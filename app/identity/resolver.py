"""Global identity resolver — wired to the runtime stream pipeline.

Pulls embeddings from ``stream:embeddings``, runs the staged retrieval
(Stage 1: same-cam recent; Stage 2: linked cameras within the
travel-time window; Stage 3: 24 h global fallback), applies the 5-factor
``final_score`` decision, persists every decision to
``identity_decisions`` for audit, and emits the decision to
``stream:identity_decisions`` for telemetry.

Hard rules (enforced by tests):
  - ``final_score`` is the threshold variable; the raw ReID cosine is
    only one of the 5 factors. (PATCH-008 / BUG-008)
  - The resolver MUST use ``camera_links`` to filter candidates.
    (BUG-010 / PATCH-016)
  - Cross-camera search is filtered by travel-time window
    (``[ts - max_travel_seconds, ts - min_travel_seconds]``) so a
    23 h-old candidate is excluded. (PATCH-016)
  - Ambiguous candidates are NOT auto-merged. (BUG-008)
  - Qdrant search ALWAYS uses payload filters (camera + model + ts).
  - The resolver updates ``tracklets.global_id`` so the joinable
    column matches the audit. (BUG-011)
  - The resolver publishes to ``stream:identity_decisions`` so the
    telemetry worker can act on it. (PATCH-010)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from ..storage.postgres import PostgresStore
from ..storage.qdrant_store import QdrantStore
from ..storage.redis_state import RedisState
from ..telemetry.metrics import REGISTRY
from ..utils.time import age_seconds, now_ts
from .ambiguity import CandidateHit, decide_ambiguity
from .camera_topology import CameraTopology
from .scoring import ScoreWeights, score_breakdown
from .session import current_session_id, mint_global_id

logger = logging.getLogger(__name__)


@dataclass
class ResolverConfig:
    auto_match_threshold: float = 0.82
    candidate_threshold: float = 0.72
    # PATCH (2026-06-15): operator spec is 0.05. The previous
    # 0.04 let near-tied top-1/top-2 pairs auto-match and led to
    # 1 person = N stored embeddings (the operator's complaint).
    ambiguous_margin: float = 0.05
    prefer_new_id_when_ambiguous: bool = True
    use_camera_topology: bool = True
    use_zone_transitions: bool = True
    persistence_window_seconds: int = 86_400
    temporal_sigma_seconds: float = 60.0
    enable_stage3_24h_fallback: bool = True
    # Stage 3 threshold is higher than Stage 2's auto_match (default
    # 0.92). This is per the audit's "Stage 3 is low-confidence only".
    stage3_auto_match_threshold: float = 0.92
    # PATCH (2026-06-17, BUG-1): high-confidence ReID cosine
    # short-circuit. When ``top1.score >= reid_override_threshold`` AND
    # the margin from top-2 is at least ``ambiguous_margin``, the
    # answer is "match" regardless of the 5-factor final_score. This
    # is the operator's plan requirement: "real ReID features must
    # dedup even when time_diff pushes final_score below threshold".
    # Default 0.95 — strong-enough cosine to be safe, with the
    # margin check still requiring a clear winner.
    reid_override_threshold: Optional[float] = 0.95
    consumer_name: str = "resolver-worker-01"
    weights: ScoreWeights = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = ScoreWeights()


class GlobalIdentityResolver:
    """Async consumer of ``stream:embeddings``. Single instance shared across cameras."""

    def __init__(
        self,
        *,
        pg: PostgresStore,
        qdrant: QdrantStore,
        redis: RedisState,
        topology: CameraTopology,
        config: Optional[ResolverConfig] = None,
        model_name: str = "transreid_msmt",
        model_version: str = "v1",
        consumer_name: Optional[str] = None,
    ) -> None:
        self.pg = pg
        self.qdrant = qdrant
        self.redis = redis
        self.topology = topology
        self.config = config or ResolverConfig()
        self.model_name = model_name
        self.model_version = model_version
        # Always resolve self.consumer_name from a single, ordered set
        # of sources. This must not raise — the run() consumer thread
        # depends on it.
        #
        # Priority:
        #   1. explicit consumer_name kwarg (tests, multi-process)
        #   2. config.consumer_name (operator override via YAML/env)
        #   3. ResolverConfig dataclass default "resolver-worker-01"
        #   4. last-resort literal "resolver-worker-01" (config is None
        #      AND no ResolverConfig default — defensive)
        self.consumer_name = (
            consumer_name
            or getattr(self.config, "consumer_name", None)
            or "resolver-worker-01"
        )

    # ---- staged candidate retrieval (Stage 1, 2, 3) ----
    def _stage1_candidates(self, source_camera_id: str) -> list[str]:
        """Stage 1 — same camera only. Always allowed."""
        return [source_camera_id]

    def _stage2_candidates(self, source_camera_id: str) -> list[str]:
        """Stage 2 — topology-linked cameras only (enabled links)."""
        return sorted(self.topology.candidate_cameras_for(source_camera_id))

    def _stage3_candidates(self, source_camera_id: str) -> list[str]:
        """Stage 3 — 24 h global fallback: ALL cameras that have been
        seen in the last 24 h, regardless of topology.

        This is the "rare re-identification" path: e.g. CAM_01 →
        CAM_05 where no direct link is configured. We use the
        ``camera:last_seen:{cam_id}`` Redis key as a witness.
        """
        try:
            # We do not have a list of all cameras cached; iterate over
            # the cameras we know about (from topology) and add any
            # recently-seen camera that the resolver has touched.
            all_cams = {link.from_camera_id for link in self.topology._links.values()}
            all_cams |= {link.to_camera_id for link in self.topology._links.values()}
            all_cams.add(source_camera_id)
            # Filter by "seen in the last 24 h"
            out: list[str] = []
            for cam in all_cams:
                last_seen = self.redis.get_camera_last_seen(cam)
                if last_seen is None:
                    continue
                if (now_ts() - float(last_seen)) <= 86_400:
                    out.append(cam)
            return sorted(out)
        except Exception as e:  # noqa: BLE001
            logger.debug("stage3 candidate enumeration failed: %s", e)
            return []

    def _candidate_cameras(
        self,
        source_camera_id: str,
        ts: float,
    ) -> tuple[list[str], str]:
        """Return (candidate_camera_ids, stage_name) for the resolver.

        Stage 1: same camera (always)
        Stage 2: linked cameras
        Stage 3 (optional): all seen-in-24h cameras
        """
        stage1 = self._stage1_candidates(source_camera_id)
        stage2 = self._stage2_candidates(source_camera_id)
        if not stage2 and not self.config.enable_stage3_24h_fallback:
            return stage1, "stage1_same_cam_only"
        if not stage2 and self.config.enable_stage3_24h_fallback:
            return self._stage3_candidates(source_camera_id), "stage3_24h_fallback"
        # Linked cams present — start with stage1 + stage2, then
        # opportunistically add stage3 (low-confidence only).
        union = sorted(set(stage1) | set(stage2))
        if self.config.enable_stage3_24h_fallback:
            stage3 = self._stage3_candidates(source_camera_id)
            union = sorted(set(union) | set(stage3) - set(union))
            if stage3:
                return union, "stage2_3_combined"
        return union, "stage2_linked_cams"

    def _search_with_filters(
        self,
        *,
        query_vec: np.ndarray,
        candidate_cams: Iterable[str],
        ts: float,
        source_camera_id: str,
        # PATCH (2026-06-15, transreid-only, real-video demo):
        # the default was 0.5 but MOT tracklets from the real
        # 2h production videos often have ``quality_score=0.0``
        # (the side-channel emits the detection score, not a
        # ReID quality score). Filtering at >=0.5 excluded
        # every real-video tracklet from the candidate pool,
        # making dedup impossible. Default 0.0 disables the
        # quality filter; ops can re-enable in the smoke path.
        quality_gte: float = 0.0,
        top_k: int = 10,
        travel_window_only: bool = True,
    ) -> list[Any]:
        """Filtered Qdrant search with optional travel-time narrowing.

        PATCH-016 fix: for each candidate camera we build a
        per-camera ``[gte, lte]`` timestamp window based on the
        topology's ``min_travel_seconds`` / ``max_travel_seconds``,
        and run a per-camera Qdrant sub-query. This is the official
        Qdrant pattern for "different bounds per partition" — there
        is no first-class OR-of-bounds; one Filter per camera is the
        correct shape.

        Same-camera candidates (Stage 1) get a wide window
        ``[ts - persistence_window, ts]``.
        Cross-camera candidates (Stage 2) get a tight window
        ``[ts - max_travel, ts - min_travel]`` so a 23h-old CAM_01
        candidate for a CAM_02 tracklet is excluded.

        If ``travel_window_only=False`` (smoke tests) we fall back to
        the broad ``[ts - persistence_window, ts]`` window for all
        candidates.
        """
        candidate_list = list(candidate_cams)
        if not candidate_list:
            return []
        if not travel_window_only:
            return self.qdrant.search(
                self._qdrant_collection_for(self.model_name),
                query_vec,
                timestamp_gte=int(ts - self.config.persistence_window_seconds),
                candidate_camera_ids=candidate_list,
                model_name=self.model_name,
                model_version=self.model_version,
                quality_score_gte=quality_gte,
                top_k=top_k,
            )
        per_camera: dict[str, tuple[Optional[int], Optional[int]]] = {}
        for cam in candidate_list:
            if cam == source_camera_id:
                # Stage 1: same-cam recent. Allow up to the full
                # persistence window. No upper bound (still in
                # the active session).
                per_camera[cam] = (
                    int(ts - self.config.persistence_window_seconds),
                    None,
                )
                continue
            # ``CameraLink`` keys are ``(from_camera_id, to_camera_id)``
            # where ``from`` is where the person was last seen and
            # ``to`` is the new camera. The candidate ``cam`` is the
            # last-seen camera and ``source_camera_id`` is the new one,
            # so the link key is ``(cam, source_camera_id)``.
            link = self.topology._links.get((cam, source_camera_id))
            if link is None or not link.enabled:
                # No enabled link from this candidate to the source
                # camera: skip. The topology hard-block in
                # `decide_ambiguity` will turn the decision into
                # "new" anyway, but excluding them here reduces the
                # candidate pool and saves Qdrant cycles.
                continue
            gte = int(ts - link.max_travel_seconds)
            lte = int(ts - link.min_travel_seconds)
            # A negative `lte` would mean "the tracklet is in the
            # future relative to the candidate" — physically
            # impossible. The Qdrant filter accepts negatives (treated
            # as 1970) but we cap at 0 to keep filter semantics sane.
            if lte < 0:
                lte = 0
            per_camera[cam] = (gte, lte)
        if not per_camera:
            return []
        return self.qdrant.search_per_camera(
            self._qdrant_collection_for(self.model_name),
            query_vec,
            per_camera_windows=per_camera,
            model_name=self.model_name,
            model_version=self.model_version,
            quality_score_gte=quality_gte,
            top_k=top_k,
        )

    @staticmethod
    def _qdrant_collection_for(model_name: str) -> str:
        # PATCH (2026-06-17, operator spec, transreid-only): the
        # only active ReID model is TransReID MSMT17. The
        # pphuman_strongbaseline / vanilla-transreid / clipreid
        # collections were dropped. We still accept the
        # historical ``transreid`` alias for back-compat (the
        # docs/transreid_msmt_setup.md describes a transition
        # window where the sidecar could publish under either
        # name) and fall back to the MSMT17 collection when the
        # model_name is anything else.
        if model_name in ("transreid_msmt", "transreid"):
            return "person_reid_transreid_msmt"
        # Defensive: any unknown model_name lands in the MSMT17
        # collection. The ResolverConfig.model_name default is
        # ``transreid_msmt``; misconfiguration here is logged by
        # the caller.
        return "person_reid_transreid_msmt"

    # ---- PATCH (2026-06-15): Qdrant global_id payload backfill ----
    def _backfill_qdrant_global_id(
        self,
        tracklet_id: str,
        global_id: Optional[str],
        collection: str,
    ) -> None:
        """Update the Qdrant payload's ``global_id`` field for every point
        matching this ``tracklet_id``.

        PATCH (2026-06-15): this is the operator's bug #3 fix. Without
        it, subsequent tracklets from the same person search Qdrant,
        find prior embeddings, but the resolver's Stage 1/2/3
        candidate loop skips them
        (``if not gid or not cid: continue`` line ~314) because the
        Qdrant payload's ``global_id`` is still ``None``. The dedup
        chain silently fails and 1 person = N stored embeddings.

        Best-effort: log and continue on errors. The resolver must
        not fail just because Qdrant is briefly slow.
        """
        if not global_id or not tracklet_id:
            return
        try:
            from qdrant_client import models as _models
            # Scroll all points with this tracklet_id, fetch vectors,
            # and re-upsert with updated payload.
            points, _next = self.qdrant.client.scroll(
                collection_name=collection,
                scroll_filter=_models.Filter(
                    must=[
                        _models.FieldCondition(
                            key="tracklet_id",
                            match=_models.MatchValue(value=tracklet_id),
                        )
                    ]
                ),
                with_vectors=True,
                with_payload=True,
                limit=128,
            )
            if not points:
                return
            updated_points = []
            for pt in points:
                payload = dict(pt.payload or {})
                payload["global_id"] = global_id
                # Preserve the original point id so we re-upsert
                # in-place (no duplicate).
                updated_points.append(
                    _models.PointStruct(
                        id=pt.id,
                        vector=pt.vector,
                        payload=payload,
                    )
                )
            self.qdrant.client.upsert(
                collection_name=collection,
                points=updated_points,
            )
            logger.debug(
                "Qdrant backfill: tracklet=%s gid=%s collection=%s pts=%d",
                tracklet_id, global_id, collection, len(updated_points),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Qdrant global_id backfill failed for tracklet=%s collection=%s: %s",
                tracklet_id, collection, e,
            )

    # ---- PATCH (2026-06-15): identity:active:{cam}:{local} writes ----
    def _set_active_if_possible(
        self,
        global_id: Optional[str],
        camera_id: str,
        local_track_id: Optional[int],
    ) -> None:
        """Write the ``active:{camera_id}:{local_track_id}`` Redis
        binding so a restarted IdentityOverlayCache (or external
        consumer) can recover the ``G:{global_id}`` overlay.

        PATCH (2026-06-15): this is the operator's bug #4 fix. The
        overlay cache reads from this Redis key, but the resolver
        never called ``set_active``. The cache lookup always returned
        ``None`` and the overlay showed only the local_track_id.

        PATCH (2026-06-17): the live evidence shows the keys are
        still empty after many tracklets. The likely cause: the
        resolver's ``_set_active_if_possible`` is called, but the
        ``local_track_id`` from the embedding event is ``None`` (the
        api's reid_worker does not always include it). This patch
        adds (a) a one-shot info log on first set_active call and
        (b) a debug log on every set_active, so the operator can
        verify the contract end-to-end.
        """
        if not global_id or not camera_id or local_track_id is None:
            # PATCH (2026-06-17): log when the binding is skipped
            # so the operator can see why ``active:*`` is empty.
            if not getattr(self, "_logged_set_active_skip", False):
                logger.info(
                    "set_active SKIP: gid=%r cam=%r local=%r "
                    "(one of them is None — this is normal when "
                    "the embedding event lacks local_track_id)",
                    global_id, camera_id, local_track_id,
                )
                self._logged_set_active_skip = True
            return
        try:
            self.redis.set_active(camera_id, int(local_track_id), global_id)
            if not getattr(self, "_logged_set_active_ok", False):
                logger.info(
                    "set_active OK: gid=%s cam=%s local=%s "
                    "(first successful write — subsequent writes "
                    "logged at DEBUG only)",
                    global_id, camera_id, local_track_id,
                )
                self._logged_set_active_ok = True
            logger.debug(
                "set_active: gid=%s cam=%s local=%s",
                global_id, camera_id, local_track_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "set_active failed: cam=%s local=%s gid=%s: %s",
                camera_id, local_track_id, global_id, e,
            )

    # ---- main entry point ----
    def resolve(
        self,
        *,
        tracklet_id: str,
        camera_id: str,
        ts: float,
        mean_embedding: np.ndarray,
        tracklet_quality: Optional[float] = None,
        new_zone_id: Optional[str] = None,
        prev_zone_id: Optional[str] = None,
        local_track_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve a tracklet's embedding into a global_id decision.

        Returns a dict summarizing the decision (used by telemetry and
        tests). The decision is persisted to ``identity_decisions`` and
        ``tracklets.global_id`` is updated.

        PATCH (2026-06-15): accept ``local_track_id`` (optional) so the
        stream:identity_decisions publish payload can include it as
        the join key for the IdentityOverlayCache. If not provided,
        ``local_track_id`` is None in the published event.

        PATCH (2026-06-15, sidecar): accept ``model_name`` override
        from the embedding event. With the TransReID sidecar
        publishing embeddings to Qdrant collection
        ``person_reid_transreid_msmt`` (model_name=
        ``transreid_msmt``) and the api's reid_worker publishing to
        ``person_reid_pphuman`` (model_name=``pphuman_strongbaseline``),
        the resolver must look at the model_name in the event
        itself, not the resolver's default. Otherwise the resolver
        filters by the api's model_name and never sees the
        sidecar's embeddings. (Without this fix, the chain is
        broken end-to-end.)
        """
        # Resolve which model produced this embedding. The event is
        # the source of truth; the resolver's default is the
        # fallback for callers that do not provide one.
        effective_model_name = model_name or self.model_name
        # Save the original for the post-decision side-effects.
        original_model_name = self.model_name
        self.model_name = effective_model_name
        try:
            return self._resolve_inner(
                tracklet_id=tracklet_id,
                camera_id=camera_id,
                ts=ts,
                mean_embedding=mean_embedding,
                tracklet_quality=tracklet_quality,
                new_zone_id=new_zone_id,
                prev_zone_id=prev_zone_id,
                local_track_id=local_track_id,
            )
        finally:
            self.model_name = original_model_name

    def _resolve_inner(
        self,
        *,
        tracklet_id: str,
        camera_id: str,
        ts: float,
        mean_embedding: np.ndarray,
        tracklet_quality: Optional[float],
        new_zone_id: Optional[str],
        prev_zone_id: Optional[str],
        local_track_id: Optional[int],
    ) -> dict[str, Any]:
        candidate_cams, stage = self._candidate_cameras(camera_id, ts)
        hits = self._search_with_filters(
            query_vec=mean_embedding,
            candidate_cams=candidate_cams,
            ts=ts,
            source_camera_id=camera_id,
            top_k=10,
        )
        # Filter out the current tracklet itself
        hits = [h for h in hits if (h.payload or {}).get("tracklet_id") != tracklet_id]

        # Build candidate hits (per-cam best). BUG-009 fix: use `>` not
        # `>=` so equal scores do not silently drop the second camera.
        top1: Optional[CandidateHit] = None
        top2: Optional[CandidateHit] = None
        per_cam: dict[str, float] = {}
        for h in hits:
            payload = h.payload or {}
            gid = payload.get("global_id")
            cid = payload.get("camera_id")
            last_seen = float(payload.get("timestamp", ts))
            if not gid or not cid:
                continue
            score = float(h.score)
            if cid in per_cam and per_cam[cid] > score:
                continue
            per_cam[cid] = score
            cand = CandidateHit(global_id=gid, camera_id=cid, score=score, last_seen_at=last_seen)
            if top1 is None or cand.score > top1.score:
                top2 = top1
                top1 = cand
            elif top2 is None or cand.score > top2.score:
                top2 = cand

        # Apply 5-factor scoring on top-1
        reid_sim = top1.score if top1 is not None else 0.0
        time_diff = age_seconds(top1.last_seen_at, ts) if top1 is not None else 0.0
        is_link = (
            self.topology.is_known_link(top1.camera_id, camera_id)
            if self.config.use_camera_topology and top1 is not None
            else None
        )
        breakdown = score_breakdown(
            reid_similarity=reid_sim,
            time_diff_seconds=time_diff,
            is_known_link=is_link,
            tracklet_quality=tracklet_quality,
            prev_zone_id=prev_zone_id,
            new_zone_id=new_zone_id if self.config.use_zone_transitions else None,
            weights=self.config.weights,
            sigma_seconds=self.config.temporal_sigma_seconds,
        )
        final = breakdown["final_score"]

        # PATCH-008: pass the final_score to decide_ambiguity so it
        # thresholds against the weighted score, not the raw cosine.
        # For Stage 3 the threshold is tighter (per the audit's
        # "low-confidence only" rule).
        auto_match_threshold = self.config.auto_match_threshold
        if stage.startswith("stage3"):
            auto_match_threshold = self.config.stage3_auto_match_threshold
        # PATCH (2026-06-17, BUG-1): forward the reid_override_threshold
        # so a near-perfect cosine match (>= 0.95) bypasses the 5-factor
        # weighted score. Without this, time_diff=150s would give
        # final_score=0.658 even with cosine=0.998, and the resolver
        # would mint a new global_id per person (the operator's
        # "1 person = N stored embeddings" complaint).
        decision = decide_ambiguity(
            top1,
            top2,
            auto_match_threshold=auto_match_threshold,
            candidate_threshold=self.config.candidate_threshold,
            ambiguous_margin=self.config.ambiguous_margin,
            is_known_link=is_link,
            prefer_new_id_when_ambiguous=self.config.prefer_new_id_when_ambiguous,
            final_score=final,
            reid_override_threshold=self.config.reid_override_threshold,
        )

        # Top-2 fields for audit
        top1_id, top1_cam, top1_score = (
            (top1.global_id, top1.camera_id, top1.score) if top1 else (None, None, None)
        )
        top2_id, top2_cam, top2_score = (
            (top2.global_id, top2.camera_id, top2.score) if top2 else (None, None, None)
        )

        # Apply decision
        assigned_global_id: Optional[str] = None
        confidence_state = "firm"
        if decision == "match":
            assigned_global_id = top1.global_id
            confidence_state = "firm"
            self.pg.update_global_identity_seen(assigned_global_id, ts, camera_id)
            self.pg.update_tracklet_global_id(tracklet_id, assigned_global_id)  # BUG-011
            self.redis.mark_recent(assigned_global_id, ts, camera_id)
            self.redis.mark_camera_last_seen(camera_id, ts)
        elif decision in ("candidate", "ambiguous", "held"):
            assigned_global_id = None
            confidence_state = "ambiguous" if decision == "ambiguous" else "held"
        else:  # "new"
            assigned_global_id = mint_global_id(camera_id, ts)
            confidence_state = "firm"
            self.pg.create_global_identity(
                global_id=assigned_global_id,
                session_id=current_session_id(ts),
                first_seen_at=ts,
                last_seen_at=ts,
                first_camera_id=camera_id,
                last_camera_id=camera_id,
                confidence_state=confidence_state,
            )
            self.pg.update_tracklet_global_id(tracklet_id, assigned_global_id)  # BUG-011
            self.redis.mark_recent(assigned_global_id, ts, camera_id)
            self.redis.mark_camera_last_seen(camera_id, ts)
        # PATCH (2026-06-15): BUG-3 + BUG-4 fixes. Wire the backfill
        # of the Qdrant payload's global_id and the Redis active
        # binding for the overlay cache. These are best-effort and
        # must not fail the resolver.
        if assigned_global_id:
            self._backfill_qdrant_global_id(
                tracklet_id=tracklet_id,
                global_id=assigned_global_id,
                collection=self._qdrant_collection_for(self.model_name),
            )
            self._set_active_if_possible(
                global_id=assigned_global_id,
                camera_id=camera_id,
                local_track_id=local_track_id,
            )
            self.redis.mark_camera_last_seen(camera_id, ts)
            # PATCH-010: publish a zone_event for the new global_id at
            # the camera (so dwell sessions can be opened).
            if new_zone_id:
                self.redis.publish(
                    "stream:zone_events",
                    {
                        "global_id": assigned_global_id,
                        "tracklet_id": tracklet_id,
                        "camera_id": camera_id,
                        "zone_id": new_zone_id,
                        "event_type": "enter",
                        "timestamp": ts,
                    },
                )

        reason = (
            f"stage={stage} decision={decision} "
            f"reid={breakdown['reid_similarity']:.3f} "
            f"topo={breakdown['camera_topology_score']:.2f} "
            f"temporal={breakdown['temporal_score']:.3f} "
            f"quality={breakdown['quality_score']:.3f} "
            f"zone={breakdown['zone_score']:.3f} "
            f"final={final:.3f}"
        )

        # Persist decision (audit)
        self.pg.insert_identity_decision(
            tracklet_id=tracklet_id,
            source_camera_id=camera_id,
            candidate_camera_id=top1_cam,
            assigned_global_id=assigned_global_id,
            decision_type=decision,
            top1_global_id=top1_id,
            top1_camera_id=top1_cam,
            top1_score=top1_score,
            top2_global_id=top2_id,
            top2_camera_id=top2_cam,
            top2_score=top2_score,
            reid_similarity=breakdown["reid_similarity"],
            temporal_score=breakdown["temporal_score"],
            camera_topology_score=breakdown["camera_topology_score"],
            quality_score=breakdown["quality_score"],
            zone_score=breakdown["zone_score"],
            final_score=breakdown["final_score"],
            reason=reason,
        )

        # PATCH-010: publish the decision to stream:identity_decisions
        # so the telemetry worker can forward it to ThingsBoard.
        # PATCH (2026-06-15): include local_track_id (join key for
        # the IdentityOverlayCache).
        try:
            self.redis.publish(
                "stream:identity_decisions",
                {
                    "tracklet_id": tracklet_id,
                    "camera_id": camera_id,
                    "local_track_id": local_track_id,
                    "ts": ts,
                    "decision": decision,
                    "assigned_global_id": assigned_global_id,
                    "confidence_state": confidence_state,
                    "stage": stage,
                    "top1_global_id": top1_id,
                    "top1_camera_id": top1_cam,
                    "top1_score": top1_score,
                    "top2_global_id": top2_id,
                    "top2_camera_id": top2_cam,
                    "top2_score": top2_score,
                    "final_score": final,
                    "reason": reason,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("publish to stream:identity_decisions failed: %s", e)

        REGISTRY.identity_decisions.inc()
        return {
            "tracklet_id": tracklet_id,
            "camera_id": camera_id,
            "ts": ts,
            "decision": decision,
            "assigned_global_id": assigned_global_id,
            "confidence_state": confidence_state,
            "stage": stage,
            "top1": {"global_id": top1_id, "camera_id": top1_cam, "score": top1_score},
            "top2": {"global_id": top2_id, "camera_id": top2_cam, "score": top2_score},
            "breakdown": breakdown,
            "reason": reason,
        }

    # ---- stream consumer (PATCH-006) ----
    def run(self, stop_event=None) -> None:
        """Consume ``stream:embeddings`` and resolve each event.

        PATCH-006 fix: the resolver is now a real stream consumer.
        """
        # Defensive: __init__ already sets self.consumer_name with a
        # safe default. If a future refactor breaks that contract, do
        # NOT crash the worker thread — log a structured warning and
        # fall back to a known-good name. The resolver keeps running.
        consumer_name = getattr(self, "consumer_name", None) or "resolver-worker-01"
        if consumer_name != getattr(self, "consumer_name", None):
            logger.warning(
                "resolver.consumer_name missing or empty; falling back to %r",
                consumer_name,
            )
        self.redis.ensure_group("stream:embeddings", "resolver_workers")
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            msgs = self.redis.consume(
                "stream:embeddings",
                "resolver_workers",
                consumer_name,
                count=4,
                block_ms=1000,
            )
            for msg_id, fields in msgs:
                try:
                    self._handle_one(fields)
                except Exception as e:  # noqa: BLE001
                    logger.exception("resolver._handle_one failed: %s", e)
                    continue
                else:
                    self.redis.ack("stream:embeddings", "resolver_workers", msg_id)

    def _handle_one(self, fields: dict[str, Any]) -> dict[str, Any]:
        tracklet_id = fields["tracklet_id"]
        camera_id = fields["camera_id"]
        # PATCH (2026-06-15, dedup-debug): one-shot log of decoded
        # fields so the operator can see exactly what the
        # resolver received from the sidecar's embedding event.
        if not getattr(self, "_logged_handle_one", False):
            logger.info(
                "resolver DEBUG _handle_one: tracklet=%s cam=%r ts=%r "
                "model_name=%r mean_vec_len=%d",
                tracklet_id[:12] if isinstance(tracklet_id, str) else tracklet_id,
                camera_id,
                fields.get("ts"),
                fields.get("model_name"),
                len(fields.get("mean_vec", [])),
            )
            self._logged_handle_one = True
        ts = float(fields.get("ts") or now_ts())
        end_zone_id = fields.get("end_zone_id")
        quality = float(fields.get("quality_score") or 0.0)
        mean_vec = np.asarray(fields.get("mean_vec", []), dtype=np.float32)
        if mean_vec.size == 0:
            logger.warning("resolver: empty mean_vec for %s", tracklet_id)
            return {}
        # PATCH (2026-06-15, sidecar): the embedding event carries
        # its own ``model_name`` (e.g. ``transreid_msmt`` from the
        # sidecar or ``pphuman_strongbaseline`` from the api's
        # reid_worker). Pass it through so the resolver routes to
        # the right Qdrant collection. Without this, the resolver
        # would always filter by its own default and never see
        # cross-model embeddings.
        event_model_name = fields.get("model_name") or None
        # PATCH (2026-06-15, anti-flicker): read the embedding
        # event's ``local_track_id`` (the api's reid_worker always
        # includes it; the sidecar also includes it after the
        # ``stream:tracklets`` -> sidecar hop). Without this, the
        # ``stream:identity_decisions`` event has
        # ``local_track_id=null`` and the IdentityOverlayCache
        # cannot bind a ``(camera_id, local_track_id) -> global_id``
        # entry — every new MOT re-id then renders ``Person``
        # without a G:<gid> label in the HLS overlay.
        raw_ltid = fields.get("local_track_id")
        try:
            local_track_id = int(raw_ltid) if raw_ltid is not None else None
        except (TypeError, ValueError):
            local_track_id = None
        return self.resolve(
            tracklet_id=tracklet_id,
            camera_id=camera_id,
            ts=ts,
            mean_embedding=mean_vec,
            tracklet_quality=quality,
            new_zone_id=end_zone_id,
            model_name=event_model_name,
            local_track_id=local_track_id,
        )
