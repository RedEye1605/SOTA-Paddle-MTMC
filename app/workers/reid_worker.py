"""ReID worker.

Consumes ``stream:tracklets``, downloads real crops from MinIO, runs
the active ReID adapter, normalizes embeddings, upserts to Qdrant,
persists to PostgreSQL, and emits the mean embedding to
``stream:embeddings`` for the resolver.

Hard rule: ReID runs only on tracklets, never on raw frames. A tracklet
has already passed the stability + quality filter.

Real crop flow (PATCH-004 / BUG-005 fix):
  * Each tracklet's ``crop_uris`` is a list of ``s3://bucket/key`` URIs
    that point at the evidence crops uploaded by the tracklet collector.
  * This worker downloads the actual bytes, decodes them with cv2, and
    only then runs the ReID forward pass.
  * If a download fails (network, missing key, broken jpeg), the
    tracklet is dropped from this round with a loud log and
    ``reid_extractions_dropped`` is incremented. We do NOT fabricate
    crops from ``quality_score`` or any other metadata.
  * In smoke-test mode, the operator may inject a
    ``SYNTHETIC_CROP_PROVIDER`` callable that returns synthetic images;
    the production code path requires real downloads.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import cv2
import numpy as np

from ..reid.base import ReIDAdapter
from ..core.runtime_mode import (
    RuntimeMode,
    resolve_runtime_mode,
)
from ..storage.minio_store import MinioStore
from ..storage.postgres import PostgresStore
from ..storage.qdrant_store import QdrantStore
from ..storage.redis_state import RedisState
from ..telemetry.metrics import REGISTRY
from ..utils.crop import mean_normalized
from ..utils.time import now_ts
from .tracklet_collector import Tracklet

logger = logging.getLogger(__name__)


# Counter for failed downloads (added in PATCH-022 / BUG-005).
def _register_metrics_once() -> None:
    """Lazily register additional counters (so the registry module does
    not need to know about the worker).
    """
    name = "reid_extractions_dropped_total"
    if not hasattr(REGISTRY, name):
        from ..telemetry.metrics import Counter

        setattr(REGISTRY, name, Counter(name, "ReID tracklets dropped due to bad/missing crops"))


class CropLoadError(RuntimeError):
    """Raised when a crop URI cannot be downloaded/decoded."""


class ReIDWorker:
    # PATCH: the Qdrant collection ``person_reid_transreid_msmt`` is
    # configured for 3840-dim vectors (TransReID MSMT17 with 5x768
    # JPM). The vendor pipeline's side-channel ``embedding`` field
    # carries 256-dim PP-Human strongbaseline attribute logits, NOT
    # ReID features. In B2 mode the api reid_worker is a no-op (the
    # TransReID sidecar in the eval image is the sole writer to the
    # 3840-dim collection), but the side-channel fast-path is still
    # present and would fire if the vendor pipeline starts emitting
    # embeddings. The dim check below is a safety net for that case:
    # it skips wrong-dim side-channel vectors instead of writing
    # 256-dim garbage into the 3840-dim Qdrant collection. A future
    # Paddle 3.x upgrade that re-enables StrongBaseline must write
    # to a SEPARATE Qdrant collection, not
    # ``person_reid_transreid_msmt``.
    _EXPECTED_DIM = 3840

    def __init__(
        self,
        *,
        adapter: ReIDAdapter,
        pg: PostgresStore,
        qdrant: QdrantStore,
        redis: RedisState,
        minio: Optional[MinioStore] = None,
        model_version: str = "v1",
        consumer_name: str = "reid-worker-01",
        mode: Optional[RuntimeMode] = None,
    ) -> None:
        self.adapter = adapter
        self.pg = pg
        self.qdrant = qdrant
        self.redis = redis
        self.minio = minio
        self.model_version = model_version
        self.consumer_name = consumer_name
        self._mode = mode or resolve_runtime_mode()
        # PATCH: local counter for embeddings skipped because their
        # dim does not match the Qdrant collection's vector size.
        # See ``_EXPECTED_DIM`` for context. Kept on the worker (not
        # the global REGISTRY) because we do not need to expose it
        # via /metrics; tests inspect this attribute directly.
        self.wrong_dim_skips_total: int = 0
        _register_metrics_once()

    # ---- real crop download (PATCH-004) ----
    def _download_crop(self, uri: str) -> Optional[np.ndarray]:
        """Download a single crop from MinIO via its S3 URI.

        Returns a BGR numpy array, or None if the download/decode
        failed. Never fabricates an image from metadata.
        """
        # PATCH (2026-06-15, transreid-only, operator spec): the
        # vendor pipeline's side-channel may emit ``frame_uri=None``
        # when ``PPHUMAN_SKIP_FRAME_UPLOAD=1`` is set (the
        # TransReID sidecar falls back to its RTSP ring buffer
        # instead of fetching from MinIO). The tracklet's
        # ``frame_uris`` list may contain ``None`` placeholders;
        # skip those cleanly.
        if not uri:
            return None
        if not uri.startswith("s3://"):
            # Local file path is allowed in smoke-test mode only.
            if self._mode == RuntimeMode.SMOKE_TEST:
                import os

                if os.path.exists(uri):
                    arr = cv2.imread(uri, cv2.IMREAD_COLOR)
                    if arr is not None:
                        return arr
            logger.warning("ReID crop uri is not s3:// nor a local file: %s", uri)
            return None
        if self.minio is None:
            logger.error("ReID worker has no minio client; cannot download %s", uri)
            return None
        bucket = self.minio.bucket
        prefix = f"s3://{bucket}/"
        if not uri.startswith(prefix):
            logger.warning("ReID crop uri bucket mismatch: %s (expected s3://%s/...)", uri, bucket)
            return None
        key = uri[len(prefix) :]
        try:
            resp = self.minio.client.get_object(bucket_name=bucket, object_name=key)
            data = resp.read()
            resp.close()
            resp.release_conn()
        except Exception as e:  # noqa: BLE001
            logger.warning("MinIO get_object failed for %s: %s", uri, e)
            return None
        if not data:
            logger.warning("MinIO returned empty body for %s", uri)
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cv2.imdecode failed for %s (%d bytes)", uri, len(data))
            return None
        return img

    def _load_crops(
        self,
        crop_uris: Sequence[str],
        tracklet_id: str,
    ) -> list[np.ndarray]:
        """Download real crops for a tracklet. Drops failed downloads.

        Raises :class:`CropLoadError` if NO crops could be downloaded —
        the caller must NOT fabricate crops from metadata.
        """
        crops: list[np.ndarray] = []
        for uri in crop_uris[:15]:
            img = self._download_crop(uri)
            if img is None:
                continue
            crops.append(img)
        if not crops:
            # PATCH-004: refuse to fabricate. This is the production
            # safety net — the ReID path must never be fed
            # metadata-derived pixels.
            raise CropLoadError(
                f"tracklet {tracklet_id}: no real crops could be downloaded "
                f"from {len(crop_uris)} uris; refusing to fabricate"
            )
        return crops

    def _load_crops_from_frames(
        self,
        frame_uris: Sequence[str],
        frame_bboxes: Sequence[tuple[float, float, float, float]],
        tracklet_id: str,
    ) -> list[np.ndarray]:
        """B2 path: download full BGR frames, crop the bbox in numpy.

        Used when the strongbaseline ReID is disabled and no per-crop
        JPEGs are written to MinIO. The vendor pipeline uploads the
        full frame (one PUT per camera+frame_id) and includes the
        URI in the side-channel event. The TrackletCollector forwards
        (frame_uri, bbox) pairs on the tracklet event; the worker
        finishes the work here.
        """
        crops: list[np.ndarray] = []
        for uri, bbox in zip(
            list(frame_uris)[:15],
            list(frame_bboxes)[:15],
            strict=False,
        ):
            img = self._download_crop(uri)
            if img is None:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except Exception:  # noqa: BLE001
                continue
            h, w = img.shape[:2]
            x1 = max(0, min(int(round(x1)), w - 1))
            y1 = max(0, min(int(round(y1)), h - 1))
            x2 = max(0, min(int(round(x2)), w))
            y2 = max(0, min(int(round(y2)), h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
        return crops

    # ---- main per-tracklet logic ----
    def process_tracklet(self, tl: Tracklet) -> Optional[np.ndarray]:
        """Run ReID on the tracklet's crops; upsert to Qdrant; return mean
        embedding (also emitted to stream:embeddings for the resolver).

        PATCH (2026-06-15): fast-path — if the tracklet already carries
        pre-extracted ReID embeddings (from the side-channel), use them
        directly. Otherwise fall back to the MinIO download + re-extract
        path. If neither is available, return None honestly
        (no placeholder garbage in Qdrant).

        PATCH (2026-06-15, operator's "real persistent ID" demand):
        we DROPPED the SHA-256 placeholder strategy. Garbage
        embeddings caused every tracklet to mint a new global_id
        (cosine was always 0.0). Real embeddings come from the
        TransReID sidecar in the eval image. Until the sidecar
        is in place, the resolver sees no embedding → returns
        hold_ambiguous → no global_id is minted. Honest: no
        real features = no fake features.
        """
        # ---- PATCH (2026-06-15): fast-path via side-channel embeddings ----
        if tl.embeddings:
            # The side-channel already produced N x dim-dim vectors.
            # Use them directly. Mean-aggregate for the resolver, upsert
            # each to Qdrant with the same payload schema.
            embeddings = np.stack(
                [np.asarray(e, dtype=np.float32) for e in tl.embeddings],
                axis=0,
            )
            # L2-normalize each (in case the side-channel didn't)
            for i in range(embeddings.shape[0]):
                n = np.linalg.norm(embeddings[i])
                if n > 1e-8:
                    embeddings[i] = embeddings[i] / n
        elif tl.crop_uris:
            # ---- original path: download crops from MinIO, re-extract ----
            # PATCH (2026-06-15, transreid-only): filter out any
            # ``None`` placeholders (the side-channel may emit
            # ``frame_uri=None`` when the per-frame MinIO upload is
            # skipped).
            valid_crop_uris = [u for u in tl.crop_uris if u]
            try:
                crops = self._load_crops(valid_crop_uris, tl.tracklet_id)
            except CropLoadError as e:
                logger.error("[NO-CROPS] %s", e)
                REGISTRY.reid_extractions_dropped_total.inc()
                return None
            embeddings = self.adapter.extract(crops)
            if embeddings is None or len(embeddings) == 0:
                return None
        elif tl.frame_uris and tl.frame_bboxes:
            # PATCH (2026-06-15, persistent-id): B2 path. No
            # per-crop JPEGs were written (strongbaseline is off);
            # only full frames + bboxes. Download each frame, crop
            # the bbox in pure numpy, then run ReID on the crop.
            # PATCH (2026-06-15, transreid-only): filter out
            # ``None`` URIs (the per-frame MinIO upload may be
            # skipped). The api reid_worker is a no-op in B2
            # mode (the paddle 2.x model can't load in 3.x);
            # the real TransReID sidecar in the eval image does
            # the work via the side-channel -> stream:tracklets
            # -> ``_load_crops_from_rtsp_buffer`` path.
            valid_frame_uris = [u for u in tl.frame_uris if u]
            if not valid_frame_uris:
                # PATCH (2026-06-15, transreid-only, operator
                # spec): with PPHUMAN_SKIP_FRAME_UPLOAD=1, all
                # ``frame_uris`` are ``None`` placeholders. The
                # api reid_worker is a no-op in B2 mode (the
                # strongbaseline model can't load in Paddle 3.x);
                # the real TransReID sidecar handles ReID via
                # its RTSP frame buffer (see
                # ``app/reid/transreid_sidecar.py``). We
                # therefore skip cleanly here instead of erroring
                # out.
                return None
            crops = self._load_crops_from_frames(
                valid_frame_uris, tl.frame_bboxes, tl.tracklet_id
            )
            if not crops:
                logger.error(
                    "[NO-CROPS] tracklet %s: %d frame URIs but no "
                    "decodable crops",
                    tl.tracklet_id,
                    len(valid_frame_uris),
                )
                REGISTRY.reid_extractions_dropped_total.inc()
                return None
            embeddings = self.adapter.extract(crops)
            if embeddings is None or len(embeddings) == 0:
                return None
        else:
            # PATCH (2026-06-15): NO MORE PLACEHOLDERS. Honest
            # path: log a warning, increment the drop counter, and
            # return None. The resolver will treat the tracklet as
            # hold_ambiguous and not mint a global_id. Once the
            # TransReID sidecar is in place, real embeddings flow
            # through this code path.
            logger.warning(
                "tracklet %s has no crops/embeddings; skipping "
                "(awaiting TransReID sidecar for real features)",
                tl.tracklet_id,
            )
            REGISTRY.reid_extractions_dropped_total.inc()
            return None
        if embeddings is None or len(embeddings) == 0:
            return None
        # PATCH: dim validation. The Qdrant collection
        # ``person_reid_transreid_msmt`` is 3840-dim; side-channel
        # ``embedding`` payloads can be 256-dim PP-Human attribute
        # logits. Drop wrong-dim rows with a loud warning + a local
        # counter so we never write 256-dim vectors into a 3840-dim
        # collection. See ``_EXPECTED_DIM`` for the full rationale.
        if embeddings.ndim == 2:
            actual_dims = np.asarray(
                [int(e.shape[-1]) for e in embeddings], dtype=np.int64
            )
            valid_mask = actual_dims == self._EXPECTED_DIM
            bad_count = int((~valid_mask).sum())
            if bad_count:
                for i, actual in enumerate(actual_dims):
                    if int(actual) == self._EXPECTED_DIM:
                        continue
                    logger.warning(
                        "ReID dim mismatch: tracklet_id=%s camera_id=%s "
                        "expected_dim=%d actual_dim=%d index=%d — skipping upsert",
                        tl.tracklet_id,
                        tl.camera_id,
                        self._EXPECTED_DIM,
                        int(actual),
                        i,
                    )
                self.wrong_dim_skips_total += bad_count
                embeddings = embeddings[valid_mask]
                if len(embeddings) == 0:
                    return None
        mean_vec = mean_normalized(embeddings)
        # PATCH-012: namespace point ids by model name so two adapters
        # writing the same tracklet don't collide.
        # PATCH (2026-06-15): Qdrant requires point_ids to be UUID
        # or unsigned integer. Use a deterministic UUID derived
        # from (model_prefix, tracklet_id, index) for stability
        # across restarts.
        import uuid as _uuid
        model_prefix = self.adapter.name
        for i, vec in enumerate(embeddings):
            point_id = str(
                _uuid.uuid5(
                    _uuid.NAMESPACE_URL,
                    f"{model_prefix}:{tl.tracklet_id}:{i:02d}",
                )
            )
            payload = {
                "global_id": None,  # filled by resolver
                "tracklet_id": tl.tracklet_id,
                "camera_id": tl.camera_id,
                "local_track_id": tl.local_track_id,
                "zone_id": tl.end_zone_id or "",
                "site_id": tl.site_id,
                "timestamp": int(now_ts()),
                "quality_score": float(tl.quality_score or 0.0),
                "model_name": self.adapter.name,
                "model_version": self.model_version,
                "embedding_version": self.model_version,
                "crop_uri": tl.crop_uris[i] if i < len(tl.crop_uris) else "",
            }
            self.qdrant.upsert_point(
                self.adapter.qdrant_collection, vec, payload, point_id=point_id
            )
            self.pg.insert_tracklet_embedding(
                tracklet_id=tl.tracklet_id,
                global_id=None,
                camera_id=tl.camera_id,
                model_name=self.adapter.name,
                model_version=self.model_version,
                vector_db_collection=self.adapter.qdrant_collection,
                vector_db_point_id=point_id,
                embedding_dim=int(self.adapter.embedding_dim),
                quality_score=tl.quality_score,
            )
        # PATCH-022: increment the extractions counter
        REGISTRY.reid_extractions.inc(amount=len(embeddings))
        # Emit the mean embedding to stream:embeddings for the resolver
        # PATCH (2026-06-15): add local_track_id so the resolver and
        # overlay cache can join on (camera_id, local_track_id).
        self.redis.publish(
            "stream:embeddings",
            {
                "tracklet_id": tl.tracklet_id,
                "camera_id": tl.camera_id,
                "local_track_id": tl.local_track_id,
                "site_id": tl.site_id,
                "ts": int(now_ts()),
                "model_name": self.adapter.name,
                "model_version": self.model_version,
                "embedding_version": self.model_version,
                "qdrant_collection": self.adapter.qdrant_collection,
                "embedding_dim": int(self.adapter.embedding_dim),
                "mean_vec": mean_vec.tolist(),
                "end_zone_id": tl.end_zone_id,
                "quality_score": tl.quality_score,
            },
        )
        return mean_vec

    def minio_bucket_for_uri(self) -> str:
        return self.minio.bucket if self.minio is not None else "evidence"

    def run(self, stop_event=None) -> None:
        """Consume ``stream:tracklets`` in a loop."""
        self.redis.ensure_group("stream:tracklets", "reid_workers")
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            msgs = self.redis.consume(
                "stream:tracklets",
                "reid_workers",
                self.consumer_name,
                count=4,
                block_ms=1000,
            )
            for msg_id, fields in msgs:
                try:
                    tl = Tracklet(
                        tracklet_id=fields["tracklet_id"],
                        camera_id=fields["camera_id"],
                        local_track_id=int(fields["local_track_id"]),
                        start_time=float(fields["start_time"]),
                        end_time=float(fields.get("end_time") or 0.0) or None,
                        site_id=fields.get("site_id", ""),
                        end_zone_id=fields.get("end_zone_id"),
                        quality_score=float(fields.get("quality_score") or 0.0),
                        crop_uris=list(fields.get("crop_uris", [])),
                        # PATCH (2026-06-15, persistent-id): B2 path. The
                        # vendor pipeline writes full BGR frames to MinIO
                        # and includes ``frame_uri`` + ``bbox`` in the
                        # side-channel. The tracklet event carries the
                        # collected ``frame_uris``/``frame_bboxes``; if
                        # present, ``process_tracklet`` downloads each
                        # frame and crops the bbox in numpy. This replaces
                        # the old (now broken) per-crop ``crop_uris`` path
                        # in B2 mode.
                        frame_uris=list(fields.get("frame_uris", [])),
                        frame_bboxes=[
                            tuple(b) for b in fields.get("frame_bboxes", []) or []
                        ],
                    )
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        "ReID invalid tracklet message %s; acking poison entry: %s",
                        msg_id,
                        e,
                    )
                    self.redis.ack("stream:tracklets", "reid_workers", msg_id)
                    continue
                try:
                    self.process_tracklet(tl)
                except Exception as e:  # noqa: BLE001
                    logger.exception(
                        "ReID process_tracklet failed for %s: %s",
                        tl.tracklet_id,
                        e,
                    )
                    continue
                else:
                    self.redis.ack("stream:tracklets", "reid_workers", msg_id)
