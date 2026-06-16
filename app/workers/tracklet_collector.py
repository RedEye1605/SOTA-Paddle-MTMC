"""Tracklet collector.

For each local track that is *stable* (>= min_track_age_frames), it
collects 5–15 candidate crops, applies image-quality filter, picks the
best one, uploads to MinIO, and emits a tracklet event to Redis Stream
`stream:tracklets` and PostgreSQL `tracklets`.

Hard rule: ReID does NOT run here. The ReID worker consumes
`stream:tracklets` separately.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..storage.minio_store import MinioStore
from ..storage.postgres import PostgresStore
from ..storage.redis_state import RedisState
from ..utils.crop import crop_with_padding
from ..utils.image_quality import crop_quality_score
from ..utils.time import now_ts
from ..zones.assignment import assign_bbox
from ..zones.polygon import parse_zones
from .pphuman_worker import FrameResult

logger = logging.getLogger(__name__)


@dataclass
class Tracklet:
    tracklet_id: str
    camera_id: str
    local_track_id: int
    start_time: float
    end_time: Optional[float] = None
    best_crop_uri: Optional[str] = None
    quality_score: Optional[float] = None
    frame_count: int = 0
    embedding_count: int = 0
    start_zone_id: Optional[str] = None
    end_zone_id: Optional[str] = None
    crop_uris: list[str] = field(default_factory=list)
    # PATCH-009 fix: per-crop quality scores; the highest wins.
    crop_quality_scores: list[float] = field(default_factory=list)
    # PATCH (2026-06-15, persistent-id): s3 URIs of full BGR frames for
    # the detections in this tracklet. The TransReID sidecar downloads
    # each frame, crops the bbox, and feeds the crop to TransReID. One
    # URI per detection, capped at ``max_crops_per_tracklet``.
    frame_uris: list[str] = field(default_factory=list)
    # PATCH (2026-06-15, persistent-id): per-detection bbox tuples
    # parallel to ``frame_uris`` (x1, y1, x2, y2 in pixel coords of the
    # full frame). The sidecar uses these to crop the frame at ReID
    # time.
    frame_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    site_id: str = ""
    # PATCH (2026-06-15): per-crop 256-dim ReID embeddings carried from
    # the detection event side-channel. ReIDWorker prefers these over
    # the MinIO download path (faster, no re-extraction).
    embeddings: list[np.ndarray] = field(default_factory=list)
    # PATCH (2026-06-16, SAHI integration): source of the tracklet
    # (pphuman = MOT-tracked; sahi = auxiliary short-lived track).
    # Provisional = True for SAHI tracklets; downstream treats them
    # as low-confidence until seen in 2+ frames.
    source: str = "pphuman"
    provisional: bool = False


# PATCH (2026-06-15): DetectionEvent — the structured per-detection event
# consumed from stream:detections. Mirrors the operator's spec schema.
@dataclass
class DetectionEvent:
    schema_version: str = "1.0"
    event_id: str = ""
    source: str = "pphuman"
    run_id: str = ""
    camera_id: str = ""
    frame_id: int = 0
    timestamp_ms: int = 0
    received_at_ms: int = 0
    local_track_id: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    score: float = 0.0
    crop_path: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    # PATCH (2026-06-15, persistent-id): s3 URI of the full BGR frame
    # for this detection (uploaded to MinIO by the vendor pipeline's
    # RedisSideChannel). The TransReID sidecar downloads the frame,
    # crops the bbox in numpy, and feeds the crop to TransReID. In
    # B2 mode the strongbaseline ReID is disabled so no per-crop
    # MinIO writes exist; ``frame_uri`` is the only path to real
    # features.
    frame_uri: Optional[str] = None


class TrackletCollector:
    """Per-local-track state machine + async emission."""

    def __init__(
        self,
        *,
        pg: PostgresStore,
        redis: RedisState,
        minio: MinioStore,
        site_id: str,
        zone_rows: list[dict],
        min_track_age_frames: int = 15,
        min_crops_per_tracklet: int = 5,
        max_crops_per_tracklet: int = 15,
        min_person_height_px: float = 60.0,
        camera_width: int = 1920,
        camera_height: int = 1080,
        stale_tracklet_seconds: float = 30.0,
        auto_finalize: bool = True,
    ) -> None:
        self.pg = pg
        self.redis = redis
        self.minio = minio
        self.site_id = site_id
        self.zones_by_cam = parse_zones(zone_rows)
        self.min_track_age_frames = min_track_age_frames
        self.min_crops = min_crops_per_tracklet
        self.max_crops = max_crops_per_tracklet
        self.min_person_height_px = min_person_height_px
        self.camera_width = camera_width
        self.camera_height = camera_height
        # PATCH-024 fix: 5 s was too aggressive (caused ID fragmentation).
        # Default to 30 s — a tracklet is only closed after 30 s of silence.
        self.stale_tracklet_seconds = stale_tracklet_seconds
        self._lock = threading.RLock()
        # local_track_id -> { Tracklet | None when in collection phase }
        self._in_flight: dict[tuple[str, int], Tracklet] = {}
        # PATCH (2026-06-15): background finalize loop. In production
        # mode the main loop never calls finalize_stale() (because
        # runner.stream() blocks — PaddleDetection's subprocess owns
        # the .mp4 reading). This background thread runs the same
        # finalize+emit cycle on a 5 s interval so tracklets
        # emitted from on_detection() actually get flushed to
        # stream:tracklets. Without this, persistent-ID would never
        # progress past the in-flight state.
        self._auto_finalize = auto_finalize
        self._finalize_stop = threading.Event()
        self._finalize_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background finalize loop (if enabled)."""
        if not self._auto_finalize:
            return
        if self._finalize_thread is not None:
            return
        self._finalize_thread = threading.Thread(
            target=self._finalize_loop, daemon=True, name="tracklet-auto-finalize"
        )
        self._finalize_thread.start()
        logger.info("TrackletCollector auto-finalize loop started")

    def stop(self) -> None:
        """Stop the background finalize loop."""
        self._finalize_stop.set()
        if self._finalize_thread is not None:
            self._finalize_thread.join(timeout=2.0)
            self._finalize_thread = None

    def _finalize_loop(self) -> None:
        """Background loop: every 5s, finalize stale tracklets and emit
        them to stream:tracklets. This is the persistent-ID chain's
        heartbeat in production mode where runner.stream() blocks.

        PATCH (2026-06-15): use TRACKLET_IDLE_TIMEOUT_MS from env
        (default 3000ms = 3s) for the stale window. The original
        default of 30s was too long for production-mode persistent
        ID; operators expect tracklets to close within a few seconds
        of silence so the resolver can fire.
        """
        import os as _os
        idle_ms = int(_os.environ.get("TRACKLET_IDLE_TIMEOUT_MS", "3000"))
        idle_sec = max(0.5, idle_ms / 1000.0)
        while not self._finalize_stop.is_set():
            try:
                closed = self.finalize_stale(max_age_seconds=idle_sec)
                if closed:
                    self.emit_closed_tracklets(closed)
            except Exception as e:  # noqa: BLE001
                logger.warning("auto-finalize error: %s", e)
            self._finalize_stop.wait(timeout=5.0)

    def _zone_for(self, camera_id: str, bbox) -> Optional[str]:
        zones = self.zones_by_cam.get(camera_id, [])
        z = assign_bbox(bbox, zones, self.camera_width, self.camera_height)
        return z.zone_id if z else None

    def on_frame(self, result: FrameResult) -> Optional[Tracklet]:
        """Consume one frame. Returns a *closed* tracklet if one became
        ready, else None.
        """
        if result.skipped or result.frame is None:
            return None
        with self._lock:
            self._on_frame_locked(result)
        return None

    def _on_frame_locked(self, result: FrameResult) -> None:
        for tr in result.tracks:
            if tr.age_frames < self.min_track_age_frames:
                continue
            key = (tr.camera_id, tr.local_track_id)
            tl = self._in_flight.get(key)
            if tl is None:
                tl = Tracklet(
                    tracklet_id=str(uuid.uuid4()),
                    camera_id=tr.camera_id,
                    local_track_id=tr.local_track_id,
                    start_time=tr.ts,
                    site_id=self.site_id,
                    start_zone_id=self._zone_for(tr.camera_id, tr.bbox),
                )
                self._in_flight[key] = tl
            # Collect a crop
            crop = crop_with_padding(result.frame, tr.bbox)
            q = crop_quality_score(crop, tr.bbox, min_height_px=self.min_person_height_px)
            if q <= 0.0:
                continue
            ts = now_ts()
            zone_id = self._zone_for(tr.camera_id, tr.bbox)
            try:
                uri = self.minio.put_crop(
                    site_id=self.site_id,
                    camera_id=tr.camera_id,
                    zone_id=zone_id or "Z_NONE",
                    ts=ts,
                    global_id="UNASSIGNED",
                    tracklet_id=tl.tracklet_id,
                    image=crop,
                    kind="debug",
                    frame_id=result.frame_id,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("minio put_crop failed: %s", e)
                continue
            self.redis.append_crop(tr.camera_id, tr.local_track_id, uri)
            tl.crop_uris.append(uri)
            tl.crop_quality_scores.append(q)
            tl.frame_count += 1
            tl.end_zone_id = zone_id
            tl.end_time = tr.ts

    def on_detection(self, event: DetectionEvent) -> Optional[Tracklet]:
        """Consume one structured detection event from stream:detections.

        The side-channel carries bbox + score + (optionally) a 256-dim
        ReID embedding. We do NOT have a frame image here, so we cannot
        upload a MinIO crop — but we can:
          1. Update the in-flight tracklet's frame_count, end_time, last
             bbox/score (so finalize_stale() closes it correctly).
          2. Append the embedding to Tracklet.embeddings (preferred by
             ReIDWorker over the MinIO download path).
        The first time a (camera_id, local_track_id) is seen, we open
        a new in-flight tracklet.
        """
        with self._lock:
            self._on_detection_locked(event)
        return None

    def _on_detection_locked(self, event: DetectionEvent) -> None:
        key = (event.camera_id, event.local_track_id)
        tl = self._in_flight.get(key)
        ts = event.timestamp_ms / 1000.0 if event.timestamp_ms else now_ts()
        if tl is None:
            tl = Tracklet(
                tracklet_id=str(uuid.uuid4()),
                camera_id=event.camera_id,
                local_track_id=event.local_track_id,
                start_time=ts,
                site_id=self.site_id,
                start_zone_id=self._zone_for(event.camera_id, event.bbox),
            )
            self._in_flight[key] = tl
        # Update last-seen
        tl.end_time = ts
        tl.end_zone_id = self._zone_for(event.camera_id, event.bbox)
        # frame_count counts detections, not frames. We treat each
        # detection as one "frame" for the min_track_age_frames gate.
        tl.frame_count += 1
        tl.embedding_count = len(tl.embeddings) + (1 if event.embedding is not None else 0)
        if event.embedding is not None and event.embedding.size > 0:
            # Cap to max_crops to bound memory
            if len(tl.embeddings) < self.max_crops:
                tl.embeddings.append(np.asarray(event.embedding, dtype=np.float32))
        # PATCH (2026-06-15, persistent-id): capture the (frame_uri,
        # bbox) pair so the TransReID sidecar can download the full
        # BGR frame and crop the bbox at ReID time. B2 mode disables
        # the strongbaseline ReID model that used to write per-crop
        # JPEGs to MinIO, so the tracklet's ``crop_uris`` is empty in
        # B2. ``frame_uris`` is the alternative path: full frame
        # PATCH (2026-06-15, transreid-only, operator spec):
        # always capture the bbox (it drives the sidecar's crop
        # step). The frame_uri is optional: when present, the
        # sidecar downloads from MinIO; when None (which happens
        # when ``PPHUMAN_SKIP_FRAME_UPLOAD=1`` is set), the
        # sidecar falls back to its RTSP ring buffer (see
        # ``RTSPFrameBuffer.get_frame()`` in
        # ``app/reid/transreid_sidecar.py``). The sidecar's
        # best-effort BGR frame + the bbox gives a usable crop
        # for TransReID inference. The 5-15 frame time-skew
        # between the PP-Human frame_id and the RTSP buffer's
        # own counter is fine for cross-camera dedup (both
        # query and candidate are skewed identically).
        if event.bbox is not None and len(tl.frame_bboxes) < self.max_crops:
            tl.frame_bboxes.append(event.bbox)
            tl.frame_uris.append(event.frame_uri)  # may be None

    def on_sahi_detection(
        self,
        *,
        camera_id: str,
        frame_id: int,
        timestamp_ms: int,
        bbox: tuple,
        score: float,
    ) -> Optional[Tracklet]:
        """Create a synthetic tracklet for a SAHI-only detection.

        PATCH (2026-06-16, SAHI integration): the SAHITrackletBridge
        calls this method when a SAHI detection does NOT match any
        active PP-Human track. The resulting tracklet has
        source="sahi" and provisional=True. Downstream chain
        (ReIDWorker, GlobalIdentityResolver) treats it as
        low-confidence until seen in 2+ frames.
        """
        with self._lock:
            return self._on_sahi_detection_locked(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                bbox=bbox,
                score=score,
            )

    def _on_sahi_detection_locked(
        self,
        *,
        camera_id: str,
        frame_id: int,
        timestamp_ms: int,
        bbox: tuple,
        score: float,
    ) -> Optional[Tracklet]:
        ts = timestamp_ms / 1000.0 if timestamp_ms else now_ts()
        # PATCH (2026-06-16, SAHI integration): one tracklet per
        # (camera_id, frame_id) for SAHI — there is no MOT, so we
        # cannot assign a true local_track_id. Using the frame_id
        # as the synthetic local id keeps it stable for that frame
        # and trivially distinct across frames.
        key = (camera_id, frame_id)
        if key in self._in_flight:
            # Already have a SAHI tracklet for this frame; update
            # end_time and bump frame_count / embedding_count.
            tl = self._in_flight[key]
            tl.end_time = ts
            tl.embedding_count += 1
            return None
        tl = Tracklet(
            tracklet_id=str(uuid.uuid4()),
            camera_id=camera_id,
            local_track_id=frame_id,
            start_time=ts,
            end_time=ts,
            site_id=self.site_id,
            end_zone_id=None,
            source="sahi",
            provisional=True,
        )
        # Capture the bbox so finalize_stale has something to write
        # to PG/stream:tracklets. The side-channel path (ReIDWorker)
        # does not download from MinIO for SAHI tracklets — it
        # already receives the embedding via the latest:* key or
        # the stream event side-channel.
        tl.frame_bboxes.append(tuple(bbox))
        tl.frame_count = 1
        self._in_flight[key] = tl
        return None

    def finalize_stale(self, max_age_seconds: Optional[float] = None) -> list[Tracklet]:
        """Close any tracklets that have stopped emitting tracks. Returns
        the list of closed tracklets.

        ``max_age_seconds`` overrides the constructor default; the
        default is 30 s (was 5 s; raised in PATCH-024 / BUG-023 to avoid
        ID fragmentation from short gaps).
        """
        if max_age_seconds is None:
            max_age_seconds = self.stale_tracklet_seconds
        with self._lock:
            return self._finalize_stale_locked(max_age_seconds)

    def _finalize_stale_locked(self, max_age_seconds: float) -> list[Tracklet]:
        now = now_ts()
        closed: list[Tracklet] = []
        for key, tl in list(self._in_flight.items()):
            if tl.end_time is None or (now - tl.end_time) < max_age_seconds:
                continue
            # PATCH (2026-06-15): relax the min_crops gate for side-
            # channel tracklets. A side-channel tracklet (built via
            # on_detection) carries no MinIO crop_uris because there's
            # no frame image — only bbox + score + embedding. We
            # accept it as long as the detection count is high
            # enough (min_track_age_frames) OR embeddings are present.
            has_enough_crops = len(tl.crop_uris) >= self.min_crops
            has_enough_detections = tl.frame_count >= self.min_track_age_frames
            has_embeddings = len(tl.embeddings) > 0
            if not (has_enough_crops or has_enough_detections or has_embeddings):
                logger.debug(
                    "Dropping short tracklet %s (crops=%d, frames=%d, embeddings=%d)",
                    tl.tracklet_id,
                    len(tl.crop_uris),
                    tl.frame_count,
                    len(tl.embeddings),
                )
                del self._in_flight[key]
                continue
            # PATCH-009 fix: pick the highest-quality crop as best_crop_uri.
            if tl.crop_uris and tl.crop_quality_scores:
                best_idx = max(range(len(tl.crop_uris)), key=lambda i: tl.crop_quality_scores[i])
                tl.best_crop_uri = tl.crop_uris[best_idx]
                tl.quality_score = tl.crop_quality_scores[best_idx]
            elif tl.crop_uris:
                tl.best_crop_uri = tl.crop_uris[0]
                tl.quality_score = 0.0
            closed.append(tl)
            del self._in_flight[key]
            self.redis.clear_buffer(tl.camera_id, tl.local_track_id)
        return closed

    def emit_closed_tracklets(self, closed: list[Tracklet]) -> None:
        """Persist closed tracklets to PostgreSQL + publish to Redis Stream."""
        for tl in closed:
            # PATCH-029: best.jpg is now also uploaded (not just debug
            # crops). best_crop_uri was set by finalize_stale.
            if tl.best_crop_uri is not None and self.minio is not None:
                try:
                    self.minio.copy_object_within_bucket(
                        src_key=tl.best_crop_uri.replace(
                            f"s3://{self.minio.bucket}/",
                            "",
                        ),
                        dst_key=self.minio.evidence_key(
                            site_id=self.site_id,
                            camera_id=tl.camera_id,
                            zone_id=tl.end_zone_id or "Z_NONE",
                            ts=tl.start_time,
                            global_id=tl.best_crop_uri.split("/")[-2]
                            if "/" in tl.best_crop_uri
                            else "UNASSIGNED",
                            tracklet_id=tl.tracklet_id,
                            kind="best",
                        ),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("best.jpg copy skipped: %s", e)
            # PATCH-009 fix: we always have a quality_score now
            self.pg.insert_tracklet(
                tracklet_id=tl.tracklet_id,
                global_id=None,
                camera_id=tl.camera_id,
                local_track_id=tl.local_track_id,
                start_time=tl.start_time,
                end_time=tl.end_time,
                start_zone_id=tl.start_zone_id,
                end_zone_id=tl.end_zone_id,
                best_crop_uri=tl.best_crop_uri,
                quality_score=tl.quality_score,
                frame_count=tl.frame_count,
                embedding_count=0,
            )
            # Publish to Redis Stream for the ReID worker
            self.redis.publish(
                self._tracklet_stream(),
                {
                    "tracklet_id": tl.tracklet_id,
                    "camera_id": tl.camera_id,
                    "local_track_id": tl.local_track_id,
                    "start_time": tl.start_time,
                    "end_time": tl.end_time,
                    "site_id": self.site_id,
                    "quality_score": tl.quality_score,
                    "best_crop_uri": tl.best_crop_uri,
                    "crop_uris": tl.crop_uris[: self.max_crops],
                    # PATCH (2026-06-15, persistent-id): full-frame
                    # URIs and bboxes for the TransReID sidecar to
                    # use. The B2 pipeline writes these; the sidecar
                    # downloads each frame and crops the bbox at
                    # ReID time. ``crop_uris`` is empty in B2 mode.
                    "frame_uris": tl.frame_uris[: self.max_crops],
                    "frame_bboxes": [
                        list(b) for b in tl.frame_bboxes[: self.max_crops]
                    ],
                    # PATCH (2026-06-16, SAHI integration): source
                    # and provisional flags. The ReIDWorker and
                    # GlobalIdentityResolver use these to apply the
                    # SAHI-specific weights (higher ambiguity margin,
                    # require 2+ sightings before global_id).
                    "source": tl.source,
                    "provisional": tl.provisional,
                },
            )
            logger.info(
                "Closed tracklet %s cam=%s local_id=%d crops=%d q=%.3f",
                tl.tracklet_id,
                tl.camera_id,
                tl.local_track_id,
                len(tl.crop_uris),
                tl.quality_score or 0.0,
            )

    @staticmethod
    def _tracklet_stream() -> str:
        return "stream:tracklets"
