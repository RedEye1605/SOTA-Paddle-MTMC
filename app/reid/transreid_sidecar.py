"""TransReID sidecar — runs in the eval image (torch-enabled).

This is the operator's chosen path for real ReID features: the api
image stays Paddle-only (no torch), while a sidecar process running
in the eval image consumes ``stream:tracklets``, runs the real
TransReID backbone (vit_base_patch16_224_TransReID, MSMT17-pretrained
via ``/models/vit_transreid_msmt.pth``), and writes 3840-dim
L2-normalized embeddings to Qdrant
(``person_reid_transreid_msmt``).

It then emits a compact summary to ``stream:embeddings`` with the
mean vector and Qdrant point_ids, so the api's
``GlobalIdentityResolver`` can read it like any other embedding
event — no schema change required in the resolver.

The sidecar uses the existing :class:`TransReIDAdapter` (which loads
the operator's MSMT17 checkpoint, with ``ignore_classifier_head=True``
and weights_only=True) — no new model code is introduced; only a
new stream-driven run loop.

Hard rules:
  * No SHA-256 placeholder, no histogram fallback. If the real
    model fails to load, the sidecar refuses to start (matches
    the adapter's production-safety contract).
  * The sidecar must not break the api's HLS path. It is an
    XREADGROUP consumer of ``stream:tracklets`` and an XADD producer
    of ``stream:embeddings`` — both additive to the existing
    pipeline.

Configuration (env, set by docker-compose):
  * ``TRANSREID_WEIGHT``: path to the .pth (default
    ``/models/vit_transreid_msmt.pth``).
  * ``REID_SIDECAR_PROFILE``: ``msmt17`` (default) — selects
    ``num_class=1041, embedding_dim=3840``.
  * ``REID_SIDECAR_DEVICE``: ``cuda`` (default) or ``cpu``.
  * ``REID_SIDECAR_RUN``: ``1`` to enable the sidecar run loop in
    the main api (for smoke tests). The eval-image service uses
    ``app.reid.transreid_sidecar:run_sidecar`` as its entrypoint.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Optional

import numpy as np

from ..telemetry.metrics import REGISTRY
from ..utils.frame_buffer import RTSPFrameBuffer
from ..utils.time import now_ts
from .base import ReIDConfig
from .transreid_adapter import TransReIDAdapter

logger = logging.getLogger(__name__)


SIDECAR_STREAM_TRACKLETS = "stream:tracklets"
SIDECAR_STREAM_EMBEDDINGS = "stream:embeddings"
SIDECAR_CONSUMER_GROUP = "reid_sidecar_workers"
SIDECAR_COLLECTION = "person_reid_transreid_msmt"
SIDECAR_MODEL_NAME = "transreid_msmt"


def build_sidecar_adapter(
    *,
    weight_path: Optional[str] = None,
    profile: str = "msmt17",
    device: Optional[str] = None,
    use_fp16: bool = True,
) -> TransReIDAdapter:
    """Construct the production TransReID adapter with the
    MSMT17 profile (num_class=1041, embedding_dim=3840).
    """
    weight = weight_path or os.environ.get(
        "TRANSREID_WEIGHT", "/models/vit_transreid_msmt.pth"
    )
    if device is None:
        device = os.environ.get("REID_SIDECAR_DEVICE", "cuda")
    # The msmt17 profile pins num_class=1041 and embedding_dim=3840
    # (5 x 768 JPM concat, L2-normalized).
    return TransReIDAdapter(
        config=ReIDConfig(
            name=SIDECAR_MODEL_NAME,
            embedding_dim=3840,
            qdrant_collection=SIDECAR_COLLECTION,
            use_fp16=use_fp16,
            input_size=(128, 256),
        ),
        weight_path=weight,
        weights_only=True,
        num_class=1041,
        profile=profile,
        camera_num=0,  # SIE disabled for inference
        view_num=0,
        device=device,
        ignore_classifier_head=True,  # feature-extractor only
        require_checkpoint_in_production=True,
    )


class TransReIDSidecar:
    """Consumes ``stream:tracklets`` and writes real TransReID embeddings.

    The sidecar:
      1. Pulls crops from MinIO (s3://bucket/key URIs from
         ``Tracklet.crop_uris``).
      2. Runs the TransReID backbone on each crop.
      3. L2-normalizes and upserts each 3840-dim vector to
         ``person_reid_transreid_msmt`` with the standard payload
         schema.
      4. Emits a compact summary to ``stream:embeddings`` with the
         mean vector + Qdrant point_ids, so the api resolver can
         consume it.

    If crop downloads fail for a tracklet, the sidecar logs and
    skips — never fabricates.
    """

    def __init__(
        self,
        *,
        adapter: TransReIDAdapter,
        qdrant,
        pg,
        redis,
        minio,
        model_version: str = "v1",
        consumer_name: str = "reid-sidecar-01",
        max_crops: int = 8,
    ) -> None:
        self.adapter = adapter
        self.qdrant = qdrant
        self.pg = pg
        self.redis = redis
        self.minio = minio
        self.model_version = model_version
        self.consumer_name = consumer_name
        self.max_crops = max_crops

    def _download_crop(self, uri: str) -> Optional[np.ndarray]:
        import cv2

        if not uri.startswith("s3://"):
            return None
        if self.minio is None:
            return None
        bucket = self.minio.bucket
        prefix = f"s3://{bucket}/"
        if not uri.startswith(prefix):
            return None
        key = uri[len(prefix) :]
        try:
            resp = self.minio.client.get_object(bucket_name=bucket, object_name=key)
            data = resp.read()
            resp.close()
            resp.release_conn()
        except Exception as e:  # noqa: BLE001
            logger.warning("sidecar MinIO get_object failed for %s: %s", uri, e)
            return None
        if not data:
            return None
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return None
        return img

    def _load_crops(self, crop_uris, tracklet_id: str) -> list[np.ndarray]:
        crops: list[np.ndarray] = []
        for uri in list(crop_uris)[: self.max_crops]:
            img = self._download_crop(uri)
            if img is None:
                continue
            crops.append(img)
        if not crops:
            return []
        return crops

    def _load_crops_from_frames(
        self,
        frame_uris: list[str],
        frame_bboxes: list[tuple[float, float, float, float]],
        tracklet_id: str,
    ) -> list[np.ndarray]:
        """Download each (frame_uri, bbox) pair and crop the bbox in pure
        numpy.

        This is the B2 path: the strongbaseline ReID is disabled, so no
        per-crop JPEGs are written to MinIO. The vendor pipeline writes
        full BGR frames to ``s3://{bucket}/frames/{run_id}/{camera}/
        {frame_id:09d}.jpg`` and includes the URI in the side-channel
        event. The TrackletCollector forwards (frame_uri, bbox) pairs
        on the tracklet event, and the sidecar finishes the work.

        Returns a list of BGR crops aligned 1:1 with the input list
        (None entries for failed downloads / degenerate bboxes are
        silently dropped).
        """
        crops: list[np.ndarray] = []
        for uri, bbox in zip(
            list(frame_uris)[: self.max_crops],
            list(frame_bboxes)[: self.max_crops],
            strict=False,
        ):
            img = self._download_crop(uri)
            if img is None:
                logger.debug(
                    "sidecar: frame download failed for %s (tracklet=%s)",
                    uri,
                    tracklet_id,
                )
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except Exception:  # noqa: BLE001
                continue
            h, w = img.shape[:2]
            # Clamp to the image; degenerate bboxes (x2<=x1, y2<=y1)
            # become empty.
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

    def _load_crops_from_rtsp_buffer(
        self,
        frame_buffer,
        camera_id: str,
        frame_bboxes: list[tuple[float, float, float, float]],
        tracklet_id: str,
    ) -> list[np.ndarray]:
        """Pull the latest BGR frame from the RTSP ring buffer
        and apply each bbox. Used when the MinIO frame upload
        is disabled (``PPHUMAN_SKIP_FRAME_UPLOAD=1``).

        ``frame_buffer`` is the ``camera_id`` -> ``RTSPFrameBuffer``
        mapping exposed by ``run_sidecar``; we look up the per-camera
        buffer for ``camera_id`` and pull its most recent BGR frame.
        """
        crops: list[np.ndarray] = []
        # ``frame_buffer`` may be either a dict (new API, post-2026-06-16)
        # or a single buffer (legacy; kept for backward compat with tests
        # that construct ``TransReIDSidecar`` directly).
        per_cam = (
            frame_buffer.get(camera_id)
            if isinstance(frame_buffer, dict)
            else frame_buffer
        )
        if per_cam is None:
            return crops
        for bbox in frame_bboxes[: self.max_crops]:
            frame = per_cam.get_frame(timeout_sec=1.0)
            if frame is None:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except Exception:  # noqa: BLE001
                continue
            h, w = frame.shape[:2]
            x1 = max(0, min(int(round(x1)), w - 1))
            y1 = max(0, min(int(round(y1)), h - 1))
            x2 = max(0, min(int(round(x2)), w))
            y2 = max(0, min(int(round(y2)), h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
        return crops

    def process_tracklet(self, fields: dict) -> bool:
        """Process one tracklet from ``stream:tracklets``.

        Returns True if at least one embedding was written.

        PATCH (2026-06-15, persistent-id): prefer ``frame_uris`` over
        ``crop_uris``. In B2 mode the strongbaseline ReID model is
        disabled and no per-crop JPEGs are written; the only path to
        real features is the full-frame URI emitted by the vendor
        pipeline's RedisSideChannel. TrackletCollector captures both
        the URI and the bbox, and the sidecar downloads each frame
        and crops the bbox at ReID time.
        """
        import uuid as _uuid

        started = time.perf_counter()
        tracklet_id = fields.get("tracklet_id", "")
        camera_id = fields.get("camera_id", "")
        local_track_id = int(fields.get("local_track_id", 0) or 0)
        site_id = fields.get("site_id", "")
        end_zone_id = fields.get("end_zone_id")
        quality_score = float(fields.get("quality_score") or 0.0)
        crop_uris = list(fields.get("crop_uris", []) or [])
        frame_uris = list(fields.get("frame_uris", []) or [])
        frame_bboxes_raw = fields.get("frame_bboxes", []) or []
        frame_bboxes = [tuple(b) for b in frame_bboxes_raw]
        if not tracklet_id:
            return False
        # PATCH (2026-06-15, transreid-only, operator spec): the
        # operator's spec is "no per-frame MinIO upload". The
        # vendor pipeline sets ``frame_uris=[]`` (skipped MinIO
        # PUT) so the sidecar falls back to the RTSP frame
        # buffer. The buffer subscribes to the same RTSP stream
        # that PP-Human is pushing to MediaMTX and gives us a
        # best-effort BGR frame for the crop. Skew is fine for
        # cross-camera dedup since both query and candidate are
        # skewed identically.
        frame_buffer = getattr(self, "frame_buffer", None)
        # PATCH (2026-06-15): ``frame_uris`` may contain ``None``
        # placeholders when the per-frame MinIO upload was
        # skipped (see tracklet_collector on_detection). Filter
        # to keep only valid ``s3://`` URIs.
        valid_frame_uris = [u for u in frame_uris if u]
        if valid_frame_uris and frame_bboxes:
            crops = self._load_crops_from_frames(
                valid_frame_uris, frame_bboxes, tracklet_id
            )
        elif crop_uris:
            crops = self._load_crops(crop_uris, tracklet_id)
        elif frame_buffer is not None and frame_bboxes:
            # Fallback: pull the latest BGR frame from the
            # RTSP ring buffer and apply the tracklet's bbox.
            crops = self._load_crops_from_rtsp_buffer(
                frame_buffer, camera_id, frame_bboxes, tracklet_id
            )
        else:
            crops = []
        if not crops:
            logger.warning(
                "sidecar: tracklet %s has no decodable crops; skipping", tracklet_id
            )
            try:
                REGISTRY.reid_extractions_dropped_total.inc()
            except Exception:  # noqa: BLE001
                pass
            return False
        embeddings = self.adapter.extract(crops)
        if embeddings is None or len(embeddings) == 0:
            logger.warning(
                "sidecar: TransReID extracted 0 embeddings for tracklet %s", tracklet_id
            )
            return False
        # Re-normalize (defensive; the adapter already L2-normalizes
        # before returning).
        embeddings = np.asarray(embeddings, dtype=np.float32)
        for i in range(embeddings.shape[0]):
            n = np.linalg.norm(embeddings[i])
            if n > 1e-8:
                embeddings[i] = embeddings[i] / n
        mean_vec = embeddings.mean(axis=0)
        n = np.linalg.norm(mean_vec)
        if n > 1e-8:
            mean_vec = mean_vec / n
        mean_vec = mean_vec.astype(np.float32)
        # Upsert each per-crop embedding to the MSMT Qdrant collection
        # with deterministic point_ids.
        written_point_ids: list[str] = []
        ts_ms = int(now_ts())
        for i, vec in enumerate(embeddings):
            pid = str(
                _uuid.uuid5(
                    _uuid.NAMESPACE_URL,
                    f"{SIDECAR_MODEL_NAME}:{tracklet_id}:{i:02d}",
                )
            )
            payload = {
                "global_id": None,  # filled by the resolver's backfill
                "tracklet_id": tracklet_id,
                "camera_id": camera_id,
                "local_track_id": local_track_id,
                "zone_id": end_zone_id or "",
                "site_id": site_id,
                "timestamp": ts_ms,
                "quality_score": quality_score,
                "model_name": SIDECAR_MODEL_NAME,
                "model_version": self.model_version,
                "embedding_version": self.model_version,
                "crop_uri": crop_uris[i] if i < len(crop_uris) else "",
            }
            try:
                self.qdrant.upsert_point(
                    SIDECAR_COLLECTION, vec, payload, point_id=pid
                )
                written_point_ids.append(pid)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "sidecar: Qdrant upsert failed for %s idx=%d: %s",
                    tracklet_id, i, e,
                )
        if len(written_point_ids) != len(embeddings):
            logger.warning(
                "sidecar: only wrote %d/%d Qdrant points for %s; retrying later",
                len(written_point_ids),
                len(embeddings),
                tracklet_id,
            )
            return False
        for point_id in written_point_ids:
            try:
                self.pg.insert_tracklet_embedding(
                    tracklet_id=tracklet_id,
                    global_id=None,
                    camera_id=camera_id,
                    model_name=SIDECAR_MODEL_NAME,
                    model_version=self.model_version,
                    vector_db_collection=SIDECAR_COLLECTION,
                    vector_db_point_id=point_id,
                    embedding_dim=3840,
                    quality_score=quality_score,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "sidecar: pg.insert_tracklet_embedding failed for %s: %s",
                    point_id,
                    e,
                )
                return False
        # Emit mean embedding summary to stream:embeddings for the
        # resolver. The resolver will read it like any other
        # embedding event (model_name="transreid_msmt").
        try:
            self.redis.publish(
                SIDECAR_STREAM_EMBEDDINGS,
                {
                    "tracklet_id": tracklet_id,
                    "camera_id": camera_id,
                    "local_track_id": local_track_id,
                    "site_id": site_id,
                    "ts": ts_ms,
                    "model_name": SIDECAR_MODEL_NAME,
                    "model_version": self.model_version,
                    "embedding_version": self.model_version,
                    "qdrant_collection": SIDECAR_COLLECTION,
                    "embedding_dim": 3840,
                    "mean_vec": mean_vec.tolist(),
                    "end_zone_id": end_zone_id,
                    "quality_score": quality_score,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("sidecar: publish stream:embeddings failed: %s", e)
            return False
        try:
            REGISTRY.reid_extractions.inc(amount=len(embeddings))
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "sidecar: tracklet=%s cam=%s local=%d crops=%d points=%d ms=%.0f",
            tracklet_id, camera_id, local_track_id, len(crops), len(written_point_ids),
            (time.perf_counter() - started) * 1000,
        )
        return True

    def run(self, stop_event=None) -> None:
        """Consume ``stream:tracklets`` in a loop."""
        self.redis.ensure_group(SIDECAR_STREAM_TRACKLETS, SIDECAR_CONSUMER_GROUP)
        logger.info(
            "TransReID sidecar running: collection=%s model=%s weight=%s device=%s",
            SIDECAR_COLLECTION,
            SIDECAR_MODEL_NAME,
            self.adapter._weight_path,
            self.adapter._device_name,
        )
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            msgs = self.redis.consume(
                SIDECAR_STREAM_TRACKLETS,
                SIDECAR_CONSUMER_GROUP,
                self.consumer_name,
                count=4,
                block_ms=1000,
            )
            for msg_id, fields in msgs:
                if not fields.get("tracklet_id") or not fields.get("camera_id"):
                    logger.warning(
                        "sidecar invalid tracklet message %s; acking poison entry",
                        msg_id,
                    )
                    try:
                        self.redis.ack(
                            SIDECAR_STREAM_TRACKLETS,
                            SIDECAR_CONSUMER_GROUP,
                            msg_id,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("sidecar poison ack failed for %s: %s", msg_id, e)
                    continue
                try:
                    processed = self.process_tracklet(fields)
                except Exception as e:  # noqa: BLE001
                    logger.exception("sidecar process_tracklet failed: %s", e)
                    continue
                if not processed:
                    continue
                try:
                    self.redis.ack(
                        SIDECAR_STREAM_TRACKLETS,
                        SIDECAR_CONSUMER_GROUP,
                        msg_id,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "sidecar ack failed for %s/%s: %s",
                        SIDECAR_STREAM_TRACKLETS,
                        msg_id,
                        e,
                    )


def run_sidecar() -> None:
    """Entry point for the eval-image service.

    Constructs the real TransReID adapter (no fallback), wires up
    the sidecar consumer + an RTSP frame buffer, and blocks on the
    run loop. SIGTERM triggers a clean shutdown.
    """
    import logging as _logging

    _logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Lazy imports so the api image (no torch) does not need them.
    # The four storage modules each expose a module-level ``from_env()``
    # factory (not a classmethod) — match the convention used by
    # ``app/main.py``.
    from ..storage.redis_state import from_env as redis_from_env
    from ..storage.minio_store import from_env as minio_from_env
    from ..storage.postgres import from_env as pg_from_env
    from ..storage.qdrant_store import from_env as qdrant_from_env

    redis = redis_from_env()
    redis.connect()
    qdrant = qdrant_from_env()
    qdrant.connect()
    qdrant.init_collections()  # creates person_reid_transreid_msmt if missing
    pg = pg_from_env()
    pg.connect()
    minio = minio_from_env()
    minio.connect()

    adapter = build_sidecar_adapter()
    adapter.load()
    if adapter._fallback_active:
        raise RuntimeError(
            "sidecar: TransReID weights could not be loaded and the "
            "production safety net refused to start. Check "
            f"TRANSREID_WEIGHT={adapter._weight_path} and torch install."
        )
    adapter.warmup()

    sidecar = TransReIDSidecar(
        adapter=adapter,
        qdrant=qdrant,
        pg=pg,
        redis=redis,
        minio=minio,
    )

    # PATCH (2026-06-15, transreid-only, operator spec): the
    # sidecar needs BGR frames to crop the bbox. The vendor
    # pipeline's side-channel can upload full frames to MinIO
    # (200-2000 ms per frame, throttles the chain) but the
    # operator's spec is "transreid-only + no per-frame MinIO
    # upload". Instead, the sidecar subscribes to the RTSP
    # stream that PP-Human is pushing to MediaMTX and maintains
    # a small per-camera ring buffer of recent BGR frames. When
    # a tracklet comes in, the sidecar finds the frame by
    # (camera_id, frame_id) and crops it.
    #
    # PATCH (2026-06-16, SAHI integration): the
    # ``RTSPFrameBuffer`` was extracted to ``app.utils.frame_buffer``
    # as a per-camera primitive (one URL, one buffer, one thread).
    # The sidecar builds one buffer per known camera and exposes
    # the ``camera_id`` -> ``RTSPFrameBuffer`` mapping as
    # ``sidecar.frame_buffer`` so ``_load_crops_from_rtsp_buffer``
    # can look up the right buffer.
    rtsp_base = os.environ.get(
        "SIDECAR_RTSP_BASE", "rtsp://198.51.100.20:8554/sota-paddle-mtmc"
    ).rstrip("/")
    cam_to_basename = {
        "CAM_01": "cam1_merged",
        "CAM_02": "cam2_merged",
    }
    frame_buffers: dict[str, RTSPFrameBuffer] = {
        cam_id: RTSPFrameBuffer(
            url=f"{rtsp_base}/{basename}",
            camera_id=cam_id,
            ring_size=300,
        )
        for cam_id, basename in cam_to_basename.items()
    }
    for buf in frame_buffers.values():
        buf.start()
    sidecar.frame_buffer = frame_buffers

    stop = _Event()

    def _on_signal(signum, _frame):  # noqa: ANN001
        logger.info("sidecar: received signal %s; stopping", signum)
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        sidecar.run(stop_event=stop)
    finally:
        for buf in frame_buffers.values():
            buf.stop()


# Lightweight stdlib Event so we don't import threading at module
# load time. The eval image has torch; this is fine.
class _Event:
    def __init__(self) -> None:

        self._e = threading.Event()

    def set(self) -> None:
        self._e.set()

    def is_set(self) -> bool:
        return self._e.is_set()
