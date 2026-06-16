"""DetectionEventConsumer — XREADGROUP consumer for ``stream:detections``.

The PP-Human subprocess (per the vendor hotfix) emits one structured
detection event per tracked detection to ``stream:detections``. This
consumer pulls those events and feeds them into
``TrackletCollector.on_detection()``, which in turn drives
``stream:tracklets`` → ``ReIDWorker`` → ``stream:embeddings`` →
``GlobalIdentityResolver`` for the persistent-ID pipeline.

Design (matches the operator's spec):
  - Consumer group ``detection_consumers`` (XREADGROUP, not XREAD)
  - Bounded in-process queue (size 1024) with drop counter
  - XACK after successful processing
  - Bounded retries; dead-letter via metric on schema/parse errors
  - Never blocks the main api loop

If Redis is down, the consumer logs and retries; TrackletCollector
remains idle (which is correct: no detections means no tracklets).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from ..storage.redis_state import RedisState
from .tracklet_collector import DetectionEvent, TrackletCollector

logger = logging.getLogger(__name__)


# Prometheus-style counter holder (lightweight; no external dep)
class _Counters:
    def __init__(self) -> None:
        self.parsed = 0
        self.dropped_queue_full = 0
        self.dropped_schema_invalid = 0
        self.acked = 0
        self.redis_errors = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "detection_events_parsed": self.parsed,
            "detection_events_dropped_queue_full": self.dropped_queue_full,
            "detection_events_dropped_schema_invalid": self.dropped_schema_invalid,
            "detection_events_acked": self.acked,
            "detection_events_redis_errors": self.redis_errors,
        }


class DetectionEventConsumer:
    """Pulls structured detection events from ``stream:detections`` and
    feeds them into ``TrackletCollector.on_detection()``.

    Args:
        redis: RedisState instance.
        collector: TrackletCollector (target sink).
        group: Consumer group name.
        consumer: This consumer's name (unique per replica).
        queue_size: Bounded in-process queue size.
        block_ms: XREADGROUP block timeout in ms.
    """

    STREAM = "stream:detections"
    GROUP = "detection_consumers"

    def __init__(
        self,
        *,
        redis: RedisState,
        collector: TrackletCollector,
        group: str = GROUP,
        consumer: str = "detection-consumer-01",
        queue_size: int = 1024,
        block_ms: int = 1000,
    ) -> None:
        self._redis = redis
        self._collector = collector
        self._group = group
        self._consumer = consumer
        self._queue_size = queue_size
        self._block_ms = block_ms
        self._queue: list[DetectionEvent] = []
        self._counters = _Counters()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending_ids: list[str] = []
        self._pending_events: list[DetectionEvent] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            self._redis.ensure_group(self.STREAM, self._group)
        except Exception as e:  # noqa: BLE001
            logger.warning("ensure_group failed (continuing): %s", e)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="detection-event-consumer"
        )
        self._thread.start()
        logger.info("DetectionEventConsumer started (group=%s)", self._group)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def counters(self) -> dict[str, int]:
        return self._counters.as_dict()

    # ------------------------------------------------------------------
    # Internal: main loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msgs = self._redis.consume(
                    self.STREAM,
                    self._group,
                    self._consumer,
                    count=16,
                    block_ms=self._block_ms,
                )
            except Exception as e:  # noqa: BLE001
                self._counters.redis_errors += 1
                logger.warning("XREADGROUP error (retrying): %s", e)
                time.sleep(0.5)
                continue
            if not msgs:
                continue
            for msg_id, payload in msgs:
                event = self._parse(payload)
                if event is None:
                    self._counters.dropped_schema_invalid += 1
                    # ACK bad messages so they don't pile up in pending
                    try:
                        self._redis.ack(self.STREAM, self._group, msg_id)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                # Enqueue; drop on overflow with counter
                if len(self._queue) >= self._queue_size:
                    self._counters.dropped_queue_full += 1
                    try:
                        self._redis.ack(self.STREAM, self._group, msg_id)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                self._queue.append(event)
                self._pending_ids.append(msg_id)
                # Drain the in-process queue synchronously (we're in a
                # dedicated thread; the collector's on_detection is
                # cheap so the drain is inline).
                self._drain()

    def _drain(self) -> None:
        while self._queue:
            event = self._queue.pop(0)
            try:
                self._collector.on_detection(event)
                self._counters.parsed += 1
            except Exception as e:  # noqa: BLE001
                # Don't ACK if processing failed; let it retry
                self._queue.insert(0, event)
                logger.warning("on_detection failed (will retry): %s", e)
                return
            # ACK the corresponding message id
            if self._pending_ids:
                msg_id = self._pending_ids.pop(0)
                try:
                    self._redis.ack(self.STREAM, self._group, msg_id)
                    self._counters.acked += 1
                except Exception as e:  # noqa: BLE001
                    logger.warning("XACK failed (id=%s): %s", msg_id, e)

    def _parse(self, payload: dict) -> Optional[DetectionEvent]:
        """Parse a Redis-stream payload into a DetectionEvent.

        Defensive: missing fields, bad types, and JSON failures all
        return None and the message is acked as schema-invalid.

        PATCH (2026-06-15): PP-Human's `video_out_name` (used as the
        basename in the RTSP push path) is the source file basename
        without extension, e.g. ``cam1_merged``. The operator's
        ``cameras`` table uses operator-chosen IDs like ``CAM_01``.
        Map ``cam1_merged`` → ``CAM_01`` (strip ``_merged``) so the
        tracklets' foreign key to ``cameras.camera_id`` resolves.
        """
        try:
            raw_camera_id = str(payload["camera_id"])
            camera_id = self._normalize_camera_id(raw_camera_id)
            frame_id = int(payload["frame_id"])
            local_track_id = int(payload["local_track_id"])
            timestamp_ms = int(payload.get("timestamp_ms", int(time.time() * 1000)))
            received_at_ms = int(
                payload.get("received_at_ms", int(time.time() * 1000))
            )
            score = float(payload.get("score", 0.0))
            bbox = payload.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return None
            bbox = [float(b) for b in bbox[:4]]
            embedding = payload.get("embedding")
            emb_arr: Optional[np.ndarray] = None
            if embedding is not None:
                try:
                    emb_arr = np.asarray(embedding, dtype=np.float32)
                except Exception:  # noqa: BLE001
                    emb_arr = None
            return DetectionEvent(
                schema_version=str(payload.get("schema_version", "1.0")),
                event_id=str(payload.get("event_id", f"det_{camera_id}_{frame_id}_{local_track_id}")),
                source=str(payload.get("source", "pphuman")),
                run_id=str(payload.get("run_id", "")),
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                received_at_ms=received_at_ms,
                local_track_id=local_track_id,
                bbox=tuple(bbox),  # type: ignore[arg-type]
                score=score,
                crop_path=payload.get("crop_path"),
                embedding=emb_arr,
                # PATCH (2026-06-15, persistent-id): full-frame s3
                # URI uploaded by the vendor pipeline's
                # RedisSideChannel. Empty string is normalised to
                # None so TrackletCollector can use ``if event.frame_uri``.
                frame_uri=(payload.get("frame_uri") or None),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("DetectionEvent parse error: %s (payload=%r)", e, payload)
            return None

    @staticmethod
    def _normalize_camera_id(raw: str) -> str:
        """Map PP-Human's file-basename camera_id to the operator's
        ``cameras.camera_id``. The current operator uses
        ``camN_merged`` file basenames which map to ``CAM_0N`` (strip
        ``_merged`` suffix, upper-case). Unknown IDs pass through."""
        s = raw.strip()
        if s.endswith("_merged"):
            s = s[: -len("_merged")]
        if s.startswith("cam") and len(s) > 3 and s[3:].isdigit():
            return "CAM_" + s[3:].zfill(2)
        return s.upper()
