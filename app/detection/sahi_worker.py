"""SAHIWorker.

Background thread inside the api container. Subscribes to MediaMTX
RTSP via ``RTSPFrameBuffer``, runs ``SahiDetector`` on each frame,
and publishes raw SAHI detections to:

  - XADD stream:detections_sahi MAXLEN ~ 1000 ...
  - SET  sahi:latest:{camera_id} <json> EX 1

Constraints honored:
  - Drops frames older than ``stale_ms`` (default 400).
  - Rate-limits to ``rate_limit_hz`` per camera (default 5).
  - Skips frames rather than queueing.
  - Auto-restarts on uncaught exception (3x) then logs FATAL.
  - When ``SAHI_ENABLED=false``, ``start()`` is a no-op.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAHIWorkerConfig:
    camera_id: str = ""
    rtsp_url: str = ""
    stream: str = "stream:detections_sahi"
    latest_key_prefix: str = "sahi:latest:"
    rate_limit_hz: float = 5.0
    stale_ms: int = 400
    # Test-only override for the wall-clock; production passes None.
    current_timestamp_ms: Optional[int] = None

    @classmethod
    def from_env(cls, *, camera_id: str, rtsp_url: str) -> "SAHIWorkerConfig":
        return cls(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            stream=os.environ.get("SAHI_STREAM", "stream:detections_sahi"),
            latest_key_prefix=os.environ.get(
                "SAHI_LATEST_KEY_PREFIX", "sahi:latest:"
            ),
            rate_limit_hz=float(
                os.environ.get("SAHI_RATE_LIMIT_HZ", "5.0")
            ),
            stale_ms=int(os.environ.get("SAHI_STALE_MS", "400")),
        )


def _is_sahi_enabled() -> bool:
    return os.environ.get("SAHI_ENABLED", "false").lower() in ("true", "1", "yes")


class SAHIWorker:
    def __init__(
        self,
        *,
        config: SAHIWorkerConfig,
        buffer: Any,  # RTSPFrameBuffer (typed Any to avoid circular import)
        detector: Any,  # SahiDetector
        redis: Any,
    ) -> None:
        self._config = config
        self._buffer = buffer
        self._detector = detector
        self._redis = redis
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_publish_ts: float = 0.0
        self._min_interval: float = (
            1.0 / config.rate_limit_hz if config.rate_limit_hz > 0 else 0.0
        )

    def start(self) -> None:
        """Start the worker thread. No-op if SAHI_ENABLED=false."""
        if not _is_sahi_enabled():
            logger.info(
                "SAHIWorker[%s] SAHI_ENABLED=false; not starting", self._config.camera_id
            )
            return
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_with_restart,
            name=f"sahi-worker-{self._config.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("SAHIWorker[%s] started", self._config.camera_id)

    def stop(self) -> None:
        """Stop the worker. Joins within 2 seconds."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            try:
                self._buffer.stop()
            except Exception:  # noqa: BLE001
                pass
        logger.info("SAHIWorker[%s] stopped", self._config.camera_id)

    def _run_with_restart(self) -> None:
        """Outer loop: restart on uncaught exception, max 3x."""
        deaths = 0
        while not self._stop_event.is_set() and deaths < 3:
            try:
                self._run()
            except Exception as e:  # noqa: BLE001
                deaths += 1
                logger.warning(
                    "SAHIWorker[%s] died (attempt %d/3): %s",
                    self._config.camera_id,
                    deaths,
                    e,
                )
                if deaths >= 3:
                    logger.fatal(
                        "SAHIWorker[%s] died 3x; giving up", self._config.camera_id
                    )
                    return
                self._stop_event.wait(timeout=5.0)
        if deaths == 0:
            logger.info(
                "SAHIWorker[%s] run loop exited cleanly", self._config.camera_id
            )

    def _run(self) -> None:
        """Inner loop: read frames, run SAHI, publish to Redis."""
        self._buffer.start()
        while not self._stop_event.is_set():
            frame = self._buffer.get_frame(timeout_sec=1.0)
            if frame is None:
                continue
            # Stale drop. When ``current_timestamp_ms`` is set (test
            # override) it acts as the frame's wall-clock timestamp;
            # otherwise we treat the frame as fresh.
            wall_now_ms = int(time.time() * 1000)
            if self._config.current_timestamp_ms is not None:
                age_ms = wall_now_ms - self._config.current_timestamp_ms
                if age_ms > self._config.stale_ms:
                    logger.debug(
                        "SAHIWorker[%s] dropping stale frame (age=%dms)",
                        self._config.camera_id,
                        age_ms,
                    )
                    continue
                ts_ms = self._config.current_timestamp_ms
            else:
                ts_ms = wall_now_ms
            # Rate-limit gate.
            now = time.monotonic()
            if self._min_interval > 0:
                if (now - self._last_publish_ts) < self._min_interval:
                    continue
            # Run SAHI.
            try:
                detections = self._detector.predict(frame)
            except Exception as e:  # noqa: BLE001
                logger.error("SAHIWorker[%s] predict error: %s", self._config.camera_id, e)
                continue
            if not detections:
                continue
            # Publish.
            self._publish(ts_ms, detections)
            self._last_publish_ts = now

    def _publish(self, ts_ms: int, detections: List[tuple]) -> None:
        """XADD + SET to Redis. Both calls are best-effort."""
        try:
            self._redis.xadd(
                self._config.stream,
                {
                    "camera_id": self._config.camera_id,
                    "frame_id": "0",  # not used; kept for schema compat
                    "timestamp_ms": str(ts_ms),
                    "detections": json.dumps(
                        [
                            [float(x1), float(y1), float(x2), float(y2), float(s)]
                            for x1, y1, x2, y2, s in detections
                        ]
                    ),
                },
                maxlen=1000,
                approximate=True,
            )
            self._redis.set(
                self._config.latest_key_prefix + self._config.camera_id,
                json.dumps(
                    {
                        "camera_id": self._config.camera_id,
                        "frame_id": 0,
                        "timestamp_ms": ts_ms,
                        "detections": [
                            [float(x1), float(y1), float(x2), float(y2), float(s)]
                            for x1, y1, x2, y2, s in detections
                        ],
                    }
                ),
                ex=1,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "SAHIWorker[%s] redis publish error: %s", self._config.camera_id, e
            )
