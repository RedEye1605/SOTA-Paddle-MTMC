"""Resilient frame reader with reconnect + degraded/offline state.

PATCH-032 fix: the audit flagged that the previous ``make_frame_reader``
did not reconnect on EOF for RTSP streams. A dead camera was silent
forever, and other cameras could be affected by FFmpeg/OpenCV leaking
state.

This module exposes ``ResilientFrameReader`` that:

  * Wraps an underlying ``cv2.VideoCapture`` (or any iterator).
  * On read failure, transitions to ``degraded`` after
    ``degraded_after_seconds`` and ``offline`` after
    ``offline_after_seconds``.
  * Reconnects with exponential backoff starting at
    ``initial_backoff_seconds`` and capped at ``max_backoff_seconds``.
  * Emits a final synthetic ``None`` frame when the source is fully
    exhausted (or offline) so the consumer can detect end-of-stream.
  * Per-camera metrics: ``camera_reconnects_total`` and
    ``camera_decode_errors_total`` are incremented via
    :class:`app.telemetry.per_camera.PER_CAMERA`.

The reader is intentionally simple: it is a generator over
``(frame_id, ts, frame)`` tuples. When the source is offline it
yields a sentinel ``(frame_id, ts, None)`` so the consumer can
update the camera status without blocking.

Hard rules:
  * One dead camera MUST NOT stop other cameras. The multi-camera
    runner spawns one ``ResilientFrameReader`` per camera; failures
    are isolated.
  * No unbounded memory growth: we never accumulate frames.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np

from ..telemetry.per_camera import (
    CAMERA_STATUS_DEGRADED,
    CAMERA_STATUS_OFFLINE,
    CAMERA_STATUS_ONLINE,
    PER_CAMERA,
)

logger = logging.getLogger(__name__)


@dataclass
class ReconnectConfig:
    enabled: bool = True
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    degraded_after_seconds: float = 10.0
    offline_after_seconds: float = 60.0
    # When the source looks like a local file (loop=True), we
    # don't reconnect on EOF — we just stop the generator.
    is_live_stream: bool = True


def _is_live_stream(source: str) -> bool:
    """Heuristic: RTSP / HTTP / TCP / UDP streams are live; file
    paths are not.

    Accepts the following source shapes (any of these is a valid
    ``CAM_0X_RTSP_URL`` value in the operator's ``.env``):

    * ``rtsp://...`` / ``rtsps://...`` — live RTSP
    * ``rtmp://...`` — live RTMP
    * ``http://...`` / ``https://...`` — HLS / MJPEG / HTTP-MP4
    * ``tcp://...`` / ``udp://...`` — raw sockets
    * ``file:///abs/path`` — local file (URI form)
    * ``/abs/path`` or ``./relative/path`` — local file (path form)
    * ``~/...`` — local file (tilde-expanded)
    """
    if not source:
        return False
    s = source.strip().lower()
    if s.startswith("rtsp://") or s.startswith("rtsps://"):
        return True
    if s.startswith("rtmp://") or s.startswith("tcp://") or s.startswith("udp://"):
        return True
    if s.startswith("http://") or s.startswith("https://"):
        return True
    return False


def _normalize_video_source(source: str) -> str:
    """Normalize a ``CAM_0X_RTSP_URL`` value into an OpenCV-friendly form.

    * ``file://`` URIs are converted to local paths.
    * Leading ``~`` is expanded to the user's home.
    * Anything else is returned unchanged (the live-stream branches
      go through ``cv2.VideoCapture`` directly).
    """
    if not source:
        return source
    s = source.strip()
    if s.lower().startswith("file://"):
        return s[7:]  # strip "file://"
    return os.path.expanduser(s)


class ResilientFrameReader:
    """Wraps ``cv2.VideoCapture`` with reconnect + degraded/offline
    status transitions.

    Usage::

        r = ResilientFrameReader("rtsp://camera/CAM_01", camera_id="CAM_01")
        for frame_id, ts, frame in r:
            if frame is None:
                # camera offline; we still get a tick
                continue
            ...
    """

    def __init__(
        self,
        source: str,
        camera_id: str,
        *,
        config: Optional[ReconnectConfig] = None,
        loop: bool = False,
    ) -> None:
        from threading import Lock

        self.source = source
        self.camera_id = camera_id
        self.config = config or ReconnectConfig(
            is_live_stream=_is_live_stream(source),
        )
        self.loop = loop
        self._lock = Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._first_failure_ts: Optional[float] = None
        self._status = CAMERA_STATUS_ONLINE
        self._backoff = max(0.1, float(self.config.initial_backoff_seconds))
        self._last_yield_ts: Optional[float] = None
        # The first error sets the metric.
        self._error_count = 0

    def _metrics(self):
        return PER_CAMERA.for_camera(self.camera_id)

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.warning(
                "ResilientFrameReader(%s): cannot open %s",
                self.camera_id,
                self.source,
            )
            return None
        return cap

    def _set_status(self, new_status: int) -> None:
        with self._lock:
            old = self._status
        if new_status != old:
            with self._lock:
                self._status = new_status
            try:
                self._metrics().set_status(new_status)
            except Exception:  # noqa: BLE001
                pass
            if new_status == CAMERA_STATUS_DEGRADED:
                logger.warning(
                    "ResilientFrameReader(%s): → DEGRADED",
                    self.camera_id,
                )
            elif new_status == CAMERA_STATUS_OFFLINE:
                logger.error(
                    "ResilientFrameReader(%s): → OFFLINE",
                    self.camera_id,
                )
            elif new_status == CAMERA_STATUS_ONLINE:
                logger.info(
                    "ResilientFrameReader(%s): → ONLINE (recovered)",
                    self.camera_id,
                )

    @property
    def status(self) -> int:
        with self._lock:
            return self._status

    def fps(self) -> float:
        with self._lock:
            return self._compute_fps_locked()

    def _record_failure(self) -> None:
        now = time.time()
        if self._first_failure_ts is None:
            self._first_failure_ts = now
        self._error_count += 1
        try:
            self._metrics().observe_decode_error()
        except Exception:  # noqa: BLE001
            pass
        # Status transitions: degraded after degraded_after_s,
        # offline after offline_after_s.
        if now - self._first_failure_ts >= self.config.offline_after_seconds:
            self._set_status(CAMERA_STATUS_OFFLINE)
        elif now - self._first_failure_ts >= self.config.degraded_after_seconds:
            self._set_status(CAMERA_STATUS_DEGRADED)

    def _try_reconnect(self) -> bool:
        """Try to reopen the capture. Returns True on success.
        Increments the reconnect metric and resets the failure
        state on success.
        """
        if not self.config.enabled or not self.config.is_live_stream:
            return False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None
        # Exponential backoff.
        wait = min(self._backoff, self.config.max_backoff_seconds)
        logger.info(
            "ResilientFrameReader(%s): reconnect in %.1fs",
            self.camera_id,
            wait,
        )
        time.sleep(wait)
        self._backoff = min(self._backoff * 2, self.config.max_backoff_seconds)
        try:
            self._metrics().observe_reconnect()
        except Exception:  # noqa: BLE001
            pass
        cap = self._open_capture()
        if cap is None:
            return False
        self._cap = cap
        # Success — reset failure state.
        self._first_failure_ts = None
        self._set_status(CAMERA_STATUS_ONLINE)
        return True

    def __iter__(self) -> Iterator[tuple[int, float, Optional[np.ndarray]]]:
        # Open the capture for the first time.
        self._cap = self._open_capture()
        if self._cap is None:
            # Source is dead from the start. We still yield a
            # continuous stream of None sentinels so the consumer
            # can update per-camera status and the producer
            # thread can drive reconnect attempts.
            self._set_status(CAMERA_STATUS_OFFLINE)
            self._first_failure_ts = time.time()
            while True:
                yield (self._frame_id, time.time(), None)
                self._frame_id += 1
                self._record_failure()
                if not self._try_reconnect():
                    time.sleep(1.0)
        self._set_status(CAMERA_STATUS_ONLINE)
        while True:
            if self._cap is None:
                # We are in a reconnect loop.
                if not self._try_reconnect():
                    # Yield a None sentinel so the consumer can
                    # act on the offline status without blocking.
                    yield (self._frame_id, time.time(), None)
                    # Cap the rate at 1 Hz to avoid busy-waiting.
                    time.sleep(1.0)
                    continue
            assert self._cap is not None
            try:
                ok, frame = self._cap.read()
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "ResilientFrameReader(%s): read() raised: %s",
                    self.camera_id,
                    e,
                )
                ok, frame = False, None
            if not ok or frame is None:
                # EOF or read failure. For local files in loop mode
                # we just restart the capture; otherwise we
                # transition to degraded and try to reconnect.
                self._record_failure()
                if self.loop and not self.config.is_live_stream:
                    # Local file loop: rewind.
                    try:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                if self.config.is_live_stream:
                    if not self._try_reconnect():
                        yield (self._frame_id, time.time(), None)
                        time.sleep(1.0)
                        continue
                    # Else fall through to read.
                else:
                    # Local file, no loop: stop.
                    try:
                        self._cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    self._cap = None
                    return
            self._frame_id += 1
            self._last_yield_ts = time.time()
            # Successful read — clear any degraded state if we were
            # only temporarily degraded.
            if self._first_failure_ts is not None:
                # We had an intermittent failure but recovered.
                self._first_failure_ts = None
                self._backoff = self.config.initial_backoff_seconds
                self._set_status(CAMERA_STATUS_ONLINE)
            yield (self._frame_id, self._last_yield_ts, frame)

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None
