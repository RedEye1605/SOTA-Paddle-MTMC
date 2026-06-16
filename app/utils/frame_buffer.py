"""Shared RTSP frame buffer.

Used by both ``SAHIWorker`` (api process, Paddle SAHI inference) and
``TransReIDSidecar`` (eval image, torch TransReID inference). The class
subscribes to an RTSP feed, decodes BGR frames, and serves the latest
frame to callers via a thread-safe ring buffer.

Force UDP transport for MediaMTX by setting OPENCV_FFMPEG_CAPTURE_OPTIONS
at import time.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Force UDP transport for MediaMTX. Set before any VideoCapture opens.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp"
)


class RTSPFrameBuffer:
    """Thread-safe ring buffer of BGR frames from an RTSP source.

    Parameters
    ----------
    url : str
        RTSP URL (e.g., ``rtsp://mediamtx:8554/cam1_merged/``).
    camera_id : str
        Logical camera name, used in log lines.
    ring_size : int
        Maximum frames kept in the ring. Older frames are dropped.
    """

    def __init__(
        self,
        *,
        url: str,
        camera_id: str,
        ring_size: int = 300,
    ) -> None:
        self.url = url
        self.camera_id = camera_id
        self.ring_size = ring_size
        self._frames: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=ring_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """Start the reader thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rtsp-buffer-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the reader thread. Joins within 2 seconds."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None

    def get_frame(self, *, timeout_sec: float = 1.0) -> Optional[np.ndarray]:
        """Return the most recent BGR frame, or None on timeout/disconnect."""
        try:
            return self._frames.get(timeout=timeout_sec)
        except queue.Empty:
            return None

    def _run(self) -> None:
        """Reader loop. Reconnects with exponential backoff on failure."""
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if not self._cap.isOpened():
                    raise RuntimeError(f"failed to open {self.url}")
                logger.info(
                    "RTSPFrameBuffer[%s] connected to %s", self.camera_id, self.url
                )
                backoff = 1.0
                self._read_loop()  # returns normally on stop, raises on read failure
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "RTSPFrameBuffer[%s] error: %s; sleeping %.1fs",
                    self.camera_id,
                    e,
                    backoff,
                )
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    self._cap = None
                # Sleep with stop-event awareness.
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, 30.0)

    def _read_loop(self) -> None:
        """Inner read loop. Returns on stop; raises on read failure (triggers reconnect)."""
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise RuntimeError("read failed")
            self._drop_oldest_and_put(frame)

    def _drop_oldest_and_put(self, frame: np.ndarray) -> None:
        """Drop the oldest frame if queue is full, then put the new frame.

        Invariant: this is the single producer of self._frames. The second
        put_nowait cannot fail under that invariant.
        """
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass  # queue drained between full-check and get; retry below
            self._frames.put_nowait(frame)
