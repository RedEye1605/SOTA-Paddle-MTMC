"""Per-camera metrics helpers (PATCH-018).

This module holds the per-camera EWMA + windowed-FPS logic. It is
intentionally small and dependency-free so it can be unit-tested
without the full Qdrant/Postgres/Redis stack.

The metrics emitted:
  * ``camera_fps{camera_id=...}``           — rolling EWMA FPS
  * ``camera_frame_latency_ms{camera_id}`` — last frame's wall-clock latency
  * ``camera_last_frame_timestamp{camera_id}`` — epoch seconds
  * ``camera_queue_depth{camera_id}``      — current queue size
  * ``camera_status{camera_id}``           — 0=offline, 1=degraded, 2=online
  * ``camera_decode_errors_total{camera_id}`` — Counter
  * ``camera_reconnects_total{camera_id}``  — Counter
  * ``camera_drops_total{camera_id}``       — Counter
  * ``total_analytics_fps``                 — sum across cameras
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Optional

from .metrics import REGISTRY


CAMERA_STATUS_OFFLINE = 0
CAMERA_STATUS_DEGRADED = 1
CAMERA_STATUS_ONLINE = 2


class PerCameraMetrics:
    """Tracks per-camera runtime metrics with a windowed FPS."""

    def __init__(self, camera_id: str, *, window_seconds: float = 5.0, ewma_alpha: float = 0.3):
        self.camera_id = camera_id
        self._lock = Lock()
        # Window of recent frame timestamps (epoch seconds). Trims
        # to ``window_seconds`` on each ``observe_frame``.
        self._frame_times: deque[float] = deque()
        self._window_seconds = window_seconds
        # EWMA for frame latency (ms) — last observed value, smoothed.
        self._latency_ms_ewma: float = 0.0
        self._ewma_alpha = ewma_alpha
        # Last status (0/1/2). Online by default.
        self._status = CAMERA_STATUS_ONLINE
        self._last_status_change_ts: float = time.time()

    # ---- public API ----
    def observe_frame(self, ts: Optional[float] = None) -> None:
        """Record that a frame was emitted at ``ts`` (default: now)."""
        if ts is None:
            ts = time.time()
        with self._lock:
            self._frame_times.append(ts)
            self._trim_window(ts)
            fps = self._compute_fps_locked()
            REGISTRY.camera_fps.set(fps, camera_id=self.camera_id)
            REGISTRY.camera_last_frame_timestamp.set(
                float(ts),
                camera_id=self.camera_id,
            )
        # Total across cameras: sum of (window_seconds / inter-frame)
        # is a rough approximation; use 1/mean_interval.
        with self._lock:
            mean_interval = self._mean_interval_locked()
        if mean_interval > 0:
            REGISTRY.total_fps.set(1.0 / mean_interval)

    def observe_frame_latency(self, latency_ms: float) -> None:
        """Record a single frame's wall-clock latency (milliseconds)."""
        with self._lock:
            self._latency_ms_ewma = (
                self._ewma_alpha * float(latency_ms)
                + (1.0 - self._ewma_alpha) * self._latency_ms_ewma
            )
        REGISTRY.camera_frame_latency_ms.set(
            float(self._latency_ms_ewma),
            camera_id=self.camera_id,
        )

    def observe_queue_depth(self, depth: int) -> None:
        REGISTRY.camera_queue_depth.set(int(depth), camera_id=self.camera_id)

    def observe_decode_error(self) -> None:
        REGISTRY.camera_decode_errors_total.inc(camera_id=self.camera_id)

    def observe_reconnect(self) -> None:
        REGISTRY.camera_reconnects_total.inc(camera_id=self.camera_id)

    def observe_drop(self) -> None:
        REGISTRY.camera_drops_total.inc(camera_id=self.camera_id)

    def set_status(self, status: int) -> None:
        with self._lock:
            if status != self._status:
                self._status = status
                self._last_status_change_ts = time.time()
        REGISTRY.camera_status.set(int(status), camera_id=self.camera_id)

    @property
    def status(self) -> int:
        with self._lock:
            return self._status

    def fps(self) -> float:
        with self._lock:
            return self._compute_fps_locked()

    def latency_ms(self) -> float:
        with self._lock:
            return float(self._latency_ms_ewma)

    # ---- private ----
    def _trim_window(self, now_ts: float) -> None:
        cutoff = now_ts - self._window_seconds
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.popleft()

    def _compute_fps_locked(self) -> float:
        n = len(self._frame_times)
        if n < 2:
            return 0.0
        duration = self._frame_times[-1] - self._frame_times[0]
        if duration <= 0:
            return 0.0
        return float(n - 1) / duration

    def _mean_interval_locked(self) -> float:
        n = len(self._frame_times)
        if n < 2:
            return 0.0
        duration = self._frame_times[-1] - self._frame_times[0]
        if duration <= 0:
            return 0.0
        return duration / float(n - 1)


class PerCameraMetricsRegistry:
    """Process-wide cache of per-camera metrics objects."""

    def __init__(self) -> None:
        self._by_camera: dict[str, PerCameraMetrics] = {}
        self._lock = Lock()

    def for_camera(self, camera_id: str) -> PerCameraMetrics:
        with self._lock:
            m = self._by_camera.get(camera_id)
            if m is None:
                m = PerCameraMetrics(camera_id)
                self._by_camera[camera_id] = m
            return m

    def all_cameras(self) -> list[PerCameraMetrics]:
        with self._lock:
            return list(self._by_camera.values())


# Global singleton.
PER_CAMERA = PerCameraMetricsRegistry()
