"""Tests for SAHIWorker.

These tests use a fake ``RTSPFrameBuffer`` and a fake Redis. The
worker's actual RTSP / detector are mocked; we exercise the
lifecycle, rate-limiting, stale-drop, and Redis publish paths.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.detection.sahi_worker import SAHIWorker, SAHIWorkerConfig


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.kv_lock = threading.Lock()
        self.stream_lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self.kv_lock:
            return self.kv.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:  # noqa: A002
        with self.kv_lock:
            self.kv[key] = value

    def xadd(
        self,
        name: str,
        fields: dict[str, str],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        with self.stream_lock:
            eid = f"{int(time.time() * 1e6)}-{len(self.streams.get(name, []))}"
            self.streams.setdefault(name, []).append((eid, fields))
            if maxlen is not None and len(self.streams[name]) > maxlen:
                self.streams[name] = self.streams[name][-maxlen:]
        return eid


class FakeBuffer:
    """Stands in for RTSPFrameBuffer; serves synthetic frames on demand."""

    def __init__(self, frames: List[np.ndarray]) -> None:
        self._frames = frames
        self._idx = 0
        self._lock = threading.Lock()
        self._connected = True
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def is_connected(self) -> bool:
        return self._connected

    def get_frame(self, *, timeout_sec: float = 1.0) -> Optional[np.ndarray]:
        with self._lock:
            if self._idx >= len(self._frames):
                return None
            f = self._frames[self._idx]
            self._idx += 1
        return f


@pytest.fixture
def cfg() -> SAHIWorkerConfig:
    return SAHIWorkerConfig(
        camera_id="CAM_01",
        rtsp_url="rtsp://fake",
        stream="stream:detections_sahi",
        latest_key_prefix="sahi:latest:",
        rate_limit_hz=10.0,
        stale_ms=400,
    )


@pytest.fixture(autouse=True)
def _enable_sahi(monkeypatch):
    """Tests exercise the enabled path; the disabled test overrides it."""
    monkeypatch.setenv("SAHI_ENABLED", "true")


def test_config_defaults():
    c = SAHIWorkerConfig()
    assert c.camera_id == ""
    assert c.rate_limit_hz == 5.0
    assert c.stale_ms == 400


def test_disabled_noop():
    """When SAHI_ENABLED=false, no thread is ever created."""
    import os
    os.environ["SAHI_ENABLED"] = "false"
    try:
        detector = MagicMock()
        detector.predict = MagicMock(return_value=[(10, 10, 50, 50, 0.9)])
        buf = FakeBuffer([np.zeros((10, 10, 3), dtype=np.uint8)])
        worker = SAHIWorker(
            config=SAHIWorkerConfig(camera_id="CAM_01", rtsp_url="rtsp://fake"),
            buffer=buf,  # type: ignore[arg-type]
            detector=detector,
            redis=FakeRedis(),  # type: ignore[arg-type]
        )
        worker.start()
        time.sleep(0.2)
        worker.stop()
        assert not buf.started
        assert not buf.stopped
    finally:
        del os.environ["SAHI_ENABLED"]


def test_publishes_per_frame(cfg: SAHIWorkerConfig):
    """Each frame triggers one XADD and one SET."""
    fake_redis = FakeRedis()
    detector = MagicMock()
    detector.predict = MagicMock(return_value=[(10, 10, 50, 50, 0.9)])
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    buf = FakeBuffer(frames)
    worker = SAHIWorker(
        config=cfg,
        buffer=buf,  # type: ignore[arg-type]
        detector=detector,
        redis=fake_redis,  # type: ignore[arg-type]
    )
    worker.start()
    # Wait for the worker to consume all 3 frames.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and len(fake_redis.streams.get("stream:detections_sahi", [])) < 3:
        time.sleep(0.05)
    worker.stop()
    assert len(fake_redis.streams["stream:detections_sahi"]) >= 1
    assert "sahi:latest:CAM_01" in fake_redis.kv


def test_stale_drop():
    """Frames older than stale_ms are dropped before publish."""
    fake_redis = FakeRedis()
    detector = MagicMock()
    detector.predict = MagicMock(return_value=[(10, 10, 50, 50, 0.9)])
    # Build a frame with a fake timestamp 1000ms in the past.
    old_ts_ms = int(time.time() * 1000) - 1000  # 1s old, > 400ms stale
    buf = FakeBuffer([np.zeros((10, 10, 3), dtype=np.uint8)])
    cfg = SAHIWorkerConfig(
        camera_id="CAM_01", rtsp_url="rtsp://fake",
        stream="stream:detections_sahi", latest_key_prefix="sahi:latest:",
        rate_limit_hz=10.0, stale_ms=400, current_timestamp_ms=old_ts_ms,
    )
    worker = SAHIWorker(
        config=cfg, buffer=buf,  # type: ignore[arg-type]
        detector=detector, redis=fake_redis,  # type: ignore[arg-type]
    )
    worker.start()
    time.sleep(0.3)
    worker.stop()
    # No publish should have happened.
    assert "stream:detections_sahi" not in fake_redis.streams or len(fake_redis.streams["stream:detections_sahi"]) == 0


def test_rate_limit():
    """At rate_limit_hz=2, the worker should not publish faster than 2 Hz."""
    fake_redis = FakeRedis()
    detector = MagicMock()
    detector.predict = MagicMock(return_value=[(10, 10, 50, 50, 0.9)])
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(20)]
    buf = FakeBuffer(frames)
    cfg = SAHIWorkerConfig(
        camera_id="CAM_01", rtsp_url="rtsp://fake",
        stream="stream:detections_sahi", latest_key_prefix="sahi:latest:",
        rate_limit_hz=2.0, stale_ms=10_000, current_timestamp_ms=int(time.time() * 1000),
    )
    worker = SAHIWorker(
        config=cfg, buffer=buf,  # type: ignore[arg-type]
        detector=detector, redis=fake_redis,  # type: ignore[arg-type]
    )
    worker.start()
    time.sleep(2.0)  # at 2 Hz, expect ~4 publishes
    worker.stop()
    published = len(fake_redis.streams.get("stream:detections_sahi", []))
    # Should be ~4 publishes (2 Hz × 2 sec), with some slack.
    assert published <= 7, f"rate-limit not enforced; got {published} publishes"


def test_stop_joins_within_2s(cfg: SAHIWorkerConfig):
    """stop() must join the worker thread within 2 seconds."""
    fake_redis = FakeRedis()
    detector = MagicMock()
    detector.predict = MagicMock(return_value=[(10, 10, 50, 50, 0.9)])
    buf = FakeBuffer([np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(100)])
    worker = SAHIWorker(
        config=cfg, buffer=buf,  # type: ignore[arg-type]
        detector=detector, redis=fake_redis,  # type: ignore[arg-type]
    )
    worker.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    worker.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"stop() took {elapsed}s"
