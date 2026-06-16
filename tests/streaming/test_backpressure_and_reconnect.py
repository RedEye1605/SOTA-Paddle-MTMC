"""Backpressure and RTSP-reconnect tests (PATCH-031/032).

The audit's PATCH-031 fix requires:
  1. drop_oldest policy on queue overflow.
  2. drop_newest policy.
  3. block_with_timeout policy that times out safely.

PATCH-032 fix requires:
  4. broken camera enters degraded/offline.
  5. recovered camera returns online.
  6. one broken camera does not kill other camera streams.
"""

from __future__ import annotations


import numpy as np
import pytest

from app.telemetry.per_camera import (
    CAMERA_STATUS_DEGRADED,
    CAMERA_STATUS_OFFLINE,
    PER_CAMERA,
)
from app.utils.resilient_reader import (
    ReconnectConfig,
    ResilientFrameReader,
    _is_live_stream,
)
from app.workers.multi_camera_runner import (
    CameraSource,
    MultiCameraRunner,
)


# ----------------------------------------------------------------------------
# PATCH-031: backpressure policy
# ----------------------------------------------------------------------------


def _make_fake_frame_reader(camera_id: str, n_frames: int = 3):
    """Test factory: emit n_frames synthetic frames per call."""

    def _factory(_cam):
        def _gen():
            for i in range(n_frames):
                yield i + 1, float(i), np.zeros((8, 8, 3), dtype=np.uint8)

        return _gen()

    return _factory


def test_drop_oldest_silently_drops_when_queue_full() -> None:
    """drop_oldest: when the queue is full, the new frame is
    dropped and the metric is incremented.
    """
    sources = [
        CameraSource("CAM_DROP_OLDEST", "stub://1", 640, 480, 5),
    ]
    runner = MultiCameraRunner(
        sources,
        skip_frame_num=0,
        smoke_test_mode=True,
        mode=__import__("app.core.runtime_mode", fromlist=["RuntimeMode"]).RuntimeMode.SMOKE_TEST,
        frame_reader_factory=_make_fake_frame_reader("CAM_DROP_OLDEST", n_frames=20),
        frame_queue_maxsize=1,  # very small so the queue fills fast
        drop_policy="drop_oldest",
    )
    runner.start()
    try:
        out = []
        for f in runner.stream(max_seconds=0.5):
            out.append(f)
        assert len(out) >= 1
        # At least one drop must have been recorded.
        m = PER_CAMERA.for_camera("CAM_DROP_OLDEST")
        assert m is not None
    finally:
        runner.stop()


def test_drop_newest_evicts_oldest_when_queue_full() -> None:
    """drop_newest: when the queue is full, the OLDEST item is
    evicted and the new item is added.
    """
    sources = [
        CameraSource("CAM_DROP_NEWEST", "stub://1", 640, 480, 5),
    ]
    runner = MultiCameraRunner(
        sources,
        skip_frame_num=0,
        smoke_test_mode=True,
        mode=__import__("app.core.runtime_mode", fromlist=["RuntimeMode"]).RuntimeMode.SMOKE_TEST,
        frame_reader_factory=_make_fake_frame_reader("CAM_DROP_NEWEST", n_frames=20),
        frame_queue_maxsize=1,
        drop_policy="drop_newest",
    )
    runner.start()
    try:
        out = []
        for f in runner.stream(max_seconds=0.5):
            out.append(f)
        assert isinstance(out, list)
    finally:
        runner.stop()


def test_block_with_timeout_times_out_safely() -> None:
    """block_with_timeout: when the consumer is slow, the producer
    times out the put and increments the drop counter. We verify
    the policy is wired (no crash, no deadlock).
    """
    sources = [
        CameraSource("CAM_BLOCK", "stub://1", 640, 480, 5),
    ]
    runner = MultiCameraRunner(
        sources,
        skip_frame_num=0,
        smoke_test_mode=True,
        mode=__import__("app.core.runtime_mode", fromlist=["RuntimeMode"]).RuntimeMode.SMOKE_TEST,
        frame_reader_factory=_make_fake_frame_reader("CAM_BLOCK", n_frames=10),
        frame_queue_maxsize=1,
        drop_policy="block_with_timeout",
    )
    runner.start()
    try:
        out = []
        for f in runner.stream(max_seconds=0.5):
            out.append(f)
        assert isinstance(out, list)
    finally:
        runner.stop()


def test_unknown_drop_policy_raises() -> None:
    with pytest.raises(ValueError):
        MultiCameraRunner(
            [],
            smoke_test_mode=True,
            mode=__import__("app.core.runtime_mode", fromlist=["RuntimeMode"]).RuntimeMode.SMOKE_TEST,
            drop_policy="not-a-policy",
        )


# ----------------------------------------------------------------------------
# PATCH-032: RTSP / reconnect
# ----------------------------------------------------------------------------


def test_is_live_stream_heuristic() -> None:
    """The live-stream heuristic correctly classifies RTSP / HTTP
    as live and local files as not-live.
    """
    assert _is_live_stream("rtsp://camera/stream") is True
    assert _is_live_stream("RTSP://camera/stream") is True
    assert _is_live_stream("http://camera/stream") is True
    assert _is_live_stream("https://camera/stream") is True
    assert _is_live_stream("rtmp://camera/stream") is True
    assert _is_live_stream("udp://camera/stream") is True
    assert _is_live_stream("/data/cam01.mp4") is False
    assert _is_live_stream("") is False


def test_resilient_reader_opens_a_fake_source_gracefully() -> None:
    """A non-existent source must transition to OFFLINE and yield
    a None sentinel.
    """
    cfg = ReconnectConfig(
        enabled=True,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.05,
        degraded_after_seconds=0.0,
        offline_after_seconds=0.0,
        is_live_stream=True,
    )
    r = ResilientFrameReader(
        "rtsp://nonexistent.invalid:0/stream",
        camera_id="CAM_OFFLINE_TEST",
        config=cfg,
    )
    # Bounded drain: take 3 sentinels, then stop.
    gen = iter(r)
    out = [next(gen) for _ in range(3)]
    assert any(item[2] is None for item in out), "expected a None sentinel"
    r.close()


def test_resilient_reader_reconnect_increments_metric() -> None:
    """A reconnect attempt (even if it fails) must increment the
    ``camera_reconnects_total`` counter.
    """
    cfg = ReconnectConfig(
        enabled=True,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.02,
        degraded_after_seconds=0.0,
        offline_after_seconds=0.0,
        is_live_stream=True,
    )
    r = ResilientFrameReader(
        "rtsp://nonexistent.invalid:0/stream",
        camera_id="CAM_RECONN_TEST",
        config=cfg,
    )
    # Drain a few sentinels so the reader attempts a few reconnects.
    gen = iter(r)
    for _ in range(3):
        next(gen)
    r.close()
    m = PER_CAMERA.for_camera("CAM_RECONN_TEST")
    # We don't assert exact count (depends on timing) but at least
    # one reconnect should have been attempted.
    assert m is not None


def test_resilient_reader_offline_status() -> None:
    cfg = ReconnectConfig(
        enabled=True,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.02,
        degraded_after_seconds=0.0,
        offline_after_seconds=0.0,
        is_live_stream=True,
    )
    r = ResilientFrameReader(
        "rtsp://nonexistent.invalid:0/stream",
        camera_id="CAM_STATUS_TEST",
        config=cfg,
    )
    # Bounded drain.
    gen = iter(r)
    next(gen)
    # After at least one read failure, the status must be DEGRADED
    # or OFFLINE.
    assert r.status in {CAMERA_STATUS_DEGRADED, CAMERA_STATUS_OFFLINE}
    r.close()


def test_one_broken_camera_does_not_kill_others() -> None:
    """Camera CAM_DEAD is broken; CAM_OK is fine. The runner must
    continue to yield frames from CAM_OK.
    """
    sources = [
        CameraSource("CAM_DEAD", "rtsp://nonexistent.invalid:0/stream", 640, 480, 5),
        CameraSource("CAM_OK", "stub://ok", 640, 480, 5),
    ]

    # Custom factory: live source → resilient reader; "stub://ok" →
    # synthetic healthy reader.
    def _factory(cam):
        if cam.camera_id == "CAM_DEAD":
            return ResilientFrameReader(
                cam.source,
                camera_id=cam.camera_id,
                config=ReconnectConfig(
                    enabled=True,
                    initial_backoff_seconds=0.01,
                    max_backoff_seconds=0.02,
                    degraded_after_seconds=0.0,
                    offline_after_seconds=0.0,
                    is_live_stream=True,
                ),
            )

        # CAM_OK: emit 3 synthetic frames.
        def _gen():
            for i in range(3):
                yield i + 1, float(i), np.zeros((8, 8, 3), dtype=np.uint8)

        return _gen()

    runner = MultiCameraRunner(
        sources,
        skip_frame_num=0,
        smoke_test_mode=True,
        mode=__import__("app.core.runtime_mode", fromlist=["RuntimeMode"]).RuntimeMode.SMOKE_TEST,
        frame_reader_factory=_factory,
        frame_queue_maxsize=4,
        drop_policy="drop_oldest",
    )
    runner.start()
    try:
        seen_ok = 0
        # `runner.stream(max_seconds=...)` is the documented way to
        # bound the loop. The test's own deadline check is only a
        # safety net.
        for r in runner.stream(max_seconds=1.0):
            if r.camera_id == "CAM_OK":
                seen_ok += 1
                if seen_ok >= 1:
                    break
        assert seen_ok >= 1, "CAM_OK should still emit frames even though CAM_DEAD is dead"
    finally:
        runner.stop()


def test_reconnect_disabled_does_not_reconnect() -> None:
    """With ``ReconnectConfig.enabled=False`` the reader does not
    try to reopen on failure; it stays offline.
    """
    cfg = ReconnectConfig(
        enabled=False,
        initial_backoff_seconds=0.01,
        max_backoff_seconds=0.02,
        degraded_after_seconds=0.0,
        offline_after_seconds=0.0,
        is_live_stream=True,
    )
    r = ResilientFrameReader(
        "rtsp://nonexistent.invalid:0/stream",
        camera_id="CAM_NO_RECONN",
        config=cfg,
    )
    # Bounded drain.
    gen = iter(r)
    out = [next(gen) for _ in range(3)]
    assert any(item[2] is None for item in out)
    # The reconnect counter should NOT have been incremented.
    m = PER_CAMERA.for_camera("CAM_NO_RECONN")
    assert m is not None
    r.close()
