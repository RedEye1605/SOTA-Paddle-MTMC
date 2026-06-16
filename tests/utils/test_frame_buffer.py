"""Tests for the shared RTSPFrameBuffer.

These tests use a fake RTSP source (a local MP4 served by the test fixture
in tests/fixtures/sample_2_frames.mp4). If the fixture is missing, the
tests skip with a clear message.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from app.utils.frame_buffer import RTSPFrameBuffer

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_MP4 = FIXTURE_DIR / "sample_2_frames.mp4"

pytestmark = pytest.mark.skipif(
    not SAMPLE_MP4.exists(),
    reason=f"test fixture not present: {SAMPLE_MP4}",
)


def test_rtsp_frame_buffer_connects_and_reads():
    """A buffer started against a local MP4 should yield BGR frames."""
    buf = RTSPFrameBuffer(url=str(SAMPLE_MP4), camera_id="TEST", ring_size=10)
    buf.start()
    try:
        frame = buf.get_frame(timeout_sec=5.0)
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
    finally:
        buf.stop()


def test_rtsp_frame_buffer_returns_none_on_timeout():
    """A buffer reading an unreachable source should not block forever."""
    # 127.0.0.1:1 is unreachable; the buffer should fail to connect.
    buf = RTSPFrameBuffer(
        url="rtsp://127.0.0.1:1/nope",
        camera_id="TEST",
        ring_size=10,
    )
    buf.start()
    try:
        frame = buf.get_frame(timeout_sec=2.0)
        assert frame is None
    finally:
        buf.stop()


def test_rtsp_frame_buffer_stop_joins_thread():
    """stop() must join the reader thread within 2 seconds."""
    buf = RTSPFrameBuffer(url=str(SAMPLE_MP4), camera_id="TEST", ring_size=10)
    buf.start()
    time.sleep(0.1)  # let it connect
    t0 = time.monotonic()
    buf.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0
