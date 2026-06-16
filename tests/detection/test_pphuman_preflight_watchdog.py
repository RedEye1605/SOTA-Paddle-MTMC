"""Tests for the PP-Human decoder preflight + runtime watchdog.

Pinned contract (per FixReports/UNIFIED_STREAM_2026-06-14.md §5):

* :func:`preflight_video_source` returns a :class:`PreflightResult`
  with codec, width, height, fps, duration for any file OpenCV
  can open.
* It returns a :class:`PreflightResult` with ``ok=False`` and a
  non-empty ``error`` for missing files, empty files, and
  unreadable files (e.g. a plain text file masquerading as mp4).
* The public path-mapping helper
  :func:`expected_publish_path` returns the URL PP-Human would
  publish to given a pushurl_base and a video_file path. This
  is the operator-facing contract assertion.
* :class:`StreamWatchdog` raises after ``stall_timeout_seconds``
  with no frames seen, exposes ``healthy`` and ``stall_reason``.
* :class:`StreamWatchdog` reports healthy only after at least one
  frame has been recorded via :meth:`note_frame` AND the
  subprocess is still alive.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. preflight_video_source
# ---------------------------------------------------------------------------


def test_preflight_passes_on_valid_smoke_clip(tmp_path: Path) -> None:
    """Real smoke clip — exists, has bytes, OpenCV opens first frame."""
    # Arrange: the data/smoke/CAM_01.mp4 file we just created in the
    # host working dir. If it isn't there (CI without data), skip
    # rather than fail.
    smoke = Path(__file__).resolve().parents[2] / "data" / "smoke" / "CAM_01.mp4"
    if not smoke.exists():
        pytest.skip("smoke clip not built; run scripts/build_smoke_clips.sh first")

    from app.detection.pphuman_pipeline import preflight_video_source

    result = preflight_video_source(smoke)

    assert result.ok is True
    assert result.error == ""
    # codec is best-effort (ffprobe when available, "" otherwise);
    # the test only checks that the preflight didn't crash.
    assert isinstance(result.codec, str)
    assert result.width == 960
    assert result.height == 540
    assert result.fps > 0
    assert result.duration_seconds > 0
    assert result.size_bytes > 0


def test_preflight_fails_on_missing_file(tmp_path: Path) -> None:
    from app.detection.pphuman_pipeline import preflight_video_source

    missing = tmp_path / "does-not-exist.mp4"
    result = preflight_video_source(missing)

    assert result.ok is False
    assert result.error  # non-empty
    assert "missing" in result.error.lower() or "not found" in result.error.lower()


def test_preflight_fails_on_empty_file(tmp_path: Path) -> None:
    from app.detection.pphuman_pipeline import preflight_video_source

    empty = tmp_path / "empty.mp4"
    empty.touch()
    result = preflight_video_source(empty)

    assert result.ok is False
    assert "empty" in result.error.lower() or "0" in result.error


def test_preflight_fails_on_unreadable_file(tmp_path: Path) -> None:
    """A text file named .mp4 — file exists, has bytes, but OpenCV
    cannot decode the first frame."""
    from app.detection.pphuman_pipeline import preflight_video_source

    fake = tmp_path / "fake.mp4"
    fake.write_text("not really a video")
    result = preflight_video_source(fake)

    assert result.ok is False
    assert result.error  # non-empty
    # The error should mention decoding / opening
    assert (
        "decode" in result.error.lower()
        or "open" in result.error.lower()
        or "frame" in result.error.lower()
    )


def test_preflight_returns_size_bytes(tmp_path: Path) -> None:
    """The preflight reports the file size so operators can spot
    half-downloaded / truncated sources."""
    from app.detection.pphuman_pipeline import preflight_video_source

    smoke = Path(__file__).resolve().parents[2] / "data" / "smoke" / "CAM_01.mp4"
    if not smoke.exists():
        pytest.skip("smoke clip not built; run scripts/build_smoke_clips.sh first")

    result = preflight_video_source(smoke)
    assert result.size_bytes == smoke.stat().st_size


# ---------------------------------------------------------------------------
# 2. expected_publish_path
# ---------------------------------------------------------------------------


def test_expected_publish_path_cam_01_basename() -> None:
    """The public contract: CAM_01.mp4 → sota-paddle-mtmc/CAM_01."""
    from app.detection.pphuman_pipeline import expected_publish_path

    pushurl = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    got = expected_publish_path(pushurl, "/data/smoke/CAM_01.mp4")
    assert got == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/CAM_01"


def test_expected_publish_path_cam_02_basename() -> None:
    from app.detection.pphuman_pipeline import expected_publish_path

    pushurl = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    got = expected_publish_path(pushurl, "/data/smoke/CAM_02.mp4")
    assert got == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/CAM_02"


def test_expected_publish_path_legacy_merged_collision() -> None:
    """Pinned regression test: with the old `cam1_merged.mp4`
    filename, PP-Human would have published to
    `sota-paddle-mtmc/cam1_merged`. The expected_publish_path
    helper must surface this so the operator sees the path
    mismatch in the preflight report."""
    from app.detection.pphuman_pipeline import expected_publish_path

    pushurl = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    got = expected_publish_path(pushurl, "/data/cam1_merged.mp4")
    assert got == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/cam1_merged"


def test_expected_publish_path_handles_path_without_extension() -> None:
    """Defensive: if the operator forgets the .mp4, the helper
    shouldn't crash and the result is the same basename."""
    from app.detection.pphuman_pipeline import expected_publish_path

    pushurl = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    got = expected_publish_path(pushurl, "/data/smoke/CAM_01")
    assert got == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/CAM_01"


# ---------------------------------------------------------------------------
# 3. StreamWatchdog
# ---------------------------------------------------------------------------


def test_watchdog_starts_unhealthy() -> None:
    """A fresh watchdog with no frames seen reports unhealthy."""
    from app.detection.pphuman_pipeline import StreamWatchdog

    wd = StreamWatchdog(stall_timeout_seconds=10.0)
    assert wd.healthy is False
    assert wd.stall_reason == "no_frames_yet"


def test_watchdog_healthy_after_first_frame() -> None:
    from app.detection.pphuman_pipeline import StreamWatchdog

    wd = StreamWatchdog(stall_timeout_seconds=60.0)
    wd.note_frame(frame_id=1, ts=time.monotonic())
    assert wd.healthy is True
    assert wd.stall_reason == ""


def test_watchdog_flags_stall_after_timeout() -> None:
    """If no frames for `stall_timeout_seconds`, healthy flips
    False and stall_reason explains."""
    from app.detection.pphuman_pipeline import StreamWatchdog

    wd = StreamWatchdog(stall_timeout_seconds=0.05)  # 50ms
    wd.note_frame(frame_id=1, ts=time.monotonic())
    assert wd.healthy is True
    time.sleep(0.10)
    # Force re-evaluation by querying healthy
    assert wd.healthy is False
    assert "stall" in wd.stall_reason.lower() or "timeout" in wd.stall_reason.lower()


def test_watchdog_resets_after_frame() -> None:
    """A fresh frame resets the stall timer."""
    from app.detection.pphuman_pipeline import StreamWatchdog

    wd = StreamWatchdog(stall_timeout_seconds=0.05)
    wd.note_frame(frame_id=1, ts=time.monotonic())
    time.sleep(0.03)
    wd.note_frame(frame_id=2, ts=time.monotonic())
    time.sleep(0.03)
    assert wd.healthy is True  # still alive, last frame < timeout ago


def test_watchdog_subprocess_died_marks_unhealthy() -> None:
    """If the subprocess exits but frames had been flowing, the
    watchdog flips unhealthy with a different reason."""
    from app.detection.pphuman_pipeline import StreamWatchdog

    wd = StreamWatchdog(stall_timeout_seconds=60.0)
    wd.note_frame(frame_id=1, ts=time.monotonic())
    assert wd.healthy is True
    wd.note_subprocess_exit(returncode=1)
    assert wd.healthy is False
    assert "subprocess" in wd.stall_reason.lower() or "exit" in wd.stall_reason.lower()


def test_watchdog_clean_exit_is_acceptable_when_no_frames_expected() -> None:
    """For a short smoke clip that finishes naturally, the
    subprocess exits with rc=0 — that is NOT a stall. The
    watchdog reports the natural completion as a benign state."""
    from app.detection.pphuman_pipeline import StreamWatchdog

    wd = StreamWatchdog(stall_timeout_seconds=60.0)
    wd.note_frame(frame_id=300, ts=time.monotonic())  # clip finished
    wd.note_subprocess_exit(returncode=0)
    # Healthy is sticky: once frames were emitted, the smoke
    # succeeded, so the watchdog reports healthy=False only on
    # a stall, not on a clean end-of-clip. The caller can ask
    # ``finished_cleanly`` separately if it cares.
    assert wd.healthy is True
    assert wd.finished_cleanly is True


# ---------------------------------------------------------------------------
# 4. Watchdog wiring into PPHumanFrameStateAdapter (UNIFIED_STREAM_2026-06-14)
# ---------------------------------------------------------------------------


def test_frame_state_adapter_watchdog_notes_frames() -> None:
    """``PPHumanFrameStateAdapter`` exposes a ``watchdog`` attribute
    and ``_tail_loop`` calls ``note_frame`` for every detection
    it receives from the manager — so the watchdog becomes
    healthy as soon as the first MOT line is parsed.
    """
    from app.detection.pphuman_pipeline import (
        PPHumanDetectorAdapter,
        PPHumanFrameStateAdapter,
        PPHumanPipelineSubprocessManager,
    )
    from app.detection.pphuman_pipeline import Detection

    # Stub adapter + manager: we don't launch a real PP-Human.
    adapter = PPHumanDetectorAdapter()
    PPHumanPipelineSubprocessManager(
        adapter=adapter,
        cameras=[("CAM_01", "stub")],
    )

    class _StubManager:
        def __init__(self, detections):
            self._detections = list(detections)
        def start(self):
            pass
        def stream(self):
            for cam, det in self._detections:
                yield cam, det
        crashed_cameras: set = set()

    state = PPHumanFrameStateAdapter(manager=_StubManager([
        ("CAM_01", Detection(frame_id=1, track_id=1, bbox=(0,0,10,10), confidence=0.9)),
    ]))
    # Start the tailer thread; it will consume the stub stream
    # and call ``note_frame``. Wait for it.
    state.start()
    state._tailer.join(timeout=2.0)
    assert state.watchdog.healthy is True
    assert state.watchdog.last_frame_id == 1
    state.stop()


def test_frame_state_adapter_watchdog_flips_unhealthy_on_manager_crash() -> None:
    """When the manager's ``crashed_cameras`` becomes non-empty,
    the adapter's ``crashed_cameras`` property marks the watchdog
    with a non-zero exit so the operator gets an explicit
    failure rather than a silent stall.
    """
    from app.detection.pphuman_pipeline import (
        PPHumanFrameStateAdapter,
    )

    class _StubManager:
        def start(self):
            pass
        def stream(self):
            # Yield nothing — empty stream.
            if False:
                yield None, None  # pragma: no cover
        crashed_cameras: set = set()

    state = PPHumanFrameStateAdapter(manager=_StubManager())
    # No frames seen, no exit → unhealthy with "no_frames_yet".
    assert state.watchdog.healthy is False
    assert "no_frames" in state.watchdog.stall_reason
    # Simulate manager crash.
    state._crashed_cameras.add("CAM_01")
    crashed = state.crashed_cameras
    assert "CAM_01" in crashed
    # Reading crashed_cameras should have flipped the watchdog
    # via note_subprocess_exit(returncode=1).
    assert state.watchdog.healthy is False
    assert "subprocess" in state.watchdog.stall_reason.lower()
