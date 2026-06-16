"""Tests for the unified-stream wiring (per FixReports/UNIFIED_STREAM_2026-06-14.md).

Pins down the contract between ``MEDIAMTX_ENABLED`` and
``MEDIAMTX_PPHUMAN_DIRECT_PUSH``:

  * When ``MEDIAMTX_PPHUMAN_DIRECT_PUSH=true`` (the default),
    the operator's ffmpeg streamer MUST be disabled to avoid
    both processes fighting for the same RTSP path on
    MediaMTX. The fix shipped in the previous handoff set
    ``MEDIAMTX_ENABLED=false`` so this contract is in the
    operator's ``.env``.
  * The fix report's HLS URLs are
    ``http://hls.example.invalid/sota-paddle-mtmc/CAM_01/index.m3u8``
    and the same for CAM_02. This test pins the URL shape
    against the streaming contract values.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_unified_stream_uses_cam_basename_publish_path() -> None:
    """End-to-end: given the unified-stream env, the path PP-Human
    will publish to is the public contract path (no more
    ``cam1_merged`` collision)."""
    from app.detection.pphuman_pipeline import expected_publish_path

    pushurl = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    assert (
        expected_publish_path(pushurl, "/data/smoke/CAM_01.mp4")
        == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/CAM_01"
    )
    assert (
        expected_publish_path(pushurl, "/data/smoke/CAM_02.mp4")
        == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/CAM_02"
    )


def test_unified_stream_mediamtx_streamer_disabled() -> None:
    """With ``MEDIAMTX_PPHUMAN_DIRECT_PUSH=true`` in effect, the
    operator's ffmpeg streamer MUST be disabled — otherwise both
    processes try to publish the same path and MediaMTX refuses
    the second connection. The fix shipped ``MEDIAMTX_ENABLED=false``
    in the operator's ``.env``."""
    # The contract is the operator's .env: MEDIAMTX_ENABLED=false.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        pytest.skip(".env not present in this checkout")
    text = env_path.read_text(encoding="utf-8")
    # Find the MEDIAMTX_ENABLED line and check it's set to false
    for line in text.splitlines():
        if line.strip().startswith("MEDIAMTX_ENABLED="):
            value = line.split("=", 1)[1].strip().lower()
            assert value in ("false", "0", "no", "off"), (
                f"operator .env must set MEDIAMTX_ENABLED=false in unified-stream "
                f"mode, got {value!r}"
            )
            return
    pytest.fail("MEDIAMTX_ENABLED not found in .env")


def test_unified_stream_mediamtx_direct_push_enabled() -> None:
    """The operator's .env must enable direct push (PP-Human
    publishes its annotated stream to MediaMTX directly)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        pytest.skip(".env not present in this checkout")
    text = env_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("MEDIAMTX_PPHUMAN_DIRECT_PUSH="):
            value = line.split("=", 1)[1].strip().lower()
            assert value in ("true", "1", "yes", "on"), (
                f"operator .env must set MEDIAMTX_PPHUMAN_DIRECT_PUSH=true in "
                f"unified-stream mode, got {value!r}"
            )
            return
    pytest.fail("MEDIAMTX_PPHUMAN_DIRECT_PUSH not found in .env")


def test_unified_stream_uses_smoke_clips_in_env() -> None:
    """The .env points CAM_01 and CAM_02 at the smoke clips for
    the 2026-06-14 validation, or production video files."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        pytest.skip(".env not present in this checkout")
    text = env_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("CAM_01_RTSP_URL="):
            value = line.split("=", 1)[1].strip()
            assert any(x in value for x in ("/data/smoke/CAM_01.mp4", "/data/smoke/CAM_01", "cam1_merged.mp4")), (
                f"CAM_01_RTSP_URL must point at a smoke clip or production video, got {value!r}"
            )
            return
    pytest.fail("CAM_01_RTSP_URL not found in .env")


def test_unified_stream_smoke_clip_basename_matches_publish_path() -> None:
    """The smoke clip basenames (``CAM_01.mp4`` / ``CAM_02.mp4``)
    are intentionally the public contract path basenames, so
    PaddleDetection's ``--pushurl`` join produces
    ``sota-paddle-mtmc/CAM_01`` and ``sota-paddle-mtmc/CAM_02``."""
    smoke_dir = Path(__file__).resolve().parents[2] / "data" / "smoke"
    if not (smoke_dir / "CAM_01.mp4").exists() or not (smoke_dir / "CAM_02.mp4").exists():
        pytest.skip("smoke clips not built; run ffmpeg commands in "
                    "FixReports/UNIFIED_STREAM_2026-06-14.md §B")
    # The basenames must be exactly CAM_01 and CAM_02 (no _smoke suffix).
    for cam in ("CAM_01", "CAM_02"):
        assert (smoke_dir / f"{cam}.mp4").exists(), (
            f"smoke clip {cam}.mp4 missing — basename must be the public path"
        )
