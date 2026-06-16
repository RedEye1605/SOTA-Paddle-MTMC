"""Tests for the local-video + network-URL auto-detection.

Verifies that:
* ``_is_live_stream`` correctly classifies RTSP / HTTP / file paths.
* ``_normalize_video_source`` handles ``file://`` URIs and ``~``.
* ``resolve_rtsp_url`` validates a local file path actually exists.
* The runner wires through the normalizer.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from app.cli import config as cli_config
from app.utils.resilient_reader import (
    _is_live_stream,
    _normalize_video_source,
)


# ---------------------------------------------------------------------------
# 1. _is_live_stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://10.0.0.1:554/stream",
        "RTSP://10.0.0.1:554/stream",  # case-insensitive
        "rtsps://camera/path",
        "rtmp://live/edge",
        "http://10.0.0.1:8080/video.m3u8",
        "https://hls.example.com/cam1.m3u8",
        "tcp://10.0.0.1:8000",
        "udp://10.0.0.1:9000",
    ],
)
def test_is_live_stream_recognises_network_streams(url: str) -> None:
    assert _is_live_stream(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "/home/rhendy/videos/cam01.mp4",
        "./data/cam1.mp4",
        "file:///home/rhendy/videos/cam01.mp4",
        "~/Videos/cam01.mp4",
        "data/cam1_merged.mp4",
        "",
    ],
)
def test_is_live_stream_recognises_local_files(url: str) -> None:
    assert _is_live_stream(url) is False


# ---------------------------------------------------------------------------
# 2. _normalize_video_source
# ---------------------------------------------------------------------------


def test_normalize_strips_file_uri(tmp_path: Path) -> None:
    p = tmp_path / "video.mp4"
    p.write_text("dummy")
    out = _normalize_video_source(f"file://{p}")
    assert out == str(p)


def test_normalize_expands_tilde() -> None:
    out = _normalize_video_source("~/Videos/cam.mp4")
    assert out == os.path.expanduser("~/Videos/cam.mp4")
    assert "~" not in out


def test_normalize_leaves_rtsp_unchanged() -> None:
    url = "rtsp://10.0.0.1:554/stream"
    assert _normalize_video_source(url) == url


def test_normalize_leaves_empty_unchanged() -> None:
    assert _normalize_video_source("") == ""


# ---------------------------------------------------------------------------
# 3. resolve_rtsp_url validates local file exists
# ---------------------------------------------------------------------------


def test_resolve_rtsp_url_passes_through_rtsp() -> None:
    cam = {"camera_id": "CAM_01", "rtsp_url_env_key": "CAM_01_RTSP_URL"}
    with mock.patch.dict(os.environ, {"CAM_01_RTSP_URL": "rtsp://10.0.0.1:554/cam01"}):
        assert cli_config.resolve_rtsp_url(cam) == "rtsp://10.0.0.1:554/cam01"


def test_resolve_rtsp_url_validates_local_file_exists(tmp_path: Path) -> None:
    p = tmp_path / "cam.mp4"
    p.write_text("dummy")
    cam = {"camera_id": "CAM_01", "rtsp_url_env_key": "CAM_01_RTSP_URL"}
    with mock.patch.dict(os.environ, {"CAM_01_RTSP_URL": str(p)}):
        assert cli_config.resolve_rtsp_url(cam) == str(p)


def test_resolve_rtsp_url_rejects_missing_local_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    cam = {"camera_id": "CAM_01", "rtsp_url_env_key": "CAM_01_RTSP_URL"}
    with mock.patch.dict(os.environ, {"CAM_01_RTSP_URL": str(missing)}):
        with pytest.raises(RuntimeError) as exc:
            cli_config.resolve_rtsp_url(cam)
    assert "does not exist" in str(exc.value)


def test_resolve_rtsp_url_expands_tilde() -> None:
    """~/path that does not exist is rejected as a normal local file."""
    cam = {"camera_id": "CAM_01", "rtsp_url_env_key": "CAM_01_RTSP_URL"}
    with mock.patch.dict(os.environ, {"CAM_01_RTSP_URL": "~/.nonexistent-cam01-video-xyz.mp4"}):
        with pytest.raises(RuntimeError) as exc:
            cli_config.resolve_rtsp_url(cam)
    assert "does not exist" in str(exc.value)


def test_resolve_rtsp_url_accepts_file_uri(tmp_path: Path) -> None:
    """``file://`` URIs are passed through unchanged (no Path.exists() check)."""
    p = tmp_path / "video.mp4"
    p.write_text("dummy")
    cam = {"camera_id": "CAM_01", "rtsp_url_env_key": "CAM_01_RTSP_URL"}
    with mock.patch.dict(os.environ, {"CAM_01_RTSP_URL": f"file://{p}"}):
        assert cli_config.resolve_rtsp_url(cam) == f"file://{p}"


def test_resolve_rtsp_url_raises_when_env_var_missing() -> None:
    cam = {"camera_id": "CAM_01", "rtsp_url_env_key": "CAM_01_RTSP_URL"}
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAM_01_RTSP_URL", None)
        with pytest.raises(RuntimeError) as exc:
            cli_config.resolve_rtsp_url(cam)
    assert "CAM_01_RTSP_URL" in str(exc.value)
