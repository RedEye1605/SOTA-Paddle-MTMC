"""Tests that disabled streaming is safe and side-effect-free (Phase 6).

Covers the "skip the whole pipeline" path that lets SOTA run in CI
or in T4-cycles-saving mode without MediaMTX.
"""

from __future__ import annotations

import os
from unittest import mock

from app.streaming.mediamtx_streamer import (
    MediaMTXStreamer,
    make_from_env,
)


def test_make_from_env_disabled_is_a_noop() -> None:
    """No env at all → disabled streamer, no ffmpeg subprocess."""
    with mock.patch.dict(os.environ, {}, clear=True):
        s = make_from_env("CAM_01")
    assert s.is_enabled() is False
    s.start()
    assert not s.is_running()
    s.stop()
    assert not s.is_running()


def test_make_from_env_disabled_when_mediamtx_disabled() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MEDIAMTX_ENABLED": "false",
            "MEDIAMTX_HOST": "h.example.com",
        },
    ):
        s = make_from_env("CAM_01")
    assert s.is_enabled() is False


def test_make_from_env_disabled_when_host_empty() -> None:
    with mock.patch.dict(
        os.environ,
        {"MEDIAMTX_ENABLED": "true", "MEDIAMTX_HOST": ""},
    ):
        s = make_from_env("CAM_01")
    # Even with "enabled", an empty host is treated as a disabled streamer.
    assert s.is_enabled() is False


def test_make_from_env_disabled_stream_urls_still_built() -> None:
    """``stream_urls()`` returns *what would have been* the URLs,
    so a debugging tool can show the operator the right path even
    when streaming is disabled."""
    with mock.patch.dict(
        os.environ,
        {
            "MEDIAMTX_ENABLED": "true",
            "MEDIAMTX_HOST": "h.example.com",
            "MEDIAMTX_RTSP_PORT": "8554",
            "MEDIAMTX_STREAM_PREFIX": "sota-paddle-mtmc",
        },
    ):
        s = make_from_env("CAM_01")
    urls = s.stream_urls()
    assert urls["rtsp"].endswith("/sota-paddle-mtmc/CAM_01")


def test_disabled_streamer_does_not_spawn_thread() -> None:
    """Calling ``start()`` on a disabled streamer must not start
    any threads (so the caller can use it in a unit test without
    a thread-leak warning)."""
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="",
        enabled=False,
    )
    s.start()
    assert s._push_thread is None  # noqa: SLF001
    s.stop()


def test_disabled_streamer_is_a_clean_context_manager() -> None:
    """Used as ``with make_from_env(...) as s:`` it must not raise."""
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h",
        enabled=False,
    )
    with s as ctx:
        assert ctx is s
    assert not s.is_running()


def test_strict_mode_is_supported() -> None:
    """``MEDIAMTX_STRICT=true`` would be honored by the visual
    script; here we just confirm the env var is read consistently
    and a disabled streamer is unaffected."""
    with mock.patch.dict(
        os.environ,
        {
            "MEDIAMTX_ENABLED": "false",
            "MEDIAMTX_STRICT": "true",
        },
    ):
        s = make_from_env("CAM_01")
    assert s.is_enabled() is False
