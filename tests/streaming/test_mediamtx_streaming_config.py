"""MediaMTX streaming config tests (Phase 6).

Covers:
  1. stream URL is built from env/config
  2. CAM_01 and CAM_02 get different stream paths
  3. ffmpeg command does not include secrets
  4. streaming disabled is safe
  5. one failed stream does not kill all cameras (per-camera isolation)
"""

from __future__ import annotations

import logging
from unittest import mock

from app.streaming.ffmpeg_writer import (
    DEFAULT_PUBLISH_URL_TEMPLATE,
    build_ffmpeg_command,
    build_hls_url,
    build_publish_url,
    build_webrtc_url,
)
from app.streaming.mediamtx_streamer import (
    MediaMTXStreamer,
    make_from_env,
)


# ---------------------------------------------------------------------------
# 1. URL builder
# ---------------------------------------------------------------------------


def test_default_publish_url_uses_prefix_and_camera_id() -> None:
    url = build_publish_url(
        template=None,
        host="mediamtx.example.com",
        rtsp_port=8554,
        prefix="sota-paddle-mtmc",
        camera_id="CAM_01",
    )
    assert url == "rtsp://mediamtx.example.com:8554/sota-paddle-mtmc/CAM_01"


def test_custom_publish_url_template_substitutes() -> None:
    url = build_publish_url(
        template="rtsp://{host}:{port}/custom/{camera_id}",
        host="h",
        rtsp_port=8554,
        prefix="ignored",
        camera_id="CAM_02",
    )
    assert url == "rtsp://h:8554/custom/CAM_02"


def test_hls_url() -> None:
    url = build_hls_url(host="h", hls_port=8888, prefix="p", camera_id="CAM_01")
    assert url == "http://h:8888/p/CAM_01/index.m3u8"


def test_webrtc_url() -> None:
    url = build_webrtc_url(host="h", webrtc_port=8889, prefix="p", camera_id="CAM_01")
    assert url == "http://h:8889/p/CAM_01"


def test_default_template_constant() -> None:
    assert DEFAULT_PUBLISH_URL_TEMPLATE == "rtsp://{host}:{port}/{prefix}/{camera_id}"


# ---------------------------------------------------------------------------
# 2. CAM_01 and CAM_02 get different stream paths
# ---------------------------------------------------------------------------


def test_cam_01_and_cam_02_have_distinct_paths() -> None:
    url_01 = build_publish_url(
        template=None,
        host="h",
        rtsp_port=8554,
        prefix="sota-paddle-mtmc",
        camera_id="CAM_01",
    )
    url_02 = build_publish_url(
        template=None,
        host="h",
        rtsp_port=8554,
        prefix="sota-paddle-mtmc",
        camera_id="CAM_02",
    )
    assert url_01 != url_02
    assert "CAM_01" in url_01
    assert "CAM_02" in url_02
    # They share the prefix and host, so we test for the suffix being different.
    assert url_01.endswith("/sota-paddle-mtmc/CAM_01")
    assert url_02.endswith("/sota-paddle-mtmc/CAM_02")


def test_streamer_caches_distinct_urls_per_camera() -> None:
    s1 = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h",
        stream_prefix="p",
        enabled=False,
    )
    s2 = MediaMTXStreamer(
        camera_id="CAM_02",
        width=64,
        height=64,
        host="h",
        stream_prefix="p",
        enabled=False,
    )
    assert s1._output_url != s2._output_url  # noqa: SLF001


# ---------------------------------------------------------------------------
# 3. ffmpeg command does not include secrets
# ---------------------------------------------------------------------------


def test_ffmpeg_command_contains_no_secrets() -> None:
    """The argv list must never include passwords / tokens."""
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=960,
        height=540,
        fps=10,
        bitrate_kbps=1800,
        output_url="rtsp://h:8554/p/CAM_01",
    )
    flat = " ".join(cmd)
    assert "password" not in flat.lower()
    assert "token" not in flat.lower()
    assert "secret" not in flat.lower()


def test_ffmpeg_command_uses_rtsp_tcp() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=960,
        height=540,
        fps=10,
        bitrate_kbps=1800,
        output_url="rtsp://h:8554/p/CAM_01",
    )
    # The arg `-rtsp_transport tcp` must be present.
    assert "tcp" in cmd
    idx = cmd.index("-rtsp_transport")
    assert idx >= 0
    assert cmd[idx + 1] == "tcp"


def test_ffmpeg_command_zerolatency_preset() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=960,
        height=540,
        fps=10,
        bitrate_kbps=1800,
        output_url="rtsp://h:8554/p/CAM_01",
    )
    assert "ultrafast" in cmd
    assert "zerolatency" in cmd
    # GOP = fps * 2 = 20
    gop_idx = cmd.index("-g")
    assert cmd[gop_idx + 1] == "20"


def test_ffmpeg_command_uses_bgr24_pipe() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=960,
        height=540,
        fps=10,
        bitrate_kbps=1800,
        output_url="rtsp://h:8554/p/CAM_01",
    )
    assert "rawvideo" in cmd
    assert "bgr24" in cmd
    assert "pipe:0" in cmd
    assert "960x540" in cmd


def test_ffmpeg_command_output_url_at_end() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=960,
        height=540,
        fps=10,
        bitrate_kbps=1800,
        output_url="rtsp://h:8554/p/CAM_01",
    )
    # Last arg is the output URL.
    assert cmd[-1] == "rtsp://h:8554/p/CAM_01"


# ---------------------------------------------------------------------------
# 4. streaming disabled is safe
# ---------------------------------------------------------------------------


def test_streamer_disabled_does_not_spawn_ffmpeg() -> None:
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h",
        enabled=False,
    )
    s.start()
    assert not s.is_running()
    s.stop()
    assert s.stop_reason() is None


def test_streamer_disabled_when_host_empty() -> None:
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="",
        enabled=True,  # host empty disables
    )
    s.start()
    assert not s.is_running()
    assert s.stop_reason() == "host_unset"


def test_streamer_disabled_logs_no_secrets(caplog) -> None:
    """The disable / no-host path must not log any sensitive value."""
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h.example.com",
        enabled=False,
    )
    with caplog.at_level(logging.INFO):
        s.start()
    flat = caplog.text.lower()
    assert "password" not in flat
    assert "token" not in flat


def test_make_from_env_returns_disabled_when_env_off() -> None:
    with mock.patch.dict(
        "os.environ",
        {
            "MEDIAMTX_ENABLED": "false",
            "MEDIAMTX_HOST": "h.example.com",
        },
        clear=False,
    ):
        s = make_from_env("CAM_01")
    assert s.is_enabled() is False


def test_make_from_env_returns_enabled_when_env_on() -> None:
    with mock.patch.dict(
        "os.environ",
        {
            "MEDIAMTX_ENABLED": "true",
            "MEDIAMTX_HOST": "h.example.com",
            "MEDIAMTX_RTSP_PORT": "8554",
            "MEDIAMTX_STREAM_PREFIX": "sota-paddle-mtmc",
        },
    ):
        s = make_from_env("CAM_01")
    assert s.is_enabled() is True
    assert s._host == "h.example.com"  # noqa: SLF001
    assert "CAM_01" in s._output_url  # noqa: SLF001


# ---------------------------------------------------------------------------
# 5. one failed stream does not kill all cameras
# ---------------------------------------------------------------------------


def test_streamer_isolation() -> None:
    """Two streamers with the same args are independent objects."""
    s1 = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h",
        enabled=False,
    )
    s2 = MediaMTXStreamer(
        camera_id="CAM_02",
        width=64,
        height=64,
        host="h",
        enabled=False,
    )
    s1.start()
    s2.start()
    # Stopping one does not affect the other.
    s1.stop()
    assert not s1.is_running()
    assert not s2.is_running()  # was never started
    s2.stop()


def test_streamer_stop_idempotent() -> None:
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h",
        enabled=False,
    )
    s.start()
    s.stop()
    # Second stop is a no-op.
    s.stop()
    assert not s.is_running()


def test_push_frame_after_stop_is_noop() -> None:
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h",
        enabled=False,
    )
    s.start()
    s.stop()
    # push_frame is a no-op when not running
    import numpy as np

    s.push_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    # No exception, no state change.
    assert not s.is_running()


def test_stream_urls_returns_three_formats() -> None:
    s = MediaMTXStreamer(
        camera_id="CAM_01",
        width=64,
        height=64,
        host="h.example.com",
        rtsp_port=8554,
        hls_port=8888,
        webrtc_port=8889,
        stream_prefix="p",
        enabled=False,
    )
    urls = s.stream_urls()
    assert urls["rtsp"] == "rtsp://h.example.com:8554/p/CAM_01"
    assert urls["hls"] == "http://h.example.com:8888/p/CAM_01/index.m3u8"
    assert urls["webrtc"] == "http://h.example.com:8889/p/CAM_01"
