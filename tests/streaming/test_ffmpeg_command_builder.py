"""FFmpeg command builder tests (Phase 6).

Covers:
  1. argv shape matches Service's reference
  2. resolution / fps / bitrate propagate
  3. RTSP transport is TCP
  4. the output URL is always the last argument
"""

from __future__ import annotations

from app.streaming.ffmpeg_writer import build_ffmpeg_command


def test_argv_first_is_ffmpeg() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=1000,
        output_url="rtsp://h:1/c",
    )
    assert cmd[0] == "ffmpeg"


def test_argv_loglevel_warning() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=1000,
        output_url="rtsp://h:1/c",
    )
    assert cmd[1:3] == ["-loglevel", "warning"]


def test_argv_resolution_and_fps() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=1280,
        height=720,
        fps=15,
        bitrate_kbps=4000,
        output_url="rtsp://h:1/c",
    )
    assert cmd[cmd.index("-s") + 1] == "1280x720"
    assert cmd[cmd.index("-r") + 1] == "15"


def test_argv_bitrate() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=2500,
        output_url="rtsp://h:1/c",
    )
    # -b:v 2500k -maxrate 2500k -bufsize 5000k
    assert cmd[cmd.index("-b:v") + 1] == "2500k"
    assert cmd[cmd.index("-maxrate") + 1] == "2500k"
    assert cmd[cmd.index("-bufsize") + 1] == "5000k"


def test_argv_gop_is_twice_fps() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=20,
        bitrate_kbps=1000,
        output_url="rtsp://h:1/c",
    )
    assert cmd[cmd.index("-g") + 1] == "40"


def test_argv_no_scene_cut() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=1000,
        output_url="rtsp://h:1/c",
    )
    sc_idx = cmd.index("-sc_threshold")
    assert cmd[sc_idx + 1] == "0"


def test_argv_rtsp_format_and_tcp_transport() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=1000,
        output_url="rtsp://h:1/c",
    )
    # `-f rtsp` immediately before `-rtsp_transport tcp` (the
    # second occurrence of `-f` — the first is the rawvideo input
    # format earlier in the argv).
    f_indices = [i for i, arg in enumerate(cmd) if arg == "-f"]
    assert len(f_indices) >= 2, cmd
    assert cmd[f_indices[1] + 1] == "rtsp"
    rt_idx = cmd.index("-rtsp_transport")
    assert cmd[rt_idx + 1] == "tcp"


def test_argv_output_url_is_last() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=1000,
        output_url="rtsp://h:8554/sota-paddle-mtmc/CAM_01",
    )
    assert cmd[-1] == "rtsp://h:8554/sota-paddle-mtmc/CAM_01"


def test_argv_custom_ffmpeg_bin() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="/usr/local/bin/ffmpeg-static",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=1000,
        output_url="rtsp://h:1/c",
    )
    assert cmd[0] == "/usr/local/bin/ffmpeg-static"


def test_argv_contains_no_password_or_token() -> None:
    cmd = build_ffmpeg_command(
        ffmpeg_bin="ffmpeg",
        width=100,
        height=100,
        fps=10,
        bitrate_kbps=1000,
        output_url="rtsp://host.example.com:8554/path",
    )
    flat = " ".join(cmd)
    # The caller may pass a credentialed URL; the test asserts
    # that this builder does not inject extra credential params
    # of its own (no ``-password``, ``-token`` flags, etc.).
    assert "password=" not in flat
    assert "token=" not in flat
