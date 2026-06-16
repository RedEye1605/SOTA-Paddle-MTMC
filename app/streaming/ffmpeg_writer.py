"""Pure-function FFmpeg command + URL builders (Phase 6).

Adapted from the upstream ``Service/offline-people-counting`` pattern
(:mod:`app.io.streamer_command`). The builder is intentionally
dependency-free so it can be unit-tested without spawning a
subprocess.

The publish URL is built from a small template that supports a few
substitutions:

* ``{host}``    — the MediaMTX host
* ``{port}``    — the RTSP port
* ``{prefix}``  — the operator's stream prefix (e.g. ``sota-paddle-mtmc``)
* ``{camera_id}`` — the camera identifier (``CAM_01`` / ``CAM_02``)

The default template produces::

    rtsp://{host}:{port}/{prefix}/{camera_id}

which matches the per-camera namespacing the user task spec asked
for.

The FFmpeg command mirrors Service's argv exactly: raw BGR24 on
stdin → libx264 zerolatency → RTSP/TCP push.
"""

from __future__ import annotations

from typing import Optional

FFMPEG_LOGLEVEL = "warning"

DEFAULT_PUBLISH_URL_TEMPLATE = "rtsp://{host}:{port}/{prefix}/{camera_id}"


def build_publish_url(
    *,
    template: Optional[str],
    host: str,
    rtsp_port: int,
    prefix: str,
    camera_id: str,
) -> str:
    """Build the publish URL from a small template.

    If ``template`` is empty, the default
    ``rtsp://{host}:{port}/{prefix}/{camera_id}`` form is used.

    Unknown ``{placeholders}`` are left in place so the operator can
    spot template typos at the first publish attempt.
    """
    template = (template or "").strip() or DEFAULT_PUBLISH_URL_TEMPLATE
    return template.format(host=host, port=rtsp_port, prefix=prefix, camera_id=camera_id)


def build_hls_url(
    *,
    host: str,
    hls_port: int,
    prefix: str,
    camera_id: str,
) -> str:
    return f"http://{host}:{hls_port}/{prefix}/{camera_id}/index.m3u8"


def build_webrtc_url(
    *,
    host: str,
    webrtc_port: int,
    prefix: str,
    camera_id: str,
) -> str:
    return f"http://{host}:{webrtc_port}/{prefix}/{camera_id}"


def build_ffmpeg_command(
    *,
    ffmpeg_bin: str,
    width: int,
    height: int,
    fps: int,
    bitrate_kbps: int,
    output_url: str,
    rtsp_transport: str = "tcp",
) -> list[str]:
    """Build the ffmpeg argv list for a raw BGR24 pipe → RTSP push.

    Mirrors Service's argv.  Notes:

    * ``-f rtsp -rtsp_transport tcp`` matches Service (TCP is more
      reliable than UDP for a single-host LAN).
    * The libx264 encoder is configured for zero-latency, ultrafast
      preset — quality is sacrificed for stream responsiveness.
    * GOP = ``fps * 2`` so the stream is seekable every 2 s.
    * ``-sc_threshold 0`` disables scene-cut detection (keeps the
      GOP length consistent).
    """
    return [
        ffmpeg_bin,
        "-loglevel",
        FFMPEG_LOGLEVEL,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-b:v",
        f"{bitrate_kbps}k",
        "-maxrate",
        f"{bitrate_kbps}k",
        "-bufsize",
        f"{bitrate_kbps * 2}k",
        "-g",
        str(fps * 2),
        "-sc_threshold",
        "0",
        "-f",
        "rtsp",
        "-rtsp_transport",
        rtsp_transport,
        output_url,
    ]
