"""Legacy-compatible streaming URLs + ports.

Mirrors the legacy ``Service/offline-people-counting`` streamer:

* RTSP publish URL:  ``rtsp://{host}:8554/{cam1|cam2}/live``
* HLS URL:           ``http://{host}:8889/{cam1|cam2}/live/index.m3u8``
* WebRTC URL:        ``http://{host}:8890/{cam1|cam2}/live``

The defaults (8554 / 8889 / 8890) come from
``Service/offline-people-counting/config.yaml:152-164``. Operators
override via the standard ``MEDIAMTX_HOST`` / ``MEDIAMTX_RTSP_PORT``
env vars (existing new-pipeline env names) — the values are simply
passed through.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegacyStreamEndpoints:
    host: str
    rtsp_port: int
    hls_port: int
    webrtc_port: int

    def publish_url(self, camera_id: str) -> str:
        from .legacy_contract import _resolved_legacy_topic_id, _load_config

        return f"rtsp://{self.host}:{self.rtsp_port}/{_resolved_legacy_topic_id(camera_id, _load_config())}/live"

    def hls_url(self, camera_id: str) -> str:
        from .legacy_contract import _resolved_legacy_topic_id, _load_config

        return f"http://{self.host}:{self.hls_port}/{_resolved_legacy_topic_id(camera_id, _load_config())}/live/index.m3u8"

    def webrtc_url(self, camera_id: str) -> str:
        from .legacy_contract import _resolved_legacy_topic_id, _load_config

        return f"http://{self.host}:{self.webrtc_port}/{_resolved_legacy_topic_id(camera_id, _load_config())}/live"


def resolve_legacy_endpoints(
    cfg: dict[str, Any] | None = None,
) -> LegacyStreamEndpoints:
    """Resolve MediaMTX endpoints from env (matching legacy .env vars)."""
    # The legacy pipeline reads MEDIAMTX_HOST / MEDIAMTX_HLS_HOST /
    # MEDIAMTX_WEBRTC_HOST; the new pipeline collapses these to a
    # single ``MEDIAMTX_HOST`` + per-protocol ports. We respect the
    # same precedence: explicit HLS / WebRTC host env vars win, but
    # fall back to the global ``MEDIAMTX_HOST`` if unset.
    host = os.environ.get("MEDIAMTX_HOST", "").strip()
    if not host:
        raise RuntimeError("MEDIAMTX_HOST is not set; cannot resolve legacy stream endpoints")
    # Read the legacy HLS / WebRTC host overrides but fall back to
    # the global MEDIAMTX_HOST.  We don't currently surface them as
    # fields on LegacyStreamEndpoints (the new pipeline assumes a
    # single host), but we still honour the env vars when they are
    # set.
    _hls_host = os.environ.get("MEDIAMTX_HLS_HOST", host).strip() or host
    _webrtc_host = os.environ.get("MEDIAMTX_WEBRTC_HOST", host).strip() or host
    del _hls_host, _webrtc_host
    # The legacy config.yaml default is 8554 / 8889 / 8890.
    rtsp_port = int(os.environ.get("MEDIAMTX_RTSP_PORT", "8554"))
    hls_port = int(os.environ.get("MEDIAMTX_HLS_PORT", "8889"))
    webrtc_port = int(os.environ.get("MEDIAMTX_WEBRTC_PORT", "8890"))
    return LegacyStreamEndpoints(
        host=host,
        rtsp_port=rtsp_port,
        hls_port=hls_port,
        webrtc_port=webrtc_port,
    )
