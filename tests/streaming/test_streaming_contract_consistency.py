"""Streaming contract consistency test.

Pins down the values that MUST match across the codebase:

  * RTSP port     = 8554
  * HLS port      = 8889
  * WebRTC port   = 8890
  * Stream prefix = sota-paddle-mtmc

This catches silent drift between ``.env.example``, the streamer
defaults, and the helper modules in ``app.integrations``.

Bug history: a prior version of
``app/streaming/mediamtx_streamer.py::make_from_env`` had
``MEDIAMTX_HLS_PORT`` defaulting to ``8888`` and
``MEDIAMTX_WEBRTC_PORT`` defaulting to ``8889`` — both wrong. The
operator's ``.env`` provided the correct values, so the bug was
latent until the day someone rebooted with a fresh ``.env.example``
and got cross-talk between HLS and WebRTC listeners.
"""

from __future__ import annotations

import os
from unittest import mock

from app.streaming.mediamtx_streamer import make_from_env


# Authoritative contract values. Change here AND in .env.example
# AND in legacy_stream.py AND in legacy_contract.py if these ever
# change.
RTSP_PORT = 8554
HLS_PORT = 8889
WEBRTC_PORT = 8890
STREAM_PREFIX = "sota-paddle-mtmc"


def test_make_from_env_default_ports_match_contract() -> None:
    """No MEDIAMTX_*_PORT set → the defaults must equal the contract.

    ``MediaMTXStreamer`` does not store ``_rtsp_port`` as an
    attribute (the RTSP URL is baked into ``_output_url``), so we
    verify the contract via the URLs the streamer actually
    publishes.
    """
    with mock.patch.dict(
        "os.environ",
        {
            "MEDIAMTX_ENABLED": "true",
            "MEDIAMTX_HOST": "h.example.com",
            "MEDIAMTX_STREAM_PREFIX": STREAM_PREFIX,
        },
        clear=False,
    ):
        # Strip any inherited port envs.
        for k in (
            "MEDIAMTX_RTSP_PORT",
            "MEDIAMTX_HLS_PORT",
            "MEDIAMTX_WEBRTC_PORT",
        ):
            os.environ.pop(k, None)
        s = make_from_env("CAM_01")

    urls = s.stream_urls()
    assert urls["rtsp"].startswith(f"rtsp://h.example.com:{RTSP_PORT}/")
    assert urls["hls"].startswith(f"http://h.example.com:{HLS_PORT}/")
    assert urls["webrtc"].startswith(f"http://h.example.com:{WEBRTC_PORT}/")
    assert urls["rtsp"].endswith(f"/{STREAM_PREFIX}/CAM_01")
    assert urls["hls"].endswith(f"/{STREAM_PREFIX}/CAM_01/index.m3u8")
    assert urls["webrtc"].endswith(f"/{STREAM_PREFIX}/CAM_01")


def test_make_from_env_explicit_ports_are_honored() -> None:
    with mock.patch.dict(
        "os.environ",
        {
            "MEDIAMTX_ENABLED": "true",
            "MEDIAMTX_HOST": "h.example.com",
            "MEDIAMTX_RTSP_PORT": "18554",
            "MEDIAMTX_HLS_PORT": "18889",
            "MEDIAMTX_WEBRTC_PORT": "18890",
            "MEDIAMTX_STREAM_PREFIX": STREAM_PREFIX,
        },
    ):
        s = make_from_env("CAM_01")

    urls = s.stream_urls()
    assert urls["rtsp"].startswith("rtsp://h.example.com:18554/")
    assert urls["hls"].startswith("http://h.example.com:18889/")
    assert urls["webrtc"].startswith("http://h.example.com:18890/")


def test_streaming_defaults_match_legacy_stream_module() -> None:
    """The ``app.integrations.legacy_stream`` module's port docs
    must agree with the contract. A drift here means the legacy
    integration and the new pipeline are wired to different
    MediaMTX instances.

    The legacy module's module docstring hard-codes the ports. We
    assert the docstring contains all three port numbers.
    """
    import app.integrations.legacy_stream as legacy_stream

    doc = legacy_stream.__doc__ or ""
    assert str(RTSP_PORT) in doc
    assert str(HLS_PORT) in doc
    assert str(WEBRTC_PORT) in doc


def test_stream_urls_have_distinct_cam_paths() -> None:
    """CAM_01 and CAM_02 must yield distinct URLs across RTSP, HLS,
    WebRTC — same prefix and host, different camera suffix."""
    with mock.patch.dict(
        "os.environ",
        {
            "MEDIAMTX_ENABLED": "true",
            "MEDIAMTX_HOST": "h.example.com",
            "MEDIAMTX_RTSP_PORT": str(RTSP_PORT),
            "MEDIAMTX_HLS_PORT": str(HLS_PORT),
            "MEDIAMTX_WEBRTC_PORT": str(WEBRTC_PORT),
            "MEDIAMTX_STREAM_PREFIX": STREAM_PREFIX,
        },
    ):
        s01 = make_from_env("CAM_01")
        s02 = make_from_env("CAM_02")

    u01 = s01.stream_urls()
    u02 = s02.stream_urls()

    for proto in ("rtsp", "hls", "webrtc"):
        assert u01[proto] != u02[proto], f"{proto} urls collided"
        assert u01[proto].endswith("/CAM_01") or "CAM_01" in u01[proto]
        assert u02[proto].endswith("/CAM_02") or "CAM_02" in u02[proto]


def test_rtsp_publish_url_uses_correct_ports_and_prefix() -> None:
    """The ffmpeg-publish URL must use the RTSP port, not HLS or
    WebRTC. A common drift is to accidentally pass the HLS port to
    the publish URL — that breaks ffmpeg → MediaMTX."""
    with mock.patch.dict(
        "os.environ",
        {
            "MEDIAMTX_ENABLED": "true",
            "MEDIAMTX_HOST": "h.example.com",
            "MEDIAMTX_RTSP_PORT": str(RTSP_PORT),
            "MEDIAMTX_HLS_PORT": str(HLS_PORT),
            "MEDIAMTX_WEBRTC_PORT": str(WEBRTC_PORT),
            "MEDIAMTX_STREAM_PREFIX": STREAM_PREFIX,
        },
    ):
        s = make_from_env("CAM_01")

    u = s.stream_urls()
    assert u["rtsp"].startswith(f"rtsp://h.example.com:{RTSP_PORT}/")
    assert u["hls"].startswith(f"http://h.example.com:{HLS_PORT}/")
    assert u["webrtc"].startswith(f"http://h.example.com:{WEBRTC_PORT}/")
    for url in u.values():
        assert f"/{STREAM_PREFIX}/" in url
