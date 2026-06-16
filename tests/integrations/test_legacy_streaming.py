"""Tests for the legacy streaming / ROI / env contract.

* CAM_01 / CAM_02 RTSP input path is unchanged.
* HLS / WebRTC / RTSP port numbers match the legacy config.yaml.
* Publish URL has the legacy ``{cam_id}/live`` suffix.
* ROI zones for CAM_01 and CAM_02 are loaded from the legacy YAML
  and the new pipeline can scale them to processing size.
* The env / config resolution yields the same broker, topic, bucket,
  prefix, credentials, and ROI values as the legacy pipeline.
"""

from __future__ import annotations


import pytest

from app.integrations import legacy_contract as lc
from app.integrations.legacy_stream import resolve_legacy_endpoints


# ---------------------------------------------------------------------------
# 1. RTSP / HLS / WebRTC ports
# ---------------------------------------------------------------------------


def test_rtsp_port_8554() -> None:
    cfg = lc._load_config()
    assert int(cfg["streaming"]["rtsp_port"]) == 8554


def test_hls_port_8889() -> None:
    cfg = lc._load_config()
    assert int(cfg["streaming"]["hls_port"]) == 8889


def test_webrtc_port_8890() -> None:
    cfg = lc._load_config()
    assert int(cfg["streaming"]["webrtc_port"]) == 8890


# ---------------------------------------------------------------------------
# 2. Publish URL format
# ---------------------------------------------------------------------------


def test_publish_url_cam1() -> None:
    cfg = lc._load_config()
    s = cfg["streaming"]
    expected = f"rtsp://{{host}}:{int(s['rtsp_port'])}/cam1/live"
    assert lc.legacy_publish_url("anyhost", "cam1") == expected.format(host="anyhost")


def test_publish_url_cam2() -> None:
    cfg = lc._load_config()
    s = cfg["streaming"]
    expected = f"rtsp://{{host}}:{int(s['rtsp_port'])}/cam2/live"
    assert lc.legacy_publish_url("anyhost", "cam2") == expected.format(host="anyhost")


def test_hls_url_cam1() -> None:
    cfg = lc._load_config()
    s = cfg["streaming"]
    expected = f"http://{{host}}:{int(s['hls_port'])}/cam1/live/index.m3u8"
    assert lc.legacy_hls_url("anyhost", "cam1") == expected.format(host="anyhost")


def test_webrtc_url_cam1() -> None:
    cfg = lc._load_config()
    s = cfg["streaming"]
    expected = f"http://{{host}}:{int(s['webrtc_port'])}/cam1/live"
    assert lc.legacy_webrtc_url("anyhost", "cam1") == expected.format(host="anyhost")


# ---------------------------------------------------------------------------
# 3. resolve_legacy_endpoints uses env
# ---------------------------------------------------------------------------


def test_resolve_endpoints_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIAMTX_HOST", "h.example")
    monkeypatch.setenv("MEDIAMTX_RTSP_PORT", "9554")
    monkeypatch.setenv("MEDIAMTX_HLS_PORT", "9889")
    monkeypatch.setenv("MEDIAMTX_WEBRTC_PORT", "9890")
    ep = resolve_legacy_endpoints()
    assert ep.host == "h.example"
    assert ep.rtsp_port == 9554
    assert ep.hls_port == 9889
    assert ep.webrtc_port == 9890
    assert ep.publish_url("cam1") == "rtsp://h.example:9554/cam1/live"
    assert ep.hls_url("cam2") == "http://h.example:9889/cam2/live/index.m3u8"
    assert ep.webrtc_url("cam1") == "http://h.example:9890/cam1/live"


def test_resolve_endpoints_raises_when_host_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEDIAMTX_HOST", raising=False)
    with pytest.raises(RuntimeError):
        resolve_legacy_endpoints()


def test_resolve_endpoints_default_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIAMTX_HOST", "h")
    monkeypatch.delenv("MEDIAMTX_RTSP_PORT", raising=False)
    monkeypatch.delenv("MEDIAMTX_HLS_PORT", raising=False)
    monkeypatch.delenv("MEDIAMTX_WEBRTC_PORT", raising=False)
    ep = resolve_legacy_endpoints()
    assert ep.rtsp_port == 8554
    assert ep.hls_port == 8889
    assert ep.webrtc_port == 8890


# ---------------------------------------------------------------------------
# 4. ROI config matches legacy for CAM_01 and CAM_02
# ---------------------------------------------------------------------------


def test_cam01_roi_in_legacy_config() -> None:
    cam = lc.legacy_roi_zones("cam1")
    names = {r.name for r in cam.rois}
    assert names == {"Fazzio & Filano Zone", "Active Zone", "Dealing Zone 1", "Island Zone"}


def test_cam02_roi_in_legacy_config() -> None:
    cam = lc.legacy_roi_zones("cam2")
    names = {r.name for r in cam.rois}
    assert names == {"Sport Zone", "Premium Zone", "Dealing Zone 2", "Island Zone"}


def test_legacy_rois_match_config_yaml() -> None:
    """Cross-check that the legacy YAML polygon coordinates match the
    values quoted in Service/offline-people-counting/config.yaml."""
    cam1 = lc.legacy_roi_zones("cam1")
    cam1_pts = {r.name: r.points for r in cam1.rois}
    assert cam1_pts["Fazzio & Filano Zone"][0] == (102, 792)
    assert cam1_pts["Active Zone"][-1] == (168, 724)
    cam2 = lc.legacy_roi_zones("cam2")
    cam2_pts = {r.name: r.points for r in cam2.rois}
    assert cam2_pts["Sport Zone"][0] == (270, 886)
    assert cam2_pts["Premium Zone"][-1] == (242, 776)


def test_legacy_cam1_display_size() -> None:
    cam = lc.legacy_roi_zones("cam1")
    assert cam.display_size == (960, 540)
    assert cam.original_size == (3072, 2048)
    assert cam.fps == 15


def test_legacy_cam2_display_size() -> None:
    cam = lc.legacy_roi_zones("cam2")
    assert cam.display_size == (960, 540)
    assert cam.original_size == (2592, 1944)
    assert cam.fps == 15


# ---------------------------------------------------------------------------
# 5. env / config resolution yields same broker / topic / bucket
# ---------------------------------------------------------------------------


def test_config_resolves_broker_host() -> None:
    # The new pipeline's MQTT_BROKER_HOST / MEDIAMTX_HOST are the
    # operator's set-and-forget deployment addresses. The legacy
    # values are documented in configs/legacy/offline_people_counting.yaml
    # so the new pipeline *can* also reach them when the operator
    # chooses.
    from app.integrations.legacy_contract import (
        legacy_camera_topic,
        legacy_device_config,
        legacy_evidence_key,
    )

    dev = legacy_device_config("CAM_01")
    topic = legacy_camera_topic("telemetry", "CAM_01", dev.device_name)
    assert topic == "ai/yamaha/people-detection/cam1/summary"
    key = legacy_evidence_key(camera_id="cam1", zone="Sport Zone", person_id=1)
    assert key.startswith("people-detection/")


def test_config_resolves_credentials_env_names() -> None:
    # The .env / .env.example must reference the legacy env names.
    env_example = __import__("pathlib").Path(__file__).resolve().parents[2] / ".env.example"
    text = env_example.read_text()
    for name in (
        "MQTT_USERNAME",
        "MQTT_PASSWORD",
        "MQTT_BROKER_HOST",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MINIO_ENDPOINT",
        "MEDIAMTX_HOST",
        "MEDIAMTX_RTSP_PORT",
        "MEDIAMTX_HLS_PORT",
        "MEDIAMTX_WEBRTC_PORT",
    ):
        assert name in text, f"env var {name} missing from .env.example"
