"""Tests for the legacy Service/offline-people-counting contract.

Every value asserted here is mirrored exactly from
``Service/offline-people-counting`` — MQTT topic, MQTT topic base,
MinIO bucket / object_prefix / object-key format, streaming ports
and URL templates, ROI polygons for CAM_01 and CAM_02, thresholds.

If a test fails, the legacy pipeline has changed and the new
pipeline must be updated to match.
"""

from __future__ import annotations

import re

import pytest

from app.integrations import legacy_contract as lc
from app.integrations.legacy_contract import (
    LegacyCameraRois,
    LegacyDeviceConfig,
    all_flags,
    all_legacy_cameras,
    all_legacy_devices,
    flag_enabled,
    legacy_camera_topic,
    legacy_client_id,
    legacy_device_config,
    legacy_evidence_key,
    legacy_hls_url,
    legacy_publish_url,
    legacy_roi_zones,
    legacy_thresholds,
    legacy_webrtc_url,
    normalize_legacy_camera_id,
    toggle_names,
)


# ---------------------------------------------------------------------------
# 1. camera id normalization
# ---------------------------------------------------------------------------


def test_normalize_legacy_camera_id_cam1() -> None:
    assert normalize_legacy_camera_id("CAM_01", "cam_1") == "cam1"


def test_normalize_legacy_camera_id_cam2() -> None:
    assert normalize_legacy_camera_id("CAM_02", "cam_2") == "cam2"


def test_normalize_legacy_camera_id_already_legacy() -> None:
    assert normalize_legacy_camera_id("cam1", "cam1") == "cam1"
    assert normalize_legacy_camera_id("cam2", "cam2") == "cam2"


# ---------------------------------------------------------------------------
# 2. MQTT topic for CAM_01 / CAM_02
# ---------------------------------------------------------------------------


def test_cam01_mqtt_topic_matches_legacy() -> None:
    assert (
        legacy_camera_topic("telemetry", "CAM_01", "cam_1")
        == "ai/yamaha/people-detection/cam1/summary"
    )


def test_cam02_mqtt_topic_matches_legacy() -> None:
    assert (
        legacy_camera_topic("telemetry", "CAM_02", "cam_2")
        == "ai/yamaha/people-detection/cam2/summary"
    )


def test_mqtt_topic_base_matches_legacy() -> None:
    assert lc._load_config()["mqtt"]["topic_base"] == "ai/yamaha/people-detection"


def test_mqtt_other_channels() -> None:
    assert (
        legacy_camera_topic("attributes", "CAM_01", "cam_1")
        == "ai/yamaha/people-detection/cam1/attributes"
    )
    assert (
        legacy_camera_topic("events", "CAM_02", "cam_2") == "ai/yamaha/people-detection/cam2/event"
    )
    assert (
        legacy_camera_topic("status", "CAM_01", "cam_1") == "ai/yamaha/people-detection/cam1/status"
    )
    assert (
        legacy_camera_topic("command", "CAM_01", "cam_1") == "ai/yamaha/people-detection/+/command"
    )


def test_mqtt_qos_and_retain_match_legacy() -> None:
    mqtt = lc._load_config()["mqtt"]
    assert mqtt["qos"] == 1
    assert mqtt["retain"] is False
    assert mqtt["keepalive_seconds"] == 60
    assert mqtt["publish_interval_seconds"] == 3


# ---------------------------------------------------------------------------
# 3. Client id format
# ---------------------------------------------------------------------------


def test_legacy_client_id_format() -> None:
    cid = legacy_client_id("cam_1", now_epoch=1_700_000_000.0, rand=1234)
    assert cid == "people_counter_cam_1_1700000000_1234"


def test_legacy_client_id_random_suffix_in_range() -> None:
    cid = legacy_client_id("cam_1", now_epoch=1_700_000_000.0)
    # Pattern: people_counter_cam_1_<int>_<int 1000-9999>
    m = re.fullmatch(r"people_counter_cam_1_\d{10}_(\d{4})", cid)
    assert m is not None
    suffix = int(m.group(1))
    assert 1000 <= suffix <= 9999


# ---------------------------------------------------------------------------
# 4. Device config (CAM_01 / CAM_02)
# ---------------------------------------------------------------------------


def test_cam01_device_config() -> None:
    dev = legacy_device_config("CAM_01")
    assert dev == LegacyDeviceConfig(
        camera_id="CAM_01",
        device_name="cam_1",
        device_type="People Counting",
        location="Main Entrance - Cam 1",
        category="ymh",
        integration="yamaha",
        subsystem="demo",
        site_id="site_001",
    )


def test_cam02_device_config() -> None:
    dev = legacy_device_config("CAM_02")
    assert dev.device_name == "cam_2"
    assert dev.site_id == "site_001"
    assert dev.integration == "yamaha"


def test_all_legacy_devices_has_cam01_cam02() -> None:
    devices = all_legacy_devices()
    ids = {d.camera_id for d in devices}
    assert "CAM_01" in ids
    assert "CAM_02" in ids


# ---------------------------------------------------------------------------
# 5. MinIO contract
# ---------------------------------------------------------------------------


def test_minio_bucket_and_prefix() -> None:
    cfg = lc._load_config()
    assert cfg["minio"]["bucket"] == "yamaha-poc"
    assert cfg["minio"]["object_prefix"] == "people-detection"


def test_minio_object_key_format() -> None:
    key = legacy_evidence_key(
        camera_id="cam1",
        zone="Sport Zone",
        person_id=42,
        timestamp_epoch=1_700_000_000.0,
    )
    # prefix/cam1/sport-zone/2023-11-14/1700000000000_42.jpg
    assert key.startswith("people-detection/cam1/sport-zone/")
    assert key.endswith("_42.jpg")
    assert "2023-11-14" in key
    assert "/1700000000000_42.jpg" in key


def test_minio_object_key_uses_location_title_when_zone_missing() -> None:
    key = legacy_evidence_key(
        camera_id="cam1",
        zone=None,
        person_id=1,
        timestamp_epoch=1_700_000_000.0,
    )
    # The legacy pipeline uses location_title ("main_hallway") as the
    # fallback zone folder when no zone is provided.
    assert "main-hallway" in key


def test_minio_object_key_ampersand_slug() -> None:
    key = legacy_evidence_key(
        camera_id="cam1",
        zone="Fazzio & Filano Zone",
        person_id=1,
        timestamp_epoch=1_700_000_000.0,
    )
    # Ampersand → " and ", then non-alphanum → "-"
    assert "/fazzio-and-filano-zone/" in key


# ---------------------------------------------------------------------------
# 6. Streaming
# ---------------------------------------------------------------------------


def test_legacy_publish_url_cam1() -> None:
    assert legacy_publish_url("198.51.100.10", "cam1") == "rtsp://198.51.100.10:8554/cam1/live"


def test_legacy_publish_url_cam2() -> None:
    assert legacy_publish_url("198.51.100.10", "cam2") == "rtsp://198.51.100.10:8554/cam2/live"


def test_legacy_publish_url_new_id_cam01() -> None:
    # Even when the caller passes the new id (CAM_01), the URL must
    # use the legacy normalized id.
    assert legacy_publish_url("h", "CAM_01", lc._load_config()) == "rtsp://h:8554/cam1/live"


def test_legacy_hls_url() -> None:
    assert (
        legacy_hls_url("hls.example.invalid", "cam1")
        == "http://hls.example.invalid:8889/cam1/live/index.m3u8"
    )


def test_legacy_webrtc_url() -> None:
    assert legacy_webrtc_url("rtc.example.invalid", "cam1") == "http://rtc.example.invalid:8890/cam1/live"


def test_legacy_streaming_ports() -> None:
    s = lc._load_config()["streaming"]
    assert int(s["rtsp_port"]) == 8554
    assert int(s["hls_port"]) == 8889
    assert int(s["webrtc_port"]) == 8890


# ---------------------------------------------------------------------------
# 7. ROI / zone config
# ---------------------------------------------------------------------------


def test_legacy_cam1_rois_exact_polygons() -> None:
    cam = legacy_roi_zones("cam1")
    assert isinstance(cam, LegacyCameraRois)
    assert cam.original_size == (3072, 2048)
    assert cam.display_size == (960, 540)
    name_to_points = {r.name: r.points for r in cam.rois}
    assert name_to_points["Fazzio & Filano Zone"] == [
        (102, 792),
        (742, 608),
        (2968, 2048),
        (534, 2048),
    ]
    assert name_to_points["Active Zone"] == [
        (112, 512),
        (1090, 300),
        (1806, 702),
        (1236, 856),
        (782, 556),
        (168, 724),
    ]
    assert name_to_points["Dealing Zone 1"] == [
        (1372, 882),
        (2016, 688),
        (2644, 1032),
        (2016, 1288),
    ]
    assert name_to_points["Island Zone"] == [(1902, 470), (2436, 230), (3040, 582), (2510, 802)]


def test_legacy_cam2_rois_exact_polygons() -> None:
    cam = legacy_roi_zones("cam2")
    assert cam.original_size == (2592, 1944)
    name_to_points = {r.name: r.points for r in cam.rois}
    assert name_to_points["Sport Zone"] == [
        (270, 886),
        (1390, 588),
        (2592, 1358),
        (2592, 1944),
        (786, 1944),
    ]
    assert name_to_points["Premium Zone"] == [(108, 260), (710, 84), (988, 260), (242, 776)]
    assert name_to_points["Dealing Zone 2"] == [(912, 380), (1246, 304), (1312, 520), (968, 604)]
    assert name_to_points["Island Zone"] == [(1572, 516), (1912, 288), (2480, 596), (2174, 896)]


def test_legacy_cam1_resolution_via_cam01_alias() -> None:
    cam = legacy_roi_zones("CAM_01")
    name_to_points = {r.name: r.points for r in cam.rois}
    assert name_to_points["Fazzio & Filano Zone"] == [
        (102, 792),
        (742, 608),
        (2968, 2048),
        (534, 2048),
    ]


def test_legacy_zone_colors_cam1() -> None:
    cam = legacy_roi_zones("cam1")
    assert cam.zone_colors["Fazzio & Filano Zone"] == (0, 242, 255)
    assert cam.zone_colors["Active Zone"] == (255, 0, 255)
    assert cam.zone_colors["Dealing Zone 1"] == (0, 0, 255)
    assert cam.zone_colors["Island Zone"] == (255, 255, 255)


def test_legacy_zone_colors_cam2() -> None:
    cam = legacy_roi_zones("cam2")
    assert cam.zone_colors["Sport Zone"] == (0, 255, 51)
    assert cam.zone_colors["Premium Zone"] == (221, 0, 0)
    assert cam.zone_colors["Dealing Zone 2"] == (0, 0, 255)
    assert cam.zone_colors["Island Zone"] == (255, 255, 255)


def test_legacy_roi_scaling_to_processing_size() -> None:
    cam = legacy_roi_zones("cam1")
    out = cam.to_processing_polygons((960, 540))
    # The 3072x2048 -> 960x540 scale is 0.3125 / 0.26367...
    name_to_points = dict(out)
    fz = name_to_points["Fazzio & Filano Zone"]
    # (102, 792) -> (int(102*0.3125), int(792*0.263671875)) = (31, 208)
    assert fz[0] == (int(round(102 * 960 / 3072)), int(round(792 * 540 / 2048)))


def test_legacy_unknown_camera_raises() -> None:
    with pytest.raises(KeyError):
        legacy_roi_zones("cam999")


def test_all_legacy_cameras_returns_cam1_cam2() -> None:
    cams = all_legacy_cameras()
    keys = {c.camera_id for c in cams}
    assert "cam1" in keys
    assert "cam2" in keys


# ---------------------------------------------------------------------------
# 8. Thresholds
# ---------------------------------------------------------------------------


def test_legacy_thresholds_match_config_yaml() -> None:
    t = legacy_thresholds()
    assert float(t["conf_threshold"]) == 0.5
    assert float(t["gallery_match_threshold"]) == 0.70
    assert float(t["gallery_same_camera_threshold"]) == 0.80
    assert float(t["gallery_cross_camera_threshold"]) == 0.55
    assert float(t["gallery_prototype_min_cosine"]) == 0.80
    assert float(t["dwell_min_seconds"]) == 120.0
    assert int(t["tracklet_ttl_frames"]) == 90
    assert int(t["tracklet_min_length"]) == 10


# ---------------------------------------------------------------------------
# 9. Feature toggles
# ---------------------------------------------------------------------------


def test_default_toggle_values() -> None:
    # All toggles default to True.
    for name in toggle_names():
        assert flag_enabled(name) is True, f"{name} should default to True"


def test_env_overrides_default_toggle_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_SEND_MQTT", "false")
    assert flag_enabled("ENABLE_SEND_MQTT") is False


def test_env_overrides_default_toggle_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOW_ROI_ZONES", "true")
    assert flag_enabled("SHOW_ROI_ZONES") is True


def test_all_flags_returns_all_toggles() -> None:
    flags = all_flags()
    expected = {
        "ENABLE_SEND_MQTT",
        "ENABLE_MINIO_UPLOAD",
        "ENABLE_TRACK_ID",
        "SHOW_ROI_ZONES",
        "SHOW_CONFIDENCE_SCORE",
        "SHOW_TRACK_ID",
        "SHOW_DETECTION_BOX",
        "SHOW_CAMERA_LABEL",
        "SHOW_COUNTING_OVERLAY",
    }
    assert set(flags.keys()) == expected


def test_unknown_toggle_returns_true() -> None:
    # "Unknown == on" — the new pipeline only adds toggles, never
    # removes a default-true behaviour.
    assert flag_enabled("DOES_NOT_EXIST") is True


# ---------------------------------------------------------------------------
# 10. Toggle visual-affects-only invariant (documentation test)
# ---------------------------------------------------------------------------


def test_toggles_documented_names_match_yaml() -> None:
    cfg = lc._load_config()
    yaml_toggles = set(cfg.get("toggles", {}).keys())
    assert set(toggle_names()) == yaml_toggles
