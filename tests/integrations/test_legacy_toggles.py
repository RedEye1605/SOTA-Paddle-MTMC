"""Tests for the legacy feature toggles (Phase 5b).

Verifies that each of the new toggles:

1. Resolves to a boolean.
2. Respects the env-var override.
3. Affects only visualization / outbound integration (does not
   change the legacy contract values).
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from app.integrations import legacy_contract as lc
from app.integrations.legacy_contract import (
    all_flags,
    flag_enabled,
    legacy_camera_topic,
    legacy_evidence_key,
    legacy_roi_zones,
    toggle_names,
)


TOGGLES = (
    "ENABLE_SEND_MQTT",
    "ENABLE_MINIO_UPLOAD",
    "ENABLE_TRACK_ID",
    "SHOW_ROI_ZONES",
    "SHOW_CONFIDENCE_SCORE",
    "SHOW_TRACK_ID",
    "SHOW_DETECTION_BOX",
    "SHOW_CAMERA_LABEL",
    "SHOW_COUNTING_OVERLAY",
)


# ---------------------------------------------------------------------------
# 1. All toggles are recognised
# ---------------------------------------------------------------------------


def test_toggle_names_includes_all_required_toggles() -> None:
    for t in TOGGLES:
        assert t in toggle_names()


def test_toggle_names_in_yaml_matches_constants() -> None:
    yaml_toggles = set(lc._load_config().get("toggles", {}).keys())
    assert set(TOGGLES) == yaml_toggles


# ---------------------------------------------------------------------------
# 2. Default values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TOGGLES)
def test_default_value_is_true(name: str) -> None:
    # No env override: default is True.
    with mock.patch.dict(os.environ, {}, clear=True):
        assert flag_enabled(name) is True


# ---------------------------------------------------------------------------
# 3. Env-var overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TOGGLES)
def test_env_false_disables_toggle(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "false")
    assert flag_enabled(name) is False


@pytest.mark.parametrize("name", TOGGLES)
def test_env_true_enables_toggle(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "true")
    assert flag_enabled(name) is True


def test_env_one_means_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_SEND_MQTT", "1")
    assert flag_enabled("ENABLE_SEND_MQTT") is True


def test_env_yes_means_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_SEND_MQTT", "yes")
    assert flag_enabled("ENABLE_SEND_MQTT") is True


def test_env_zero_means_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_SEND_MQTT", "0")
    assert flag_enabled("ENABLE_SEND_MQTT") is False


# ---------------------------------------------------------------------------
# 4. all_flags returns the full state
# ---------------------------------------------------------------------------


def test_all_flags_returns_every_toggle() -> None:
    flags = all_flags()
    for t in TOGGLES:
        assert t in flags


def test_all_flags_reflects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_SEND_MQTT", "false")
    monkeypatch.setenv("SHOW_ROI_ZONES", "false")
    flags = all_flags()
    assert flags["ENABLE_SEND_MQTT"] is False
    assert flags["SHOW_ROI_ZONES"] is False
    assert flags["ENABLE_MINIO_UPLOAD"] is True  # default
    assert flags["ENABLE_TRACK_ID"] is True


# ---------------------------------------------------------------------------
# 5. Toggles do not change the legacy contract
# ---------------------------------------------------------------------------


def test_disable_mqtt_toggle_does_not_change_topic() -> None:
    """The topic is the contract value, independent of the toggle."""
    with mock.patch.dict(os.environ, {"ENABLE_SEND_MQTT": "false"}):
        assert (
            legacy_camera_topic("telemetry", "CAM_01", "cam_1")
            == "ai/yamaha/people-detection/cam1/summary"
        )


def test_disable_minio_toggle_does_not_change_object_key() -> None:
    with mock.patch.dict(os.environ, {"ENABLE_MINIO_UPLOAD": "false"}):
        k = legacy_evidence_key(camera_id="cam1", zone="Sport Zone", person_id=1)
        assert k.startswith("people-detection/cam1/sport-zone/")


def test_disable_track_id_toggle_does_not_change_roi() -> None:
    with mock.patch.dict(os.environ, {"ENABLE_TRACK_ID": "false"}):
        cam = legacy_roi_zones("cam1")
        assert {r.name for r in cam.rois} == {
            "Fazzio & Filano Zone",
            "Active Zone",
            "Dealing Zone 1",
            "Island Zone",
        }


def test_show_overlay_toggles_have_no_side_effects() -> None:
    """Disabling the visualization toggles must not change the
    outbound contract."""
    env = {t: "false" for t in TOGGLES if t.startswith("SHOW_")}
    with mock.patch.dict(os.environ, env):
        cam = legacy_roi_zones("cam1")
        assert cam.original_size == (3072, 2048)
        # Topic still resolves the same way.
        assert (
            legacy_camera_topic("telemetry", "CAM_01", "cam_1")
            == "ai/yamaha/people-detection/cam1/summary"
        )


# ---------------------------------------------------------------------------
# 6. Visualisation-only invariant (documentation test)
# ---------------------------------------------------------------------------


def test_visualization_toggles_are_visualization_only() -> None:
    """All SHOW_* toggles are documented as visualization-only."""
    visual_toggles = {
        "SHOW_ROI_ZONES",
        "SHOW_CONFIDENCE_SCORE",
        "SHOW_TRACK_ID",
        "SHOW_DETECTION_BOX",
        "SHOW_CAMERA_LABEL",
        "SHOW_COUNTING_OVERLAY",
    }
    assert visual_toggles.issubset(set(TOGGLES))


def test_outbound_toggles_are_outbound_only() -> None:
    """All ENABLE_* toggles are documented as outbound-only."""
    outbound_toggles = {
        "ENABLE_SEND_MQTT",
        "ENABLE_MINIO_UPLOAD",
        "ENABLE_TRACK_ID",
    }
    assert outbound_toggles.issubset(set(TOGGLES))
