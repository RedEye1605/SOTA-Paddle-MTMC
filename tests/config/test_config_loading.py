"""Config loading — YAML schema validation."""

from __future__ import annotations

from pathlib import Path


from app.cli.config import load_all_configs

ROOT = Path(__file__).resolve().parents[2]


def test_load_all_configs() -> None:
    configs = load_all_configs(
        ROOT / "configs/app.yaml",
        ROOT / "configs/cameras.yaml",
        ROOT / "configs/zones.yaml",
        ROOT / "configs/camera_links.yaml",
    )
    assert "app" in configs
    assert "cameras" in configs
    assert "zones" in configs
    assert "links" in configs

    # app.yaml required sections
    app = configs["app"]
    assert "runtime" in app
    assert "detection_tracking" in app
    assert "reid" in app
    assert "identity" in app
    assert "telemetry" in app
    assert "storage" in app

    # 5 active cameras
    assert len([c for c in configs["cameras"]["cameras"] if c.get("is_active")]) >= 2

    # zones + camera_links
    assert len(configs["zones"]["zones"]) >= 2
    assert len(configs["links"]["camera_links"]) >= 1


def test_app_yaml_initial_thresholds() -> None:
    configs = load_all_configs(
        ROOT / "configs/app.yaml",
        ROOT / "configs/cameras.yaml",
        ROOT / "configs/zones.yaml",
        ROOT / "configs/camera_links.yaml",
    )
    identity = configs["app"]["identity"]
    assert identity["auto_match_threshold"] == 0.82
    assert identity["candidate_threshold"] == 0.72
    assert identity["ambiguous_margin"] == 0.04
    assert identity["prefer_new_id_when_ambiguous"] is True
    assert identity["use_camera_topology"] is True
    assert identity["use_zone_transitions"] is True
