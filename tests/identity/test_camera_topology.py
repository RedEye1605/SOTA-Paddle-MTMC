"""Camera topology — link enforcement."""

from __future__ import annotations


from app.identity.camera_topology import CameraTopology


def test_unknown_link_returns_none() -> None:
    t = CameraTopology()
    assert t.is_known_link("CAM_A", "CAM_B") is None


def test_enabled_link_returns_true() -> None:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_A",
                "to_camera_id": "CAM_B",
                "min_travel_seconds": 10,
                "max_travel_seconds": 60,
                "transition_probability": 0.7,
                "enabled": True,
                "notes": "",
            }
        ]
    )
    assert t.is_known_link("CAM_A", "CAM_B") is True


def test_disabled_link_returns_false() -> None:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_A",
                "to_camera_id": "CAM_C",
                "min_travel_seconds": 0,
                "max_travel_seconds": 0,
                "transition_probability": 0.0,
                "enabled": False,
                "notes": "impossible",
            }
        ]
    )
    assert t.is_known_link("CAM_A", "CAM_C") is False


def test_travel_window_enforced() -> None:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_A",
                "to_camera_id": "CAM_B",
                "min_travel_seconds": 10,
                "max_travel_seconds": 60,
                "transition_probability": 0.7,
                "enabled": True,
                "notes": "",
            }
        ]
    )
    assert t.is_within_travel_window("CAM_A", "CAM_B", 30) is True
    assert t.is_within_travel_window("CAM_A", "CAM_B", 5) is False
    assert t.is_within_travel_window("CAM_A", "CAM_B", 90) is False
    # Disabled link -> always False
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_A",
                "to_camera_id": "CAM_X",
                "min_travel_seconds": 0,
                "max_travel_seconds": 0,
                "transition_probability": 0.0,
                "enabled": False,
                "notes": "",
            }
        ]
    )
    assert t.is_within_travel_window("CAM_A", "CAM_X", 30) is False


def test_candidate_cameras_for() -> None:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_01",
                "to_camera_id": "CAM_02",
                "min_travel_seconds": 1,
                "max_travel_seconds": 60,
                "transition_probability": 0.7,
                "enabled": True,
                "notes": "",
            },
            {
                "from_camera_id": "CAM_01",
                "to_camera_id": "CAM_03",
                "min_travel_seconds": 1,
                "max_travel_seconds": 60,
                "transition_probability": 0.5,
                "enabled": True,
                "notes": "",
            },
            {
                "from_camera_id": "CAM_01",
                "to_camera_id": "CAM_04",
                "min_travel_seconds": 0,
                "max_travel_seconds": 0,
                "transition_probability": 0.0,
                "enabled": False,
                "notes": "",
            },
        ]
    )
    out = t.candidate_cameras_for("CAM_01")
    assert "CAM_02" in out
    assert "CAM_03" in out
    assert "CAM_04" not in out  # disabled -> not a candidate
