"""ThingsBoard `{ts, values}` payload format."""

from __future__ import annotations


from app.telemetry.thingsboard_payload import (
    build_dwell_payload,
    build_global_count_payload,
    build_zone_event_payload,
)


def test_global_count_payload_shape() -> None:
    p = build_global_count_payload(
        global_id="GID-1", camera_id="CAM_01", site_id="showroom", ts_ms=1000
    )
    assert set(p.keys()) == {"ts", "values"}
    assert p["ts"] == 1000
    assert p["values"]["global_id"] == "GID-1"
    assert p["values"]["site_id"] == "showroom"
    assert p["values"]["camera_id"] == "CAM_01"
    assert p["values"]["global_id_active"] == 1


def test_zone_event_payload_shape() -> None:
    p = build_zone_event_payload(
        zone_id="Z1", camera_id="CAM_01", event_type="enter", global_id="GID-1", ts_ms=2000
    )
    assert p["ts"] == 2000
    assert p["values"]["zone_event"] == "enter"
    assert p["values"]["zone_id"] == "Z1"


def test_dwell_payload_shape() -> None:
    p = build_dwell_payload(
        global_id="GID-1", zone_id="Z1", camera_id="CAM_01", duration_seconds=120, ts_ms=3000
    )
    assert p["ts"] == 3000
    assert p["values"]["dwell_duration_seconds"] == 120


def test_payload_ts_is_ms() -> None:
    # Ensure the default ts is in milliseconds (not seconds).
    p = build_global_count_payload(global_id="GID-1", camera_id=None, site_id="x")
    # Heuristic: ts should be > 10^12 for any reasonable date after 2001
    assert p["ts"] > 1_000_000_000_000
