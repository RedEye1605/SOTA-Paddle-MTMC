"""ThingsBoard payload contract tests (Phase 5).

Covers:
  1. {ts, values} envelope on every helper
  2. ts is in milliseconds
  3. legacy payload shapes are preserved
  4. the user-spec example payload matches exactly
"""

from __future__ import annotations

import time

from app.telemetry.thingsboard_payload import (
    build_dwell_payload,
    build_global_count_payload,
    build_system_health_payload,
    build_zone_event_payload,
    build_zone_summary_payload,
)


def test_envelope_keys_are_exactly_ts_and_values() -> None:
    for payload in [
        build_global_count_payload(global_id="G", camera_id=None, site_id="s"),
        build_zone_event_payload(zone_id="z", camera_id="c", event_type="enter", global_id="g"),
        build_dwell_payload(global_id="g", zone_id="z", camera_id="c", duration_seconds=10),
        build_zone_summary_payload(
            cam_id="c",
            zone_id="z",
            people_count=0,
            entries=0,
            exits=0,
            dwell_avg_seconds=0.0,
            active_global_ids=0,
        ),
        build_system_health_payload(
            site_id="s",
            camera_id="c",
            fps=1.0,
            detector_backend="x",
            reid_backend="y",
        ),
    ]:
        assert set(payload.keys()) == {"ts", "values"}


def test_default_ts_is_unix_milliseconds() -> None:
    """All helpers default to ``int(time.time() * 1000)``."""
    before = int(time.time() * 1000)
    p = build_zone_summary_payload(
        cam_id="c",
        zone_id="z",
        people_count=0,
        entries=0,
        exits=0,
        dwell_avg_seconds=0.0,
        active_global_ids=0,
    )
    after = int(time.time() * 1000)
    assert before <= p["ts"] <= after


def test_user_spec_example_payload_matches_exactly() -> None:
    """The user spec listed an example payload verbatim; verify it."""
    p = build_zone_summary_payload(
        cam_id="CAM_01",
        zone_id="ZONE_A",
        people_count=3,
        entries=1,
        exits=0,
        dwell_avg_seconds=42.5,
        active_global_ids=3,
        ts_ms=1_710_000_000_000,
    )
    assert p == {
        "ts": 1_710_000_000_000,
        "values": {
            "cam_id": "CAM_01",
            "zone_id": "ZONE_A",
            "people_count": 3,
            "entries": 1,
            "exits": 0,
            "dwell_avg_seconds": 42.5,
            "active_global_ids": 3,
        },
    }


def test_extra_values_overlay_into_values() -> None:
    p = build_global_count_payload(
        global_id="G",
        camera_id="C",
        site_id="s",
        extra_values={"foo": 1, "bar": 2},
    )
    assert p["values"]["foo"] == 1
    assert p["values"]["bar"] == 2


def test_global_count_payload_does_not_leak_optional_fields() -> None:
    p = build_global_count_payload(global_id="G", camera_id=None, site_id="s")
    # camera_id is omitted when None (no spurious empty key).
    assert "camera_id" not in p["values"]


def test_system_health_includes_workers_crashed_default_zero() -> None:
    p = build_system_health_payload(
        site_id="s",
        camera_id="C",
        fps=1.0,
        detector_backend="pphuman",
        reid_backend="transreid",
    )
    assert p["values"]["workers_crashed"] == 0
    assert p["values"]["stream_healthy"] is True


def test_dwell_duration_coerced_to_int() -> None:
    p = build_dwell_payload(global_id="g", zone_id="z", camera_id="c", duration_seconds=12.7)
    assert p["values"]["dwell_duration_seconds"] == 12
