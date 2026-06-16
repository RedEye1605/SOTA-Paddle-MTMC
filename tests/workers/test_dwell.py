"""Dwell session bookkeeping."""

from __future__ import annotations


from app.zones.dwell import DwellBookkeeper


def test_open_close_dwell() -> None:
    d = DwellBookkeeper()
    e = d.on_event(global_id="G1", zone_id="Z1", camera_id="CAM_01", event_type="enter", ts=100.0)
    assert e["kind"] == "open"
    e2 = d.on_event(global_id="G1", zone_id="Z1", camera_id="CAM_01", event_type="exit", ts=160.0)
    assert e2["kind"] == "close"
    assert e2["duration_seconds"] == 60


def test_duplicate_enter_ignored() -> None:
    d = DwellBookkeeper()
    d.on_event(global_id="G1", zone_id="Z1", camera_id="CAM_01", event_type="enter", ts=100.0)
    e2 = d.on_event(global_id="G1", zone_id="Z1", camera_id="CAM_01", event_type="enter", ts=110.0)
    assert e2 is None  # already open


def test_exit_without_enter_ignored() -> None:
    d = DwellBookkeeper()
    e = d.on_event(global_id="G1", zone_id="Z1", camera_id="CAM_01", event_type="exit", ts=100.0)
    assert e is None


def test_force_close_stale() -> None:
    d = DwellBookkeeper(max_open_seconds=10)
    d.on_event(global_id="G1", zone_id="Z1", camera_id="CAM_01", event_type="enter", ts=0.0)
    out = d.force_close_stale(now=20.0)
    assert len(out) == 1
    assert out[0]["duration_seconds"] == 20
