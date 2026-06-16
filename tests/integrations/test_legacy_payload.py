"""Tests for the legacy ThingsBoard ``{ts, values}`` payload.

Every key emitted by ``LegacyPayloadBuilder.build`` is asserted here,
along with the delta-tracking behaviour (ThingsBoard ``SUM``
aggregation requires emitting the *delta* per tick).
"""

from __future__ import annotations

import time

import pytest

from app.integrations.legacy_payload import (
    FakeZoneStats,
    LegacyPayloadBuilder,
)


def _stats(name: str = "Active Zone") -> FakeZoneStats:
    s = FakeZoneStats(name=name)
    s.current_count = 3
    s.total_entered = 10
    s.total_exited = 7
    s.dwell_times.extend([12.5, 30.0, 5.0])
    s.valid_entries.extend(
        [
            {"dwell_time": 12.5, "person_id": 1},
            {"dwell_time": 30.0, "person_id": 2},
            {"dwell_time": 5.0, "person_id": 3},
        ]
    )
    s.total_entered_hourly = 4
    s.total_entered_daily = 10
    s.total_exited_hourly = 3
    s.total_exited_daily = 7
    s.dwell_times_hourly.extend([12.5, 5.0])
    s.dwell_times_daily.extend([12.5, 30.0, 5.0])
    s.valid_entries_hourly.extend(
        [{"dwell_time": 12.5, "person_id": 1}, {"dwell_time": 5.0, "person_id": 3}]
    )
    s.valid_entries_daily.extend(
        [
            {"dwell_time": 12.5, "person_id": 1},
            {"dwell_time": 30.0, "person_id": 2},
            {"dwell_time": 5.0, "person_id": 3},
        ]
    )
    s.pending_entries.append({"dwell_time": 1.0, "person_id": 99})
    return s


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_payload_shape_ts_and_values() -> None:
    b = LegacyPayloadBuilder()
    payload = b.build(
        zone_stats={"Active Zone": _stats()},
        unique_counts=(0, 0, 0),
        frame_count=0,
        camera_dwell_times=[],
        timestamp=1_700_000_000_000,
    )
    assert set(payload.keys()) == {"ts", "values"}
    assert payload["ts"] == 1_700_000_000_000


def test_payload_timestamp_is_unix_ms() -> None:
    b = LegacyPayloadBuilder()
    payload = b.build(
        zone_stats={},
        unique_counts=(0, 0, 0),
        frame_count=0,
        camera_dwell_times=[],
        timestamp=time.time(),
    )
    assert payload["ts"] > 1_700_000_000_000


# ---------------------------------------------------------------------------
# Per-zone gauges
# ---------------------------------------------------------------------------


def test_total_people_is_sum_of_current_counts() -> None:
    b = LegacyPayloadBuilder()
    s1 = _stats("Active Zone")
    s1.current_count = 3
    s2 = _stats("Island Zone")
    s2.current_count = 5
    payload = b.build(
        zone_stats={"Active Zone": s1, "Island Zone": s2},
        unique_counts=(0, 0, 0),
        frame_count=0,
        camera_dwell_times=[],
        timestamp=1,
    )
    assert payload["values"]["TotalPeople"] == 8


def test_per_zone_count_field_uses_zonename_without_spaces() -> None:
    b = LegacyPayloadBuilder()
    payload = b.build(
        zone_stats={"Active Zone": _stats("Active Zone")},
        unique_counts=(0, 0, 0),
        frame_count=0,
        camera_dwell_times=[],
        timestamp=1,
    )
    assert payload["values"]["ActiveZoneCount"] == 3
    # AvgDwell uses lower_underscore form
    assert payload["values"]["active_zoneAvgDwell"] == pytest.approx((12.5 + 30 + 5) / 3, abs=0.01)
    # longestDwell<ZoneKey>
    assert payload["values"]["longestDwellActiveZone"] == 30.0
    # pending<ZoneKey>
    assert payload["values"]["pendingActiveZone"] == 1


# ---------------------------------------------------------------------------
# Windowed gauges
# ---------------------------------------------------------------------------


def test_windowed_dwell_keys_present() -> None:
    b = LegacyPayloadBuilder()
    payload = b.build(
        zone_stats={"Sport Zone": _stats("Sport Zone")},
        unique_counts=(0, 0, 0),
        frame_count=0,
        camera_dwell_times=[],
        timestamp=1,
    )
    v = payload["values"]
    assert "sport_zoneAvgDwellHourly" in v
    assert "sport_zoneAvgDwellDaily" in v
    assert "longestDwellSportZoneHourly" in v
    assert "longestDwellSportZoneDaily" in v


# ---------------------------------------------------------------------------
# Aggregate gauges
# ---------------------------------------------------------------------------


def test_aggregate_gauges_match_legacy() -> None:
    b = LegacyPayloadBuilder()
    s1 = _stats("Active Zone")
    s1.current_count = 3
    s1.total_entered = 10
    s2 = _stats("Island Zone")
    s2.current_count = 1
    s2.total_entered = 4
    payload = b.build(
        zone_stats={"Active Zone": s1, "Island Zone": s2},
        unique_counts=(5, 2, 1),
        frame_count=0,
        camera_dwell_times=[12.5, 30.0],
        timestamp=1,
    )
    v = payload["values"]
    assert v["busiestZone"] == "Active Zone"
    assert v["mostVisitedZone"] == "Active Zone"
    assert v["uniquePeopleToday"] == 5
    assert v["uniquePeopleHourly"] == 2
    assert v["uniquePeopleDaily"] == 1
    assert v["avgCameraDwell"] == pytest.approx(21.25, abs=0.01)
    assert v["longestCameraDwell"] == 30.0
    assert v["totalPendingEntries"] == 2


# ---------------------------------------------------------------------------
# Delta tracking (ThingsBoard SUM)
# ---------------------------------------------------------------------------


def test_first_emit_includes_full_deltas() -> None:
    b = LegacyPayloadBuilder()
    s = _stats("Active Zone")
    s.total_entered = 10
    s.total_exited = 7
    payload = b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=5,
        camera_dwell_times=[],
        timestamp=1,
    )
    v = payload["values"]
    # First tick emits the full value as the delta.
    assert v["countActiveZone"] == 10
    assert v["exitActiveZone"] == 7
    assert v["validEntriesActiveZone"] == 3
    assert v["totalFrames"] == 5
    assert v["totalValidEntries"] == 3


def test_subsequent_emit_increments_delta_only() -> None:
    b = LegacyPayloadBuilder()
    s = _stats("Active Zone")
    s.total_entered = 10
    s.total_exited = 7
    b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=5,
        camera_dwell_times=[],
        timestamp=1,
    )
    # Frame 6: +1 entered, +1 exited, +1 valid entry, +1 frame.
    s.total_entered = 11
    s.total_exited = 8
    s.valid_entries.append({"dwell_time": 2, "person_id": 5})
    payload2 = b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=6,
        camera_dwell_times=[],
        timestamp=2,
    )
    v = payload2["values"]
    assert v["countActiveZone"] == 1
    assert v["exitActiveZone"] == 1
    # initial_valid -> initial_valid+1, so delta is 1
    assert v["validEntriesActiveZone"] == 1
    assert v["totalFrames"] == 1


def test_zero_delta_is_omitted() -> None:
    b = LegacyPayloadBuilder()
    s = _stats("Active Zone")
    b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=5,
        camera_dwell_times=[],
        timestamp=1,
    )
    # Same stats on the second tick.
    payload2 = b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=5,
        camera_dwell_times=[],
        timestamp=2,
    )
    v = payload2["values"]
    assert "countActiveZone" not in v
    assert "exitActiveZone" not in v


def test_windowed_snapshots_always_present() -> None:
    """Unlike cumulative deltas, windowed snapshots are full values."""
    b = LegacyPayloadBuilder()
    s = _stats("Active Zone")
    b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=0,
        camera_dwell_times=[],
        timestamp=1,
    )
    payload2 = b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=0,
        camera_dwell_times=[],
        timestamp=2,
    )
    v = payload2["values"]
    assert v["countActiveZoneHourly"] == s.total_entered_hourly
    assert v["countActiveZoneDaily"] == s.total_entered_daily
    assert v["exitActiveZoneHourly"] == s.total_exited_hourly
    assert v["exitActiveZoneDaily"] == s.total_exited_daily


def test_reset_clears_deltas() -> None:
    b = LegacyPayloadBuilder()
    s = _stats("Active Zone")
    s.total_entered = 10
    b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=5,
        camera_dwell_times=[],
        timestamp=1,
    )
    b.reset()
    payload = b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(0, 0, 0),
        frame_count=5,
        camera_dwell_times=[],
        timestamp=2,
    )
    # After reset, the baseline is 0 again, so the delta is the
    # current value.
    assert payload["values"]["countActiveZone"] == 10
