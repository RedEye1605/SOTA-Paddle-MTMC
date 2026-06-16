"""ThingsBoard ``{ts, values}`` payload builder for the legacy contract.

Port of ``Service/offline-people-counting/app/counting/payload.py`` to
the new pipeline.  The schema is preserved exactly so the same
ThingsBoard dashboard works against the new pipeline without
modification.

Field reference (every key the legacy payload emits, with the same
naming convention)::

    TotalPeople
    <ZoneKey>Count
    <zone>_avgDwell
    longestDwell<ZoneKey>
    pending<ZoneKey>
    <zone>_avgDwellHourly
    <zone>_avgDwellDaily
    longestDwell<ZoneKey>Hourly
    longestDwell<ZoneKey>Daily
    avgDwellTime
    longestDwellTime
    busiestZone
    mostVisitedZone
    longestVisitedZone
    mostVisitedZoneHourly / Daily
    longestVisitedZoneHourly / Daily
    avgCameraDwell
    longestCameraDwell
    uniquePeopleToday / Hourly / Daily
    totalPendingEntries
    count<ZoneKey>            (delta, ThingsBoard SUM)
    exit<ZoneKey>             (delta, ThingsBoard SUM)
    validEntries<ZoneKey>     (delta, ThingsBoard SUM)
    totalFrames               (delta)
    totalValidEntries         (delta)
    count<ZoneKey>Hourly / Daily  (snapshot)
    exit<ZoneKey>Hourly / Daily
    validEntries<ZoneKey>Hourly / Daily

The builder is stateful only for the delta baselines. One instance
per camera; share across threads under an external lock.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Iterable

from ..telemetry.mqtt_client import MqttPublisher  # noqa: F401  (re-export)


def _avg(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _max0(values: Iterable[float]) -> float:
    values = list(values)
    return float(max(values)) if values else 0.0


def _zone_key(zone_name: str) -> str:
    """``"Fazzio & Filano Zone"`` -> ``"Fazzio&FilanoZone"``."""
    return zone_name.replace(" ", "")


def _zone_lower_us(zone_name: str) -> str:
    """``"Sport Zone"`` -> ``"sport_zone"``."""
    return zone_name.lower().replace(" ", "_")


def _argmax(zone_stats: dict[str, Any], key_fn, default: str) -> str:
    if not zone_stats:
        return default
    return max(zone_stats.items(), key=lambda kv: key_fn(kv[1]))[0]


def _to_unix_ms(timestamp) -> int:
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return int(timestamp.timestamp() * 1000)
    try:
        v = float(timestamp)
    except (TypeError, ValueError):
        return int(time.time() * 1000)
    return int(v) if v > 1_000_000_000_000 else int(v * 1000)


class LegacyPayloadBuilder:
    """Builds the legacy ThingsBoard ``{ts, values}`` payload."""

    def __init__(self) -> None:
        self._last_published: dict[str, float] = {}
        self._last_ts_debug: float = 0.0
        self._debug_log_interval: float = 5.0

    def reset(self) -> None:
        """Forget all delta baselines. Used by ``reset_statistics``."""
        self._last_published.clear()

    # ---- public ----
    def build(
        self,
        *,
        zone_stats: dict[str, Any],
        unique_counts: tuple[int, int, int],
        frame_count: int,
        camera_dwell_times: Iterable[float],
        timestamp: Any,
    ) -> dict[str, Any]:
        ts_ms = _to_unix_ms(timestamp)
        values: dict[str, Any] = {}
        self._emit_gauges(values, zone_stats)
        self._emit_windowed_gauges(values, zone_stats)
        self._emit_aggregate_gauges(values, zone_stats, camera_dwell_times, unique_counts)
        self._emit_cumulative_deltas(values, zone_stats, frame_count)
        self._emit_windowed_snapshots(values, zone_stats)
        return {"ts": ts_ms, "values": values}

    # ---- internals ----
    def _emit_gauges(self, values: dict, zone_stats: dict) -> None:
        values["TotalPeople"] = sum(
            int(getattr(s, "current_count", 0)) for s in zone_stats.values()
        )
        for name, stats in zone_stats.items():
            key = _zone_key(name)
            dwell = [e.get("dwell_time", 0.0) for e in getattr(stats, "valid_entries", [])]
            values[f"{key}Count"] = int(getattr(stats, "current_count", 0))
            values[f"{_zone_lower_us(name)}AvgDwell"] = round(_avg(dwell), 2)
            values[f"longestDwell{key}"] = round(_max0(dwell), 2)
            values[f"pending{key}"] = len(getattr(stats, "pending_entries", []))

    def _emit_windowed_gauges(self, values: dict, zone_stats: dict) -> None:
        for name, stats in zone_stats.items():
            lower = _zone_lower_us(name)
            hourly = list(getattr(stats, "dwell_times_hourly", []))
            daily = list(getattr(stats, "dwell_times_daily", []))
            values[f"{lower}AvgDwellHourly"] = round(_avg(hourly), 2)
            values[f"{lower}AvgDwellDaily"] = round(_avg(daily), 2)
            values[f"longestDwell{_zone_key(name)}Hourly"] = round(_max0(hourly), 2)
            values[f"longestDwell{_zone_key(name)}Daily"] = round(_max0(daily), 2)

    def _emit_aggregate_gauges(
        self,
        values: dict,
        zone_stats: dict,
        camera_dwell_times: Iterable[float],
        unique_counts: tuple[int, int, int],
    ) -> None:
        seen, hourly, daily = unique_counts
        valid_all = [e for s in zone_stats.values() for e in getattr(s, "valid_entries", [])]
        dwell_all = [e.get("dwell_time", 0.0) for e in valid_all]
        avg_cam = _avg(camera_dwell_times)
        max_cam = _max0(camera_dwell_times)
        values.update(
            {
                "avgDwellTime": round(_avg(dwell_all), 2),
                "longestDwellTime": round(_max0(dwell_all), 2),
                "busiestZone": _argmax(zone_stats, lambda s: s.current_count, default="N/A"),
                "mostVisitedZone": _argmax(zone_stats, lambda s: s.total_entered, default="N/A"),
                "longestVisitedZone": _argmax(
                    zone_stats, lambda s: _avg(getattr(s, "dwell_times", [])), default="N/A"
                ),
                "mostVisitedZoneHourly": _argmax(
                    zone_stats, lambda s: s.total_entered_hourly, default="N/A"
                ),
                "longestVisitedZoneHourly": _argmax(
                    zone_stats, lambda s: _avg(getattr(s, "dwell_times_hourly", [])), default="N/A"
                ),
                "mostVisitedZoneDaily": _argmax(
                    zone_stats, lambda s: s.total_entered_daily, default="N/A"
                ),
                "longestVisitedZoneDaily": _argmax(
                    zone_stats, lambda s: _avg(getattr(s, "dwell_times_daily", [])), default="N/A"
                ),
                "avgCameraDwell": round(avg_cam, 2),
                "longestCameraDwell": round(max_cam, 2),
                "uniquePeopleToday": int(seen),
                "uniquePeopleHourly": int(hourly),
                "uniquePeopleDaily": int(daily),
                "totalPendingEntries": sum(
                    len(getattr(s, "pending_entries", [])) for s in zone_stats.values()
                ),
            }
        )

    def _emit_cumulative_deltas(self, values: dict, zone_stats: dict, frame_count: int) -> None:
        for name, stats in zone_stats.items():
            key = _zone_key(name)
            self._emit_delta(values, f"count{key}", int(getattr(stats, "total_entered", 0)))
            self._emit_delta(values, f"exit{key}", int(getattr(stats, "total_exited", 0)))
            self._emit_delta(values, f"validEntries{key}", len(getattr(stats, "valid_entries", [])))
        self._emit_delta(values, "totalFrames", int(frame_count))
        valid_all = [e for s in zone_stats.values() for e in getattr(s, "valid_entries", [])]
        self._emit_delta(values, "totalValidEntries", len(valid_all))

    def _emit_windowed_snapshots(self, values: dict, zone_stats: dict) -> None:
        for name, stats in zone_stats.items():
            key = _zone_key(name)
            values[f"count{key}Hourly"] = int(getattr(stats, "total_entered_hourly", 0))
            values[f"count{key}Daily"] = int(getattr(stats, "total_entered_daily", 0))
            values[f"exit{key}Hourly"] = int(getattr(stats, "total_exited_hourly", 0))
            values[f"exit{key}Daily"] = int(getattr(stats, "total_exited_daily", 0))
            values[f"validEntries{key}Hourly"] = len(getattr(stats, "valid_entries_hourly", []))
            values[f"validEntries{key}Daily"] = len(getattr(stats, "valid_entries_daily", []))

    def _emit_delta(self, values: dict, key: str, current_value: int | float) -> None:
        last = self._last_published.get(key, 0)
        delta = current_value - last
        self._last_published[key] = current_value
        if delta:
            values[key] = delta


# ---------------------------------------------------------------------------
# Convenience: a tiny zone-stat fake used by tests + the visual
# validation script. Mirrors the attributes of
# Service/offline-people-counting/app/counting/aggregator.py::_ZoneStats.
# ---------------------------------------------------------------------------


class FakeZoneStats:
    """Mutable zone stats with all the fields LegacyPayloadBuilder expects.

    Used by tests and by the visual validation script. Production code
    does not use this — it receives real :class:`_ZoneStats` objects
    from the per-camera aggregator.
    """

    def __init__(self, *, name: str = "fake") -> None:
        self.name = name
        self.current_count: int = 0
        self.total_entered: int = 0
        self.total_exited: int = 0
        self.dwell_times: deque = deque(maxlen=1000)
        self.valid_entries: deque = deque(maxlen=1000)
        self.pending_entries: deque = deque(maxlen=1000)
        self.total_entered_hourly: int = 0
        self.total_entered_daily: int = 0
        self.total_exited_hourly: int = 0
        self.total_exited_daily: int = 0
        self.dwell_times_hourly: deque = deque(maxlen=1000)
        self.dwell_times_daily: deque = deque(maxlen=1000)
        self.valid_entries_hourly: deque = deque(maxlen=1000)
        self.valid_entries_daily: deque = deque(maxlen=1000)
