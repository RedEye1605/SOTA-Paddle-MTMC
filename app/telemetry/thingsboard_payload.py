"""ThingsBoard `{ts, values}` payload builders.

The ThingsBoard telemetry RPC expects every payload to be
``{ts: <unix_ms>, values: {...}}``. The helpers in this module build
that shape for the four payload types the SOTA pipeline emits:

* ``build_zone_summary_payload`` — per-camera, per-zone analytics
  (people_count, entries, exits, dwell_avg_seconds, …).
* ``build_global_count_payload`` — fires when a new global_id is
  minted.
* ``build_zone_event_payload`` — zone enter/exit.
* ``build_dwell_payload`` — dwell-time event.
* ``build_system_health_payload`` — engine health summary.

Nothing in this module ever logs credentials. The MQTT client owns
auth and never echoes passwords; the payloads themselves contain
only domain values.
"""

from __future__ import annotations

import time
from typing import Any, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_global_count_payload(
    *,
    global_id: str,
    camera_id: Optional[str],
    site_id: str,
    ts_ms: Optional[int] = None,
    extra_values: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a ThingsBoard telemetry payload: `{ts, values}`."""
    if ts_ms is None:
        ts_ms = _now_ms()
    values: dict[str, Any] = {
        "global_id_active": 1,
        "global_id": global_id,
        "site_id": site_id,
    }
    if camera_id:
        values["camera_id"] = camera_id
    if extra_values:
        values.update(extra_values)
    return {"ts": ts_ms, "values": values}


def build_zone_event_payload(
    *,
    zone_id: str,
    camera_id: str,
    event_type: str,
    global_id: str,
    ts_ms: Optional[int] = None,
) -> dict[str, Any]:
    if ts_ms is None:
        ts_ms = _now_ms()
    return {
        "ts": ts_ms,
        "values": {
            "zone_event": event_type,
            "zone_id": zone_id,
            "camera_id": camera_id,
            "global_id": global_id,
        },
    }


def build_dwell_payload(
    *,
    global_id: str,
    zone_id: str,
    camera_id: str,
    duration_seconds: int,
    ts_ms: Optional[int] = None,
) -> dict[str, Any]:
    if ts_ms is None:
        ts_ms = _now_ms()
    return {
        "ts": ts_ms,
        "values": {
            "dwell_duration_seconds": int(duration_seconds),
            "zone_id": zone_id,
            "camera_id": camera_id,
            "global_id": global_id,
        },
    }


def build_zone_summary_payload(
    *,
    cam_id: str,
    zone_id: str,
    people_count: int,
    entries: int,
    exits: int,
    dwell_avg_seconds: float,
    active_global_ids: int,
    ts_ms: Optional[int] = None,
    extra_values: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Per-camera, per-zone summary payload.

    The values block matches the user-spec example::

        {
          "cam_id": "CAM_01",
          "zone_id": "ZONE_A",
          "people_count": 3,
          "entries": 1,
          "exits": 0,
          "dwell_avg_seconds": 42.5,
          "active_global_ids": 3
        }
    """
    if ts_ms is None:
        ts_ms = _now_ms()
    values: dict[str, Any] = {
        "cam_id": cam_id,
        "zone_id": zone_id,
        "people_count": int(people_count),
        "entries": int(entries),
        "exits": int(exits),
        "dwell_avg_seconds": float(dwell_avg_seconds),
        "active_global_ids": int(active_global_ids),
    }
    if extra_values:
        values.update(extra_values)
    return {"ts": ts_ms, "values": values}


def build_system_health_payload(
    *,
    site_id: str,
    camera_id: str,
    fps: float,
    detector_backend: str,
    reid_backend: str,
    workers_crashed: int = 0,
    stream_healthy: bool = True,
    ts_ms: Optional[int] = None,
) -> dict[str, Any]:
    """System-health summary for ThingsBoard dashboards."""
    if ts_ms is None:
        ts_ms = _now_ms()
    return {
        "ts": ts_ms,
        "values": {
            "site_id": site_id,
            "cam_id": camera_id,
            "fps": float(fps),
            "detector_backend": detector_backend,
            "reid_backend": reid_backend,
            "workers_crashed": int(workers_crashed),
            "stream_healthy": bool(stream_healthy),
        },
    }
