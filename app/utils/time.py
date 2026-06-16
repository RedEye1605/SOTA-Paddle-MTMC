"""Time helpers.

All timestamps in the pipeline are stored in UTC seconds since epoch (float).
PostgreSQL stores them as TIMESTAMPTZ; Qdrant payloads store them as int (Unix
seconds); ThingsBoard telemetry uses milliseconds.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


def now_ts() -> float:
    return time.time()


def now_ms() -> int:
    return int(time.time() * 1000)


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def session_id_for_window(ts: float, window_seconds: int = 86_400) -> str:
    """Compute a 24 h rolling window session id like 'S-2026-06-12'."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    # The session boundary is the UTC midnight
    return f"S-{dt.strftime('%Y-%m-%d')}"


def age_seconds(then_ts: float, now: float | None = None) -> float:
    if now is None:
        now = time.time()
    return max(0.0, now - then_ts)


def is_within_window(ts: float, window_seconds: int, now: float | None = None) -> bool:
    return age_seconds(ts, now) <= window_seconds
