"""Dwell session bookkeeping (in-memory, persists to PostgreSQL).

A dwell session is opened on the first `enter` event for a (global_id, zone)
pair, and closed on the next `exit` event. Sessions that don't see an
`exit` within `max_open_seconds` are force-closed by the cleanup job.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _DwellState:
    entered_at: float
    zone_id: str
    camera_id: str


class DwellBookkeeper:
    def __init__(self, max_open_seconds: float = 24 * 3600) -> None:
        self._open: dict[tuple[str, str], _DwellState] = {}
        self.max_open_seconds = max_open_seconds

    def on_event(
        self,
        *,
        global_id: str,
        zone_id: str,
        camera_id: str,
        event_type: str,  # "enter" | "exit"
        ts: float,
    ) -> Optional[dict]:
        """Process a zone event. Returns a dict describing the transition
        (used by the caller to write to PostgreSQL), or None.
        """
        key = (global_id, zone_id)
        if event_type == "enter":
            if key in self._open:
                return None
            self._open[key] = _DwellState(entered_at=ts, zone_id=zone_id, camera_id=camera_id)
            return {
                "kind": "open",
                "global_id": global_id,
                "zone_id": zone_id,
                "camera_id": camera_id,
                "entered_at": ts,
            }
        if event_type == "exit":
            st = self._open.pop(key, None)
            if st is None:
                return None
            duration = max(0, int(ts - st.entered_at))
            return {
                "kind": "close",
                "global_id": global_id,
                "zone_id": zone_id,
                "camera_id": camera_id,
                "entered_at": st.entered_at,
                "exited_at": ts,
                "duration_seconds": duration,
            }
        return None

    def force_close_stale(self, now: Optional[float] = None) -> list[dict]:
        """Close any dwell sessions older than `max_open_seconds`."""
        if now is None:
            now = time.time()
        out: list[dict] = []
        stale = [k for k, v in self._open.items() if (now - v.entered_at) > self.max_open_seconds]
        for k in stale:
            st = self._open.pop(k)
            duration = max(0, int(now - st.entered_at))
            out.append(
                {
                    "kind": "close",
                    "global_id": k[0],
                    "zone_id": k[1],
                    "camera_id": st.camera_id,
                    "entered_at": st.entered_at,
                    "exited_at": now,
                    "duration_seconds": duration,
                }
            )
        return out
