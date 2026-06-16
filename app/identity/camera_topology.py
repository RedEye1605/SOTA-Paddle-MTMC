"""Camera topology — read `camera_links` from Postgres and answer
`is_transition_valid(from_camera, to_camera)`.

If the pair is not in the table, the answer is `None` (unknown) — the
resolver treats unknown transitions as 0.5 score, not auto-rejected.
If the row exists but `enabled = false`, the answer is `False` and the
resolver MUST NOT auto-match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CameraLink:
    from_camera_id: str
    to_camera_id: str
    min_travel_seconds: int
    max_travel_seconds: int
    transition_probability: float
    enabled: bool
    notes: str = ""

    def is_within_travel_window(self, travel_seconds: float) -> bool:
        return self.min_travel_seconds <= travel_seconds <= self.max_travel_seconds


class CameraTopology:
    """Loaded from Postgres on startup. Read-only in-process cache."""

    def __init__(self) -> None:
        self._links: dict[tuple[str, str], CameraLink] = {}

    def load_from_rows(self, rows: list[dict]) -> None:
        self._links.clear()
        for r in rows:
            link = CameraLink(
                from_camera_id=r["from_camera_id"],
                to_camera_id=r["to_camera_id"],
                min_travel_seconds=int(r["min_travel_seconds"]),
                max_travel_seconds=int(r["max_travel_seconds"]),
                transition_probability=float(r["transition_probability"]),
                enabled=bool(r["enabled"]),
                notes=str(r.get("notes") or ""),
            )
            self._links[(link.from_camera_id, link.to_camera_id)] = link
        logger.info("Loaded %d camera links", len(self._links))

    def is_known_link(self, from_camera_id: str, to_camera_id: str) -> Optional[bool]:
        link = self._links.get((from_camera_id, to_camera_id))
        if link is None:
            return None
        return link.enabled

    def is_within_travel_window(
        self,
        from_camera_id: str,
        to_camera_id: str,
        travel_seconds: float,
    ) -> bool:
        link = self._links.get((from_camera_id, to_camera_id))
        if link is None or not link.enabled:
            return False
        return link.is_within_travel_window(travel_seconds)

    def candidate_cameras_for(self, from_camera_id: str) -> list[str]:
        """All `to_camera_id`s that have an enabled link from `from_camera_id`."""
        out: list[str] = []
        for (frm, to), link in self._links.items():
            if frm == from_camera_id and link.enabled:
                out.append(to)
        return out
