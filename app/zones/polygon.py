"""Polygon zone geometry + ray-casting point-in-polygon test."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class Zone:
    zone_id: str
    camera_id: str
    name: str
    polygon: list[tuple[float, float]]
    zone_type: str
    is_entry_zone: bool
    is_exit_zone: bool
    enabled: bool = True

    def to_pixel_polygon(
        self,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int]]:
        """Convert normalized [0..1] polygon to absolute pixels."""
        return [(int(round(x * frame_w)), int(round(y * frame_h))) for (x, y) in self.polygon]


def parse_zones(rows: Iterable[dict]) -> dict[str, list[Zone]]:
    """Returns {camera_id: [Zone, ...]}."""
    by_cam: dict[str, list[Zone]] = {}
    for r in rows:
        try:
            polygon = json.loads(r["polygon_json"])
            polygon = [(float(x), float(y)) for (x, y) in polygon]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"Invalid polygon_json for zone {r.get('zone_id')}: {e}") from e
        zone = Zone(
            zone_id=r["zone_id"],
            camera_id=r["camera_id"],
            name=r["name"],
            polygon=polygon,
            zone_type=r.get("zone_type", "generic"),
            is_entry_zone=bool(r.get("is_entry_zone", False)),
            is_exit_zone=bool(r.get("is_exit_zone", False)),
            enabled=bool(r.get("enabled", True)),
        )
        by_cam.setdefault(zone.camera_id, []).append(zone)
    return by_cam


def point_in_polygon(
    px: float,
    py: float,
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Ray-casting algorithm. Polygon is closed implicitly (last edge connects
    last vertex to first)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def bbox_centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
