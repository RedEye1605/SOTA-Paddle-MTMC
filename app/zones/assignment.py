"""Tracklet → zone assignment + entry/exit event emission.

This module is intentionally side-effect-free (no I/O). The caller (the
tracklet collector worker) is responsible for persisting events.
"""

from __future__ import annotations

import logging
from typing import Optional

from .polygon import Zone, bbox_centroid, point_in_polygon

logger = logging.getLogger(__name__)


def zone_for_point(
    px: float,
    py: float,
    zones: list[Zone],
    frame_w: int,
    frame_h: int,
) -> Optional[Zone]:
    """Return the first zone (in declaration order) whose pixel polygon
    contains the normalized point (px, py in [0..1]).
    """
    for z in zones:
        if not z.enabled:
            continue
        pixel_poly = z.to_pixel_polygon(frame_w, frame_h)
        if point_in_polygon(px * frame_w, py * frame_h, pixel_poly):
            return z
    return None


def assign_bbox(
    bbox: tuple[float, float, float, float],
    zones: list[Zone],
    frame_w: int,
    frame_h: int,
) -> Optional[Zone]:
    cx, cy = bbox_centroid(bbox)
    nx, ny = cx / max(frame_w, 1), cy / max(frame_h, 1)
    return zone_for_point(nx, ny, zones, frame_w, frame_h)
