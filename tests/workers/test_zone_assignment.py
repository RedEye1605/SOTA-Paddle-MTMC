"""Zone assignment — polygon point-in-polygon tests."""

from __future__ import annotations

import json

import pytest

from app.zones.assignment import assign_bbox
from app.zones.polygon import (
    Zone,
    parse_zones,
    point_in_polygon,
    bbox_centroid,
)


def test_point_in_polygon_inside_outside() -> None:
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon(50, 50, square) is True
    assert point_in_polygon(150, 50, square) is False
    # Boundary: ray-casting treats it as either; the safe invariant is
    # that a clearly outside point returns False.
    assert point_in_polygon(150, 150, square) is False


def test_point_in_polygon_concave() -> None:
    # L-shape: outside the concave bite
    poly = [(0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)]
    # Outside the bite
    assert point_in_polygon(75, 75, poly) is False
    # Inside the rectangle (top-left)
    assert point_in_polygon(25, 75, poly) is True


def test_bbox_centroid() -> None:
    assert bbox_centroid((0, 0, 100, 50)) == (50, 25)


def test_parse_zones() -> None:
    rows = [
        {
            "zone_id": "Z1",
            "camera_id": "CAM_01",
            "name": "test",
            "polygon_json": json.dumps([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
            "zone_type": "entry",
            "is_entry_zone": True,
            "is_exit_zone": False,
            "enabled": True,
        }
    ]
    by_cam = parse_zones(rows)
    assert "CAM_01" in by_cam
    assert by_cam["CAM_01"][0].zone_id == "Z1"
    assert by_cam["CAM_01"][0].is_entry_zone is True


def test_parse_zones_invalid_json_raises() -> None:
    rows = [
        {
            "zone_id": "BAD",
            "camera_id": "CAM_01",
            "name": "x",
            "polygon_json": "not-json",
            "zone_type": "x",
            "is_entry_zone": False,
            "is_exit_zone": False,
            "enabled": True,
        }
    ]
    with pytest.raises(ValueError):
        parse_zones(rows)


def test_assign_bbox_returns_zone() -> None:
    zone = Zone(
        zone_id="Z1",
        camera_id="CAM_01",
        name="test",
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        zone_type="entry",
        is_entry_zone=True,
        is_exit_zone=False,
        enabled=True,
    )
    # bbox centered at (50, 50) in a 100x100 frame -> centroid at (0.5, 0.5)
    z = assign_bbox((0, 0, 100, 100), [zone], frame_w=100, frame_h=100)
    assert z is not None
    assert z.zone_id == "Z1"
