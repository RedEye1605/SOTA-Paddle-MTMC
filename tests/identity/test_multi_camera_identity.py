"""Multi-camera identity — staged retrieval semantics + topology gating.

The resolver is integration-tested elsewhere (it needs a live Qdrant +
Postgres + Redis). Here we test the *decisions* and *staged retrieval*
in isolation.
"""

from __future__ import annotations

import time

from app.identity.ambiguity import CandidateHit, decide_ambiguity
from app.identity.camera_topology import CameraTopology


def test_no_topology_link_blocks_match() -> None:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_01",
                "to_camera_id": "CAM_04",
                "min_travel_seconds": 0,
                "max_travel_seconds": 0,
                "transition_probability": 0.0,
                "enabled": False,
                "notes": "",
            },
        ]
    )
    top1 = CandidateHit("GID-X", "CAM_04", 0.99, time.time())
    # A disabled row from CAM_01 -> CAM_04 is a hard block.
    decision = decide_ambiguity(
        top1,
        None,
        auto_match_threshold=0.5,
        candidate_threshold=0.4,
        ambiguous_margin=0.01,
        is_known_link=t.is_known_link("CAM_01", "CAM_04"),
    )
    assert decision == "new"


def test_candidate_cameras_respect_enabled_only() -> None:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_01",
                "to_camera_id": "CAM_02",
                "min_travel_seconds": 1,
                "max_travel_seconds": 60,
                "transition_probability": 0.7,
                "enabled": True,
                "notes": "",
            },
            {
                "from_camera_id": "CAM_01",
                "to_camera_id": "CAM_99",
                "min_travel_seconds": 1,
                "max_travel_seconds": 60,
                "transition_probability": 0.5,
                "enabled": False,
                "notes": "",
            },
        ]
    )
    candidates = t.candidate_cameras_for("CAM_01")
    assert "CAM_02" in candidates
    assert "CAM_99" not in candidates


def test_24h_fallback_not_in_topology() -> None:
    """The 24h fallback is OUTSIDE the topology. The resolver uses
    `cameras_by_seen_recent` for the fallback, not `camera_links`."""
    t = CameraTopology()
    # No rows
    out = t.candidate_cameras_for("CAM_01")
    assert out == []
