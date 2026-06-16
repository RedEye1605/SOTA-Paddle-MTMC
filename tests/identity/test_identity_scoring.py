"""Identity scoring — 5-factor weighted score + decision policy."""

from __future__ import annotations

import math
import time

import pytest

from app.identity.ambiguity import CandidateHit, decide_ambiguity
from app.identity.scoring import (
    ScoreWeights,
    camera_topology_score,
    final_score,
    score_breakdown,
    temporal_score,
    zone_transition_score,
)


def test_temporal_score_at_zero_diff() -> None:
    assert temporal_score(0.0) == pytest.approx(1.0)


def test_temporal_score_falls_off() -> None:
    assert temporal_score(60.0) < 1.0
    assert temporal_score(120.0) < temporal_score(60.0)


def test_camera_topology_score() -> None:
    assert camera_topology_score(True) == 1.0
    assert camera_topology_score(False) == 0.0
    assert camera_topology_score(None) == 0.5


def test_zone_transition_score() -> None:
    assert zone_transition_score("Z1", "Z1") == 1.0
    assert zone_transition_score("Z1", "Z2") == 0.5
    assert zone_transition_score(None, "Z1") == 0.5
    assert zone_transition_score("Z1", None) == 0.5
    assert zone_transition_score(None, None) == 0.5


def test_weights_normalize_to_one() -> None:
    w = ScoreWeights()
    n = w.normalized()
    assert math.isclose(
        n.reid_weight + n.temporal_weight + n.camera_weight + n.quality_weight + n.zone_weight,
        1.0,
        abs_tol=1e-9,
    )


def test_final_score_in_unit_interval() -> None:
    w = ScoreWeights()
    for s in [0.0, 0.3, 0.7, 1.0]:
        for t in [0.0, 60.0, 600.0]:
            for link in [None, True, False]:
                for q in [0.0, 0.5, 1.0]:
                    for z1, z2 in [(None, None), ("Z1", "Z1"), ("Z1", "Z2")]:
                        v = final_score(s, t, link, q, z1, z2, w)
                        assert 0.0 <= v <= 1.0


def test_breakdown_includes_all_components() -> None:
    b = score_breakdown(
        reid_similarity=0.85,
        time_diff_seconds=30.0,
        is_known_link=True,
        tracklet_quality=0.7,
        prev_zone_id="Z1",
        new_zone_id="Z2",
        weights=ScoreWeights(),
    )
    for k in [
        "reid_similarity",
        "temporal_score",
        "camera_topology_score",
        "quality_score",
        "zone_score",
        "final_score",
    ]:
        assert k in b


# ---- Decision policy ----
def test_decide_match_when_score_and_margin_and_topology_ok() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.9, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.7, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
    )
    assert decision == "match"


def test_decide_new_when_no_candidate() -> None:
    decision = decide_ambiguity(
        None,
        None,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=None,
    )
    assert decision == "new"


def test_decide_new_when_topology_disabled() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.99, time.time())
    decision = decide_ambiguity(
        top1,
        None,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=False,
    )
    assert decision == "new"


def test_decide_ambiguous_when_margin_too_close() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.9, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.88, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
    )
    assert decision in ("ambiguous", "candidate", "new")


def test_decide_candidate_when_score_below_match_but_above_thresh() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.75, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.5, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
    )
    assert decision == "candidate"
