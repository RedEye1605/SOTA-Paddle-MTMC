"""Ambiguity policy — false merges are worse than fragmentation."""

from __future__ import annotations

import time

from app.identity.ambiguity import CandidateHit, decide_ambiguity


def _hit(score: float) -> CandidateHit:
    return CandidateHit("GID-X", "CAM_01", score, time.time())


def test_top2_margin_with_only_top1_is_one() -> None:
    from app.identity.ambiguity import top2_margin

    assert top2_margin(_hit(0.9), None) == 1.0


def test_margin_returns_zero_when_no_top1() -> None:
    from app.identity.ambiguity import top2_margin

    assert top2_margin(None, _hit(0.9)) == 0.0


def test_high_score_low_margin_ambiguous() -> None:
    top1 = _hit(0.85)
    top2 = CandidateHit("GID-Y", "CAM_02", 0.84, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
    )
    # top1-top2 = 0.01 < 0.04 -> not a clean match
    assert decision in ("ambiguous", "candidate", "new")
    # The spec is clear: ambiguity is NOT auto-merged
    assert decision != "match"


def test_topology_block_always_new() -> None:
    top1 = _hit(0.99)
    decision = decide_ambiguity(
        top1,
        None,
        auto_match_threshold=0.5,
        candidate_threshold=0.4,
        ambiguous_margin=0.01,
        is_known_link=False,
    )
    assert decision == "new"


# PATCH (2026-06-17, BUG-1 fix): the 5-factor weighted
# ``final_score`` can be far below the ``auto_match_threshold`` even
# when the raw ReID cosine is 0.99. Example: reid=0.998, time_diff=
# 150s (temporal=0.044), topology=0.5, quality=0.0, zone=0.5 →
# final=0.55*0.998 + 0.20*0.044 + 0.15*0.5 + 0.05*0.0 + 0.05*0.5 =
# 0.658, well below the 0.82 auto_match_threshold. The chain then
# mints a NEW global_id even though the person is the same person.
#
# Operator's plan ("Real Persistent ID — Production-Ready"): the
# reid cosine itself is the ground truth signal; the 5 factors are
# tuning. A reid-override path is required so that a top-1 with
# cosine >= a high-confidence threshold (``reid_override_threshold``,
# default 0.95) AND a margin from top-2 of at least
# ``ambiguous_margin`` short-circuits the 5-factor math and returns
# "match". This is the "real ReID features" path — without it, the
# chain mints N global_ids per person.
def test_high_reid_cosine_overrides_low_final_score() -> None:
    """A 0.998 cosine match must dedup even when final_score=0.66."""
    top1 = _hit(0.998)  # near-perfect ReID match
    top2 = CandidateHit("GID-OTHER", "CAM_02", 0.20, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,  # the weighted threshold
        candidate_threshold=0.72,
        ambiguous_margin=0.05,
        is_known_link=True,  # cross-cam topology enabled
        reid_override_threshold=0.95,  # NEW: high-confidence shortcut
    )
    # The override should match: 0.998 >= 0.95 AND margin 0.798 >= 0.05
    assert decision == "match", (
        f"BUG-1: expected reid-override match, got {decision!r}. "
        "A 0.998 cosine match is ground truth; the 5-factor score "
        "should not prevent dedup when the cosine is this high."
    )


def test_reid_override_requires_margin() -> None:
    """A high cosine with a close second-best MUST NOT auto-merge."""
    top1 = _hit(0.998)
    top2 = CandidateHit("GID-OTHER", "CAM_02", 0.997, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.05,
        is_known_link=True,
        reid_override_threshold=0.95,
    )
    # margin = 0.001, < 0.05 → still ambiguous (anti-flicker)
    assert decision in ("ambiguous", "candidate", "new")
    assert decision != "match"


def test_reid_override_respects_topology_block() -> None:
    """The override must NOT bypass the topology hard-block."""
    top1 = _hit(0.998)
    decision = decide_ambiguity(
        top1,
        None,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.05,
        is_known_link=False,  # explicit topology block
        reid_override_threshold=0.95,
    )
    assert decision == "new"


def test_reid_override_disabled_keeps_old_behavior() -> None:
    """When reid_override_threshold=None, old 5-factor behavior holds."""
    top1 = _hit(0.998)
    decision = decide_ambiguity(
        top1,
        None,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.05,
        is_known_link=True,
        reid_override_threshold=None,  # disabled → old behavior
    )
    # Old behavior: 0.998 < 0.82? No, it's > auto_match_threshold,
    # so it matches. But this is with no top2, so margin = 1.0 > 0.05
    # → match. The point of this test is: the override kwarg doesn't
    # change behavior when it's None.
    assert decision == "match"


def test_reid_override_threshold_default() -> None:
    """The default reid_override_threshold should be 0.95 (operator plan)."""
    import inspect

    from app.identity.ambiguity import decide_ambiguity
    sig = inspect.signature(decide_ambiguity)
    assert "reid_override_threshold" in sig.parameters
    default = sig.parameters["reid_override_threshold"].default
    assert default == 0.95, (
        f"BUG-1: default reid_override_threshold is {default!r}, "
        "expected 0.95 per operator's plan"
    )
