"""Identity decisions — wire final_score, topology gate, ambiguous margin.

This is the single function that decides whether the resolver returns
"match" / "new" / "candidate" / "ambiguous" / "held".

The decision policy mirrors the task spec and the audit:

  * ``final_score`` is the THRESHOLD variable (NOT the raw ReID cosine).
    Fix for PATCH-008 / BUG-008.
  * The topology hard-block is a pre-check; if the candidate camera is
    not a known link, the answer is "new" even when the visual
    similarity is 0.99. Fix for BUG-008.
  * The ambiguous margin separates "match" from "ambiguous" and "held";
    if the top-1 and top-2 are too close, we hold instead of
    auto-merging. Fix for BUG-008.
  * We pass the actual ``final_score`` value (not the ReID cosine) to
    this function so the operator-tunable thresholds in
    ``app.yaml::identity`` work as documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CandidateHit:
    global_id: str
    camera_id: str
    score: float
    last_seen_at: float


def top2_margin(top1: Optional[CandidateHit], top2: Optional[CandidateHit]) -> float:
    if top1 is None:
        return 0.0
    if top2 is None:
        return 1.0
    return top1.score - top2.score


def decide_ambiguity(
    top1: Optional[CandidateHit],
    top2: Optional[CandidateHit],
    *,
    auto_match_threshold: float,
    candidate_threshold: float,
    ambiguous_margin: float,
    is_known_link: Optional[bool],
    prefer_new_id_when_ambiguous: bool = True,
    final_score: Optional[float] = None,
    # PATCH (2026-06-17, BUG-1 fix): the 5-factor weighted
    # ``final_score`` can be far below ``auto_match_threshold`` even
    # when the raw ReID cosine is 0.99. With weights
    # reid=0.55, temporal=0.20, camera=0.15, quality=0.05, zone=0.05,
    # a 150-second-old match gives final_score=0.658 even when
    # cosine=0.998 — well below the 0.82 auto_match_threshold. The
    # chain then mints a NEW global_id per person, which is the
    # operator's "1 person = N stored embeddings" complaint.
    #
    # The ReID cosine is the ground-truth signal that the 5 factors
    # are tuning around. We add a high-confidence short-circuit:
    # when ``top1.score >= reid_override_threshold`` AND the margin
    # from top-2 is at least ``ambiguous_margin``, the answer is
    # "match" regardless of ``final_score``. The override is
    # opt-in (default 0.95) and respects the topology hard-block
    # (``is_known_link is False`` still returns "new").
    reid_override_threshold: Optional[float] = 0.95,
) -> str:
    """Returns one of: 'match' | 'candidate' | 'ambiguous' | 'new' | 'held'.

    Decision policy (per the task spec, PATCH-008 + BUG-1 fix):
      - Topology hard-block: ``is_known_link is False`` → 'new' regardless
        of any score.
      - ReID override (BUG-1): if ``top1.score >= reid_override_threshold``
        AND ``margin > ambiguous_margin`` AND topology is not False
        → 'match' (regardless of ``final_score``).
      - If ``final_score`` is provided, use it as the threshold variable.
        If absent, fall back to ``top1.score`` (ReID cosine) for backward
        compatibility with the original test suite.
      - final_score >= auto_match_threshold AND margin > ambiguous_margin
        AND topology valid (None or True) → 'match'.
      - candidate_threshold <= final_score < auto_match_threshold AND
        margin > ambiguous_margin → 'candidate'.
      - final_score >= candidate_threshold AND margin <= ambiguous_margin
        → 'ambiguous' (or 'held' if prefer_new_id_when_ambiguous=False).
      - final_score < candidate_threshold → 'new'.
    """
    if top1 is None:
        return "new"
    # explicit topology hard-block
    if is_known_link is False:
        return "new"
    # PATCH (2026-06-17, BUG-1): high-confidence ReID override.
    # Only triggers when the override is enabled, the top-1 cosine
    # is at or above the override threshold, AND the margin from
    # top-2 is clear (otherwise we still want a human-eyeball
    # ambiguous hold, not a silent merge).
    margin = top2_margin(top1, top2)
    if (
        reid_override_threshold is not None
        and top1.score >= reid_override_threshold
        and margin > ambiguous_margin
    ):
        return "match"
    # Decide which score drives the threshold.
    score_for_threshold = float(final_score) if final_score is not None else float(top1.score)
    if score_for_threshold >= auto_match_threshold and margin > ambiguous_margin:
        return "match"
    if score_for_threshold >= candidate_threshold and margin > ambiguous_margin:
        return "candidate"
    if score_for_threshold >= candidate_threshold:
        # score is ok but ambiguous with top-2
        return "ambiguous" if prefer_new_id_when_ambiguous else "held"
    return "new"
