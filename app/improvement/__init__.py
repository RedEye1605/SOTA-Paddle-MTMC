"""Improvement-loop package — first minimal version.

Implements Components 1, 3, 4, 5, 6, 7, 8, 12 of the audit's
``IMPROVEMENT_LOOP_PLAN.md``:

  - ``evidence_sampler``        — every Nth tracklet's ``best_crop_uri``
                                 is staged to a labeler bucket.
  - ``dataset_manifest``        — frozen benchmark manifest format.
  - ``offline_evaluator``       — skeleton that re-runs the resolver
                                 against a labeled ground-truth set and
                                 produces a metrics report.
  - ``promotion_gate``          — deploy gate that fails non-zero on
                                 regression of the headline metrics.
  - ``metrics_report``          — JSON output of the 22 metrics in the
                                 audit (defined in
                                 ``IMPROVEMENT_LOOP_PLAN.md``).
"""

from __future__ import annotations

# Re-export the public submodules so callers can do
#   from app.improvement import evidence_sampler
# without a deeper import path.
from . import (
    dataset_manifest,
    evidence_sampler,
    offline_evaluator,
    promotion_gate,
)

__all__ = [
    "dataset_manifest",
    "evidence_sampler",
    "offline_evaluator",
    "promotion_gate",
]
