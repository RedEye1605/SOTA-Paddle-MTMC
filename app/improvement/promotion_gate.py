"""Promotion gate — fail non-zero on regression of headline metrics.

A promotion candidate (new model, new threshold, new config) is
rejected if any of the gate conditions in the audit's
``IMPROVEMENT_LOOP_PLAN.md`` Component 10 fails:

  1. ``false_merge_rate > 0.05`` (or > baseline + 0.01)
  2. ``cross_camera_match_accuracy < 0.85``
  3. ``fps < 5.0`` per camera at peak load
  4. ``gpu_memory_used_mb > 12_000`` (T4 budget)
  5. ``qdrant_query_latency_p99_ms > 200``
  6. ``postgres_write_latency_p99_ms > 50``
  7. ``ambiguous_auto_merge_rate > 0`` — the system should never
     auto-merge ambiguous decisions.

The gate is a Python script that reads a ``OfflineReport`` JSON and
exits non-zero if any condition fails. The ``PromotionGate.check()``
method is also importable so a CI can integrate it without spawning
a process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GateThresholds:
    false_merge_rate_max: float = 0.05
    cross_camera_match_accuracy_min: float = 0.85
    fps_min: float = 5.0
    gpu_memory_used_mb_max: float = 12_000.0
    qdrant_query_latency_p99_ms_max: float = 200.0
    postgres_write_latency_p99_ms_max: float = 50.0
    ambiguous_auto_merge_rate_max: float = 0.0
    id_fragmentation_rate_max: float = 0.20
    # Hard-rule enforcement (task spec rule #8 — do not claim
    # READY_FOR_LIMITED_PRODUCTION without real-model + real
    # recorded multi-camera benchmark metrics).  When True, the
    # following metrics MUST be present in the report; missing
    # values fail the gate instead of silently passing.
    require_real_metrics: bool = True
    required_metric_keys: tuple[str, ...] = (
        "false_merge_rate",
        "cross_camera_match_accuracy",
        "id_fragmentation_rate",
    )


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
        }


class PromotionGate:
    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.thresholds = thresholds or GateThresholds()

    def check(self, report: dict) -> GateResult:
        """Evaluate the gate against a flat report dict.

        The dict is the ``to_dict()`` of ``OfflineReport`` OR a
        top-level benchmark JSON. We look for the metric under
        both ``metrics.*`` (OfflineReport shape) and at the
        top level (benchmark JSON shape).
        """
        fails: list[str] = []
        warns: list[str] = []
        metrics = report.get("metrics", {}) or {}
        m = self.thresholds

        def _get(*keys: str) -> Any:
            """Look for a key at the top level first, then inside
            ``metrics``. Returns None if not present.
            """
            for k in keys:
                if k in report:
                    return report[k]
            for k in keys:
                if k in metrics:
                    return metrics[k]
            return None

        fmr = _get("false_merge_rate")
        if fmr is not None and fmr > m.false_merge_rate_max:
            fails.append(f"false_merge_rate={fmr:.4f} > threshold={m.false_merge_rate_max:.4f}")
        cma = _get("cross_camera_match_accuracy")
        if cma is not None and cma < m.cross_camera_match_accuracy_min:
            fails.append(
                f"cross_camera_match_accuracy={cma:.4f} < threshold={m.cross_camera_match_accuracy_min:.4f}"
            )
        ifr = _get("id_fragmentation_rate")
        if ifr is not None and ifr > m.id_fragmentation_rate_max:
            fails.append(
                f"id_fragmentation_rate={ifr:.4f} > threshold={m.id_fragmentation_rate_max:.4f}"
            )
        fps = _get("per_camera_analytics_fps")
        if isinstance(fps, dict):
            for cam, v in fps.items():
                if v is not None and v < m.fps_min:
                    fails.append(f"fps[{cam}]={v:.2f} < threshold={m.fps_min:.2f}")
        gpu = _get("gpu_memory_used_mb")
        if gpu is not None and gpu > m.gpu_memory_used_mb_max:
            fails.append(f"gpu_memory_used_mb={gpu:.0f} > threshold={m.gpu_memory_used_mb_max:.0f}")
        qlat = _get("qdrant_query_latency_p99_ms")
        if qlat is not None and qlat > m.qdrant_query_latency_p99_ms_max:
            fails.append(
                f"qdrant_query_latency_p99_ms={qlat:.0f} > threshold={m.qdrant_query_latency_p99_ms_max:.0f}"
            )
        plat = _get("postgres_write_latency_p99_ms")
        if plat is not None and plat > m.postgres_write_latency_p99_ms_max:
            fails.append(
                f"postgres_write_latency_p99_ms={plat:.0f} > threshold={m.postgres_write_latency_p99_ms_max:.0f}"
            )
        # Hard-rule enforcement: production_benchmark reports must
        # contain real measured accuracy metrics or the gate
        # refuses.  A benchmark that ran but did not record
        # ``false_merge_rate`` (and friends) is NOT proof that
        # the system meets the LIMITED_PRODUCTION bar — it is
        # proof that the benchmark ran.  Per task-spec hard rule
        # #8 the verdict MUST cap at READY_FOR_SHADOW_TEST until
        # the operator records a real labelled multi-camera run.
        if m.require_real_metrics and report.get("mode") == "production_benchmark":
            missing = [k for k in m.required_metric_keys if _get(k) is None]
            if missing:
                fails.append(
                    f"production_benchmark missing required real-model metrics: "
                    f"{', '.join(sorted(missing))} (required to claim "
                    f"READY_FOR_LIMITED_PRODUCTION; cap at READY_FOR_SHADOW_TEST)",
                )
        passed = not fails
        return GateResult(passed=passed, failures=fails, warnings=warns)
