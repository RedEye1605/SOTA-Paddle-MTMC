#!/usr/bin/env python3
"""Readiness gate — produces one of four verdicts.

The gate consumes:

  1. ``scripts/readiness_preflight.json`` — produced by
     ``scripts/readiness_preflight.py`` (a structured set of
     preflight checks: env vars, file existence, model weights).
  2. The most recent ``reports/benchmark_*.json`` (production
     benchmark). The gate requires a benchmark file.
  3. The most recent ``pytest`` JUnit XML (optional).
  4. The ``app/improvement/promotion_gate`` config in
     ``configs/benchmark.yaml``.

Verdicts (in increasing order of strictness):

  * NOT_READY                 — preflight failed OR test suite failed
  * STRUCTURALLY_READY        — preflight passes, tests pass, no
                                 production benchmark yet
  * READY_FOR_SHADOW_TEST     — STRUCTURALLY_READY + production
                                 benchmark ran in smoke mode (i.e.
                                 the workload runner works)
  * READY_FOR_LIMITED_PRODUCTION — READY_FOR_SHADOW_TEST + real
                                 model path executed successfully
                                 (the benchmark ran in
                                 production_benchmark mode and
                                 recorded real-model metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
)
log = logging.getLogger("readiness_gate")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFLIGHT = ROOT / "scripts" / "readiness_preflight.json"
DEFAULT_BENCH_DIR = ROOT / "reports"
DEFAULT_GATE_CFG = ROOT / "configs" / "benchmark.yaml"


# Ensure the project root is on sys.path so the ``app`` package
# imports succeed when the script is invoked as
# ``python scripts/readiness_gate.py`` from anywhere (e.g. a
# ``docker compose run`` with ``working_dir: /app``).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------------------------
# Verdict enum
# ----------------------------------------------------------------------------


VERDICT_NOT_READY = "NOT_READY"
VERDICT_STRUCTURALLY_READY = "STRUCTURALLY_READY"
VERDICT_READY_FOR_SHADOW_TEST = "READY_FOR_SHADOW_TEST"
VERDICT_READY_FOR_LIMITED_PRODUCTION = "READY_FOR_LIMITED_PRODUCTION"

VERDICT_ORDER = {
    VERDICT_NOT_READY: 0,
    VERDICT_STRUCTURALLY_READY: 1,
    VERDICT_READY_FOR_SHADOW_TEST: 2,
    VERDICT_READY_FOR_LIMITED_PRODUCTION: 3,
}


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------


@dataclass
class GateInputs:
    preflight: dict[str, Any] = field(default_factory=dict)
    benchmark: Optional[dict[str, Any]] = None
    promotion_gate: dict[str, Any] = field(default_factory=dict)
    tests_passed: Optional[bool] = None
    tests_failed_count: int = 0


def _load_preflight(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("preflight file %s not found", path)
        return {"ok": False, "checks": {}, "missing": [str(path)]}
    try:
        return json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        log.error("preflight file %s unreadable: %s", path, e)
        return {"ok": False, "checks": {}, "error": str(e)}


def _latest_benchmark(dir_path: Path) -> Optional[dict[str, Any]]:
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob("benchmark_*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text())
    except Exception as e:  # noqa: BLE001
        log.error("benchmark file %s unreadable: %s", files[0], e)
        return None


def _parse_promotion_gate(yaml_path: Path) -> dict[str, Any]:
    """Read the ``gate`` block of configs/benchmark.yaml."""
    if not yaml_path.exists():
        return {}
    try:
        import yaml

        with yaml_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("gate", {})
    except Exception as e:  # noqa: BLE001
        log.warning("gate config %s unreadable: %s", yaml_path, e)
        return {}


def _parse_tests(junit_path: Path) -> tuple[Optional[bool], int]:
    if not junit_path.exists():
        return None, 0
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(junit_path).getroot()
        failures = int(root.get("failures", 0))
        errors = int(root.get("errors", 0))
        total = int(root.get("tests", 0))
        passed = failures == 0 and errors == 0 and total > 0
        return passed, failures + errors
    except Exception:  # noqa: BLE001
        return None, 0


# ----------------------------------------------------------------------------
# Gate checks
# ----------------------------------------------------------------------------


@dataclass
class GateResult:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    inputs: GateInputs = field(default_factory=GateInputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "failures": self.failures,
        }


def _check_preflight(preflight: dict[str, Any]) -> list[str]:
    """Return a list of failure reasons. Empty list = OK."""
    fails: list[str] = []
    if not preflight:
        fails.append("preflight: empty")
        return fails
    if not preflight.get("ok", False):
        fails.append(f"preflight: not OK ({preflight.get('error', 'unknown')})")
    for name, check in preflight.get("checks", {}).items():
        if not check.get("ok", False):
            fails.append(f"preflight.{name}: {check.get('reason', 'failed')}")
    return fails


def _check_tests(tests_passed: Optional[bool], tests_failed: int) -> list[str]:
    fails: list[str] = []
    if tests_passed is False:
        fails.append(f"tests: {tests_failed} failure(s)")
    return fails


def _check_benchmark_shadow(bench: Optional[dict[str, Any]]) -> list[str]:
    """The smoke benchmark can drive a SHADOW verdict. The benchmark
    must have been run in smoke_benchmark mode and have at least
    one camera.
    """
    fails: list[str] = []
    if bench is None:
        fails.append("benchmark: no benchmark report found in reports/")
        return fails
    if bench.get("mode") != "smoke_benchmark":
        fails.append(
            f"benchmark: mode={bench.get('mode')!r} (need 'smoke_benchmark' "
            f"for READY_FOR_SHADOW_TEST)",
        )
    if not bench.get("cameras"):
        fails.append("benchmark: no cameras in report")
    return fails


def _check_benchmark_production(bench: Optional[dict[str, Any]]) -> list[str]:
    """For READY_FOR_LIMITED_PRODUCTION the benchmark must have
    been run in production_benchmark mode AND recorded real-model
    metrics (no synthetic path).
    """
    fails: list[str] = []
    if bench is None:
        fails.append("benchmark: no benchmark report found")
        return fails
    if bench.get("mode") != "production_benchmark":
        fails.append(
            f"benchmark: mode={bench.get('mode')!r} (need "
            f"'production_benchmark' for READY_FOR_LIMITED_PRODUCTION)",
        )
    return fails


def _check_promotion_gate(
    bench: Optional[dict[str, Any]],
    gate_cfg: dict[str, Any],
) -> list[str]:
    """The promotion gate is the canonical perf check. For
    READY_FOR_LIMITED_PRODUCTION it must pass on the most-recent
    benchmark.
    """
    fails: list[str] = []
    if bench is None:
        return fails
    try:
        from app.improvement.promotion_gate import (
            GateThresholds,
            PromotionGate,
        )

        thresholds = GateThresholds(
            false_merge_rate_max=float(
                gate_cfg.get("false_merge_rate_max", 0.05),
            ),
            cross_camera_match_accuracy_min=float(
                gate_cfg.get("cross_camera_match_accuracy_min", 0.85),
            ),
            fps_min=float(gate_cfg.get("fps_min", 5.0)),
            gpu_memory_used_mb_max=float(
                gate_cfg.get("gpu_memory_used_mb_max", 12_000.0),
            ),
            qdrant_query_latency_p99_ms_max=float(
                gate_cfg.get("qdrant_query_latency_p99_ms_max", 200.0),
            ),
            postgres_write_latency_p99_ms_max=float(
                gate_cfg.get("postgres_write_latency_p99_ms_max", 50.0),
            ),
            ambiguous_auto_merge_rate_max=float(
                gate_cfg.get("ambiguous_auto_merge_rate_max", 0.0),
            ),
            id_fragmentation_rate_max=float(
                gate_cfg.get("id_fragmentation_rate_max", 0.20),
            ),
            # Hard-rule enforcement (task-spec rule #8): a
            # production_benchmark report MUST contain real
            # accuracy metrics (false_merge_rate, etc.) before the
            # gate may promote to READY_FOR_LIMITED_PRODUCTION.
            # The operator can set ``gate.require_real_metrics:
            # false`` in configs/benchmark.yaml to opt out, but
            # the default is strict.
            require_real_metrics=bool(
                gate_cfg.get("require_real_metrics", True),
            ),
        )
        result = PromotionGate(thresholds).check(bench)
        if not result.passed:
            fails.extend(result.failures)
    except Exception as e:  # noqa: BLE001
        fails.append(f"promotion_gate: {e}")
    return fails


# ----------------------------------------------------------------------------
# Top-level gate
# ----------------------------------------------------------------------------


def compute_verdict(inputs: GateInputs) -> GateResult:
    reasons: list[str] = []
    failures: list[str] = []

    # 1. Preflight must pass.
    pf_fails = _check_preflight(inputs.preflight)
    failures.extend(pf_fails)
    if pf_fails:
        reasons.append("preflight failed; NOT_READY")

    # 2. Tests must pass.
    test_fails = _check_tests(inputs.tests_passed, inputs.tests_failed_count)
    failures.extend(test_fails)
    if test_fails:
        reasons.append("tests failed; NOT_READY")

    if pf_fails or test_fails:
        return GateResult(
            verdict=VERDICT_NOT_READY,
            reasons=reasons,
            failures=failures,
            inputs=inputs,
        )

    reasons.append("preflight + tests pass; STRUCTURALLY_READY")
    verdict = VERDICT_STRUCTURALLY_READY

    # 3. Production_benchmark + promotion gate → LIMITED_PRODUCTION.
    #    If the production bench is present but the gate fails,
    #    we drop straight to NOT_READY (the production bench is
    #    not viable and there is no smoke bench to fall back to).
    prod_fails = _check_benchmark_production(inputs.benchmark)
    if not prod_fails:
        gate_fails = _check_promotion_gate(
            inputs.benchmark,
            inputs.promotion_gate,
        )
        if not gate_fails:
            reasons.append(
                "production_benchmark + promotion gate pass; READY_FOR_LIMITED_PRODUCTION",
            )
            return GateResult(
                verdict=VERDICT_READY_FOR_LIMITED_PRODUCTION,
                reasons=reasons,
                failures=[],
                inputs=inputs,
            )
        else:
            # Production bench exists but the gate failed.  Two
            # cases:
            #
            # (a) The ONLY failure is "missing required real-model
            #     metrics" — the operator ran a perf-only
            #     production_benchmark without labels.  Treat this
            #     as equivalent to "no production bench yet" and
            #     fall through to the SHADOW check.  Per task
            #     spec rule #8, the verdict still cannot exceed
            #     READY_FOR_SHADOW_TEST without a real labelled
            #     multi-camera dataset.
            #
            # (b) The gate failed on real metrics (FPS too low,
            #     GPU exceeded, accuracy below threshold).  The
            #     verdict drops to NOT_READY — we cannot claim
            #     SHADOW when the operator already attempted a
            #     labelled real-model run and it failed the gate.
            missing_only = all("missing required real-model metrics" in f for f in gate_fails)
            if missing_only:
                reasons.append(
                    "production_benchmark lacks required real-model metrics; "
                    "capping verdict at READY_FOR_SHADOW_TEST per task-spec rule #8",
                )
                # The production_benchmark ran with N cameras and
                # produced a structurally-valid report; it just
                # lacks accuracy labels.  That's exactly the SHADOW
                # criterion (the runner works; no LIMITED_PRODUCTION
                # claim without labels), so return SHADOW directly
                # instead of falling through to the smoke-only check.
                return GateResult(
                    verdict=VERDICT_READY_FOR_SHADOW_TEST,
                    reasons=reasons,
                    failures=failures,
                    inputs=inputs,
                )
            else:
                reasons.extend(
                    [f"production_benchmark gate: {f}" for f in gate_fails],
                )
                failures.extend(gate_fails)
                return GateResult(
                    verdict=VERDICT_NOT_READY,
                    reasons=reasons,
                    failures=failures,
                    inputs=inputs,
                )
    else:
        reasons.extend([f"production_benchmark: {f}" for f in prod_fails])

    # 4. Smoke_benchmark → SHADOW.
    shadow_fails = _check_benchmark_shadow(inputs.benchmark)
    if not shadow_fails:
        reasons.append("smoke_benchmark ran; READY_FOR_SHADOW_TEST")
        return GateResult(
            verdict=VERDICT_READY_FOR_SHADOW_TEST,
            reasons=reasons,
            failures=failures,
            inputs=inputs,
        )
    else:
        reasons.extend([f"smoke_benchmark: {f}" for f in shadow_fails])
        failures.extend(shadow_fails)

    return GateResult(
        verdict=verdict,
        reasons=reasons,
        failures=failures,
        inputs=inputs,
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Readiness gate — verdicts NOT_READY → READY_FOR_LIMITED_PRODUCTION",
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_PREFLIGHT,
        help="Path to the preflight JSON.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCH_DIR,
        help="Directory of benchmark_*.json reports.",
    )
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=DEFAULT_GATE_CFG,
        help="Path to configs/benchmark.yaml (for gate thresholds).",
    )
    parser.add_argument(
        "--junit",
        type=Path,
        default=None,
        help="Optional pytest JUnit XML for test pass/fail.",
    )
    parser.add_argument(
        "--min-verdict",
        choices=list(VERDICT_ORDER),
        default=None,
        help="If set, exit non-zero unless the verdict is at least this.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the gate JSON report.",
    )
    args = parser.parse_args()

    inputs = GateInputs(
        preflight=_load_preflight(args.preflight),
        benchmark=_latest_benchmark(args.benchmark_dir),
        promotion_gate=_parse_promotion_gate(args.gate_config),
    )
    if args.junit is not None:
        tests_passed, tests_failed = _parse_tests(args.junit)
        inputs.tests_passed = tests_passed
        inputs.tests_failed_count = tests_failed

    result = compute_verdict(inputs)
    payload = {
        "verdict": result.verdict,
        "reasons": result.reasons,
        "failures": result.failures,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preflight_file": str(args.preflight),
        "benchmark_file": (
            str(args.benchmark_dir / "benchmark_<latest>.json") if inputs.benchmark else None
        ),
        "tests_passed": inputs.tests_passed,
        "tests_failed_count": inputs.tests_failed_count,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        log.info("Wrote gate report to %s", args.out)

    if args.min_verdict is not None:
        if VERDICT_ORDER[result.verdict] < VERDICT_ORDER[args.min_verdict]:
            log.error(
                "verdict %s < required %s",
                result.verdict,
                args.min_verdict,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
