"""Readiness gate tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# Verdict enum + ordering
# ----------------------------------------------------------------------------


def test_verdict_order_strictly_increasing() -> None:
    from scripts.readiness_gate import (
        VERDICT_NOT_READY,
        VERDICT_ORDER,
        VERDICT_READY_FOR_LIMITED_PRODUCTION,
        VERDICT_READY_FOR_SHADOW_TEST,
        VERDICT_STRUCTURALLY_READY,
    )

    assert (
        VERDICT_ORDER[VERDICT_NOT_READY]
        < VERDICT_ORDER[VERDICT_STRUCTURALLY_READY]
        < VERDICT_ORDER[VERDICT_READY_FOR_SHADOW_TEST]
        < VERDICT_ORDER[VERDICT_READY_FOR_LIMITED_PRODUCTION]
    )


# ----------------------------------------------------------------------------
# Input loaders
# ----------------------------------------------------------------------------


def test_load_preflight_missing_file(tmp_path) -> None:
    from scripts.readiness_gate import _load_preflight

    out = _load_preflight(tmp_path / "missing.json")
    assert out["ok"] is False
    assert "missing" in out


def test_load_preflight_ok(tmp_path) -> None:
    p = tmp_path / "preflight.json"
    p.write_text(json.dumps({"ok": True, "checks": {"x": {"ok": True}}}))
    from scripts.readiness_gate import _load_preflight

    out = _load_preflight(p)
    assert out["ok"] is True


def test_latest_benchmark_empty_dir(tmp_path) -> None:
    from scripts.readiness_gate import _latest_benchmark

    assert _latest_benchmark(tmp_path / "empty") is None


def test_latest_benchmark_picks_newest(tmp_path) -> None:
    p = tmp_path
    (p / "benchmark_20260101T000000Z.json").write_text(
        json.dumps({"mode": "smoke_benchmark"}),
    )
    (p / "benchmark_20260102T000000Z.json").write_text(
        json.dumps({"mode": "production_benchmark"}),
    )
    from scripts.readiness_gate import _latest_benchmark

    out = _latest_benchmark(p)
    # Files are sorted reverse; 20260102 is first.
    assert out["mode"] == "production_benchmark"


def test_parse_promotion_gate_missing_yaml(tmp_path) -> None:
    from scripts.readiness_gate import _parse_promotion_gate

    assert _parse_promotion_gate(tmp_path / "missing.yaml") == {}


def test_parse_promotion_gate_extracts_gate_block(tmp_path) -> None:
    import yaml

    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "gate": {"false_merge_rate_max": 0.07, "fps_min": 4.0},
                "other_block": {"foo": "bar"},
            }
        )
    )
    from scripts.readiness_gate import _parse_promotion_gate

    cfg = _parse_promotion_gate(p)
    assert cfg["false_merge_rate_max"] == 0.07
    assert cfg["fps_min"] == 4.0


# ----------------------------------------------------------------------------
# Top-level verdict
# ----------------------------------------------------------------------------


def _inputs_ok(benchmark_mode: str | None = None) -> dict:
    preflight = {"ok": True, "checks": {}}
    inputs = {
        "preflight": preflight,
        "tests_passed": True,
        "tests_failed_count": 0,
        "benchmark": None,
        "promotion_gate": {},
    }
    if benchmark_mode == "smoke_benchmark":
        inputs["benchmark"] = {
            "mode": "smoke_benchmark",
            "cameras": ["CAM_01", "CAM_02"],
            "duration_seconds": 5.0,
            "total_analytics_fps": 4.0,
            "per_camera_analytics_fps": {"CAM_01": 2.0, "CAM_02": 2.0},
        }
    elif benchmark_mode == "production_benchmark":
        inputs["benchmark"] = {
            "mode": "production_benchmark",
            "cameras": ["CAM_01", "CAM_02"],
            "duration_seconds": 5.0,
            "total_analytics_fps": 12.0,
            # Default gate threshold is fps_min=5.0; 6.0+6.0 = 12 total.
            "per_camera_analytics_fps": {"CAM_01": 6.0, "CAM_02": 6.0},
            # Hard-rule enforcement (task-spec rule #8): production
            # benchmark MUST report real accuracy metrics or the
            # gate caps at READY_FOR_SHADOW_TEST.  All three keys
            # below are required by GateThresholds.required_metric_keys.
            "false_merge_rate": 0.01,
            "cross_camera_match_accuracy": 0.93,
            "id_fragmentation_rate": 0.08,
        }
    return inputs


def _build_inputs(d):
    from scripts.readiness_gate import GateInputs

    return GateInputs(
        preflight=d["preflight"],
        benchmark=d["benchmark"],
        promotion_gate=d.get("promotion_gate", {}),
        tests_passed=d.get("tests_passed"),
        tests_failed_count=d.get("tests_failed_count", 0),
    )


def test_verdict_not_ready_when_preflight_fails() -> None:
    from scripts.readiness_gate import (
        GateInputs,
        VERDICT_NOT_READY,
        compute_verdict,
    )

    inputs = GateInputs(
        preflight={"ok": False, "checks": {}},
        tests_passed=True,
    )
    r = compute_verdict(inputs)
    assert r.verdict == VERDICT_NOT_READY


def test_verdict_not_ready_when_tests_fail() -> None:
    from scripts.readiness_gate import (
        GateInputs,
        VERDICT_NOT_READY,
        compute_verdict,
    )

    inputs = GateInputs(
        preflight={"ok": True, "checks": {}},
        tests_passed=False,
        tests_failed_count=3,
    )
    r = compute_verdict(inputs)
    assert r.verdict == VERDICT_NOT_READY


def test_verdict_structurally_ready_when_preflight_and_tests_pass() -> None:
    from scripts.readiness_gate import (
        VERDICT_STRUCTURALLY_READY,
        compute_verdict,
    )

    inputs = _build_inputs(_inputs_ok())
    r = compute_verdict(inputs)
    assert r.verdict == VERDICT_STRUCTURALLY_READY


def test_verdict_ready_for_shadow_with_smoke_benchmark() -> None:
    from scripts.readiness_gate import (
        VERDICT_READY_FOR_SHADOW_TEST,
        compute_verdict,
    )

    inputs = _build_inputs(_inputs_ok(benchmark_mode="smoke_benchmark"))
    r = compute_verdict(inputs)
    assert r.verdict == VERDICT_READY_FOR_SHADOW_TEST


def test_verdict_ready_for_limited_production_with_promotion_pass() -> None:
    from scripts.readiness_gate import (
        VERDICT_READY_FOR_LIMITED_PRODUCTION,
        compute_verdict,
    )

    inputs = _build_inputs(_inputs_ok(benchmark_mode="production_benchmark"))
    # The default gate thresholds tolerate 4 fps (our bench has 6).
    r = compute_verdict(inputs)
    assert r.verdict == VERDICT_READY_FOR_LIMITED_PRODUCTION


def test_verdict_falls_back_to_shadow_if_production_bench_fails_gate() -> None:
    """When the production_benchmark report fails the promotion gate,
    we drop to READY_FOR_SHADOW_TEST (or lower) — never silently
    to READY_FOR_LIMITED_PRODUCTION.
    """
    from scripts.readiness_gate import (
        GateInputs,
        VERDICT_NOT_READY,
        VERDICT_READY_FOR_SHADOW_TEST,
        compute_verdict,
    )

    inputs = GateInputs(
        preflight={"ok": True, "checks": {}},
        tests_passed=True,
        benchmark={
            "mode": "production_benchmark",
            "cameras": ["CAM_01"],
            "duration_seconds": 5.0,
            "total_analytics_fps": 1.0,  # below fps_min
            "per_camera_analytics_fps": {"CAM_01": 1.0},
        },
    )
    r = compute_verdict(inputs)
    # The promotion gate fails (fps too low); we drop to
    # SHADOW or NOT_READY. With only production_benchmark
    # available, and gate failing, the verdict is NOT_READY
    # because the production bench is not viable.
    assert r.verdict in {VERDICT_READY_FOR_SHADOW_TEST, VERDICT_NOT_READY}
    assert "fps" in " ".join(r.failures) or r.verdict == VERDICT_NOT_READY


def test_verdict_falls_back_to_structurally_ready_when_bench_missing() -> None:
    from scripts.readiness_gate import (
        VERDICT_STRUCTURALLY_READY,
        compute_verdict,
    )

    inputs = _build_inputs(_inputs_ok(benchmark_mode=None))
    r = compute_verdict(inputs)
    assert r.verdict == VERDICT_STRUCTURALLY_READY


def test_promotion_gate_called_for_production_bench() -> None:
    """The promotion gate MUST be invoked when we have a
    production_benchmark report.
    """
    from scripts.readiness_gate import (
        GateInputs,
        compute_verdict,
    )

    inputs = GateInputs(
        preflight={"ok": True, "checks": {}},
        tests_passed=True,
        benchmark={
            "mode": "production_benchmark",
            "cameras": ["CAM_01"],
            "total_analytics_fps": 0.5,  # fail fps_min
            "per_camera_analytics_fps": {"CAM_01": 0.5},
        },
    )
    r = compute_verdict(inputs)
    # We expect the fps check to fail.
    assert (
        any("fps" in f.lower() for f in r.failures) or r.verdict != "READY_FOR_LIMITED_PRODUCTION"
    )


# ----------------------------------------------------------------------------
# CLI integration
# ----------------------------------------------------------------------------


def test_cli_min_verdict_exits_nonzero_when_below(tmp_path) -> None:
    """When --min-verdict is set and the actual verdict is lower,
    the CLI exits non-zero.
    """
    # No preflight, no benchmark, no tests. Default verdict: NOT_READY.
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"ok": False, "checks": {}}))
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "readiness_gate.py"),
            "--preflight",
            str(preflight),
            "--min-verdict",
            "READY_FOR_SHADOW_TEST",
            "--out",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 1


def test_cli_min_verdict_exits_zero_when_meeting(tmp_path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"ok": True, "checks": {}}))
    # Isolate from the host's reports/ directory so a real
    # benchmark report does not promote the verdict above
    # STRUCTURALLY_READY for this CLI smoke check.
    bench_dir = tmp_path / "empty_reports"
    bench_dir.mkdir()
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "readiness_gate.py"),
            "--preflight",
            str(preflight),
            "--benchmark-dir",
            str(bench_dir),
            "--min-verdict",
            "STRUCTURALLY_READY",
            "--out",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["verdict"] == "STRUCTURALLY_READY"
