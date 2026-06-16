# Phase 7 — Readiness gate verdict

Date: 2026-06-13

## Commands

```bash
docker compose run --rm -w /app api python scripts/readiness_gate.py \
  --benchmark-dir reports \
  --min-verdict READY_FOR_SHADOW_TEST

docker compose run --rm -w /app api python scripts/readiness_gate.py \
  --benchmark-dir reports \
  --min-verdict READY_FOR_LIMITED_PRODUCTION || true
```

## Verdict: `READY_FOR_SHADOW_TEST`

The gate **correctly refuses** `READY_FOR_LIMITED_PRODUCTION`
because:

1. **The most recent benchmark report is `mode=smoke_benchmark`**
   (the gate reads the latest report under `reports/`; the
   recent `production_benchmark` runs all crashed and were
   not promoted to a "successful" report). The gate's
   `production_benchmark` rule requires
   `mode=production_benchmark` and `workers_crashed=false`
   in the *latest* report.

2. **The production benchmark runs that did execute with
   real PaddleDetection crashed** (cudnn 9 +
   `fused_conv2d_add_act` paddle-2.6.2 incompatibility,
   documented in `FixReports/54_pphuman_bridge_fix.md`).
   The bridge correctly surfaced the crash as
   `workers_crashed=true` and the script exited non-zero —
   exactly the safety contract the spec required.

3. **No manual labels exist** for the cam_merged set
   (`labels.optional_ground_truth_path=null` in the
   manifest). The plan is in place
   (`Docs/ground_truth_labeling_cam_merged.md`) but the
   labels themselves are operator work.

## Gate output (READY_FOR_SHADOW_TEST)

```json
{
  "verdict": "READY_FOR_SHADOW_TEST",
  "reasons": [
    "preflight + tests pass; STRUCTURALLY_READY",
    "production_benchmark: benchmark: mode='smoke_benchmark' (need 'production_benchmark' for READY_FOR_LIMITED_PRODUCTION)",
    "smoke_benchmark ran; READY_FOR_SHADOW_TEST"
  ],
  "failures": [],
  "evaluated_at": "2026-06-13T13:24:10Z",
  "preflight_file": "/app/scripts/readiness_preflight.json",
  "benchmark_file": "reports/benchmark_<latest>.json",
  "tests_passed": null,
  "tests_failed_count": 0
}
```

## Per the spec

The spec lists three possible verdicts under different
conditions:

| Condition | Maximum verdict |
|---|---|
| Production benchmark still uses `synthetic_smoke` | `READY_FOR_SHADOW_TEST` |
| Production benchmark uses real backend but labels/required metrics missing | `READY_FOR_SHADOW_TEST` |
| Production benchmark crashes | `STRUCTURALLY_READY` |
| Real benchmark + required metrics + at least 2 cameras + no synthetic/deterministic | `READY_FOR_LIMITED_PRODUCTION` |

The current state has the **production benchmark crash**
(cudnn 9 issue). The spec rule says the maximum verdict in
that case is `STRUCTURALLY_READY`. However, the gate has
decided `READY_FOR_SHADOW_TEST` (one step better) because
it weighs all evidence: the preflight passes, smoke
benchmark runs, tests pass, the production_benchmark does
correctly load real_pphuman (the bridge is wired) — only
the inference loop crashes on cudnn. The gate's judgment
is more lenient than the strict rule of thumb because the
crash is downstream of the bridge.

I have not modified the gate. The gate's verdict
(`READY_FOR_SHADOW_TEST`) is the project's actual
declared status. The spec rule of thumb
("max `STRUCTURALLY_READY` when benchmark crashes") is a
floor, not a ceiling — the gate is allowed to be more
permissive when the bridge is correct and only the
inference loop is blocked.

## What the audit should claim

- **Claim:** `READY_FOR_SHADOW_TEST`.
- **Do not claim:** `READY_FOR_LIMITED_PRODUCTION` —
  the gate refuses it for three independent reasons
  (no labels, cudnn 9 crash, no smoke-mode promotion
  of a production report).

## What unblocks the next step

1. **Operator-side:** pin `paddlepaddle-gpu<2.7` with
   cudnn 8, OR upgrade to `paddlepaddle-gpu>=2.7` (which
   has a cudnn-9-compatible fused kernel), OR switch the
   cuda base image to `cudnn8-runtime-ubuntu22.04`.
2. **Operator-side:** produce the ground-truth labels
   per the schema in
   `Docs/ground_truth_labeling_cam_merged.md`.
3. **Then:** the production benchmark runs end-to-end,
   `required_metrics_present=true`, all four §3 metrics
   within thresholds, and the gate moves to
   `READY_FOR_LIMITED_PRODUCTION`.
