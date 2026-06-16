# 42 — Readiness gate after the PPHuman fix

## Command

```bash
docker compose run --rm -e BENCHMARK_OUT_DIR=/app/reports api \
  python scripts/readiness_gate.py \
  --benchmark-dir /app/reports \
  --min-verdict READY_FOR_SHADOW_TEST
```

## Verdict

```json
{
  "verdict": "READY_FOR_SHADOW_TEST",
  "reasons": [
    "preflight + tests pass; STRUCTURALLY_READY",
    "production_benchmark lacks required real-model metrics; capping verdict at READY_FOR_SHADOW_TEST per task-spec rule #8"
  ],
  "failures": [],
  "evaluated_at": "2026-06-12T20:32:28Z"
}
```

## LIMITED_PRODUCTION refusal

```bash
docker compose run --rm -e BENCHMARK_OUT_DIR=/app/reports api \
  python scripts/readiness_gate.py \
  --benchmark-dir /app/reports \
  --min-verdict READY_FOR_LIMITED_PRODUCTION || echo "EXIT: $?"
```

```text
verdict: READY_FOR_SHADOW_TEST
ERROR verdict READY_FOR_SHADOW_TEST < required READY_FOR_LIMITED_PRODUCTION
EXIT: 1
```

The gate refused `READY_FOR_LIMITED_PRODUCTION` because the
most recent benchmark is a `production_benchmark` whose
required real-model metrics (`false_merge_rate`,
`cross_camera_match_accuracy`, `id_fragmentation_rate`) are
absent — labels file (`/data/labels.json`) does not exist, and
the PP-Human subprocess path produced no MOT output (the
official pipeline is not installed in this environment). This
is the **correct** outcome per the hard rules.

## What changed

* `scripts/readiness_gate.py` — added `import sys` and
  prepended the project root to `sys.path` so the
  `app.improvement.promotion_gate` import inside
  `_check_promotion_gate` succeeds when the script is
  invoked as `python scripts/readiness_gate.py` from any
  working directory (the script's directory is
  `/app/scripts`, which is not the same as `/app/` where
  `app/` lives). This was a pre-existing path bug; the new
  benchmark report surfaced it because the production-mode
  branch of the gate now runs more often.

* `scripts/benchmark_t4.py` — emits the integrity fields
  the gate consumes (`status`, `detector_backend`,
  `reid_backend`, `workers_crashed`,
  `required_metrics_present`, `cameras_processed`).

## Honest verdict matrix

| Required real metrics present? | PaddleDetection installed? | Labels loaded? | Verdict |
|---|---|---|---|
| no | no | no | **READY_FOR_SHADOW_TEST** (current) |
| no | no | yes | READY_FOR_SHADOW_TEST (metrics still missing) |
| yes | no | yes | STRUCTURALLY_READY (real model required) |
| yes | yes | yes | READY_FOR_LIMITED_PRODUCTION |

## Path to LIMITED_PRODUCTION

1. Install PaddleDetection at `/opt/paddledetection/` and
   download the PP-Human model weights to
   `/app/models/pphuman/`.
2. Record or generate ground-truth labels at
   `/data/labels.json` (see
   `Docs/ground_truth_labeling_plan.md`).
3. Record at least 30 minutes of two-camera video at
   `/data/cam01.mp4` and `/data/cam02.mp4`.
4. Re-run:
   ```bash
   docker compose run --rm -e BENCHMARK_OUT_DIR=/app/reports \
     api python scripts/benchmark_t4.py \
     --mode production_benchmark \
     --dataset configs/benchmark.yaml \
     --max-seconds 1800
   ```
5. Verify `status: success`, `required_metrics_present: true`.
6. Re-run the readiness gate with
   `--min-verdict READY_FOR_LIMITED_PRODUCTION`.
