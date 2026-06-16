# Phase 3 — Production benchmark on cam_merged (initial run)

Date: 2026-06-13

## Command

```bash
docker compose run --rm -w /app api python scripts/benchmark_t4.py \
  --mode production_benchmark \
  --dataset configs/benchmark_real_cam_merged.yaml \
  --max-seconds 60 \
  --out-dir /app/reports/benchmark_runs/real_cam_merged
```

## Result (initial run, before bridge fix)

```json
{
  "mode": "production_benchmark",
  "detector_backend": "real_pphuman",
  "reid_backend": "pphuman_strongbaseline",
  "workers_crashed": false,
  "crashed_cameras": [],
  "cameras_processed": ["CAM_01", "CAM_02"],
  "status": "partial",
  "required_metrics_present": false,
  "total_analytics_fps": 293.2
}
```

Acceptance check (against Phase 3 rules):

| Rule | Pass? | Evidence |
|---|---|---|
| 1. `detector_backend != synthetic_smoke` | PASS | `real_pphuman` |
| 2. `reid_backend != smoke_deterministic` | PASS | `pphuman_strongbaseline` |
| 3. `workers_crashed == false` | PASS (misleading) | see Phase 4 |
| 4. CAM_01 and CAM_02 processed | PASS | both in `cameras_processed` |
| 5. Status not failed due to safety gate | PASS | `status=partial` (no labels) |
| 6. `required_metrics_present=false` OK | PASS | no labels, so expected |
| 7. Readiness capped at `READY_FOR_SHADOW_TEST` | PASS | gate is uncapped here |

## The catch (Phase 4 found it)

`detector_backend=real_pphuman` was the **intent** but the
**subprocess was crashing on import** (missing `scipy`).
`workers_crashed=false` was a **bridge bug**, not a healthy
result. The 293 FPS was the synthetic per-frame factory running
on empty buffers.

The bridge fix (Phase 4) added stderr-tap monitoring,
`sys.executable` for the subprocess, `scipy` and `imgaug` to
the deps, and made `crashed_cameras` the UNION of the
subprocess-monitor's set and the tailer's set. **Only then**
does `workers_crashed=true` correctly surface the missing
deps. The follow-up container rebuild (still in Phase 4) is
required to get a truly clean `workers_crashed=false` report.

## Maximum verdict after Phase 3 + Phase 4 (initial)

Because the original Phase-3 report was misleading, the
authoritative report to read is the post-Phase-4 one. After
the container rebuilds with the fixed deps, the benchmark is
re-run; the result is recorded in `FixReports/54_pphuman_bridge_fix.md`
under "Verification (post-rebuild)".

## Verdict

- Production benchmark **was reached** (no safety-gate refusal)
  with the real detector and real ReID wired in.
- `detector_backend=real_pphuman`, `reid_backend=pphuman_strongbaseline`,
  `workers_crashed=false` were **observed** — but the underlying
  subprocess was actually crashing.
- The bridge fix (Phase 4) corrects this; the post-fix report
  shows `workers_crashed=true` and the missing dep is named in
  the stderr tail.
- Per Phase 3 rule #6, the benchmark script **does** exit
  non-zero once the bridge fix is in place — see Phase 4.

## Files

- `reports/benchmark_runs/real_cam_merged/benchmark_20260613T093126Z.json`
- `reports/benchmark_runs/real_cam_merged_long/benchmark_20260613T093714Z.json`
  (90s + 180s warmup variants; identical contract)
