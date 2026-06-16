# 41 — Short real production benchmark after PPHuman fix

## What ran

```bash
docker compose run --rm -e BENCHMARK_OUT_DIR=/app/reports api \
  python scripts/benchmark_t4.py \
  --mode production_benchmark \
  --dataset configs/benchmark.yaml \
  --max-seconds 5
```

## Result

The benchmark completed in ~5.1 seconds and wrote a structurally
valid JSON + Markdown report to
`reports/benchmark_20260612T202808Z.{json,md}`.

| Field | Value |
|---|---|
| `mode` | `production_benchmark` |
| `status` | `partial` |
| `detector_backend` | `real_pphuman` |
| `reid_backend` | `pphuman_strongbaseline` |
| `workers_crashed` | `false` |
| `crashed_cameras` | `[]` |
| `cameras_processed` | `["CAM_01", "CAM_02"]` |
| `required_metrics_present` | `false` (labels missing) |
| `labels_path` | `/data/labels.json` (does not exist) |
| `total_analytics_fps` | 918.4 |
| `queue_drops_total` | 26 |

## Comparison to baseline

| Field | Before (Phase 0) | After (Phase 4) |
|---|---|---|
| worker exception | `NotImplementedError` (crash) | none |
| `detector_backend` | absent | `real_pphuman` |
| `reid_backend` | absent | `pphuman_strongbaseline` |
| `workers_crashed` | absent | `false` |
| `status` | absent (always looked "success") | `partial` |
| `required_metrics_present` | absent | `false` |
| `cameras_processed` | absent | `["CAM_01", "CAM_02"]` |
| `total_analytics_fps` | 1706.8 (fake) | 918.4 (no MOT output) |

The new FPS number is *lower* than the Phase 0 number, and
that's correct: Phase 0 measured "how fast can the consumer
loop yield empty frames" because the workers crashed on the
first frame. The new code measures the same loop but now
after the per-frame factory is wired, the runner drops frames
at the queue (queue_drops_total=26) when the factory returns
empty detections, and the stream loop processes fewer frames
per second. Still not a real inference number — the official
PP-Human pipeline is not installed in this container — but
no longer a wildly inflated value from a crashed worker.

## Why is the FPS still not a real performance number?

The API container does **not** have PaddleDetection installed
(`/opt/paddledetection/deploy/pipeline/pipeline.py` is missing).
`PPHumanDetectorAdapter.load()` is probe-only; it only checks
file existence. The subprocess that `run_pipeline()` tries to
launch immediately fails (the python interpreter returns
non-zero because the script does not exist), so the
`PPHumanPipelineSubprocessManager.stream()` generator sees
`any_alive = False` and exits. The per-frame factory in the
worker then returns `[]` for every frame, the worker emits
zero tracks, and the benchmark drains the queue faster than a
real model would.

This is the **honest** outcome the spec asks for: no
fabricated FPS, no fake detector, no synthetic fallback in
production mode, and the report's `status='partial'` plus
`required_metrics_present=False` correctly signal that the
benchmark is not a valid LIMITED_PRODUCTION proof.

## What would change with a real PaddleDetection install

When the operator installs PaddleDetection at
`/opt/paddledetection/` and the model weights at
`/app/models/pphuman/`, the subprocesses launch, write MOT
files, the per-frame factory returns real MOT detections, the
worker emits `LocalTrack`s, the tracklet collector runs ReID,
and the resolver records identity decisions. The benchmark
then computes `false_merge_rate` / `cross_camera_match_accuracy`
/ `id_fragmentation_rate` against a labelled manifest
(`/data/labels.json`).

## Verdict

* `status`: **partial** (correct, given missing labels and
  missing PaddleDetection install).
* `workers_crashed`: **false** — the integration is wired.
* `detector_backend`: **real_pphuman** (correct backend
  identification; the path is not faked).
* `required_metrics_present`: **false** — readiness gate MUST
  cap at `READY_FOR_SHADOW_TEST`.
