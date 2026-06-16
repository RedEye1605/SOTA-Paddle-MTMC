# FixReport 48 — Production benchmark on cam_merged

**Date**: 2026-06-13
**Scope**: short production benchmark on the
``configs/benchmark_real_cam_merged.yaml`` manifest, end-to-end.

---

## 1. Dataset manifest

New file: `configs/benchmark_real_cam_merged.yaml`. The two cameras
are wired to the multi-hour videos copied in Phase 2:

```yaml
dataset:
  name: cam_merged_validation
  site_id: yamaha_showroom
  cameras:
    - camera_id: CAM_01
      video_path: data/cam1_merged.mp4
    - camera_id: CAM_02
      video_path: data/cam2_merged.mp4
  labels:
    optional_ground_truth_path: null   # no labels yet
```

The `optional_ground_truth_path: null` is deliberate: no manual
labels exist for the cam_merged set, so the readiness gate will
**cap the verdict at READY_FOR_SHADOW_TEST** regardless of
detector/ReID quality on this benchmark.

## 2. Production benchmark attempt

Command:

```bash
uv run python scripts/benchmark_t4.py \
  --mode production_benchmark \
  --dataset configs/benchmark_real_cam_merged.yaml \
  --max-seconds 60
```

Result (verbatim, from the run log):

```text
[benchmark_t4] Benchmark mode=production_benchmark cameras=2
[benchmark_t4] Sources: ['CAM_01', 'CAM_02']
[app.runtime_mode] PRODUCTION REFUSED: PPHumanDetectorAdapter
  attempted "missing PaddleDetection pipeline at
  '/opt/paddledetection/deploy/pipeline/pipeline.py' (clone
  PaddleDetection there or set PPHUMAN_PIPELINE_PATH /
  PPHUMAN_INFER_CONFIG)" but runtime mode is 'production' which
  disallows synthetic / deterministic paths. Set --mode smoke_test
  or ALLOW_SYNTHETIC_SMOKE_TEST=true to allow this in dev/CI only.
[benchmark_t4] production_benchmark: failed to load
  PPHumanDetectorAdapter: PRODUCTION REFUSED: ...
```

**The production safety gate correctly refused to start.** The
`runtime_mode` module blocks any attempt to use the synthetic
detector when `RuntimeMode.PRODUCTION` is in effect. This is the
expected and required behaviour per the hard rules in the task
spec:

> 4. Do not weaken RuntimeMode safety gates.
> 5. Production must refuse synthetic detector and deterministic
>    ReID.

The dev environment does not have the PaddleDetection pipeline
checked out at `/opt/paddledetection`; the gate blocks the
production run with a clear, actionable error message.

## 3. Smoke benchmark fallback (allowed for dev/CI)

To get a baseline smoke report on the cam_merged videos, the
operator may opt in to the explicit smoke path:

```bash
uv run python scripts/benchmark_t4.py \
  --mode smoke_benchmark \
  --dataset configs/benchmark_real_cam_merged.yaml \
  --max-seconds 30
```

This run is *explicitly* a smoke run, the JSON + Markdown reports
label every field accordingly, and the readiness gate will not
promote to `READY_FOR_LIMITED_PRODUCTION` based on its numbers.

Latest report (`reports/benchmark_20260613T084533Z.json`):

```jsonc
{
  "mode": "smoke_benchmark",
  "dataset_name": "cam_merged_validation",
  "site_id": "yamaha_showroom",
  "duration_seconds": 30.01,
  "cameras": ["CAM_01", "CAM_02"],
  "cameras_processed": ["CAM_01", "CAM_02"],
  "detector_backend": "synthetic_smoke",
  "reid_backend": "smoke_deterministic",
  "workers_crashed": false,
  "crashed_cameras": [],
  "labels_path": null,
  "labels_loaded": false,
  "required_metrics_present": false,
  "status": "success",
  "total_analytics_fps": 198.98,
  "per_camera_analytics_fps": {
    "CAM_01_fps": 99.47,
    "CAM_02_fps": 99.51
  },
  "gpu_memory_used_mb_max": 224.0,
  "queue_drops_total": 3077,
  "camera_reconnects_total": 0
}
```

## 4. Required-metrics check

| Field | Value | Required for READY_FOR_LIMITED_PRODUCTION? |
|---|---|---|
| `detector_backend` | `synthetic_smoke` | ❌ must be real (production) |
| `reid_backend` | `smoke_deterministic` | ❌ must be real (production) |
| `workers_crashed` | `false` | ✅ required |
| `cameras_processed` | 2 / 2 | ✅ ≥ 2 cameras required |
| `required_metrics_present` | `false` | ❌ labels absent |
| `status` | `success` | ✅ required |

The current run satisfies 2 of 4 hard requirements. The other
two (real detector, real ReID) require the PaddleDetection stack
in the dev container, which the operator must provision. The
required_metrics_present field is gated on the existence of a
`labels.json` for the cam_merged set, which is not yet available.

## 5. Verdict

- **Production benchmark run**: correctly refused by the
  `runtime_mode` safety gate (PaddleDetection pipeline not
  installed). No report generated for the production path because
  no production worker ever started.
- **Smoke benchmark run**: completed successfully, both cameras
  processed at ~99.5 fps each, no workers crashed, but the
  detector/ReID backends are explicit-synthetic and required
  metrics are absent.
- **Status**: capped at **READY_FOR_SHADOW_TEST** until the
  operator provisions the real PaddleDetection stack and produces
  ground-truth labels for the cam_merged dataset.
- **To promote to READY_FOR_LIMITED_PRODUCTION** the operator
  must:
  1. install PaddleDetection (`git clone
     https://github.com/PaddlePaddle/PaddleDetection` at
     `/opt/paddledetection` or set `PPHUMAN_PIPELINE_PATH`),
  2. re-run the production benchmark and confirm
     `detector_backend` is real (e.g. `paddledetection_pphuman`),
  3. produce a labels file for the cam_merged videos and rerun
     so `required_metrics_present` becomes `true`.

The safety gate worked exactly as designed: it prevented the
project from being marked production-ready without the required
infrastructure.
