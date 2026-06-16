# Real Benchmark Runbook

> **Operator runbook for PATCH-048/049.**
> How to run `scripts/benchmark_t4.py` against a real recorded
> multi-camera dataset.

## TL;DR

```bash
# 1. Smoke benchmark (no real model needed; works in CI)
python scripts/benchmark_t4.py \
    --mode smoke_benchmark \
    --dataset configs/benchmark.yaml \
    --max-seconds 30 \
    --out-dir reports/

# 2. Production benchmark (requires Paddle + ReID weights)
python scripts/benchmark_t4.py \
    --mode production_benchmark \
    --dataset configs/benchmark.yaml \
    --max-seconds 300 \
    --out-dir reports/

# 3. Readiness gate
python scripts/readiness_preflight.py --out scripts/readiness_preflight.json
python scripts/readiness_gate.py \
    --preflight scripts/readiness_preflight.json \
    --out reports/readiness.json
```

## Dataset manifest

The manifest is a YAML file with a `dataset:` key. See
`configs/benchmark.yaml` for the live example.

```yaml
dataset:
  name: yamaha_showroom_day1
  site_id: yamaha demo
  timezone: Asia/Jakarta
  cameras:
    - camera_id: CAM_01
      video_path: /data/cam01.mp4
    - camera_id: CAM_02
      video_path: /data/cam02.mp4
  labels:
    optional_ground_truth_path: /data/labels.json
```

* `cameras[].video_path` is a local file path or an
  `rtsp://…` URL. RTSP streams are reconnected automatically
  via `ResilientFrameReader`.
* `labels.optional_ground_truth_path` is optional. When
  present, the benchmark report includes a `labels_loaded`
  count and a note that the operator should cross-reference
  with the `identity_decisions` table to compute
  `false_merge_rate` / `id_fragmentation_rate`.

## Reports

Each run writes two files to `--out-dir`:

* `benchmark_<timestamp>.json` — the structured report.
* `benchmark_<timestamp>.md` — the human-readable summary.

Fields:

| Key | Description |
|---|---|
| `mode` | `smoke_benchmark` or `production_benchmark` |
| `started_at` | UTC timestamp of the run |
| `dataset_name` | the manifest's `name` |
| `site_id` | the manifest's `site_id` |
| `duration_seconds` | wall-clock duration |
| `cameras` | list of camera_ids that ran |
| `per_camera_analytics_fps` | `{cam_id: fps}` |
| `total_analytics_fps` | sum across cameras |
| `queue_drops_total` | backpressure drops (PATCH-031) |
| `camera_reconnects_total` | reconnects (PATCH-032) |
| `gpu_memory_used_mb_max` | GPU memory peak |
| `cpu_usage_percent_avg` | CPU peak (if `psutil` available) |
| `qdrant_query_latency_p50_ms` | p50 of `qdrant_query_latency_seconds` |
| `qdrant_query_latency_p95_ms` | p95 |
| `postgres_write_latency_p50_ms` | p50 |
| `postgres_write_latency_p95_ms` | p95 |
| `labels_loaded` | count of labels (if path provided) |

## Readiness gate integration

The promotion gate is invoked by `readiness_gate.py` for
production_benchmark reports. The gate thresholds are in
`configs/benchmark.yaml`:

```yaml
gate:
  false_merge_rate_max: 0.05
  cross_camera_match_accuracy_min: 0.85
  fps_min: 5.0
  gpu_memory_used_mb_max: 12000
  qdrant_query_latency_p99_ms_max: 200
  postgres_write_latency_p99_ms_max: 50
  ambiguous_auto_merge_rate_max: 0.0
  id_fragmentation_rate_max: 0.20
```

The gate returns:
* `READY_FOR_LIMITED_PRODUCTION` — all thresholds pass.
* `NOT_READY` — any threshold fails.

## Smoke vs production

* `smoke_benchmark` does NOT require the real Paddle +
  TransReID weights. The synthetic detector + histogram
  ReID are used. Use this in CI to verify the data plane
  is wired end-to-end.
* `production_benchmark` requires the real weights (the
  runner refuses to start without them). Use this on the
  T4 host with a recorded dataset.

## Common issues

* `rtsp://` stream not reachable: the reader enters
  offline state. The benchmark report's
  `per_camera_analytics_fps` will be 0 for that camera.
  The other cameras continue to run.
* Disk full: the `best_crop_uri` upload fails; the
  collector logs a warning and continues. The benchmark
  report itself is unaffected.
* `psutil` not installed: `cpu_usage_percent_avg` is `null`.
  The other metrics are still recorded.
