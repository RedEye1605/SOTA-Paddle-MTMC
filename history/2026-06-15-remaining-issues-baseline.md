# Phase 0 — Remaining-Issues Baseline Verification

> **Date:** 2026-06-12
> **Scope:** Audit SOTA-Paddle-MTMC against the 8 remaining partial / deferred patches
> (PATCH-011, 016, 018, 029, 031, 032, 047, 048/049) and prepare a hardening plan.

## 1. Current test result

```
119 passed, 1 skipped, 1 warning in 1.05 s
```

The 1 skipped test is `tests/test_transreid_vendor.py` (requires torch, not installed
on the dev host). `python3 -m compileall app scripts tests` is clean and
`docker compose config` is valid.

## 2. Current code evidence per remaining patch

### PATCH-011 — TransReID weight/config alignment

* `configs/reid/transreid.yaml` still says `weight: /models/transreid/transformer_120.pth`
  (Market-1501, num_class=751) but the file shipped on disk is
  `models/vit_transreid_msmt.pth` (MSMT17, num_class=1041).
* `app/reid/transreid_adapter.py:_try_load_real()` does `load_state_dict(strict=False)`
  and logs missing/unexpected keys but does not abort on a classifier mismatch.
* No `inspect_transreid_checkpoint` script exists.
* No profile selector in `configs/reid/transreid.yaml` (no `profile: market1501`).

**Already implemented:**
* `weights_only=True` enforced (security).
* `load_state_dict(strict=False)` so feature-extractor mode is allowed.

**Still missing:**
* Profile selector (market1501 / msmt17 / custom).
* Checkpoint inspector script.
* Classifier-head mismatch detection (with `ignore_classifier_head` toggle).
* Production preflight that checks checkpoint vs. config.

**Risk of change:** low — additive inspector + adapter config; no existing
production path is touched.

### PATCH-016 — Strict travel-time Qdrant filter

* `app/identity/resolver.py:_search_with_filters()` calls
  `self.qdrant.search(..., timestamp_gte=int(ts - persistence_window_seconds), ...)`
  — a single `Range(gte=...)` is applied. **No `lte` bound per camera.**
* `app/identity/camera_topology.py` exposes `min_travel_seconds` / `max_travel_seconds`
  but the resolver does not use them when building the Qdrant filter.
* `QdrantStore.search()` signature does not accept per-camera `lte` bounds.

**Already implemented:**
* Topology hard-block (`is_known_link=False → "new"`).
* Stage 1/2/3 staged candidate enumeration.
* `QdrantStore.search` enforces `timestamp_gte > 0` (PATCH-034).

**Still missing:**
* Per-camera `Range(gte=..., lte=...)` payload filter.
* `QdrantStore.search` does per-camera queries (or per-camera bounds in a single
  query — Qdrant supports multiple `FieldCondition`s on the same key with
  `should[]`).
* 24h fallback cannot auto-match unless `enable_stage3_24h_fallback=true` and
  threshold is `stage3_auto_match_threshold` (already implemented).

**Risk of change:** medium — touches the resolver hot path. The implementation
is per-camera sub-queries, which is exactly the Qdrant recommended pattern.

### PATCH-018 — Per-camera FPS / latency logging

* `app/telemetry/metrics.py:REGISTRY.analytics_fps` is a `Gauge("analytics_fps_per_camera")`
  — but it is never `.set()` anywhere in the code.
* `MultiCameraRunner.stream()` polls queues with `q.get(timeout=0.1)` but does
  NOT measure per-camera wall-clock FPS.
* `app/main.py:268-275` measures global GPU memory every 10 s, but no per-camera
  counters.

**Already implemented:**
* `analytics_fps_per_camera` gauge exists in registry.
* `gpu_memory_used` is set every 10 s.
* Qdrant / Postgres latency histograms are observed.

**Still missing:**
* Per-camera FPS counter (per `camera_id` label).
* Per-camera frame latency (frame → emit).
* Per-camera queue depth gauge.
* Per-camera decode-error / reconnect counters.

**Risk of change:** low — additive metric writes; no production logic touched.

### PATCH-029 — Evidence re-key after global_id assignment

* `app/storage/minio_store.py:evidence_key()` always uses `global_id="UNASSIGNED"`.
* `app/workers/tracklet_collector.py:emit_closed_tracklets()` writes
  `best_crop_uri` with `global_id="UNASSIGNED"`.
* The resolver writes the assigned `global_id` to `tracklets.global_id`
  (via `update_tracklet_global_id`) but does NOT re-key the MinIO object.

**Already implemented:**
* `best_crop_uri` column is updated in the `tracklets` table.
* `MinioStore.copy_object_within_bucket()` exists (server-side copy).
* `MinioStore.get_object_bytes()` exists.

**Still missing:**
* Re-key from `pending/{tracklet_id}/best.jpg` to
  `evidence/{site}/{cam}/{zone}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg`
  after the resolver assigns a global_id.
* Retry / dead-letter on re-key failure.
* Configurable `keep_pending_copy` / `rekey_retry_max`.

**Risk of change:** medium — touches the resolver hot path. We must NOT block
the resolver waiting for MinIO. Re-key must be done asynchronously (in a worker)
or fire-and-forget with logging.

### PATCH-031 / PATCH-032 — Backpressure and RTSP reconnect

* `app/workers/multi_camera_runner.py:_run_worker` does `q.put(result, timeout=0.5)`
  with a silent drop on `Full`. No metric, no policy selector.
* `make_frame_reader` does `cap = cv2.VideoCapture(source)` once and loops
  with `cap.read()`. **No reconnect on EOF for RTSP.** A dead camera is silent
  forever (only seen if the queue stops producing).
* `drop_policy` is not configurable.
* Camera status (`online` / `degraded` / `offline` / `recovered`) does not exist.

**Already implemented:**
* `Queue(maxsize=64)` is bounded (PATCH-007).
* `multi_camera_runner.stop()` sets `_stop_event` (PATCH-023).

**Still missing:**
* Configurable `drop_policy` (`drop_oldest` / `drop_newest` / `block_with_timeout`).
* Per-camera status state machine.
* Reconnect with exponential backoff for RTSP.
* Per-camera metrics (`camera_reconnects_total`, `camera_decode_errors_total`).
* `frame_queue_maxsize` per-stage (we have one Queue per stage today, but the
  `queues.frame_queue_maxsize` config is not read).

**Risk of change:** medium — touches the runner's threading model. The change
must keep the audit's "one model per process" rule and not break the
architecture-guard test.

### PATCH-047 — Docker `api` HEALTHCHECK

* `docker-compose.yaml:api` does not have a `healthcheck:` block.
* `/health` endpoint exists and is public (good).
* The Docker HEALTHCHECK is not wired.

**Already implemented:**
* `/health` returns `{"status": "ok", "postgres": "ok" | "timeout" | "down"}`.
* PG healthcheck is wrapped in `asyncio.wait_for(timeout=2)`.

**Still missing:**
* `healthcheck:` block in the `api` service.

**Risk of change:** low — single YAML edit + verify `/health` does not leak
secrets.

### PATCH-048 / PATCH-049 — `benchmark_t4.py` real workload

* `scripts/benchmark_t4.py` is a 5-scenario skeleton that records `elapsed=0`
  and writes `note: "Skeleton..."`. It does NOT actually run the runner.
* The improvement loop has `DatasetManifest`, `OfflineEvaluator`,
  `PromotionGate` — but `scripts/benchmark_t4.py` does not call any of them.

**Already implemented:**
* `app/improvement/dataset_manifest.py` (CameraClip, DatasetManifest).
* `app/improvement/offline_evaluator.py` (OfflineEvaluator, MetricBlock, OfflineReport).
* `app/improvement/promotion_gate.py` (PromotionGate, GateThresholds).
* `configs/benchmark.yaml` (retention + gate + identity thresholds).

**Still missing:**
* `benchmark_t4.py` accepts a dataset manifest + runs the actual
  `MultiCameraRunner` (in a subprocess if real model, in-process if smoke).
* `smoke_benchmark` vs. `production_benchmark` mode.
* JSON + Markdown report.
* Real metrics: FPS, GPU memory, Qdrant/Postgres p50/p95, Redis backlog,
  reconnect count, queue drops, ambiguous rate, false merge rate (if labels),
  id fragmentation (if labels).

**Risk of change:** low — additive; the existing skeleton is preserved.

## 3. What is already implemented (do not touch)

* `QdrantStore.search` always uses `Filter(must=[...])` (Qdrant docs compliance).
* Redis `r.setex` / `r.xadd` / `r.xgroup_create(mkstream=True)` / `r.xreadgroup`
  (Redis docs compliance).
* `psycopg_pool.ConnectionPool` (psycopg 3.2+ API).
* `MinioStore.copy_object_within_bucket` uses `CopySource(bucket, object)`.
* FastAPI `HTTPBearer` + `Depends(verify)` (FastAPI 0.115+ security tutorial).
* `paddle.inference.Config.enable_tensorrt_engine` and `precision_mode=inference.PrecisionType.Half`
  per the PaddleDetection deployment doc.
* `pp_ts_infer --run_mode=trt_fp16` is the documented T4 path.

## 4. Risk of each change (summary)

| Patch | Risk | Touched files | Reversible |
|---|---|---|:---:|
| PATCH-011 | LOW | `scripts/inspect_transreid_checkpoint.py` (new), `configs/reid/transreid.yaml`, `app/reid/transreid_adapter.py` | yes |
| PATCH-016 | MEDIUM | `app/storage/qdrant_store.py`, `app/identity/resolver.py` | yes |
| PATCH-018 | LOW | `app/workers/multi_camera_runner.py`, `app/telemetry/metrics.py` | yes |
| PATCH-029 | MEDIUM | `app/workers/reid_worker.py` or new `app/workers/evidence_rekey_worker.py`, `app/storage/minio_store.py` | yes |
| PATCH-031/32 | MEDIUM | `app/workers/multi_camera_runner.py`, `app/utils/backpressure.py` (new) | yes |
| PATCH-047 | LOW | `docker-compose.yaml` | yes |
| PATCH-048/49 | LOW | `scripts/benchmark_t4.py` (rewrite), `configs/benchmark.yaml` | yes |

## 5. Plan

Phase 1 → 7: implement each patch with focused tests.
Phase 8: targeted-improvement search (low-risk, non-rewrite).
Phase 9: readiness gate that consumes the new benchmark JSON.
Phase 10: docs.
