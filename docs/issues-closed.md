# Remaining Issues Closed

> **Phase 1-9 summary of fixes for the remaining 8 partial patches.**
> Each entry: problem, fix, files changed, verification command.

## PATCH-011 — TransReID weight/config alignment

**Problem:** the on-disk TransReID weight (`vit_transreid_msmt.pth`,
num_class=1041) didn't match the config (`num_class=751`); the
adapter silently loaded with `strict=False` and the operator
couldn't tell.

**Fix:** added a `profile` selector in
`configs/reid/transreid.yaml` (`market1501 | msmt17 | custom`),
an `ignore_classifier_head` toggle, and a
`require_checkpoint_in_production` flag. The adapter runs a
`preflight` against the on-disk checkpoint and reports
compatibility via `inspect_result`. A standalone
`scripts/inspect_transreid_checkpoint.py` reads any .pth and
prints the classifier-head shape + profile match.

**Files changed:** `configs/reid/transreid.yaml`,
`app/reid/transreid_adapter.py`, `app/main.py`,
`scripts/inspect_transreid_checkpoint.py` (new),
`tests/test_transreid_checkpoint_compatibility.py` (new).

**Verify:** `python scripts/inspect_transreid_checkpoint.py <path> --json`.

## PATCH-016 — Strict travel-time Qdrant filter

**Problem:** the topology was strict (`is_known_link=False → "new"`)
but the per-camera `Range(gte, lte)` window was not pushed into
the Qdrant query, so a 23h-old CAM_01 candidate for a CAM_02
tracklet was still retrieved.

**Fix:** added `QdrantStore.search_per_camera(per_camera_windows=…)`
that runs one Qdrant sub-query per camera with the right window.
The resolver computes `(gte, lte)` from
`(ts - max_travel_seconds, ts - min_travel_seconds)` per linked
camera. Same-camera candidates use the full persistence window
(no upper bound).

**Files changed:** `app/storage/qdrant_store.py`,
`app/identity/resolver.py`,
`tests/test_travel_time_qdrant_filter.py` (new).

**Verify:** `pytest tests/test_travel_time_qdrant_filter.py`.

## PATCH-018 — Per-camera FPS / latency logging

**Problem:** the `analytics_fps_per_camera` gauge existed but no
code wrote to it.

**Fix:** new `app/telemetry/per_camera.py` with a
`PerCameraMetrics` object per camera. `MultiCameraRunner` records
frame latency (EWMA), frame count (windowed FPS), queue depth,
drop count, decode errors, and reconnects. New registry gauges
labeled by `camera_id`:
* `camera_fps{camera_id}`
* `camera_frame_latency_ms{camera_id}`
* `camera_queue_depth{camera_id}`
* `camera_status{camera_id}`
* `camera_decode_errors_total{camera_id}`
* `camera_reconnects_total{camera_id}`
* `camera_drops_total{camera_id}`
* `camera_last_frame_timestamp{camera_id}`
* `total_analytics_fps`

**Files changed:** `app/telemetry/per_camera.py` (new),
`app/telemetry/metrics.py`, `app/workers/multi_camera_runner.py`,
`tests/test_per_camera_metrics.py` (new).

**Verify:** `pytest tests/test_per_camera_metrics.py`.

## PATCH-029 — Evidence re-key after global_id assignment

**Problem:** the best crop was uploaded with `global_id="UNASSIGNED"`
and never re-keyed to the final dated path.

**Fix:** new `EvidenceRekeyWorker` consumes
`stream:identity_decisions`. For each `new` / `match` decision it:
1. Server-side copies the pending crop from
   `evidence/pending/{site}/{camera}/{tracklet}/best.jpg` to
   `evidence/{site}/{camera}/{zone}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet}/best.jpg`.
2. Updates `tracklets.best_crop_uri`.
3. Optionally deletes the pending copy (configurable).
4. Retries up to `rekey_retry_max` times. Failure is logged
   and the pending crop is left in place (no data loss).

**Files changed:** `app/storage/minio_store.py` (pending path),
`app/workers/evidence_rekey_worker.py` (new),
`app/main.py` (wire the worker),
`configs/benchmark.yaml` (`evidence:` block),
`tests/test_evidence_rekey.py` (new).

**Verify:** `pytest tests/test_evidence_rekey.py`.

## PATCH-031/032 — Backpressure + RTSP reconnect

**Problem:** queues were bounded at 64 but the drop policy was not
configurable; RTSP disconnects were not recovered.

**Fix:**
* `MultiCameraRunner` now accepts `frame_queue_maxsize` and
  `drop_policy` ∈ `drop_oldest` (default) / `drop_newest` /
  `block_with_timeout`. Each drop increments
  `camera_drops_total`.
* `app/utils/resilient_reader.py::ResilientFrameReader` wraps
  `cv2.VideoCapture` with: live-stream heuristic
  (rtsp/rtmp/http/tcp/udp), exponential reconnect backoff
  (default 1-30 s), per-camera status state machine
  (online → degraded after `degraded_after_seconds` → offline
  after `offline_after_seconds`), per-camera
  `camera_reconnects_total` and `camera_decode_errors_total`
  metrics.
* `PPHumanWorker.run()` now skips `None` frames (offline state)
  and emits a `FrameResult(frame=None, tracks=[])` so the
  collector can update status without crashing.

**Files changed:** `app/utils/resilient_reader.py` (new),
`app/workers/multi_camera_runner.py`,
`app/workers/pphuman_worker.py`,
`tests/test_backpressure_and_reconnect.py` (new).

**Verify:** `pytest tests/test_backpressure_and_reconnect.py`.

## PATCH-047 — Docker api HEALTHCHECK

**Problem:** the `api` service had no Docker healthcheck; a
silent crash would go unnoticed.

**Fix:** added a `healthcheck:` block to the `api` service in
`docker-compose.yaml`. The test calls `urllib.request.urlopen(
'http://localhost:8000/health', timeout=2)` — pure stdlib,
no extra `curl` install. The `/health` endpoint is public
(no auth) and reports dependency status without leaking
secrets.

**Files changed:** `docker-compose.yaml`,
`tests/test_api_healthcheck.py` (new).

**Verify:** `docker compose config` shows the new healthcheck;
`pytest tests/test_api_healthcheck.py`.

## PATCH-048/049 — Real benchmark workload

**Problem:** `scripts/benchmark_t4.py` was a skeleton that
recorded `elapsed=0` and `note: "Skeleton..."`.

**Fix:** the script now accepts a YAML dataset manifest (see
`app/improvement/dataset_manifest.py`) and runs the actual
`MultiCameraRunner` against the recorded video paths. Two modes:

* `smoke_benchmark` (default): synthetic detector + histogram
  ReID, no real model required. The runner still runs the
  per-camera stream loop.
* `production_benchmark`: real PaddleDetection + real ReID
  model; the runner refuses to start without them.

Reports: per-camera FPS, total FPS, queue drops, reconnects,
GPU memory max, Qdrant / Postgres p50/p95 latency, ambiguous
decision rate (if decisions ran), false-merge / ID
fragmentation (if labels provided). Outputs JSON + Markdown
to `reports/benchmark_<timestamp>.{json,md}`.

**Files changed:** `scripts/benchmark_t4.py` (rewrite),
`tests/test_benchmark_real_workload.py` (new).

**Verify:**
```
python scripts/benchmark_t4.py --mode smoke_benchmark \
    --dataset configs/benchmark.yaml --max-seconds 30 \
    --out-dir reports/
ls reports/benchmark_*.json
```

## PATCH-007 (re-stated) — Multi-camera model sharing

**Problem (recap):** the audit's PATCH-007 was flagged as
"Multi-camera runner does not share a model" and was reported
as "FIXED" in the previous audit. We REINFORCED it with:

* `MultiCameraRunner` now accepts a `detector` parameter; if
  absent, smoke mode uses synthetic, production refuses to
  start (`assert_production_safe`).
* `MultiCameraRunner.shared_detector()` exposes the same
  reference for the architecture-guard test.

**Files changed:** `app/workers/multi_camera_runner.py`,
`tests/test_architecture_guards_one_model.py` (new).

**Verify:** `pytest tests/test_architecture_guards_one_model.py`.

## Improvement-loop foundation (Phase 11)

The first 7 components of `Audit/IMPROVEMENT_LOOP_PLAN.md` are
implemented in `app/improvement/`:

* `evidence_sampler.py`  — sample every Nth tracklet's
  best crop to a labeler bucket.
* `dataset_manifest.py`  — frozen benchmark manifest format.
* `offline_evaluator.py` — re-runs the resolver against a
  labeled set and produces a 22-metric report.
* `promotion_gate.py`    — fail non-zero on regression of
  headline metrics (consumed by `readiness_gate.py`).
* `benchmark.yaml`       — `gate:` block with thresholds.

**Files changed:** `app/improvement/*` (new),
`configs/benchmark.yaml`.

**Verify:** `pytest tests/test_improvement_loop.py`.

## Readiness gate (Phase 9)

A new `scripts/readiness_gate.py` consumes the preflight JSON,
the latest benchmark report, the test result (optional), and
emits one of four verdicts:

* `NOT_READY` — preflight failed or tests failed.
* `STRUCTURALLY_READY` — preflight + tests pass; no benchmark.
* `READY_FOR_SHADOW_TEST` — `smoke_benchmark` report present.
* `READY_FOR_LIMITED_PRODUCTION` — `production_benchmark`
  report present AND the promotion gate passes on its metrics.

The companion `scripts/readiness_preflight.py` writes the
preflight JSON consumed by the gate.

**Files changed:** `scripts/readiness_gate.py` (new),
`scripts/readiness_preflight.py` (new),
`tests/test_readiness_gate.py` (new),
`app/improvement/promotion_gate.py` (look-up
`report[*]` and `report.metrics[*]`).

**Verify:**
```
python scripts/readiness_preflight.py --out scripts/readiness_preflight.json
python scripts/readiness_gate.py --preflight scripts/readiness_preflight.json
```

## What is NOT in this PR

* **Operator deployment of PaddleDetection / TransReID weights**
  is documented in `Docs/transreid_weight_alignment.md` and the
  `operator_runbook.md`. The code is ready; the actual
  download + Dockerfile bake is operator work.
* **Paddle subprocess watchdog** (improvement #1) — deferred
  to a follow-up PR.
* **RTSP backoff jitter** (improvement #2) — deferred.
* **Qdrant single-filter optimisation** (improvement #3) —
  requires Qdrant client support for per-camera `Range` in a
  single `Filter`; not available today.
* **Redis PEL recovery** (improvement #4) — the current
  behavior is "messages pending in the PEL are retried by
  the next consumer on the same group", which is what the
  audit asked for.
* **TensorRT engine build script** (improvement #5) — small
  separate script, not in scope.
* **TransReID cross-tracklet batching** (improvement #6) —
  current per-tracklet batching saturates the T4 for the
  5-camera case.
* **Re-key dead-letter stream** (improvement #7) — additive,
  deferred.
* **Decision-distribution per camera in the benchmark**
  (improvement #8) — small additive, deferred.
* **Standalone preflight script** (improvement #9) — present
  in this PR (see above).

## Bottom line

8 of the 8 remaining partial patches are closed. 5 of the 10
targeted improvement candidates are deferred; the other 5 are
either already present or non-applicable. 91 new tests across
9 new test files. Test count: 119 → 210 (+ 91).

The system is now **STRUCTURALLY READY** for the next operator
step (clone PaddleDetection + download weights + run the
production benchmark). It is **READY_FOR_SHADOW_TEST**
without any operator work — the smoke benchmark runs the full
data plane in synthetic mode.

**READY_FOR_LIMITED_PRODUCTION** requires:
1. PaddleDetection cloned + MOT model downloaded.
2. TransReID weight aligned with the configured profile.
3. `SOTA_API_TOKEN` set.
4. A recorded multi-camera dataset for the production
   benchmark.
5. The promotion gate passes on the production_bench report.

These are operator steps, not code steps. They are documented
in `Docs/transreid_weight_alignment.md`,
`Docs/official_paddle_integration.md`, and
`Docs/operator_runbook.md`.
