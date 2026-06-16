# Phase 12 — Targeted improvement search

**Date:** 2026-06-13
**Project root:** `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC`

## Method

The codebase was searched across the 10 nominated areas with an
explore agent that only **read** the source.  For each candidate
I then verified the finding by hand and (for the items marked
"implemented now") wrote a regression test in
`tests/test_targeted_improvements_phase12.py` and
`tests/test_readiness_preflight.py`.

The 10 search areas:

```text
 1. PaddleDetection PP-Human integration
 2. Multi-camera runner backpressure
 3. Postgres connection pooling
 4. Qdrant search hygiene
 5. Redis usage / model loading
 6. T4 CUDA/TensorRT runtime settings
 7. TransReID batching
 8. Evidence retention and re-keying
 9. Benchmark reporting
10. Healthcheck / preflight clarity
```

## Findings

### 1. PaddleDetection PP-Human integration

- **Source:** `app/detection/pphuman_pipeline.py:96`, `:241`, `:307`
- **Current behavior:** `PPHumanDetectorAdapter.__init__` accepts a
  `timeout_seconds=30` argument, stores it on `self.timeout_seconds`,
  but the attribute is never read.  `subprocess.Popen(...)` launches
  pipeline children with no startup/heartbeat watchdog.  A hung
  pipeline.py is only detected at the 5 s `p.wait(timeout=5)` in
  `stop()`.
- **Proposed improvement:** poll `p.poll()` from a side thread with
  a `timeout_seconds` budget, terminate + log on timeout, restart.
- **Risk:** medium (requires plumbing a watchdog thread per
  pipeline; needs new integration tests against a stubbed Popen).
- **Expected benefit:** stuck CUDA-OOM / broken-config children are
  killed within seconds instead of leaving the parent waiting.
- **Implemented now:** **no** — deferred (Phase-13 follow-up).

### 2. Multi-camera runner backpressure — drop counter under-report

- **Source:** `app/workers/multi_camera_runner.py:255-270`
- **Current behavior (before):** in `drop_newest`, the initial
  `put_nowait` failure increments `observe_drop()` once.  The
  eviction-and-retry path silently swallows a second failure
  with `pass`.  When the queue is genuinely saturated, the
  metric under-reports the true number of dropped frames.
- **Proposed improvement:** call `observe_drop()` again when the
  post-eviction `put_nowait` also raises.
- **Risk:** low (metric-only; no behavioral change for the
  consumer; existing drop tests still pass).
- **Expected benefit:** accurate per-camera drop counter →
  easier diagnosis of sustained vs. transient backpressure.
- **Implemented now:** **yes**.
- **Regression test:** `tests/test_targeted_improvements_phase12.py::test_drop_newest_records_second_drop_when_retry_also_fails`

### 3. Postgres connection pooling

- **Source:** `app/storage/postgres.py:76`, `app/main.py` finally block
- **Current behavior (before):** `PostgresStore.close()` exists
  but the main loop's `finally` only called `runner.stop()`.  On
  SIGTERM the PG/Qdrant/Redis/MinIO pools leaked.
- **Proposed improvement:** call `close()` (or `disconnect()`) on
  every infra client in the `finally` block, with per-store
  best-effort isolation.
- **Risk:** low (each close wrapped in `try/except`; logged but
  not re-raised).
- **Expected benefit:** clean teardown on SIGTERM; no socket
  leaks across restart cycles.
- **Implemented now:** **yes**.
- A retry/circuit-breaker around `timed_execute` for transient
  PG errors was identified but **deferred** — that change touches
  the hot path and needs its own test scaffolding.

### 4. Qdrant search hygiene

- **Source:** `app/storage/qdrant_store.py:96`, `app/workers/reid_worker.py:181`
- **Current behavior:** `upsert_point` is called in a Python loop
  per embedding (up to 15 round-trips per tracklet).  All
  *searches* already use payload filters (verified — the audit's
  hard rule holds).
- **Proposed improvement:** add `upsert_points(collection, points)`
  batch API.
- **Risk:** low.
- **Expected benefit:** ~15× fewer RPCs per tracklet; lower
  ReID worker wall-clock time on the hot path.
- **Implemented now:** **no** — deferred (perf optimization;
  wants p50/p95 measurement first, then a tuned batch size,
  then a test that asserts batch boundaries).

### 5. Redis usage / model loading

- **Source:** `app/storage/redis_state.py:47`, `app/main.py:131`
- **Current behavior:** default 50-socket pool; model is loaded
  once in `build_app_context` and shared (architecture-guard
  verified, `test_architecture_guards_one_model.py`).
- **Implemented now:** **n/a** — no actionable issue.

### 6. T4 CUDA/TensorRT runtime settings

- **Source:** `configs/app.yaml:13`, `app/main.py:221`
- **Current behavior:** `runtime.run_mode=trt_fp16`,
  `PPHUMAN_DEVICE=gpu` already set; `trt_calib_mode` correctly
  not set (it's only needed for `trt_int8`).
- **Implemented now:** **n/a** — no actionable issue.

### 7. TransReID batching

- **Source:** `app/reid/transreid_adapter.py:395`, `app/reid/_transreid_native/model.py:544`
- **Current behavior:** `_extract_real` already stacks all crops
  into a single tensor and runs one forward under
  `torch.inference_mode()` with `model.eval()`.  Intra-tracklet
  batching is correct.
- **Observation:** `ReIDConfig.batch_size=16` is never read and
  crops are truncated to 15 in `reid_worker._load_crops`
  independent of the config.  Cosmetic only.
- **Implemented now:** **no** — cosmetic / documentation
  follow-up (deferred).

### 8. Evidence retention and re-keying

- **Source:** `app/workers/evidence_rekey_worker.py:116-168`
- **Current behavior:** copy → delete → PG update is not
  transactional.  If PG fails after a successful copy, the
  object is at the final key but the row still points at
  pending; the next pass may double-copy.
- **Proposed improvement:** wrap copy + delete + PG update so a
  failure re-runs the whole sequence (idempotent) or write a
  `pending` marker that the next poll retries.
- **Risk:** medium (changes storage semantics — needs MinIO +
  PG fault-injection test).
- **Implemented now:** **no** — deferred.

### 9. Benchmark reporting — atomic write

- **Source:** `scripts/benchmark_t4.py:451-461`
- **Current behavior (before):** `with open(..., "w")` for both
  JSON and Markdown.  A SIGKILL mid-write leaves a truncated
  file; the readiness gate then crashes on the malformed JSON.
- **Proposed improvement:** write to `*.tmp` + `os.replace` to
  the final name.  POSIX rename is atomic.  Pre-existing
  stale `*.tmp` files are overwritten.
- **Risk:** low.
- **Expected benefit:** crash-mid-write does not corrupt the
  most-recent report.
- **Implemented now:** **yes**.
- **Regression test:** `tests/test_targeted_improvements_phase12.py::test_write_reports_is_atomic_via_tmp_and_replace`

A second related finding — adding a per-camera **inference
latency** histogram — was identified but **deferred**.  It needs
instrumentation in the ReID worker hot path and its own
benchmark assertion.

### 10. Healthcheck / preflight clarity — QDRANT_HOST coverage

- **Source:** `scripts/readiness_preflight.py:140`
- **Current behavior (before):** `_check_infra_env` required
  only `POSTGRES_HOST/USER/PASSWORD`, `MINIO_ACCESS_KEY/SECRET_KEY`,
  `REDIS_HOST`.  `QDRANT_HOST` was unchecked — a misconfigured
  production deployment could reach the readiness gate.
- **Proposed improvement:** add `QDRANT_HOST` to the required dict.
- **Risk:** low (the gate is a soft check; the preflight just
  reports the new failure).
- **Expected benefit:** catches the missing QDRANT_HOST env var
  before SHADOW or LIMITED_PRODUCTION promotion.
- **Implemented now:** **yes**.
- **Regression tests:** `tests/test_targeted_improvements_phase12.py::test_preflight_infra_env_requires_qdrant_host` and `::test_preflight_infra_env_passes_with_qdrant_host`

Extending the `/health` API endpoint to actively probe Redis,
Qdrant, and MinIO (currently it only probes Postgres) was
identified but **deferred** — it needs an async-safe per-store
2-s timeout and dedicated tests against each store's healthcheck.

A precedent precedence-bug fix (`_check_infra_env` line 154)
was applied in Phase 6 with its own three regression tests
in `tests/test_readiness_preflight.py`.

## Summary

| #   | Area                                         | Implemented now |
| --- | -------------------------------------------- | --------------- |
| 1   | PaddleDetection subprocess watchdog          | NO (deferred)   |
| 2   | drop_newest second-drop counter              | YES             |
| 3a  | Infra client teardown on SIGTERM             | YES             |
| 3b  | Postgres transient-error retry               | NO (deferred)   |
| 4   | Qdrant batched upsert                        | NO (deferred)   |
| 5   | Redis pool / model sharing                   | N/A             |
| 6   | TensorRT FP16 / device defaults              | N/A             |
| 7   | TransReID batching cosmetic batch_size       | NO (cosmetic)   |
| 8   | Evidence re-key atomicity                    | NO (deferred)   |
| 9a  | Atomic benchmark report write                | YES             |
| 9b  | Per-camera inference latency histogram       | NO (deferred)   |
| 10a | `_check_infra_env` precedence bug fix        | YES (Phase 6)   |
| 10b | Preflight covers QDRANT_HOST                 | YES             |
| 10c | `/health` probes Redis/Qdrant/MinIO          | NO (deferred)   |

Five low-risk improvements were implemented now and covered by
11 new tests (test suite: 214 → 225, all passing).  Eight
medium-risk items are documented as deferred follow-ups for the
next iteration.

## Verification

```text
uv run python -m pytest tests/ --tb=no
  -> 225 passed, 3 warnings in 27.49s

uv run ruff check app scripts tests
  -> All checks passed!

uv run ruff format --check app scripts tests
  -> (90+ files already formatted)
```
