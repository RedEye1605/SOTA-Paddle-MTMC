# 39 — PPHumanWorker ↔ PPHumanDetectorAdapter gap baseline

## 1. Exact crash location

`app/workers/pphuman_worker.py::PPHumanWorker._detector_dets`
line 170-175, inside the `if self._detector is not None:` branch.

```python
if self._detector is not None:
    raise NotImplementedError(
        "PPHumanDetectorAdapter is intended to be invoked via its "
        "subprocess manager; the worker delegates per-frame via "
        "detector_factory (a callable) instead. ..."
    )
```

The `MultiCameraRunner` always constructs a worker with
`detector=self._shared_detector`. In production_benchmark the
shared detector is the real `PPHumanDetectorAdapter`, so as soon as
the first non-skipped frame arrives, `_detector_dets` raises
`NotImplementedError`. The thread dies, `runner.stream()` keeps
yielding the empty/queued frames (or stops once all workers
crash), and the benchmark report is written with a high
`total_analytics_fps` value (1706 FPS in the latest
`reports/benchmark_20260612T193807Z.json`) that reflects *inter-queue
drain intervals*, not real inference.

## 2. Exact caller/callee mismatch

* `MultiCameraRunner.start()` (app/workers/multi_camera_runner.py:198-220)
  passes `detector=self._shared_detector` to every `PPHumanWorker`.
* `PPHumanWorker._detector_dets` (line 156-195) expects
  `self._detector_factory` to be a *callable*; the real adapter
  instance is NOT callable.
* `scripts/benchmark_t4.py::run_scenario` (line 293-327) constructs
  a real `PPHumanDetectorAdapter` and passes it to
  `MultiCameraRunner(..., detector=detector)`. This is the
  path that crashes.

## 3. Adapter shape

`PPHumanDetectorAdapter` is **subprocess-only**. It exposes:

* `load()` — probe-only, fails fast in production.
* `synthetic_stream()` — smoke-test fallback (used by
  MultiCameraRunner as a separate path, not by the worker).
* `run_pipeline()` / `build_pipeline_command()` — invokes the
  official `pipeline.py` as a subprocess.
* `parse_mot_file()` — tail the per-camera MOT txt file.
* `PPHumanPipelineSubprocessManager` — manages one subprocess per
  camera, tails MOT outputs, yields `(camera_id, Detection)` tuples.

There is **no `__call__` / per-frame API** on the adapter today.
The worker expects `self._detector` to be either a smoke fallback
(None) or a per-frame callable. The two sides disagree.

## 4. Which production path `benchmark_t4.py` currently uses

`scripts/benchmark_t4.py` line 293-317:
1. constructs `PPHumanDetectorAdapter(...)` in production mode
2. calls `detector.load()` (probe-only; passes when the pipeline
   is missing only in smoke mode)
3. hands the adapter to `MultiCameraRunner(..., detector=detector)`
4. the runner's per-camera threads call `worker._detector_dets(frame)`
   → `NotImplementedError`

The `PPHumanPipelineSubprocessManager` defined in the same module
is *not* instantiated by `benchmark_t4.py`. The audit's
"subprocess-per-stream" design exists in code but is not wired to
the runner path that the benchmark exercises.

## 5. Why current report is structurally valid but not performance-valid

The reported `total_analytics_fps` comes from
`runner.stream()`'s inter-yield interval (the time between two
`q.get()` returns). When workers crash early, the runner's
`stream()` loop still iterates but mostly produces no work; the
interval collapses to "how fast can the consumer loop", which
inflates the FPS metric without any detection having happened.

The report also lacks:
* `detector_backend` (no record of "real_pphuman" vs
  "synthetic_smoke")
* `reid_backend`
* `workers_crashed` boolean
* `required_metrics_present` boolean
* `false_merge_rate` / `cross_camera_match_accuracy` /
  `id_fragmentation_rate` (no real model ran, no real labels
  were scored)

The same is true of `gpu_memory_used_mb_max` and the Qdrant /
Postgres latencies — they are real but reflect the idle benchmark
environment, not an inference load.

## 6. Current verified state (this run)

* `uv run ruff check app scripts tests` — all checks passed
* `uv run ruff format --check app scripts tests` — 92 files already
  formatted
* `uv run python -m pytest tests/` — **229 passed** in 28.80s
* `uv run python -m compileall app scripts tests` — clean
* `docker compose config` — valid
* `docker compose ps` — all 5 services healthy (api, minio,
  postgres, qdrant, redis)

## 7. Verdict

* Smoke benchmark: passes — synthetic detector path works end to
  end, readiness gate caps at `READY_FOR_SHADOW_TEST` correctly.
* Production benchmark: structurally runs (JSON + MD written) but
  the per-camera worker crashes, so the report is not
  performance-valid. Gate correctly refuses
  `READY_FOR_LIMITED_PRODUCTION` because the required real
  metrics are missing.
* Maximum honest verdict right now: **`READY_FOR_SHADOW_TEST`**.

## 8. Next steps

Phase 1 will pick the lowest-risk integration design that uses the
real PP-Human model without re-architecting the runner. The
constraint set is documented in
`FixReports/40_pphuman_integration_design.md`.
