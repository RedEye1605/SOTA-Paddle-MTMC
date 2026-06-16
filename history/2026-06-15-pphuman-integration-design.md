# 40 — PPHuman integration design

## Decision

**Option B (subprocess-per-stream with a per-frame adapter)**.

## Why not Option A (callable in-process)

* Loads the full PP-Human model in the *parent* Python process.
  The audit (PATCH-007) and the architecture-guard test
  `tests/test_architecture_guards_one_model.py::test_one_model_per_process`
  require "one model instance per process". A single-process
  model shared across N camera workers would also serialise
  inference through the GIL and crush throughput.
* Requires `paddle` / `PaddleInference` imports in the API
  image. The current image is the orchestrator; the inference
  model lives in the subprocess (the official pattern).
* Would require re-implementing MOT output tail logic *and* the
  per-frame dispatch path.

## Why Option B is preferred

* The official PaddleDetection multi-stream pattern is
  *subprocess-per-stream* (per
  https://github.com/PaddlePaddle/PaddleDetection `multi_camera_mtmct_en.md`).
  Each subprocess runs `deploy/pipeline/pipeline.py --video_file=…`
  and writes a MOT text file under `{output_dir}/mot_results/`.
* `PPHumanDetectorAdapter` already exposes
  `run_pipeline()` / `parse_mot_file()` for this — the
  implementation is **correct**; the integration glue to the
  worker is what's missing.
* No custom inference code: we consume the official MOT text
  output, parse it into `Detection` records, and the worker
  emits `LocalTrack` for the current frame. Smoke tests for
  `parse_mot_file` already pass.

## The bridge

A new class `PPHumanFrameStateAdapter` (added to
`app/detection/pphuman_pipeline.py`) wraps the subprocess
manager and exposes a per-frame lookup:

```python
def detections_for_frame(self, camera_id, frame_id) -> list[Detection]:
    """Return the MOT detections for ``frame_id`` (already
    accumulated from the subprocess), or [] if not yet seen.
    """
```

`MultiCameraRunner.start()` builds the subprocess manager *when
and only when* a real `PPHumanDetectorAdapter` is present, then
constructs each `PPHumanWorker` with a per-camera
`detector_factory` that calls
`adapter.detections_for_frame(camera_id, frame_id)`. The
worker contract stays unchanged (per-frame callable via
`detector_factory`); only the adapter changes.

The synthetic / smoke path is preserved: the worker still falls
back to `_synthetic_detect(frame)` when both `self._detector`
and `self._detector_factory` are None *and* the mode is
`SMOKE_TEST`. Production mode refuses to start without a
detector (the existing `assert_production_safe` call stays).

## Subprocess output format

The official MOT txt is
`frame,id,x1,y1,w,h,score,-1,-1,-1`. `parse_mot_file()` already
converts `(x1, y1, w, h)` to `(x1, y1, x2, y2)`. The
frame-keyed adapter accumulates these per frame so a
back-pressure-driven late-arriving subprocess write is still
attributable to its `frame_id`.

## Hard guarantees (unchanged)

* Local `local_track_id` is camera-local (the resolver
  decides global identity).
* Synthetic detector is **only** allowed in `SMOKE_TEST`.
  Production benchmark fails fast if no detector.
* One model instance per process (subprocess-per-camera, but
  each subprocess holds exactly one model).

## Fallback / when the official pipeline is not installed

If the operator does not have `pipeline.py` on disk, the
adapter's `load()` will (in production) call
`assert_production_safe` and refuse. The operator then must
either (a) install PaddleDetection, or (b) run the staged
offline pipeline (Option C — `python deploy/pipeline/pipeline.py`
once per camera, write the MOT to disk, parse). Option C is
not implemented in code; it is documented here as the fallback
the operator can run by hand, then feed the MOT files into a
thin reader.

## Phase 2 deliverable

* `app/detection/pphuman_pipeline.py` — add
  `PPHumanFrameStateAdapter` + a small `MOTAccumulator`.
* `app/workers/multi_camera_runner.py` — when a real
  `PPHumanDetectorAdapter` is passed, start a
  `PPHumanPipelineSubprocessManager`, and wire each
  `PPHumanWorker`'s `detector_factory` to the frame-state
  adapter.
* `app/workers/pphuman_worker.py` — keep the
  `detector_factory` path; remove the
  `NotImplementedError` in the `self._detector is not None`
  branch (replace it with "should be unreachable: caller
  passes `detector_factory` not `detector`").
* `scripts/benchmark_t4.py` — same wiring; if
  `PPHumanDetectorAdapter.load()` fails in production mode,
  the benchmark must exit non-zero (current behaviour already
  raises; we ensure the report records `detector_backend =
  "real_pphuman"` on success and `status = "failed"` on
  load failure).
* Tests: `tests/test_pphuman_detector_adapter.py` (new),
  `tests/test_production_benchmark_real_detector.py` (new).
