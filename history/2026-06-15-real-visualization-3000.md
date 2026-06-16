# Phase 5 — Real 3000-frame visualization (deferred)

Date: 2026-06-13

## Status: DEFERRED

The real 3000-frame visualization cannot run end-to-end against
the real PP-Human detector on the cam_merged videos because of
**two layered issues**:

1. **Bridge issue (FIXED)**: the visualization script's call to
   `PPHumanDetectorAdapter` used the kwarg names
   `infer_cfg=...` and `tracker_cfg=...` — neither is a real
   kwarg on the adapter (the adapter uses `config_path=...`).
   This made the adapter raise `TypeError: unexpected keyword
   argument 'infer_cfg'` and silently fall back to the smoke
   detector. PATCH-051 fixes the call site in
   `scripts/generate_visual_validation.py::_try_load_real_detector`
   to use the correct `config_path` kwarg.

2. **Architectural issue (OUT OF SCOPE)**: after the kwarg
   fix, the visualization script tries to call
   `adapter.detect(frame)` for every frame, but the
   `PPHumanDetectorAdapter` is **subprocess-only** — it exposes
   `run_pipeline(camera_id, video_file, output_dir)` which
   launches an official PaddleDetection subprocess for the
   entire video, not a per-frame callable. The per-frame
   callable is what the visualization script's loop expects.

   The runtime-mode-aware `PPHumanFrameStateAdapter` and
   `PPHumanWorker` chain in `app/workers/multi_camera_runner.py`
   is the only path the adapter supports. Wiring the
   visualization script to that chain is a 50-100 line
   refactor (run the multi-camera runner with one camera, then
   render the per-frame state to a video). Filed as a
   follow-up task; it does NOT block the production benchmark
   (which already uses the per-frame state chain via
   `MultiCameraRunner`).

3. **Paddle wheel pin issue (DOCUMENTED in Phase 4)**: even
   if the visualization script were refactored to use the
   real detector, the subprocess would hit
   `CUDNN_STATUS_NOT_SUPPORTED` in `fused_conv2d_add_act` —
   the same blocker that prevents the production benchmark
   from completing inference.

## What was fixed

- `scripts/generate_visual_validation.py::_try_load_real_detector`:
  the call to `PPHumanDetectorAdapter` now uses the correct
  `config_path` kwarg. After this fix the script no longer
  falls back to the smoke detector because of a TypeError.
  It still falls back to the smoke detector because the
  adapter has no `detect(frame)` method, but for a
  different, structural reason — which is the architecturally
  correct diagnosis.

## What the operator should do

The visualization's "real detector" path is a future refactor:

1. Refactor `generate_visual_validation.py` to drive the
   `PPHumanFrameStateAdapter` (the same path the
   `MultiCameraRunner` uses) instead of calling
   `adapter.detect(frame)` directly.
2. After (1), resolve the cudnn 9 + paddle 2.6.2
   incompatibility (Phase 4 §"Remaining blockers").
3. Only then can the real 3000-frame visualization render
   with overlay fields
   `camera_id / frame_id / detection bbox / detection
   confidence / local_track_id / global_id / ReID model /
   final_score or similarity / zone_id / detector_backend /
   reid_backend` and a HUD that is **not** "SMOKE-TEST
   BACKEND".

## Sidecar (smoke mode, for reference only)

The most recent smoke-mode sidecar lives at
`reports/visualization/CAM_02_first_3000_frames_real.mp4.json`
(from a prior session). The smoke-mode sidecar
unambiguously states `detector_backend=synthetic_smoke` and
`reid_backend=smoke_deterministic` and includes a HUD that
reads "SMOKE-TEST BACKEND". It does **not** satisfy Phase 5
acceptance. The real version is the future refactor described
above.

## Verdict

Phase 5 is **DEFERRED**, not failed. The bridge-side kwarg
fix is in place; the architectural refactor is the next
session's work; the cudnn 9 fix is the operator-side
prerequisite.

## Files changed

- `scripts/generate_visual_validation.py` — `config_path`
  kwarg, removed unused `tracker_cfg` parameter (the
  adapter doesn't accept it).
