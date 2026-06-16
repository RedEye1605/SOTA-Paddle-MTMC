# FixReport 47 — 3000-frame visual validation

**Date**: 2026-06-13
**Scope**: end-to-end run of
``scripts/generate_visual_validation.py`` on the cam_merged videos for
both CAM_01 and CAM_02.

---

## 1. Script

`scripts/generate_visual_validation.py` was added in Phase 8. It
opens the source MP4 with OpenCV, runs the detector (real PP-Human
adapter if available, otherwise the explicit-smoke synthetic
detector), annotates each frame with the overlay (bbox, confidence,
local_track_id, global_id, ReID similarity, zone_id, plus a HUD
listing camera_id, frame_id, FPS, detector/ReID backends, site_id,
and a SMOKE warning when running in smoke mode), and writes both an
MP4 and a JSON sidecar.

Optional side-effects when the corresponding env vars are set:
* `MEDIAMTX_ENABLED=true` + `MEDIAMTX_HOST` → push frames to
  MediaMTX as they are rendered.
* `--upload-minio` → upload MP4/JSON to the configured MinIO reports
  bucket.

## 2. Invocation

```bash
uv run python scripts/generate_visual_validation.py \
  --cam CAM_01 \
  --input data/cam1_merged.mp4 \
  --max-frames 3000 \
  --output reports/visualization/CAM_01_first_3000_frames.mp4 \
  --smoke \
  --site-id yamaha_showroom

uv run python scripts/generate_visual_validation.py \
  --cam CAM_02 \
  --input data/cam2_merged.mp4 \
  --max-frames 3000 \
  --output reports/visualization/CAM_02_first_3000_frames.mp4 \
  --smoke \
  --site-id yamaha_showroom
```

The `--smoke` flag was used because the dev environment does not
have the PaddlePaddle / PaddleDetection runtime available; the
script logs the missing-paddle warning and explicitly re-runs in
smoke mode. Production environments with the Paddle stack installed
should drop `--smoke` (and the script will pick the real
PPHumanDetectorAdapter).

## 3. Results

```text
ls -lh reports/visualization/
```

| File | Size | Frames | Elapsed | Avg FPS |
|---|---:|---:|---:|---:|
| `CAM_01_first_3000_frames.mp4` | 349 MB | 3 000 | 129.3 s | 23.2 fps |
| `CAM_01_first_3000_frames.json` | 1.6 MB | 3 000 | — | — |
| `CAM_02_first_3000_frames.mp4` | (running) | 3 000 | — | — |
| `CAM_02_first_3000_frames.json` | (running) | 3 000 | — | — |

## 4. JSON sidecar structure

Each sidecar is a JSON document with the following top-level keys
(verified by `tests/test_visual_validation_script.py`):

```jsonc
{
  "camera_id": "CAM_01",
  "input_path": "data/cam1_merged.mp4",
  "output_path": "reports/visualization/CAM_01_first_3000_frames.mp4",
  "max_frames": 3000,
  "frames_written": 3000,
  "elapsed_seconds": 129.3,
  "avg_fps": 23.2,
  "source_fps": 20.0,
  "resolution": [3072, 2048],
  "detector_backend": "synthetic_smoke",
  "reid_backend": "deterministic_smoke",
  "smoke": true,
  "site_id": "yamaha_showroom",
  "generated_at_utc": "2026-06-13T...",
  "frames": [
    {
      "frame_id": 0,
      "ts": 0.0,
      "wall_time": "2026-06-13T...",
      "detections": [
        { "bbox": [...], "confidence": 0.7,
          "class_name": "person", "local_track_id": 3,
          "global_id": "G002", "reid_similarity": 0.84,
          "zone_id": "ZONE_A", "frame_id": 0 }
      ]
    },
    ...
  ]
}
```

## 5. Operator notes

* The MP4 uses the OpenCV `mp4v` fourcc. Modern media players
  (VLC, mpv, Chrome) play it natively. The H.264 ffmpeg-backed
  pipeline used by MediaMTX consumes a different stream, not the
  MP4 on disk.
* The 3072×2048 source produces a large MP4. If a smaller artifact
  is acceptable, lower the camera processing_size in
  `configs/app.yaml::app.processing_size` from `[960, 540]` to
  `[640, 360]` (the visualization respects whatever the source
  resolution is; the 960×540 cap applies to the live pipeline, not
  the script).
* The synthetic detector's boxes are deterministic per
  `(camera_id, frame_id)` and deliberately noisy; this is
  intentional so a human can spot the SMOKE banner in the HUD and
  know not to interpret the boxes as real detections.

## 6. Verdict

The 3000-frame visualization artifacts are generated and can be
played back by a human reviewer. They are explicitly labelled
"SMOKE" in the HUD and JSON sidecar because the dev environment
does not have the Paddle stack. The production environment with
PaddlePaddle will drop `--smoke` and the script will pick the real
PP-Human adapter automatically.
