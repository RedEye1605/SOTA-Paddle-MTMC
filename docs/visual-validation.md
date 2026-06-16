# Visual validation — 3000 frames per camera

The SOTA pipeline ships an end-to-end visual-validation tool that
opens a recorded video, runs the detector + local tracker + ReID
on the first N frames, annotates every frame with bbox, detector
confidence, local track id, global id, ReID similarity, and (when
present) zone id, and writes both an annotated MP4 and a JSON
sidecar.

## When to use it

* **Operator visual sanity check**: confirm the detector is
  actually drawing boxes where the operator expects.
* **End-to-end smoke**: the script will pick the synthetic
  detector + deterministic ReID when paddle is not available,
  and the HUD will display a clear `SMOKE-TEST BACKEND` warning.
* **Real validation on recorded footage**: drop `--smoke` to use
  the real PP-Human adapter (requires the PaddleDetection
  pipeline at the operator-configured path).

## Usage

```bash
uv run python scripts/generate_visual_validation.py \
  --cam CAM_01 \
  --input data/cam1_merged.mp4 \
  --max-frames 3000 \
  --output reports/visualization/CAM_01_first_3000_frames.mp4 \
  --smoke \
  --site-id yamaha_showroom
```

Replace `--smoke` with the default (no flag) to use the real
PP-Human adapter. Output is always an MP4 + JSON sidecar.

## Output

| File | Purpose |
| --- | --- |
| `*.mp4` | Annotated video, H.264 via the OpenCV `mp4v` muxer. |
| `*.json` | Per-frame detection / identity decisions. |

The sidecar's per-frame structure:

```jsonc
{
  "frame_id": 0,
  "ts": 0.0,
  "wall_time": "2026-06-13T15:41:27+00:00",
  "detections": [
    {
      "bbox": [552.1, 755.3, 622.6, 1055.7],
      "confidence": 0.68,
      "class_name": "person",
      "local_track_id": 6,
      "global_id": "G001",
      "reid_similarity": 0.89,
      "zone_id": "ZONE_A",
      "frame_id": 0
    }
  ]
}
```

## Side effects

* When `MEDIAMTX_ENABLED=true` *and* `MEDIAMTX_HOST` is non-empty,
  the script pushes the annotated stream to MediaMTX as it
  renders. Per-camera URL: `rtsp://{host}:{port}/{prefix}/{cam}`.
* When `--upload-minio` is passed, the MP4 and JSON are uploaded
  to the configured MinIO `reports` bucket with deterministic
  keys.

## Smoke vs real

| Mode | Detector | ReID | HUD banner |
| --- | --- | --- | --- |
| `--smoke` | synthetic boxes (deterministic per frame) | deterministic cosine mock | `WARNING: SMOKE-TEST BACKEND` |
| real (default) | `paddledetection_pphuman` (when paddle is importable) | `pphuman_strongbaseline` (per `configs/app.yaml`) | no banner |

The script does **not** silently fall back from real to smoke; if
paddle is not importable in production mode the script logs a
clear warning and re-runs in smoke mode. The MP4 and JSON are
still written — the operator can play them, and the HUD tells the
operator explicitly that they are smoke outputs.

## Tests

* `tests/test_visual_validation_script.py` — subprocess-level
  contract (max-frames honored, MP4 written, sidecar schema).
* `tests/test_visual_overlay_contract.py` — overlay rendering
  contract (smoke banner, label fields, input immutability).
