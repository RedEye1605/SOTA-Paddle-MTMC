#!/usr/bin/env bash
# Run a local video single-camera smoke test.
set -euo pipefail

VIDEO_FILE="${1:-demo/sample.mp4}"
CAMERA_ID="${2:-CAM_01}"

cd "$(dirname "$0")/.."
export LOG_LEVEL=INFO
export SMOKE_MAX_FRAMES="${SMOKE_MAX_FRAMES:-300}"

python -m app.main \
  --mode single_cam_smoke \
  --camera-id "$CAMERA_ID" \
  --video-file "$VIDEO_FILE"
