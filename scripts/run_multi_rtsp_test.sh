#!/usr/bin/env bash
# Run the multi-camera RTSP smoke test.
set -euo pipefail

cd "$(dirname "$0")/.."
export LOG_LEVEL=INFO
export HEADLESS=true
export DRAWING_ENABLED=false
export SMOKE_MAX_FRAMES="${SMOKE_MAX_FRAMES:-300}"
export SMOKE_MAX_SECONDS="${SMOKE_MAX_SECONDS:-30}"

python -m app.main --mode multi_rtsp
