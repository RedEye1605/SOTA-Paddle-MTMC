#!/usr/bin/env bash
# =============================================================================
# Download PP-Human official models.
# Source: PaddleDetection official model zoo
# (https://github.com/PaddlePaddle/PaddleDetection/blob/develop/README.md,
# rows 394-395 — confirmed 2026-06-12).
#
# Both models live under
#   https://bj.bcebos.com/v1/paddledet/models/pipeline/<NAME>.zip
# and the file extension is .zip (not .tar.gz).
# =============================================================================
set -euo pipefail

MODEL_DIR="${PPHUMAN_MODEL_DIR:-/models/pphuman}"
mkdir -p "$MODEL_DIR"

echo "Downloading PP-Human detector+tracker (mot_ppyoloe_l_36e_pipeline, 182 MB)..."
wget -q \
  -O "$MODEL_DIR/mot_ppyoloe_l_36e_pipeline.zip" \
  https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
unzip -q -o "$MODEL_DIR/mot_ppyoloe_l_36e_pipeline.zip" -d "$MODEL_DIR"
rm "$MODEL_DIR/mot_ppyoloe_l_36e_pipeline.zip"

echo "Downloading PP-Human StrongBaseline ReID (strongbaseline_r50_30e_pa100k, 86 MB)..."
wget -q \
  -O "$MODEL_DIR/strongbaseline_r50_30e_pa100k.zip" \
  https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip
unzip -q -o "$MODEL_DIR/strongbaseline_r50_30e_pa100k.zip" -d "$MODEL_DIR"
rm "$MODEL_DIR/strongbaseline_r50_30e_pa100k.zip"

echo "Done. Models in $MODEL_DIR:"
ls -la "$MODEL_DIR"
echo "Detector contents:"
ls "$MODEL_DIR/mot_ppyoloe_l_36e_pipeline/" 2>/dev/null || echo "  (not yet extracted)"
echo "ReID contents:"
ls "$MODEL_DIR/strongbaseline_r50_30e_pa100k/" 2>/dev/null || echo "  (not yet extracted)"
