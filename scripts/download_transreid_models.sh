#!/usr/bin/env bash
# =============================================================================
# Download TransReID (damo-cv/TransReID) checkpoint for Market-1501.
# Verified via Context7: configs/Market/vit_transreid_stride.yml +
# test.py --config_file <config> TEST.WEIGHT <weight>.
# =============================================================================
set -euo pipefail

MODEL_DIR="${TRANSREID_MODEL_DIR:-/models/transreid}"
mkdir -p "$MODEL_DIR"

echo "Downloading ViT-B/16 pretrained (jx_vit_base_p16_224-80ecf9dd.pth)..."
wget -q --show-progress \
  -O "$MODEL_DIR/jx_vit_base_p16_224-80ecf9dd.pth" \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

echo "Downloading TransReID Market-1501 checkpoint (transformer_120.pth)..."
wget -q --show-progress \
  -O "$MODEL_DIR/transformer_120.pth" \
  "https://drive.google.com/uc?export=download&id=1UX8q2_u1KuZcUc6hJEDO1J52XHQpi-b_" \
  || echo "FAILED: download from Google Drive requires manual install; see https://github.com/damo-cv/TransReID#testing"

echo "Done. Models in $MODEL_DIR."
ls -la "$MODEL_DIR"
