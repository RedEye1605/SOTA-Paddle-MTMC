#!/bin/sh
# scripts/fetch_torch_wheels.sh — pre-download torch + torchvision
# cu124 wheels into ./wheels/ so the sidecar Dockerfile can COPY
# them locally instead of fetching from PyPI during the build.
#
# Run once after `git pull` (or when the wheels/ dir is missing):
#   ./scripts/fetch_torch_wheels.sh
# Result: ./wheels/torch-2.4.0+cu124-cp312-cp312-linux_x86_64.whl
#         ./wheels/torchvision-0.19.0+cu124-cp312-cp312-linux_x86_64.whl
set -eu
mkdir -p wheels
pip download \
    --index-url https://download.pytorch.org/whl/cu124 \
    --dest wheels/ \
    --no-deps \
    torch==2.4.0+cu124 \
    torchvision==0.19.0+cu124
echo "Wheels downloaded to ./wheels/"
