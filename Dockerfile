# syntax=docker/dockerfile:1.7
# =============================================================================
# Dockerfile — multi-stage build for the yamaha-mtmct stack
#
# Targets:
#   base    — paddlepaddle-gpu + python deps + app user + entrypoint
#             (shared by api and sidecar)
#   api     — production image for the detect-pipeline service
#             (Paddle-only, no torch)
#   sidecar — embedding-sidecar image (base + torch + torchvision)
#
# PaddleDetection (release/2.9) is NOT baked into the image. It lives in a
# named docker volume (`paddledetection`) and is auto-cloned on first start
# by the api container's entrypoint. This keeps the api image at ~6 GB
# (down from ~25 GB) and the build at ~8 min (down from ~30 min).
#
# Build (single command, both targets):
#   docker compose build
# Or one at a time:
#   docker build --target api      -t yamaha-mtmct:api      .
#   docker build --target sidecar  -t yamaha-mtmct:sidecar  .
#
# Run:
#   docker compose up -d
# =============================================================================

ARG PYTHON_VERSION=3.12

# =============================================================================
# base stage — shared by api and sidecar
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ARG PYTHON_VERSION
USER root
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# ---- 1. System deps: deadsnakes PPA for Python 3.12 + build tools + git ----
# BuildKit cache mount on apt cache so subsequent builds skip re-downloads.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
        build-essential gcc g++ make cmake \
        ffmpeg libsm6 libxext6 libxrender1 libgl1 libglib2.0-0 \
        libpq-dev curl wget git \
        libgomp1 libgeos-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. Create venv at /opt/venv ----
RUN python${PYTHON_VERSION} -m venv /opt/venv

# ---- 3. Bootstrap pip ----
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && /opt/venv/bin/python /tmp/get-pip.py --no-cache-dir \
    && rm -f /tmp/get-pip.py

# ---- 4. Install production runtime deps (requirements.txt) ----
# BuildKit cache mount for pip so wheel downloads persist across builds.
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# ---- 5. Install PaddlePaddle GPU 2.6.2 (cu118 wheel, the last GPU
#       release on PyPI) + pin cuDNN to 8.x for ABI compatibility
#       with Paddle 2.6.2's `fused_conv2d_add_act_kernel` (compiled
#       against cuDNN 8.6 + CUDA 11.8). The symlink in step 6 makes
#       the runtime loader pick the 8.x library.
#       ``lap`` / ``numba`` / ``motmetrics`` are HARD requirements
#       of PaddleDetection's MOT pipeline (JDE/FairMOT/ByteTrack
#       linear assignment, JITed image ops, MTMCT evaluator).
#       Without them the subprocess prints its config banner then
#       hangs on the first inference frame because the import
#       fails inside the tracker (the 64 KiB pipe fills with
#       warnings + traceback and the inference thread blocks
#       on the full stderr pipe). PATCH 2026-06-18. ----
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/pip install --no-cache-dir \
        paddlepaddle-gpu==2.6.2 \
        nvidia-cudnn-cu12==8.9.7.29 \
        lap==0.5.13 \
        numba==0.65.1 \
        motmetrics==1.4.0

# ---- 6. cuDNN 8.x runtime symlinks (PATCH-051) ----
# The nvidia-cudnn-cu12==8.9.7.29 wheel ships the individual
# cuDNN libraries (libcudnn_cnn_infer.so.8, libcudnn_adv_infer.so.8, etc.)
# but does NOT ship a libcudnn.so.8 runtime stub. We create one by
# symlinking libcudnn.so.8 to the main inference library
# (libcudnn_cnn_infer.so.8), which the loader uses as the entry point.
# The remaining libcudnn_*.so.8 are then transitively loaded by
# libcudnn.so.8 via its DT_NEEDED entries.
#
# PATCH (2026-06-18): Reverted to ``nvidia-cudnn-cu12==8.9.7.29``
# after an experiment with cuDNN 9.1.0.70 (the system version)
# produced a worse failure mode: Paddle 2.6.2's binary is
# statically linked against its own bundled cuDNN symbols and
# cannot find them in cuDNN 9's renamed ABI. cuDNN 8.9 at least
# loads (the bundled cuDNN finds its DT_NEEDED entries via
# ``libcudnn.so.8`` -> ``libcudnn_cnn_infer.so.8``), but the
# ``fused_conv2d_add_act`` kernel fails on cuDNN 8.9 with
# CUDNN_STATUS_NOT_SUPPORTED. The upstream fix is Paddle
# 2.7+ (which uses cuDNN 9.x natively). Until then, the
# pipeline can be run in inference-only mode (no MOT
# tracking) by setting ``MOT.enable=False`` in the PP-Human
# config — the bare detector path doesn't trigger the fused
# kernel and works on cuDNN 8.9.
RUN CUDNN_DIR=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib \
    && if [ -f "$CUDNN_DIR/libcudnn_cnn_infer.so.8" ]; then \
         cd "$CUDNN_DIR" \
         && rm -f libcudnn.so libcudnn.so.8 \
         && ln -sf libcudnn_cnn_infer.so.8 libcudnn.so.8 \
         && ln -sf libcudnn.so.8 libcudnn.so \
         && for f in libcudnn_adv_infer libcudnn_adv_train \
                    libcudnn_cnn_infer libcudnn_cnn_train \
                    libcudnn_ops_infer libcudnn_ops_train; do \
                ln -sf "$f.so.8" "$f.so"; \
            done; \
       fi

# ---- 6b. cuBLAS unversioned symlinks (PATCH 2026-06-18) ----
# The nvidia-cublas-cu12 wheel ships ``libcublas.so.12`` /
# ``libcublasLt.so.12`` but Paddle 2.6.2's C++ runtime asks the
# loader for the unversioned ``libcublas.so``. Without these
# symlinks the attribute (ATTR) predictor fails on the first
# frame with ``libcublas.so: cannot open shared object file`` and
# the subprocess is killed by SIGTERM after the model-load banner
# is printed. PATCH 2026-06-18 after a direct
# ``pipeline.py --video_file=...`` run exposed the missing symbol.
RUN CUBLAS_DIR=/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cublas/lib \
    && if [ -d "$CUBLAS_DIR" ]; then \
         cd "$CUBLAS_DIR" \
         && for pair in "libcublas.so.12 libcublas.so" \
                        "libcublasLt.so.12 libcublasLt.so"; do \
                set -- $pair; \
                if [ -f "$1" ] && [ ! -e "$2" ]; then \
                    ln -sf "$1" "$2"; \
                fi; \
            done; \
       fi

ENV LD_LIBRARY_PATH="/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_cupti/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/nccl/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cublas/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cufft/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/curand/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cusolver/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cusparse/lib:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/nvtx/lib"

# ---- 7. Create `app` user ----
RUN useradd --create-home --uid 1000 --shell /bin/bash app

# ---- 8. Copy app code (used by api; sidecar inherits it too) ----
COPY --chown=app:app app/             /app/app/
COPY --chown=app:app configs/         /app/configs/
COPY --chown=app:app scripts/         /app/scripts/
COPY --chown=app:app pyproject.toml   /app/pyproject.toml

# ---- 9. Pre-create writable dirs (bind-mount / volume targets) ----
RUN mkdir -p /models /app/reports /app/data /opt/paddledetection \
    && chown -R app:app /models /app/reports /app/data

# ---- 10. Entrypoint: chown bind-mount volumes, auto-clone
#        PaddleDetection if missing, then drop to `app` user ----
RUN cat > /usr/local/bin/entrypoint.sh <<'EOF'
#!/bin/sh
set -eu
# Auto-clone PaddleDetection release/2.9 on first start. The named
# volume `paddledetection` is empty on a fresh `docker compose up`;
# this populates it once. Subsequent runs skip the clone (idempotent
# fast-path check). Runs as root before privilege drop.
PD_MARKER=/opt/paddledetection/deploy/pipeline/pipeline.py
if [ ! -f "$PD_MARKER" ]; then
    echo "[entrypoint] PaddleDetection missing — cloning release/2.9 ..."
    git clone --depth 1 --branch release/2.9 \
        https://github.com/PaddlePaddle/PaddleDetection /opt/paddledetection \
        && echo "[entrypoint] PaddleDetection cloned" \
        || { echo "[entrypoint] FATAL: PaddleDetection clone failed"; exit 1; }
else
    echo "[entrypoint] PaddleDetection present, skipping clone"
fi
# chown bind-mount / volume targets so the unprivileged `app` user
# (uid 1000) can write to them.
for d in /app/reports /app/data /models /opt/paddledetection; do
    if [ -d "$d" ]; then
        chown -R app:app "$d" 2>/dev/null || true
    fi
done
# Re-exec as the `app` user, passing through the CMD args.
exec runuser -u app -- /opt/venv/bin/python "$@"
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

USER root
WORKDIR /app

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["-m", "app.main"]


# =============================================================================
# api target — production detect-pipeline image (Paddle-only)
# =============================================================================
FROM base AS api

# Inherit everything from base; nothing extra needed.
# The api container runs `python -m app.main` (the default CMD).


# =============================================================================
# sidecar target — embedding-sidecar with torch for TransReID MSMT17
# =============================================================================
FROM base AS sidecar

# Install torch + torchvision. These pull `nvidia-cudnn-cu12==9.x`
# which REPLACES the cuDNN 8.x from `base`. That's fine: the sidecar
# never imports paddle, so the cuDNN ABI mismatch doesn't apply.
# Pin to torch 2.4 + cu124 (matches nvidia/cuda:12.4.x base).
#
# We use curl + retry instead of `pip install` directly because the
# 800 MB torch wheel download via pip is OOM-prone on machines with
# < 16 GB RAM. curl streams the file to disk and doesn't need
# pip's full metadata cache in memory. The wheels are cached by
# BuildKit (target=/root/.cache) so subsequent builds skip the
# download entirely.
RUN --mount=type=cache,target=/root/.cache/torch-wheels,sharing=locked \
    set -eu; \
    WHEEL_DIR=/root/.cache/torch-wheels; \
    mkdir -p "$WHEEL_DIR"; \
    TORCH_WHL="$WHEEL_DIR/torch-2.4.0+cu124-cp312-cp312-linux_x86_64.whl"; \
    TV_WHL="$WHEEL_DIR/torchvision-0.19.0+cu124-cp312-cp312-linux_x86_64.whl"; \
    if [ ! -f "$TORCH_WHL" ]; then \
        for try in 1 2 3 4 5; do \
            echo "[sidecar] downloading torch (attempt $try) ..."; \
            if curl -fSL --retry 3 --retry-delay 5 -o "$TORCH_WHL" \
                'https://download.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp312-cp312-linux_x86_64.whl'; then \
                break; \
            fi; \
            sleep 10; \
        done; \
    fi; \
    if [ ! -f "$TV_WHL" ]; then \
        curl -fSL --retry 3 --retry-delay 5 -o "$TV_WHL" \
            'https://download.pytorch.org/whl/cu124/torchvision-0.19.0%2Bcu124-cp312-cp312-linux_x86_64.whl'; \
    fi; \
    /opt/venv/bin/pip install --no-cache-dir \
        "$TORCH_WHL" \
        "$TV_WHL"
