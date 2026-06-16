# Docker build

> **Multi-stage CUDA Dockerfile for the Paddle API image and the
> torch TransReID sidecar.** Runtime Python packages are installed
> from `requirements.txt`; the sidecar adds torch in its own target
> so torch's cuDNN 9.x dependencies do not land in the API image.

## Stages

```text
base:
  nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
  Python 3.12 venv at /opt/venv
  requirements.txt + paddlepaddle-gpu==2.6.2 + cuDNN 8.x
  app/configs/scripts copied into /app

api:
  inherits base
  Paddle-only; no torch

sidecar:
  inherits base
  adds torch==2.4.0+cu124 and torchvision==0.19.0+cu124
```

## Build

```bash
docker compose build detect-pipeline embedding-sidecar
```

> **Heavy build.**  Layer 2 downloads PaddlePaddle GPU and its
> CUDA deps, and the sidecar target downloads torch wheels. The
> first build is network-heavy; subsequent builds reuse BuildKit
> cache layers.

## Run via docker compose

```bash
docker compose up -d relation-store vector-store message-bus
sleep 20
docker compose ps
docker compose up -d detect-pipeline
docker compose logs detect-pipeline --tail=100
```

## Mounts

`docker-compose.yaml` mounts:

```text
./configs    -> /app/configs:ro
./app        -> /app/app:ro
./scripts    -> /app/scripts:ro
./models     -> /models:ro             # host model weights
```

The TransReID weight (~420 MB) is **not** baked into the image —
it must exist on the host at `./models/vit_transreid_msmt.pth` and
is mounted read-only into the containers at `/models/vit_transreid_msmt.pth`.

The PP-Human models must exist on the host under `./models/pphuman`
and are mounted read-only at `/models/pphuman`.  Run
`scripts/download_pphuman_models.sh` if the directory is missing.

## Required env

Every required env var is documented in `.env.example` and
verified by `scripts/readiness_preflight.py`.  Run the preflight
from inside the detect-pipeline container after a fresh build:

```bash
docker compose run --rm detect-pipeline python scripts/readiness_preflight.py
```

## Healthcheck

The detect-pipeline `/health` endpoint is probed every 30 s
with a 5 s timeout; the container is `unhealthy` after 3
consecutive failures.  See
[`preflight_and_readiness_gate.md`](preflight_and_readiness_gate.md).

## Dependency source of truth

```text
requirements.txt  -> Docker runtime dependencies
pyproject.toml    -> local uv/dev dependency declaration
uv.lock           -> local uv lockfile
```

Keep the PaddleDetection MOT deps (`lap`, `numba`, `motmetrics`)
listed in both `requirements.txt` and `pyproject.toml`. The Dockerfile
also installs them explicitly after Paddle as a belt-and-suspenders
guard for the PP-Human tracker path.

## Tear-down

```bash
docker compose down              # leaves named volumes
docker compose down -v           # wipes infra volumes; host ./models remains
docker image prune -f            # cleans dangling layers
```
