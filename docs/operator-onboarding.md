# Operator Onboarding with uv

> **End-to-end onboarding for `SOTA-Paddle-MTMC` using uv as the
> sole package manager.  Every command in this guide has been
> verified to run on a host with 1 NVIDIA T4 + 8 CPU cores.**

Hard rules (enforced):

```text
- uv is the only normal package manager.  pip is reserved for
  documented emergency diagnostics (see pyproject.toml).
- Service/ is read-only.  Only SOTA-Paddle-MTMC/ may be modified.
- Production refuses synthetic detectors and deterministic ReID.
- Smoke-test paths are gated by RuntimeMode.SMOKE_TEST.
```

## 1. Provision

```bash
git clone <repo> /opt/sota-paddle-mtmc
cd /opt/sota-paddle-mtmc
# uv install (idempotent — skip if already on the host)
curl -LsSf https://astral.sh/uv/0.5.31/install.sh | sh
uv --version    # >= 0.5.31
```

## 2. Environment file + secrets

```bash
cp .env.example .env
chmod 600 .env
# Edit .env and set the five required secrets:
#   SOTA_API_TOKEN        -> at least 32 random chars
#   POSTGRES_PASSWORD     -> not the default "change_me_in_production"
#   MINIO_ACCESS_KEY      -> not the default
#   MINIO_SECRET_KEY      -> not the default
#   MQTT_PASSWORD         -> not the default (only if MQTT is enabled)
```

Verify:

```bash
grep -c "change_me_in_production" .env       # must return 0
stat -c "%a %n" .env                         # must show 600 .env
```

## 3. Python deps via uv

```bash
uv sync --frozen --extra dev
uv run python -m pytest tests/ -q --tb=no    # 225 passed
uv run python -m compileall app scripts tests
```

> **No `pip install` here.**  If a hot diagnostic ever needs `pip`,
> document it as a temporary emergency command, never as the
> normal workflow.

## 4. PaddleDetection clone + PP-Human models

See [`pphuman_model_setup.md`](pphuman_model_setup.md) for the
exact URLs.  Summary:

```bash
git clone --depth 1 https://github.com/PaddlePaddle/PaddleDetection.git \
    /home/$USER/paddledetection
echo "PPHUMAN_PIPELINE_PATH=/home/$USER/paddledetection/deploy/pipeline/pipeline.py" >> .env

mkdir -p models/pphuman
bash scripts/download_pphuman_models.sh
```

## 5. TransReID weight

See [`transreid_msmt_setup.md`](transreid_msmt_setup.md).  Summary:

```bash
bash scripts/download_transreid_models.sh
uv run python scripts/inspect_transreid_checkpoint.py \
    models/vit_transreid_msmt.pth --profile msmt17
```

## 6. Lint + format gate

See [`ruff_quality_gate.md`](ruff_quality_gate.md).  Summary:

```bash
uv run ruff check app scripts tests
uv run ruff format --check app scripts tests
```

## 7. Bring up infra

```bash
docker compose up -d relation-store vector-store message-bus
sleep 20
docker compose ps             # all 3 must be (healthy)
```

The Qdrant healthcheck must use `bash -c "echo > /dev/tcp/..."` —
the image is too minimal for `curl` / `wget`.  This is already
configured in `docker-compose.yaml`.

## 8. Migrations + Qdrant init

Migrations run automatically on first postgres init via
`/docker-entrypoint-initdb.d`.  Verify:

```bash
docker exec yamaha-mtmct-relation-store-1 \
    psql -U yamaha -d yamaha_mtmct -c "\dt"
# Expect 13 tables (cameras, global_identities, zones, etc.)
```

Initialize Qdrant collections (idempotent):

```bash
# From inside the detect-pipeline container (recommended):
docker compose run --rm detect-pipeline python scripts/init_qdrant.py

# OR from the host (for diagnostics):
PYTHONPATH=. QDRANT_HOST=localhost \
    uv run python scripts/init_qdrant.py
# -> "Qdrant ready. Collections: ['person_reid_transreid', ...]"
```

## 9. Production preflight

See [`preflight_and_readiness_gate.md`](preflight_and_readiness_gate.md).

```bash
docker compose run --rm detect-pipeline python scripts/readiness_preflight.py \
    --out scripts/readiness_preflight.json

# OR from the host:
set -a; source .env; set +a
QDRANT_HOST=localhost POSTGRES_HOST=localhost REDIS_HOST=localhost \
    uv run python scripts/readiness_preflight.py \
    --out scripts/readiness_preflight.json
```

All 6 checks must pass:

```text
[OK  ] sota_api_token
[OK  ] transreid_weight
[OK  ] pphuman_pipeline
[OK  ] infra_env
[OK  ] docker_compose
[OK  ] benchmark_dir
```

## 10. Smoke benchmark + readiness gate

```bash
PYTHONPATH=. QDRANT_HOST=localhost POSTGRES_HOST=localhost REDIS_HOST=localhost \
    uv run python scripts/benchmark_t4.py \
    --mode smoke_benchmark \
    --dataset configs/benchmark_smoke.yaml \
    --max-seconds 10 --out-dir reports

PYTHONPATH=. uv run python scripts/readiness_gate.py \
    --preflight scripts/readiness_preflight.json \
    --benchmark-dir reports \
    --out reports/readiness_gate.json
```

The maximum verdict reachable without a real recorded
multi-camera dataset is:

```text
READY_FOR_SHADOW_TEST
```

`READY_FOR_LIMITED_PRODUCTION` requires a real production
benchmark.  Never claim it based on a smoke benchmark.

## Tear-down

```bash
docker compose down              # leaves volumes
docker compose down -v           # wipes relation-store-data, vector-store-data, ...
                                 # NOTE: MinIO data lives on the operator's external
                                 # cluster (minio.example.invalid:9000) and is
                                 # not touched by `docker compose down -v`.
```
