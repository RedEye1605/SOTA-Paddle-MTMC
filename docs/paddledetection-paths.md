# PaddleDetection — host vs container paths

This project runs the same code in two contexts:

1. **Host (`uv run ...`)** — the operator's local checkout.
2. **Container (`docker compose run --rm api ...`)** — the production image.

Each context has its **own** filesystem and its **own** correct path to the
PaddleDetection clone. The `.env` file is the source of truth for the **host**
context; `docker-compose.yaml` is the source of truth for the **container**
context. They must never cross.

## Why two paths?

`/opt/paddledetection` is the upstream-recommended path, but on a typical
dev box `/opt` is root-owned. Operators usually clone to
`/home/<user>/paddledetection` instead and set
`PPHUMAN_PIPELINE_PATH` to point at it.

The Docker image, by contrast, has full control of `/opt`, so the
Dockerfile clones the same upstream repo to `/opt/paddledetection`
(Layer 4 of the image). The `docker-compose.yaml` env block pins the
container-side env vars to the container paths, **overriding** any
host `.env` value of the same name.

## What the runtime sees

| Env var | Host (uv run) | Container (docker compose) |
|---|---|---|
| `PPHUMAN_PIPELINE_PATH` | `$(grep ^PPHUMAN_PIPELINE_PATH .env)` | `/opt/paddledetection/deploy/pipeline/pipeline.py` (compose) |
| `PPHUMAN_INFER_CONFIG`  | `$(grep ^PPHUMAN_INFER_CONFIG .env)`  | `/opt/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml` (compose) |
| `PPHUMAN_MODEL_DIR`     | `$(grep ^PPHUMAN_MODEL_DIR .env)`     | `/models/pphuman` (compose) |
| `PPHUMAN_REID_MODEL_DIR`| inherited from `PPHUMAN_MODEL_DIR`    | `/models/pphuman/strongbaseline_r50_30e_pa100k` (compose) |
| `TRANSREID_WEIGHT`      | `$(grep ^TRANSREID_WEIGHT .env)`      | `/models/vit_transreid_msmt.pth` (compose) |

The reason this works: in `docker-compose.yaml`, the api service has both
`env_file: .env` and a literal `environment:` block. Compose merges
them with the literal `environment:` block winning, so the container
**always** sees the container paths, even if `.env` mentions the host
ones. This is intentional and is called out in the comment block above
the `environment:` key.

## Do not

- Do **not** clone PaddleDetection to `/opt` on the host — `/opt` is
  root-owned, and the host does not need it there.
- Do **not** set `PPHUMAN_PIPELINE_PATH` in `.env` to a container path
  (`/opt/...`); it will not be reachable from the host and will fail
  the host preflight. The container overrides this anyway, so the
  value is silently wrong for the host context.
- Do **not** remove the `environment:` block from `docker-compose.yaml`
  and rely on `.env` — the container would then inherit the host path
  and the production benchmark would fail with `FileNotFoundError`
  on `pipeline.py`.

## How to fix a path failure

If `readiness_preflight.py` reports:

```text
[FAIL] pphuman_pipeline: pipeline='/home/rhendy/paddledetection/...'
       reason='file not found'
```

…on the **host**, the clone is missing. Run:

```bash
test -f /home/rhendy/paddledetection/deploy/pipeline/pipeline.py || \
  git clone --depth 1 https://github.com/PaddlePaddle/PaddleDetection.git \
    /home/rhendy/paddledetection
```

If the same message appears **inside the container**, the image build
failed on Layer 4. Rebuild:

```bash
docker compose build --no-cache api
```

If only the model dir fails (`PPHUMAN_MODEL_DIR`), the model weights
are missing. Run `scripts/download_pphuman_models.sh` on the host
so `./models/pphuman` is populated before starting docker compose.

## Sanity check

Inside the container, after `docker compose run --rm api ...`:

```bash
docker compose run --rm api python scripts/readiness_preflight.py \
  --out /tmp/preflight.json
jq '.checks[] | select(.name == "pphuman_pipeline")' /tmp/preflight.json
# pipeline MUST be /opt/paddledetection/...
```

On the host:

```bash
uv run python scripts/readiness_preflight.py \
  --out /tmp/preflight.json
jq '.checks[] | select(.name == "pphuman_pipeline")' /tmp/preflight.json
# pipeline MUST be /home/rhendy/paddledetection/...
```
