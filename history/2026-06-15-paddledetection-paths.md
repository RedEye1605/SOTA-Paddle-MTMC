# Phase 1 — PaddleDetection Path Resolution (host vs container)

Date: 2026-06-13

## Goal

Ensure that the PaddleDetection clone is correctly resolved:

- Host (uv run) uses `/home/rhendy/paddledetection/...`
- Container (docker compose) uses `/opt/paddledetection/...`
- The container's paths are pinned in `docker-compose.yaml` so the
  host `.env` cannot leak through.

## Inspection

```text
rg -n "PPHUMAN_PIPELINE_PATH|PPHUMAN_INFER_CONFIG|PPHUMAN_MODEL_DIR|PPHUMAN_REID_MODEL_DIR|TRANSREID_WEIGHT|/opt/paddledetection|/home/rhendy/paddledetection" \
  .env docker-compose.yaml Dockerfile configs app scripts Docs tests
```

Result: 58 matches across 18 files. Key findings:

| Location | Path | Notes |
|---|---|---|
| `.env` | `/home/rhendy/paddledetection/...` | Host-only. Comment notes /opt is root-owned. |
| `docker-compose.yaml` env block | `/opt/paddledetection/...` and `/app/models/pphuman` | Container-only. **Overrides** env_file with literal values. |
| `Dockerfile` Layer 4 | clones to `/opt/paddledetection` | Image-baked. |
| `app/detection/pphuman_pipeline.py` | defaults to `/opt/paddledetection/...` | Default; overridden by env. |
| `scripts/benchmark_t4.py` | defaults to `/opt/paddledetection/...` | Default; overridden by env. |
| `app/main.py` | defaults to `/opt/paddledetection/...` | Default; overridden by env. |

## Verdict

Path resolution is **already correct**:

1. **Host uv run** uses `.env` which points at
   `/home/rhendy/paddledetection/...`. The host clone is present and
   has the required files (`deploy/pipeline/pipeline.py`,
   `deploy/pipeline/config/infer_cfg_pphuman.yml`).
2. **Docker** uses `docker-compose.yaml`'s `environment:` block which
   pins `/opt/paddledetection/...`. The Dockerfile Layer 4 clones the
   same repo into `/opt/paddledetection` at build time. The literal
   `environment:` block wins over `env_file`, so host paths cannot
   leak into the container.
3. **Docs**: `Docs/official_paddle_integration.md` and
   `Docs/pphuman_model_setup.md` both tell operators to clone to a
   user-owned path and to set `PPHUMAN_PIPELINE_PATH` accordingly. No
   instruction instructs cloning to `/opt` on the host.

## Doc added

New: `Docs/paddledetection_host_vs_container_paths.md` — explains
the host-vs-container split, the merge order in compose (env_file
loses to environment), what NOT to do, and how to fix path failures
on either side.

## No code changes needed

The path layer was already clean. The only addition is the
host-vs-container doc, which makes the contract explicit for
operators who arrive later.

Proceed to Phase 2.
