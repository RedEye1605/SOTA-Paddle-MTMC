# Phase 0 — Baseline Verification (Real PP-Human Benchmark)

Date: 2026-06-13
Branch: people-detection

## Commands

```bash
cd /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC

uv run ruff check app scripts tests
uv run ruff format --check app scripts tests
uv run python -m pytest tests/ -q
uv run python -m compileall app scripts tests
docker compose config
docker compose ps
```

## Results

- `ruff check` — All checks passed!
- `ruff format --check` — 111 files already formatted
- `pytest` — **360 passed, 7 warnings in 50.61s**
  - Warnings are `PytestUnhandledThreadExceptionWarning` from smoke tests
    that intentionally start `pphuman-CAM_0X` workers pointing at
    `stub://cam0X`. They are expected for smoke tests and tests pass.
- `compileall` — Listed all packages, no syntax errors.
- `docker compose config` — Valid, output JSON without errors.
- `docker compose ps` — All 5 services (api, minio, postgres, qdrant, redis) healthy.

## State Confirmation

- Ruff clean.
- 360 tests passed.
- Docker compose config passes.
- API + infra containers healthy.
- `data/cam1_merged.mp4` and `data/cam2_merged.mp4` already copied in earlier phases.
- 3000-frame visualization for CAM_01 and CAM_02 exists in smoke mode.

## Verdict

Baseline OK. Proceed to Phase 1.
