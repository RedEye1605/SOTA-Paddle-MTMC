# Phase 2 — uv-only Workflow Verification

**Date:** 2026-06-13
**Project root:** `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC`

## Verification commands run

```text
uv sync --frozen --extra dev
uv run python -m pytest tests/ -q
uv run python -m compileall app scripts tests
uv run ruff check app scripts tests
uv run ruff format --check app scripts tests
```

## Results

| Check                            | Result                                | Pass |
| -------------------------------- | ------------------------------------- | ---- |
| `uv.lock` present                | yes (198 KB)                          | YES  |
| `uv sync --frozen --extra dev`   | `Checked 83 packages in 0.89ms`        | YES  |
| pytest                           | `214 passed, 3 warnings, 28.90s`      | YES  |
| compileall                       | clean                                 | YES  |
| ruff check                       | `All checks passed!`                  | YES  |
| ruff format --check              | `90 files already formatted`          | YES  |

## Search for pip usage in normal workflow

```text
grep -rn "pip install|python3 -m pip|pip3 " README.md Docs/ scripts/ docker-compose.yaml Dockerfile pyproject.toml
```

Only matches:

```text
pyproject.toml:44:    # CPU torch (`uv pip install torch==<v>+cpu --index-url
```

This is a comment in the `[project.dependencies]` block describing a documented
exception path (operator preinstalls CPU torch on a host without a GPU, for
running the vendored vendor forward-pass test). It is annotated as a
temporary diagnostic command and is **not** the normal workflow. The
`uv pip install …` form is intentional.

No docs, scripts, Dockerfile, docker-compose.yaml, or README instruct
`pip install` for the normal workflow. All current onboarding commands
already use `uv sync` / `uv run`.

## Dockerfile uv pattern (verified)

```text
Layer 1: COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /usr/local/bin/
Layer 2: uv sync --frozen --extra gpu --no-dev --no-install-project
Layer 3: pre-download model weights + clone PaddleDetection
Layer 4: uv sync --frozen --extra gpu --no-dev
```

Two-stage CUDA build using `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` base.

## Legacy `requirements*.txt` files

`requirements.txt` and `requirements-gpu.txt` remain on disk as legacy
compatibility artifacts. Their headers mention `pip install -r requirements.txt`
as a historical install path. They are **not** the normal workflow.

Hard rule: "Normal workflow must be uv" — confirmed satisfied.

## Documentation status

- `README.md` — no `pip install` command (only mentions `uv` and `uv run`).
- `Docs/operator_runbook.md` — no `pip install` command.
- `Docs/architecture.md` — no install commands.
- `Docs/comparison_with_existing_service.md` — no install commands.
- `Docs/shadow_test_readiness.md` — no install commands.
- `Docs/model_selection.md` — no install commands.
- `Docs/research_sources.md` — no install commands.
- `Docs/official_paddle_integration.md` — no install commands.

(Other matches for the substring "pip" are words like "pipeline", not the
package manager.)

## Hard rule compliance

| Rule                                                           | Status |
| -------------------------------------------------------------- | ------ |
| uv.lock exists                                                 | YES    |
| uv sync works                                                  | YES    |
| Tests run through uv                                           | YES    |
| Normal docs/commands use uv, not pip                          | YES    |
| requirements.txt may remain as legacy, not the normal workflow | YES  |

## Verdict

Phase 2 uv workflow verification: PASS. uv is the only normal package-manager
workflow. No docs need to be updated.
