# Preflight and readiness gate

> **Two scripts.  One produces a structured JSON of environment
> checks; the other consumes that JSON plus the latest benchmark
> report and emits one of four verdicts.**

## Preflight: `scripts/readiness_preflight.py`

Checks (six total):

| Name              | Verifies                                              |
| ----------------- | ----------------------------------------------------- |
| `sota_api_token`  | `SOTA_API_TOKEN` is set AND not the docs default      |
| `transreid_weight`| When `reid.active_model=transreid`, the weight file exists |
| `pphuman_pipeline`| The PaddleDetection `pipeline.py` is on disk         |
| `infra_env`       | `POSTGRES_HOST/USER/PASSWORD`, `QDRANT_HOST`, `MINIO_ACCESS_KEY/SECRET_KEY`, `REDIS_HOST` set AND not defaults |
| `docker_compose`  | `docker compose config --quiet` returns 0            |
| `benchmark_dir`   | `reports/` exists (created if missing)               |

### Run via docker compose (recommended)

```bash
docker compose run --rm detect-pipeline python scripts/readiness_preflight.py \
    --out scripts/readiness_preflight.json
```

### Run from the host (for diagnostics)

```bash
set -a; source .env; set +a
QDRANT_HOST=localhost POSTGRES_HOST=localhost REDIS_HOST=localhost \
    uv run python scripts/readiness_preflight.py \
    --out scripts/readiness_preflight.json
```

Expected output (verified on this host, 2026-06-13):

```text
[OK  ] sota_api_token: len=64
[OK  ] transreid_weight: active_model='pphuman_strongbaseline'; skipped
[OK  ] pphuman_pipeline: pipeline='/home/rhendy/paddledetection/deploy/pipeline/pipeline.py'
[OK  ] infra_env: all env vars present
[OK  ] docker_compose: valid
[OK  ] benchmark_dir: path=/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC/reports
```

### Exit code

```text
0  -> all checks passed
1  -> at least one check failed
```

## Readiness gate: `scripts/readiness_gate.py`

### Verdicts (strictly increasing)

```text
0  NOT_READY                          preflight OR tests failed
1  STRUCTURALLY_READY                 preflight + tests pass, no benchmark
2  READY_FOR_SHADOW_TEST              smoke_benchmark report present
3  READY_FOR_LIMITED_PRODUCTION       production_benchmark + promotion gate pass
```

### Run

```bash
docker compose run --rm detect-pipeline python scripts/readiness_gate.py \
    --preflight scripts/readiness_preflight.json \
    --benchmark-dir reports \
    --out reports/readiness_gate.json

# OR from the host:
PYTHONPATH=. uv run python scripts/readiness_gate.py \
    --preflight scripts/readiness_preflight.json \
    --benchmark-dir reports \
    --out reports/readiness_gate.json
```

### Verified verdict on this host (2026-06-13)

```json
{
  "verdict": "READY_FOR_SHADOW_TEST",
  "reasons": [
    "preflight + tests pass; STRUCTURALLY_READY",
    "production_benchmark: benchmark: mode='smoke_benchmark' (need 'production_benchmark' for READY_FOR_LIMITED_PRODUCTION)",
    "smoke_benchmark ran; READY_FOR_SHADOW_TEST"
  ],
  "failures": []
}
```

### `--min-verdict` flag (CI gate)

```bash
PYTHONPATH=. uv run python scripts/readiness_gate.py \
    --preflight scripts/readiness_preflight.json \
    --benchmark-dir reports \
    --min-verdict READY_FOR_SHADOW_TEST
# exit 0 if verdict >= READY_FOR_SHADOW_TEST, else exit 1
```

## Hard rules (enforced by tests)

```text
1. Never claim READY_FOR_LIMITED_PRODUCTION based on a
   smoke benchmark.  The gate refuses it.
2. A production_benchmark report that fails the promotion gate
   does NOT silently fall back to READY_FOR_SHADOW_TEST — it
   becomes NOT_READY.
3. The preflight refuses default credentials
   (`change_me_in_production`) and refuses an empty
   SOTA_API_TOKEN.
4. The /health endpoint is NOT a substitute for the preflight.
   The preflight covers env-var hygiene; /health covers
   live dependency probes.
```

## Tests

Regression coverage:

```text
tests/test_readiness_gate.py          12 tests
tests/test_readiness_preflight.py      7 tests
tests/test_targeted_improvements_phase12.py  4 tests
```

## Improvements applied in Phase 12

* **Precedence bug fix** in `_check_infra_env` — previously any
  env var ending in `KEY` was flagged as a "default credential"
  regardless of its value.  Fix: group the boolean correctly.
* **Coverage extension** — `QDRANT_HOST` is now required.
* **Atomic benchmark write** in `_write_reports` — writes
  `*.tmp` and `os.replace`s.  A SIGKILL mid-write no longer
  truncates the JSON the gate reads.
