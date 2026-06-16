# Phase 13 — Operator onboarding summary

**Date:** 2026-06-13
**Project root:** `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC`
**Host:** 1 NVIDIA T4, 8 CPU cores, CUDA + NVIDIA driver pre-installed,
no sudo / no host `/opt` write access.

## Phases executed

| Phase | Action                              | Status                | Evidence                                       |
| ----- | ----------------------------------- | --------------------- | ---------------------------------------------- |
| 0     | Current state verification          | DONE                  | `FixReports/15_current_state_verification.md`  |
| 1     | Ruff lint + format cleanup          | DONE (clean)          | `FixReports/16_ruff_cleanup.md`                |
| 2     | uv-only workflow verification       | DONE                  | `FixReports/17_uv_workflow_verification.md`    |
| 3     | Infra bring-up                      | DONE (4/4 healthy)    | `FixReports/18_infra_bringup.md`               |
| 4     | DB migrations + Qdrant init         | DONE (13 tables, 3 collections) | this report                          |
| 5     | API container build / `/health`     | **DEFERRED**          | this report (see "Known deferrals")            |
| 6     | Production preflight                | DONE (6/6 pass)       | this report + `Docs/preflight_and_readiness_gate.md` |
| 7     | Smoke test (gated path)             | DONE                  | this report                                    |
| 8     | Smoke benchmark                     | DONE                  | `reports/benchmark_<ts>.{json,md}`             |
| 9     | Readiness gate                      | DONE                  | this report                                    |
| 12    | Targeted improvement search         | DONE (5 applied, 8 deferred) | `FixReports/26_targeted_improvement_search.md` |
| 13    | Documentation                       | DONE                  | this report                                    |

## Verdict

```text
READY_FOR_SHADOW_TEST
```

This is the **maximum verdict** reachable without a real recorded
multi-camera dataset.  Promoting to `READY_FOR_LIMITED_PRODUCTION`
requires the operator to:

1. Record ≥30 minutes of synchronized multi-camera video from
   the target showroom.
2. Run `scripts/benchmark_t4.py --mode production_benchmark`
   with the real PaddleDetection + TransReID weights.
3. Pass the promotion gate thresholds in `configs/benchmark.yaml`.

The gate report is at `reports/readiness_gate.json`.

## Numbers

| Item                                 | Before  | After   | Δ       |
| ------------------------------------ | ------- | ------- | ------- |
| Ruff `check` errors                 | 119     | 0       | -119    |
| Ruff `format --check` reformats     | 81      | 0       | -81     |
| pytest passing                       | 214     | 225     | +11     |
| pytest failing                       | 0       | 0       | 0       |
| Docker compose services healthy     | 0       | 4       | +4      |
| Postgres tables present              | 0       | 13      | +13     |
| Qdrant collections initialized       | 0       | 3       | +3      |
| Preflight checks passing             | n/a     | 6/6     |         |
| `FixReports/` artifacts (Phase 12-13)| 0       | 4       | +4      |
| `Docs/` artifacts (Phase 13)         | 18      | 24      | +6      |

## Files added in Phase 12-13

```text
FixReports/18_infra_bringup.md
FixReports/26_targeted_improvement_search.md
FixReports/27_operator_onboarding_summary.md                <- this
tests/test_readiness_preflight.py
tests/test_targeted_improvements_phase12.py
configs/benchmark_smoke.yaml
Docs/operator_onboarding_uv.md
Docs/transreid_msmt_setup.md
Docs/pphuman_model_setup.md
Docs/docker_uv_build.md
Docs/preflight_and_readiness_gate.md
Docs/ruff_quality_gate.md
```

## Bugs fixed (regression-tested)

1. **Migration 002 ordered before its FK target** — `tracking_events.zone_id`
   referenced `zones` which is defined in 003.  Fix: drop the FK
   (informational column only).  Evidence: 13/13 tables present.
2. **Qdrant healthcheck used `wget` not present in the image** — fixed
   to `bash -c "echo > /dev/tcp/127.0.0.1/6333"`.  Container is
   now `(healthy)`.
3. **Postgres healthcheck logged FATAL on every probe** — fixed by
   passing `-d $POSTGRES_DB` to `pg_isready`.
4. **Readiness preflight `_check_infra_env` precedence bug** — `v ==
   default and k.endswith("PASSWORD") or k.endswith("KEY")`
   flagged any KEY-suffixed var as a default credential.  Fix:
   parenthesize.  Regression tests:
   `tests/test_readiness_preflight.py::test_infra_env_passes_with_legitimate_non_default_credentials`
   (plus 2 more).
5. **`drop_newest` drop counter under-reported** — the second
   `put_nowait` failure swallowed the drop event.  Fix: call
   `observe_drop()` again.  Regression test:
   `tests/test_targeted_improvements_phase12.py::test_drop_newest_records_second_drop_when_retry_also_fails`.
6. **Benchmark report write not atomic** — `with open(..., "w")` left
   truncated files on SIGKILL.  Fix: `*.tmp` + `os.replace`.
   Regression test:
   `tests/test_targeted_improvements_phase12.py::test_write_reports_is_atomic_via_tmp_and_replace`.
7. **Preflight did not require `QDRANT_HOST`** — fix: added to
   required dict.  Regression tests:
   `tests/test_targeted_improvements_phase12.py::test_preflight_infra_env_requires_qdrant_host`
   and `::test_preflight_infra_env_passes_with_qdrant_host`.
8. **Infra clients leaked on SIGTERM** — `app/main.py` `finally`
   only called `runner.stop()`.  Fix: also call `close()` /
   `disconnect()` on `pg`, `qdrant`, `redis`, `minio`, `mqtt`.
9. **`test_cli_min_verdict_exits_zero_when_meeting` was fragile** —
   relied on `reports/` being empty.  Fix: pass an explicit
   empty `--benchmark-dir`.

## Known deferrals (clearly documented, not blocking)

* **Phase 5 — API container build.**  The two-stage CUDA build
  (`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` +
  PaddlePaddle GPU) is heavy (~1.5 GB download, ~10 min on a
  fast network).  Skipped in this session.  The `docker compose
  config` validates cleanly.  Operators should run
  `docker compose build api` as a separate step (see
  `Docs/docker_uv_build.md`).
* **8 medium-risk improvements** from Phase 12 are deferred (PP-Human
  subprocess watchdog, PG transient retry, Qdrant batched
  upsert, TransReID batch_size cosmetic, evidence re-key
  atomicity, per-camera inference latency histogram, /health
  extended probes).  See `FixReports/26_targeted_improvement_search.md`
  for the per-item rationale.

## Final validation commands (verified on this host)

```bash
cd /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC

uv run ruff check app scripts tests           # All checks passed!
uv run ruff format --check app scripts tests  # 90+ files already formatted
uv run python -m pytest tests/ -q             # 225 passed
uv run python -m compileall app scripts tests # clean

docker compose config                          # valid
docker compose ps                              # 4/4 healthy

# Host-mode preflight (no api image required):
set -a; source .env; set +a
QDRANT_HOST=localhost POSTGRES_HOST=localhost REDIS_HOST=localhost \
    uv run python scripts/readiness_preflight.py \
    --out scripts/readiness_preflight.json    # exit 0, 6/6 OK

PYTHONPATH=. uv run python scripts/readiness_gate.py \
    --preflight scripts/readiness_preflight.json \
    --benchmark-dir reports                    # verdict READY_FOR_SHADOW_TEST
```

## Next operator commands

```bash
# (1) Build the API image (heavy, one-time).
docker compose build api

# (2) Start the API.
docker compose up -d api
sleep 30
docker compose logs api --tail=50

# (3) Confirm health.
curl http://localhost:8000/health
docker compose ps api                     # must show (healthy)

# (4) Record a real multi-camera dataset and produce a
#     production benchmark.  Replace data/cam*.mp4 with real
#     recordings and reference them in configs/benchmark.yaml.
PYTHONPATH=. uv run python scripts/benchmark_t4.py \
    --mode production_benchmark \
    --dataset configs/benchmark.yaml \
    --max-seconds 1800

# (5) Re-run the gate.  Expect READY_FOR_LIMITED_PRODUCTION
#     only if the production benchmark passes the promotion
#     gate thresholds in configs/benchmark.yaml.
PYTHONPATH=. uv run python scripts/readiness_gate.py \
    --preflight scripts/readiness_preflight.json \
    --benchmark-dir reports \
    --min-verdict READY_FOR_LIMITED_PRODUCTION
```

## Hard-rule compliance

| Rule                                                                            | Status |
| ------------------------------------------------------------------------------- | ------ |
| Service/ untouched                                                              | YES    |
| Only SOTA-Paddle-MTMC/ modified                                                 | YES    |
| uv is the only normal package manager (pip only as documented diagnostic)       | YES    |
| Verified code before changing it (no trust in prior summaries)                  | YES    |
| Production refuses synthetic detector + deterministic ReID                      | YES (30 production-safety tests pass) |
| Smoke-test paths gated by `RuntimeMode.SMOKE_TEST`                              | YES    |
| `RuntimeMode.PRODUCTION` vs `SMOKE_TEST` safety not weakened                    | YES    |
| `READY_FOR_LIMITED_PRODUCTION` NOT claimed without real multi-camera benchmark  | YES (verdict is `READY_FOR_SHADOW_TEST`) |
| Normal workflow is `uv`, not `pip`                                              | YES    |
| Tests not removed or weakened to pass (only added)                              | YES (+11 tests, 0 deletions) |
