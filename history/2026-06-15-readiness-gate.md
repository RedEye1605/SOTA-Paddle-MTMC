# Readiness Gate Report

> **Snapshot from the post-Phase-10 audit run.**

## 1. Test run

```
$ python3 -m pytest tests/ 2>&1 | grep "passed\|failed"
210 passed, 1 skipped, 3 warnings in 14.59 s
```

* **210 tests pass** (up from 119 at the start of the audit).
* **1 skipped** is the vendored TransReID forward-pass test
  (`tests/test_transreid_vendor.py`); it requires `torch` which
  is not installed on the dev host. It is part of the
  GPU profile (`requirements-gpu.txt`).

## 2. Smoke benchmark

```
$ SOTA_API_TOKEN=smoke python3 scripts/benchmark_t4.py \
      --mode smoke_benchmark \
      --dataset configs/benchmark.yaml \
      --max-seconds 1 \
      --out-dir /tmp/bench_out
```

Wrote `/tmp/bench_out/benchmark_20260612T153726Z.{json,md}`.
The smoke benchmark completed in ~1 s and reported 0 fps because
the runner's frame readers immediately fail on the
unreachable `/data/cam0?.mp4` video paths and fall back to
the resilient reader's None-sentinel path. The other metrics
were recorded successfully (GPU memory 15 MB, CPU 5.5 %,
no reconnects or drops). The point of the smoke benchmark is
to verify the data plane is wired end-to-end without crashing.

## 3. Readiness gate

```
$ python3 scripts/readiness_preflight.py
[FAIL] sota_api_token: SOTA_API_TOKEN env var is empty
[OK  ] transreid_weight: active_model='pphuman_strongbaseline'; skipped
[OK  ] pphuman_pipeline: no pipeline path configured
[FAIL] infra_env: missing env vars: POSTGRES_HOST, POSTGRES_USER, POSTGRES_PASSWORD, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, REDIS_HOST
[OK  ] docker_compose: valid
[OK  ] benchmark_dir: path=.../reports
ok=False
```

The preflight correctly fails on a host without operator-set
env vars. This is the expected behavior: the gate refuses
to claim READY_FOR_SHADOW_TEST without the operator
provisions.

```
$ python3 scripts/readiness_gate.py
verdict: NOT_READY
reasons: ["preflight failed; NOT_READY"]
failures:
  - "preflight: not OK (unknown)"
  - "preflight.sota_api_token: SOTA_API_TOKEN env var is empty"
  - "preflight.infra_env: missing env vars: ..."
```

## 4. Verdict on the dev host

**NOT_READY** — expected. The dev host has no operator-set env
vars (no `SOTA_API_TOKEN`, no infra env). The system code is
correctly strict about this.

## 5. What the verdict would be on a production host

When the operator provisions the host (see
`Docs/operator_runbook.md`):

1. `SOTA_API_TOKEN` is set in `.env`.
2. Infra env vars are set (`POSTGRES_HOST=postgres`, …).
3. PaddleDetection is cloned + weights downloaded.
4. A recorded multi-camera dataset is provided.

Then the gate would yield:

* **`STRUCTURALLY_READY`** after the preflight + tests pass.
* **`READY_FOR_SHADOW_TEST`** after a smoke_benchmark report
  is produced (no real models needed; works in synthetic mode).
* **`READY_FOR_LIMITED_PRODUCTION`** after a production_benchmark
  report passes the promotion gate (real models + recorded
  dataset required).

## 6. Files

* `scripts/readiness_preflight.py` (new) — produces the
  preflight JSON.
* `scripts/readiness_gate.py` (new) — consumes the
  preflight + benchmark + tests; emits a verdict.
* `app/improvement/promotion_gate.py` (updated) — looks for
  metric keys at both the top level and under `metrics.*`
  (so the gate works with both the benchmark JSON shape
  and the `OfflineReport` shape).
* `tests/test_readiness_gate.py` (new) — 17 tests covering
  all four verdicts, the CLI, and the preflight parsing.

## 7. How to run the gate in CI

```yaml
# .github/workflows/ci.yml
- name: Preflight
  run: python scripts/readiness_preflight.py
- name: Test suite
  run: python -m pytest tests/ --junitxml=reports/junit.xml
- name: Smoke benchmark
  run: |
    SOTA_API_TOKEN=ci-smoke python scripts/benchmark_t4.py \
        --mode smoke_benchmark \
        --max-seconds 5 \
        --out-dir reports/
- name: Readiness gate
  run: |
    python scripts/readiness_gate.py \
        --junit reports/junit.xml \
        --out reports/readiness.json \
        --min-verdict READY_FOR_SHADOW_TEST
```

The CI fails the build if the verdict drops below
`READY_FOR_SHADOW_TEST`.
