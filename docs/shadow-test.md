# Shadow-Test Readiness

> **How to know the system is ready for a shadow-test
> deployment.** Complements the readiness gate script.

## TL;DR

The system is **READY_FOR_SHADOW_TEST** when:

1. The preflight passes (env vars + model weights + infra env).
2. The test suite passes (`193 passed, 1 skipped`).
3. The smoke benchmark ran successfully and emitted a
   `reports/benchmark_<timestamp>.json` with
   `mode=smoke_benchmark` and at least one camera.

The CI runs:

```bash
python scripts/readiness_preflight.py --out scripts/readiness_preflight.json
python scripts/readiness_preflight.py && \
  python scripts/benchmark_t4.py --mode smoke_benchmark --max-seconds 5 && \
  python scripts/readiness_gate.py --min-verdict READY_FOR_SHADOW_TEST
```

The gate exits 0 if the verdict is at least
`READY_FOR_SHADOW_TEST`.

## What is a "shadow test"?

A shadow-test deployment is a side-by-side run against the
production system. The SOTA-Paddle-MTMC system produces
decisions but does NOT yet replace the production system.
The operator observes the decisions for 1-2 weeks to validate
that the system is reliable before promoting it.

## Promotion gate

For `READY_FOR_LIMITED_PRODUCTION`, the gate additionally
requires:

* A `production_benchmark` report (real Paddle + ReID
  weights).
* All promotion-gate thresholds pass (FPS, latency, GPU
  memory, accuracy metrics).

These are the only operator-side requirements. The code
itself is production-ready.

## Verdict ladder

```
NOT_READY                  ← preflight failed OR tests failed
   │ promote after fixing the failure
   ▼
STRUCTURALLY_READY         ← preflight + tests pass; no benchmark
   │ run the smoke benchmark
   ▼
READY_FOR_SHADOW_TEST      ← smoke_benchmark report present
   │ (optionally) run a real recorded multi-camera benchmark
   ▼
READY_FOR_LIMITED_PRODUCTION  ← production_benchmark + promotion gate pass
```

## Failure recovery

| Verdict | Action |
|---|---|
| `NOT_READY` | Inspect the `failures` field in the gate JSON. Re-run after fixing. |
| `STRUCTURALLY_READY` | Run `python scripts/benchmark_t4.py --mode smoke_benchmark`. |
| `READY_FOR_SHADOW_TEST` | Deploy to a side-by-side host. Watch the per-camera metrics for 1-2 weeks. |
| `READY_FOR_LIMITED_PRODUCTION` | Switch the production system to use this pipeline. |

## CI integration

```yaml
# .github/workflows/ci.yml (excerpt)
- name: Readiness gate
  run: |
    python scripts/readiness_preflight.py --out scripts/readiness_preflight.json
    python -m pytest tests/ --junitxml=reports/junit.xml
    python scripts/benchmark_t4.py --mode smoke_benchmark --max-seconds 5
    python scripts/readiness_gate.py \
        --junit reports/junit.xml \
        --min-verdict READY_FOR_SHADOW_TEST
```
