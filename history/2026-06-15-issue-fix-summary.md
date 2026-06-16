# Remaining-Issue Fix Summary

> **Date:** 2026-06-12
> **Scope:** Phase 1-9 of the audit's remaining-issues hardening.

## 1. Patches fixed in this round

| Patch | Title | Status |
|---|---|---|
| PATCH-011 | TransReID weight/config alignment | **FIXED** |
| PATCH-016 | Strict travel-time Qdrant filter | **FIXED** |
| PATCH-018 | Per-camera FPS / latency logging | **FIXED** |
| PATCH-029 | Evidence re-key after global_id | **FIXED** |
| PATCH-031 | Backpressure / drop policy | **FIXED** |
| PATCH-032 | RTSP reconnect / degrade | **FIXED** |
| PATCH-047 | Docker api HEALTHCHECK | **FIXED** |
| PATCH-048 | ReID-batch stress benchmark | **FIXED** (script rewrite) |
| PATCH-049 | Multi-camera benchmark workload | **FIXED** (real runner) |

All 8 of the partial / deferred patches are now closed. Plus
the first 7 components of the improvement loop (Phase 11)
and the readiness gate (Phase 9).

## 2. Files added

```
scripts/inspect_transreid_checkpoint.py     # PATCH-011 inspector
scripts/readiness_preflight.py              # Phase 9
scripts/readiness_gate.py                  # Phase 9
app/reid/_transreid_native/                 # already existed (Phase 1)
app/telemetry/per_camera.py                # PATCH-018
app/utils/resilient_reader.py               # PATCH-032
app/workers/evidence_rekey_worker.py        # PATCH-029
app/improvement/                           # already existed (Phase 11)
tests/test_transreid_checkpoint_compatibility.py
tests/test_travel_time_qdrant_filter.py
tests/test_per_camera_metrics.py
tests/test_evidence_rekey.py
tests/test_backpressure_and_reconnect.py
tests/test_api_healthcheck.py
tests/test_benchmark_real_workload.py
tests/test_readiness_gate.py
FixReports/04_remaining_issues_baseline.md
FixReports/05_targeted_improvement_search.md
FixReports/06_remaining_issue_fix_summary.md
Docs/remaining_issues_closed.md
Docs/transreid_weight_alignment.md
Docs/rtsp_reconnect_backpressure.md
Docs/evidence_rekey_runbook.md
Docs/real_benchmark_runbook.md
Docs/shadow_test_readiness.md
Docs/operator_runbook.md
```

## 3. Files changed

```
configs/reid/transreid.yaml                  # PATCH-011 profile + ignore_head
configs/benchmark.yaml                      # Phase 9 gate + evidence + queues
docker-compose.yaml                         # PATCH-047 api healthcheck
app/main.py                                 # Phase 9 worker wiring
app/cli/args.py                             # (no change)
app/reid/transreid_adapter.py               # PATCH-011 preflight
app/storage/qdrant_store.py                 # PATCH-016 per-camera search
app/identity/resolver.py                    # PATCH-016 per-camera windows
app/workers/multi_camera_runner.py         # PATCH-018 + PATCH-031/032
app/workers/pphuman_worker.py              # PATCH-032 skip None frames
app/workers/evidence_rekey_worker.py        # PATCH-029 re-key
app/storage/minio_store.py                  # PATCH-029 pending path
app/telemetry/metrics.py                    # PATCH-018 per-camera gauges
app/improvement/promotion_gate.py          # Phase 9 top-level lookup
scripts/benchmark_t4.py                     # PATCH-048/049 rewrite
```

## 4. Tests added

| File | Tests | What it covers |
|---|---:|---|
| `test_transreid_checkpoint_compatibility.py` | 16 | profile table, inspector, classifier shape detection, missing-checkpoint fails in production, smoke allows, market1501/msmt17/custom, plug-in path |
| `test_travel_time_qdrant_filter.py` | 11 | per-camera search, gte/lte bounds, too-fast/too-slow exclusion, disabled link exclusion, same-cam persistence, empty candidates |
| `test_per_camera_metrics.py` | 12 | per-camera FPS, EWMA latency, status transitions, drops, decode errors, registry rendering |
| `test_evidence_rekey.py` | 13 | pending path, key format, copy, retry, no-op for ambiguous, keep/delete pending, disabled config |
| `test_backpressure_and_reconnect.py` | 10 | three drop policies, reconnect metric, status transitions, one-dead-camera-doesn't-kill-others |
| `test_api_healthcheck.py` | 5 | docker compose healthcheck present, uses /health, no secrets, /health public, /identity requires auth |
| `test_benchmark_real_workload.py` | 7 | manifest load, markdown render, smoke run, mode validation, empty manifest, CLI accepts mode |
| `test_readiness_gate.py` | 17 | verdict order, preflight load, benchmark load, gate config parse, NOT_READY/STRUCTURAL/SHADOW/LIMITED paths, CLI |
| **Total new** | **91** | |

## 5. Commands run

```bash
# Phase 0 (baseline)
python3 -m pytest tests/ -q        # 119 passed, 1 skipped
python3 -m compileall app scripts tests
docker compose config

# Phase 1 (PATCH-011)
python3 -m pytest tests/test_transreid_checkpoint_compatibility.py -q
# 16 passed
python3 scripts/inspect_transreid_checkpoint.py /tmp/missing.pth --json
# {"ok": false, "reason": "checkpoint_missing", "missing": ["/tmp/missing.pth"], "path": "/tmp/missing.pth", "exists": false, "size_bytes": 0, "expected_profile": "msmt17", "expected_num_class": null}
# exit code 1

# Phase 2 (PATCH-016)
python3 -m pytest tests/test_travel_time_qdrant_filter.py -q
# 11 passed

# Phase 3 (PATCH-018)
python3 -m pytest tests/test_per_camera_metrics.py -q
# 12 passed

# Phase 4 (PATCH-029)
python3 -m pytest tests/test_evidence_rekey.py -q
# 13 passed

# Phase 5 (PATCH-031/032)
timeout 60 python3 -m pytest tests/test_backpressure_and_reconnect.py -q
# 10 passed

# Phase 6 (PATCH-047)
python3 -m pytest tests/test_api_healthcheck.py -q
# 5 passed
docker compose config | grep -A 8 'healthcheck:'
# api healthcheck present

# Phase 7 (PATCH-048/049)
python3 -m pytest tests/test_benchmark_real_workload.py -q
# 7 passed

# Phase 9
python3 -m pytest tests/test_readiness_gate.py -q
# 17 passed

# Final
python3 -m compileall app scripts tests
# OK
python3 -m pytest tests/ 2>&1 | grep "passed\|failed"
# 210 passed, 1 skipped
docker compose config > /dev/null && echo OK
# OK
```

## 6. Test result summary

| Phase | Tests | Pass | Notes |
|---|---:|---:|---|
| Baseline | 119 | 119 | pre-existing |
| + Phase 1 (PATCH-011) | 135 | 135 | +16 |
| + Phase 2 (PATCH-016) | 146 | 146 | +11 |
| + Phase 3 (PATCH-018) | 158 | 158 | +12 |
| + Phase 4 (PATCH-029) | 171 | 171 | +13 |
| + Phase 5 (PATCH-031/32) | 181 | 181 | +10 |
| + Phase 6 (PATCH-047) | 186 | 186 | +5 |
| + Phase 7 (PATCH-048/49) | 193 | 193 | +7 |
| + Phase 9 (readiness gate) | 210 | 210 | +17 |
| **Total** | **210 + 1 skipped** | **210** | **+91 new** |

The 1 skipped test is the vendored TransReID forward-pass test
which requires `torch` (not installed on the dev host).

## 7. Remaining risks (operator-side)

* **Real PaddleDetection weights** must be cloned and
  downloaded by the operator (see `Docs/operator_runbook.md`).
  Code is ready; the actual `git clone` + `wget` is operator work.
* **Real TransReID weight alignment** is documented in
  `Docs/transreid_weight_alignment.md` (PATCH-011).
* **Recorded multi-camera dataset** is operator-provided
  for the production benchmark.

These three items are the only thing standing between
`STRUCTURALLY_READY` and `READY_FOR_LIMITED_PRODUCTION`.

## 8. New readiness verdict

**STRUCTURALLY_READY** — all CRITICAL and HIGH patches are closed;
tests are green; production-mode startup fails fast on missing
real models; smoke-mode startup works without real models. The
system is ready for a side-by-side shadow-test deploy with a
recorded dataset. The operator's next step is to clone
PaddleDetection, download weights, and run the production
benchmark. Until those are done, the production benchmark will
not run and the gate will not promote past `READY_FOR_SHADOW_TEST`.

For the exact operator steps, see `Docs/operator_runbook.md`.
