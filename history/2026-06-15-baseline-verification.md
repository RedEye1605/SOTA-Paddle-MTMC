# Phase 0 — Baseline Verification Report

> **Date:** 2026-06-12
> **Scope:** Audit SOTA-Paddle-MTMC against the audit findings.
> **Method:** Run audit-recommended commands; do not trust test counts.

## Commands run and outputs

### 1. `python3 -m pytest tests/ -q`

```
....................................................................     [100%]
68 passed in 0.74s
```

The 68 tests pass and execute in <1 s. They are unit tests; none of them
exercise real Paddle, real TransReID, real Qdrant, real PG, real Redis, or
real MinIO. This is consistent with `TEST_QUALITY_AUDIT.md`.

### 2. `python3 -m compileall app scripts tests`

Compiles cleanly. No syntax errors.

### 3. `docker compose config`

Parses cleanly. The migrator is missing `--single-transaction`
(PATCH-013). All ports are exposed on `0.0.0.0` (S3).

### 4. `python3 -c "from app.storage.postgres import PostgresStore"`

**FAILS** with the expected error:
```
ImportError: cannot import name 'pool' from 'psycopg'
```
This is **PATCH-001** (psycopg ≥ 3.2 moved `pool` to `psycopg_pool`).

### 5. `grep -E "paddle|torch|psycopg_pool" requirements.txt`

**ZERO matches.** PaddlePaddle, PyTorch, and psycopg_pool are missing.
This is **PATCH-005**.

### 6. `grep -R "fallback|synthetic|histogram|smoke|fake|quality_score" app tests scripts configs`

Hits found:
- `app/reid/pphuman_adapter.py:42,47,68,81,82,86,87,91,107` — histogram fallback
- `app/reid/transreid_adapter.py:44,46,71,84,85,88,89,101` — histogram fallback
- `app/reid/clipreid_adapter_optional.py:24,28,49` — histogram fallback (always-on)
- `app/workers/pphuman_worker.py:83-109` — synthetic random-box detector
- `app/workers/reid_worker.py:66` — fabricates crops from `quality_score`
- `app/workers/tracklet_collector.py:160` — `tl.quality_score = 0.6` placeholder
- `app/identity/resolver.py:10` — Stage 3 (24h fallback) is documented but
  not implemented
- `app/main.py:8,189,194,212-218,260` — `single_cam_smoke` is the only path
  in production
- `app/storage/qdrant_store.py:38,120,135` — `quality_score` field/index
  (real usage, not fake)
- `app/identity/scoring.py` — `quality_score` in the 5-factor score (real)

### 7. `grep -R "xreadgroup|consume|stream:tracklets|stream:embeddings" app`

Hits found:
- `app/storage/redis_state.py:149-158` — `consume()` is implemented
- `app/workers/reid_worker.py:124-148` — consumes `stream:tracklets` (real)
- `app/workers/telemetry_worker.py:78-92` — consumes `stream:identity_decisions`
  and `stream:zone_events` (but nothing publishes — **PATCH-006** and
  **PATCH-010**)
- `app/identity/resolver.py:3,57` — documents that it should consume
  `stream:embeddings` but has **no** `run()` / `consume()` method — this is
  **PATCH-006** in its raw form.

### 8. `grep -R "final_score" app tests`

`final_score` is computed in `app/identity/scoring.py` and
`app/identity/resolver.py:195,260` and is **persisted to
`identity_decisions`**. But the actual `decide_ambiguity()` function in
`app/identity/ambiguity.py:34-71` decides based on
`top1.score` (ReID cosine), not the weighted `final_score`. This is
**PATCH-008** confirmed.

## Summary of baseline findings

| ID | Severity | Status | Location |
|---|---|---|---|
| PATCH-001 | CRITICAL | CONFIRMED | `app/storage/postgres.py:15` (psycopg_pool import) |
| PATCH-002 | CRITICAL | CONFIRMED | `app/workers/pphuman_worker.py:83-109,197-200` (synthetic detector) |
| PATCH-003 | CRITICAL | CONFIRMED | `app/reid/{pphuman,transreid,clipreid}_adapter.py` (histogram fallback) |
| PATCH-004 | CRITICAL | CONFIRMED | `app/workers/reid_worker.py:62-69` (fabricated crops) |
| PATCH-005 | CRITICAL | CONFIRMED | `requirements.txt` (no paddle/torch/psycopg_pool) |
| PATCH-006 | CRITICAL | CONFIRMED | `app/identity/resolver.py` (no `run` consumer) |
| PATCH-007 | HIGH | CONFIRMED | `app/workers/multi_camera_runner.py:79-94` (no shared model) |
| PATCH-008 | HIGH | CONFIRMED | `app/identity/ambiguity.py:34-71` (final_score unused) |
| PATCH-009 | HIGH | PARTIAL | `app/workers/tracklet_collector.py:160` (placeholder quality) |
| PATCH-010 | HIGH | CONFIRMED | `app/workers/telemetry_worker.py:78-89` (no publishers) |
| PATCH-011 | HIGH | PARTIAL | `models/vit_transreid_msmt.pth` is MSMT17, config says Market-1501 |
| PATCH-012 | HIGH | CONFIRMED | `app/workers/reid_worker.py:79` (point_id not namespaced) |
| PATCH-013 | HIGH | CONFIRMED | `docker-compose.yaml:81-85` (no --single-transaction) |
| PATCH-014 | HIGH | CONFIRMED | `app/api/server.py` (no auth) |
| PATCH-015 | HIGH | CONFIRMED | (no retention worker anywhere) |
| PATCH-021 | HIGH | CONFIRMED | `Dockerfile` (no paddle/torch) |
| PATCH-022 | HIGH | CONFIRMED | `requirements.txt` (no psycopg_pool) |
| PATCH-037 | LOW | CONFIRMED | `tests/test_architecture_guards.py` (no one-model-per-process) |
| PATCH-040 | LOW/MED | CONFIRMED | `app/reid/*_adapter.py:42` (WARNING, not ERROR) |

## What is already correct (do not touch)

- Qdrant: `init_collections`, payload indexes, filtered `search` — all real
- Redis: TTL keys, Streams, consumer groups — all real
- PostgreSQL: migrations, FKs, indexes — all real (one import broken — PATCH-001)
- MinIO: bucket ensure, deterministic path scheme, JPEG upload — all real
- MQTT: ThingsBoard `{ts, values}` payload format — correct
- FastAPI: endpoints implemented (but no auth — PATCH-014)
- `tracker_config.yml`: matches the official Paddle OC-SORT keys

## Decisions for the fix plan

1. **Fix the import chain first** (PATCH-001, PATCH-005) so the system can
   import. Required for all other tests.
2. **Implement a real Paddle adapter boundary** that loads the PP-Human
   `pipeline.py` as a child process — this is the official path documented
   in `PaddleDetection/deploy/pipeline/`. The "model" is then `pipeline.py`,
   not Python.
3. **Implement a real TransReID adapter** using the upstream
   `vit_base_patch16_224_TransReID` (vendor a minimal copy).
4. **Wire the resolver** to consume `stream:embeddings` as a real worker.
5. **Production-mode startup guard** that refuses to start if ReID/detector
   are in fallback state.
6. **Final-score drives the decision** in `decide_ambiguity`.
7. **Shared model object** between workers.
8. **FastAPI auth** + **retention worker** + **CHECK constraint on
   `decision_type`** (PATCH-030) + other quick wins.
9. **Integration tests** against real Redis (the only one available
   locally without Docker), real Qdrant (also local), real MinIO
   (also local). Tests are skipped if infra is unavailable.
10. **Improvement loop skeleton** (Phase 11): evidence sampler, manifest,
    offline evaluator skeleton, promotion gate, ambiguous export.
