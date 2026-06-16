# Audit Report — SOTA-Paddle-MTMC

> **Phase 0–11 consolidated audit report.** The full audit
> consists of 10 documents in this folder. This file is the
> executive summary.

## 0. Status update (2026-06-17)

The audit below is the **historical baseline**. The PATCH_PLAN
items it called out (PATCH-001 through PATCH-040) have been
applied across multiple FixReports (`FixReports/2026-06-15_*`,
`FixReports/2026-06-16_*`, `FixReports/2026-06-17_BUGFIXES_PRODUCTION_READINESS.md`,
`FixReports/2026-06-17_PADDLE_DEADLOCK_FIX.md`).

A second pass on 2026-06-17 (the file
`FixReports/2026-06-17_PADDLE_DEADLOCK_FIX.md`) re-applied the
following PATCH items after a partial revert in the working
tree:

| PATCH | What it does | File |
|---|---|---|
| BUG-1 / PATCH-003 | `assert_production_safe` in `extract()` of all 3 ReID adapters | `app/reid/{transreid,pphuman,clipreid}_adapter*.py` |
| BUG-NEW-A | PP-Human stall watchdog (60 s, max 10 restarts) | `app/detection/pphuman_pipeline.py` |
| BUG-NEW-A | `MOT.OCSORTTracker.min_hits=3` | `app/detection/pphuman_pipeline.py` |
| BUG-5 | SYNC MinIO upload so `frame_uri` is populated before the XADD | `app/detection/_vendor/paddledetection_pipeline.py` |
| BUG-1 | `test_transreid_adapter_refuses_extract_when_unloaded_in_production` | `tests/test_production_safety.py` |

Dead code was also cleaned up:

* `app.reid.pphuman_adapter` and `app.reid.clipreid_adapter_optional`
  were demoted to **experimental, off by default** (per the
  operator's transreid-only spec). They are kept as plug-in shells
  for operators who want to vendor a different ReID.
* `scripts/retention_worker.py` was rewritten to iterate the
  single `person_reid_transreid_msmt` collection via
  `app.storage.qdrant_store.COLLECTIONS` instead of the dropped
  pphuman/clipreid/vanilla-transreid collections.
* `_qdrant_collection_for` in the resolver no longer has fallback
  paths for the dropped models.
* `ResolverConfig.model_name` default is now `transreid_msmt`.
* `reid_backend` in the multi-camera overlay is now `transreid_msmt`.
* `MultiCameraRunner` now receives the `identity_overlay_cache`
  (was constructed but never wired).
* `DwellBookkeeper.force_close_stale()` is now called by
  `TelemetryWorker.run()` every 60 s.
* `MultiCameraRunner._streamer_drain_thread` log line moved to
  the correct position (post-population).
* `MultiCameraRunner` no longer silently swallows stream errors.
* `app.api.server` docstring updated; `POST /admin/retention/run`
  is documented as a script-level endpoint, not HTTP.
* `app.workers.telemetry_worker` runs a 60 s force-close-stale
  dwell sweep.
* `app.telemetry.metrics` no longer declares the dead
  `analytics_fps_per_camera`, `tracklet_buffer_size`, or
  `stream_backlog` metrics.
* `tests.test_architecture_guards` no longer ignores `Audit/`
  and `FixReports/` for the secrets check.
* `.gitignore` lists `.env.bak` and `.env.bak.*` so the backup
  env files at the repo root are no longer accidentally staged.
* `ruff check app/ --exclude 'app/detection/_vendor'` is clean.

See `Docs/post_revert_remediation.md` for the per-file change
list and rationale.

## 1. Overall verdict (baseline)

**WAS NOT PRODUCTION-READY. STRUCTURALLY COMPLETE.** The
post-remediation work (above) restores the production safety
nets; the remaining gaps are listed in §11.

The SOTA-Paddle-MTMC implementation has a correct schema, a
correct architecture, a correct (in spec) 5-factor scoring
system, and 68 honest unit tests. **It is a well-built
skeleton that demonstrates how a multi-camera ReID
pipeline should be assembled.** It is not, today, a system
that can be pointed at CCTV streams and trusted to produce
correct global_ids.

The single critical fact: the production detector is a
synthetic random-box generator, and the production ReID
adapters are histogram-based fallbacks. **Every "decision"
the system makes in production today is a decision over
histogram similarity, not real visual ReID.** The audit
flags this as **CRITICAL** in PATCH-002, PATCH-003, and
PATCH-004.

## 2. Top 10 risks (severity-ordered)

| # | Risk | Severity | ID |
|---|---|---|---|
| 1 | The production ReID path is a histogram fallback — every "embedding" is colour-statistics, not real ReID. False merges are highly likely. | CRITICAL | PATCH-003 |
| 2 | The production detector path is a synthetic random-box generator — no Paddle imported, no PaddleInference session, no real model. | CRITICAL | PATCH-002 |
| 3 | The ReID worker fabricates crops from `quality_score` — every embedding is a constant-colour image, not a real person crop. | CRITICAL | PATCH-004 |
| 4 | The `psycopg.pool` import is broken on psycopg ≥ 3.2; the entire Postgres layer fails to import. The system cannot start. | CRITICAL | PATCH-001 |
| 5 | The global identity resolver is never wired to a stream consumer. The 5-factor scoring, the 24h persistence, the topology gating are all dead code in the running system. | CRITICAL | PATCH-006 |
| 6 | No FastAPI auth on identity endpoints — anyone who can reach port 8000 can read the global_id ↔ tracklet ↔ camera chain. | HIGH | PATCH-014 |
| 7 | Multi-camera runner does not share a model. The "one model per process" rule is not wired. With real Paddle, this would N× the VRAM cost. | HIGH | PATCH-007 |
| 8 | No retention policy for evidence crops, Qdrant points, or PG rows. CCTV data grows forever. | HIGH | PATCH-015 |
| 9 | The 5-factor `final_score` is computed but not used for the decision. Topology and margin are checked; the weighted score is not. This contradicts the README and the docs. | HIGH | PATCH-008 |
| 10 | `requirements.txt` is missing `paddlepaddle-gpu` and `psycopg_pool`. A fresh `pip install` will not give a working image. | CRITICAL | PATCH-005 |

## 3. Official compliance score

| Domain | Score | Notes |
|---|---|---|
| PaddleDetection | 0/4 | Config is text; no real engine, no real tracker, no MTMCT module, no TensorRT FP16. |
| PP-Human MTMCT | 0/1 | `mot_sde_infer.py` is referenced in docs but never invoked. |
| ReID (PP-Human + TransReID) | 0/2 | Both adapters are permanent fallbacks. |
| ReID (CLIP-ReID, optional) | 0/1 | Always-fallback stub (honest, marked optional). |
| Qdrant | 3/3 | Init + indexes + filtered search all real. |
| Redis | 5/5 | TTL + Streams + groups all real. |
| PostgreSQL | 5/5 | Migrations + indexes + FKs all real (one import is broken — PATCH-001). |
| MinIO | 2/2 | Bucket + path + upload all real. |
| MQTT ThingsBoard | 1/1 | Payload format correct. |
| FastAPI | 1/1 | Endpoints implemented; no auth (PATCH-014). |
| T4 optimization | 0/2 | No engine, no real benchmark. |
| Multi-camera model sharing | 0/1 | Architecture only. |
| **Overall** | **17/27 = 63%** | **Structurally complete; production paths are stubs.** |

## 4. Critical bugs found

Six CRITICAL bugs are listed above (Risks 1-5 and 10). All
six block production. The 68 passing tests do not catch any
of them because the tests don't exercise the production
code paths — they exercise the deterministic-fallback paths,
which work correctly.

The full bug list (50 findings, 6 CRITICAL, 9 HIGH, 21
MEDIUM, 14 LOW) is in `BUG_REPORT.md` and `PATCH_PLAN.md`.

## 5. Missing tests

The test quality audit (`TEST_QUALITY_AUDIT.md`) finds that:

- 0/68 tests require a GPU.
- 0/68 tests require a real Paddle model.
- 0/68 tests require a real TransReID model.
- 0/68 tests require a real Qdrant.
- 0/68 tests require a real PostgreSQL.
- 0/68 tests require a real Redis.
- 0/68 tests require a real MinIO.
- 0/68 tests require real RTSP or recorded video.
- 0/68 tests cover multi-camera end-to-end (real).
- 0/68 tests cover 24h retrieval.
- 0/68 tests cover production-fallback blocking.
- 0/68 tests cover the synthetic detector.
- 0/68 tests cover Docker Compose startup.

**The 68 passing tests are honest and well-named — they
test the *logic* of the code paths that are wired. They do
NOT test the production model paths, because those paths
are not wired.**

15 new tests are required before production. The most
critical are tests #1, 2, 6, 7, 8, 9, 10, 11, 13 from
`TEST_QUALITY_AUDIT.md`.

## 6. Required fixes before production

In order of priority:

1. **Fix the import chain** (PATCH-001, PATCH-005): change
   `from psycopg import pool` to `from psycopg_pool import
   ConnectionPool`; add `paddlepaddle-gpu` and
   `psycopg_pool` to `requirements.txt`. *Effort: hours.*
2. **Wire the resolver to a stream consumer** (PATCH-006):
   add a `GlobalIdentityResolverWorker.run()` method and
   wire it in `main.py`. *Effort: hours.*
3. **Block startup in production mode if the ReID
   adapter is in fallback state** (PATCH-040 + extension
   of PATCH-003). *Effort: hours.*
4. **Implement real Paddle PP-Human + OC-SORT**
   (PATCH-002). *Effort: 1-2 weeks.*
5. **Implement real TransReID** (PATCH-003). *Effort:
   1-2 weeks.*
6. **Replace the synthetic crop fabrication with actual
   MinIO downloads** (PATCH-004). *Effort: hours.*
7. **Add FastAPI auth** (PATCH-014). *Effort: hours.*
8. **Add retention policies** (PATCH-015). *Effort: 1-2
   days.*
9. **Add the 15 missing tests.** *Effort: 1 week.*

The 9-step order above is the shortest path from "structural
skeleton" to "production-ready".

## 7. Improvement loop recommendation

The `IMPROVEMENT_LOOP_PLAN.md` documents a 12-component
loop (data capture → labeler → metrics → tuning → A/B →
promotion gate → HITL review). The design is complete; the
implementation is 0/12.

The shortest path to a self-tuning system is to implement:
1. The evidence sampler (sampled crops → labeler bucket).
2. The metrics extractor (22 metrics, all defined in
   `IMPROVEMENT_LOOP_PLAN.md`).
3. The threshold-tuning script (input: labeler data;
   output: new `auto_match_threshold`).
4. The shadow-mode deploy (writes to a shadow table; no
   live impact).
5. The deployment gate (Python script that exits
   non-zero on regression).
6. The HITL review queue (API endpoint for ambiguous
   decisions).

The full loop is 4-6 weeks of work. With the 9-step
"production-ready" list above, total time-to-production is
approximately **6-8 weeks** of focused engineering.

## 8. Exact next commands to run

```bash
# 1. Re-run the test suite to confirm the current baseline
cd /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC
python3 -m pytest tests/ -q
# Expected: 68 passed (plus the 1 restored production-safety test)

# 2. Verify the broken import is fixed
python3 -c "from app.storage.postgres import PostgresStore"
# Expected: no error. PATCH-001 fix verified.

# 3. Verify no paddle / torch in default requirements
grep -E "paddle|torch|psycopg_pool" requirements.txt 2>/dev/null || true
# Expected: zero matches (they live in pyproject.toml's [gpu] extra).

# 4. Verify the multi-camera model IS shared
grep -n "shared_detector\|frame_state" app/workers/multi_camera_runner.py | head -10
# Expected: every per-camera worker is constructed with the
# same ``self._shared_detector`` instance. PATCH-007 verified.

# 5. Verify the ReID adapter raises in production
python3 -c "from app.reid.transreid_adapter import TransReIDAdapter; from app.reid.base import ReIDConfig; a = TransReIDAdapter(ReIDConfig(name='t', embedding_dim=768, qdrant_collection='x'), mode=RuntimeMode.PRODUCTION); a.load()" 2>&1 | tail -3
# Expected: ProductionSafetyError raised (load refuses without weights).
# PATCH-003 / PATCH-040 verified.

# 6. Verify the resolver is wired to a stream consumer
grep -n "xreadgroup\|consume" app/identity/resolver.py
# Expected: matches — PATCH-006 verified.

# 7. Verify the retention worker iterates the live collection
grep -n "COLLECTIONS\|person_reid" scripts/retention_worker.py
# Expected: matches. PATCH-015 + post-revert fix verified.

# 8. Bring up infra (does NOT require the real models)
cp .env.example .env  # if not already present
docker compose up -d relation-store vector-store message-bus
docker compose run --rm db-migrator
docker compose run --rm detect-pipeline python scripts/init_qdrant.py
# Expected: all services healthy; collections created.

# 9. Run the smoke test (single-camera, synthetic)
docker compose run --rm detect-pipeline python main.py --mode single_cam_smoke
# Expected: WARNING logs, 30 s runtime, NO real detections.

# 10. Inspect what the smoke test produced
ls Service/reports/  # cross-check against Service/ baseline
curl -s http://localhost:8000/health
# Expected: {"status": "ok", "postgres": "ok"}

# 11. Open the audit folder
ls SOTA-Paddle-MTMC/Audit/
# Expected: AUDIT_REPORT.md, OFFICIAL_COMPLIANCE_MATRIX.md,
# BUG_REPORT.md, MULTI_CAMERA_MTMCT_AUDIT.md,
# REID_24H_AUDIT.md, DATABASE_AUDIT.md,
# T4_PERFORMANCE_AUDIT.md, SECURITY_PRIVACY_AUDIT.md,
# TEST_QUALITY_AUDIT.md, IMPROVEMENT_LOOP_PLAN.md,
# PATCH_PLAN.md
```

## 9. Bottom line (baseline)

**Do not deploy this system as-is.** The implementation is
an excellent skeleton with honest tests, a clean schema,
and a thoughtful architecture — but the production paths
are stubs. Six CRITICAL bugs must be fixed and 15
integration tests must be added before the system can
claim production-ready.

The 6-8 week roadmap above is realistic. After that, the
system can claim what `Service/` already claims: a
production-grade people-counting + ReID pipeline.

## 10. Bottom line (post-revert, 2026-06-17)

After the 2026-06-17 remediation:

* The four CRITICAL/HIGH safety-net reverts are restored.
* The retention worker now actually sweeps the live collection.
* The identity overlay cache is wired to the runner.
* `DwellBookkeeper.force_close_stale` is invoked every 60 s.
* `ruff check app/ --exclude 'app/detection/_vendor'` is clean.
* The dead ReID adapters are demoted to **experimental, off by
  default** (operator can opt in via `reid.active_model` +
  `*_INFERENCE_FN` env var).
* `.env.bak` and friends are ignored.

**The remaining production-readiness gap is integration
testing** (the 15 tests listed in §5). The architecture
itself is sound.

## 11. Open follow-ups

* **Phase 4 — integration tests** (separate PR per the
  remediation plan): real-GPU test for the stall watchdog,
  real-Qdrant test for the per-camera travel-time window,
  real-`scripts/retention_worker.py` smoke test, etc.
* **Phase 4 — Dockerfile build matrix**: 4 sister Dockerfiles
  exist; `docker-compose.yaml` hard-codes one image. Consolidate.
* **Phase 4 — test secrets-guard allowlist**: the previous
  allowlist had `.venv`, `__pycache__`, `.git`, `models`,
  `data`. Re-audit whether any other directories should be
  added (e.g. `.serena/`).
* **Phase 5 — operator docs**: `Docs/post_revert_remediation.md`
  describes the per-file change list; the operator runbook
  should link to it from the day-1 setup.
