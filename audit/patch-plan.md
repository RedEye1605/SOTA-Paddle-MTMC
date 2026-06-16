# Patch Plan — SOTA-Paddle-MTMC

> **Phase 11 — Patch plan.** Every finding is classified
> CRITICAL/HIGH/MEDIUM/LOW with file, evidence, reproduction,
> proposed fix, risk-of-fix, and required test. **Do not patch
> without explicit operator approval.**

## Severity legend

- **CRITICAL** — production-unsafe; broken official integration; false
  identity merge risk; security/privacy issue; data-loss risk.
- **HIGH** — likely runtime bug; wrong DB behavior; wrong MTMCT
  behavior; fake production path.
- **MEDIUM** — performance issue; missing test; weak error handling;
  benchmark weakness.
- **LOW** — docs mismatch; style; minor maintainability.

---

## CRITICAL findings

### PATCH-001 — `psycopg.pool` import is broken
**Severity:** CRITICAL
**File:** `app/storage/postgres.py:15`
**Official source:** psycopg 3.3 release notes — `ConnectionPool`
moved to the `psycopg_pool` package.
**Code evidence:**
```python
from psycopg import pool as pg_pool
```
**Reproduction:**
```bash
$ python3 -c "from app.storage.postgres import PostgresStore"
ImportError: cannot import name 'pool' from 'psycopg'
```
**Fix:**
```python
from psycopg_pool import ConnectionPool
```
And in `requirements.txt`:
```
psycopg_pool>=3.2
```
**Risk of fix:** low. The `ConnectionPool` constructor is
the same; only the import path changes.
**Test required:** re-run `python3 -m pytest tests/ -q`;
add an `import app.storage.postgres` smoke test in CI.

---

### PATCH-002 — Production detector path is a synthetic stub
**Severity:** CRITICAL
**File:** `app/workers/pphuman_worker.py:83-109, 197-200`
**Official source:** `PaddleDetection/deploy/pipeline/pipeline.py`
must be invoked with a real PaddleInference session.
**Code evidence:**
```python
if self._smoke_test_mode or self._detector_factory is None:
    detections = self._synthetic_detect(frame)
```
**Reproduction:** every production run produces random
boxes. The audit already confirmed zero real detections.
**Fix:**
1. Add `paddlepaddle-gpu==2.6.x` and
   `paddlepaddle-inference==...` to `requirements.txt`.
2. Implement `app/workers/pphuman_worker.py:_try_load_paddle()`
   to instantiate `paddle.inference.create_predictor(...)`
   against the `mot_ppyoloe_l_36e_pipeline` model dir, with
   TensorRT FP16 enabled.
3. Add a startup guard in `app/main.py:build_app_context()`:
   if `mode != "single_cam_smoke"` and the real model
   didn't load, raise `RuntimeError("Real Paddle PP-Human
   model not loaded; refuse to start in production")`.
4. Vendor Paddle's `OC-SORT` Python tracker (or invoke
   `pipeline.py` as a subprocess) and replace the
   `_update_tracks` naive IoU.
**Risk of fix:** high. The Paddle Python API is not
import-friendly; vendor Paddle's tracker code is several
hundred lines. Estimate 2-4 days of work.
**Test required:** the "Real Paddle integration test" from
`TEST_QUALITY_AUDIT.md` (test #1).

---

### PATCH-003 — Production ReID paths are histogram fallbacks
**Severity:** CRITICAL
**Files:** `app/reid/transreid_adapter.py:38-86`,
`app/reid/pphuman_adapter.py:36-84`
**Official source:** `damo-cv/TransReID` model
construction is via `model.make_model(cfg, num_class, …)`;
Paddle's `StrongBaseline` is via PaddleInference.
**Code evidence:** the `_try_load()` methods raise
`RuntimeError("…not configured")` and the adapter
silently flips to `_fallback_active=True`.
**Reproduction:** any call to `extract()` returns a
histogram feature, not a real ReID feature.
**Fix:**
1. Vendor `damo-cv/TransReID/model/` into a new
   `app/reid/_transreid_native/` submodule.
2. Implement `_try_load()` to construct
   `vit_base_patch16_224_TransReID(img_size=(256, 128),
   stride_size=12, sie_xishu=3.0, local_feature=True)`,
   load the `state_dict` from
   `models/vit_transreid_msmt.pth` with
   `weights_only=True`, move to GPU, switch to FP16.
3. Implement JPM aggregation (concat of global + 4 local
   features, L2-normalize, take CLS-nek feature per
   official config).
4. Same pattern for Paddle StrongBaseline: vendor
   `deploy/pptracking/python/nets/strongbaseline.py`
   and instantiate via PaddleInference.
5. Add a startup guard identical to PATCH-002.
6. Add BUG-034 fix: rename
   `models/vit_transreid_msmt.pth` or download the
   Market-1501 checkpoint to match the config.
**Risk of fix:** high. The TransReID forward pass has
specific arguments (`cam_label`, `view_label`); the
JPM output is 5×768; the `neck_feat: 'before'` config
requires a custom feature extraction. Estimate 3-5
days of work.
**Test required:** the "Real TransReID integration test"
(test #2); a batch test with 32 crops, expect 768-dim
output, expect unit norm.

---

### PATCH-004 — ReID worker fabricates crops from quality score
**Severity:** CRITICAL
**File:** `app/workers/reid_worker.py:62-69`
**Code evidence:**
```python
arr = np.full((128, 64, 3), int(tl.quality_score or 128), dtype=np.uint8)
crops.append(arr)
```
**Reproduction:** every ReID extraction in production
uses a flat-color crop whose pixel values are the
quality score. Two unrelated people with the same
quality will produce identical embeddings.
**Fix:** download the actual crop from MinIO:
```python
from minio import Minio
def _load_crop(uri: str) -> np.ndarray:
    assert uri.startswith(f"s3://{self.minio_bucket_for_uri()}/")
    key = uri.split(f"s3://{self.minio_bucket_for_uri()}/", 1)[-1]
    resp = self.minio.client.get_object(self.minio.bucket, key)
    data = resp.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
```
Raise loud error if the download fails.
**Risk of fix:** low. The MinIO client is already
imported; `get_object` is a stable API.
**Test required:** an integration test that uploads a
test crop to MinIO, publishes the URI to
`stream:tracklets`, asserts the ReID worker actually
loaded the crop (e.g. by adding a `_crops_loaded`
counter).

---

### PATCH-005 — `requirements.txt` is missing `paddlepaddle-gpu` and `psycopg_pool`
**Severity:** CRITICAL
**Files:** `requirements.txt`
**Code evidence:** the file lists `psycopg[binary]>=3.2` but
not `psycopg_pool>=3.2`; no `paddlepaddle-gpu` entry.
**Reproduction:** `pip install -r requirements.txt` and
run `python -c "import app.storage.postgres"` — fails
on `from psycopg import pool`.
**Fix:** add the missing entries; pin the versions.
**Risk of fix:** medium. The paddlepaddle-gpu package
is large (~1 GB) and has CUDA-version pinning. The
team must verify the CUDA version on the target T4.
**Test required:** a fresh `docker compose build detect-pipeline`
followed by a startup smoke test.

---

### PATCH-006 — Resolver is not wired to `stream:embeddings`
**Severity:** CRITICAL
**File:** `app/identity/resolver.py` (no consumer method);
`app/main.py` (no `ResolverWorker`)
**Code evidence:** `GlobalIdentityResolver.resolve(...)` is
an in-process function. No `run(stop_event)` method on
it. `main.py:228-231` starts `reid_worker` and
`telemetry_worker` as threads, but no resolver worker.
**Reproduction:** the Resolver class is never
instantiated in production code paths (it's instantiated
in `build_app_context()` but never called).
**Fix:** add a `GlobalIdentityResolverWorker` class with
`run(stop_event)` that consumes `stream:embeddings` and
calls `self.resolve(...)`. Wire it in `main.py` like the
other workers.
**Risk of fix:** low. It's an orchestration fix.
**Test required:** a stream consumer test (test #13 in
`TEST_QUALITY_AUDIT.md`).

---

## HIGH findings

### PATCH-007 — Multi-camera runner does not share a model
**Severity:** HIGH
**File:** `app/workers/multi_camera_runner.py:79-94`
**Code evidence:** no `model=…` parameter on the worker
or the runner.
**Fix:** add `model: ReIDAdapter | Detector` to both
classes; pass the same instance to every worker. Add
an `assert id(w1.model) == id(w2.model)` in test.
**Risk of fix:** low. Architecture-only change.
**Test required:** test #10 in `TEST_QUALITY_AUDIT.md`.

---

### PATCH-008 — `final_score` from 5-factor scoring is unused
**Severity:** HIGH
**Files:** `app/identity/resolver.py:185-204`,
`app/identity/ambiguity.py:34-71`
**Code evidence:** `decide_ambiguity` checks only
`top1.score` (raw cosine), not the `final_score`.
**Fix:** modify `decide_ambiguity` to take
`final_score: float` and use it as the threshold.
**Risk of fix:** low. The threshold tuning values may
need to be re-tuned after this change.
**Test required:** unit test for `decide_ambiguity` with
`final_score` argument; integration test that a
high-cosine-low-quality match is rejected.

---

### PATCH-009 — Quality placeholder (`0.6`) is hard-coded
**Severity:** HIGH
**File:** `app/workers/tracklet_collector.py:160`
**Code evidence:** `tl.quality_score = 0.6   # placeholder`
**Fix:** score every crop with `crop_quality_score()`,
keep the max, write to `tl.quality_score` and
`tl.best_crop_uri`.
**Risk of fix:** low.
**Test required:** unit test that the quality score of a
good crop > 0.5 and a bad crop < 0.5.

---

### PATCH-010 — `stream:identity_decisions` and `stream:zone_events` have no publishers
**Severity:** HIGH
**Files:** `app/identity/resolver.py` (no publish),
`app/workers/tracklet_collector.py` (no zone-event
publish)
**Code evidence:** `TelemetryWorker` consumes from
these streams; nothing publishes. `dwell_sessions`
never gets opened.
**Fix:** in the resolver's `resolve()`, after the
decision, publish to `stream:identity_decisions`. In
the tracklet collector, detect zone enter/exit
transitions and publish to `stream:zone_events`.
**Risk of fix:** low.
**Test required:** stream consumer test (test #13).

---

### PATCH-011 — TransReID weight is MSMT17 but config points to Market-1501
**Severity:** HIGH
**File:** `models/vit_transreid_msmt.pth` (400 MB),
`configs/reid/transreid.yaml:9` says
`weight: /models/transreid/transformer_120.pth`
**Code evidence:** the file at
`models/vit_transreid_msmt.pth` was downloaded from
`syliz-lcz/CLIP-ReID`, not `damo-cv/TransReID`. The
classifier head dimensions differ (MSMT17=1041,
Market-1501=751).
**Fix:** either (a) download the official Market-1501
checkpoint from `damo-cv/TransReID#testing` (requires
Google Drive, see `download_transreid_models.sh`),
or (b) update the config to `num_class: 1041` and
document that the model is MSMT17-trained.
**Risk of fix:** low (config change) or medium
(download dependency on Google Drive).
**Test required:** the model-load test (test #2)
verifies the output shape.

---

### PATCH-012 — `tracklet_embeddings.vector_db_point_id` is not namespaced
**Severity:** HIGH
**File:** `app/workers/reid_worker.py:79`
**Code evidence:** `point_id = f"{tl.tracklet_id}-{i:02d}"`
**Fix:** `point_id = f"{self.adapter.name}-{tl.tracklet_id}-{i:02d}"`
**Risk of fix:** low. New points are fine; existing
points are not migrated (one-time backfill is OK).
**Test required:** unit test that two adapters
writing the same tracklet get different point IDs.

---

### PATCH-013 — `migrator` runs migrations in implicit transactions
**Severity:** HIGH
**File:** `docker-compose.yaml:81-85`
**Code evidence:** no `--single-transaction`; if a
migration fails, the DB is in inconsistent state.
**Fix:**
```yaml
command: |
  "for f in /migrations/*.sql; do
     echo \"Applying $$f\";
     psql -h postgres -U $${POSTGRES_USER} -d $${POSTGRES_DB}
        --single-transaction --set ON_ERROR_STOP=on -f $$f;
   done"
```
**Risk of fix:** low. `--single-transaction` is
non-destructive for idempotent migrations.
**Test required:** a "fail-fast" test where a
malformed migration is added and the migrator is
expected to exit non-zero without partial state.

---

### PATCH-014 — No FastAPI auth on identity endpoints
**Severity:** HIGH
**File:** `app/api/server.py`
**Code evidence:** `@app.get("/identity/{global_id}")` has
no auth dependency.
**Fix:** add a `Depends(verify_token)` dependency that
reads `Authorization: Bearer <token>` and validates
against a shared secret. Issue tokens via a separate
auth service (or use mutual TLS).
**Risk of fix:** medium. The token issuance is
out of scope; the validation is straightforward.
**Test required:** an auth test that asserts
`/identity/{id}` returns 401 without a token.

---

### PATCH-015 — No retention policy for evidence crops, Qdrant, PG
**Severity:** HIGH (privacy)
**Files:** MinIO, Qdrant, PG
**Code evidence:** no scheduled retention worker.
**Fix:** add `scripts/retention_worker.py` that:
1. Calls `pg.expire_old_identities(older_than_seconds=N)`.
2. Iterates Qdrant collections and deletes points with
   `timestamp < now() - N*86400`.
3. Configures MinIO lifecycle policy:
   ```python
   client.set_bucket_lifecycle(bucket, LifecycleConfig(
       rules=[LifecycleRule(
           status=Status.ENABLED,
           expiration=Expiration(days=N_EVIDENCE_DAYS),
       )]
   ))
   ```
**Risk of fix:** medium. Retention deletion is
destructive; ensure the `evidence_sampler` bucket has
its own (longer) retention.
**Test required:** a retention test that inserts old
data, runs the worker, asserts deletion.

---

## MEDIUM findings

### PATCH-016 — Cross-camera retrieval doesn't filter by travel-time window
**Severity:** MEDIUM
**File:** `app/identity/resolver.py:101-119`
**Fix:** add a Stage 1.5 filter: for each linked
`to_camera_id`, compute
`[ts - max_travel_seconds(from_cam, to_cam),
 ts - min_travel_seconds(from_cam, to_cam)]`
and add it to the Qdrant filter.
**Risk of fix:** low.
**Test required:** the "Impossible-travel-time test"
in `MULTI_CAMERA_MTMCT_AUDIT.md` (test #3).

### PATCH-017 — The 24h fallback Stage 3 of staged retrieval is missing
**Severity:** MEDIUM
**File:** `app/identity/resolver.py:79-99`
**Fix:** add a `_candidate_cameras_stage3()` that
returns all active cameras (cameras seen in the
last 24h) for fallback search. The threshold for
Stage 3 is higher (`auto_match_threshold + 0.1`).
**Risk of fix:** low.
**Test required:** the 24h retrieval test
(test #8 in `TEST_QUALITY_AUDIT.md`).

### PATCH-018 — No FPS per camera logging
**Severity:** MEDIUM
**File:** `app/workers/multi_camera_runner.py`
**Fix:** track wall-clock between `FrameResult`
emissions, push to `REGISTRY.analytics_fps`.
**Risk of fix:** low.
**Test required:** integration test with 2 cameras
emitting 5 frames/s, assert `analytics_fps` reads
~5 after 10 s.

### PATCH-019 — `qdrant_query_latency_seconds` histogram is never observed
**Severity:** MEDIUM
**File:** `app/storage/qdrant_store.py:152-153`
**Fix:** after `self.client.search(...)`,
`REGISTRY.qdrant_query_latency.observe(latency)`.
**Risk of fix:** low.
**Test required:** a unit test that observes the
histogram and asserts non-empty after a fake search.

### PATCH-020 — `postgres_write_latency_seconds` histogram is never observed
**Severity:** MEDIUM
**File:** `app/storage/postgres.py:97-102`
**Fix:** in `timed_execute`, return the latency
*and* observe the histogram.
**Risk of fix:** low.
**Test required:** same as PATCH-019.

### PATCH-021 — `tracklet_buffer_size` and `stream_backlog` gauges are never set
**Severity:** MEDIUM
**File:** `app/telemetry/metrics.py:130-132`
**Fix:** in `TrackletCollector.on_frame` and the
Redis consumer, set these gauges.
**Risk of fix:** low.
**Test required:** unit test.

### PATCH-022 — `reid_extractions_total` counter is never incremented
**Severity:** MEDIUM
**File:** `app/workers/reid_worker.py:72-100`
**Fix:** in `process_tracklet`, after `extract()`,
`REGISTRY.reid_extractions.inc()`.
**Risk of fix:** low.
**Test required:** unit test.

### PATCH-023 — `MultiCameraRunner.stop()` is a no-op
**Severity:** MEDIUM
**File:** `app/workers/multi_camera_runner.py:116-122`
**Fix:** keep a per-worker `threading.Event`; signal
on `stop()`; join the threads; close the
`cv2.VideoCapture`s.
**Risk of fix:** low.
**Test required:** unit test that calls
`runner.stop()` and asserts the threads exit within
1 s.

### PATCH-024 — `finalize_stale` 5 s timeout is too aggressive
**Severity:** MEDIUM
**File:** `app/workers/tracklet_collector.py:139-164`
**Fix:** raise default to 30 s; or track "frames
since last emit" and close after N frames.
**Risk of fix:** low. May produce more
`id_fragmentation`.
**Test required:** unit test.

### PATCH-025 — `MultiCameraRunner.start()` is not idempotent
**Severity:** MEDIUM
**File:** `app/workers/multi_camera_runner.py:74-95`
**Fix:** guard with `if self._workers: return`.
**Risk of fix:** low.
**Test required:** unit test.

### PATCH-026 — `migrator` doesn't depend on `api`; `init_qdrant.py` is run as `api` image
**Severity:** MEDIUM
**File:** `docker-compose.yaml:81-85, 86-117`
**Fix:** add a `qdrant-init` service that depends on
`qdrant` and uses a minimal image (no Paddle/Torch
needed for `init_collections`).
**Risk of fix:** low.
**Test required:** `docker compose up -d` smoke test.

### PATCH-027 — `Dockerfile` uses `pip` despite copying `uv`
**Severity:** MEDIUM
**File:** `Dockerfile:29-36`
**Fix:** use `uv pip install` or `uv sync` with a
`uv.lock` for reproducibility.
**Risk of fix:** low.
**Test required:** build time benchmark.

### PATCH-028 — `Dockerfile` doesn't install `paddlepaddle` / `torch`
**Severity:** MEDIUM
**File:** `Dockerfile` (already addressed by PATCH-005)
**Fix:** same as PATCH-005.
**Risk of fix:** medium (large base image).
**Test required:** fresh build.

### PATCH-029 — `evidence_key()` uses `global_id="UNASSIGNED"` and is never re-keyed
**Severity:** MEDIUM
**File:** `app/storage/minio_store.py:89-116`,
`app/workers/tracklet_collector.py:117-128`
**Fix:** after the resolver assigns a `global_id`, do
a MinIO copy-object to a re-keyed path; delete the
`UNASSIGNED` original.
**Risk of fix:** medium. Race conditions between
two ReID workers re-keying the same tracklet; use
a small `evidence_rekey` table to dedupe.
**Test required:** integration test.

### PATCH-030 — `identity_decisions.decision_type` lacks a CHECK constraint
**Severity:** MEDIUM
**File:** `db/migrations/002_identity_tables.sql:97`
**Fix:**
```sql
CHECK (decision_type IN ('match','new','candidate','ambiguous','held'))
```
**Risk of fix:** low. Existing data must conform.
**Test required:** migration test.

### PATCH-031 — Per-camera QoS / back-pressure is absent
**Severity:** MEDIUM
**File:** `app/workers/multi_camera_runner.py`
**Fix:** add per-camera rate limiting in
`stream()`; respect `CameraSource.fps_target`.
**Risk of fix:** low.
**Test required:** integration test with one fast
camera and one slow; assert no overflow.

### PATCH-032 — No reconnect logic for RTSP
**Severity:** MEDIUM
**File:** `app/workers/multi_camera_runner.py:38-55`
**Fix:** on EOF from `cv2.VideoCapture`, sleep
`reconnect_backoff_seconds` and reopen. Track
reconnect count in `REGISTRY.rtsp_reconnects`.
**Risk of fix:** low.
**Test required:** integration test with a stream
that disconnects.

### PATCH-033 — Resolver stage 1 (`same-cam recent 60s`) is not actually executed
**Severity:** MEDIUM
**File:** `app/identity/resolver.py:79-99`
**Fix:** explicitly run a Qdrant search with
`candidate_camera_ids={source_cam}` and
`timestamp_gte=ts-60` before the linked-cams
search.
**Risk of fix:** low.
**Test required:** unit test.

### PATCH-034 — `QdrantStore.search()` allows `timestamp_gte=0` (no time bound)
**Severity:** MEDIUM
**File:** `app/storage/qdrant_store.py:111-138`
**Fix:** enforce `timestamp_gte > 0` and
`timestamp_gte <= ts`. (Today the resolver sets it
to `ts - 86400`, which is fine. But the
`QdrantStore` doesn't defend against direct
callers.)
**Risk of fix:** low.
**Test required:** unit test.

### PATCH-035 — `get_recent` returns the wrong type on JSON parse error
**Severity:** MEDIUM
**File:** `app/storage/redis_state.py:101-107`
**Fix:** log a warning and return None on
`json.JSONDecodeError`. (Already does this. OK.)
(Remove from patch plan.)

### PATCH-036 — `psql -h localhost` in README assumes Docker port-forward
**Severity:** MEDIUM
**File:** `README.md:93-95`
**Fix:** use `docker compose exec relation-store psql -U …`
instead.
**Risk of fix:** low.
**Test required:** doc walkthrough.

---

## LOW findings

### PATCH-037 — `architecture_guards` does not enforce "one model instance"
**Severity:** LOW
**File:** `tests/test_architecture_guards.py`
**Fix:** add a test that constructs
`MultiCameraRunner(sources, model=…)` with two
cameras and asserts both workers share the model.
**Risk of fix:** low.
**Test required:** see test #10.

### PATCH-038 — Secret scanner regex misses YAML key:value
**Severity:** LOW
**File:** `tests/test_architecture_guards.py:90-97`
**Fix:** add `(r'^\s*(password|token|key|secret)[\s:='"]+\S+",
"yaml secret literal")` to `SECRET_PATTERNS`.
**Risk of fix:** low.
**Test required:** add a `.env` with `password=foo` and
assert the test fails.

### PATCH-039 — `.gitignore` is missing
**Severity:** LOW
**File:** `.gitignore`
**Fix:** add `.gitignore` excluding `.env`,
`__pycache__/`, `models/`, `*.pth`, `*.onnx`.
**Risk of fix:** low.
**Test required:** manual.

### PATCH-040 — `tmp_fallback` is logged at WARNING, not ERROR
**Severity:** LOW (should be MEDIUM in production)
**File:** `app/reid/transreid_adapter.py:42-45`,
`app/reid/pphuman_adapter.py:42-48`
**Fix:** change to `logger.error(...)`. In
production mode, raise instead of log.
**Risk of fix:** low.
**Test required:** log capture test.

### PATCH-041 — `min_person_height_px` is hard-coded in collector
**Severity:** LOW
**File:** `app/workers/tracklet_collector.py:67`
**Fix:** read from `reid.min_person_height_px` config.
**Risk of fix:** low.
**Test required:** unit test.

### PATCH-042 — `health` endpoint blocks the FastAPI threadpool
**Severity:** LOW
**File:** `app/api/server.py:21-24`
**Fix:** wrap `pg.healthcheck()` in
`asyncio.wait_for(timeout=2)`.
**Risk of fix:** low.
**Test required:** integration test.

### PATCH-043 — `MultiCameraRunner.stream` doesn't enforce `max_seconds` in production
**Severity:** LOW
**File:** `app/workers/multi_camera_runner.py:101-114`
**Fix:** pass `max_seconds` from config in production
mode.
**Risk of fix:** low.
**Test required:** unit test.

### PATCH-044 — No `.gitignore` for the `models/` directory in SOTA
**Severity:** LOW
**File:** `.gitignore`
**Fix:** `models/*.pth` and `models/*.onnx` should
not be committed.
**Risk of fix:** low.
**Test required:** manual.

### PATCH-045 — `docs/architecture.md` claims Stage 3 (24h fallback) that doesn't exist
**Severity:** LOW
**File:** `Docs/architecture.md:39-42`
**Fix:** either implement (PATCH-017) or remove
from doc.
**Risk of fix:** low.
**Test required:** doc audit.

### PATCH-046 — Telemetry worker reads `stream:identity_decisions` but resolver doesn't publish
**Severity:** LOW (covered by PATCH-010)
**Fix:** covered.

### PATCH-047 — Dockerfile has no HEALTHCHECK
**Severity:** LOW
**File:** `Dockerfile`
**Fix:** add `HEALTHCHECK CMD curl -f
http://localhost:8000/health || exit 1`.
**Risk of fix:** low.
**Test required:** `docker inspect` walkthrough.

### PATCH-048 — ReID-batch stress not benchmarked
**Severity:** LOW
**File:** `scripts/benchmark_t4.py:21-43`
**Fix:** implement the actual benchmark (PATCH-049).
**Risk of fix:** low.
**Test required:** manual benchmark.

### PATCH-049 — `scripts/benchmark_t4.py` is a stub
**Severity:** MEDIUM
**File:** `scripts/benchmark_t4.py`
**Fix:** implement real scenario runners; record real
metrics.
**Risk of fix:** medium (requires real models).
**Test required:** manual.

### PATCH-050 — `scripts/compare_with_service_baseline.py` is a stub
**Severity:** LOW
**File:** `scripts/compare_with_service_baseline.py`
**Fix:** implement the comparison runner.
**Risk of fix:** medium (requires both systems
operational).
**Test required:** manual.

---

## Summary

| Severity | Count |
|---|---|
| CRITICAL | 6 (PATCH-001..006) |
| HIGH | 9 (PATCH-007..015) |
| MEDIUM | 21 (PATCH-016..036) |
| LOW | 14 (PATCH-037..050) |
| **Total** | **50** |

## Recommended order of application

1. **PATCH-001, PATCH-005** — unblock startup (1 hour).
2. **PATCH-006, PATCH-010, PATCH-014, PATCH-015** —
   orchestration + security (1-2 days).
3. **PATCH-002, PATCH-003, PATCH-004, PATCH-007** —
   real Paddle + real TransReID (1-2 weeks).
4. **PATCH-008, PATCH-009, PATCH-011, PATCH-012** —
   correctness (1 week).
5. **PATCH-016, PATCH-017, PATCH-033** — staged
   retrieval correctness (1 week).
6. **PATCH-018..022, PATCH-027, PATCH-028, PATCH-031,
   PATCH-032** — observability + ops (1 week).
7. **PATCH-029, PATCH-030, PATCH-034..036** —
   retention + DB correctness (1 week).
8. **PATCH-023..026, PATCH-037..050** — hardening
   (1 week).
9. **PATCH-048, PATCH-049, PATCH-050** — benchmarks
   (1 week).

After all CRITICAL + HIGH patches, plus tests #1, 2, 6,
7, 8, 9, 10, 11, 13 from `TEST_QUALITY_AUDIT.md`, the
system may claim production-ready — provided the
operator has:
- Real Paddle PP-Human weights on disk.
- Real TransReID weights (Market-1501, not MSMT17) on
  disk.
- A recorded multi-camera test dataset for regression.
- An operator to review ambiguous decisions.
- A retention policy in place.
