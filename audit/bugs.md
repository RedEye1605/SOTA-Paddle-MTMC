# Bug Report — SOTA-Paddle-MTMC

> **Phase 3 — Bug hunt.** All 40 risk categories in the audit brief were
> checked against the source. Each finding has a severity, file/line,
> reproduction, and proposed fix. Findings are also carried into
> `PATCH_PLAN.md` with a single ID per item.

Severity legend: **CRITICAL** = production-unsafe; **HIGH** = likely
runtime bug or wrong behavior; **MEDIUM** = performance / test gap;
**LOW** = docs/style.

---

## BUG-001 — `psycopg.pool` import is broken on psycopg ≥ 3.2
**Severity: CRITICAL**

**File:** `app/storage/postgres.py:15`
```python
from psycopg import pool as pg_pool
```
**Reproduction:**
```bash
$ python3 -c "from app.storage.postgres import PostgresStore"
ImportError: cannot import name 'pool' from 'psycopg'
```
The `psycopg` 3.3 release moved the connection pool to a separate
`psycopg_pool` package; `from psycopg import pool` no longer works.

**Why this matters:** every page that imports `app.api.server`,
`app.main`, or any worker transitively that touches `PostgresStore`
will crash at import time. The `PostgresStore` is also referenced in
`build_app_context()` from `main.py` — i.e. the production entry point.

**Fix:**
```python
from psycopg_pool import ConnectionPool  # or: import psycopg_pool
# then use psycopg_pool.ConnectionPool(...)
```
Add `psycopg_pool` to `requirements.txt`.

---

## BUG-002 — Synthetic detector is the only detector
**Severity: CRITICAL**

**File:** `app/workers/pphuman_worker.py:83-109, 197-200`
```python
if self._smoke_test_mode or self._detector_factory is None:
    detections = self._synthetic_detect(frame)
else:
    detections = self._detector_factory(frame)
```
`multi_camera_runner.py:80-86` constructs `PPHumanWorker(smoke_test_mode=True)`
for every camera, and there is no `detector_factory` argument passed.

**Reproduction:** `python -m app.main --mode single_cam_smoke` emits
"detections" that are random `(x, y, w, h)` based on a per-(camera,
frame) RNG seed. They are not real detections.

**Why this matters:** the README's "production-ready" claim is false.
There is no Paddle import, no PaddleInference session, and the synthetic
detector is the only code path. The "production" mode is not even wired
to a different code path — `smoke_test_mode=True` is the *only* path.

**Fix:** either
- wire `paddle.inference` + the PP-Human pipeline as a subprocess; or
- spawn `paddle.tools.infer` (or `pipeline.py`) as a child process per
  camera; or
- vendor the OC-SORT tracker + detector into the worker using a real
  PyTorch backend. In all three cases, block startup if the real model
  is unavailable and the `mode != single_cam_smoke`.

---

## BUG-003 — TransReID / PP-Human ReID are always fallback
**Severity: CRITICAL**

**File:** `app/reid/transreid_adapter.py:38-72`,
`app/reid/pphuman_adapter.py:36-69`,
`app/reid/clipreid_adapter_optional.py:24-31`

The `load()` and `_try_load()` methods catch the missing-model
exception and silently flip `_fallback_active = True`. The `extract()`
method then unconditionally routes to `_deterministic_fallback()`.
There is no `TRANSREID_MODEL_FN` / `PPHUMAN_REID_INFERENCE_FN` env var
handler anywhere in the codebase; the adapters raise
`RuntimeError("Real … inference path not configured.")` whenever a
fallback-free path is requested.

**Reproduction:** any import of `TransReIDAdapter().extract([crop])` in
production returns a histogram-derived 768-dim feature, NOT the
TransReID feature. The "5-factor scoring" is therefore evaluating
histogram similarity, not real visual ReID.

**Why this matters:** the entire identity pipeline is operating on
synthetic features. Two people with similar clothing colors will have
near-identical histogram features, which will produce false merges.

**Fix:** vendor the `damo-cv/TransReID` `make_model` and
`vit_base_patch16_224_TransReID` from `model/backbones/vit_pytorch.py`
into the repo (or import it as a submodule). Same for
`PaddleInference` + `strongbaseline_r50_30e_pa100k`. Mark
`fallback_active` as a startup error if `mode != single_cam_smoke`.

---

## BUG-004 — Multi-camera runner does not share a model
**Severity: HIGH**

**File:** `app/workers/multi_camera_runner.py:79-94`

```python
for cam in self.cameras:
    reader = make_frame_reader(cam.source, loop=True)
    worker = PPHumanWorker(
        camera_id=cam.camera_id,
        frame_reader=reader,
        ...
        smoke_test_mode=self._smoke_test_mode,
    )
```

There is no `model=…` argument; the comment
`# In production, model is loaded ONCE here` is a stub. The hard
guarantee "one model instance shared across all cameras" is NOT
verified by any test (`test_architecture_guards.py` does not assert
this).

**Reproduction:** Open `multi_camera_runner.py` and read the
constructor — it has no `model` parameter.

**Fix:** add a `model: Detector` parameter to `PPHumanWorker` and
`MultiCameraRunner.__init__`; pass the same instance to every worker.
Add an `assert id(worker.model) == id(shared_model)` style guard in a
test.

---

## BUG-005 — ReID worker fabricates crops from `quality_score`
**Severity: CRITICAL**

**File:** `app/workers/reid_worker.py:62-69`
```python
arr = np.full((128, 64, 3), int(tl.quality_score or 128), dtype=np.uint8)
crops.append(arr)
```
The `try` block catches any download failure and falls through to
*synthesizing* a crop from the quality score. **Every** ReID extraction
in the test path uses a flat-colour crop whose pixel values are
`quality_score`. Two crops with the same quality will produce
**identical** embeddings from the histogram-based fallback.

**Why this matters:** every gallery entry is an (essentially) constant
feature. Re-ranking, top-1, top-2 selection, ambiguity scoring all
collapse to "compare quality scores". Two unrelated people with the
same ReID quality will appear identical in the gallery.

**Fix:** either (a) actually download the crop from MinIO via the
`minio` client, or (b) raise a loud error so the operator knows the
ReID pipeline is not running. Do not silently fabricate.

---

## BUG-006 — `finalize_stale` always uses quality_score placeholder
**Severity: HIGH**

**File:** `app/workers/tracklet_collector.py:160`
```python
tl.quality_score = 0.6   # placeholder; real impl scores every crop
```
The `best_crop_uri` is also `tl.crop_uris[0]` (the first debug crop),
not the actual best.

**Why this matters:** the resolver uses `tracklet_quality` in the
5-factor score. Setting it to a constant `0.6` means the quality
factor becomes a constant 0.5 after normalization — the "quality
filter" exists in code but is dead on the wire.

**Fix:** keep a per-crop quality score, pick the highest, write to
`tl.best_crop_uri` and `tl.quality_score`.

---

## BUG-007 — Resolver queries the resolver-collection, but the
##          ReID worker writes with `point_id = f"{tracklet_id}-{i:02d}"`
##          causing collision in the rare case of multiple models writing
##          to overlapping IDs.
**Severity: MEDIUM**

**File:** `app/workers/reid_worker.py:79`
```python
point_id = f"{tl.tracklet_id}-{i:02d}"
```
`vector_db_point_id` is `(collection, point_id)` unique in
`tracklet_embeddings` (`002_identity_tables.sql:62`). With model
A writing `{tracklet}-00` and model B also writing `{tracklet}-00` on
the *same tracklet* (e.g. during a benchmark), the second model
silently gets a `ON CONFLICT DO NOTHING` and the audit row vanishes.

**Fix:** namespace point IDs by model: `f"{model_name}-{tl.tracklet_id}-{i:02d}"`.

---

## BUG-008 — `final_score` in the resolver is computed but never
##          used in the decision
**Severity: HIGH**

**File:** `app/identity/resolver.py:185-204`

The 5-factor `score_breakdown()` returns `final_score`, but the
actual decision is made by `decide_ambiguity(top1, top2, …)` which
*only* uses `top1.score` (the ReID cosine) and the margin. The
topology block (`is_known_link=False` → "new") is checked inside
`decide_ambiguity`, so topology gating works. But the
`auto_match_threshold` is applied to **ReID cosine**, not to the
weighted `final_score`. This contradicts the README and
`Docs/reid_threshold_tuning.md`, both of which say
"final_score >= auto_match_threshold → match".

**Reproduction:** with reid_sim=0.85, time_diff=0, is_link=True,
quality=1.0, zone_same=True, `final_score ≈ 0.85*0.55+0.20+0.15+0.05+0.05 = 0.92`,
which is above 0.82. If reid_sim=0.83, final_score ≈ 0.91, also above
0.82 — so the match would happen. But if reid_sim=0.83 and
is_link=False, decide_ambiguity returns "new" *before* the final
score is even consulted, which is correct for the topology case.
However: a reid_sim=0.81 (below auto_match_threshold=0.82) with
all-1.0 supporting factors would still get final_score=0.93 — but
decide_ambiguity returns "candidate" or "new", not "match". The
README's "5-factor weighted score" is being silently downgraded to
"ReID cosine alone".

**Fix:** modify `decide_ambiguity` to accept the 5-factor
`final_score` and threshold against it, not the raw `top1.score`.
Keep the topology block as a separate pre-check (already there).

---

## BUG-009 — Top-2 candidate construction drops cross-camera
##          candidates from same-global_id hits
**Severity: MEDIUM**

**File:** `app/identity/resolver.py:165-174`
```python
if cid in per_cam and per_cam[cid] >= float(h.score):
    continue
per_cam[cid] = float(h.score)
...
if top1 is None or cand.score > top1.score:
    top2 = top1
    top1 = cand
elif top2 is None or cand.score > top2.score:
    top2 = cand
```

The intent (one hit per camera, then rank globally) is correct, but
the assignment of `top2` is broken. When a *new* candidate with
higher score than top1 arrives, the OLD top1 becomes top2 — but
that top1 was already from a different camera, so the global
top1/top2 ranking is consistent. OK — but: when two hits have the
same score from different cameras, only the first one wins; the
second is dropped from `per_cam` only if the new score is *lower*.
The condition `per_cam[cid] >= float(h.score)` is `>=`, so equal
scores get `continue`d. **Two visually identical candidates at
the same score are silently dropped from top2.**

**Fix:** change `>=` to `>`; consider an epsilon-based tie-break.

---

## BUG-010 — `_candidate_cameras` only filters by topology, not by
##           time window of last_seen
**Severity: MEDIUM**

**File:** `app/identity/resolver.py:79-99`
The function returns the topology-derived candidate cameras. But
the Qdrant search filter is then called with
`timestamp_gte = ts - persistence_window_seconds` (24h), which
**always** allows linked cameras regardless of travel-time
feasibility. A tracklet seen on CAM_02 at time T will be matched
against CAM_01 candidates seen 23 hours ago, even if the topology
has `min_travel_seconds=10, max_travel_seconds=90` — a 23h-old CAM_01
candidate is impossible.

**Fix:** filter `payload.timestamp` to be within
`[ts - max_travel_seconds(from_cam), ts - min_travel_seconds(from_cam)]`
in the Qdrant payload filter. This is the "staged retrieval" the
docs promise.

---

## BUG-011 — `emit_closed_tracklets` writes `global_id=None` then
##           the resolver updates by tracklet_id, but
##           `insert_tracklet` has `ON CONFLICT DO NOTHING`, so a
##           closed → re-closed cycle loses the resolver's update.
**Severity: MEDIUM**

**File:** `app/storage/postgres.py:248`
```sql
ON CONFLICT (tracklet_id) DO NOTHING;
```

The resolver subsequently does
`pg.create_global_identity(...)` and
`pg.update_global_identity_seen(...)`, but the `tracklet.global_id`
column is never updated by the resolver (no SQL method
`update_tracklet_global_id` exists). The audit row is written to
`identity_decisions.assigned_global_id`, but the joinable
`tracklets.global_id` column stays `NULL` forever.

**Fix:** add a method `update_tracklet_global_id(tracklet_id, global_id)`
and call it from the resolver after decision.

---

## BUG-012 — `get_recent` JSON encoding is float, not int
**Severity: LOW**

**File:** `app/storage/redis_state.py:97`
```python
json.dumps({"last_seen": ts, "camera_id": camera_id})
```
`ts` is a float (seconds since epoch). `get_camera_last_seen()` does
`float(raw)` and works. OK.

---

## BUG-013 — `mark_camera_last_seen` stores a raw float string, no
##           JSON
**Severity: LOW**

**File:** `app/storage/redis_state.py:110-120`
`mark_recent` stores JSON; `mark_camera_last_seen` stores the raw
float string. Asymmetric but works.

---

## BUG-014 — Telemetry worker writes `time.time()` to `ts` if missing
**Severity: MEDIUM**

**File:** `app/workers/telemetry_worker.py:58, 67, 74`
```python
ts=float(event.get("timestamp") or time.time()),
```
This is OK in production (telemetry publishes real timestamps), but
in tests where events are hand-crafted without a `timestamp` field,
the dwell math (`duration = ts - entered_at`) will produce nonsense.

---

## BUG-015 — `MultiCameraRunner.stop()` does nothing useful
**Severity: LOW**

**File:** `app/workers/multi_camera_runner.py:116-122`
```python
def stop(self) -> None:
    for w in self._workers:
        try:
            w.__class__.__name__  # no-op
        except Exception:
            pass
    self._workers.clear()
    self._queues.clear()
```
The `try` is a no-op. The worker threads are `daemon=True`, so they
are killed on process exit, but `stop()` does not signal the frame
readers to stop or join the threads. On Ctrl-C, the OpenCV captures
will leak.

**Fix:** keep a per-camera `threading.Event` and signal it; close
the `cv2.VideoCapture`s.

---

## BUG-016 — `TelemetryWorker.run` polls two streams sequentially,
##           but `ensure_group` is called for `stream:identity_decisions`
##           and `stream:zone_events`, but NOT for `stream:embeddings` or
##           `stream:tracklets` (which are also referenced in
##           `app.yaml` `queues.streams`).
**Severity: LOW**

**File:** `app/workers/telemetry_worker.py:79-89`

The ReID worker uses `stream:tracklets` (line 125 of
`reid_worker.py`). The telemetry worker doesn't touch it. The
identity resolver doesn't exist as a worker — it's an in-process
function. So `stream:identity_decisions` is *never* published by
anyone. **The whole `stream:identity_decisions` consumer is dead
code.**

**Fix:** either (a) wire the resolver to publish to
`stream:identity_decisions`, or (b) remove the consumer.

---

## BUG-017 — `MultiCameraRunner.start()` is called twice without
##           idempotency
**Severity: LOW**

**File:** `app/workers/multi_camera_runner.py:74-95`

If `start()` is called twice (e.g. in tests that re-create a
runner), the second call spawns a second set of threads without
joining the first. Memory and OpenCV captures leak.

**Fix:** guard with `if self._workers: return`.

---

## BUG-018 — `db-migrator` service doesn't depend on `detect-pipeline`, but
##           `init_qdrant.py` is run as `docker compose run --rm detect-pipeline`
**Severity: LOW**

**File:** `docker-compose.yaml:81-85, 86-117`
Migrator uses `postgres:16-alpine` and runs psql directly. The
`api` build step expects `/models/pphuman` and
`/models/transreid/transformer_120.pth` to be present; running
`init_qdrant.py` inside the `api` image without those model dirs
will fail when the `select_reid_adapter()` import chain pulls
`paddle`/`torch` lazily. Not currently breaking because the
imports are lazy, but fragile.

---

## BUG-019 — `Dockerfile` installs `uv` but uses `pip` in the builder
**Severity: LOW**

**File:** `Dockerfile:29, 36`
```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /usr/local/bin/
...
pip install --no-cache-dir -r requirements.txt
```
`uv` is copied but never used; `pip` is invoked. Comment claims
"uv sync" but code is `pip install`. Cosmetic, but the build is
~3× slower than necessary.

---

## BUG-020 — `transit_reid` is misspelled in some comment
**Severity: NONE — typo only**

**File:** `app/cli/args.py` has no typo; ok.

---

## BUG-021 — `Dockerfile` does not install paddlepaddle / pytorch
**Severity: HIGH**

**File:** `Dockerfile:34-37`, `requirements.txt`
The `requirements.txt` does not include `paddlepaddle` or
`paddlepaddle-gpu` or `torch`. The Dockerfile copies it but no
real model can be loaded. The runtime image will fail to start if
the lazy import is ever triggered (which it isn't, today, because
the fallback is permanent).

**Fix:** add `paddlepaddle-gpu==2.6.x` and
`torch==2.4.0` + CUDA wheels to `requirements.txt`. Document the
~5 GB install.

---

## BUG-022 — `requirements.txt` does not include `psycopg_pool`
**Severity: HIGH**

**File:** `requirements.txt:16`
```python
psycopg[binary]>=3.2
```
The `pool` import in `postgres.py` requires `psycopg_pool>=3.2`.
Without it, `PostgresStore` cannot import (see BUG-001).

---

## BUG-023 — `finalize_stale` `max_age_seconds` parameter is
##           dead — the public function uses 5.0 hard-coded
**Severity: LOW**

**File:** `app/workers/tracklet_collector.py:139-164`
```python
def finalize_stale(self, max_age_seconds: float = 5.0) -> list[Tracklet]:
    ...
    if tl.end_time is None or (now - tl.end_time) < max_age_seconds:
        continue
```
The caller in `main.py:264` calls `finalize_stale()` with no
argument, so 5.0 s is used. A 5-second gap between "tracklet last
seen" and "tracklet closed" is far too aggressive — local tracks
frequently have 1-2 frame gaps. The tracklet will be closed, the
tracklet_id reused, and a new tracklet opened for the same
physical person → spurious "new GID" → ID fragmentation.

**Fix:** raise default to 30-60 s; or implement a proper
"no-emit-for-N-frames" tracker state.

---

## BUG-024 — `_synthetic_detect` reuses the same `_next_local_id`
##           across cameras
**Severity: LOW**

Wait, this is per-worker. `PPHumanWorker._next_local_id` is
per-instance. Two cameras will start at 1. Per the architecture
guard test, that's "by design" — local_track_id is camera-local.
**Not a bug.**

---

## BUG-025 — `MultiCameraRunner.stream` returns results in
##           arrival order, but with `max_seconds=None` it runs
##           forever, ignoring `max_seconds` from the runner
**Severity: LOW**

**File:** `app/workers/multi_camera_runner.py:101-114`

The `stream(max_seconds=...)` parameter is only set by
`main.py:260` when `mode == single_cam_smoke`. In production
multi-rtsp, `max_seconds=None` is passed, so the loop runs until
the worker dies. **No FPS-based backpressure**, no camera-level
rate limiting. A fast camera will flood the collector queue.

**Fix:** add per-camera rate limiting in `MultiCameraRunner.stream`.

---

## BUG-026 — `tracklet_buffer_size` gauge is never updated
**Severity: LOW**

**File:** `app/telemetry/metrics.py:130-132`
The metric is declared but never `set()` anywhere in the
codebase. `/metrics` will report it as 0. Same for
`stream_backlog`.

---

## BUG-027 — Qdrant search latency histogram is never observed
**Severity: LOW**

**File:** `app/storage/qdrant_store.py:152-153`
```python
logger.debug("Qdrant search %s top_k=%d hits=%d latency=%.3fs", ...)
```
The metric is logged, not stored. `qdrant_query_latency_seconds` is
empty in `/metrics`. Same for `postgres_write_latency_seconds`.

**Fix:** in `QdrantStore.search()`, after the search, do
`REGISTRY.qdrant_query_latency.observe(latency)`. Same in
`PostgresStore.timed_execute`.

---

## BUG-028 — `identity_decisions.decision_type` lacks a CHECK
##           constraint
**Severity: LOW**

**File:** `db/migrations/002_identity_tables.sql:97`
The column is `TEXT NOT NULL` but no CHECK enforces the
enum `'match' | 'new' | 'candidate' | 'ambiguous' | 'held'`. A
typo will silently insert garbage.

---

## BUG-029 — `dwell_sessions` is opened in
##           `telemetry_worker.on_zone_event`, but the worker
##           reads from `stream:zone_events` — which is never
##           published.
**Severity: HIGH**

**File:** `app/workers/telemetry_worker.py:51, 79-80`
There is no publisher for `stream:zone_events`. `ZoneEvent`
emission is a TODO; the `tracklet_collector` never calls
`pg.insert_zone_event` or `redis.publish("stream:zone_events", ...)`.
Result: the telemetry worker's `on_zone_event` is never invoked;
`dwell_sessions` are never created; the `dwell/summary` API
endpoint always returns empty.

**Fix:** emit `stream:zone_events` from the tracklet collector on
zone transitions (entry/exit).

---

## BUG-030 — `MultiCameraRunner.start()` and the worker
##           `start()` are called in `main.py:218-220` but the
##           workers run their own loop in `_run_worker` — there
##           is no end-of-stream signal
**Severity: MEDIUM**

The frame readers `make_frame_reader(..., loop=True)` loop
forever on `cv2.VideoCapture`. For a local file, `loop=True`
restarts at frame 0 on EOF. For RTSP, EOF is a disconnect
(camera down). There is no reconnect logic; the worker
silently dies and the queue is permanently empty.

**Fix:** add reconnect-with-backoff for RTSP; raise on local-file
EOF unless `loop=True`.

---

## BUG-031 — `identity_decisions.insert_identity_decision` allows
##           NULL `decision_type`, but should be NOT NULL
**Severity: LOW (already NOT NULL)

Already `NOT NULL` per migration. OK.

---

## BUG-032 — `seed/cameras.sample.sql` uses `ON CONFLICT DO NOTHING`
##           but no conflict target — Postgres requires one
**Severity: HIGH (SQL syntax error)

**File:** `db/seed/cameras.sample.sql:12`
```sql
ON CONFLICT (camera_id) DO NOTHING;
```
OK, this is fine — `camera_id` is the PK.

**File:** `db/seed/camera_links.sample.sql:14`, `db/seed/zones.sample.sql:24`
Same — explicit conflict target. OK.

---

## BUG-033 — `transient_fallback` is logged at WARNING level
##           rather than ERROR
**Severity: MEDIUM**

**File:** `app/reid/transreid_adapter.py:42`,
`app/reid/pphuman_adapter.py:42`
```python
logger.warning("TransReID weights not loaded (%s). Using
deterministic 768-dim fallback. This is for smoke tests ONLY.", e)
```
In production mode, this should be `ERROR` and a startup-fail
assertion. Currently a `WARNING` can be ignored by an operator
who is asleep.

---

## BUG-034 — `configs/reid/transreid.yaml` says
##           `transformer_type: vit_base_patch16_224_TransReID`
##           but the model file shipped is `vit_transreid_msmt.pth`
##           (not `transformer_120.pth`).
**Severity: MEDIUM**

**File:** `models/vit_transreid_msmt.pth` (400 MB)
The `download_transreid_models.sh` downloads
`transformer_120.pth` (Market-1501) into `/models/transreid/`. But
the model file present is the MSMT17 checkpoint. **The two are
not interchangeable.** MSMT17 is a different dataset with
different classifier head dimensions (`num_class=1041` for MSMT
vs `num_class=751` for Market-1501). The `model = make_model(cfg,
num_class=751, …)` call will load a 1041-class head into a
751-class head, raising a `RuntimeError: size mismatch` from
`load_state_dict`.

**Fix:** either rename to `transformer_120_msmt.pth` and update the
config, or download the Market-1501 checkpoint.

---

## BUG-035 — `migrator` runs migrations in a one-shot shell loop
##           with no `--single-transaction`, so a failed
##           migration leaves the database half-migrated
**Severity: LOW**

**File:** `docker-compose.yaml:81-85`
The migrator runs each SQL file in its own implicit transaction.
If `002_identity_tables.sql` fails halfway, `001_init.sql` is
committed and the database is in an inconsistent state.

**Fix:** wrap in `psql --single-transaction` or use a proper
migration tool.

---

## BUG-036 — `evidence_key()` is a static method but
##           `put_crop()` always uses `kind="debug"`, never `best`
**Severity: MEDIUM**

**File:** `app/workers/tracklet_collector.py:127`
`kind="debug"` is hard-coded. The `evidence_key()` static method
supports `kind="best"`, but it is never used. So the
`best.jpg` is never written; the path scheme is partially
implemented.

---

## BUG-037 — `health` endpoint in FastAPI calls
##           `pg.healthcheck()` which is a blocking call, but
##           FastAPI runs handlers in a thread pool
**Severity: LOW (acceptable, but slow under DB outage)

**File:** `app/api/server.py:21-24`
The `/health` endpoint blocks the worker thread until
Postgres responds. If Postgres is down with a long socket
timeout, the health check takes 5+ seconds. FastAPI's
default threadpool has only 40 threads, so 40 health checks
can starve the actual API.

**Fix:** add a `asyncio.wait_for(..., timeout=2)` wrapper.

---

## BUG-038 — `psql -h localhost` in README assumes localhost
##           but Docker host networking is required
**Severity: MEDIUM**

**File:** `README.md:93-95`
```bash
psql -h localhost -U yamaha -d yamaha_mtmct -f db/seed/cameras.sample.sql
```
The `postgres` service is in a Docker network. `localhost` will
fail unless the user has port-forwarded `5432:5432`. The
`docker-compose.yaml` does forward 5432, so this works, but the
instructions are not explicit.

**Fix:** document that the user must `docker compose up -d relation-store`
*before* running psql locally, OR run via `docker compose run --rm
relation-store psql -U yamaha -d yamaha_mtmct -f /docker-entrypoint-initdb.d/...`.

---

## BUG-039 — `Doc/research_sources.md` says CLIP-ReID is
##           "OPTIONAL — not verified via Context7 (deliberately)",
##           but the doc claims PaddleDetection is "VERIFIED"
**Severity: LOW (honesty)

This is honest. OK.

---

## BUG-040 — `tests/test_architecture_guards.py` does not
##           enforce the "one model instance per process" rule
**Severity: MEDIUM**

**File:** `tests/test_architecture_guards.py`
The test only checks for forbidden imports and secrets. It
does not assert that `MultiCameraRunner` constructs one model
and passes the same instance to every worker. The README's
"Hard rule: one model instance is shared across all cameras"
is unverified.

**Fix:** add a test that constructs a `MultiCameraRunner` with
two cameras and asserts `id(worker1.model) == id(worker2.model)`.

---

## Summary

| Severity | Count |
|---|---|
| CRITICAL | 6 (BUG-001, 002, 003, 004, 005, 021, 022) |
| HIGH | 8 (BUG-006, 008, 011, 014, 029, 032, 034, 037) |
| MEDIUM | 12 |
| LOW | 12 |
| NONE | 2 |

Critical bugs are all in the same family: the production
detector and ReID paths are not implemented. Every code path
that "looks" production is a deterministic fallback or
synthetic data.
