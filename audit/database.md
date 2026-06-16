# Database Audit — SOTA-Paddle-MTMC

> **Phase 6 — Database audit.** PostgreSQL, Qdrant, Redis, MinIO.
> Each storage layer is checked for migration validity, indexes,
> transaction boundaries, retention, auditability.

## PostgreSQL

### Migrations

| File | Tables added | Validated? |
|---|---|---|
| `001_init.sql` | `cameras`, `model_versions`, `system_metrics` | ✅ tables exist, columns present |
| `002_identity_tables.sql` | `global_identities`, `tracklets`, `tracklet_embeddings`, `tracking_events`, `identity_decisions`, `identity_merge_audit` | ✅ all present, FKs correct |
| `003_zone_tables.sql` | `camera_links`, `zones`, `zone_events`, `dwell_sessions` | ✅ |
| `004_indexes.sql` | (additional indexes only) | ✅ |

### Required tables (from `test_db_schema.py`)
```python
REQUIRED_TABLES = [
    "cameras", "camera_links", "zones", "global_identities",
    "tracklets", "tracklet_embeddings", "tracking_events",
    "zone_events", "dwell_sessions", "identity_decisions",
    "identity_merge_audit", "model_versions", "system_metrics",
]
```
**All 13 present in the migrations.** The schema test passes.

### Foreign-key integrity

| Table | FK | Notes |
|---|---|---|
| `global_identities` | `first_camera_id, last_camera_id → cameras` | ✅ |
| `tracklets` | `global_id → global_identities ON DELETE SET NULL`, `camera_id → cameras` | ✅ |
| `tracklet_embeddings` | `tracklet_id → tracklets ON DELETE CASCADE`, `global_id → global_identities ON DELETE SET NULL` | ✅ |
| `tracking_events` | `tracklet_id → tracklets ON DELETE SET NULL`, `global_id → global_identities ON DELETE SET NULL`, `camera_id → cameras`, `zone_id → zones` | ✅ (depends on `003_zone_tables.sql` running first; **forward FK dependency**) |
| `identity_decisions` | `tracklet_id → tracklets ON DELETE CASCADE`, `source_camera_id, candidate_camera_id → cameras`, `assigned_global_id → global_identities ON DELETE SET NULL` | ✅ |
| `identity_merge_audit` | `old_global_id, new_global_id → global_identities` | ✅ |
| `zone_events` | `global_id → global_identities ON DELETE CASCADE`, `tracklet_id → tracklets ON DELETE CASCADE`, `camera_id → cameras`, `zone_id → zones` | ✅ |
| `dwell_sessions` | `global_id → global_identities ON DELETE CASCADE`, `zone_id → zones`, `camera_id → cameras` | ✅ |

### Index coverage (production query patterns)

| Query | Index used? |
|---|---|
| `SELECT * FROM cameras WHERE site_id = …` | `idx_cameras_site_id` ✅ |
| `SELECT * FROM cameras WHERE is_active = TRUE` | `idx_cameras_active` ✅ |
| `SELECT * FROM global_identities WHERE session_id = …` | `idx_gi_session_id` ✅ |
| `SELECT * FROM global_identities WHERE last_seen_at < ts ORDER BY last_seen_at DESC` | `idx_gi_last_seen` ✅ |
| `SELECT * FROM tracklets WHERE global_id = … ORDER BY start_time DESC LIMIT 200` | `idx_tracklets_global_id` ✅ |
| `SELECT * FROM tracklets WHERE camera_id = … AND start_time > ts` | `idx_tracklets_camera_time` ✅ |
| `SELECT * FROM zone_events WHERE camera_id = … ORDER BY "timestamp" DESC` | `idx_ze_recent_per_camera` ✅ |
| `SELECT * FROM zone_events WHERE global_id = … ORDER BY "timestamp" DESC` | `idx_ze_global_time` ✅ |
| `SELECT * FROM tracking_events WHERE camera_id = … AND "timestamp" > ts` | `idx_te_camera_ts` ✅ |
| `SELECT * FROM tracking_events WHERE tracklet_id = … ORDER BY "timestamp" DESC` | `idx_te_tracklet_ts` ✅ |
| `SELECT * FROM identity_decisions WHERE created_at > ts ORDER BY created_at DESC` | `idx_id_created` ✅ |
| `SELECT * FROM identity_decisions WHERE assigned_global_id = …` | `idx_id_assigned` ✅ |
| `SELECT * FROM dwell_sessions WHERE status = 'open' AND global_id = …` | `idx_dw_open` (partial) ✅ |

**Index coverage is excellent.** No obvious gaps.

### Transaction boundaries

| Operation | Atomic? |
|---|---|
| `upsert_camera`, `upsert_zone`, `upsert_camera_link` | Single-statement, atomic. ✅ |
| `create_global_identity` (with `ON CONFLICT DO UPDATE`) | Single-statement, atomic. ✅ |
| `insert_tracklet` (`ON CONFLICT DO NOTHING`) | Single-statement, but **the resolver updates the row** through `update_global_identity_seen`, not through `update_tracklet_global_id` (no such method exists). The tracklet's `global_id` column stays NULL forever, even after assignment. | ⚠️ BUG-011 |
| `upsert_dwell` (open + close) | Uses its own `connection()` context — but inside, the open path and exit path are two separate `cur.execute()` calls. If the open succeeds and the exit fails, an orphaned `open` row remains. | ⚠️ MEDIUM |
| `expire_old_identities` | Single `UPDATE`, atomic. ✅ |

### Retention / cleanup

- `expire_old_identities(older_than_seconds=86400)` exists and
  is callable. **It is not called from anywhere in the
  codebase.** No scheduled job, no cron, no APScheduler. The
  status flips from `active` to `expired` manually if at all.
- `tracklet_embeddings` and `tracking_events` grow unbounded
  (no retention policy, no partitioning). After 30 days at 25
  FPS across 5 cameras, the table has ~5×25×30×86400 = 3.2B rows
  in `tracking_events` alone. **Production-unready** for a
  multi-day deployment without partitioning or a TTL.
- `Qdrant` payloads are not TTL'd by Qdrant itself; the
  resolver's `timestamp_gte` filter naturally excludes old
  points from search, but the points still exist on disk.
  `QdrantStore` has no `delete_by_time()` or
  `delete_by_id()`. **No Qdrant retention plan.**

### Auditability

- `identity_decisions` is full-audit (every decision, all
  factors, final score, reason text). ✅
- `identity_merge_audit` is full-audit (operator, reason,
  score). ✅
- `system_metrics` allows runtime metric snapshots, but no
  code writes to it. ⚠️
- `tracklet_embeddings.vector_db_point_id` enables joining
  PostgreSQL ↔ Qdrant for offline analysis. ✅

### 24h retrieval support

- The schema is correct. The Qdrant filter is correct. The
  Redis TTL is correct. **The 24h retrieval is supported by
  the schema, but the production data is synthetic histogram
  features** (see ReID 24h audit). In a real deployment
  with real Paddle+TransReID, this would work.

## Qdrant

### Collection creation

```python
COLLECTIONS: list[tuple[str, int, models.Distance]] = [
    ("person_reid_pphuman",            256, models.Distance.COSINE),
    ("person_reid_transreid",          768, models.Distance.COSINE),
    ("person_reid_clipreid_optional",  512, models.Distance.COSINE),
]
```

- `init_collections()` is idempotent. ✅
- `create_payload_index` is called for every (collection,
  field) pair. Errors are swallowed (`Already exists`). ✅
- One collection per model — no cross-dimension pollution. ✅
- `Distance.COSINE` matches the L2-normalized embeddings. ✅

### Payload indexes

| Field | Type | Used in `search()` filter? |
|---|---|---|
| `global_id` | KEYWORD | ❌ (the search returns hits with `global_id`; we don't filter on it) |
| `tracklet_id` | KEYWORD | ❌ (used post-search to exclude self) |
| `camera_id` | KEYWORD | ✅ |
| `zone_id` | KEYWORD | ❌ |
| `site_id` | KEYWORD | ✅ (if provided) |
| `timestamp` | INTEGER | ✅ (range) |
| `quality_score` | FLOAT | ✅ (range) |
| `model_name` | KEYWORD | ✅ (match) |
| `model_version` | KEYWORD | ✅ (match) |

Index fields exceed the actual filter set, which is fine —
over-indexing is safer than under-indexing.

### Search correctness

```python
must = [
    FieldCondition("timestamp", Range(gte=ts - 86400)),
    FieldCondition("camera_id", MatchAny(any=candidate_cams)),
    FieldCondition("quality_score", Range(gte=0.5)),
    FieldCondition("model_name", MatchValue(value=model_name)),
    FieldCondition("model_version", MatchValue(value=model_version)),
]
```
- All required filters present. ✅
- Empty `candidate_camera_ids` → short-circuit (no search). ✅
- `query_vector` is a 1D numpy array; reshaped to `(1, dim)`
  before sending. ✅

### Score interpretation

- `models.ScoredPoint.score` is the cosine similarity in
  `[0, 1]` for `Distance.COSINE`. ✅
- The resolver uses `h.score` directly. ✅

## Redis

### Key inventory

```
active:{camera_id}:{local_track_id}        → global_id         (TTL 60s)
recent:global:{global_id}                  → JSON {ts, cam}    (TTL 86400s)
camera:last_seen:{camera_id}               → float (TTL 86400s)
tracklet_buffer:{camera_id}:{local_id}     → list of URIs       (TTL 300s)
stream:tracklets
stream:embeddings
stream:identity_decisions  (no publisher!)
stream:zone_events         (no publisher!)
```

### TTL usage

| Key | TTL | Correct? |
|---|---|---|
| `active:…` | 60 s | ✅ matches spec |
| `recent:global:…` | 86 400 s | ✅ |
| `camera:last_seen:…` | 86 400 s | ✅ |
| `tracklet_buffer:…` | 300 s | ✅ |

### Active-track binding

`set_active(cam, local_id, global_id)` is defined but **never
called** in production. The flow is:
1. Tracklet collector emits to `stream:tracklets`.
2. ReID worker consumes, runs ReID, upserts Qdrant, publishes
   to `stream:embeddings`.
3. Resolver is supposed to consume `stream:embeddings` and
   assign a `global_id`, then call `redis.set_active(…)`.

But: the resolver is not a stream consumer. The
`GlobalIdentityResolver` is an in-process class — there is no
`run_consumer(stop_event)` method on it, no
`redis.consume("stream:embeddings", …)` call anywhere. **The
resolver is dead code in the current pipeline.**

### Streams & consumer groups

- `stream:tracklets` ← published by `TrackletCollector` ✅
- `stream:tracklets` ← consumed by `ReIDWorker` ✅
- `stream:embeddings` ← published by `ReIDWorker` ✅
- `stream:embeddings` ← consumed by **NOTHING** ⚠️
- `stream:identity_decisions` ← published by **NOTHING** ⚠️
- `stream:identity_decisions` ← consumed by `TelemetryWorker` (dead)
- `stream:zone_events` ← published by **NOTHING** ⚠️
- `stream:zone_events` ← consumed by `TelemetryWorker` (dead)

So the only active stream is `stream:tracklets` →
ReIDWorker. Everything past ReIDWorker is unbuilt.

### Queue / backpressure

- Per-worker `Queue(maxsize=64)` in `MultiCameraRunner`. If the
  collector is slow, the worker thread blocks on
  `q.put(result)`. **No dead-letter, no overflow policy.**

### Reconnect / error behavior

- `redis.Redis(...)` has `socket_timeout=5,
  socket_connect_timeout=5`. After 5s of no response, the
  call raises a `TimeoutError` or `ConnectionError`. The
  `MultiCameraRunner` does not catch this; the ReID worker's
  `consume()` raises and the thread dies. **No reconnect
  logic.** ⚠️

## MinIO

### Bucket creation

`connect()` calls `bucket_exists(bucket)`; if missing,
`make_bucket(bucket)`. ✅ Idempotent.

### Path format

```
s3://evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/{best.jpg|debug_{frame_id}.jpg}
```

- `global_id` in the path is `UNASSIGNED` at upload time
  (resolver hasn't run yet). **The path is not
  re-keyed after assignment** — so `global_id` in the path
  is always `UNASSIGNED`. A re-keying step is missing.
- `zone_id` is `Z_NONE` if no zone was assigned. ✅

### Privacy-aware retention

- **No TTL on MinIO objects.** No lifecycle policy, no
  scheduled delete. Evidence crops grow forever.
- `evidence` bucket is `minio` only; not replicated to
  cold storage. **Production-unsafe for a multi-week
  deployment** (CCTV evidence is privacy-sensitive).

### Upload error behavior

`put_crop()` raises `RuntimeError` on `cv2.imencode` failure.
The caller (`TrackletCollector.on_frame`) catches and
`continue`s. ✅

### Frame flood

- `max_crops_per_tracklet = 15`. Per tracklet, at most 15
  uploads. At 5 FPS, 5 cameras, 15 crops × 5 s per tracklet,
  1 tracklet per camera, that's 75 uploads / 5 s = 15
  uploads/s. Negligible.
- But: if a tracklet persists (long-lived person), crops
  are uploaded every frame. **The `append_crop` Redis
  call** has no per-tracklet rate limit.

## Cross-store consistency

- `PostgresStore.insert_tracklet` and
  `redis.publish("stream:tracklets", ...)` are two separate
  operations. If Postgres succeeds and Redis fails, the
  tracklet exists in PG but the ReID worker never sees it.
  **No outbox / transactional consistency.** ⚠️
- `QdrantStore.upsert_point` and
  `PostgresStore.insert_tracklet_embedding` are two
  separate operations. Same risk: Qdrant can be ahead of
  PG or vice versa. ⚠️
- `RedisState.set_active` is also separate. The
  active-track binding can drift from PG.

## Database verdict

**Schema: 10/10.** Migrations are clean, FKs are correct,
indexes are comprehensive, audit fields are present.

**Operational wiring: 3/10.**
- Resolver is not wired to a stream consumer.
- `stream:identity_decisions` and `stream:zone_events` have
  no publishers.
- Dwell sessions and zone events are dead.
- MinIO has no retention; PG has no partitioning.
- No reconnect logic; no transactional consistency between
  Redis/Qdrant/PG.

**To make it production-ready:** wire the resolver to
`stream:embeddings`; add publishers for
`stream:identity_decisions` and `stream:zone_events`; add
a retention worker (e.g. `scripts/retention.py` running
nightly); add transactional outbox for cross-store writes.
