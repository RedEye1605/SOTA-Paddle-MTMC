# Database Design

## PostgreSQL: source of truth

Migrations live in `db/migrations/`. The full DDL is split into:

- `001_init.sql` — `cameras`, `model_versions`, `system_metrics`
- `002_identity_tables.sql` — `global_identities`, `tracklets`, `tracklet_embeddings`,
  `identity_decisions`, `identity_merge_audit`
- `003_zone_tables.sql` — `zones`, `camera_links`, `zone_events`, `dwell_sessions`
- `004_indexes.sql` — secondary indexes (timestamp, camera_id, global_id)

## Qdrant: vector search

- **Collections** (one per active ReID model):
  - `person_reid_pphuman` (256-dim, cosine)
  - `person_reid_transreid` (768-dim, cosine)
  - `person_reid_clipreid_optional` (512-dim, cosine)
- **Payload indexes** (created on init via `scripts/init_qdrant.py`):
  - `global_id` (KEYWORD)
  - `tracklet_id` (KEYWORD)
  - `camera_id` (KEYWORD)
  - `zone_id` (KEYWORD)
  - `site_id` (KEYWORD)
  - `timestamp` (INTEGER, range queries)
  - `quality_score` (FLOAT, range queries)
  - `model_name` (KEYWORD)
  - `model_version` (KEYWORD)

## Redis: active state + streams

Keys (with TTLs):

```
active:{camera_id}:{local_track_id}       → global_id         (TTL 60s)
recent:global:{global_id}                 → last_seen_ts     (TTL 86400s)
camera:last_seen:{camera_id}              → last_seen_ts     (TTL 86400s)
tracklet_buffer:{camera_id}:{local_track_id} → list of crop URIs (TTL 300s)
```

Streams (with consumer groups):

```
stream:tracklets          — emitted by tracklet_collector, consumed by reid_worker
stream:embeddings         — emitted by reid_worker,       consumed by resolver
stream:identity_decisions — emitted by resolver,          consumed by telemetry
stream:telemetry          — emitted by telemetry_worker,  consumed by mqtt_client
```

## MinIO: evidence crops

```
s3://evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg
s3://evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/debug_{frame_id}.jpg
```

`best.jpg` is the single highest-quality crop of the tracklet.
`debug_{frame_id}.jpg` are diagnostic crops used for labeler reviews.
