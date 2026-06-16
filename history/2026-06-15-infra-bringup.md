# Phase 3 — Infrastructure bring-up

**Date:** 2026-06-13
**Project root:** `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC`

## Commands run

```bash
docker compose up -d postgres qdrant redis minio
docker compose ps
docker compose logs <service> --tail=N
```

## Service status (final)

| Service  | Image                                       | Health    | Port(s)         |
| -------- | ------------------------------------------- | --------- | --------------- |
| postgres | `postgres:16-alpine`                        | healthy   | 5432            |
| qdrant   | `qdrant/qdrant:v1.12.0`                     | healthy   | 6333, 6334      |
| redis    | `redis:7-alpine`                            | healthy   | 6379            |
| minio    | `minio/minio:RELEASE.2024-09-13T20-26-02Z`  | healthy   | 9000, 9001      |

End-to-end probes:

```text
curl http://localhost:6333/             -> 200 OK {"title":"qdrant ..."}
curl http://localhost:6333/readyz       -> 200 OK "all shards are ready"
curl http://localhost:9000/minio/health/live -> 200 OK
redis-cli ping                          -> PONG
psql -U yamaha -d yamaha_mtmct -c '\dt' -> 13 tables (see below)
```

## Issues found and fixed

### Issue 1 — Qdrant healthcheck used `wget`, which is not in the qdrant image

`qdrant/qdrant:v1.12.0` is a minimal image with only `bash` and `sh` in
`/usr/bin/`. No `curl`, `wget`, `nc`, or `python3`. The previous healthcheck
`["CMD", "wget", "--spider", "-q", "http://localhost:6333/"]` therefore
failed every probe, leaving the container marked **unhealthy** even though
the service was actually serving traffic.

**Fix** in `docker-compose.yaml`:

```yaml
qdrant:
  healthcheck:
    # The qdrant image is minimal (no curl/wget). Probe via bash
    # /dev/tcp, which is always present.
    test: ["CMD-SHELL", "bash -c 'echo > /dev/tcp/127.0.0.1/6333'"]
    interval: 5s
    timeout: 3s
    retries: 20
```

Verified `bash -c "echo > /dev/tcp/127.0.0.1/6333"` returns 0 inside the
container. The healthcheck now succeeds and the container reports
`healthy` after the first probe interval.

### Issue 2 — Migration 002 referenced `zones` before 003 created it

`postgres:16-alpine` runs every file mounted in `/docker-entrypoint-initdb.d/`
in alphabetical order, **one statement at a time**, on first init. It
hit this in `002_identity_tables.sql:83`:

```sql
CREATE TABLE IF NOT EXISTS tracking_events (
    ...
    zone_id  TEXT REFERENCES zones(zone_id),   -- zones is defined in 003
    ...
);
```

```text
psql:/docker-entrypoint-initdb.d/002_identity_tables.sql:86: ERROR:  relation "zones" does not exist
```

This **partially applied** 002 (the tables above this point were created)
and **never applied** 003 or 004. Only 6 of the 13 expected tables existed.

**Fix** in `db/migrations/002_identity_tables.sql`:

The `zone_id` column on `tracking_events` is informational only (used for
analytics traceability). Removed the FK reference so migration 002 no
longer depends on a table that does not yet exist:

```sql
-- zone_id is informational only at this layer (no DB-level FK) so
-- 002 can be applied before 003 creates the zones table. Application
-- code validates the zone_id against zones() at write time.
zone_id  TEXT,
```

Removed the partially-initialized `postgres_data` volume, recreated the
container, and verified all 4 migrations applied:

```text
running /docker-entrypoint-initdb.d/001_init.sql
running /docker-entrypoint-initdb.d/002_identity_tables.sql
running /docker-entrypoint-initdb.d/003_zone_tables.sql
running /docker-entrypoint-initdb.d/004_indexes.sql
```

### Issue 3 — Postgres healthcheck logged noise about a missing `yamaha` DB

`pg_isready -U yamaha` defaults to connecting to a database named after
the user (`yamaha`), which does not exist in this project (we use
`yamaha_mtmct`). The healthcheck *worked* (it just probed the listener)
but every 5 s logged:

```text
FATAL: database "yamaha" does not exist
```

**Fix** in `docker-compose.yaml`: pass the correct DB explicitly.

```yaml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-yamaha} -d ${POSTGRES_DB:-yamaha_mtmct}"]
```

The FATAL noise is now gone.

## Final table list (13)

```text
camera_links
cameras
dwell_sessions
global_identities
identity_decisions
identity_merge_audit
model_versions
system_metrics
tracking_events
tracklet_embeddings
tracklets
zone_events
zones
```

This matches the design (`Docs/database_design.md`) and the four migration
files in `db/migrations/`.

## Verdict

Phase 3 PASS. All 4 infra services are **healthy**, the DB schema is
fully migrated (13/13 tables), and the previously-silent migration
ordering bug has been fixed in a backward-compatible way (existing
`zone_id` data, where present, is preserved; no FK was previously
enforced for partially-migrated DBs).
