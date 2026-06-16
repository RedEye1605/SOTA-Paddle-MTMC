# MTMCT Operational Runbook

## Day-1: bring up the stack

```bash
# 1. Clone and configure
cd SOTA-Paddle-MTMC
cp .env.example .env
# Edit .env with credentials (NEVER commit)

# 2. Start infra (PostgreSQL, Qdrant, Redis)
#    MinIO is external (operator-managed at minio.example.invalid:9000);
#    no local minio container.
docker compose up -d relation-store vector-store message-bus

# 3. Run migrations
docker compose run --rm db-migrator

# 4. Initialize Qdrant collections + payload indexes
docker compose run --rm detect-pipeline python scripts/init_qdrant.py

# 5. Download PP-Human + TransReID model weights
bash scripts/download_pphuman_models.sh
bash scripts/download_transreid_models.sh

# 6. Seed cameras, zones, camera_links from your site survey
psql -h localhost -U yamaha -d yamaha_mtmct -f db/seed/cameras.sample.sql
psql -h localhost -U yamaha -d yamaha_mtmct -f db/seed/zones.sample.sql
psql -h localhost -U yamaha -d yamaha_mtmct -f db/seed/camera_links.sample.sql

# 7. Smoke test (single camera, 30 s)
docker compose run --rm detect-pipeline python main.py --mode single_cam_smoke

# 8. Full multi-cam
docker compose up -d detect-pipeline
```

## Day-2: monitor and tune

```bash
# Health
curl http://localhost:8000/health

# Per-camera FPS, GPU memory, Qdrant latency
curl http://localhost:8000/metrics

# Recent decisions
curl 'http://localhost:8000/identity/decisions?limit=50'

# Identity lookup
curl 'http://localhost:8000/identity/GID-12345?include_tracklets=true'
```

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| All global_ids are new (no matches) | ReID threshold too high, or ReID model not loaded | Check `/metrics` for `reid_extractions_total`; lower `auto_match_threshold` |
| Same person gets 2+ GIDs in the same camera | Local track fragmentation + crop quality issue | Raise `min_track_age_frames`, raise `min_crops_per_tracklet` |
| Matches across impossible camera pairs | `camera_links` table empty | Populate `camera_links` from your site survey |
| Stream backlog growing | ReID worker saturated | Lower `max_crops_per_tracklet`, batch ReID more aggressively |
| GPU OOM | CLIP-ReID accidentally enabled | Set `reid.optional_benchmark: clipreid → active: false` |
| Qdrant queries slow | Missing payload indexes | Re-run `scripts/init_qdrant.py` |
| `identity_decisions` writes slow | PostgreSQL under-sized | Check `postgres_write_latency_seconds`; raise DB instance |

## Maintenance

- **Daily**: review `identity_decisions` for `ambiguous` outcomes.
- **Weekly**: review `identity_merge_audit` (should be operator-driven only).
- **Monthly**: re-tune `auto_match_threshold` based on the score histogram.
- **Quarterly**: backfill embeddings if the ReID model was upgraded.
