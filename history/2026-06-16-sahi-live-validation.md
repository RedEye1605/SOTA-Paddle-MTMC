# SAHI Live Validation — 2026-06-16

## Goal

Verify the SAHI integration improves detection recall on the
real 2h Yamaha production videos (cam1_merged.mp4, cam2_merged.mp4)
without disrupting the 15 fps HLS path or the persistent-ID chain.

## Architecture (operator-approved, 2026-06-16)

Auxiliary SAHI stream + `SAHITrackletBridge`. SAHI does NOT
modify SDE_Detector or any PaddleDetection internal. SAHI
detections become provisional tracklets with `source="sahi"`,
deduplicated against active PP-Human tracks via IoU matching
within the last 1 second.

## Step 1 — Baseline (SAHI off, 30 minutes)

```bash
# In .env:
SAHI_ENABLED=false

cd /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC
docker compose up -d --force-recreate api
sleep 60  # let the chain stabilize
```

Record (30 minutes later):
```bash
docker exec -it redis redis-cli XLEN stream:detections
docker exec -it redis redis-cli XLEN stream:tracklets
docker exec -it redis redis-cli XLEN stream:identity_decisions
docker exec -it postgres psql -U yamaha -d yamaha_mtmct \
    -c "SELECT COUNT(DISTINCT global_id) FROM tracklets WHERE global_id IS NOT NULL;"
```

Save these 4 numbers as `BASELINE_*` in this report.

## Step 2 — SAHI on, low rate (30 minutes)

```bash
# In .env:
SAHI_ENABLED=true
SAHI_RATE_LIMIT_HZ=2

docker compose up -d --force-recreate api
sleep 60
```

Record the same 4 numbers as `LOW_RATE_*`. Also:
```bash
docker compose logs api | grep "sahi.worker:" | tail -30
```

## Step 3 — SAHI on, default rate (1 hour)

```bash
# In .env:
SAHI_ENABLED=true
SAHI_RATE_LIMIT_HZ=5

docker compose up -d --force-recreate api
sleep 60
```

Record the same 4 numbers as `DEFAULT_RATE_*`. Also check:
```bash
docker compose logs api | grep -i "fatal\|error" | tail -30
```

## Acceptance

- DEFAULT_RATE detections per minute ≥ 1.10 × BASELINE
- DEFAULT_RATE match decisions per minute ≥ 1.10 × BASELINE
- DEFAULT_RATE global_id count ≤ BASELINE (more dedup)
- 15 fps HLS path uninterrupted (visually verify the HLS stream)
- No FATAL log lines over 1 hour
- SAHI tracklets are marked `source="sahi"` and `provisional=True`
  in `stream:tracklets` (verify with `redis-cli XRANGE`)

## Rollback

If any acceptance criterion fails:
1. Set `SAHI_ENABLED=false` in `.env`.
2. `docker compose restart api`.
3. Total rollback time: <30 seconds.
4. The chain reverts to PP-Human-only behavior.
5. No code change, no image rebuild, no migration.
