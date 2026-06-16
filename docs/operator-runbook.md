# Operator Runbook

> **Single-page operator guide for the Yamaha People Detection
> System. Each phase has: what to do, how to verify.**

## Phases

| Phase | Action | Verify |
|---|---|---|
| 0. Clone | `git clone … /app` | `ls app/` |
| 1. Env | `cp .env.example .env`; edit secrets | `grep -c change_me .env` returns 0 |
| 2. Paddle clone | `git clone https://github.com/PaddlePaddle/PaddleDetection /opt/paddledetection` | `test -f /opt/paddledetection/deploy/pipeline/pipeline.py` |
| 3. Paddle model | Download `mot_ppyoloe_l_36e_pipeline.zip` to `/models/pphuman`; unzip | `ls /models/pphuman/mot_ppyoloe_l_36e_pipeline/inference.pdmodel` |
| 4. ReID weight | (a) use shipped MSMT17 or (b) download Market-1501 | `python scripts/inspect_transreid_checkpoint.py <path> --json` |
| 5. Env | `SOTA_API_TOKEN=…` in `.env` | `grep SOTA_API_TOKEN .env` |
| 6. Infra up | `docker compose up -d relation-store vector-store message-bus` | `docker compose ps` shows all `healthy` |
| 7. Migrate | `docker compose run --rm db-migrator` | `docker compose exec relation-store psql -U yamaha -d yamaha_mtmct -c '\dt'` |
| 8. Qdrant init | `docker compose run --rm detect-pipeline python scripts/init_qdrant.py` | `curl localhost:6333/collections` shows 3 collections |
| 9. Start | `docker compose up -d detect-pipeline` | `docker compose ps detect-pipeline` shows `healthy` |
| 10. Preflight | `docker compose run --rm detect-pipeline python scripts/readiness_preflight.py` | exit 0, all checks OK |
| 11. Smoke bench | `docker compose run --rm detect-pipeline python scripts/benchmark_t4.py --mode smoke_benchmark --max-seconds 30` | `ls reports/benchmark_*.json` |
| 12. Gate | `python scripts/readiness_gate.py --min-verdict READY_FOR_SHADOW_TEST` | exit 0 |
| 13. Shadow deploy | route one showroom's traffic to the SOTA pipeline | watch `camera_status` in `/metrics` |
| 14. Production bench | record 30 min of multi-camera video; run production benchmark | gate verdict = `READY_FOR_LIMITED_PRODUCTION` |
| 15. Promote | switch the showroom to the SOTA pipeline as the source of truth | operator runbook |

## Verifications

| Check | Command |
|---|---|
| Tests pass | `python -m pytest tests/ -q` (target: 210 passed, 1 skipped) |
| Production refuses without models | `SOTA_RUNTIME_MODE=production docker compose run --rm detect-pipeline python -m app.main` (expect ProductionSafetyError) |
| Smoke allows synthetic | `ALLOW_SYNTHETIC_SMOKE_TEST=true SOTA_API_TOKEN=smoke docker compose run --rm detect-pipeline python -m app.main --mode smoke_test` (expect startup) |
| Prometheus metrics include per-camera labels | `curl localhost:8000/metrics` (look for `camera_fps{camera_id=…}`) |
| Healthcheck | `curl localhost:8000/health` (returns 200 + status json) |
| Healthcheck from Docker | `docker inspect --format='{{json .State.Health.Status}}' yamaha-mtmct-detect-pipeline-1` (expect `healthy`) |

## Failure recovery

| Symptom | First check | Fix |
|---|---|---|
| `READY_FOR_SHADOW_TEST` is `NOT_READY` | `cat reports/readiness.json` | fix the failing check; rerun |
| Camera status is 0 (offline) in `/metrics` | `docker logs yamaha-mtmct-detect-pipeline-1` | check RTSP URL; `camera_reconnects_total` shows attempts |
| `best_crop_uri` is `s3://evidence/pending/...` for hours | rekey worker may be down | restart detect-pipeline; check `stream:identity_decisions` for backlog |
| Qdrant latency p99 > 200 ms | `qdrant_query_latency_p99_ms` | increase `frame_queue_maxsize` or scale the resolver |
| False merge rate increases | `false_merge_rate` in latest benchmark | lower `auto_match_threshold` to 0.85 in `configs/app.yaml` |
| 401 from /identity/* | `SOTA_API_TOKEN` env var | set the token; restart detect-pipeline |

## SAHI Operations

SAHI (Slicing Aided Hyper Inference) augments the PP-Human detector
with a sliced detection pass to catch small/distant people that
PP-Human misses at 1920x1080 (bbox heights 30-50 px). It runs as a
background thread inside the detect-pipeline container.

### Architecture (operator-approved, 2026-06-16)

SAHIWorker (in-process) subscribes to clean MediaMTX RTSP, runs
the same PaddleDetection model on 320x320 patches, and publishes
raw SAHI detections to `stream:detections_sahi`. SAHITrackletBridge
consumes that stream, deduplicates against active PP-Human tracks
(IoU > 0.3 within the last 1 second), and emits NEW auxiliary
tracklets to the existing TrackletCollector. SAHI tracklets are
marked `source="sahi"` and `provisional=True`; the ReIDWorker and
GlobalIdentityResolver treat them as low-confidence.

The chain does NOT modify SDE_Detector or any PaddleDetection
internal. SAHI is purely additive.

### Enable / Disable

Production default is **OFF**. To enable per environment:

```bash
# in .env or docker-compose.yaml
SAHI_ENABLED=true
SAHI_RATE_LIMIT_HZ=2     # start low; raise to 5 for full rate
docker compose restart detect-pipeline
```

To disable:

```bash
SAHI_ENABLED=false
docker compose restart detect-pipeline
```

### Rollout phases

| Phase | Setting | Duration | Goal |
|---|---|---|---|
| 0 (deploy) | `SAHI_ENABLED=false` | 1 day | All existing tests pass. No behavior change. |
| 1 (smoke) | `SAHI_ENABLED=true SAHI_RATE_LIMIT_HZ=1` | 1 day | Verify logs. No FATAL. Detections +5%. |
| 2 (low) | `SAHI_ENABLED=true SAHI_RATE_LIMIT_HZ=2` | 2 days | Detections +10%. Matches +5%. 15 fps holds. |
| 3 (default) | `SAHI_ENABLED=true SAHI_RATE_LIMIT_HZ=5` | 1 week | Detections +15-25%. Matches +10-20%. |

### Monitoring

```bash
# Per-minute summary (grep the detect-pipeline container's stdout)
docker compose logs -f detect-pipeline | grep "sahi.worker:"

# Redis stream length (should be ≤ ~2000)
docker exec -it message-bus redis-cli XLEN stream:detections_sahi

# Latest snapshot for a camera (1s TTL)
docker exec -it message-bus redis-cli GET sahi:latest:CAM_01

# SAHI bridge consumer group state
docker exec -it message-bus redis-cli XINFO GROUPS stream:detections_sahi
```

### Troubleshooting

| Symptom | Action |
|---|---|
| False positives (motorcycle parts, posters) | `SAHI_ENABLED=false` |
| Rate too low (GPU contention) | Lower `SAHI_PATCH_SIZE=256` or raise `SAHI_RATE_LIMIT_HZ=10` |
| Rate too high (GPU saturated) | Lower `SAHI_RATE_LIMIT_HZ=2` |
| `stream:detections_sahi` > 5000 | `docker exec -it message-bus redis-cli XTRIM stream:detections_sahi MAXLEN 1000` |
| SAHIWorker thread dead (3x auto-restart) | `docker compose restart detect-pipeline` |
| Build fails with "SAHI PULLED TORCH" | A `sahi` version bump re-introduced torch. Pin to a different version or use `--no-deps`. |

### Acceptance (when is SAHI considered "working")

1. All tests in `tests/test_sahi_*` pass.
2. Detections per minute increase ≥ 10% vs SAHI-off baseline on the 2h Yamaha video.
3. Cross-camera `match` decisions increase ≥ 10%.
4. PP-Human's 15 fps HLS path is uninterrupted.
5. No FATAL log lines over 24 hours.
