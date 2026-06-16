# External services setup

This document explains how `SOTA-Paddle-MTMC` talks to the operator's
**external** MinIO, MQTT, and MediaMTX services — the same managed
endpoints that the upstream `Service/offline-people-counting` stack
already uses.

> The local Docker Compose stack (detect-pipeline + relation-store +
> vector-store + message-bus) is for **development only**. The
> internal minio service was removed; all evidence uploads go to the
> operator's external MinIO cluster at
> `minio.example.invalid:9000` via `MINIO_ENDPOINT` in `.env`.
> Production runs the same detect-pipeline image, but with the
> container's `MINIO_*`, `MQTT_*`, and `MEDIAMTX_*` env vars pointing
> at the operator's managed services.

## 1. Service addresses (operator-managed)

The following are the addresses SOTA has been wired to in the
provided `.env` (see `SOTA-Paddle-MTMC/.env`):

| Service | Address | Notes |
| --- | --- | --- |
| MinIO | `minio.example.invalid:9000` | external, HTTP (port 9000), `MINIO_SECURE=false` |
| MQTT broker | `mqtt.example.invalid:1883` | external, plain MQTT |
| MediaMTX | operator-managed host | not used by default (see Phase 6 doc) |

These are the same addresses `Service/offline-people-counting/.env`
uses. **Do not copy the password values** from Service — SOTA's
`.env` already has its own credentials.

## 2. MinIO — three buckets

`SOTA-Paddle-MTMC` deliberately splits evidence / reports / models
into three named buckets:

| Bucket env var | Default | Used for |
| --- | --- | --- |
| `MINIO_BUCKET_EVIDENCE` | `evidence` | person crops, debug frames, best.jpg |
| `MINIO_BUCKET_REPORTS`  | `reports`  | benchmark JSON, visualization MP4 sidecars |
| `MINIO_BUCKET_MODELS`   | `models`   | reserved for future model artefact sync |

The buckets are **operator-managed**. We do not create them
implicitly in production unless `MINIO_CREATE_BUCKETS=true` is set
explicitly. Each `put_*` call goes to the bucket selected by the
caller, not a single hardcoded bucket.

Object paths are deterministic:

```text
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/debug_{frame_id}.jpg
visualization/{site_id}/{camera_id}/{yyyy}/{mm}/{dd}/first_3000_frames.mp4
reports/{site_id}/{yyyy}/{mm}/{dd}/benchmark_{timestamp}.json
```

The path scheme is the same as the upstream Service's path scheme
(prefix-per-bucket) but more deeply nested so a per-day, per-tracklet
listing is straightforward.

## 3. MQTT / ThingsBoard

We publish to:

- `v1/devices/me/telemetry` (default, when no token is set)
- `v1/devices/<THINGSBOARD_DEVICE_TOKEN>/telemetry` (token-based)

The payload is the canonical ThingsBoard envelope:

```json
{ "ts": 1710000000000, "values": { ... } }
```

Per the user spec the values block carries:

- `cam_id`, `zone_id`
- `people_count`, `entries`, `exits`
- `dwell_avg_seconds`, `active_global_ids`

Auth: `MQTT_USERNAME` + `MQTT_PASSWORD`, **or** `THINGSBOARD_DEVICE_TOKEN`
(used as the password). TLS is opt-in via `MQTT_TLS_ENABLED=true`.

## 4. MediaMTX

MediaMTX is opt-in. Set:

```text
MEDIAMTX_ENABLED=true
MEDIAMTX_HOST=...
MEDIAMTX_RTSP_PORT=8554
MEDIAMTX_STREAM_PREFIX=sota-paddle-mtmc
```

Per camera, the streamer publishes to:

```text
rtsp://{MEDIAMTX_HOST}:{MEDIAMTX_RTSP_PORT}/{MEDIAMTX_STREAM_PREFIX}/{camera_id}
```

For example, `CAM_01` → `rtsp://host:8554/sota-paddle-mtmc/CAM_01`.

See `Docs/mediamtx_streaming_setup.md` for the full command set.

## 5. Quick start (production)

```bash
cd /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC

# 1. Verify env
grep -E "^(MINIO_ENDPOINT|MQTT_BROKER_HOST|MQTT_HOST|MQTT_USERNAME|MQTT_PASSWORD|THINGSBOARD_DEVICE_TOKEN|MEDIAMTX_HOST)=" .env

# 2. Run production preflight (refuses READY_FOR_LIMITED_PRODUCTION
#    if external services are unreachable)
uv run python scripts/readiness_preflight.py

# 3. Visualize 3000 frames per camera (Phase 8/9)
uv run python scripts/generate_visual_validation.py \
  --cam CAM_01 \
  --input "$(find data -maxdepth 1 -type f -name 'cam1_merged*' | head -n 1)" \
  --max-frames 3000 \
  --output reports/visualization/CAM_01_first_3000_frames.mp4
```

## 6. Disabling external services

| Service | Disable by |
| --- | --- |
| MinIO | set `MINIO_ENABLED=false` in `configs/app.yaml` |
| MQTT | set `MQTT_ENABLED=false` in `configs/app.yaml` (env) |
| MediaMTX | set `MEDIAMTX_ENABLED=false` |

The preflight refuses `READY_FOR_LIMITED_PRODUCTION` when telemetry is
required but unreachable.
