# Phase 1 — Service Reference Mapping

Date: 2026-06-13
Source: `/home/rhendy/Projects/yamaha/Service/offline-people-counting` (read-only)
Target: `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC`

This document maps Service's proven patterns to the corresponding
SOTA modules we will adapt. **No Service file is modified.**
We copy *patterns*, never secrets.

## 1. Service files inspected

### I/O — MediaMTX / MQTT / MinIO
- `app/io/streamer.py` — per-camera `VideoStreamer` that pushes BGR
  frames to MediaMTX via an FFmpeg subprocess. Owns a bounded queue,
  daemon push loop, reconnect-with-exponential-backoff.
- `app/io/streamer_command.py` — pure builders:
  - `build_ffmpeg_command(...)` — argv for raw BGR24 → RTSP via libx264.
  - `build_publish_url(...)` — defaults to `rtsp://{host}:{port}/{camera_id}/live`.
  - `build_hls_url(...)`, `build_webrtc_url(...)`.
- `app/io/mqtt_connection.py` — paho-mqtt v2 wrapper with `username`/
  `password`/`TLS`, lazy env load via `os.getenv`, reconnect delay
  config, `connection_event`, `on_publish` callback.
- `app/io/mqtt_publisher.py` — async publisher: bounded queue, daemon
  retry thread, exponential backoff (cap 30 s, max 3 attempts),
  `publish_count` accounting, milestone logs.
- `app/io/mqtt_service.py` — facade combining connection + publisher.
- `app/io/mqtt_topics.py` — topic builder:
  `ai/yamaha/people-detection/{camera_id}/{summary,event,status,attributes}`.
- `app/io/minio_client.py` — S3 client lifecycle (`connect`,
  `_ensure_bucket`, `record_failure`/`record_success`/`try_reconnect`,
  `is_available` flag).
- `app/io/minio_uploader.py` — per-crop upload with retry/presign.
  Object path pattern: `{object_prefix}/{camera_id}/{slug-zone}/{date}/`
  `{unix_ms}_{person_id}.jpg` (e.g. `people-detection/cam1/main-hallway/2026-04-29/1234_5.jpg`).
- `app/io/minio_service.py` — facade.
- `app/io/history.py` — `HistoryService` queue + per-person payloads
  with `to_unix_ms()` and `build_person_payload()` returning
  `{ts, values}` (ThingsBoard shape).

### Engine — overlay + evidence
- `app/engine/overlay.py` — pure drawing: `draw_detection()`,
  `draw_hud()` (top-left status: device, time, FPS, per-zone counts,
  MQTT status), `draw_detection_box()` (corner-style box + label).
- `app/engine/evidence.py` — `EvidenceManager`: TTL cache of latest
  per-person crops + bounded async upload queue + daemon worker.

### Counting
- `app/counting/payload.py` — `PayloadBuilder` produces a
  `{ts, values}` per-zone telemetry dict with deltas (ThingsBoard
  SUM aggregation friendly).

### Config
- `config.yaml` — has explicit `mqtt:`, `minio:`, `mediamtx:` blocks,
  with `topic_base`, ports, `presigned_url_expiry_days`, `rtsp_port`,
  `hls_port`, `webrtc_port`, etc.
- `.env.example` — operational + secrets layout; deployment addresses
  blank by default and **filled in by the operator in `.env`**.

### Deployed hosts (from Service `.env`)
- MQTT: `mqtt.example.invalid`
- MinIO: `minio.example.invalid`
- MediaMTX: `198.51.100.10` (HLS via `hls.example.invalid`, WebRTC via `rtc.example.invalid`)

These are the **same external servers** the user has already wired
into SOTA's `.env` (see Phase 3 below). No new host info is needed.

### Dataset layout
- `data/cam1_merged.mp4` — 2.07 GiB
- `data/cam2_merged.mp4` — 1.81 GiB
- Other folders (`CCTV 29 APR 2026`, `CCTV AI`, `datasets`, etc.) are
  **not** to be copied.

## 2. How Service publishes an annotated stream to MediaMTX

`VideoStreamer` keeps a daemon thread that drains a bounded
`queue.Queue(maxsize=2)` at the configured `fps`. For each frame:
1. Resize to `(width, height)`.
2. Write raw BGR bytes to FFmpeg's stdin.
3. FFmpeg re-encodes with `libx264` + `zerolatency` and pushes the
   RTSP stream to MediaMTX at the publish URL.

The streamer is fed by the engine's `process_video` loop, which draws
the annotated frame via `overlay.draw_detection()` / `draw_hud()` and
calls `streamer.push_frame(annotated)`.

## 3. How Service builds the FFmpeg command

`streamer_command.build_ffmpeg_command()` returns:

```text
ffmpeg -loglevel warning \
  -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i pipe:0 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -b:v {kbps}k -maxrate {kbps}k -bufsize {2*kbps}k \
  -g {fps*2} -sc_threshold 0 \
  -f rtsp -rtsp_transport tcp {output_url}
```

This is the exact `argv` form Service uses; we will copy it.

## 4. MediaMTX URL/path convention

Default `build_publish_url` returns:

```text
rtsp://{host}:{rtsp_port}/{camera_id}/live
```

- `camera_id` is e.g. `cam1` or `cam2`.
- The path part is `{camera_id}/live`. With MediaMTX's default
  `pathDefaults: on`, RTSP path = stream name.
- HLS: `http://{host}:{hls_port}/{camera_id}/live/index.m3u8`.
- WebRTC: `http://{host}:{webrtc_port}/{camera_id}/live`.

The user's task spec asks for a per-camera path that includes a
`MEDIAMTX_STREAM_PREFIX` (e.g. `/sota-paddle-mtmc/CAM_01`). We
**adapt** the URL builder to:

```text
rtsp://{host}:{rtsp_port}/{stream_prefix}/{camera_id}
```

This is a small, well-defined deviation: same MediaMTX semantics, but
the stream name becomes `sota-paddle-mtmc/CAM_01` so multiple apps
can share a MediaMTX instance without colliding.

## 5. How Service connects to MQTT/ThingsBoard

- `mqtt_connection._load_env()` reads `MQTT_USERNAME` / `MQTT_PASSWORD`
  via `os.getenv`. Credentials are never logged.
- `client.username_pw_set(username, password)` is called only when
  both are set; otherwise the broker uses anonymous.
- `tls_set(...)` is called when `tls_enabled=True`.
- The client id is suffixed with epoch + 4-digit random to avoid
  collisions on reconnect.

Service **does not** specifically brand itself as a ThingsBoard
publisher — it uses the generic `{ts, values}` envelope, which is
exactly the ThingsBoard telemetry RPC format. No special TB headers
are required for `v1/devices/me/telemetry`.

## 6. MQTT topic + payload shape

Topic pattern: `ai/yamaha/people-detection/{camera_id}/summary` (or
`/event`, `/status`, `/attributes`).

Payload:

```json
{ "ts": 1710000000000, "values": { ... } }
```

For a zone-aware People Counting integration, the values block carries
per-zone counts (`MainZoneCount`), per-zone dwell averages
(`main_zoneAvgDwell`), and per-camera aggregates (`avgDwellTime`,
`busiestZone`, `uniquePeopleToday`, …).

SOTA already targets `v1/devices/me/telemetry` (ThingsBoard telemetry
RPC) and uses `{ts, values}`. We will **keep** that target and add a
**per-camera topic path** for multi-camera payloads when
`MQTT_TOPIC` is configured.

## 7. How Service uses MinIO

- One bucket per deployment (`yamaha-poc` in Service's `config.yaml`).
- `MinioClient` lazily connects on first use; if credentials are
  missing, the `is_available` flag stays `False` and uploads are
  no-ops.
- `_ensure_bucket()` creates the bucket on first successful connect.
- `MinioUploader.upload_person_crop()` is the per-frame write;
  `HistoryService` enqueues from the engine loop and the publisher
  drains the queue.

Service uses **one bucket**; the user task asks for **three buckets**
(evidence / models / reports). We will adapt the pattern to support
multiple buckets selected by `MINIO_BUCKET_EVIDENCE`,
`MINIO_BUCKET_REPORTS`, `MINIO_BUCKET_MODELS`.

## 8. MinIO object-path convention

Service's pattern:

```text
{object_prefix}/{camera_id}/{slug-zone}/{YYYY-MM-DD}/{unix_ms}_{person_id}.jpg
```

Example: `people-detection/cam1/main-hallway/2026-04-29/1714425600000_5.jpg`

The user task spec requires the more deeply nested:

```text
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/debug_{frame_id}.jpg
visualization/{site_id}/{camera_id}/{yyyy}/{mm}/{dd}/first_3000_frames.mp4
reports/{site_id}/{yyyy}/{mm}/{dd}/benchmark_{timestamp}.json
```

SOTA's `MinioStore.evidence_key()` already produces this exact
structure. We will add `visualization_key()` and `report_key()` static
helpers, and the *bucket* they go to is determined by the *bucket
parameter* of the `MinioStore` instance (caller chooses the bucket).

## 9. What is copied / adapted into SOTA

| Concern | Service pattern | SOTA target |
| --- | --- | --- |
| MediaMTX streamer | `app/io/streamer.py` + `streamer_command.py` | new `app/streaming/mediamtx_streamer.py` + `ffmpeg_writer.py` (port argv builder, do not import from Service) |
| URL builder | `build_publish_url` | re-implemented in `app/streaming/ffmpeg_writer.py`, prefix-aware |
| MinIO client | `MinioClient` + `MinioUploader` | extend `app/storage/minio_store.py` to support multiple buckets, keep deterministic path helpers |
| MinIO evidence path | `{prefix}/{cam}/{zone}/{date}/{ts}_{id}.jpg` | re-use existing `evidence_key()` + add `visualization_key()` / `report_key()` |
| MQTT connection | `MQTTConnection` w/ paho v2 | extend `app/telemetry/mqtt_client.py` with TLS, token-auth, prefix-aware topic; do not import Service |
| MQTT publisher | async queue + retry | add `app/telemetry/mqtt_publisher.py` for the periodic publisher; keep `MqttPublisher` for fire-and-forget |
| MQTT payload | `{ts, values}` | already in `app/telemetry/thingsboard_payload.py`; extend with a zone-event aggregator |
| Engine overlay | `draw_detection` + `draw_hud` | new `app/visualization/overlay.py` + reuse `app/utils/crop.py` for cropping |
| data layout | `data/cam1_merged.mp4`, `data/cam2_merged.mp4` | copy **only** those two files to `SOTA-Paddle-MTMC/data/` |

## 10. What must NOT be copied

- **Secrets**: no `MQTT_PASSWORD`, `MINIO_SECRET_KEY`, or
  `POSTGRES_PASSWORD` from Service's `.env`. SOTA's `.env` already
  has its own credentials.
- **Anything under `app/io/history.py`**'s implementation details of
  Service's internal class names. We only take the `{ts, values}`
  shape and the topic pattern shape.
- **`app/io/minio_service.py`**'s coupling to Service's
  `device_config` / `mqtt_service`. We do not import these.
- **`config.yaml`** (Service's entire config). SOTA uses `configs/app.yaml`
  which has its own keys; we only borrow *names* (`mediamtx.*`, `mqtt.*`,
  `minio.*`) — not the file.
- **`docker-compose.yaml`** Service-level. SOTA's `docker-compose.yaml`
  is intentionally different and we are not changing it.
- **Models**: do NOT copy Service's `models/` directory. SOTA already
  has its own PaddleDetection/TransReID models.
- **Service datasets other than `cam1_merged.mp4` and `cam2_merged.mp4`**.
  In particular, do NOT copy `CCTV 29 APR 2026/`, `CCTV AI/`,
  `datasets/`, `crossing_*.mp4`, etc.

## 11. Summary

Service is a clean, self-contained reference for four pieces of
behaviour we lack in SOTA today:

1. **MediaMTX streaming** (FFmpeg argv + URL builders + daemon
   push loop + reconnect backoff).
2. **MinIO evidence upload** (lazy connect, ensure-bucket, retry
   queue, presigned URLs, deterministic paths).
3. **MQTT telemetry** (`{ts, values}` envelope, topic base + per-camera
   path, token / username-password auth, periodic publisher).
4. **Annotated video overlay** (corner-style box + label with
   `class`, `confidence`, `track_id`, `gid`, plus a HUD with FPS,
   time, MQTT status).

SOTA already has the *plumbing* for the in-process side of these
(pipeline workers, `MinioStore.evidence_key()`, `MqttPublisher`,
`thingsboard_payload`). Phase 4–6 will add the *deployment* layer
(per-camera streamers, bucket-aware MinIO, periodic telemetry, HUD
overlay) without changing SOTA's existing safety gates.

## 12. Phase 1 re-verification (2026-06-13)

The mapping above is current. The SOTA `app/storage/minio_store.py`
now has multi-bucket support and `evidence_key()`,
`visualization_key()`, `report_key()` static helpers. The
`app/telemetry/mqtt_client.py` supports token + username/password
auth. `app/streaming/ffmpeg_writer.py` and
`app/streaming/mediamtx_streamer.py` implement the full FFmpeg +
MediaMTX pipeline (lifecycle, reconnect, isolation per camera). The
`app/visualization/overlay.py` overlay draws all 10 required fields.

The **remaining work for Phases 4-10** is therefore:

1. **Phase 2** — copy `cam1_merged.mp4` + `cam2_merged.mp4`.
2. **Phase 4-6** — already done; verify tests pass.
3. **Phase 7** — verify PP-Human detector + TransReID model availability.
4. **Phase 8** — write `scripts/generate_visual_validation.py` end-to-end
   (the script does not exist; the overlay it will use does).
5. **Phase 9** — run the visualization for both cameras.
6. **Phase 10** — run a production benchmark on the cam_merged dataset
   and write the report.

No new MediaMTX/MQTT/MinIO modules are required: Phase 4-6
implementation already lives in `app/storage/`, `app/telemetry/`,
`app/streaming/`.
