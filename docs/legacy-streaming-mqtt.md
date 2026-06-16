# Service reference — streaming / MinIO / MQTT / overlay

This document records the **patterns copied** from
`Service/offline-people-counting` into `SOTA-Paddle-MTMC` during
the Phases 1-6 integration. It is the operator-facing map of
"where did the idea come from, where does it live now".

The full inspection notes are in
`FixReports/44_service_reference_mapping.md`; this is the
operational summary.

## MediaMTX streaming

| Concern | Service file | SOTA target |
| --- | --- | --- |
| URL builder | `app/io/streamer_command.py::build_publish_url` | `app/streaming/ffmpeg_writer.py::build_publish_url` |
| FFmpeg argv | `app/io/streamer_command.py::build_ffmpeg_command` | `app/streaming/ffmpeg_writer.py::build_ffmpeg_command` |
| Daemon push loop | `app/io/streamer.py::VideoStreamer` | `app/streaming/mediamtx_streamer.py::MediaMTXStreamer` |
| HLS / WebRTC consumer URLs | `app/io/streamer_command.py::build_hls_url` / `build_webrtc_url` | `app/streaming/ffmpeg_writer.py::build_hls_url` / `build_webrtc_url` |
| Per-camera env loader | (Service uses runtime config) | `app/streaming/mediamtx_streamer.py::make_from_env` |

The SOTA URL builder uses an explicit `MEDIAMTX_STREAM_PREFIX`
(default `sota-paddle-mtmc`) so the streams do not collide with
Service's `/{camera_id}/live` convention when both share a
MediaMTX instance:

```text
rtsp://{MEDIAMTX_HOST}:{MEDIAMTX_RTSP_PORT}/{MEDIAMTX_STREAM_PREFIX}/{camera_id}
```

## MinIO evidence storage

| Concern | Service file | SOTA target |
| --- | --- | --- |
| Client lifecycle | `app/io/minio_client.py` | `app/storage/minio_store.py::MinioStore` |
| Per-crop upload | `app/io/minio_uploader.py::MinioUploader` | `app/storage/minio_store.py::MinioStore.put_crop` |
| Path scheme | `{prefix}/{cam}/{zone}/{date}/{ts}_{id}.jpg` | `evidence/{site}/{cam}/{zone}/{yyyy}/{mm}/{dd}/{gid}/{tracklet_id}/{kind}.jpg` |
| Pending bucket | (Service does not split pending/final) | `evidence/pending/{site}/{cam}/{tracklet}/{kind}.jpg` (PATCH-029) |
| Reports bucket | (Service uses one bucket) | `reports/{site}/{yyyy}/{mm}/{dd}/benchmark_{ts}.json` |
| Visualization bucket | (Service uses one bucket) | `visualization/{site}/{cam}/{yyyy}/{mm}/{dd}/first_3000_frames.mp4` |

Service uses one bucket; SOTA splits evidence / reports / models
per the user task spec. The 3-bucket split is controlled by
`MINIO_BUCKET_EVIDENCE`, `MINIO_BUCKET_REPORTS`, and
`MINIO_BUCKET_MODELS`.

## MQTT / ThingsBoard

| Concern | Service file | SOTA target |
| --- | --- | --- |
| Connection | `app/io/mqtt_connection.py` | `app/telemetry/mqtt_client.py::MqttPublisher` |
| Async publisher | `app/io/mqtt_publisher.py::MQTTPublisher` | `app/telemetry/mqtt_publisher.py::MQTTPublisher` |
| Topic base | `app/io/mqtt_topics.py` | `app/telemetry/mqtt_client.py::MqttPublisher` (configurable) |
| Payload shape | `app/io/history.py::to_unix_ms` / `build_person_payload` | `app/telemetry/thingsboard_payload.py` |
| ThingsBoard format | `app/counting/payload.py::PayloadBuilder` | `app/telemetry/thingsboard_payload.py` |

Service uses `{ts, values}` for telemetry; SOTA targets the same
shape for ThingsBoard compatibility. Token-based auth (per-device
token) is supported alongside username/password.

## Engine overlay

| Concern | Service file | SOTA target |
| --- | --- | --- |
| Detection box | `app/engine/overlay.py::draw_detection_box` | `app/visualization/overlay.py::draw_detections` |
| Label builder | `app/engine/overlay.py::_build_label` | `app/visualization/overlay.py::_build_detection_label` |
| HUD (status panel) | `app/engine/overlay.py::draw_hud` | `app/visualization/overlay.py::draw_hud` |
| Annotation entry point | (Service calls `draw_detection_box` per detection) | `app/visualization/overlay.py::annotate_frame` |

SOTA's overlay includes a SMOKE warning line when the pipeline
runs in smoke mode. The labels include: class, confidence, local
track id (`L#`), global id (`G:#`), ReID similarity (`R#`),
and zone id (`Z:#`).

## What is NOT copied

* **No secrets** from Service's `.env` (the operator's
  credentials in SOTA's `.env` are kept separate).
* **No Service-side class names** (e.g. `device_config`,
  `mqtt_service`).
* **No Service-side `config.yaml`**, only the key *names*
  (`mediamtx.*`, `mqtt.*`, `minio.*`).
* **No models** from Service's `models/` directory.
* **No Service datasets** other than `cam1_merged.mp4` and
  `cam2_merged.mp4`.
