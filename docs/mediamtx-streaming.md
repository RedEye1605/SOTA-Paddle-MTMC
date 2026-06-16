# MediaMTX streaming setup

SOTA-Paddle-MTMC publishes annotated video frames to the operator's
external MediaMTX instance, one RTSP stream per camera.

## 1. Activation

Streaming is **opt-in**. Enable it in `.env`:

```text
MEDIAMTX_ENABLED=true
MEDIAMTX_HOST=198.51.100.10          # operator-managed MediaMTX
MEDIAMTX_RTSP_PORT=8554
MEDIAMTX_RTMP_PORT=1935
MEDIAMTX_HLS_PORT=8888
MEDIAMTX_WEBRTC_PORT=8889
MEDIAMTX_STREAM_PREFIX=sota-paddle-mtmc
MEDIAMTX_FPS=10
MEDIAMTX_BITRATE_KBPS=1800
MEDIAMTX_WIDTH=960
MEDIAMTX_HEIGHT=540
```

When `MEDIAMTX_ENABLED=false` the streamer is a no-op and the
visual-validation script writes only to local MP4 files (no ffmpeg
subprocess is spawned). This is the **default** to avoid hammering
the host when no MediaMTX is reachable.

## 2. Stream URL convention

For each camera the streamer builds:

```text
rtsp://{MEDIAMTX_HOST}:{MEDIAMTX_RTSP_PORT}/{MEDIAMTX_STREAM_PREFIX}/{camera_id}
```

Example: with `MEDIAMTX_HOST=198.51.100.10` and
`MEDIAMTX_STREAM_PREFIX=sota-paddle-mtmc`:

| Camera | Publish URL |
| --- | --- |
| `CAM_01` | `rtsp://198.51.100.10:8554/sota-paddle-mtmc/CAM_01` |
| `CAM_02` | `rtsp://198.51.100.10:8554/sota-paddle-mtmc/CAM_02` |

The default MediaMTX behaviour auto-creates a stream entry the
first time someone `PUBLISH`es to it; no `mediamtx.yml` change is
required for basic operation.

## 3. FFmpeg command

The streamer pipes BGR frames to ffmpeg using the upstream Service
argv:

```text
ffmpeg -loglevel warning \
  -f rawvideo -pix_fmt bgr24 -s 960x540 -r 10 -i pipe:0 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -b:v 1800k -maxrate 1800k -bufsize 3600k \
  -g 20 -sc_threshold 0 \
  -f rtsp -rtsp_transport tcp <publish_url>
```

The frame rate, resolution, and bitrate are configured via the
`MEDIAMTX_FPS` / `MEDIAMTX_WIDTH` / `MEDIAMTX_HEIGHT` /
`MEDIAMTX_BITRATE_KBPS` env vars. The `pipe:0` input is filled from
the producer thread; the producer drops frames if ffmpeg cannot
keep up (the streamer uses a 2-frame bounded queue).

## 4. Subscribe URLs

MediaMTX automatically exposes HLS and WebRTC for every published
stream. With the defaults above:

| Camera | HLS | WebRTC |
| --- | --- | --- |
| `CAM_01` | `http://198.51.100.10:8888/sota-paddle-mtmc/CAM_01/index.m3u8` | `http://198.51.100.10:8889/sota-paddle-mtmc/CAM_01` |
| `CAM_02` | `http://198.51.100.10:8888/sota-paddle-mtmc/CAM_02/index.m3u8` | `http://198.51.100.10:8889/sota-paddle-mtmc/CAM_02` |

(The exact host:port pair depends on the operator's MediaMTX
config — adjust to your deployment.)

## 5. Failure handling

- If `ffmpeg` is not on `$PATH`, the streamer logs
  `ffmpeg not found; install ffmpeg before enabling streaming` and
  remains a no-op.
- If the ffmpeg process dies (network blip, MediaMTX restart), the
  streamer reconnects with exponential backoff
  (`min=2 s`, `max=60 s`, `max_attempts=5`).
- If `MEDIAMTX_HOST` is empty, the streamer refuses to start
  (`MEDIAMTX_HOST must be set when MEDIAMTX_ENABLED=true`).
- A failed stream for `CAM_01` does **not** kill the visualization
  script. The script logs a warning, marks the camera as
  `stream=disabled`, and continues with the next camera.
- The `strict` mode (`MEDIAMTX_STRICT=true`) makes the script exit
  non-zero if any camera stream fails. Off by default; turn it on
  for production CI.

## 6. Quick start

```bash
# 1. Verify ffmpeg is installed
ffmpeg -version | head -n 1

# 2. Enable streaming
sed -i 's/^MEDIAMTX_ENABLED=false/MEDIAMTX_ENABLED=true/' .env

# 3. Run the visualization; ffmpeg will publish annotated frames
uv run python scripts/generate_visual_validation.py \
  --cam CAM_01 --input data/cam1_merged.mp4 \
  --max-frames 3000 \
  --output reports/visualization/CAM_01_first_3000_frames.mp4

# 4. Subscribe with VLC or a browser (HLS)
vlc http://198.51.100.10:8888/sota-paddle-mtmc/CAM_01/index.m3u8
```

## 7. References

- `app/streaming/mediamtx_streamer.py` — per-camera streamer
  (daemon thread, bounded queue, reconnect backoff).
- `app/streaming/ffmpeg_writer.py` — argv and URL builders.
- `app/visualization/overlay.py` — annotated-frame composer.
- `Docs/external_services_setup.md` — cross-service overview.
