# RTSP Reconnect & Backpressure

> **Operator runbook for PATCH-031/032.**
> How the multi-camera runner degrades gracefully and what
> per-camera metrics to watch.

## TL;DR

The multi-camera runner uses a `ResilientFrameReader` per
camera. Each reader has an independent state machine
(`online → degraded → offline → recovered`) and reconnects
with exponential backoff when the source is a live RTSP /
RTMP / HTTP / TCP / UDP stream. A dead camera does NOT stop
the other cameras.

## State machine

```
            ┌──────────┐ read fail ┌──────────┐ 60s ┌──────────┐
            │  ONLINE  │───────────│ DEGRADED │────│ OFFLINE  │
            │  status=2│           │  status=1│    │  status=0│
            └──────────┘           └──────────┘    └──────────┘
                  ▲                                         │
                  │                                         │ reconnect OK
                  └─────────────────────────────────────────┘
```

`degraded_after_seconds=10` and `offline_after_seconds=60`
are the defaults. Override via `configs/benchmark.yaml`:

```yaml
streams:
  reconnect:
    enabled: true
    initial_backoff_seconds: 1
    max_backoff_seconds: 30
    degraded_after_seconds: 10
    offline_after_seconds: 60
```

(These are read by `app/utils/resilient_reader.py`; wire them
into the runner at startup in a follow-up PR.)

## Backpressure policy

The runner's per-camera frame queue has a configurable
`maxsize` and a `drop_policy`:

```yaml
queues:
  frame_queue_maxsize: 64
  drop_policy: drop_oldest    # drop_oldest | drop_newest | block_with_timeout
```

* `drop_oldest` (default): when the queue is full, the new
  frame is silently dropped. `camera_drops_total` is
  incremented. The consumer drains at its own pace; the
  freshest data is preferred.
* `drop_newest`: when the queue is full, the OLDEST item is
  evicted and the new item is added. Same metric. The
  consumer is forced to see the freshest data.
* `block_with_timeout`: the producer's `q.put` blocks for
  up to 0.5 s. On timeout, the frame is dropped and the
  metric is incremented.

## Per-camera metrics

```text
camera_fps{camera_id="CAM_01"}                4.5
camera_frame_latency_ms{camera_id="CAM_01"} 35.0
camera_queue_depth{camera_id="CAM_01"}      2
camera_status{camera_id="CAM_01"}            2
camera_decode_errors_total{camera_id="CAM_01"} 0
camera_reconnects_total{camera_id="CAM_01"}  1
camera_drops_total{camera_id="CAM_01"}       12
camera_last_frame_timestamp{camera_id="CAM_01"} 1718210000.0
total_analytics_fps                           8.2
```

`status` is 0=offline, 1=degraded, 2=online.

## Common failure modes

* **Camera unreachable from boot (RTSP URL wrong):** the
  reader transitions to OFFLINE on the first read failure.
  The consumer's `FrameResult(frame=None, tracks=[])` is
  emitted every 1 s so the operator sees the camera is alive
  in the runner but is offline.

* **Camera goes offline mid-run (network blip):** the
  reader attempts to reconnect with exponential backoff
  (1 s → 2 s → 4 s → ... → 30 s). The metric
  `camera_reconnects_total` increments on each attempt.

* **Consumer is slow (collector is CPU-bound):** the queue
  fills. With `drop_oldest` (default), new frames are
  silently dropped and `camera_drops_total` increments.
  The operator should watch this metric and either scale
  the consumer (more CPU) or lower the camera's `fps_target`.

* **A wild camera floods the queue (high FPS, small
  bbox):** same as above; `drop_oldest` is the right policy.

## Tests

* `tests/test_backpressure_and_reconnect.py` — covers all
  three drop policies, the state transitions, the
  reconnect metric, and the "dead camera doesn't kill
  others" property.

## Tuning

The `frame_queue_maxsize` should be sized so the queue can
absorb a 1-2 s consumer stall. At 5 FPS analytics with
250 ms processing per frame, `maxsize=8` is enough. At 25
FPS with 100 ms processing, `maxsize=16` is safer. The
default of 64 is conservative for the Yamaha showroom
deployment.
