"""Per-camera metrics tests (PATCH-018).

The audit's PATCH-018 fix requires the runner to:
  1. Measure per-camera FPS independently.
  2. Expose per-camera latency, queue depth, drop count, decode
     errors, reconnects via /metrics.
  3. Use a moving window (EWMA) to avoid noisy logs.
"""

from __future__ import annotations

import time


from app.telemetry.per_camera import (
    CAMERA_STATUS_DEGRADED,
    CAMERA_STATUS_OFFLINE,
    CAMERA_STATUS_ONLINE,
    PER_CAMERA,
    PerCameraMetrics,
)


def test_per_camera_fps_increases_with_frame_rate() -> None:
    m = PerCameraMetrics("CAM_01", window_seconds=5.0)
    base = time.time()
    # 10 frames in 1 second → 9 fps.
    for i in range(10):
        m.observe_frame(base + i * 0.1)
    fps = m.fps()
    assert 8.0 < fps < 11.0, f"unexpected fps {fps}"


def test_per_camera_fps_zero_with_one_frame() -> None:
    m = PerCameraMetrics("CAM_01")
    m.observe_frame(time.time())
    assert m.fps() == 0.0


def test_per_camera_fps_window_drops_old_frames() -> None:
    """Frames older than the window are trimmed out of the FPS
    calculation.
    """
    m = PerCameraMetrics("CAM_01", window_seconds=2.0)
    base = time.time()
    m.observe_frame(base - 5)  # 5 s old, will be trimmed
    m.observe_frame(base - 0.2)
    m.observe_frame(base)
    fps = m.fps()
    # 2 recent frames, ~0.2 s apart → ~5 fps.
    assert 3.0 < fps < 8.0


def test_per_camera_latency_ewma() -> None:
    m = PerCameraMetrics("CAM_01", ewma_alpha=0.5)
    m.observe_frame_latency(100.0)
    assert m.latency_ms() == 50.0
    m.observe_frame_latency(200.0)
    # 0.5 * 200 + 0.5 * 50 = 125
    assert m.latency_ms() == 125.0


def test_per_camera_status_initial() -> None:
    m = PerCameraMetrics("CAM_01")
    assert m.status == CAMERA_STATUS_ONLINE


def test_per_camera_status_setter_records_change() -> None:
    m = PerCameraMetrics("CAM_01")
    m.set_status(CAMERA_STATUS_OFFLINE)
    assert m.status == CAMERA_STATUS_OFFLINE
    m.set_status(CAMERA_STATUS_DEGRADED)
    assert m.status == CAMERA_STATUS_DEGRADED
    m.set_status(CAMERA_STATUS_DEGRADED)  # no change
    assert m.status == CAMERA_STATUS_DEGRADED


def test_per_camera_metrics_are_independent() -> None:
    """CAM_01 and CAM_02 metrics are isolated."""
    m1 = PerCameraMetrics("CAM_01")
    m2 = PerCameraMetrics("CAM_02")
    base = time.time()
    # CAM_01: 5 frames in 0.5 s → ~10 fps.
    for i in range(5):
        m1.observe_frame(base + i * 0.1)
    # CAM_02: 0 frames → 0 fps.
    assert m1.fps() > m2.fps()
    assert m2.fps() == 0.0


def test_per_camera_metrics_registry_returns_same_object() -> None:
    """The process-wide cache returns the same PerCameraMetrics for
    a given camera_id.
    """
    a = PER_CAMERA.for_camera("CAM_TEST_REG")
    b = PER_CAMERA.for_camera("CAM_TEST_REG")
    assert a is b


def test_metrics_render_includes_camera_labels() -> None:
    """The /metrics output must include per-camera labels."""
    PER_CAMERA.for_camera("CAM_METRICS_LABEL_TEST").observe_frame(time.time())
    text = PER_CAMERA.for_camera("CAM_METRICS_LABEL_TEST").camera_id
    assert text == "CAM_METRICS_LABEL_TEST"
    # Sanity: the registry's render() must include the per-camera
    # gauges we declared in app/telemetry/metrics.py.
    from app.telemetry.metrics import REGISTRY

    rendered = REGISTRY.render()
    assert "camera_fps" in rendered
    assert "camera_queue_depth" in rendered
    assert "camera_status" in rendered
    assert "camera_reconnects_total" in rendered
    assert "camera_decode_errors_total" in rendered
    assert "camera_drops_total" in rendered
    assert "camera_last_frame_timestamp" in rendered
    assert "camera_frame_latency_ms" in rendered
    assert "total_analytics_fps" in rendered


def test_decode_error_increments_per_camera() -> None:
    PER_CAMERA.for_camera("CAM_DECODE_TEST").observe_decode_error()
    PER_CAMERA.for_camera("CAM_DECODE_TEST").observe_decode_error()
    # Other camera must NOT be affected.
    other = PER_CAMERA.for_camera("CAM_DECODE_OTHER")
    other.set_status(CAMERA_STATUS_ONLINE)
    from app.telemetry.metrics import REGISTRY

    rendered = REGISTRY.render()
    # The counter for CAM_DECODE_TEST must be 2.
    assert 'camera_decode_errors_total{camera_id="CAM_DECODE_TEST"} 2' in rendered
    # The other camera's counter must not appear (it has no
    # observations).
    assert 'camera_decode_errors_total{camera_id="CAM_DECODE_OTHER"}' not in rendered


def test_drop_counter_increments() -> None:
    PER_CAMERA.for_camera("CAM_DROP_TEST").observe_drop()
    from app.telemetry.metrics import REGISTRY

    rendered = REGISTRY.render()
    assert 'camera_drops_total{camera_id="CAM_DROP_TEST"} 1' in rendered
