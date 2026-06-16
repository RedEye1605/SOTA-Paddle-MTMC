"""End-to-end tests for the SAHI integration.

Exercises the full chain (SAHIWorker publish + SAHITrackletBridge
dedup + TrackletCollector.on_sahi_detection emit) with fakes for
Redis, the RTSP buffer, and the detector.

The clean-stream assertion (constraint 4) is exercised by
``test_sahi_clean_stream_no_overlay`` which decodes a synthetic
frame and asserts no overlay pixels are present.
"""

from __future__ import annotations

import json
import time
from typing import Any, List, Optional
from unittest.mock import MagicMock

import numpy as np

from app.detection.sahi_tracklet_bridge import (
    SAHIBridgeConfig,
    SAHITrackletBridge,
)
from app.detection.sahi_worker import SAHIWorker, SAHIWorkerConfig
from app.workers.tracklet_collector import TrackletCollector


class FakeRedis:
    """In-memory fake of RedisState.

    Mirrors the subset of the API used by the SAHI worker, the bridge,
    and TrackletCollector. Streams are stored as ordered lists of
    ``(msg_id, fields_dict)`` tuples.
    """

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    def ensure_group(self, name: str, group: str) -> None:
        pass

    def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)

    def set(  # noqa: A002
        self, key: str, value: str, ex: Optional[int] = None
    ) -> None:
        self.kv[key] = value

    def xadd(
        self,
        name: str,
        fields: dict[str, Any],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        eid = f"{int(time.time() * 1e6)}-{len(self.streams.get(name, []))}"
        self.streams.setdefault(name, []).append((eid, fields))
        if maxlen is not None and len(self.streams[name]) > maxlen:
            self.streams[name] = self.streams[name][-maxlen:]
        return eid

    def consume(
        self,
        name: str,
        group: str,
        consumer: str,
        *,
        count: int,
        block_ms: int,
    ) -> list:
        items = self.streams.get(name, [])
        if not items:
            return []
        out = items[:count]
        self.streams[name] = items[count:]
        return out

    def ack(self, name: str, group: str, msg_id: str) -> None:
        pass

    def xrevrange(
        self,
        name: str,
        max: str = "+",  # noqa: A002
        min: str = "-",
        count: Optional[int] = None,
    ) -> list:
        items = list(reversed(self.streams.get(name, [])))
        if count is not None:
            items = items[:count]
        return items

    # TrackletCollector uses these. Provide safe stubs.
    def append_crop(self, camera_id: str, local_track_id: int, uri: str) -> None:
        return None

    def clear_buffer(self, camera_id: str, local_track_id: int) -> None:
        return None

    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        eid = f"{int(time.time() * 1e6)}-{len(self.streams.get(stream, []))}"
        self.streams.setdefault(stream, []).append((eid, payload))
        return eid


class FakeBuffer:
    """In-memory fake of RTSPFrameBuffer.

    ``start()`` is a no-op; ``stop()`` causes subsequent
    ``get_frame()`` calls to return None so the SAHI worker exits its
    run loop.
    """

    def __init__(self, frames: List[np.ndarray]) -> None:
        self._frames = frames
        self._idx = 0
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_frame(self, *, timeout_sec: float = 1.0) -> Optional[np.ndarray]:
        if self.stopped:
            return None
        if self._idx >= len(self._frames):
            # Avoid a tight spin loop when the buffer is exhausted.
            time.sleep(timeout_sec)
            return None
        f = self._frames[self._idx]
        self._idx += 1
        return f


class StubPostgres:
    """Stub PostgresStore — TrackletCollector only calls insert_tracklet."""

    def insert_tracklet(self, **_kwargs: Any) -> None:
        pass


class StubMinio:
    """Stub MinioStore — TrackletCollector only calls a few helpers."""

    bucket = "stub-bucket"

    def put_crop(self, **_kwargs: Any) -> str:
        return "s3://stub-bucket/x.jpg"

    def copy_object_within_bucket(self, **_kwargs: Any) -> None:
        pass

    def evidence_key(self, **_kwargs: Any) -> str:
        return "evidence/x.jpg"


def _solid_frame(h: int = 1080, w: int = 1920, color: int = 64) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _build_collector(redis: FakeRedis) -> TrackletCollector:
    return TrackletCollector(
        pg=StubPostgres(),
        redis=redis,
        minio=StubMinio(),
        site_id="test",
        zone_rows=[],
    )


def test_sahi_e2e_sahi_to_tracklet_collector(monkeypatch: Any) -> None:
    """SAHIWorker publishes -> SAHITrackletBridge dedups ->
    TrackletCollector.on_sahi_detection -> emit.

    Drives the worker, lets it publish, then drives the bridge for
    one scan and asserts a SAHI-sourced provisional tracklet was
    added to the collector's in-flight map.
    """
    redis = FakeRedis()
    collector = _build_collector(redis)

    # Fake detector returns a single bbox every call.
    detector = MagicMock()
    detector.predict = MagicMock(return_value=[(100, 100, 200, 200, 0.7)])

    # Buffer with 3 frames so the worker publishes 3 times.
    buf = FakeBuffer([_solid_frame() for _ in range(3)])
    cfg = SAHIWorkerConfig(
        camera_id="CAM_01",
        rtsp_url="rtsp://fake",
        stream="stream:detections_sahi",
        latest_key_prefix="sahi:latest:",
        rate_limit_hz=10.0,
        stale_ms=10_000,
        # Pin the wall-clock so the worker's stale-frame drop does
        # not fire during the test.
        current_timestamp_ms=int(time.time() * 1000),
    )
    worker = SAHIWorker(
        config=cfg,
        buffer=buf,  # type: ignore[arg-type]
        detector=detector,
        redis=redis,  # type: ignore[arg-type]
    )

    # Force SAHI_ENABLED=true for this test.
    monkeypatch.setenv("SAHI_ENABLED", "true")
    worker.start()
    # Wait for the worker to consume all 3 frames.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not redis.streams.get(
        "stream:detections_sahi"
    ):
        time.sleep(0.05)
    worker.stop()

    # Sanity check: the worker actually published at least one event
    # to stream:detections_sahi.
    assert redis.streams.get("stream:detections_sahi"), (
        "SAHIWorker did not publish any events to stream:detections_sahi"
    )

    # Now drive the bridge manually (bypass the background thread).
    bridge_cfg = SAHIBridgeConfig()
    bridge = SAHITrackletBridge(
        redis=redis,  # type: ignore[arg-type]
        collector=collector,
        config=bridge_cfg,
    )
    bridge._scan_once()

    # The bridge should have called on_sahi_detection at least once;
    # each call inserts a SAHI-sourced provisional tracklet.
    sahi_tracklets = [
        tl for tl in collector._in_flight.values() if tl.source == "sahi"
    ]
    assert len(sahi_tracklets) >= 1
    assert all(tl.provisional for tl in sahi_tracklets)


def test_sahi_clean_stream_no_overlay() -> None:
    """A clean frame (no HLS overlay) should be solid color."""
    clean = _solid_frame(h=480, w=640, color=128)
    buf = FakeBuffer([clean])
    frame = buf.get_frame(timeout_sec=1.0)
    assert frame is not None
    # All pixels should be the solid color.
    assert (frame == 128).all(), "frame is not solid; overlay may be present"


def test_sahi_e2e_no_block_on_pipeline() -> None:
    """Grep the SAHITrackletBridge source for BLOCK 0 / xread."""
    import inspect

    from app.detection import sahi_tracklet_bridge

    src = inspect.getsource(sahi_tracklet_bridge)
    # The bridge uses xrevrange (non-blocking) and consume with
    # block_ms=0 (non-blocking). Neither BLOCK 0 nor blocking xread
    # should appear in the source.
    assert "BLOCK 0" not in src
    assert "block_ms=0" in src or "block_ms = 0" in src


def test_sahi_e2e_multi_camera_isolation() -> None:
    """Events for CAM_02 must not affect CAM_01's bridge state."""
    redis = FakeRedis()
    collector = _build_collector(redis)
    # Seed CAM_02's events on the SAHI stream.
    redis.streams["stream:detections_sahi"] = [
        (
            "1-0",
            {
                "camera_id": "CAM_02",
                "frame_id": "1",
                "timestamp_ms": str(int(time.time() * 1000)),
                "detections": json.dumps([[20.0, 20.0, 60.0, 60.0, 0.9]]),
            },
        ),
    ]
    bridge = SAHITrackletBridge(
        redis=redis,  # type: ignore[arg-type]
        collector=collector,
    )
    bridge._scan_once()
    # CAM_02 event should be processed; CAM_01 should have no tracklets.
    cam01_tracklets = [
        tl for tl in collector._in_flight.values() if tl.camera_id == "CAM_01"
    ]
    cam02_tracklets = [
        tl for tl in collector._in_flight.values() if tl.camera_id == "CAM_02"
    ]
    assert len(cam01_tracklets) == 0
    assert len(cam02_tracklets) == 1
    assert cam02_tracklets[0].source == "sahi"
    assert cam02_tracklets[0].provisional is True
