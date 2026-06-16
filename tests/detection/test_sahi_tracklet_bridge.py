"""Tests for SAHITrackletBridge.

Uses a fake Redis and a fake TrackletCollector. The bridge is
exercised directly via ``_handle_sahi_event`` and ``_handle_sahi_bbox``;
the background thread is exercised only by the start/stop test.
"""

from __future__ import annotations

import time
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from app.detection.sahi_tracklet_bridge import (
    SAHIBridgeConfig,
    SAHITrackletBridge,
)


class FakeRedis:
    """Minimal fake mirroring the subset of RedisState the bridge uses."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        # stream_name -> list of (msg_id, fields_dict)
        self.streams: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    def ensure_group(self, name: str, group: str) -> None:
        pass

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

    # Raw redis-py client interface used for PP-Human dedup.
    def xrevrange(
        self,
        name: str,
        max: str = "+",
        min: str = "-",
        count: Optional[int] = None,
    ) -> list:
        items = list(reversed(self.streams.get(name, [])))
        if count is not None:
            items = items[:count]
        return items


@pytest.fixture
def cfg() -> SAHIBridgeConfig:
    return SAHIBridgeConfig()


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


def test_bridge_dedup_against_pphuman(
    fake_redis: FakeRedis, cfg: SAHIBridgeConfig
):
    """SAHI bbox overlapping a PP-Human bbox is dropped (dedup)."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    # Seed an active PP-Human track.
    fake_redis.streams["stream:detections"] = [
        (
            "1-0",
            {
                "camera_id": "CAM_01",
                "timestamp_ms": str(int(time.time() * 1000)),
                "bbox": [10.0, 10.0, 100.0, 100.0],
            },
        ),
    ]
    # SAHI bbox overlaps the PP-Human bbox (IoU ~ 0.85).
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=1,
        ts_ms=int(time.time() * 1000),
        bbox=(15.0, 15.0, 95.0, 95.0),
        score=0.7,
        idx=0,
    )
    # Should NOT have been emitted (matched an active PP-Human track).
    assert collector.on_sahi_detection.call_count == 0


def test_bridge_emits_new_person(
    fake_redis: FakeRedis, cfg: SAHIBridgeConfig
):
    """SAHI bbox that does NOT match any PP-Human track is emitted."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    # No active PP-Human tracks.
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=1,
        ts_ms=int(time.time() * 1000),
        bbox=(500.0, 500.0, 600.0, 600.0),
        score=0.7,
        idx=0,
    )
    assert collector.on_sahi_detection.call_count == 1
    call_kwargs = collector.on_sahi_detection.call_args.kwargs
    assert call_kwargs["camera_id"] == "CAM_01"
    assert call_kwargs["bbox"] == (500.0, 500.0, 600.0, 600.0)
    assert call_kwargs["frame_id"] == 1
    assert call_kwargs["score"] == 0.7


def test_bridge_handles_redis_error(
    fake_redis: FakeRedis, cfg: SAHIBridgeConfig
):
    """If the PP-Human Redis read fails, the bridge treats as no match."""

    def boom(*_args: Any, **_kwargs: Any) -> list:
        raise ConnectionError("redis down")

    fake_redis.xrevrange = boom
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    # Should not raise; should treat as no match.
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=1,
        ts_ms=int(time.time() * 1000),
        bbox=(10.0, 10.0, 50.0, 50.0),
        score=0.5,
        idx=0,
    )
    assert collector.on_sahi_detection.call_count == 1


def test_bridge_stop_joins_within_2s(
    fake_redis: FakeRedis, cfg: SAHIBridgeConfig
):
    """stop() must join the thread within 2 seconds."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    bridge.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    bridge.stop()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0


def test_bridge_handle_sahi_event_decodes(
    fake_redis: FakeRedis, cfg: SAHIBridgeConfig
):
    """_handle_sahi_event decodes the JSON-decoded fields (consume()
    already decodes) and emits one tracklet per detection."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    # consume() in the real RedisState decodes JSON values, so the
    # ``detections`` field here is a list, not a string.
    bridge._handle_sahi_event(
        {
            "camera_id": "CAM_01",
            "frame_id": 42,
            "timestamp_ms": str(int(time.time() * 1000)),
            "detections": [
                [10.0, 10.0, 50.0, 50.0, 0.9],
                [200.0, 200.0, 240.0, 240.0, 0.6],
            ],
        }
    )
    assert collector.on_sahi_detection.call_count == 2
    cameras = {
        c.kwargs["camera_id"] for c in collector.on_sahi_detection.call_args_list
    }
    assert cameras == {"CAM_01"}


def test_bridge_does_not_emit_for_other_camera(
    fake_redis: FakeRedis, cfg: SAHIBridgeConfig
):
    """PP-Human bboxes for OTHER cameras must not dedup a SAHI bbox."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    fake_redis.streams["stream:detections"] = [
        (
            "1-0",
            {
                "camera_id": "CAM_OTHER",
                "timestamp_ms": str(int(time.time() * 1000)),
                "bbox": [10.0, 10.0, 100.0, 100.0],
            },
        ),
    ]
    # SAHI bbox in CAM_01 should NOT match CAM_OTHER's PP-Human bbox.
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=1,
        ts_ms=int(time.time() * 1000),
        bbox=(10.0, 10.0, 100.0, 100.0),
        score=0.7,
        idx=0,
    )
    assert collector.on_sahi_detection.call_count == 1


def test_bridge_dedup_drops_stale_pphuman(
    fake_redis: FakeRedis, cfg: SAHIBridgeConfig
):
    """PP-Human bboxes older than the window are ignored for dedup."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    # 5 seconds old — outside the 1-second pphuman_window_ms window.
    stale_ts = int(time.time() * 1000) - 5000
    fake_redis.streams["stream:detections"] = [
        (
            "1-0",
            {
                "camera_id": "CAM_01",
                "timestamp_ms": str(stale_ts),
                "bbox": [10.0, 10.0, 100.0, 100.0],
            },
        ),
    ]
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=1,
        ts_ms=int(time.time() * 1000),
        bbox=(10.0, 10.0, 100.0, 100.0),
        score=0.7,
        idx=0,
    )
    # Stale PP-Human bbox must not dedup the SAHI bbox.
    assert collector.on_sahi_detection.call_count == 1
