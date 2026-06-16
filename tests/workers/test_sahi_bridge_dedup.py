"""Regression tests for SAHITrackletBridge PP-Human dedup.

Verifies the bridge reads the correct field name (``bbox``, singular)
from ``stream:detections`` — matching the vendor
``RedisSideChannel.emit_detection`` payload schema in
``app/detection/_vendor/paddledetection_pipeline.py:156`` — and not the
older ``bboxes`` (plural) key the bridge used to read by mistake.
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


class _FakeRedis:
    """Minimal fake mirroring the subset of RedisState the bridge uses."""

    def __init__(self) -> None:
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
def fake_redis() -> _FakeRedis:
    return _FakeRedis()


def test_dedup_reads_correct_field_name(
    fake_redis: _FakeRedis, cfg: SAHIBridgeConfig
) -> None:
    """A SAHI bbox overlapping a PP-Human ``bbox`` (singular) is dropped.

    This is the direct regression: the bridge previously read
    ``bboxes`` (plural) which never matched the vendor payload, so
    the dedup was always a no-op.
    """
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
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
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=1,
        ts_ms=int(time.time() * 1000),
        bbox=(15.0, 15.0, 95.0, 95.0),
        score=0.7,
        idx=0,
    )
    assert collector.on_sahi_detection.call_count == 0


def test_dedup_against_real_pp_human_payload(
    fake_redis: _FakeRedis, cfg: SAHIBridgeConfig
) -> None:
    """Vendor-shaped payload: schema_version, source, singular bbox."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    # Exact field set produced by
    # RedisSideChannel.emit_detection in
    # app/detection/_vendor/paddledetection_pipeline.py:144-160.
    ts_ms = int(time.time() * 1000)
    fake_redis.streams["stream:detections"] = [
        (
            "1-0",
            {
                "schema_version": "1.0",
                "event_id": "det_CAM_01_42_7",
                "source": "pphuman",
                "run_id": "run-xyz",
                "camera_id": "CAM_01",
                "frame_id": "42",
                "timestamp_ms": str(ts_ms),
                "received_at_ms": str(ts_ms),
                "local_track_id": "7",
                "bbox": [100.0, 200.0, 300.0, 400.0],
                "score": "0.91",
                "crop_path": "",
                "frame_uri": "",
                "embedding": "",
            },
        ),
    ]
    # Overlaps the vendor bbox.
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=42,
        ts_ms=ts_ms,
        bbox=(110.0, 210.0, 290.0, 390.0),
        score=0.8,
        idx=0,
    )
    assert collector.on_sahi_detection.call_count == 0


def test_no_dedup_when_field_missing(
    fake_redis: _FakeRedis, cfg: SAHIBridgeConfig
) -> None:
    """A payload without ``bbox`` must not crash; SAHI is emitted."""
    collector = MagicMock()
    bridge = SAHITrackletBridge(
        redis=fake_redis, collector=collector, config=cfg
    )
    fake_redis.streams["stream:detections"] = [
        (
            "1-0",
            {
                "camera_id": "CAM_01",
                "timestamp_ms": str(int(time.time() * 1000)),
                # No ``bbox`` key at all (e.g. malformed event).
            },
        ),
    ]
    bridge._handle_sahi_bbox(
        camera_id="CAM_01",
        frame_id=1,
        ts_ms=int(time.time() * 1000),
        bbox=(10.0, 10.0, 50.0, 50.0),
        score=0.5,
        idx=0,
    )
    # No crash; no dedup; SAHI goes through.
    assert collector.on_sahi_detection.call_count == 1
