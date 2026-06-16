"""TDD tests for TrackletCollector.on_sahi_detection.

Verifies the dataclass additions (source, provisional) and the
new on_sahi_detection method work end-to-end. The integration
with the SAHITrackletBridge is exercised by
tests/test_sahi_tracklet_bridge.py.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.workers.tracklet_collector import Tracklet, TrackletCollector


class _StubRedis:
    """Minimal RedisState stub that records the published tracklet."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []
        self._buffer_keys: set[str] = set()

    def append_crop(self, camera_id: str, local_track_id: int, uri: str) -> None:
        self._buffer_keys.add(f"{camera_id}:{local_track_id}")

    def clear_buffer(self, camera_id: str, local_track_id: int) -> None:
        self._buffer_keys.discard(f"{camera_id}:{local_track_id}")

    def publish(self, stream: str, payload: dict[str, Any]) -> str:
        self.published.append((stream, payload))
        return f"{len(self.published)}-0"

    # Anything else needed by TrackletCollector.
    def __getattr__(self, item: str) -> Any:
        return MagicMock()


class _StubPostgres:
    def insert_tracklet(self, **_kwargs: Any) -> None:
        pass


class _StubMinio:
    bucket = "stub-bucket"

    def put_crop(self, **_kwargs: Any) -> str:
        return "s3://stub-bucket/x.jpg"

    def copy_object_within_bucket(self, **_kwargs: Any) -> None:
        pass

    def evidence_key(self, **_kwargs: Any) -> str:
        return "evidence/x.jpg"


@pytest.fixture
def collector() -> TrackletCollector:
    return TrackletCollector(
        pg=_StubPostgres(),
        redis=_StubRedis(),
        minio=_StubMinio(),
        site_id="site-test",
        zone_rows=[],
        min_track_age_frames=1,
        min_crops_per_tracklet=0,
        max_crops_per_tracklet=3,
        min_person_height_px=0.0,
    )


def test_tracklet_default_source_is_pphuman() -> None:
    """A Tracklet created without arguments has source='pphuman'."""
    tl = Tracklet(
        tracklet_id="x",
        camera_id="C",
        local_track_id=1,
        start_time=time.time(),
    )
    assert tl.source == "pphuman"
    assert tl.provisional is False


def test_on_sahi_detection_creates_provisional_tracklet(
    collector: TrackletCollector,
) -> None:
    """on_sahi_detection creates a Tracklet with source='sahi' and
    provisional=True."""
    collector.on_sahi_detection(
        camera_id="CAM_01",
        frame_id=42,
        timestamp_ms=int(time.time() * 1000),
        bbox=(10.0, 10.0, 50.0, 50.0),
        score=0.9,
    )
    in_flight = list(collector._in_flight.values())
    assert len(in_flight) == 1
    tl = in_flight[0]
    assert tl.source == "sahi"
    assert tl.provisional is True
    assert tl.camera_id == "CAM_01"
    assert tl.local_track_id == 42
    assert tl.frame_bboxes == [(10.0, 10.0, 50.0, 50.0)]


def test_on_sahi_detection_dedups_within_frame(
    collector: TrackletCollector,
) -> None:
    """Two SAHI detections for the same (camera, frame) update the
    same in-flight tracklet (one per frame; no MOT for SAHI)."""
    ts_ms = int(time.time() * 1000)
    collector.on_sahi_detection(
        camera_id="CAM_01",
        frame_id=1,
        timestamp_ms=ts_ms,
        bbox=(10.0, 10.0, 50.0, 50.0),
        score=0.9,
    )
    collector.on_sahi_detection(
        camera_id="CAM_01",
        frame_id=1,
        timestamp_ms=ts_ms + 50,
        bbox=(12.0, 12.0, 52.0, 52.0),
        score=0.85,
    )
    in_flight = list(collector._in_flight.values())
    assert len(in_flight) == 1


def test_emit_closed_tracklet_includes_source_and_provisional(
    collector: TrackletCollector,
) -> None:
    """emit_closed_tracklets publishes source + provisional flags."""
    ts = time.time()
    tl = Tracklet(
        tracklet_id="abc",
        camera_id="CAM_01",
        local_track_id=1,
        start_time=ts,
        end_time=ts,
        site_id="site-test",
        source="sahi",
        provisional=True,
    )
    tl.frame_bboxes.append((10.0, 10.0, 50.0, 50.0))
    tl.frame_count = 1
    collector.emit_closed_tracklets([tl])
    assert len(collector.redis.published) == 1
    _, payload = collector.redis.published[0]
    assert payload["source"] == "sahi"
    assert payload["provisional"] is True
    assert payload["tracklet_id"] == "abc"


def test_emit_closed_tracklet_default_pphuman_emits_pphuman_source(
    collector: TrackletCollector,
) -> None:
    """A PP-Human tracklet emits source='pphuman', provisional=False."""
    ts = time.time()
    tl = Tracklet(
        tracklet_id="xyz",
        camera_id="CAM_01",
        local_track_id=2,
        start_time=ts,
        end_time=ts,
        site_id="site-test",
    )
    tl.frame_bboxes.append((10.0, 10.0, 50.0, 50.0))
    tl.frame_count = 1
    collector.emit_closed_tracklets([tl])
    _, payload = collector.redis.published[0]
    assert payload["source"] == "pphuman"
    assert payload["provisional"] is False
