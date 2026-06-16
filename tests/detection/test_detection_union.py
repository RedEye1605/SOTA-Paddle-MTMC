"""Tests for the DetectionUnionMerger.

These tests use a fake Redis (``fakeredis``-style in-memory dict)
to exercise the read paths without a real Redis server. The merger
must NEVER call XREAD BLOCK 0 in the pre-MOT path; this is enforced
by ``test_union_no_block``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import pytest

from app.detection.detection_union import (
    DetectionUnionMerger,
    UnionConfig,
)


class FakeRedis:
    """Minimal in-memory stand-in for the redis-py client used in tests.

    Supports GET, SET (with EX), XADD (with MAXLEN), XREVRANGE.
    All methods are non-blocking; no BLOCK 0 anywhere.
    """

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {
            "stream:detections_sahi": []
        }

    def get(self, key: str) -> Optional[str]:
        v = self.kv.get(key)
        if v is None:
            return None
        return v

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:  # noqa: A002
        self.kv[key] = value

    def xadd(
        self,
        name: str,
        fields: dict[str, str],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        eid = f"{int(time.time() * 1e6)}-{len(self.streams.get(name, []))}"
        self.streams.setdefault(name, []).append((eid, fields))
        if maxlen is not None and len(self.streams[name]) > maxlen:
            self.streams[name] = self.streams[name][-maxlen:]
        return eid

    def xrevrange(
        self, name: str, max: str = "+", min: str = "-", count: Optional[int] = None
    ) -> list[tuple[str, dict[str, str]]]:
        items = list(reversed(self.streams.get(name, [])))
        if count is not None:
            items = items[:count]
        return items


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


@pytest.fixture
def cfg() -> UnionConfig:
    return UnionConfig(
        union_nms_iou=0.5,
        union_min_conf=0.4,
        stale_ms=400,
        timestamp_slack_ms=50,
        stream="stream:detections_sahi",
        latest_key_prefix="sahi:latest:",
    )


@pytest.fixture
def merger(fake_redis: FakeRedis, cfg: UnionConfig) -> DetectionUnionMerger:
    return DetectionUnionMerger(redis=fake_redis, config=cfg, camera_id="CAM_01")  # type: ignore[arg-type]


def _pphuman_bbox(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score}


def _seed_latest(
    fake_redis: FakeRedis,
    camera_id: str,
    timestamp_ms: int,
    detections: list[list[float]],
) -> None:
    fake_redis.set(
        f"sahi:latest:{camera_id}",
        json.dumps(
            {
                "camera_id": camera_id,
                "frame_id": 100,
                "timestamp_ms": timestamp_ms,
                "detections": detections,
            }
        ),
    )


def test_union_pphuman_only(merger: DetectionUnionMerger):
    """No SAHI data → return pphuman_dets unchanged, all tagged pphuman."""
    pphuman = [_pphuman_bbox(10, 10, 50, 50, 0.9)]
    out = merger.merge(pphuman, current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1
    assert out[0].source == "pphuman"
    assert out[0].score == 0.9


def test_union_sahi_only(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """pphuman_dets is empty → return SAHI dets, tagged sahi."""
    _seed_latest(fake_redis, "CAM_01", 1000, [[20.0, 20.0, 60.0, 60.0, 0.8]])
    out = merger.merge([], current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1
    assert out[0].source == "sahi"
    assert abs(out[0].score - 0.8) < 1e-6


def test_union_disjoint(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """PP-Human and SAHI bboxes don't overlap → all kept."""
    pphuman = [_pphuman_bbox(10, 10, 50, 50, 0.9)]
    _seed_latest(fake_redis, "CAM_01", 1000, [[200, 200, 300, 300, 0.7]])
    out = merger.merge(pphuman, current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 2
    sources = {d.source for d in out}
    assert sources == {"pphuman", "sahi"}


def test_union_overlap_higher_wins(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """Overlapping bboxes → higher score wins, lower dropped."""
    pphuman = [_pphuman_bbox(10, 10, 100, 100, 0.95)]
    _seed_latest(
        fake_redis, "CAM_01", 1000,
        [[15, 15, 95, 95, 0.7]],  # IoU ~ 0.85 with the pphuman bbox
    )
    out = merger.merge(pphuman, current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1
    assert out[0].source == "merged"
    assert out[0].score == 0.95  # pphuman wins


def test_union_timestamp_filter(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """SAHI dets outside the timestamp window are dropped."""
    pphuman = [_pphuman_bbox(10, 10, 50, 50, 0.9)]
    _seed_latest(fake_redis, "CAM_01", 5000, [[20, 20, 60, 60, 0.8]])  # 4s old
    out = merger.merge(pphuman, current_timestamp_ms=1000, current_frame_id=1)
    # SAHI dropped; only pphuman survives.
    assert len(out) == 1
    assert out[0].source == "pphuman"


def test_union_stale_drop(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """SAHI dets older than stale_ms are dropped (even if in window)."""
    pphuman = [_pphuman_bbox(10, 10, 50, 50, 0.9)]
    _seed_latest(
        fake_redis, "CAM_01", 1000 - 500,  # 500ms old, > 400ms stale
        [[20, 20, 60, 60, 0.8]],
    )
    out = merger.merge(pphuman, current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1
    assert out[0].source == "pphuman"


def test_union_min_conf_filter(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """SAHI dets below union_min_conf are dropped."""
    pphuman = [_pphuman_bbox(10, 10, 50, 50, 0.9)]
    _seed_latest(
        fake_redis, "CAM_01", 1000,
        [[200, 200, 300, 300, 0.2]],  # below default 0.4
    )
    out = merger.merge(pphuman, current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1
    assert out[0].source == "pphuman"


def test_union_no_block(merger: DetectionUnionMerger):
    """Verify the merger source does NOT use XREAD BLOCK 0."""
    import inspect

    from app.detection import detection_union

    src = inspect.getsource(detection_union)
    # The pre-MOT path must use only non-blocking reads.
    assert "BLOCK 0" not in src, "BLOCK 0 found in detection_union.py"
    assert "xread(" not in src.lower(), "xread call found in detection_union.py"


def test_union_uses_get_then_xrevrange(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """The merger must try GET first, then XREVRANGE as fallback."""
    _seed_latest(fake_redis, "CAM_01", 1000, [[20, 20, 60, 60, 0.8]])
    out = merger.merge([], current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1

    # Now clear the latest key and add only to the stream. The
    # merger should still find the data via XREVRANGE.
    fake_redis.kv.clear()
    fake_redis.streams["stream:detections_sahi"].append(
        ("1-0", {
            "camera_id": "CAM_01",
            "frame_id": "100",
            "timestamp_ms": "1000",
            "detections": json.dumps([[30, 30, 70, 70, 0.75]]),
        })
    )
    out = merger.merge([], current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1
    assert out[0].source == "sahi"


def test_union_camera_id_filter(merger: DetectionUnionMerger, fake_redis: FakeRedis):
    """SAHI events for other cameras must be ignored."""
    _seed_latest(fake_redis, "CAM_02", 1000, [[20, 20, 60, 60, 0.8]])
    out = merger.merge([], current_timestamp_ms=1000, current_frame_id=1)
    # CAM_02 event ignored; merger returns empty.
    assert out == []


def test_union_redis_error_returns_pphuman_only(
    merger: DetectionUnionMerger, monkeypatch
):
    """Redis error during read → return pphuman_dets unchanged."""
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("simulated redis down")

    monkeypatch.setattr(merger._redis, "get", boom)
    pphuman = [_pphuman_bbox(10, 10, 50, 50, 0.9)]
    out = merger.merge(pphuman, current_timestamp_ms=1000, current_frame_id=1)
    assert len(out) == 1
    assert out[0].source == "pphuman"
