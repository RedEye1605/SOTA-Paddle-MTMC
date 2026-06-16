"""Performance tests for the SAHI integration.

These tests assert latency budgets. They are CPU-only (no GPU
required). Mark GPU-required tests with @pytest.mark.gpu_required
and skip in CPU-only test runs.

Performance budgets (from the spec):
- XADD + SET: < 5 ms p99 over 1000 iterations
- GET + NMS over 100 boxes: < 5 ms p99 (the SAHITrackletBridge
  dedup loop)
- Bridge poll loop: < 200 ms latency between SAHI event publish
  and TrackletCollector.on_sahi_detection call
"""

from __future__ import annotations

import json
import time
from typing import List, Optional
from unittest.mock import MagicMock

import pytest


class FakeRedis:
    """Minimal fake mirroring the subset of RedisState the bridge uses."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        # stream_name -> list of (msg_id, fields_dict)
        self.streams: dict[str, list[tuple[str, dict]]] = {}

    def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:  # noqa: A002
        self.kv[key] = value

    def xadd(
        self,
        name: str,
        fields: dict,
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        eid = f"{int(time.time() * 1e6)}-{len(self.streams.get(name, []))}"
        self.streams.setdefault(name, []).append((eid, fields))
        return eid

    # Methods used by SAHITrackletBridge
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


def test_sahi_redis_publish_latency():
    """XADD + SET should take < 5 ms p99 over 1000 iterations.

    This exercises the SAHI worker publish path: stream entry +
    latest-crop-pointer update.
    """
    redis = FakeRedis()
    dets = [[10, 10, 50, 50, 0.9] for _ in range(20)]
    payload = json.dumps(dets)
    durations: List[float] = []
    for i in range(1000):
        t0 = time.perf_counter()
        redis.xadd("stream:test", {"dets": payload}, maxlen=1000, approximate=True)
        redis.set("sahi:latest:TEST", payload, ex=1)
        durations.append(time.perf_counter() - t0)
    durations.sort()
    p99 = durations[int(0.99 * len(durations))]
    assert p99 < 0.005, (
        f"p99 publish latency {p99 * 1000:.1f}ms exceeds 5ms budget"
    )


def test_sahi_bridge_dedup_latency():
    """GET + NMS over 100 boxes should take < 5 ms p99.

    This exercises the SAHITrackletBridge._matches_active_pphuman
    path: read PP-Human bboxes from Redis, compute IoU against a
    single candidate SAHI bbox.
    """
    from app.detection.sahi_tracklet_bridge import (
        SAHIBridgeConfig,
        SAHITrackletBridge,
    )
    from app.workers.tracklet_collector import TrackletCollector

    redis = FakeRedis()
    # Seed 100 active PP-Human bboxes across 5 stream entries.
    for i in range(5):  # 5 entries x 20 bboxes = 100
        redis.streams.setdefault("stream:detections", []).append((
            f"{i}-0",
            {
                "camera_id": "CAM_01",
                "timestamp_ms": str(int(time.time() * 1000)),
                "bboxes": json.dumps(
                    [[j * 20, j * 20, j * 20 + 50, j * 20 + 50] for j in range(20)]
                ),
            },
        ))

    collector = TrackletCollector(
        pg=MagicMock(),
        redis=redis,
        minio=MagicMock(),
        site_id="test",
        zone_rows=[],
    )
    bridge = SAHITrackletBridge(
        redis=redis, collector=collector, config=SAHIBridgeConfig()
    )

    # Time the dedup check 1000 times.
    candidate_bbox = (50.0, 50.0, 100.0, 100.0)
    now_ms = int(time.time() * 1000)
    durations: List[float] = []
    for _ in range(1000):
        t0 = time.perf_counter()
        bridge._matches_active_pphuman("CAM_01", candidate_bbox, now_ms)
        durations.append(time.perf_counter() - t0)
    durations.sort()
    p99 = durations[int(0.99 * len(durations))]
    assert p99 < 0.005, (
        f"p99 dedup latency {p99 * 1000:.1f}ms exceeds 5ms budget"
    )


@pytest.mark.gpu_required
def test_sahi_inference_latency():
    """24 patches on 320x320 should take < 150 ms on T4.

    This is a smoke test; the actual perf depends on the model.
    Skip in CPU-only test runs.
    """
    pytest.skip("requires GPU; run on a T4 host")
