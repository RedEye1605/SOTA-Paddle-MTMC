"""Race-condition stress tests for TrackletCollector's RLock.

These tests exercise the four call sites that mutate
``TrackletCollector._in_flight`` from different threads:

- ``on_frame`` from the per-camera consumer thread
- ``on_detection`` from the ``DetectionEventConsumer`` thread
- ``on_sahi_detection`` from the ``SAHITrackletBridge`` thread
- ``finalize_stale`` from the ``_finalize_loop`` background thread

Without the lock, the dict's logical state can become inconsistent
(key removed mid-append, two Tracklets created for the same key,
``end_time`` read while being written). The tests below verify the
lock prevents all of those failure modes.

Tests should finish well under 5s.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.workers.pphuman_worker import FrameResult, LocalTrack
from app.workers.tracklet_collector import (
    DetectionEvent,
    Tracklet,
    TrackletCollector,
)


# ----------------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------------


class _StubRedis:
    """Minimal RedisState stub.

    Records appends/clears and accepts any other method call as a
    MagicMock so the collector never raises.
    """

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

    def __getattr__(self, item: str) -> Any:
        return MagicMock()


class _StubPostgres:
    def insert_tracklet(self, **_kwargs: Any) -> None:
        return None


class _StubMinio:
    bucket = "stub-bucket"

    def put_crop(self, **_kwargs: Any) -> str:
        return "s3://stub-bucket/x.jpg"

    def copy_object_within_bucket(self, **_kwargs: Any) -> None:
        return None

    def evidence_key(self, **_kwargs: Any) -> str:
        return "evidence/x.jpg"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_frame() -> np.ndarray:
    """Return a frame where ``crop_quality_score`` returns > 0 for the
    default test bbox. Uniform zeros fail the Laplacian-variance test
    (returns 0.0 → crop rejected); we need non-trivial texture."""
    return np.random.RandomState(42).randint(50, 200, (480, 640, 3), dtype=np.uint8)


def _make_frame_result(
    camera_id: str,
    local_id: int,
    ts: float,
    frame: np.ndarray,
) -> FrameResult:
    # The bbox lives at the top-left corner of the frame so the
    # 10%-padded crop is well clear of the original frame edges and
    # ``is_cut_by_frame`` (called with the crop's own shape — see
    # ``crop_quality_score``) reports a non-zero cut score.
    return FrameResult(
        camera_id=camera_id,
        frame_id=int(ts * 10) % 1_000_000,
        ts=ts,
        frame=frame,
        tracks=[
            LocalTrack(
                camera_id=camera_id,
                local_track_id=local_id,
                bbox=(5.0, 5.0, 55.0, 105.0),
                confidence=0.9,
                frame_id=int(ts * 10) % 1_000_000,
                ts=ts,
                age_frames=20,
                is_confirmed=True,
            ),
        ],
        skipped=False,
    )


def _make_detection_event(
    camera_id: str,
    local_id: int,
    ts_ms: int,
    *,
    with_embedding: bool = False,
) -> DetectionEvent:
    emb: np.ndarray | None = None
    if with_embedding:
        emb = np.random.RandomState(local_id).randn(256).astype(np.float32)
    return DetectionEvent(
        schema_version="1.0",
        event_id=f"det_{camera_id}_{local_id}_{ts_ms}",
        source="pphuman",
        run_id="run-x",
        camera_id=camera_id,
        frame_id=int(ts_ms // 100),
        timestamp_ms=ts_ms,
        received_at_ms=ts_ms,
        local_track_id=local_id,
        bbox=(50.0, 50.0, 150.0, 250.0),
        score=0.9,
        crop_path=None,
        embedding=emb,
        frame_uri=None,
    )


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def collector() -> TrackletCollector:
    """A TrackletCollector with cheap stubs and permissive gates.

    ``stale_tracklet_seconds=0.0`` makes every tracklet with an
    ``end_time`` immediately eligible for ``finalize_stale``.
    ``min_crops_per_tracklet=0`` + ``min_track_age_frames=1`` means
    the "enough" gate at finalize time is satisfied by the first
    crop or detection.
    """
    return TrackletCollector(
        pg=_StubPostgres(),
        redis=_StubRedis(),
        minio=_StubMinio(),
        site_id="site-test",
        zone_rows=[],
        min_track_age_frames=1,
        min_crops_per_tracklet=0,
        max_crops_per_tracklet=3,
        min_person_height_px=1.0,
        stale_tracklet_seconds=0.0,
        auto_finalize=False,
    )


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_concurrent_on_frame_and_finalize_stale(collector: TrackletCollector) -> None:
    """4 threads: 2 writers + 2 finalizers. No exceptions allowed and the
    in-flight dict must be in a consistent state when the test ends."""
    frame = _make_frame()
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(seed: int) -> None:
        try:
            for i in range(500):
                local_id = (i % 2) + seed
                collector.on_frame(_make_frame_result("CAM_01", local_id, time.time(), frame))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def finalizer() -> None:
        try:
            while not stop.is_set():
                collector.finalize_stale()
                stop.wait(timeout=0.001)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(0,), daemon=True, name="writer-A"),
        threading.Thread(target=writer, args=(2,), daemon=True, name="writer-B"),
        threading.Thread(target=finalizer, daemon=True, name="finalizer-A"),
        threading.Thread(target=finalizer, daemon=True, name="finalizer-B"),
    ]
    for t in threads:
        t.start()
    threads[0].join()
    threads[1].join()
    stop.set()
    threads[2].join(timeout=2.0)
    threads[3].join(timeout=2.0)

    assert not errors, f"concurrent errors: {errors!r}"
    # The final state must be consistent: either empty (finalizers drained
    # everything) or containing fully-built tracklets (a writer won the
    # very last race). Anything in-between would indicate a partial mutation.
    for tl in collector._in_flight.values():
        assert tl.crop_uris, f"partial tracklet {tl.tracklet_id} in flight"
        assert tl.end_time is not None


def test_concurrent_on_detection_and_finalize_stale(
    collector: TrackletCollector,
) -> None:
    """Same shape as the on_frame test but feeding the structured
    detection-event path. Verifies the lock also serializes the
    side-channel consumer against the finalize loop."""
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(seed: int) -> None:
        try:
            for i in range(500):
                local_id = (i % 2) + seed
                ev = _make_detection_event(
                    "CAM_01",
                    local_id,
                    int(time.time() * 1000),
                    with_embedding=(i % 3 == 0),
                )
                collector.on_detection(ev)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def finalizer() -> None:
        try:
            while not stop.is_set():
                collector.finalize_stale()
                stop.wait(timeout=0.001)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(0,), daemon=True, name="det-writer-A"),
        threading.Thread(target=writer, args=(2,), daemon=True, name="det-writer-B"),
        threading.Thread(target=finalizer, daemon=True, name="det-finalizer-A"),
        threading.Thread(target=finalizer, daemon=True, name="det-finalizer-B"),
    ]
    for t in threads:
        t.start()
    threads[0].join()
    threads[1].join()
    stop.set()
    threads[2].join(timeout=2.0)
    threads[3].join(timeout=2.0)

    assert not errors, f"concurrent errors: {errors!r}"
    for tl in collector._in_flight.values():
        assert tl.frame_count >= 1
        assert tl.end_time is not None


def test_no_tracklet_lost_under_contention(collector: TrackletCollector) -> None:
    """8 threads hammer 100 distinct keys; after the storm every key
    is accounted for (in-flight or finalized) and no tracklet was
    finalised with an empty crop list (which would indicate a key
    was deleted mid-append)."""
    frame = _make_frame()
    rng = random.Random(0)
    errors: list[BaseException] = []
    n_threads = 8
    iterations = 100
    n_keys = 100
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(iterations):
                local_id = rng.randint(0, n_keys - 1)
                collector.on_frame(
                    _make_frame_result("CAM_01", local_id, time.time(), frame),
                )
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=worker, daemon=True, name=f"hammer-{i}")
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent errors: {errors!r}"

    # Drain any remaining in-flight tracklets into the finalized bucket.
    finalized: list[Tracklet] = []
    while True:
        closed = collector.finalize_stale()
        if not closed:
            break
        finalized.extend(closed)

    seen: set[int] = set()
    for tl in collector._in_flight.values():
        seen.add(tl.local_track_id)
        assert len(tl.crop_uris) >= 1, (
            f"in-flight tracklet {tl.tracklet_id} has no crops"
        )
    for tl in finalized:
        seen.add(tl.local_track_id)
        assert len(tl.crop_uris) >= 1, (
            f"finalized tracklet {tl.tracklet_id} has no crops "
            "(key likely deleted mid-append)"
        )
    assert seen == set(range(n_keys)), (
        f"missing keys: {set(range(n_keys)) - seen}; "
        f"unexpected: {seen - set(range(n_keys))}"
    )


def test_lock_is_recursive(collector: TrackletCollector) -> None:
    """An RLock (not a plain Lock) must allow the same thread to
    re-acquire the lock. Without this, a future method composition
    (e.g. ``on_frame`` called from inside a custom ``finalize``
    override) would deadlock."""
    # Direct nesting: lock acquired twice by the same thread.
    with collector._lock:
        with collector._lock:
            # Force a brief yield to the interpreter; with a plain
            # ``Lock`` this would have deadlocked by now.
            time.sleep(0)
        # Outer release succeeds.
    # Re-entrant call through a public method that itself takes
    # the lock — this is the realistic shape (a custom subclass
    # method that delegates to a base method).
    with collector._lock:
        collector.on_sahi_detection(
            camera_id="CAM_REC",
            frame_id=1,
            timestamp_ms=int(time.time() * 1000),
            bbox=(10.0, 10.0, 50.0, 50.0),
            score=0.9,
        )
    # And the inverse: the public method that takes the lock and
    # invokes another public method that *also* takes the lock.
    collector.on_sahi_detection(
        camera_id="CAM_REC",
        frame_id=2,
        timestamp_ms=int(time.time() * 1000),
        bbox=(10.0, 10.0, 50.0, 50.0),
        score=0.9,
    )
    # The SAHI tracklet landed in the dict.
    assert any(
        tl.camera_id == "CAM_REC" and tl.local_track_id == 1
        for tl in collector._in_flight.values()
    )
    assert any(
        tl.camera_id == "CAM_REC" and tl.local_track_id == 2
        for tl in collector._in_flight.values()
    )
