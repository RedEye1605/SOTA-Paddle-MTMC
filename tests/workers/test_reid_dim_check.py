"""ReID worker dim-check tests (BUG fix).

The Qdrant collection ``person_reid_transreid_msmt`` is configured for
3840-dim vectors (TransReID MSMT17 with 5x768 JPM). The vendor
pipeline's side-channel ``embedding`` field carries 256-dim PP-Human
strongbaseline attribute logits. ``ReIDWorker.process_tracklet`` must
validate the dim of every embedding before upserting, skip wrong-dim
rows with a loud warning, and increment a local counter — never write
256-dim vectors into a 3840-dim collection.
"""
from __future__ import annotations

import logging
import time
import uuid

import numpy as np

from app.core.runtime_mode import RuntimeMode
from app.reid.base import ReIDConfig, ReIDAdapter
from app.workers.reid_worker import ReIDWorker
from app.workers.tracklet_collector import Tracklet


class _FakeAdapter(ReIDAdapter):
    def __init__(
        self,
        embedding_dim: int = 3840,
        qdrant_collection: str = "person_reid_transreid_msmt",
    ) -> None:
        self.config = ReIDConfig(
            name="transreid",
            embedding_dim=embedding_dim,
            qdrant_collection=qdrant_collection,
        )

    def load(self) -> None:
        return None

    def warmup(self) -> None:
        return None

    def extract(self, crops):
        return np.zeros((0, self.embedding_dim), dtype=np.float32)


class _FakeQdrant:
    def __init__(self) -> None:
        self.upsert_calls: list[tuple] = []

    def upsert_point(self, collection, vector, payload, point_id=None) -> str:
        self.upsert_calls.append((collection, vector, payload, point_id))
        return point_id or str(uuid.uuid4())


class _FakePG:
    def __init__(self) -> None:
        self.inserts: list[dict] = []

    def insert_tracklet_embedding(self, **kwargs) -> None:
        self.inserts.append(kwargs)


class _FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    def publish(self, stream, payload) -> None:
        self.published.append((stream, payload))


def _make_worker() -> tuple[ReIDWorker, _FakeAdapter, _FakeQdrant, _FakePG, _FakeRedis]:
    adapter = _FakeAdapter(embedding_dim=3840)
    qdrant = _FakeQdrant()
    pg = _FakePG()
    redis = _FakeRedis()
    worker = ReIDWorker(
        adapter=adapter,
        pg=pg,
        qdrant=qdrant,
        redis=redis,
        minio=None,
        mode=RuntimeMode.PRODUCTION,
    )
    return worker, adapter, qdrant, pg, redis


def _make_tracklet(
    embeddings: list[np.ndarray],
    *,
    tracklet_id: str = "TL-001",
    camera_id: str = "CAM_01",
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        camera_id=camera_id,
        local_track_id=1,
        start_time=time.time(),
        embeddings=embeddings,
    )


# -----------------------------------------------------------------------------
# Test 1: 256-dim side-channel embedding is skipped.
# -----------------------------------------------------------------------------


def test_256_dim_embedding_is_skipped(caplog) -> None:
    """A 256-dim side-channel embedding must NOT be upserted to the
    3840-dim Qdrant collection. The worker must log a warning,
    increment ``wrong_dim_skips_total``, and return None.
    """
    worker, _, qdrant, pg, redis = _make_worker()
    vec = np.ones(256, dtype=np.float32) / 16.0
    tl = _make_tracklet([vec])

    with caplog.at_level(logging.WARNING, logger="app.workers.reid_worker"):
        out = worker.process_tracklet(tl)

    assert out is None
    assert qdrant.upsert_calls == []
    assert pg.inserts == []
    assert redis.published == []
    assert worker.wrong_dim_skips_total == 1
    dim_warnings = [r for r in caplog.records if "actual_dim=256" in r.message]
    assert dim_warnings, (
        f"expected a warning naming actual_dim=256; got: "
        f"{[r.message for r in caplog.records]}"
    )


# -----------------------------------------------------------------------------
# Test 2: 3840-dim embedding is upserted normally.
# -----------------------------------------------------------------------------


def test_3840_dim_embedding_is_upserted(caplog) -> None:
    """A 3840-dim side-channel embedding MUST be upserted. The
    worker must not log a dim-mismatch warning and must not
    increment ``wrong_dim_skips_total``.
    """
    worker, _, qdrant, pg, redis = _make_worker()
    vec = np.ones(3840, dtype=np.float32) / np.sqrt(3840.0)
    tl = _make_tracklet([vec])

    with caplog.at_level(logging.WARNING, logger="app.workers.reid_worker"):
        out = worker.process_tracklet(tl)

    assert out is not None
    assert out.shape == (3840,)
    assert len(qdrant.upsert_calls) == 1
    collection, vector, payload, point_id = qdrant.upsert_calls[0]
    assert collection == "person_reid_transreid_msmt"
    assert vector.shape == (3840,)
    assert payload["tracklet_id"] == "TL-001"
    assert payload["camera_id"] == "CAM_01"
    assert len(pg.inserts) == 1
    assert pg.inserts[0]["embedding_dim"] == 3840
    assert len(redis.published) == 1
    assert redis.published[0][0] == "stream:embeddings"
    assert worker.wrong_dim_skips_total == 0
    assert not any("actual_dim=" in r.message for r in caplog.records)


# -----------------------------------------------------------------------------
# Test 3: warning log includes tracklet_id and camera_id.
# -----------------------------------------------------------------------------


def test_wrong_dim_logs_tracklet_id(caplog) -> None:
    """The dim-mismatch warning must include both ``tracklet_id`` and
    ``camera_id`` so the operator can correlate with the source
    stream event.
    """
    worker, _, qdrant, pg, redis = _make_worker()
    vec = np.ones(256, dtype=np.float32) / 16.0
    tl = _make_tracklet([vec], tracklet_id="TL-XYZ-99", camera_id="CAM_42")

    with caplog.at_level(logging.WARNING, logger="app.workers.reid_worker"):
        out = worker.process_tracklet(tl)

    assert out is None
    matches = [
        r
        for r in caplog.records
        if "TL-XYZ-99" in r.message and "CAM_42" in r.message
    ]
    assert matches, (
        f"expected a warning naming both tracklet_id and camera_id; "
        f"got: {[r.message for r in caplog.records]}"
    )
    assert qdrant.upsert_calls == []
    assert pg.inserts == []
    assert redis.published == []
    assert worker.wrong_dim_skips_total == 1
