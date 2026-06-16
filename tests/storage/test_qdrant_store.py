"""Qdrant search always uses filters.

We test the search() method's required-filter contract with a fake client
(no real Qdrant needed).
"""

from __future__ import annotations

import numpy as np

from app.storage.qdrant_store import (
    COLLECTIONS,
    PAYLOAD_INDEX_FIELDS,
    QdrantStore,
    models,
)


class _FakeQueryResponse:
    def __init__(self, points=None):
        self.points = list(points or [])


class _FakeClient:
    """Tiny stand-in for qdrant_client.QdrantClient.query_points() that records
    the filter it received."""

    def __init__(self):
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeQueryResponse()


def test_qdrant_search_requires_filter(monkeypatch) -> None:
    """If the caller forgets the camera_id filter, the resolver must
    refuse. We test this at the public API surface."""
    store = QdrantStore("localhost", 6333)
    fake = _FakeClient()
    store._client = fake  # bypass connect()
    # Empty candidate list -> short-circuit (no Qdrant call issued)
    out = store.search(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        timestamp_gte=0,
        candidate_camera_ids=[],  # intentionally empty
        model_name="pphuman_strongbaseline",
        model_version="v1",
    )
    assert out == []
    assert len(fake.calls) == 0  # no search issued when no candidates


def test_qdrant_search_uses_filter(monkeypatch) -> None:
    store = QdrantStore("localhost", 6333)
    fake = _FakeClient()
    store._client = fake
    store.search(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        timestamp_gte=1_000,
        candidate_camera_ids=["CAM_01", "CAM_02"],
        model_name="pphuman_strongbaseline",
        model_version="v1",
    )
    assert len(fake.calls) == 1
    kwargs = fake.calls[0]
    assert "query_filter" in kwargs
    f = kwargs["query_filter"]
    keys = [c.key for c in f.must]
    assert "timestamp" in keys
    assert "camera_id" in keys
    assert "quality_score" in keys
    assert "model_name" in keys
    assert "model_version" in keys


def test_qdrant_collections_have_correct_dim() -> None:
    # PATCH (2026-06-15): the operator's MSMT17 TransReID model
    # produces 5x768=3840-dim features (JPM enabled), so the new
    # ``person_reid_transreid_msmt`` collection uses dim=3840.
    for name, dim, dist in COLLECTIONS:
        assert dim in (256, 768, 512, 3840), f"{name} unexpected dim {dim}"
        assert dist is models.Distance.COSINE


def test_payload_index_fields_complete() -> None:
    fields = {f for f, _ in PAYLOAD_INDEX_FIELDS}
    required = {
        "global_id",
        "tracklet_id",
        "camera_id",
        "zone_id",
        "site_id",
        "timestamp",
        "quality_score",
        "model_name",
        "model_version",
    }
    assert required.issubset(fields), f"Missing: {required - fields}"
