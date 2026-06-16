"""Travel-time Qdrant filter tests (PATCH-016).

The audit's PATCH-016 fix requires per-camera ``[gte, lte]`` timestamp
windows in the Qdrant query. We test the resolver + QdrantStore paths
using a fake Qdrant client (no live infra needed).
"""

from __future__ import annotations

import time

import numpy as np

from app.identity.camera_topology import CameraTopology
from app.identity.resolver import GlobalIdentityResolver, ResolverConfig
from app.storage.qdrant_store import QdrantStore


# ----------------------------------------------------------------------------
# QdrantStore.search_per_camera contract
# ----------------------------------------------------------------------------


class _FakeScoredPoint:
    def __init__(self, point_id: str, score: float, payload: dict):
        self.id = point_id
        self.score = score
        self.payload = payload


class _FakeQueryResponse:
    def __init__(self, points):
        self.points = list(points)


class _FakeQdrantClient:
    """Records every query_points() call and returns canned hits.

    The Qdrant ``Filter`` is a pydantic model; we introspect via
    attribute access (``f.must``), not ``f.get("must")``.
    """

    def __init__(self, hits_by_camera: dict[str, list[_FakeScoredPoint]]):
        self.calls: list[dict] = []
        self._hits = hits_by_camera

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        cam = self._extract_cam(kwargs)
        return _FakeQueryResponse(self._hits.get(cam, []))

    @staticmethod
    def _extract_cam(kwargs) -> str:
        qf = kwargs.get("query_filter")
        if qf is None:
            return ""
        for cond in getattr(qf, "must", []) or []:
            if getattr(cond, "key", None) == "camera_id":
                v = getattr(cond, "match", None)
                if v is not None:
                    return getattr(v, "value", "")
        return ""


def _qdrant_with_hits(hits: dict[str, list[_FakeScoredPoint]]) -> QdrantStore:
    store = QdrantStore("localhost", 6333)
    store._client = _FakeQdrantClient(hits)
    return store


def test_per_camera_search_emits_one_call_per_camera() -> None:
    """search_per_camera runs exactly one Qdrant call per camera."""
    hits = {
        "CAM_01": [_FakeScoredPoint("a", 0.9, {"global_id": "GID-A", "camera_id": "CAM_01"})],
        "CAM_02": [_FakeScoredPoint("b", 0.8, {"global_id": "GID-B", "camera_id": "CAM_02"})],
    }
    store = _qdrant_with_hits(hits)
    out = store.search_per_camera(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        per_camera_windows={"CAM_01": (100, 200), "CAM_02": (50, 150)},
        model_name="pphuman_strongbaseline",
        model_version="v1",
    )
    assert len(store._client.calls) == 2
    assert len(out) == 2
    # Top-1 should be the highest score (0.9).
    assert out[0].id == "a"
    assert out[1].id == "b"


def test_per_camera_search_includes_both_gte_and_lte_in_filter() -> None:
    """The Qdrant Range filter for each camera must include both gte
    and lte. This is the audit's PATCH-016 contract.
    """
    store = _qdrant_with_hits({})
    store.search_per_camera(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        per_camera_windows={"CAM_02": (100, 200)},
        model_name="pphuman_strongbaseline",
        model_version="v1",
    )
    assert len(store._client.calls) == 1
    kwargs = store._client.calls[0]
    ts_cond = None
    for c in kwargs["query_filter"].must:
        if getattr(c, "key", None) == "timestamp":
            ts_cond = c
            break
    assert ts_cond is not None
    assert ts_cond.range.gte == 100
    assert ts_cond.range.lte == 200


def test_per_camera_search_short_circuits_on_empty() -> None:
    store = _qdrant_with_hits({})
    out = store.search_per_camera(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        per_camera_windows={},
        model_name="pphuman_strongbaseline",
        model_version="v1",
    )
    assert out == []
    assert store._client.calls == []


def test_per_camera_search_dedups() -> None:
    store = _qdrant_with_hits(
        {
            "CAM_01": [
                _FakeScoredPoint("a", 0.9, {"global_id": "GID-A", "camera_id": "CAM_01"}),
                _FakeScoredPoint("b", 0.8, {"global_id": "GID-B", "camera_id": "CAM_01"}),
            ],
            "CAM_02": [
                _FakeScoredPoint("c", 0.7, {"global_id": "GID-C", "camera_id": "CAM_02"}),
            ],
        }
    )
    out = store.search_per_camera(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        per_camera_windows={"CAM_01": (0, 1000), "CAM_02": (0, 1000)},
        model_name="pphuman_strongbaseline",
        model_version="v1",
        top_k=10,
    )
    assert {h.id for h in out} == {"a", "b", "c"}


def test_per_camera_search_respects_top_k() -> None:
    hits = {
        "CAM_01": [
            _FakeScoredPoint(f"id{i}", 0.9 - i * 0.01, {"camera_id": "CAM_01"}) for i in range(20)
        ],
    }
    store = _qdrant_with_hits(hits)
    out = store.search_per_camera(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        per_camera_windows={"CAM_01": (0, 1000)},
        model_name="pphuman_strongbaseline",
        model_version="v1",
        top_k=5,
    )
    assert len(out) == 5
    # Sorted by score desc.
    scores = [h.score for h in out]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------------------
# Resolver integration: per-camera travel-time windowing
# ----------------------------------------------------------------------------


class _ResolverDouble:
    """Minimal stand-in for the real resolver that only exposes the
    search internals we want to test."""

    def __init__(self, qdrant: QdrantStore, topology: CameraTopology, persistence: int = 86_400):
        self.qdrant = qdrant
        self.topology = topology
        self.config = ResolverConfig(persistence_window_seconds=persistence)
        self.model_name = "pphuman_strongbaseline"
        self.model_version = "v1"

    _search_with_filters = GlobalIdentityResolver._search_with_filters
    _qdrant_collection_for = staticmethod(GlobalIdentityResolver._qdrant_collection_for)


def _topology_with_link(
    from_cam: str,
    to_cam: str,
    *,
    min_travel: int = 10,
    max_travel: int = 90,
    enabled: bool = True,
) -> CameraTopology:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": from_cam,
                "to_camera_id": to_cam,
                "min_travel_seconds": min_travel,
                "max_travel_seconds": max_travel,
                "transition_probability": 0.8,
                "enabled": enabled,
                "notes": "",
            }
        ]
    )
    return t


def test_resolver_search_excludes_too_fast_candidate() -> None:
    """A candidate from 5 s ago is too fast — the topology says
    min_travel=10 s. The resolver's per-camera window is
    ``[ts - 90, ts - 10]`` which excludes 5-s-old candidates.
    """
    # The link is "from CAM_02 to CAM_01": a person walks from
    # CAM_02 to CAM_01. The candidate (last-seen) is CAM_02; the
    # new (source) camera is CAM_01.
    top = _topology_with_link("CAM_02", "CAM_01", min_travel=10, max_travel=90)
    store = _qdrant_with_hits({})  # no hits returned
    r = _ResolverDouble(store, top)
    ts = time.time()
    r._search_with_filters(
        query_vec=np.zeros(256, dtype=np.float32),
        candidate_cams=["CAM_02"],
        ts=ts,
        source_camera_id="CAM_01",
    )
    # The resolver should have run ONE Qdrant call for CAM_02.
    assert len(store._client.calls) == 1
    kwargs = store._client.calls[0]
    ts_cond = next(c for c in kwargs["query_filter"].must if c.key == "timestamp")
    # Window: [ts-90, ts-10]  (the min_travel and max_travel from the link).
    assert ts_cond.range.gte == int(ts - 90)
    assert ts_cond.range.lte == int(ts - 10)


def test_resolver_search_excludes_too_slow_candidate() -> None:
    """A candidate from 5 h ago is too slow (max_travel=90 s). The
    resolver's per-camera window excludes it.
    """
    top = _topology_with_link("CAM_02", "CAM_01", min_travel=10, max_travel=90)
    store = _qdrant_with_hits({})
    r = _ResolverDouble(store, top)
    ts = time.time()
    r._search_with_filters(
        query_vec=np.zeros(256, dtype=np.float32),
        candidate_cams=["CAM_02"],
        ts=ts,
        source_camera_id="CAM_01",
    )
    kwargs = store._client.calls[0]
    ts_cond = next(c for c in kwargs["query_filter"].must if c.key == "timestamp")
    # Window: [ts-90, ts-10] — the resolver rejects anything older
    # than 90 s and anything newer than 10 s.
    assert ts_cond.range.gte == int(ts - 90)
    assert ts_cond.range.lte == int(ts - 10)


def test_resolver_search_disabled_link_excluded() -> None:
    """A candidate with ``enabled=False`` topology link is not
    queried at all (the resolver skips it).
    """
    top = _topology_with_link("CAM_02", "CAM_01", enabled=False)
    store = _qdrant_with_hits({})
    r = _ResolverDouble(store, top)
    r._search_with_filters(
        query_vec=np.zeros(256, dtype=np.float32),
        candidate_cams=["CAM_02"],
        ts=time.time(),
        source_camera_id="CAM_01",
    )
    assert store._client.calls == []  # no Qdrant call for a disabled link


def test_resolver_search_same_cam_uses_persistence_window() -> None:
    """Same-camera candidates (Stage 1) use the full persistence
    window, not the travel-time window.
    """
    top = _topology_with_link("CAM_02", "CAM_01")
    store = _qdrant_with_hits({})
    r = _ResolverDouble(store, top, persistence=86_400)
    ts = time.time()
    r._search_with_filters(
        query_vec=np.zeros(256, dtype=np.float32),
        candidate_cams=["CAM_01", "CAM_02"],
        ts=ts,
        source_camera_id="CAM_01",
    )
    # Two Qdrant calls: one for CAM_01 (same cam, persistence window)
    # and one for CAM_02 (linked, travel window).
    cams = []
    for kw in store._client.calls:
        c = next(c for c in kw["query_filter"].must if c.key == "camera_id")
        cams.append(c.match.value)
        ts_c = next(c for c in kw["query_filter"].must if c.key == "timestamp")
        if c.match.value == "CAM_01":
            # Stage 1: full persistence window.
            assert ts_c.range.gte == int(ts - 86_400)
            assert ts_c.range.lte is None
        else:
            # Stage 2: travel-time window.
            assert ts_c.range.gte == int(ts - 90)
            assert ts_c.range.lte == int(ts - 10)
    assert "CAM_01" in cams and "CAM_02" in cams


def test_resolver_search_off_when_travel_window_only_false() -> None:
    """When ``travel_window_only=False`` the resolver falls back to
    the broad 24 h window (smoke-test path).
    """
    top = _topology_with_link("CAM_02", "CAM_01")
    store = _qdrant_with_hits({})
    r = _ResolverDouble(store, top)
    r._search_with_filters(
        query_vec=np.zeros(256, dtype=np.float32),
        candidate_cams=["CAM_02"],
        ts=time.time(),
        source_camera_id="CAM_01",
        travel_window_only=False,
    )
    # Single call using the broad-window search() (not per-camera).
    assert len(store._client.calls) == 1
    kwargs = store._client.calls[0]
    ts_cond = next(c for c in kwargs["query_filter"].must if c.key == "timestamp")
    assert ts_cond.range.gte == int(time.time() - 86_400)


def test_resolver_search_no_candidates_returns_empty() -> None:
    top = _topology_with_link("CAM_01", "CAM_02")
    store = _qdrant_with_hits({})
    r = _ResolverDouble(store, top)
    out = r._search_with_filters(
        query_vec=np.zeros(256, dtype=np.float32),
        candidate_cams=[],
        ts=time.time(),
        source_camera_id="CAM_01",
    )
    assert out == []
    assert store._client.calls == []
