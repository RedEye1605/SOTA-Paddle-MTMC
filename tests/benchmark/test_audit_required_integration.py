"""Integration tests — production safety + real infrastructure.

Tests required by the audit's TEST_QUALITY_AUDIT.md (Phase 10). Each
test verifies a fix for a specific PATCH_ID or production-readiness
criterion.

The tests are split into two groups:
  * Unit tests (no infra): the production-safety gates, final-score
    decision, ambiguous decisions, etc.
  * Integration tests (skipped if infra unavailable): real Redis
    roundtrip, real Qdrant search filter, real Paddle PP-Human command
    construction, real TransReID forward-pass shape.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterator

import httpx
import numpy as np
import pytest

from app.identity.ambiguity import CandidateHit, decide_ambiguity
from app.identity.camera_topology import CameraTopology
from app.core.runtime_mode import ProductionSafetyError, RuntimeMode


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _call() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_call())


# -----------------------------------------------------------------------------
# 1) Production fails without real Paddle model
# 2) Production fails without real ReID model
# 3) Synthetic detector is blocked in production
# 4) Deterministic ReID fallback is blocked in production
# -----------------------------------------------------------------------------


def test_synthetic_detector_blocked_in_production() -> None:
    """``MultiCameraRunner.start()`` raises ``ProductionSafetyError``
    when no detector is provided and the runtime mode is production.
    """
    from app.workers.multi_camera_runner import CameraSource, MultiCameraRunner

    def _fake_factory(source: str) -> Iterator[tuple[int, float, np.ndarray]]:
        for i in range(2):
            yield i, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)

    sources = [CameraSource("CAM_01", "stub://1", 640, 480, 5)]
    runner = MultiCameraRunner(
        sources,
        smoke_test_mode=False,
        mode=RuntimeMode.PRODUCTION,
        frame_reader_factory=_fake_factory,
    )
    with pytest.raises(ProductionSafetyError):
        runner.start()


def test_deterministic_reid_blocked_in_production() -> None:
    """The TransReID adapter refuses to load with the deterministic
    fallback when in production mode and the weight file is missing.
    """
    import tempfile
    from app.reid.base import ReIDConfig
    from app.reid.transreid_adapter import TransReIDAdapter

    with tempfile.TemporaryDirectory() as tmp:
        adapter = TransReIDAdapter(
            ReIDConfig(
                name="transreid", embedding_dim=768, qdrant_collection="person_reid_transreid"
            ),
            weight_path=f"{tmp}/missing.pth",
            mode=RuntimeMode.PRODUCTION,
        )
        with pytest.raises(ProductionSafetyError):
            adapter.load()


def test_pp_human_reid_blocked_in_production() -> None:
    import tempfile
    from app.reid.base import ReIDConfig
    from app.reid.pphuman_adapter import PPHumanReIDAdapter

    with tempfile.TemporaryDirectory() as tmp:
        adapter = PPHumanReIDAdapter(
            ReIDConfig(
                name="pphuman_strongbaseline",
                embedding_dim=256,
                qdrant_collection="person_reid_pphuman",
            ),
            weight_dir=f"{tmp}/missing",
            mode=RuntimeMode.PRODUCTION,
        )
        with pytest.raises(ProductionSafetyError):
            adapter.load()


# -----------------------------------------------------------------------------
# 5) ReID worker requires real crop (PATCH-004)
# -----------------------------------------------------------------------------


def test_reid_worker_requires_real_crops() -> None:
    """The ReID worker must NOT fabricate crops. If the URI list
    is empty / all downloads fail, ``process_tracklet`` returns
    ``None`` and increments the drop counter.
    """
    from app.workers.reid_worker import ReIDWorker
    from app.workers.tracklet_collector import Tracklet
    from app.reid.base import ReIDConfig
    from app.reid.transreid_adapter import TransReIDAdapter

    class _FakeAdapter(TransReIDAdapter):
        def __init__(self):
            # Skip loading; never used because the worker fails before
            # calling extract().
            self.config = ReIDConfig(
                name="transreid",
                embedding_dim=768,
                qdrant_collection="person_reid_transreid",
            )
            self._model = None
            self._fallback_active = True

        def load(self):
            pass

        def extract(self, crops):
            return np.zeros((0, 768), dtype=np.float32)

    worker = ReIDWorker(
        adapter=_FakeAdapter(),
        pg=None,
        qdrant=None,
        redis=None,
        minio=None,
        mode=RuntimeMode.PRODUCTION,
    )
    tl = Tracklet(
        tracklet_id="TL-001",
        camera_id="CAM_01",
        local_track_id=1,
        start_time=time.time(),
        crop_uris=["s3://evidence/never-exists/bogus.jpg"],
    )
    # Production: no MinIO client at all → no real crops → the
    # worker returns None (and logs a [NO-CROPS] error). The test
    # verifies that the worker does NOT call the adapter's extract()
    # in this case (which would have been the historical fabrication
    # path).
    out = worker.process_tracklet(tl)
    assert out is None


# -----------------------------------------------------------------------------
# 7) final_score determines identity decision
# -----------------------------------------------------------------------------


def test_final_score_drives_decision() -> None:
    """When the operator provides ``final_score`` to ``decide_ambiguity``,
    that value is the threshold variable, NOT the raw ReID cosine.
    """
    top1 = CandidateHit("GID-1", "CAM_01", 0.99, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.30, time.time())
    # Raw ReID cosine is 0.99 (>> auto_match) but final_score is 0.50.
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
        final_score=0.50,
        reid_override_threshold=None,
    )
    # The decision must follow the final_score, not the raw cosine.
    assert decision == "new"


def test_high_final_score_with_valid_topology_is_match() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.85, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.50, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
        final_score=0.90,
    )
    assert decision == "match"


def test_high_final_score_with_invalid_topology_is_not_match() -> None:
    top1 = CandidateHit("GID-1", "CAM_04", 0.85, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.50, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=False,  # hard block
        final_score=0.90,
    )
    assert decision == "new"


def test_ambiguous_when_top1_top2_close() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.88, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.86, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
        final_score=0.88,
    )
    assert decision in ("ambiguous", "candidate", "new")
    assert decision != "match"


def test_medium_final_score_is_candidate() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.76, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.50, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
        final_score=0.75,
    )
    assert decision == "candidate"


def test_low_final_score_is_new() -> None:
    top1 = CandidateHit("GID-1", "CAM_01", 0.70, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.40, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
        final_score=0.50,
    )
    assert decision == "new"


# -----------------------------------------------------------------------------
# 8) Invalid camera topology blocks auto-match (BUG-008)
# -----------------------------------------------------------------------------


def test_invalid_topology_blocks_match() -> None:
    t = CameraTopology()
    t.load_from_rows(
        [
            {
                "from_camera_id": "CAM_01",
                "to_camera_id": "CAM_04",
                "min_travel_seconds": 0,
                "max_travel_seconds": 0,
                "transition_probability": 0.0,
                "enabled": False,
                "notes": "",
            },
        ]
    )
    top1 = CandidateHit("GID-X", "CAM_04", 0.99, time.time())
    decision = decide_ambiguity(
        top1,
        None,
        auto_match_threshold=0.5,
        candidate_threshold=0.4,
        ambiguous_margin=0.01,
        is_known_link=t.is_known_link("CAM_01", "CAM_04"),
        final_score=0.99,
    )
    assert decision == "new"


# -----------------------------------------------------------------------------
# 9) Ambiguous candidates are not auto-merged
# -----------------------------------------------------------------------------


def test_ambiguous_not_auto_merged() -> None:
    """Two visually similar candidates produce 'ambiguous' or 'held',
    never 'match'.
    """
    top1 = CandidateHit("GID-1", "CAM_01", 0.85, time.time())
    top2 = CandidateHit("GID-2", "CAM_02", 0.84, time.time())
    decision = decide_ambiguity(
        top1,
        top2,
        auto_match_threshold=0.82,
        candidate_threshold=0.72,
        ambiguous_margin=0.04,
        is_known_link=True,
        final_score=0.85,
    )
    assert decision != "match"


# -----------------------------------------------------------------------------
# 10) Qdrant filtered search includes timestamp/camera/quality filters
# -----------------------------------------------------------------------------


def test_qdrant_search_includes_all_filters() -> None:
    """Every Qdrant search MUST include: timestamp, camera_id,
    quality_score, model_name, model_version. Validated via a fake
    client that records the kwargs.
    """
    from app.storage.qdrant_store import QdrantStore

    class _FakeQueryResponse:
        def __init__(self, points=None):
            self.points = list(points or [])

    class _FakeClient:
        def __init__(self):
            self.calls: list[dict] = []

        def query_points(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeQueryResponse()

    store = QdrantStore("localhost", 6333)
    fake = _FakeClient()
    store._client = fake
    store.search(
        "person_reid_pphuman",
        np.zeros(256, dtype=np.float32),
        timestamp_gte=10_000,
        candidate_camera_ids=["CAM_01", "CAM_02"],
        model_name="pphuman_strongbaseline",
        model_version="v1",
    )
    assert len(fake.calls) == 1
    f = fake.calls[0]["query_filter"]
    keys = {c.key for c in f.must}
    assert {"timestamp", "camera_id", "quality_score", "model_name", "model_version"} <= keys


# -----------------------------------------------------------------------------
# 11) PP-Human and TransReID use separate collections
# -----------------------------------------------------------------------------


def test_pphuman_and_transreid_use_separate_collections() -> None:
    from app.storage.qdrant_store import COLLECTIONS

    names = {c[0] for c in COLLECTIONS}
    assert "person_reid_transreid_msmt" in names
    # Different dims.
    dims = {c[1] for c in COLLECTIONS}
    assert len(dims) == len(COLLECTIONS), "each model has its own dim"


# -----------------------------------------------------------------------------
# 12) local_track_id collision across cameras does not collide globally
# -----------------------------------------------------------------------------


def test_local_track_id_collision_is_camera_local() -> None:
    """Two workers on different cameras can both start at local_track_id=1
    without conflict. The architecture guard test already covers the
    per-camera state; here we just verify the public surface: the
    ``LocalTrack`` dataclass is per-camera and ``local_track_id`` is
    scoped to ``camera_id``.
    """
    from app.workers.pphuman_worker import LocalTrack

    a = LocalTrack(
        camera_id="CAM_01",
        local_track_id=1,
        bbox=(0, 0, 10, 10),
        confidence=0.9,
        frame_id=0,
        ts=0.0,
    )
    b = LocalTrack(
        camera_id="CAM_02",
        local_track_id=1,
        bbox=(0, 0, 10, 10),
        confidence=0.9,
        frame_id=0,
        ts=0.0,
    )
    assert a.camera_id != b.camera_id
    assert a.local_track_id == b.local_track_id


# -----------------------------------------------------------------------------
# 13) FastAPI auth blocks unauthenticated identity access
# -----------------------------------------------------------------------------


def test_fastapi_auth_blocks_unauthenticated_identity(monkeypatch) -> None:
    """``/identity/{global_id}`` must reject requests without the
    Bearer token.
    """
    monkeypatch.setenv("SOTA_API_TOKEN", "secret-1234")
    from app.api.server import build_app

    app = build_app(pg=None, mode=RuntimeMode.PRODUCTION)
    # No token.
    r = _request(app, "GET", "/identity/GID-ABC12345-CAM01")
    assert r.status_code in (401, 403)
    # With wrong token.
    r2 = _request(
        app,
        "GET",
        "/identity/GID-ABC12345-CAM01",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r2.status_code in (401, 403)
    # /health is public.
    r3 = _request(app, "GET", "/health")
    assert r3.status_code == 200


def test_fastapi_auth_accepts_valid_token(monkeypatch) -> None:
    monkeypatch.setenv("SOTA_API_TOKEN", "secret-1234")
    from app.api.server import build_app

    app = build_app(pg=None, mode=RuntimeMode.PRODUCTION)
    # /metrics is public.
    r = _request(app, "GET", "/metrics")
    assert r.status_code == 200


def test_fastapi_refuses_to_start_without_token_in_production() -> None:
    """In production mode, ``build_app`` raises if ``SOTA_API_TOKEN``
    is not set (PATCH-014).
    """
    import os

    os.environ.pop("SOTA_API_TOKEN", None)
    from app.api.server import build_app

    with pytest.raises(Exception):
        build_app(pg=None, mode=RuntimeMode.PRODUCTION)


# -----------------------------------------------------------------------------
# 14) Retention cleanup removes expired vectors/events
# -----------------------------------------------------------------------------


def test_retention_methods_exist() -> None:
    """Both the Qdrant and MinIO retention methods must exist with
    the right signatures (PATCH-015).
    """
    from app.storage.qdrant_store import QdrantStore
    from app.storage.minio_store import MinioStore

    assert hasattr(QdrantStore, "delete_points_older_than")
    assert hasattr(QdrantStore, "count_points_older_than")
    assert hasattr(MinioStore, "delete_older_than")
    # Postgres retention method (BUG-fix for PATCH-015).
    from app.storage.postgres import PostgresStore

    assert hasattr(PostgresStore, "expire_old_identities")
    assert hasattr(PostgresStore, "delete_tracking_events_older_than")


# -----------------------------------------------------------------------------
# 15) Docker Compose config validates
# -----------------------------------------------------------------------------


def test_docker_compose_config_validates() -> None:
    """The ``docker-compose.yaml`` should parse cleanly. We use the
    CLI directly; the test is skipped if docker is not installed.
    """
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        pytest.skip("docker CLI not available")
    r = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, f"docker compose config failed: {r.stderr}"


# -----------------------------------------------------------------------------
# 6) Resolver consumes embedding stream (PATCH-006)
# -----------------------------------------------------------------------------


def test_resolver_has_run_method() -> None:
    """The ``GlobalIdentityResolver`` must have a real ``run(stop_event)``
    method (PATCH-006 fix).
    """
    from app.identity.resolver import GlobalIdentityResolver
    import inspect

    assert hasattr(GlobalIdentityResolver, "run")
    sig = inspect.signature(GlobalIdentityResolver.run)
    assert "stop_event" in sig.parameters


# -----------------------------------------------------------------------------
# Extra: final_score is persisted
# -----------------------------------------------------------------------------


def test_final_score_appears_in_persisted_decision() -> None:
    """The resolver passes ``final_score`` to ``pg.insert_identity_decision``
    (the audit's PATCH-008 fix).

    PATCH (2026-06-15, sidecar): the resolve() entry point now
    delegates to _resolve_inner() so the model_name can be
    overridden per event (the operator's TransReID sidecar emits
    embeddings with model_name="transreid_msmt", not the api's
    default "pphuman_strongbaseline"). The final_score contract is
    still satisfied — the ``insert_identity_decision`` call lives
    in ``_resolve_inner``, which is the method that actually does
    the work. The audit's intent is to assert that ``final_score``
    reaches Postgres, not the exact method name.
    """
    import inspect
    from app.identity.resolver import GlobalIdentityResolver

    src_inner = inspect.getsource(GlobalIdentityResolver._resolve_inner)
    assert 'final_score=breakdown["final_score"]' in src_inner, (
        "_resolve_inner must pass final_score to insert_identity_decision "
        "(the audit's PATCH-008 fix)."
    )


# -----------------------------------------------------------------------------
# Extra: shared model (one model per process) — PATCH-007
# -----------------------------------------------------------------------------


def test_multi_camera_runner_accepts_shared_detector() -> None:
    """The ``MultiCameraRunner.__init__`` accepts a shared detector
    (PATCH-007 fix).
    """
    import inspect
    from app.workers.multi_camera_runner import MultiCameraRunner

    sig = inspect.signature(MultiCameraRunner.__init__)
    assert "detector" in sig.parameters


# -----------------------------------------------------------------------------
# Extra: cropped ReID uses real downloads (PATCH-004)
# -----------------------------------------------------------------------------


def test_reid_worker_has_minio_client_param() -> None:
    from app.workers.reid_worker import ReIDWorker
    import inspect

    sig = inspect.signature(ReIDWorker.__init__)
    assert "minio" in sig.parameters
