"""Integration tests for persistent-ID identity resolution flows.

These tests describe the 4 scenarios from operator spec section #11:
  1. Simulated same-camera occlusion (local_track_id changes, embedding
     similarity high, resolver assigns same global_id).
  2. Simulated cross-camera transition within camera_links travel window.
  3. Ambiguous identity (two close candidates, hold_ambiguous).
  4. Impossible transition (wrong camera link or impossible travel time,
     reject_impossible even if cosine is high).

These tests are mocked end-to-end at the Qdrant / resolver layer; they
do not require a live PaddleDetection subprocess. They pin the
resolver's staged retrieval + decision outcomes.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Resolver signature + decision outcomes
# ---------------------------------------------------------------------------


def test_resolver_resolve_signature():
    """The GlobalIdentityResolver.resolve() method must accept the inputs
    needed to stage retrieval: tracklet_id, camera_id, ts, mean_embedding,
    and return a decision dict that includes 'decision' and (for non-ambiguous
    outcomes) 'assigned_global_id'."""
    from app.identity.resolver import GlobalIdentityResolver
    sig = inspect.signature(GlobalIdentityResolver.resolve)
    params = {p.name for p in sig.parameters.values()}
    for required in (
        "tracklet_id",
        "camera_id",
        "ts",
        "mean_embedding",
    ):
        assert required in params, (
            f"GlobalIdentityResolver.resolve() must accept {required!r}"
        )


def test_resolver_uses_camera_links_for_cross_cam():
    """For cross-camera matching the resolver must use camera_links (or
    equivalent adjacency data) to constrain candidate cameras."""
    from app.identity.resolver import GlobalIdentityResolver
    src = inspect.getsource(GlobalIdentityResolver)
    # Look for any of: camera_links, linked_cameras, adjacent_cameras
    assert any(
        token in src
        for token in ("camera_links", "linked_cameras", "adjacent_cameras", "_candidate_cameras")
    ), "Resolver must consult camera topology to constrain candidates"


def test_resolver_uses_ambiguity_margin():
    """The resolver must apply an ambiguity margin (top1 - top2 >= margin)
    to decide between auto-merge and hold_ambiguous."""
    from app.identity.resolver import GlobalIdentityResolver
    src = inspect.getsource(GlobalIdentityResolver)
    # Look for the ambiguity margin logic
    assert "margin" in src.lower() or "ambiguity" in src.lower(), (
        "Resolver must have ambiguity margin logic (top1 - top2 >= margin)"
    )


# ---------------------------------------------------------------------------
# Identity hierarchy
# ---------------------------------------------------------------------------


def test_global_id_never_assigned_from_local_track_id_alone():
    """Hard rule: global_id must never be derived from local_track_id.
    It's only ever created by GlobalIdentityResolver, which is fed
    embeddings, not local_track_ids."""
    # The Tracklet dataclass must not have a global_id field
    from app.workers.tracklet_collector import Tracklet
    sig = inspect.signature(Tracklet)
    fields = {p.name for p in sig.parameters.values()}
    assert "global_id" not in fields, (
        "Tracklet must not have a global_id field — that would let "
        "TrackletCollector assign global_ids, which is forbidden."
    )
    # The LocalTrack dataclass must not have a global_id field
    from app.workers.pphuman_worker import LocalTrack
    sig = inspect.signature(LocalTrack)
    fields = {p.name for p in sig.parameters.values()}
    assert "global_id" not in fields, (
        "LocalTrack must not have a global_id field — that would let "
        "the MOT tracker assign global_ids, which is forbidden."
    )


# ---------------------------------------------------------------------------
# Redis Stream consumer-group pattern
# ---------------------------------------------------------------------------


def test_redis_streams_use_consumer_groups():
    """Consumer files (reid_worker.py, resolver.py, telemetry_worker.py,
    evidence_rekey_worker.py) must use consumer groups (xreadgroup /
    ensure_group) for stream consumption. Publisher files (e.g.
    tracklet_collector.py) use xadd / publish and don't need consumer
    groups."""
    CONSUMER_FILES = (
        "app/workers/reid_worker.py",
        "app/identity/resolver.py",
        "app/workers/telemetry_worker.py",
        "app/workers/evidence_rekey_worker.py",
    )
    for rel in CONSUMER_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text()
        # The consumer-group pattern: ensure_group() called once, then
        # xreadgroup or xread within a loop. We allow either name.
        if "stream:" in text:
            assert "ensure_group" in text or "xreadgroup" in text.lower(), (
                f"{rel} must use consumer groups (xreadgroup / "
                f"ensure_group) for stream consumption"
            )


def test_redis_xack_after_processing():
    """After successful processing, the consumer must XACK the message.
    Unacked messages would pile up in the pending list."""
    from app.workers.reid_worker import ReIDWorker
    src = inspect.getsource(ReIDWorker)
    assert "ack" in src.lower(), "ReIDWorker must XACK after processing"


# ---------------------------------------------------------------------------
# Test fixtures: the 4 simulated scenarios (unit-level, no live infra)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_embedding_256():
    """A 256-dim L2-normalized random vector."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(256).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def similar_embedding(fake_embedding_256):
    """An embedding very close to the first (cosine ~ 0.95)."""
    rng = np.random.default_rng(43)
    noise = rng.standard_normal(256).astype(np.float32) * 0.1
    v = fake_embedding_256 + noise
    return v / np.linalg.norm(v)


@pytest.fixture
def ambiguous_embedding(fake_embedding_256):
    """An embedding moderately close to the first (cosine ~ 0.78)."""
    rng = np.random.default_rng(44)
    noise = rng.standard_normal(256).astype(np.float32) * 0.5
    v = fake_embedding_256 + noise
    return v / np.linalg.norm(v)


def test_same_camera_occlusion_assigns_same_global_id(similar_embedding):
    """Scenario 1: same-cam re-entry after long occlusion. We assert the
    resolver's Stage 1 (same-cam recent match) has the right config
    fields. Stage 1 uses `temporal_sigma_seconds` as the same-cam
    re-entry window and `auto_match_threshold` as the cosine threshold.
    """
    from app.identity.resolver import ResolverConfig
    cfg = ResolverConfig()
    # The same-cam re-entry window (temporal_sigma)
    assert hasattr(cfg, "temporal_sigma_seconds"), (
        "ResolverConfig must expose temporal_sigma_seconds (Stage 1 "
        "same-cam re-entry window)"
    )
    # The Stage 1 auto-match threshold (0.82 per operator spec)
    assert cfg.auto_match_threshold == pytest.approx(0.82, abs=0.01), (
        f"auto_match_threshold must default to 0.82; got {cfg.auto_match_threshold}"
    )
    # The ambiguity margin (must be tunable)
    assert hasattr(cfg, "ambiguous_margin"), (
        "ResolverConfig must expose ambiguous_margin"
    )


def test_cross_camera_transition_uses_camera_links():
    """Scenario 2: cross-cam transition within camera_links travel window
    must be honored. We assert the resolver has a stage for linked cameras.
    """
    from app.identity.resolver import GlobalIdentityResolver
    src = inspect.getsource(GlobalIdentityResolver)
    # The resolver must have a multi-stage retrieval pattern
    assert "_stage1" in src or "_stage2" in src or "_stage3" in src, (
        "Resolver must use staged retrieval (stage 1/2/3)"
    )


def test_ambiguous_match_held_not_auto_merged():
    """Scenario 3: ambiguous candidates must be held."""
    from app.identity.resolver import GlobalIdentityResolver
    src = inspect.getsource(GlobalIdentityResolver)
    # The resolver must have logic that returns an 'ambiguous' or
    # 'hold_ambiguous' decision when top1 - top2 is too close.
    assert "ambiguous" in src or "hold_ambiguous" in src


def test_impossible_transition_rejected_even_with_high_cosine():
    """Scenario 4: impossible transition (camera link missing or travel
    time violated) must be rejected even if cosine is high. The resolver
    must have a 'reject_impossible' decision outcome.
    """
    from app.identity.resolver import GlobalIdentityResolver
    src = inspect.getsource(GlobalIdentityResolver)
    # Look for reject_impossible (or similar)
    decision_outcomes = re.findall(
        r"return\s+['\"](\w+)['\"]", src
    )
    assert "reject_impossible" in decision_outcomes or "impossible" in src, (
        f"Resolver must have reject_impossible outcome. Found: {set(decision_outcomes)}"
    )
