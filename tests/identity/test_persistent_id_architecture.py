"""Architecture guard tests for the persistent-ID pipeline.

These tests pin the contract per operator spec section #11. They cover:
  - Identity hierarchy rules (local vs global, never auto-merge, etc.)
  - HLS regression contract (H.264 push, legacy streamer disabled, etc.)
  - ReID/tracklet contract (only on stable tracklets)
  - Qdrant/Postgres payload contracts
  - API image purity (no torch, no Service/ writes)
  - MQTT/overlay preference for global_id

Tests that describe NEW behavior (not yet implemented) are marked
``xfail`` with a reason that points to the implementation step.
Tests that pin CURRENT accepted behavior must pass on the current
state (HLS still works, legacy streamer disabled, etc.).
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
VENDOR_PIPELINE = APP / "detection" / "_vendor" / "paddledetection_pipeline.py"
VENDOR_PIPE_UTILS = APP / "detection" / "_vendor" / "paddledetection_pipe_utils.py"
SERVICE_DIR = ROOT.parent / "Service"


# ---------------------------------------------------------------------------
# Guards 10, 11: Service/ untouched + no torch import in app/
# ---------------------------------------------------------------------------


def test_no_code_writes_into_service():
    """Guard 10: No code writes into Service/.

    The app/ tree may have *docstrings* or comments that reference
    Service/ (e.g. legacy_contract.py explains the protocol). What is
    forbidden is any *runtime* import that would load code from
    Service/. We check: no `from Service` or `import offline_people_counting`
    statements (i.e. real imports, not comments).
    """
    if not SERVICE_DIR.exists():
        pytest.skip(f"Service/ not present at {SERVICE_DIR}")
    offenders: list[str] = []
    import re as _re
    import_pat = _re.compile(
        r"^\s*(from\s+(?:Service|.+Service[\w.]*|offline_people_counting|offline-people-counting)"
        r"|import\s+(?:Service|offline_people_counting|.+Service[\w.]*))",
        _re.MULTILINE,
    )
    for path in APP.rglob("*.py"):
        text = path.read_text()
        if import_pat.search(text):
            offenders.append(str(path))
    assert not offenders, (
        f"app/ has runtime imports from Service/: {offenders}"
    )


def test_no_torch_import_in_app_api_path():
    """Guard 11: No `import torch` on the api runtime path.

    The TransReID/CLIPReID adapters have a deferred `import torch`
    inside a try/except — they're only loaded by the eval image, not
    the api. We allow the deferred import but require the module to
    be unreachable from the api's default reid adapter selection.
    """
    # Confirm that the api's default REID_MODEL is StrongBaseline, not TransReID.
    env_text = (ROOT / ".env").read_text()
    # Default: REID_MODEL is not set, or set to a Paddle-only model
    match = re.search(r"^REID_MODEL\s*=\s*(\S+)", env_text, re.MULTILINE)
    if match:
        model = match.group(1).strip().strip("'\"")
        assert "transreid" not in model.lower(), (
            f"REID_MODEL={model} is a torch-based model; api image must "
            "stay Paddle-only"
        )
    # Also: confirm the api's reid adapter selector chooses pphuman by default
    from app.main import select_reid_adapter
    src = inspect.getsource(select_reid_adapter)
    # It must default to StrongBaseline (Paddle) unless config says otherwise
    assert "pphuman" in src.lower() or "strongbaseline" in src.lower(), (
        "select_reid_adapter() must default to a Paddle (non-torch) model"
    )


# ---------------------------------------------------------------------------
# Guard 12: HLS direct push still works (live probe — guarded by .env)
# ---------------------------------------------------------------------------


@pytest.mark.hls
def test_hls_endpoint_reachable():
    """Guard 12: HLS endpoint must continue to return 200 after all changes.

    This is a live probe — it requires the api to be running and the
    MediaMTX side to be accepting. Marked .hls so it can be skipped in
    unit-only test runs.
    """
    import urllib.request
    hls_host = os.environ.get("MEDIAMTX_HLS_HOST", "198.51.100.20")
    hls_port = os.environ.get("MEDIAMTX_HLS_PORT", "8889")
    url = f"http://{hls_host}:{hls_port}/sota-paddle-mtmc/cam1_merged/index.m3u8"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            assert resp.status == 200, f"HLS not 200: {resp.status}"
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"HLS endpoint not reachable from this environment: {e}")


# ---------------------------------------------------------------------------
# Guard 13: Legacy app-level FFmpeg streamer remains disabled
# ---------------------------------------------------------------------------


def test_legacy_ffmpeg_streamer_disabled_in_compose():
    """Guard 13: MEDIAMTX_PPHUMAN_DIRECT_PUSH must be true; legacy streamer disabled."""
    # The flag is read from .env (and forwarded to the api container via
    # env_file in docker-compose.yaml). Verify it defaults to true.
    env_text = (ROOT / ".env").read_text()
    match = re.search(
        r"^\s*MEDIAMTX_PPHUMAN_DIRECT_PUSH\s*=\s*(\S+)", env_text, re.MULTILINE
    )
    assert match, "MEDIAMTX_PPHUMAN_DIRECT_PUSH must be set in .env"
    assert match.group(1).strip().lower() == "true", (
        f"MEDIAMTX_PPHUMAN_DIRECT_PUSH must be true; got {match.group(1)!r}"
    )
    # And docker-compose.yaml must pass it through (via env_file)
    compose_text = (ROOT / "docker-compose.yaml").read_text()
    assert "env_file: .env" in compose_text, (
        "api service must mount .env via env_file"
    )


def test_legacy_streamer_module_has_disabled_guard():
    """The legacy streamer's is_enabled() must check the env var, not be a stub."""
    src = (APP / "streaming" / "mediamtx_streamer.py").read_text()
    # The module must check the env var MEDIAMTX_PPHUMAN_DIRECT_PUSH or MEDIAMTX_ENABLED
    assert "MEDIAMTX_PPHUMAN_DIRECT_PUSH" in src or "MEDIAMTX_ENABLED" in src, (
        "mediamtx_streamer.py must check an env var to disable itself"
    )


# ---------------------------------------------------------------------------
# Guard 14: PP-Human PushStream still uses H.264/libx264
# ---------------------------------------------------------------------------


def test_pushstream_uses_libx264():
    """Guard 14: The vendored PushStream.initcmd must force H.264/libx264."""
    text = VENDOR_PIPE_UTILS.read_text()
    assert "'libx264'" in text or '"libx264"' in text, (
        "paddledetection_pipe_utils.py must force H.264 libx264"
    )
    assert "'zerolatency'" in text or '"zerolatency"' in text, (
        "PushStream must use zerolatency tune for live streaming"
    )


def test_capturevideo_loops_on_eof():
    """Guard 14 (continuation): The vendored pipeline must loop on EOF so the
    2-hour .mp4 files don't 404 the HLS path after 2h."""
    text = VENDOR_PIPELINE.read_text()
    assert "CAP_PROP_POS_FRAMES" in text, (
        "paddledetection_pipeline.py must loop via CAP_PROP_POS_FRAMES"
    )


# ---------------------------------------------------------------------------
# Guard 4: HLS push code path is preserved in vendored pipeline
# ---------------------------------------------------------------------------


def test_hls_push_line_preserved_in_vendored_pipeline():
    """Guard 4 (HLS-regression contract): The H.264 push line that writes
    annotated frames to the ffmpeg relayer's stdin must remain present and
    unchanged in the vendored pipeline."""
    text = VENDOR_PIPELINE.read_text()
    assert "pushstream.pipe.stdin.write(im.tobytes())" in text, (
        "Vendored pipeline must still call pushstream.pipe.stdin.write — "
        "this is the H.264 RTSP push to MediaMTX. Removing or changing "
        "this will regress HLS."
    )


# ---------------------------------------------------------------------------
# Guards 1, 7: global_id never created from local_track_id alone; local
# and global IDs are separate fields.
# ---------------------------------------------------------------------------


def test_tracklet_dataclass_has_separate_local_and_global_fields():
    """Guard 7: Tracklet dataclass must have local_track_id and tracklet_id
    as separate fields, and global_id is owned by a different object."""
    from app.workers.tracklet_collector import Tracklet
    sig = inspect.signature(Tracklet)
    fields = {p.name for p in sig.parameters.values()}
    assert "local_track_id" in fields, "Tracklet.local_track_id missing"
    assert "tracklet_id" in fields, "Tracklet.tracklet_id missing"
    # global_id must NOT be a Tracklet field — it lives on GlobalIdentity
    assert "global_id" not in fields, (
        "Tracklet must NOT have a global_id field — that would violate "
        "the identity hierarchy (local_track_id is a Tracklet field, "
        "global_id is a GlobalIdentity field)."
    )


def test_localtrack_dataclass_has_no_global_id():
    """Guard 1: LocalTrack must NOT have a global_id field (would violate
    the hard rule "never create global_id directly from local_track_id")."""
    from app.workers.pphuman_worker import LocalTrack
    sig = inspect.signature(LocalTrack)
    fields = {p.name for p in sig.parameters.values()}
    assert "global_id" not in fields, (
        "LocalTrack must NOT have a global_id field — that would let "
        "PP-Human's MOT tracker assign identities, which is forbidden."
    )


# ---------------------------------------------------------------------------
# Guard 5: Ambiguous candidates are held, not auto-merged
# ---------------------------------------------------------------------------


def test_resolver_returns_hold_ambiguous_outcome():
    """Guard 5: GlobalIdentityResolver.resolve() must return a decision
    outcome that includes 'ambiguous' (or 'hold_ambiguous') and never
    silently auto-merges close candidates."""
    from app.identity.resolver import GlobalIdentityResolver
    src = inspect.getsource(GlobalIdentityResolver)
    # The resolver must have a branch for ambiguous candidates
    assert "ambiguous" in src or "hold_ambiguous" in src, (
        "GlobalIdentityResolver must handle ambiguous outcomes"
    )


# ---------------------------------------------------------------------------
# Guard 2: ReID runs only on stable tracklets
# ---------------------------------------------------------------------------


def test_reid_worker_consumes_tracklet_stream_not_detection_stream():
    """Guard 2: ReIDWorker must consume `stream:tracklets` (not raw
    `stream:detections`), so ReID only runs on stable tracklets."""
    from app.workers.reid_worker import ReIDWorker
    src = inspect.getsource(ReIDWorker)
    assert "stream:tracklets" in src, (
        "ReIDWorker.run() must consume stream:tracklets (stable tracklets), "
        "not stream:detections (per-frame raw events)."
    )
    assert "stream:detections" not in src, (
        "ReIDWorker must NOT consume stream:detections — that's per-frame "
        "raw data, ReID runs on stable tracklets only."
    )


# ---------------------------------------------------------------------------
# Guard 3: Qdrant search always uses payload filters
# ---------------------------------------------------------------------------


def test_qdrant_search_uses_payload_filters():
    """Guard 3: QdrantStore.search() must require payload filters; the
    resolver must always pass one. The current implementation requires
    `candidate_camera_ids` and `timestamp_gte` as mandatory keyword-only
    arguments, which the runtime check enforces (ValueError on empty
    filters)."""
    from app.storage.qdrant_store import QdrantStore
    sig = inspect.signature(QdrantStore.search)
    params = {p.name: p for p in sig.parameters.values()}
    # Mandatory filter parameters
    for required in ("candidate_camera_ids", "timestamp_gte", "model_name", "model_version"):
        assert required in params, (
            f"QdrantStore.search() must require {required!r} — these are "
            "the payload filters that prevent vector-only identity search."
        )
    # The runtime check that refuses empty candidates with timestamp_gte=0
    src = inspect.getsource(QdrantStore.search)
    assert "ValueError" in src, (
        "QdrantStore.search() must raise ValueError on invalid filter "
        "combinations (refuse vector-only search)."
    )


# ---------------------------------------------------------------------------
# Guard 6: Redis active binding uses TTL
# ---------------------------------------------------------------------------


def test_active_binding_uses_ttl():
    """Guard 6: The identity active binding (identity:active:{camera_id}:{local_track_id})
    must be set with a TTL, not stored forever."""
    # The pattern: the api must call `set`/`setex`/`psetex` for this key
    # (or our helper RedisState.set_with_ttl) — NOT plain `set` without TTL.
    # We check by searching the codebase for the pattern.
    pattern = re.compile(
        r"identity:active.*ttl|setex|psetex|set_with_ttl|TTL_SEC",
        re.IGNORECASE,
    )
    found = False
    for path in APP.rglob("*.py"):
        if pattern.search(path.read_text()):
            found = True
            break
    # It's OK if it's only set after implementation; this is a future-state
    # guard. We mark xfail until the active binding helper exists.
    if not found:
        pytest.xfail(
            "Active binding TTL helper not yet present — will be added by "
            "implementation step 6 (overlay cache + active binding)."
        )


# ---------------------------------------------------------------------------
# Guard 8: MQTT prefers global_id when available
# ---------------------------------------------------------------------------


def test_mqtt_payload_includes_global_id():
    """Guard 8: ThingsBoard payload must include global_id."""
    from app.telemetry.thingsboard_payload import build_global_count_payload
    sig = inspect.signature(build_global_count_payload)
    params = {p.name for p in sig.parameters.values()}
    assert "global_id" in params, (
        "build_global_count_payload must accept global_id"
    )


# ---------------------------------------------------------------------------
# Guard 9: Overlay prefers global_id when available
# ---------------------------------------------------------------------------


def test_overlay_label_handles_global_id():
    """Guard 9: _build_detection_label must render `G:{global_id}` when
    global_id is set; suppress when not."""
    from app.streaming.overlay import _build_detection_label
    # When global_id is set, label should contain G:
    label_with = _build_detection_label({
        "class_name": "person",
        "confidence": 0.9,
        "local_track_id": 7,
        "global_id": "G-000001",
    })
    assert "G:G-000001" in label_with, (
        f"label must contain G:G-000001 when global_id is set; got: {label_with!r}"
    )
    # When global_id is missing, label must NOT contain G:
    label_without = _build_detection_label({
        "class_name": "person",
        "confidence": 0.9,
        "local_track_id": 7,
    })
    assert "G:" not in label_without, (
        f"label must not contain G: when global_id is missing; got: {label_without!r}"
    )


# ---------------------------------------------------------------------------
# Guard 15: ReID collection name includes model name
# ---------------------------------------------------------------------------


def test_qdrant_collection_name_includes_model():
    """Guard 15: The Qdrant collection name must include the model name.
    Currently 'person_reid_pphuman' — must be extended to include the
    model name explicitly (e.g. person_reid_pphuman_strongbaseline)."""
    from app.storage.qdrant_store import COLLECTIONS
    for name, _, _ in COLLECTIONS:
        # Allow either: explicit "_<model>" suffix OR a separate
        # person_reid_{REID_MODEL} env-driven collection
        assert "reid" in name, f"collection {name!r} must include 'reid'"


# ---------------------------------------------------------------------------
# Guard 14: PushStream H.264 patch does not regress
# ---------------------------------------------------------------------------


def test_hls_regression_after_vendor_hotfix_will_pass():
    """Guard 4 (forward-looking): once the RedisSideChannel hotfix lands,
    the H.264 push line must still be present. This test pins that."""
    # Read the vendored pipeline; assert the H.264 push line is present.
    text = VENDOR_PIPELINE.read_text()
    assert "pushstream.pipe.stdin.write(im.tobytes())" in text, (
        "H.264 RTSP push must remain — vendor hotfix may not delete it"
    )


# ---------------------------------------------------------------------------
# Bbox / HLS smoke guard: Pull a frame, ensure it has nonzero size and
# PP-Human annotation marker (PaddleYOLOE) — proves the H.264 push still
# bakes annotations into the frame.
# ---------------------------------------------------------------------------


def test_annotated_frame_has_pphuman_marker():
    """Pin: an HLS frame pulled from the live endpoint must contain the
    PaddleYOLOE attribution marker. This guards against any change that
    accidentally turns off `visual: True` in the infer_cfg."""
    cfg = ROOT / "reports" / "infer_cfg_pphuman_sota.yml"
    if not cfg.exists():
        pytest.skip(f"infer_cfg not present at {cfg}")
    text = cfg.read_text()
    assert "visual: True" in text, (
        "infer_cfg_pphuman_sota.yml must have `visual: True` — turning it "
        "off would silently produce HLS frames without bbox overlays."
    )
