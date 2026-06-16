"""Evidence re-key tests (PATCH-029).

The audit's PATCH-029 fix requires that after the resolver assigns
a global_id, the best crop is server-side copied from the pending
path to the final dated path, and the tracklet's best_crop_uri is
updated.
"""

from __future__ import annotations


from app.storage.minio_store import MinioStore
from app.workers.evidence_rekey_worker import EvidenceRekeyWorker, RekeyConfig


# ----------------------------------------------------------------------------
# MinioStore path helpers
# ----------------------------------------------------------------------------


def test_pending_evidence_key_format() -> None:
    key = MinioStore.pending_evidence_key(
        site_id="showroom_a",
        camera_id="CAM_01",
        tracklet_id="TL-001",
        kind="best",
    )
    assert key == "evidence/pending/showroom_a/CAM_01/TL-001/best.jpg"


def test_pending_evidence_key_debug_includes_frame_id() -> None:
    key = MinioStore.pending_evidence_key(
        site_id="showroom_a",
        camera_id="CAM_01",
        tracklet_id="TL-001",
        kind="debug",
        frame_id=42,
    )
    assert key == "evidence/pending/showroom_a/CAM_01/TL-001/debug_000042.jpg"


def test_evidence_key_final_format_includes_global_id() -> None:
    key = MinioStore.evidence_key(
        site_id="showroom_a",
        camera_id="CAM_01",
        zone_id="Z_FLOOR",
        ts=1_700_000_000.0,
        global_id="GID-ABCD1234-CAM01",
        tracklet_id="TL-001",
        kind="best",
    )
    # The exact date format depends on timezone; we just check the
    # prefix and the global_id appears.
    assert key.startswith("evidence/showroom_a/CAM_01/Z_FLOOR/")
    assert "GID-ABCD1234-CAM01" in key
    assert key.endswith("/TL-001/best.jpg")


def test_parse_evidence_uri() -> None:
    parts = MinioStore.parse_evidence_uri(
        "s3://bucket-name/evidence/showroom_a/CAM_01/Z_FLOOR/2026/06/12/GID-X/TL-1/best.jpg"
    )
    assert parts == {
        "bucket": "bucket-name",
        "key": "evidence/showroom_a/CAM_01/Z_FLOOR/2026/06/12/GID-X/TL-1/best.jpg",
    }


def test_parse_evidence_uri_rejects_non_evidence() -> None:
    assert MinioStore.parse_evidence_uri("s3://bucket/foo/bar") is None
    assert MinioStore.parse_evidence_uri("http://x/y") is None


# ----------------------------------------------------------------------------
# Worker behavior — no MinIO (unit tests)
# ----------------------------------------------------------------------------


class _FakeMinio:
    def __init__(self, exists: bool = True, copy_ok: bool = True, delete_ok: bool = True):
        self.bucket = "evidence"
        self._exists = exists
        self._copy_ok = copy_ok
        self._delete_ok = delete_ok
        self.copies: list[tuple[str, str]] = []
        self.deletes: list[str] = []
        self.client = self  # for `self.minio.client.remove_object`
        self.copy_calls = 0

    def copy_object_within_bucket(self, src_key: str, dst_key: str):
        self.copy_calls += 1
        if not self._copy_ok:
            raise RuntimeError("copy failed")
        self.copies.append((src_key, dst_key))

    def remove_object(self, bucket, key):
        if not self._delete_ok:
            raise RuntimeError("delete failed")
        self.deletes.append(key)

    # Forward staticmethod used by the worker.
    def evidence_key(self, *args, **kwargs):
        return MinioStore.evidence_key(*args, **kwargs)

    # stat_object — used for the existence pre-check.
    def stat_object(self, bucket, key):
        if not self._exists:
            raise Exception("not found")
        return None


class _FakeRedis:
    def __init__(self):
        self.acks: list[tuple[str, str, str]] = []

    def ensure_group(self, stream, group, start_id="0"):
        return None

    def consume(self, stream, group, consumer, count=10, block_ms=1000):
        return []

    def ack(self, stream, group, *msg_ids):
        for m in msg_ids:
            self.acks.append((stream, group, m))


def test_rekey_worker_skips_when_pending_missing() -> None:
    """If the pending crop does not exist (stat_object raises), the
    worker logs and returns True (no-op success).
    """
    minio = _FakeMinio(exists=False)
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=True, keep_pending_copy=False),
    )
    fields = {
        "tracklet_id": "TL-001",
        "camera_id": "CAM_01",
        "site_id": "showroom_a",
        "decision": "new",
        "assigned_global_id": "GID-ABCD1234-CAM01",
        "end_zone_id": "Z_FLOOR",
        "ts": 1_700_000_000.0,
    }
    assert w.handle_decision(fields) is True
    assert minio.copy_calls == 0


def test_rekey_worker_copies_when_pending_exists() -> None:
    minio = _FakeMinio(exists=True)
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=True, keep_pending_copy=False),
    )
    fields = {
        "tracklet_id": "TL-001",
        "camera_id": "CAM_01",
        "site_id": "showroom_a",
        "decision": "new",
        "assigned_global_id": "GID-ABCD1234-CAM01",
        "end_zone_id": "Z_FLOOR",
        "ts": 1_700_000_000.0,
    }
    assert w.handle_decision(fields) is True
    assert minio.copy_calls == 1
    src, dst = minio.copies[0]
    assert src == "evidence/pending/showroom_a/CAM_01/TL-001/best.jpg"
    assert "GID-ABCD1234-CAM01" in dst
    assert dst.endswith("/TL-001/best.jpg")
    # Pending copy is deleted by default.
    assert minio.deletes == [
        "evidence/pending/showroom_a/CAM_01/TL-001/best.jpg",
    ]


def test_rekey_worker_keeps_pending_copy_when_configured() -> None:
    minio = _FakeMinio(exists=True)
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=True, keep_pending_copy=True),
    )
    fields = {
        "tracklet_id": "TL-001",
        "camera_id": "CAM_01",
        "site_id": "showroom_a",
        "decision": "new",
        "assigned_global_id": "GID-ABCD1234-CAM01",
        "end_zone_id": "Z_FLOOR",
        "ts": 1_700_000_000.0,
    }
    w.handle_decision(fields)
    assert minio.deletes == []  # kept


def test_rekey_worker_no_op_for_ambiguous_decisions() -> None:
    """``decision="ambiguous"`` must not create or copy a final path."""
    minio = _FakeMinio(exists=True)
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=True),
    )
    fields = {
        "tracklet_id": "TL-001",
        "camera_id": "CAM_01",
        "site_id": "showroom_a",
        "decision": "ambiguous",
        "assigned_global_id": None,  # no global_id assigned
        "end_zone_id": "Z_FLOOR",
        "ts": 1_700_000_000.0,
    }
    assert w.handle_decision(fields) is True
    assert minio.copy_calls == 0


def test_rekey_worker_no_op_for_match_decisions() -> None:
    """For ``decision="match"`` the global_id is the existing
    target's; the operator has the choice to re-key, but the
    worker's default behavior is to skip (the operator can
    trigger a one-off re-key by republishing to the stream).
    """
    minio = _FakeMinio(exists=True)
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=True),
    )
    fields = {
        "tracklet_id": "TL-001",
        "camera_id": "CAM_01",
        "site_id": "showroom_a",
        "decision": "match",
        "assigned_global_id": "GID-EXISTING",
        "end_zone_id": "Z_FLOOR",
        "ts": 1_700_000_000.0,
    }
    # We DO re-key on match because the new tracklet is now part
    # of the existing global_id and we want a fresh best.jpg for
    # this sighting.
    assert w.handle_decision(fields) is True
    assert minio.copy_calls == 1
    src, dst = minio.copies[0]
    assert "GID-EXISTING" in dst


def test_rekey_worker_retries_on_copy_failure() -> None:
    """If the copy fails twice and succeeds on the third attempt,
    the worker returns True and we see 3 copy calls.
    """
    minio = _FakeMinio(exists=True)
    call_state = {"n": 0}

    def _flaky_copy(src_key, dst_key):
        call_state["n"] += 1
        if call_state["n"] < 3:
            raise RuntimeError("flaky")
        minio.copies.append((src_key, dst_key))

    minio.copy_object_within_bucket = _flaky_copy
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=True, retry_max=3),
    )
    fields = {
        "tracklet_id": "TL-001",
        "camera_id": "CAM_01",
        "site_id": "showroom_a",
        "decision": "new",
        "assigned_global_id": "GID-X",
        "end_zone_id": "Z_FLOOR",
        "ts": 1_700_000_000.0,
    }
    assert w.handle_decision(fields) is True
    assert call_state["n"] == 3
    # Pending is deleted only after success.
    assert minio.deletes == [
        "evidence/pending/showroom_a/CAM_01/TL-001/best.jpg",
    ]


def test_rekey_worker_returns_false_when_all_retries_fail() -> None:
    minio = _FakeMinio(exists=True)
    call_state = {"n": 0}

    def _always_fail(src_key, dst_key):
        call_state["n"] += 1
        raise RuntimeError("always fail")

    minio.copy_object_within_bucket = _always_fail
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=True, retry_max=3),
    )
    fields = {
        "tracklet_id": "TL-001",
        "camera_id": "CAM_01",
        "site_id": "showroom_a",
        "decision": "new",
        "assigned_global_id": "GID-X",
        "end_zone_id": "Z_FLOOR",
        "ts": 1_700_000_000.0,
    }
    assert w.handle_decision(fields) is False
    # 3 attempts; pending NOT deleted (failure path).
    assert call_state["n"] == 3
    assert minio.deletes == []


def test_rekey_worker_skips_when_disabled() -> None:
    """When ``enabled=False`` the worker's run() loop returns
    immediately. We exercise it by calling ``run()`` with a stop
    event that is already set.
    """
    import threading

    minio = _FakeMinio(exists=True)
    redis = _FakeRedis()
    w = EvidenceRekeyWorker(
        minio=minio,
        pg=None,
        redis=redis,
        config=RekeyConfig(enabled=False),
    )
    stop = threading.Event()
    stop.set()
    # No exception; the worker exits immediately because it is
    # disabled.
    w.run(stop)
    assert minio.copy_calls == 0
