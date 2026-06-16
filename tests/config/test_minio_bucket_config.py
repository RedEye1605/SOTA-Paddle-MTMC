"""Tests for the 3-bucket MinIO layout (Phase 4).

Covers:
  1. bucket name comes from env / config
  2. missing bucket fails clearly (or creates if create_buckets=true)
  3. evidence path includes camera_id and global_id when available
  4. no secret values appear in logs (covered by the test that
     inspects caplog/printable output)
  5. visualization/report keys are deterministic and bucket-aware
  6. ``from_env()`` honors the new ``MINIO_BUCKET_EVIDENCE`` /
     ``MINIO_BUCKET_REPORTS`` / ``MINIO_BUCKET_MODELS`` variables
  7. the legacy ``MINIO_BUCKET`` alias still works
"""

from __future__ import annotations

import logging
from unittest import mock

import pytest

from app.storage import minio_store
from app.storage.minio_store import MinioStore


# ---------------------------------------------------------------------------
# 1. bucket names come from env / config
# ---------------------------------------------------------------------------


def test_bucket_names_come_from_constructor() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="rp",
        bucket_models="md",
    )
    assert s.bucket == "ev"
    assert s.reports_bucket == "rp"
    assert s.models_bucket == "md"


def test_from_env_reads_three_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.example.com:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "ak")
    monkeypatch.setenv("MINIO_SECRET_KEY", "sk")
    monkeypatch.setenv("MINIO_SECURE", "false")
    monkeypatch.setenv("MINIO_BUCKET_EVIDENCE", "evidence-2026")
    monkeypatch.setenv("MINIO_BUCKET_REPORTS", "reports-2026")
    monkeypatch.setenv("MINIO_BUCKET_MODELS", "models-2026")

    s = minio_store.from_env()
    assert s.bucket == "evidence-2026"
    assert s.reports_bucket == "reports-2026"
    assert s.models_bucket == "models-2026"


def test_from_env_legacy_minio_bucket_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """If only ``MINIO_BUCKET`` is set, it acts as the evidence bucket."""
    monkeypatch.delenv("MINIO_BUCKET_EVIDENCE", raising=False)
    monkeypatch.setenv("MINIO_BUCKET", "legacy-evidence")
    monkeypatch.delenv("MINIO_BUCKET_REPORTS", raising=False)
    monkeypatch.delenv("MINIO_BUCKET_MODELS", raising=False)
    s = minio_store.from_env()
    assert s.bucket == "legacy-evidence"
    assert s.reports_bucket == ""
    assert s.models_bucket == ""


def test_from_env_evidence_preferred_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MINIO_BUCKET_EVIDENCE`` wins over ``MINIO_BUCKET``."""
    monkeypatch.setenv("MINIO_BUCKET", "legacy")
    monkeypatch.setenv("MINIO_BUCKET_EVIDENCE", "preferred")
    s = minio_store.from_env()
    assert s.bucket == "preferred"


def test_from_env_reports_bucket_omitted_disables_upload() -> None:
    """An empty reports bucket must mean ``None`` so ``put_report`` is a no-op."""
    s = minio_store.from_env()
    # Either empty-string or None is fine; the contract is "no upload".
    assert s.reports_bucket in (None, "")


# ---------------------------------------------------------------------------
# 2. missing bucket fails clearly / create_buckets works
# ---------------------------------------------------------------------------


def test_missing_evidence_bucket_raises_when_create_disabled() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
    )
    fake_client = mock.MagicMock()
    fake_client.bucket_exists.return_value = False
    s._client = fake_client
    with pytest.raises(RuntimeError) as exc:
        s._require_bucket("ev")
    assert "ev" in str(exc.value)
    assert "MINIO_CREATE_BUCKETS" in str(exc.value)
    fake_client.make_bucket.assert_not_called()


def test_create_buckets_true_lazily_creates_bucket() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        create_buckets=True,
    )
    fake_client = mock.MagicMock()
    fake_client.bucket_exists.return_value = False
    s._client = fake_client
    s._maybe_make_bucket("ev")
    fake_client.make_bucket.assert_called_once_with("ev")
    # Second call is a cache hit.
    s._maybe_make_bucket("ev")
    fake_client.make_bucket.assert_called_once()


def test_require_bucket_caches_known_buckets() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
    )
    fake_client = mock.MagicMock()
    fake_client.bucket_exists.return_value = True
    s._client = fake_client
    s._require_bucket("ev")
    s._require_bucket("ev")
    # Only one HEAD/Exists call.
    assert fake_client.bucket_exists.call_count == 1


def test_empty_bucket_name_is_silent_no_op() -> None:
    """Reports/models buckets may be unset; the helpers must not raise."""
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="",
        bucket_models="",
    )
    s._client = mock.MagicMock()
    # No raise
    s._require_bucket("")
    s._maybe_make_bucket("")
    s._client.bucket_exists.assert_not_called()


# ---------------------------------------------------------------------------
# 3. evidence path includes camera_id and global_id
# ---------------------------------------------------------------------------


def test_evidence_key_includes_camera_id_and_global_id() -> None:
    k = MinioStore.evidence_key(
        site_id="yamaha_showroom",
        camera_id="CAM_01",
        zone_id="ZONE_A",
        ts=1_700_000_000.0,
        global_id="GID_0001",
        tracklet_id="track-abc",
        kind="best",
    )
    assert "CAM_01" in k
    assert "GID_0001" in k
    assert "yamaha_showroom" in k
    assert "ZONE_A" in k
    assert k.startswith("evidence/")
    assert k.endswith("/best.jpg")


def test_pending_evidence_key_omits_global_id() -> None:
    k = MinioStore.pending_evidence_key(
        site_id="s",
        camera_id="CAM_01",
        tracklet_id="track-abc",
        kind="best",
    )
    assert k.startswith("evidence/pending/")
    assert "track-abc" in k
    assert "GID" not in k  # global_id is not used here
    assert k.endswith("/best.jpg")


# ---------------------------------------------------------------------------
# 4. no secret values appear in logs
# ---------------------------------------------------------------------------


def test_no_secret_in_warning_when_bucket_missing(caplog) -> None:
    s = MinioStore(
        endpoint="x",
        access_key="AKIASECRETSECRET",
        secret_key="this-is-a-very-secret-key",
        bucket="ev",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = False
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            s._require_bucket("ev")
    text = caplog.text
    # No secret-like strings should appear in the warning.
    assert "AKIASECRETSECRET" not in text
    assert "this-is-a-very-secret-key" not in text


# ---------------------------------------------------------------------------
# 5. visualization/report keys are deterministic
# ---------------------------------------------------------------------------


def test_visualization_key_is_deterministic() -> None:
    a = MinioStore.visualization_key("site", "CAM_01", 1_700_000_000.0)
    b = MinioStore.visualization_key("site", "CAM_01", 1_700_000_000.0)
    assert a == b
    assert a.startswith("visualization/site/CAM_01/")
    assert a.endswith("/first_3000_frames.mp4")


def test_visualization_key_differs_per_camera() -> None:
    a = MinioStore.visualization_key("site", "CAM_01", 1_700_000_000.0)
    b = MinioStore.visualization_key("site", "CAM_02", 1_700_000_000.0)
    assert a != b
    assert "CAM_01" in a
    assert "CAM_02" in b


def test_report_key_is_deterministic() -> None:
    a = MinioStore.report_key("site", 1_700_000_000.0)
    b = MinioStore.report_key("site", 1_700_000_000.0)
    assert a == b
    assert a.startswith("reports/site/")
    assert a.endswith("_1700000000.json")


# ---------------------------------------------------------------------------
# 6. put_visualization / put_report bucket routing
# ---------------------------------------------------------------------------


def test_put_visualization_uses_reports_bucket() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="rp",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = True
    s._client.fput_object.return_value = mock.MagicMock()
    uri = s.put_visualization(
        site_id="site",
        camera_id="CAM_01",
        ts=1_700_000_000.0,
        file_path=mock.MagicMock(),
    )
    assert uri is not None
    assert uri.startswith("s3://rp/")
    s._client.fput_object.assert_called_once()
    kwargs = s._client.fput_object.call_args.kwargs
    assert kwargs["bucket_name"] == "rp"


def test_put_visualization_noop_when_reports_bucket_disabled(tmp_path) -> None:
    """If the reports bucket is empty, the upload is a silent no-op."""
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="",
        bucket_models="",
    )
    s._client = mock.MagicMock()
    fp = tmp_path / "v.mp4"
    fp.write_bytes(b"x" * 8)
    uri = s.put_visualization(
        site_id="site",
        camera_id="CAM_01",
        ts=1_700_000_000.0,
        file_path=fp,
    )
    assert uri is None
    s._client.fput_object.assert_not_called()


def test_put_report_uses_reports_bucket() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="rp",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = True
    s._client.fput_object.return_value = mock.MagicMock()
    uri = s.put_report(
        site_id="site",
        ts=1_700_000_000.0,
        file_path=mock.MagicMock(),
    )
    assert uri is not None
    assert uri.startswith("s3://rp/")
    kwargs = s._client.fput_object.call_args.kwargs
    assert kwargs["bucket_name"] == "rp"


def test_put_file_uses_explicit_bucket() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="rp",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = True
    s._client.fput_object.return_value = mock.MagicMock()
    uri = s.put_file(key="k", file_path=mock.MagicMock(), bucket="rp")
    assert uri.startswith("s3://rp/")
    assert s._client.fput_object.call_args.kwargs["bucket_name"] == "rp"


def test_put_file_default_is_evidence_bucket() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="rp",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = True
    s._client.fput_object.return_value = mock.MagicMock()
    s.put_file(key="k", file_path=mock.MagicMock())
    assert s._client.fput_object.call_args.kwargs["bucket_name"] == "ev"


# ---------------------------------------------------------------------------
# 7. connect() respects create_buckets
# ---------------------------------------------------------------------------


def test_connect_with_create_buckets_calls_make_bucket(monkeypatch) -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        create_buckets=True,
    )
    fake_client = mock.MagicMock()
    fake_client.bucket_exists.return_value = False
    monkeypatch.setattr("minio.Minio", lambda *a, **k: fake_client)
    s.connect()
    fake_client.make_bucket.assert_called_once_with("ev")


def test_connect_without_create_buckets_raises(monkeypatch) -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        create_buckets=False,
    )
    fake_client = mock.MagicMock()
    fake_client.bucket_exists.return_value = False
    monkeypatch.setattr("minio.Minio", lambda *a, **k: fake_client)
    with pytest.raises(RuntimeError):
        s.connect()
    fake_client.make_bucket.assert_not_called()


def test_connect_idempotent(monkeypatch) -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
    )
    fake_client = mock.MagicMock()
    fake_client.bucket_exists.return_value = True
    monkeypatch.setattr("minio.Minio", lambda *a, **k: fake_client)
    s.connect()
    s.connect()  # second call is a no-op
    assert fake_client.bucket_exists.call_count == 1
