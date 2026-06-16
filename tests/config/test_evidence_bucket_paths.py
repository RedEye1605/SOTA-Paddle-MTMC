"""Tests for evidence path helpers + bucket routing (Phase 4).

These tests focus on the *path scheme* contract — the path is
deterministic, includes the right pieces, and a missing bucket fails
fast.
"""

from __future__ import annotations

from unittest import mock

import pytest

from app.storage.minio_store import MinioStore


# ---------------------------------------------------------------------------
# path shape contracts
# ---------------------------------------------------------------------------


def test_evidence_key_has_yyyy_mm_dd_subdirs() -> None:
    k = MinioStore.evidence_key(
        site_id="site",
        camera_id="CAM_01",
        zone_id="Z",
        ts=1_710_000_000.0,  # 2024-03-09 UTC
        global_id="GID",
        tracklet_id="trk",
        kind="best",
    )
    parts = k.split("/")
    # evidence/site/CAM_01/Z/yyyy/mm/dd/GID/trk/best.jpg  -> 10 parts
    assert parts[0] == "evidence"
    assert parts[1] == "site"
    assert parts[2] == "CAM_01"
    assert parts[3] == "Z"
    assert parts[4] == "2024"
    assert parts[5] == "03"
    assert parts[6] == "09"
    assert parts[7] == "GID"
    assert parts[8] == "trk"
    assert parts[9] == "best.jpg"


def test_debug_evidence_key_uses_frame_id() -> None:
    k = MinioStore.evidence_key(
        site_id="site",
        camera_id="CAM_01",
        zone_id="Z",
        ts=1_710_000_000.0,
        global_id="GID",
        tracklet_id="trk",
        kind="debug",
        frame_id=42,
    )
    assert k.endswith("/debug_000042.jpg")


def test_debug_without_frame_id_raises() -> None:
    with pytest.raises(ValueError):
        MinioStore.evidence_key(
            site_id="site",
            camera_id="CAM_01",
            zone_id="Z",
            ts=1_710_000_000.0,
            global_id="GID",
            tracklet_id="trk",
            kind="debug",
            frame_id=None,
        )


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        MinioStore.evidence_key(
            site_id="site",
            camera_id="CAM_01",
            zone_id="Z",
            ts=1_710_000_000.0,
            global_id="GID",
            tracklet_id="trk",
            kind="thumbnail",
        )


def test_pending_evidence_key_is_distinct_from_final() -> None:
    """Pending uses a flat path (no yyyy/mm/dd), final uses a dated path."""
    pending = MinioStore.pending_evidence_key(
        site_id="site",
        camera_id="CAM_01",
        tracklet_id="trk",
    )
    final = MinioStore.evidence_key(
        site_id="site",
        camera_id="CAM_01",
        zone_id="Z",
        ts=1_710_000_000.0,
        global_id="GID",
        tracklet_id="trk",
        kind="best",
    )
    assert pending.startswith("evidence/pending/")
    assert final.startswith("evidence/site/")
    assert pending != final


def test_parse_evidence_uri_round_trip() -> None:
    uri = MinioStore.evidence_key(
        site_id="site",
        camera_id="CAM_01",
        zone_id="Z",
        ts=1_710_000_000.0,
        global_id="GID",
        tracklet_id="trk",
        kind="best",
    )
    parsed = MinioStore.parse_evidence_uri(f"s3://evidence/{uri}")
    assert parsed is not None
    assert parsed["bucket"] == "evidence"
    assert parsed["key"] == uri


def test_parse_evidence_uri_rejects_non_evidence() -> None:
    assert MinioStore.parse_evidence_uri("s3://b/reports/x.json") is None
    assert MinioStore.parse_evidence_uri("http://x/y") is None
    assert MinioStore.parse_evidence_uri("s3://b") is None  # no key


# ---------------------------------------------------------------------------
# bucket routing for evidence uploads
# ---------------------------------------------------------------------------


def test_put_crop_writes_to_evidence_bucket() -> None:
    import numpy as np

    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        bucket_reports="rp",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = True
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    uri = s.put_crop(
        site_id="site",
        camera_id="CAM_01",
        zone_id="Z",
        ts=1_710_000_000.0,
        global_id="GID",
        tracklet_id="trk",
        image=image,
        kind="best",
    )
    assert uri.startswith("s3://ev/")
    s._client.put_object.assert_called_once()
    kwargs = s._client.put_object.call_args.kwargs
    assert kwargs["bucket_name"] == "ev"
    assert kwargs["object_name"].startswith("evidence/")


def test_put_crop_unassigned_routes_to_pending() -> None:
    import numpy as np

    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = True
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    uri = s.put_crop(
        site_id="site",
        camera_id="CAM_01",
        zone_id="Z",
        ts=1_710_000_000.0,
        global_id="UNASSIGNED",
        tracklet_id="trk",
        image=image,
        kind="best",
    )
    assert uri.startswith("s3://ev/")
    kwargs = s._client.put_object.call_args.kwargs
    assert kwargs["object_name"].startswith("evidence/pending/")


def test_put_crop_fails_fast_on_missing_evidence_bucket() -> None:
    import numpy as np

    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
        create_buckets=False,
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = False  # bucket missing
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError) as exc:
        s.put_crop(
            site_id="site",
            camera_id="CAM_01",
            zone_id="Z",
            ts=1_710_000_000.0,
            global_id="GID",
            tracklet_id="trk",
            image=image,
            kind="best",
        )
    assert "ev" in str(exc.value)
    s._client.put_object.assert_not_called()


def test_copy_object_within_bucket_stays_in_evidence() -> None:
    s = MinioStore(
        endpoint="x",
        access_key="a",
        secret_key="b",
        bucket="ev",
    )
    s._client = mock.MagicMock()
    s._client.bucket_exists.return_value = True
    uri = s.copy_object_within_bucket(
        src_key="evidence/pending/site/CAM_01/trk/best.jpg",
        dst_key="evidence/site/CAM_01/Z/2024/03/09/GID/trk/best.jpg",
    )
    assert uri.startswith("s3://ev/")
    kwargs = s._client.copy_object.call_args.kwargs
    assert kwargs["bucket_name"] == "ev"
    # Source bucket is the same as the destination bucket.
    assert kwargs["source"].bucket_name == "ev"
