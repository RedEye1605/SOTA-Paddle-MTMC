"""Tests for the legacy MinIO uploader (Phase 5b).

* bucket = ``yamaha-poc`` (configurable)
* object prefix = ``people-detection``
* object key = ``{prefix}/{cam_id}/{zone_slug}/{yyyy-mm-dd}/{epoch_ms}_{pid}.jpg``
* ``ENABLE_MINIO_UPLOAD=false`` ⇒ every call is a silent no-op
* enabled mode actually calls ``put_object`` with the legacy key
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

import numpy as np

from app.integrations.legacy_contract import legacy_evidence_key
from app.integrations.minio_uploader import (
    LegacyMinioUploader,
    crop_bbox,
    encode_jpeg,
)


def _fake_store(bucket: str = "yamaha-poc") -> mock.MagicMock:
    s = mock.MagicMock()
    s.bucket = bucket
    s._client = mock.MagicMock()  # noqa: SLF001
    return s


# ---------------------------------------------------------------------------
# Crop + encode helpers
# ---------------------------------------------------------------------------


def test_crop_bbox_clamps_to_image_bounds() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = crop_bbox(img, (10, 5, 250, 150))
    assert out is not None
    assert out.shape == (95, 190, 3)


def test_crop_bbox_invalid_returns_none() -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    assert crop_bbox(img, (10, 10, 5, 5)) is None


def test_encode_jpeg_returns_bytes() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out = encode_jpeg(img)
    assert isinstance(out, (bytes, bytearray))
    assert len(out) > 0


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------


def test_disabled_mode_skips_upload() -> None:
    uploader = LegacyMinioUploader(_fake_store(), camera_id="cam1", enabled_override=False)
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    url, key = uploader.upload_person_crop(img, (0, 0, 10, 10), person_id=1, zone="Sport Zone")
    assert url is None
    assert key is None
    uploader._store._client.put_object.assert_not_called()  # noqa: SLF001


def test_disabled_when_store_has_no_client() -> None:
    store = mock.MagicMock()
    store.bucket = "yamaha-poc"
    store._client = None  # noqa: SLF001
    uploader = LegacyMinioUploader(store, camera_id="cam1", enabled_override=True)
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    url, key = uploader.upload_person_crop(img, (0, 0, 5, 5), 1)
    assert (url, key) == (None, None)


# ---------------------------------------------------------------------------
# Enabled mode
# ---------------------------------------------------------------------------


def test_enabled_mode_calls_put_object_with_legacy_key() -> None:
    store = _fake_store(bucket="yamaha-poc")
    uploader = LegacyMinioUploader(store, camera_id="cam1", enabled_override=True)
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    ts = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    url, key = uploader.upload_person_crop(
        img, (0, 0, 20, 20), person_id=42, zone="Sport Zone", timestamp=ts
    )
    assert key is not None
    assert key.startswith("people-detection/cam1/sport-zone/2023-11-14/")
    assert key.endswith("_42.jpg")
    # The URL is a presigned URL (None here because the magic mock
    # returns a MagicMock, but we still assert the bucket + content).
    store._client.put_object.assert_called_once()  # noqa: SLF001
    kwargs = store._client.put_object.call_args.kwargs  # noqa: SLF001
    assert kwargs["bucket_name"] == "yamaha-poc"
    assert kwargs["object_name"] == key
    assert kwargs["content_type"] == "image/jpeg"


def test_enabled_mode_uses_legacy_evidence_key_for_zone_fallback() -> None:
    store = _fake_store()
    uploader = LegacyMinioUploader(store, camera_id="cam1", enabled_override=True)
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    ts = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
    url, key = uploader.upload_person_crop(
        img, (0, 0, 20, 20), person_id=1, zone=None, timestamp=ts
    )
    assert key is not None
    # Legacy fallback folder is the location_title slugified.
    assert "/main-hallway/" in key


def test_legacy_minio_bucket_and_prefix_from_yaml() -> None:
    from app.integrations import legacy_contract as lc

    cfg = lc._load_config()
    assert cfg["minio"]["bucket"] == "yamaha-poc"
    assert cfg["minio"]["object_prefix"] == "people-detection"


def test_legacy_evidence_key_consistent_with_uploader() -> None:
    """The uploader must produce keys identical to the helper."""
    expected = legacy_evidence_key(
        camera_id="cam1",
        zone="Premium Zone",
        person_id=7,
        timestamp_epoch=1_700_000_000.0,
    )
    store = _fake_store()
    uploader = LegacyMinioUploader(store, camera_id="cam1", enabled_override=True)
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    _url, key = uploader.upload_person_crop(
        img,
        (0, 0, 10, 10),
        person_id=7,
        zone="Premium Zone",
        timestamp=datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc),
    )
    assert key == expected


# ---------------------------------------------------------------------------
# Retry on failure
# ---------------------------------------------------------------------------


def test_retry_on_failure_eventually_returns_none() -> None:
    store = _fake_store()
    store._client.put_object.side_effect = RuntimeError("boom")  # noqa: SLF001
    uploader = LegacyMinioUploader(store, camera_id="cam1", enabled_override=True)
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    url, key = uploader.upload_person_crop(img, (0, 0, 5, 5), person_id=1)
    assert url is None
    assert key is None
    # 3 attempts (UPLOAD_MAX_ATTEMPTS).
    assert store._client.put_object.call_count == 3  # noqa: SLF001


def test_retry_succeeds_on_second_attempt() -> None:
    store = _fake_store()
    # First attempt fails, second succeeds.
    store._client.put_object.side_effect = [  # noqa: SLF001
        RuntimeError("transient"),
        mock.DEFAULT,
    ]
    uploader = LegacyMinioUploader(store, camera_id="cam1", enabled_override=True)
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    url, key = uploader.upload_person_crop(img, (0, 0, 5, 5), person_id=1)
    assert key is not None
    assert store._client.put_object.call_count == 2  # noqa: SLF001
