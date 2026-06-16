"""Legacy-compatible MinIO uploader (Phase 5b).

This module is a thin shim over the project's
:class:`app.storage.minio_store.MinioStore` that produces
*evidence* uploads in the same shape the legacy
``Service/offline-people-counting`` pipeline emits:

* bucket: ``yamaha-poc`` (configurable)
* object prefix: ``people-detection`` (configurable)
* object key: ``{prefix}/{cam_id}/{zone_slug}/{yyyy-mm-dd}/{epoch_ms}_{person_id}.jpg``
* content_type: ``image/jpeg``
* on upload failure: retry with exponential backoff (3 attempts, base 2s)

When ``ENABLE_MINIO_UPLOAD=false`` every public method becomes a
silent no-op that returns ``(None, None)`` so the rest of the
pipeline can run unchanged.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..storage.minio_store import MinioStore
from .legacy_contract import (
    flag_enabled,
    legacy_evidence_key,
)

logger = logging.getLogger(__name__)

UPLOAD_MAX_ATTEMPTS = 3
UPLOAD_BACKOFF_BASE_SECONDS = 2.0
DEFAULT_JPEG_QUALITY = 85


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def crop_bbox(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    """Crop *frame* to *bbox* (clamped to image bounds)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def encode_jpeg(frame: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> Optional[bytes]:
    import cv2

    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if success else None


class LegacyMinioUploader:
    """Per-frame uploader with the legacy evidence key + retry policy.

    The constructor is lightweight: no MinIO client is opened until the
    first ``upload_person_crop`` call. If
    :func:`app.integrations.legacy_contract.flag_enabled` reports
    ``ENABLE_MINIO_UPLOAD=false`` every public method is a no-op.
    """

    def __init__(
        self,
        store: MinioStore,
        *,
        camera_id: str,
        enabled_override: Optional[bool] = None,
    ) -> None:
        self._store = store
        self._camera_id = camera_id
        # ``legacy_evidence_key`` normalizes camera_id (CAM_01 -> cam1).
        self._enabled = (
            enabled_override
            if enabled_override is not None
            else flag_enabled("ENABLE_MINIO_UPLOAD")
        )
        if not self._enabled:
            logger.info(
                "legacy MinIO uploader disabled by ENABLE_MINIO_UPLOAD=false | camera=%s",
                camera_id,
            )

    @property
    def is_enabled(self) -> bool:
        return bool(self._enabled)

    @property
    def camera_id(self) -> str:
        return self._camera_id

    def upload_person_crop(
        self,
        frame: np.ndarray,
        bbox: tuple,
        person_id: int,
        zone: str | None = None,
        timestamp: datetime | None = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Upload a person crop and return ``(presigned_url, object_name)``.

        Returns ``(None, None)`` if MinIO is disabled, the bucket is
        unreachable, or the upload failed after all retries.
        """
        if not self._enabled:
            return None, None
        # If the underlying store has no client, treat as disabled.
        if self._store is None or getattr(self._store, "_client", None) is None:
            return None, None
        cropped = crop_bbox(frame, bbox)
        if cropped is None:
            return None, None
        encoded = encode_jpeg(cropped)
        if encoded is None:
            return None, None
        object_name = legacy_evidence_key(
            camera_id=self._camera_id,
            zone=zone,
            person_id=int(person_id),
            timestamp_epoch=timestamp.timestamp() if timestamp is not None else None,
        )
        return self._upload_with_retry(object_name, encoded)

    def _upload_with_retry(
        self, object_name: str, data: bytes
    ) -> tuple[Optional[str], Optional[str]]:
        client = self._store._client  # noqa: SLF001
        if client is None:
            return None, None
        last_error: Exception | None = None
        for attempt in range(UPLOAD_MAX_ATTEMPTS):
            try:
                client.put_object(
                    bucket_name=self._store.bucket,
                    object_name=object_name,
                    data=io.BytesIO(data),
                    length=len(data),
                    content_type="image/jpeg",
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < UPLOAD_MAX_ATTEMPTS - 1:
                    time.sleep(UPLOAD_BACKOFF_BASE_SECONDS**attempt)
        else:
            logger.error(
                "legacy minio upload failed | object=%s | error=%s",
                object_name,
                last_error,
            )
            return None, None
        try:
            url = client.presigned_get_object(
                bucket_name=self._store.bucket,
                object_name=object_name,
                expires=7 * 24 * 3600,  # 7 days
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("legacy minio presign failed | error=%s", exc)
            return None, object_name
        return url, object_name
