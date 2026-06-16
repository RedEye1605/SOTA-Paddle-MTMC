"""MinIO — evidence crop storage with deterministic paths.

Phase 4 multi-bucket split (see ``Docs/minio_bucket_setup.md``):

* ``MINIO_BUCKET_EVIDENCE`` (default: ``evidence``) — person crops,
  debug frames, ``best.jpg``.
* ``MINIO_BUCKET_REPORTS``  (default: ``reports``)  — benchmark
  JSON, visualization MP4 sidecars.
* ``MINIO_BUCKET_MODELS``   (default: ``models``)   — reserved.

The constructor takes the **evidence** bucket as the primary one
(kept for backward compatibility) plus optional ``bucket_reports``
and ``bucket_models``. Each public method chooses the right
bucket; if that bucket is empty/None, the call is a no-op (we do
NOT silently fall back to a different bucket).
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MinioStore:
    """S3-compatible evidence storage via the official `minio` Python SDK."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        bucket: str = "evidence",
        bucket_reports: Optional[str] = None,
        bucket_models: Optional[str] = None,
        create_buckets: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._bucket = bucket
        self._bucket_reports = bucket_reports or ""
        self._bucket_models = bucket_models or ""
        self._create_buckets = bool(create_buckets)
        self._client = None  # lazy import
        # Track which buckets we have already verified exist on this
        # connection. The official `minio` SDK does not have a
        # session-level "ensure_bucket"; we cache the result to avoid
        # a HEAD per upload.
        self._known_buckets: set[str] = set()

    def connect(self) -> None:
        if self._client is not None:
            return
        if not self._endpoint:
            # PATCH (2026-06-17): the internal minio service was removed
            # from docker-compose. An empty endpoint almost certainly
            # means the operator forgot to set MINIO_ENDPOINT in .env,
            # which would otherwise produce a confusing ``Minio: cannot
            # resolve host ''`` error from the SDK. Fail fast.
            raise RuntimeError(
                "MINIO_ENDPOINT is not set — refusing to start. Set "
                "MINIO_ENDPOINT in .env to the operator's external "
                "MinIO endpoint (e.g. minio.example.invalid:9000)."
            )
        from minio import Minio

        self._client = Minio(
            self._endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )
        if self._create_buckets:
            self._maybe_make_bucket(self._bucket)
        else:
            self._require_bucket(self._bucket)
        logger.info(
            "MinIO ready: %s bucket=%s reports=%s models=%s create_buckets=%s",
            self._endpoint,
            self._bucket,
            self._bucket_reports or "(disabled)",
            self._bucket_models or "(disabled)",
            self._create_buckets,
        )

    def close(self) -> None:
        self._client = None
        self._known_buckets.clear()

    @property
    def client(self):
        assert self._client is not None, "MinioStore.connect() first"
        return self._client

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def reports_bucket(self) -> str:
        return self._bucket_reports

    @property
    def models_bucket(self) -> str:
        return self._bucket_models

    def _require_bucket(self, name: str) -> None:
        """Verify the bucket exists. Raises ``RuntimeError`` otherwise."""
        if not name or name in self._known_buckets:
            return
        if not self._client.bucket_exists(name):
            raise RuntimeError(
                f"MinIO bucket does not exist: {name!r} (set "
                f"MINIO_CREATE_BUCKETS=true to lazily create dev buckets)",
            )
        self._known_buckets.add(name)

    def _maybe_make_bucket(self, name: str) -> None:
        """Create the bucket if it does not exist. No-op if already known."""
        if not name or name in self._known_buckets:
            return
        if not self._client.bucket_exists(name):
            self._client.make_bucket(name)
            logger.info("MinIO bucket created: %s", name)
        self._known_buckets.add(name)

    # ---- health probe ----
    def is_reachable(self, timeout: float = 2.0) -> bool:
        """Quick reachability check used by the /health endpoint.

        Issues a HEAD on the evidence bucket via the official minio
        SDK (``bucket_exists`` is read-only — the bucket is never
        created). Returns ``True`` if the call succeeds, ``False`` on
        any exception (network down, DNS failure, bucket missing,
        auth error, etc.).

        The ``timeout`` parameter is advisory: the minio Python SDK
        (>=7.x) does not expose a per-call timeout knob, so the
        caller (the /health handler) enforces the wall-clock bound
        via ``asyncio.wait_for(asyncio.to_thread(minio.is_reachable),
        timeout=...)``. We accept the kwarg for API consistency with
        the rest of the storage layer.
        """
        del timeout  # minio SDK has no per-call timeout knob.
        try:
            return bool(self.client.bucket_exists(self._bucket))
        except Exception as e:  # noqa: BLE001
            logger.debug("MinioStore.is_reachable failed: %s", e)
            return False

    # ---- path scheme ----
    @staticmethod
    def evidence_key(
        site_id: str,
        camera_id: str,
        zone_id: str,
        ts: float,
        global_id: str,
        tracklet_id: str,
        kind: str = "best",
        frame_id: Optional[int] = None,
    ) -> str:
        """Deterministic evidence path.

        s3://evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/{kind}.jpg
        """
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        yyyy = dt.strftime("%Y")
        mm = dt.strftime("%m")
        dd = dt.strftime("%d")
        if kind == "best":
            return (
                f"evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/"
                f"{global_id}/{tracklet_id}/best.jpg"
            )
        if kind == "debug" and frame_id is not None:
            return (
                f"evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/"
                f"{global_id}/{tracklet_id}/debug_{frame_id:06d}.jpg"
            )
        raise ValueError(f"Unknown evidence kind or missing frame_id: {kind}, {frame_id}")

    @staticmethod
    def pending_evidence_key(
        site_id: str,
        camera_id: str,
        tracklet_id: str,
        kind: str = "best",
        frame_id: Optional[int] = None,
    ) -> str:
        """PATCH-029: pending evidence path used while the resolver
        has not yet assigned a global_id.

        s3://{bucket}/evidence/pending/{site_id}/{camera_id}/{tracklet_id}/{kind}.jpg
        """
        if kind == "best":
            return f"evidence/pending/{site_id}/{camera_id}/{tracklet_id}/best.jpg"
        if kind == "debug" and frame_id is not None:
            return f"evidence/pending/{site_id}/{camera_id}/{tracklet_id}/debug_{frame_id:06d}.jpg"
        raise ValueError(f"Unknown evidence kind or missing frame_id: {kind}, {frame_id}")

    @staticmethod
    def visualization_key(
        site_id: str,
        camera_id: str,
        ts: float,
        kind: str = "first_3000_frames",
    ) -> str:
        """Deterministic visualization path.

        s3://{bucket}/visualization/{site_id}/{camera_id}/{yyyy}/{mm}/{dd}/{kind}.mp4
        """
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        yyyy = dt.strftime("%Y")
        mm = dt.strftime("%m")
        dd = dt.strftime("%d")
        return f"visualization/{site_id}/{camera_id}/{yyyy}/{mm}/{dd}/{kind}.mp4"

    @staticmethod
    def report_key(site_id: str, ts: float, kind: str = "benchmark") -> str:
        """Deterministic report path.

        s3://{bucket}/reports/{site_id}/{yyyy}/{mm}/{dd}/{kind}_{timestamp}.json
        """
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        yyyy = dt.strftime("%Y")
        mm = dt.strftime("%m")
        dd = dt.strftime("%d")
        return f"reports/{site_id}/{yyyy}/{mm}/{dd}/{kind}_{int(ts)}.json"

    @staticmethod
    def parse_evidence_uri(uri: str) -> dict[str, str] | None:
        """Parse an ``s3://{bucket}/evidence/...`` URI into its parts.

        Returns None if the URI is malformed or does not look like an
        evidence path. Used by the re-key worker.
        """
        if not uri.startswith("s3://"):
            return None
        path = uri.split("://", 1)[1]
        # path is "{bucket}/{key...}"
        parts = path.split("/", 1)
        if len(parts) != 2:
            return None
        bucket, key = parts
        if not key.startswith("evidence/"):
            return None
        return {"bucket": bucket, "key": key}

    # ---- upload ----
    def put_crop(
        self,
        site_id: str,
        camera_id: str,
        zone_id: str,
        ts: float,
        global_id: str,
        tracklet_id: str,
        image: np.ndarray,
        kind: str = "best",
        frame_id: Optional[int] = None,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload a BGR crop and return the s3 URI.

        PATCH-029: when ``global_id == "UNASSIGNED"`` we route the
        upload to the *pending* evidence path
        (``evidence/pending/{site}/{camera}/{tracklet}/best.jpg``) so
        the re-key worker can copy it to the final path after the
        resolver assigns a real global_id. When ``global_id`` is a
        real identifier, we go straight to the dated path.
        """
        import cv2

        if global_id == "UNASSIGNED":
            key = self.pending_evidence_key(
                site_id,
                camera_id,
                tracklet_id,
                kind=kind,
                frame_id=frame_id,
            )
        else:
            key = self.evidence_key(
                site_id,
                camera_id,
                zone_id,
                ts,
                global_id,
                tracklet_id,
                kind=kind,
                frame_id=frame_id,
            )
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise RuntimeError("cv2.imencode failed for jpeg")
        data = BytesIO(buf.tobytes())
        length = len(buf.tobytes())
        self._require_bucket(self._bucket)
        self.client.put_object(
            bucket_name=self._bucket,
            object_name=key,
            data=data,
            length=length,
            content_type=content_type,
        )
        return f"s3://{self._bucket}/{key}"

    def put_file(
        self,
        key: str,
        file_path: Path,
        content_type: str = "image/jpeg",
        bucket: Optional[str] = None,
    ) -> str:
        """Upload a local file. Defaults to the evidence bucket; pass
        ``bucket=`` to upload to the reports or models bucket.
        """
        target = bucket or self._bucket
        self._require_bucket(target)
        self.client.fput_object(
            bucket_name=target,
            object_name=key,
            file_path=str(file_path),
            content_type=content_type,
        )
        return f"s3://{target}/{key}"

    def put_visualization(
        self,
        site_id: str,
        camera_id: str,
        ts: float,
        file_path: Path,
        kind: str = "first_3000_frames",
        content_type: str = "video/mp4",
    ) -> Optional[str]:
        """Upload the annotated MP4 sidecar to the *reports* bucket.

        Returns the ``s3://`` URI, or ``None`` if the reports bucket is
        not configured (silent skip — the visualization is always
        written locally to begin with).
        """
        if not self._bucket_reports:
            logger.debug("reports bucket not configured; skipping visualization upload")
            return None
        key = self.visualization_key(site_id, camera_id, ts, kind=kind)
        return self.put_file(
            key=key,
            file_path=file_path,
            content_type=content_type,
            bucket=self._bucket_reports,
        )

    def put_report(
        self,
        site_id: str,
        ts: float,
        file_path: Path,
        kind: str = "benchmark",
        content_type: str = "application/json",
    ) -> Optional[str]:
        """Upload a benchmark / report JSON to the *reports* bucket."""
        if not self._bucket_reports:
            logger.debug("reports bucket not configured; skipping report upload")
            return None
        key = self.report_key(site_id, ts, kind=kind)
        return self.put_file(
            key=key,
            file_path=file_path,
            content_type=content_type,
            bucket=self._bucket_reports,
        )

    def copy_object_within_bucket(
        self,
        src_key: str,
        dst_key: str,
    ) -> str:
        """Server-side copy of an object within the same bucket.

        Used by the tracklet collector to materialize a ``best.jpg``
        next to the debug crops. Returns the destination s3 URI.
        """
        from minio.commonconfig import CopySource

        self.client.copy_object(
            bucket_name=self._bucket,
            object_name=dst_key,
            source=CopySource(bucket_name=self._bucket, object_name=src_key),
        )
        return f"s3://{self._bucket}/{dst_key}"

    def get_object_bytes(self, key: str) -> Optional[bytes]:
        """Read a single object's bytes. Returns None on any error."""
        try:
            resp = self.client.get_object(bucket_name=self._bucket, object_name=key)
            data = resp.read()
            resp.close()
            resp.release_conn()
            return data
        except Exception as e:  # noqa: BLE001
            logger.warning("get_object_bytes failed for %s: %s", key, e)
            return None

    def delete_older_than(self, cutoff_ts: float) -> int:
        """Delete objects whose ``Last-Modified`` is older than cutoff_ts.

        Returns the number of objects deleted. Implemented as a list +
        delete to keep the code simple; for large buckets use the
        server-side lifecycle policy (see Docs/).
        """
        deleted = 0
        try:
            objs = list(self.client.list_objects(self._bucket, recursive=True))
        except Exception as e:  # noqa: BLE001
            logger.warning("list_objects failed: %s", e)
            return 0
        for o in objs:
            if o.last_modified is None:
                continue
            # last_modified is a datetime; convert to epoch seconds.
            lm_ts = o.last_modified.timestamp()
            if lm_ts < cutoff_ts:
                try:
                    self.client.remove_object(self._bucket, o.object_name)
                    deleted += 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("remove_object %s failed: %s", o.object_name, e)
        return deleted


def from_env() -> MinioStore:
    """Build a ``MinioStore`` from the standard env vars.

    The 3-bucket layout (Phase 4) is::

        MINIO_BUCKET          -> evidence  (legacy alias)
        MINIO_BUCKET_EVIDENCE -> evidence  (preferred)
        MINIO_BUCKET_REPORTS  -> reports
        MINIO_BUCKET_MODELS   -> models

    ``MINIO_CREATE_BUCKETS=true`` enables the dev-only "make bucket
    on first connect" behaviour. Production should leave it unset and
    ensure the buckets exist beforehand.
    """
    bucket = os.environ.get("MINIO_BUCKET_EVIDENCE") or os.environ.get("MINIO_BUCKET") or "evidence"
    return MinioStore(
        endpoint=os.environ.get("MINIO_ENDPOINT", ""),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "change_me_in_production"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "change_me_in_production"),
        secure=os.environ.get("MINIO_SECURE", "false").lower() in {"1", "true", "yes"},
        bucket=bucket,
        bucket_reports=os.environ.get("MINIO_BUCKET_REPORTS", "") or None,
        bucket_models=os.environ.get("MINIO_BUCKET_MODELS", "") or None,
        create_buckets=os.environ.get("MINIO_CREATE_BUCKETS", "false").lower()
        in {"1", "true", "yes"},
    )
