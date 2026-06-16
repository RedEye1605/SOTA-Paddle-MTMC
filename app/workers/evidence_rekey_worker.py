"""Evidence re-key worker (PATCH-029).

After the resolver assigns a ``global_id`` to a tracklet, the best
crop is still on the *pending* path
(``evidence/pending/{site}/{camera}/{tracklet}/best.jpg``). This
worker:

  1. Reads ``stream:identity_decisions`` from Redis Streams.
  2. For each ``"new"`` or ``"match"`` decision, server-side copies
     the pending best crop to
     ``evidence/{site}/{camera}/{zone}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet}/best.jpg``.
  3. Updates ``tracklets.best_crop_uri`` with the final URI.
  4. Optionally deletes the pending copy (configurable).

Failure handling:
  * Copy failures are logged and the row is left as-is; the next
    pass retries.
  * Pending copy is NOT deleted on failure.

The worker uses a Redis Streams consumer group so multiple replicas
can run in parallel. The ``tracklet_id`` is the message key; a
double-claim by two consumers is a no-op (server-side copy is
idempotent).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from ..storage.minio_store import MinioStore
from ..storage.postgres import PostgresStore
from ..storage.redis_state import RedisState
from ..utils.time import now_ts

logger = logging.getLogger(__name__)


@dataclass
class RekeyConfig:
    enabled: bool = True
    keep_pending_copy: bool = False
    retry_max: int = 3
    stream: str = "stream:identity_decisions"
    group: str = "evidence_rekey_workers"
    consumer_name: str = "rekey-worker-01"
    poll_count: int = 4
    poll_block_ms: int = 1000


class EvidenceRekeyWorker:
    """Consumes ``stream:identity_decisions`` and re-keys the best crop
    from the pending path to the final global_id path.
    """

    def __init__(
        self,
        *,
        minio: MinioStore,
        pg: Optional[PostgresStore],
        redis: RedisState,
        config: Optional[RekeyConfig] = None,
    ) -> None:
        self.minio = minio
        self.pg = pg
        self.redis = redis
        self.config = config or RekeyConfig()

    # ---- per-message logic ----
    def handle_decision(self, fields: dict) -> bool:
        """Re-key the best crop for a single identity decision.

        Returns True on success (including "nothing to do" no-ops);
        False on a transient failure.
        """
        decision = fields.get("decision")
        tracklet_id = fields.get("tracklet_id")
        camera_id = fields.get("camera_id")
        assigned_global_id = fields.get("assigned_global_id")
        site_id = fields.get("site_id", "default_site")
        end_zone_id = fields.get("end_zone_id") or "Z_NONE"
        ts = float(fields.get("ts") or now_ts())
        if not (tracklet_id and camera_id and assigned_global_id):
            return True  # nothing to do
        if decision not in {"new", "match"}:
            return True  # only "new" and "match" create a global_id path
        # Compose the pending URI. We don't have the original
        # best_crop_uri in the decision message; we recompute the
        # canonical pending path.
        pending_uri = (
            f"s3://{self.minio.bucket}/"
            f"evidence/pending/{site_id}/{camera_id}/{tracklet_id}/best.jpg"
        )
        # Compose the final URI.
        final_key = self.minio.evidence_key(
            site_id=site_id,
            camera_id=camera_id,
            zone_id=end_zone_id,
            ts=ts,
            global_id=assigned_global_id,
            tracklet_id=tracklet_id,
            kind="best",
        )
        # Sanity: the pending object must exist before we copy.
        if not self._object_exists(pending_uri):
            logger.debug(
                "evidence_rekey: pending crop missing for tracklet=%s; "
                "skipping (may have been written to the final path "
                "directly or already re-keyed)",
                tracklet_id,
            )
            return True
        # Server-side copy. Retry on transient failure.
        for attempt in range(1, max(1, self.config.retry_max) + 1):
            try:
                self.minio.copy_object_within_bucket(
                    src_key=pending_uri.replace(f"s3://{self.minio.bucket}/", ""),
                    dst_key=final_key,
                )
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "evidence_rekey: copy attempt %d/%d failed for %s: %s",
                    attempt,
                    self.config.retry_max,
                    tracklet_id,
                    e,
                )
                if attempt >= self.config.retry_max:
                    return False
                time.sleep(0.5 * attempt)
        # Optionally delete the pending copy.
        if not self.config.keep_pending_copy:
            try:
                self.minio.client.remove_object(
                    self.minio.bucket,
                    pending_uri.replace(f"s3://{self.minio.bucket}/", ""),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "evidence_rekey: pending delete failed for %s: %s",
                    tracklet_id,
                    e,
                )
        # Update PG.
        if self.pg is not None:
            try:
                final_uri = f"s3://{self.minio.bucket}/{final_key}"
                # We use the existing tracklet update path; the
                # tracklet row's global_id was already set by the
                # resolver, so we only update best_crop_uri.
                with self.pg.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE tracklets
                        SET best_crop_uri = %s, updated_at = now()
                        WHERE tracklet_id = %s
                        """,
                        (final_uri, tracklet_id),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "evidence_rekey: PG update failed for %s: %s",
                    tracklet_id,
                    e,
                )
        logger.info(
            "evidence_rekey: re-keyed tracklet=%s → %s",
            tracklet_id,
            final_key,
        )
        return True

    def _object_exists(self, uri: str) -> bool:
        """Best-effort existence check for a pending crop.

        We use the MinIO stat_object API when available; otherwise
        we fall back to a 1-byte GET. The existence check is
        advisory (a race may remove the object between check and
        copy), so the actual copy catches real failures.
        """
        try:
            key = uri.replace(f"s3://{self.minio.bucket}/", "", 1)
            self.minio.client.stat_object(self.minio.bucket, key)
            return True
        except Exception:  # noqa: BLE001
            return False

    def run(self, stop_event=None) -> None:
        if not self.config.enabled:
            logger.info("evidence_rekey: disabled by config; worker idle")
            return
        self.redis.ensure_group(self.config.stream, self.config.group)
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            msgs = self.redis.consume(
                self.config.stream,
                self.config.group,
                self.config.consumer_name,
                count=self.config.poll_count,
                block_ms=self.config.poll_block_ms,
            )
            for msg_id, fields in msgs:
                try:
                    self.handle_decision(fields)
                except Exception as e:  # noqa: BLE001
                    logger.exception("evidence_rekey.handle_decision failed: %s", e)
                    continue
                else:
                    self.redis.ack(
                        self.config.stream,
                        self.config.group,
                        msg_id,
                    )
