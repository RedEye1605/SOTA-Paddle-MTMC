#!/usr/bin/env python3
"""Retention worker — periodically clean up expired PG / Qdrant / MinIO data.

PATCH-015 fix. The worker reads its config from env (with sensible
defaults) and runs forever, calling the storage-layer retention
methods at the configured interval.

Environment variables (all optional):

  ``RETENTION_IDENTITY_WINDOW_SECONDS``  (default 86400)
  ``RETENTION_QDRANT_VECTOR_SECONDS``    (default 86400)
  ``RETENTION_TRACKING_EVENT_DAYS``      (default 7)
  ``RETENTION_CROP_RETENTION_DAYS``      (default 7)
  ``RETENTION_AUDIT_RETENTION_DAYS``     (default 30)
  ``RETENTION_RUN_INTERVAL_SECONDS``     (default 3600 = 1 hour)
  ``RETENTION_DRY_RUN``                  (default false; if true, log
                                          the counts but do not delete)

Usage:

    python -m scripts.retention_worker
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from app.storage.minio_store import MinioStore, from_env as minio_from_env
from app.storage.postgres import PostgresStore, from_env as pg_from_env
from app.storage.qdrant_store import QdrantStore, from_env as qdrant_from_env

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
)
log = logging.getLogger("retention_worker")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def run_once(
    *,
    pg: PostgresStore,
    qdrant: QdrantStore,
    minio: MinioStore,
    identity_window_seconds: int,
    qdrant_vector_seconds: int,
    tracking_event_days: int,
    crop_retention_days: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Run one retention pass and return a summary dict."""
    now = time.time()
    summary: dict[str, Any] = {"started_at": now, "dry_run": dry_run}
    # ---- PG: expire old identities ----
    expired = pg.expire_old_identities(older_than_seconds=identity_window_seconds)
    summary["pg_identities_expired"] = expired
    # ---- PG: delete old tracking_events ----
    if not dry_run:
        deleted = pg.delete_tracking_events_older_than(
            cutoff_ts=now - tracking_event_days * 86_400,
        )
        summary["pg_tracking_events_deleted"] = deleted
    # ---- Qdrant: delete points older than qdrant_vector_seconds ----
    cutoff_q = int(now - qdrant_vector_seconds)
    # PATCH (2026-06-17, operator spec, transreid-only): the only
    # active Qdrant collection is the TransReID MSMT17 one. The
    # previous pphuman / vanilla-transreid / clipreid collections
    # were dropped per the operator spec. We sweep the single
    # remaining collection here. The collection name is sourced from
    # ``app.storage.qdrant_store.COLLECTIONS`` so any future
    # collection addition is picked up automatically.
    from app.storage.qdrant_store import COLLECTIONS

    for col_name, _dim, _dist in COLLECTIONS:
        try:
            cnt = qdrant.count_points_older_than(col_name, cutoff_q)
            summary.setdefault("qdrant_old_points", {})[col_name] = cnt
            if not dry_run:
                qdrant.delete_points_older_than(col_name, cutoff_q)
        except Exception as e:  # noqa: BLE001
            log.warning("qdrant retention failed on %s: %s", col_name, e)
    # ---- MinIO: delete old objects ----
    if not dry_run:
        try:
            deleted_objs = minio.delete_older_than(
                cutoff_ts=now - crop_retention_days * 86_400,
            )
            summary["minio_objects_deleted"] = deleted_objs
        except Exception as e:  # noqa: BLE001
            log.warning("minio retention failed: %s", e)
    summary["finished_at"] = time.time()
    return summary


def main() -> int:
    interval_s = _env_int("RETENTION_RUN_INTERVAL_SECONDS", 3600)
    dry_run = _env_bool("RETENTION_DRY_RUN", False)
    pg = pg_from_env()
    pg.connect()
    qdrant = qdrant_from_env()
    qdrant.connect()
    minio = minio_from_env()
    minio.connect()
    log.info(
        "Retention worker starting: interval=%ds dry_run=%s",
        interval_s,
        dry_run,
    )
    while True:
        try:
            summary = run_once(
                pg=pg,
                qdrant=qdrant,
                minio=minio,
                identity_window_seconds=_env_int("RETENTION_IDENTITY_WINDOW_SECONDS", 86400),
                qdrant_vector_seconds=_env_int("RETENTION_QDRANT_VECTOR_SECONDS", 86400),
                tracking_event_days=_env_int("RETENTION_TRACKING_EVENT_DAYS", 7),
                crop_retention_days=_env_int("RETENTION_CROP_RETENTION_DAYS", 7),
                dry_run=dry_run,
            )
            log.info("retention summary: %s", json.dumps(summary, indent=2))
        except Exception as e:  # noqa: BLE001
            log.exception("retention pass failed: %s", e)
        time.sleep(interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
