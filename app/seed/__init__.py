"""App-side wrapper for the YAML → Postgres seeder.

Lives in ``app/`` (not ``db/``) so the seeder module can import
``PostgresStore`` cleanly. The seeder implementation is
:mod:`app.seed.legacy`; this module is the thin caller that
``app.main`` uses.

Called once from ``app.main`` before the API / workers boot.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..storage.postgres import PostgresStore
from .legacy import seed_from_yaml

logger = logging.getLogger(__name__)


def seed_legacy_topology(
    pg: PostgresStore,
    *,
    repo_root: Path,
) -> dict[str, int]:
    """Read configs/{cameras,zones,camera_links}.yaml and reconcile
    the DB. Idempotent. Returns row counts (zeros if skipped)."""
    cameras_path = repo_root / "configs" / "cameras.yaml"
    zones_path = repo_root / "configs" / "zones.yaml"
    links_path = repo_root / "configs" / "camera_links.yaml"
    for p in (cameras_path, zones_path, links_path):
        if not p.exists():
            logger.warning("seed: missing %s; skipping", p)
            return {"cameras": 0, "zones": 0, "camera_links": 0, "skipped": 1}

    force = os.environ.get("SEED_FORCE") == "1"
    return seed_from_yaml(
        pg,
        cameras_path=cameras_path,
        zones_path=zones_path,
        links_path=links_path,
        force=force,
    )
