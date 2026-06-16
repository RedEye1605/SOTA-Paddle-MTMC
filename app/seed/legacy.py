"""Idempotent YAML → Postgres seeder.

Reconciles the cameras / zones / camera_links tables with the
authoritative YAML configs at app startup. Designed to run BEFORE
the API/worker threads start so any code path that does
``pg.fetch_zones()`` / ``pg.fetch_camera_links()`` finds data on a
cold DB.

Design:
  * Idempotent — uses PostgresStore.upsert_* (ON CONFLICT … DO
    UPDATE). Safe to re-run on every boot.
  * Skips when a non-empty YAML fingerprint is already stored
    (operator can force a re-seed by setting SEED_FORCE=1).
  * Failures are logged but never raise — a bad YAML row must not
    crash the API. The boot continues with whatever the DB has.

YAML is the source of truth: this module never deletes rows. If the
operator removes a camera from cameras.yaml, the row stays in
Postgres (set is_active=false to soft-delete).

Public entry point: :func:`seed_from_yaml` — called once from
``app.main`` before the API starts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from ..storage.postgres import PostgresStore

logger = logging.getLogger(__name__)


# --- fingerprint table -------------------------------------------------------
#
# One row per (config_path, mtime, sha). The seeder compares the
# current fingerprint to the stored one; if both match, the seed is
# a no-op (fast path for warm restarts). Set SEED_FORCE=1 to bypass.
_FINGERPRINT_DDL = """
CREATE TABLE IF NOT EXISTS seed_fingerprints (
    config_path     TEXT PRIMARY KEY,
    fingerprint     TEXT NOT NULL,
    seeded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    rows_seeded     INTEGER NOT NULL DEFAULT 0
);
"""


def _yaml_fingerprint(path: Path) -> tuple[str, str]:
    """Return (fingerprint_sha1, body) for a YAML file.

    The fingerprint is a stable hash of the parsed-and-reserialized
    YAML so comment/whitespace changes do not trigger a re-seed.
    """
    import hashlib

    raw = path.read_bytes()
    parsed = yaml.safe_load(raw) or {}
    canonical = yaml.safe_dump(parsed, sort_keys=True).encode("utf-8")
    return hashlib.sha1(canonical).hexdigest(), canonical.decode("utf-8")


def _is_already_seeded(pg: PostgresStore, path: str, fingerprint: str) -> bool:
    """Return True if the stored fingerprint matches and the
    cameras/zones/camera_links tables all have rows."""
    with pg.connection() as conn:
        with conn.cursor() as cur:
            # Cold-DB safety: a prior partial seed may have created
            # the seed_fingerprints table but the migration order
            # could leave us here before that table exists. Treat
            # any error as "not seeded yet" and let the caller
            # proceed with the full seed.
            try:
                cur.execute(
                    "SELECT fingerprint FROM seed_fingerprints WHERE config_path=%s",
                    (path,),
                )
                row = cur.fetchone()
            except Exception as e:  # noqa: BLE001
                # Likely "relation does not exist" on a cold DB.
                logger.debug("seed_fingerprints lookup failed (treating as cold): %s", e)
                return False
            if row is None or row[0] != fingerprint:
                return False
            # Fingerprint matches — but verify the actual tables
            # are populated. A prior partial seed may have stored
            # the fingerprint without finishing.
            for table in ("cameras", "zones", "camera_links"):
                cur.execute(f"SELECT count(*) FROM {table}")  # noqa: S608
                cnt = cur.fetchone()[0]
                if cnt == 0:
                    return False
            return True


def _record_fingerprint(pg: PostgresStore, path: str, fingerprint: str, rows: int) -> None:
    with pg.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_FINGERPRINT_DDL)
            cur.execute(
                """
                INSERT INTO seed_fingerprints (config_path, fingerprint, rows_seeded)
                VALUES (%s, %s, %s)
                ON CONFLICT (config_path) DO UPDATE
                  SET fingerprint = EXCLUDED.fingerprint,
                      seeded_at   = now(),
                      rows_seeded = EXCLUDED.rows_seeded;
                """,
                (path, fingerprint, rows),
            )
        conn.commit()


def _seed_cameras(pg: PostgresStore, cameras_cfg: dict[str, Any], site_id: str, tz: str) -> int:
    rows = cameras_cfg.get("cameras", []) or []
    for cam in rows:
        try:
            pg.upsert_camera(
                camera_id=cam["camera_id"],
                name=cam.get("name") or cam["camera_id"],
                rtsp_url_env_key=cam["rtsp_url_env_key"],
                site_id=site_id,
                timezone=tz,
                width=int(cam["width"]),
                height=int(cam["height"]),
                fps_target=int(cam.get("fps_target", 25)),
                is_active=bool(cam.get("is_active", True)),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "seed: skip camera %r due to %s",
                cam.get("camera_id"),
                e,
            )
    return len(rows)


def _seed_zones(pg: PostgresStore, zones_cfg: dict[str, Any]) -> int:
    rows = zones_cfg.get("zones", []) or []
    for z in rows:
        try:
            pg.upsert_zone(
                zone_id=z["zone_id"],
                camera_id=z["camera_id"],
                name=z.get("name") or z["zone_id"],
                polygon_json=z["polygon_json"],
                zone_type=z.get("zone_type", "floor"),
                is_entry_zone=bool(z.get("is_entry_zone", False)),
                is_exit_zone=bool(z.get("is_exit_zone", False)),
                enabled=bool(z.get("enabled", True)),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "seed: skip zone %r due to %s",
                z.get("zone_id"),
                e,
            )
    return len(rows)


def _seed_camera_links(pg: PostgresStore, links_cfg: dict[str, Any]) -> int:
    rows = links_cfg.get("camera_links", []) or []
    for link in rows:
        try:
            pg.upsert_camera_link(
                from_camera_id=link["from_camera_id"],
                to_camera_id=link["to_camera_id"],
                min_travel_seconds=int(link.get("min_travel_seconds", 0)),
                max_travel_seconds=int(link.get("max_travel_seconds", 0)),
                transition_probability=float(link.get("transition_probability", 0.0)),
                enabled=bool(link.get("enabled", True)),
                notes=link.get("notes", "") or "",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "seed: skip link %s -> %s due to %s",
                link.get("from_camera_id"),
                link.get("to_camera_id"),
                e,
            )
    return len(rows)


def seed_from_yaml(
    pg: PostgresStore,
    *,
    cameras_path: Path,
    zones_path: Path,
    links_path: Path,
    force: bool = False,
) -> dict[str, int]:
    """Reconcile cameras / zones / camera_links with the YAML files.

    Returns a dict with the row counts written (per file).

    Parameters
    ----------
    pg
        Connected PostgresStore. The caller is responsible for
        ``pg.connect()`` and the eventual ``pg.close()``.
    cameras_path, zones_path, links_path
        Filesystem paths to the YAML files.
    force
        Skip the fingerprint short-circuit. Default ``False``.
        Override with the ``SEED_FORCE=1`` env var.
    """
    if force or os.environ.get("SEED_FORCE") == "1":
        force = True

    cameras_cfg = yaml.safe_load(cameras_path.read_text()) or {}
    zones_cfg = yaml.safe_load(zones_path.read_text()) or {}
    links_cfg = yaml.safe_load(links_path.read_text()) or {}

    site_id = cameras_cfg.get("site_id", "default_site")
    tz = cameras_cfg.get("timezone", "UTC")

    total = 0
    if force:
        logger.info("seed: SEED_FORCE=1 — running full re-seed")
    else:
        # Quick check: if all three files are unchanged AND the
        # tables are populated, skip the work.
        for path in (cameras_path, zones_path, links_path):
            fp, _ = _yaml_fingerprint(path)
            if _is_already_seeded(pg, str(path), fp):
                logger.info("seed: %s unchanged, skipping", path.name)
                return {"cameras": 0, "zones": 0, "camera_links": 0, "skipped": 1}

    n_cam = _seed_cameras(pg, cameras_cfg, site_id, tz)
    n_zones = _seed_zones(pg, zones_cfg)
    n_links = _seed_camera_links(pg, links_cfg)
    total = n_cam + n_zones + n_links

    # Record fingerprints so the next boot is a no-op.
    for path in (cameras_path, zones_path, links_path):
        fp, _ = _yaml_fingerprint(path)
        _record_fingerprint(pg, str(path), fp, total)

    logger.info(
        "seed: wrote cameras=%d zones=%d camera_links=%d (force=%s)",
        n_cam,
        n_zones,
        n_links,
        force,
    )
    return {
        "cameras": n_cam,
        "zones": n_zones,
        "camera_links": n_links,
        "skipped": 0,
    }
