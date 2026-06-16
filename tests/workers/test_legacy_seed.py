"""Tests for app/seed/legacy.py — idempotent YAML → Postgres seeder.

Uses a mock PostgresStore so no live database is required. Pins down
the public contract:

  * seed_from_yaml calls upsert_camera/upsert_zone/upsert_camera_link
    in order with the right field mapping.
  * It is safe to re-run: the upserts are ON CONFLICT DO UPDATE.
  * The fingerprint short-circuit is hit only when both the file
    fingerprint matches AND the tables are populated.
  * A bad row in YAML is logged and skipped — never crashes the seed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


from app.seed import legacy as seed_legacy


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(body)


def _make_pg_with_fingerprint(row: tuple | None) -> MagicMock:
    """Build a MagicMock PostgresStore with a properly chained
    connection/cursor so fetchone side-effects work.

    The ``row`` argument is what the FIRST ``fetchone()`` call
    returns — i.e. the value of the
    ``SELECT fingerprint FROM seed_fingerprints WHERE config_path=%s``
    query. Pass ``None`` to simulate "no row stored yet" (cold boot);
    pass ``("any",)`` to simulate a stored fingerprint (warm boot).

    NB: the seeder uses ``with conn.cursor() as cur:`` which means
    the function's local ``cur`` is whatever ``cur.__enter__()``
    returns, not the cursor object itself. We must make the cursor
    act as a self-returning context manager.
    """
    pg = MagicMock(name="pg")
    conn = MagicMock(name="conn")
    cur = MagicMock(name="cur")
    pg.connection.return_value.__enter__.return_value = conn
    conn.cursor.return_value = cur
    cur.__enter__.return_value = cur  # self-returning ctx mgr
    cur.fetchone.return_value = row
    return pg


def test_seed_writes_cameras_zones_and_links(tmp_path: Path) -> None:
    cameras = tmp_path / "cameras.yaml"
    zones = tmp_path / "zones.yaml"
    links = tmp_path / "links.yaml"
    _write_yaml(
        cameras,
        """
site_id: test_site
timezone: Asia/Jakarta
cameras:
  - camera_id: CAM_01
    name: Entrance
    rtsp_url_env_key: CAM_01_RTSP_URL
    width: 1920
    height: 1080
    fps_target: 25
    is_active: true
  - camera_id: CAM_02
    name: Floor
    rtsp_url_env_key: CAM_02_RTSP_URL
    width: 1280
    height: 720
    fps_target: 15
    is_active: true
""",
    )
    _write_yaml(
        zones,
        """
zones:
  - zone_id: Z01
    camera_id: CAM_01
    name: Entry
    polygon_json: '[[0,0],[1,0],[1,1],[0,1]]'
    zone_type: entry
    is_entry_zone: true
    is_exit_zone: false
    enabled: true
""",
    )
    _write_yaml(
        links,
        """
camera_links:
  - from_camera_id: CAM_01
    to_camera_id: CAM_02
    min_travel_seconds: 10
    max_travel_seconds: 90
    transition_probability: 0.85
    enabled: true
    notes: ""
""",
    )

    pg = MagicMock()
    pg = _make_pg_with_fingerprint(None)

    counts = seed_legacy.seed_from_yaml(
        pg, cameras_path=cameras, zones_path=zones, links_path=links
    )

    assert counts == {"cameras": 2, "zones": 1, "camera_links": 1, "skipped": 0}
    assert pg.upsert_camera.call_count == 2
    assert pg.upsert_zone.call_count == 1
    assert pg.upsert_camera_link.call_count == 1

    # Spot-check the field mapping on the first camera call.
    cam_call = pg.upsert_camera.call_args_list[0]
    assert cam_call.kwargs["camera_id"] == "CAM_01"
    assert cam_call.kwargs["site_id"] == "test_site"
    assert cam_call.kwargs["timezone"] == "Asia/Jakarta"
    assert cam_call.kwargs["width"] == 1920
    assert cam_call.kwargs["height"] == 1080
    assert cam_call.kwargs["fps_target"] == 25
    assert cam_call.kwargs["is_active"] is True


def test_seed_is_idempotent_when_fingerprint_matches(tmp_path: Path) -> None:
    """Warm restart: fingerprint stored AND tables populated → no work."""
    cameras = tmp_path / "cameras.yaml"
    zones = tmp_path / "zones.yaml"
    links = tmp_path / "links.yaml"
    _write_yaml(
        cameras,
        """
site_id: x
timezone: UTC
cameras:
  - camera_id: C1
    name: C1
    rtsp_url_env_key: K
    width: 1
    height: 1
    fps_target: 1
    is_active: true
""",
    )
    _write_yaml(zones, "zones: []\n")
    _write_yaml(links, "camera_links: []\n")

    pg = MagicMock()
    conn = MagicMock()
    cur = MagicMock()
    pg.connection.return_value.__enter__.return_value = conn
    conn.cursor.return_value = cur
    cur.__enter__.return_value = cur  # self-returning ctx mgr
    # First call (cameras.yaml): return matching fingerprint.
    # Subsequent count() calls return >0 to signal populated tables.
    responses = [
        (seed_legacy._yaml_fingerprint(cameras)[0],),  # stored fingerprint
        (5,),  # count(cameras)
        (3,),  # count(zones)
        (2,),  # count(camera_links)
    ]
    cur.fetchone.side_effect = responses

    counts = seed_legacy.seed_from_yaml(
        pg, cameras_path=cameras, zones_path=zones, links_path=links
    )

    # Skipped path returns the sentinel dict.
    assert counts == {"cameras": 0, "zones": 0, "camera_links": 0, "skipped": 1}
    assert pg.upsert_camera.call_count == 0
    assert pg.upsert_zone.call_count == 0
    assert pg.upsert_camera_link.call_count == 0


def test_seed_skips_bad_row_without_crashing(tmp_path: Path, caplog) -> None:
    """A malformed YAML row must be logged and skipped, not raise."""
    cameras = tmp_path / "cameras.yaml"
    zones = tmp_path / "zones.yaml"
    links = tmp_path / "links.yaml"
    _write_yaml(
        cameras,
        """
site_id: x
timezone: UTC
cameras:
  - camera_id: CAM_OK
    name: ok
    rtsp_url_env_key: K
    width: 1
    height: 1
    fps_target: 1
    is_active: true
  - camera_id: CAM_BAD
    name: bad
    # missing rtsp_url_env_key → will raise KeyError
    width: 1
    height: 1
    fps_target: 1
    is_active: true
""",
    )
    _write_yaml(zones, "zones: []\n")
    _write_yaml(links, "camera_links: []\n")

    pg = _make_pg_with_fingerprint(None)

    counts = seed_legacy.seed_from_yaml(
        pg, cameras_path=cameras, zones_path=zones, links_path=links
    )

    # Only the good row was upserted; the bad one was logged and skipped.
    assert counts["cameras"] == 2  # total YAML rows (good + bad)
    assert pg.upsert_camera.call_count == 1  # only the good one
    assert any("skip camera" in r.message for r in caplog.records)


def test_force_flag_bypasses_fingerprint_check(tmp_path: Path) -> None:
    cameras = tmp_path / "cameras.yaml"
    zones = tmp_path / "zones.yaml"
    links = tmp_path / "links.yaml"
    _write_yaml(
        cameras,
        """
site_id: x
timezone: UTC
cameras:
  - camera_id: C1
    name: c1
    rtsp_url_env_key: K
    width: 1
    height: 1
    fps_target: 1
    is_active: true
""",
    )
    _write_yaml(zones, "zones: []\n")
    _write_yaml(links, "camera_links: []\n")

    pg = _make_pg_with_fingerprint(("any",))

    counts = seed_legacy.seed_from_yaml(
        pg,
        cameras_path=cameras,
        zones_path=zones,
        links_path=links,
        force=True,
    )

    assert counts["cameras"] == 1
    assert pg.upsert_camera.call_count == 1
