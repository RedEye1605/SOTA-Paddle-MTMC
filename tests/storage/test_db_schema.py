"""DB schema validation — every required table is present in the migrations."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "db" / "migrations"

REQUIRED_TABLES = [
    "cameras",
    "camera_links",
    "zones",
    "global_identities",
    "tracklets",
    "tracklet_embeddings",
    "tracking_events",
    "zone_events",
    "dwell_sessions",
    "identity_decisions",
    "identity_merge_audit",
    "model_versions",
    "system_metrics",
]


def _migration_files() -> list[Path]:
    return sorted(MIGRATIONS.glob("*.sql"))


def _all_sql_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _migration_files())


def test_all_required_tables_present() -> None:
    text = _all_sql_text()
    for table in REQUIRED_TABLES:
        # accept "CREATE TABLE <name>" or "CREATE TABLE IF NOT EXISTS <name>"
        pat = rf"\bCREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+{re.escape(table)}\b"
        assert re.search(pat, text), f"Missing required table in migrations: {table}"


def test_cameras_table_has_required_columns() -> None:
    text = _all_sql_text()
    block = _extract_create("cameras", text)
    for col in [
        "camera_id",
        "name",
        "rtsp_url_env_key",
        "site_id",
        "timezone",
        "width",
        "height",
        "fps_target",
        "is_active",
        "created_at",
        "updated_at",
    ]:
        assert col in block, f"cameras.{col} missing"


def test_global_identities_table_has_required_columns() -> None:
    text = _all_sql_text()
    block = _extract_create("global_identities", text)
    for col in [
        "global_id",
        "session_id",
        "first_seen_at",
        "last_seen_at",
        "first_camera_id",
        "last_camera_id",
        "status",
        "confidence_state",
        "created_at",
        "updated_at",
    ]:
        assert col in block, f"global_identities.{col} missing"


def test_identity_decisions_table_has_required_columns() -> None:
    text = _all_sql_text()
    block = _extract_create("identity_decisions", text)
    for col in [
        "decision_id",
        "tracklet_id",
        "source_camera_id",
        "candidate_camera_id",
        "assigned_global_id",
        "decision_type",
        "top1_global_id",
        "top1_camera_id",
        "top1_score",
        "top2_global_id",
        "top2_camera_id",
        "top2_score",
        "reid_similarity",
        "temporal_score",
        "camera_topology_score",
        "quality_score",
        "zone_score",
        "final_score",
        "reason",
        "created_at",
    ]:
        assert col in block, f"identity_decisions.{col} missing"


def test_camera_links_disables_impossible_transitions() -> None:
    """The sample seed data MUST contain an `enabled = false` row for
    impossible transitions (e.g. CAM_01 -> CAM_04)."""
    text = (ROOT / "db" / "seed" / "camera_links.sample.sql").read_text()
    assert "FALSE" in text, (
        "Sample camera_links must include impossible transitions (enabled=FALSE)"
    )


def _extract_create(table: str, text: str) -> str:
    """Naive extract: return the CREATE TABLE block for `table`."""
    pat = re.compile(
        rf"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+{re.escape(table)}\s*\((.*?)\);",
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(text)
    if not m:
        return ""
    return m.group(1)
