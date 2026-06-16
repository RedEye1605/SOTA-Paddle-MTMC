"""PostgreSQL — durable source of truth.

Uses ``psycopg_pool`` (psycopg ≥ 3.2). The pool was moved out of the
``psycopg`` package into ``psycopg_pool`` in the 3.2 release; the
``from psycopg import pool`` form is no longer valid. No silent JSON
fallback; if Postgres is disabled in config, the system refuses to start
the identity layer.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


class PostgresStore:
    """Thin wrapper around a psycopg connection pool.

    All public methods are blocking (psycopg is sync). If the workload needs
    async, wrap calls in `asyncio.to_thread`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        pool_size: int = 10,
        max_overflow: int = 20,
    ) -> None:
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool: Optional[ConnectionPool] = None

    # ---- lifecycle ----
    def connect(self) -> None:
        if self._pool is not None:
            return
        conninfo = (
            f"host={self._host} port={self._port} dbname={self._database} "
            f"user={self._user} password={self._password} "
            f"application_name=sota-paddle-mtmct"
        )
        self._pool = ConnectionPool(
            conninfo=conninfo,
            min_size=1,
            max_size=self._pool_size + self._max_overflow,
            kwargs={"row_factory": dict_row},
            open=True,
            timeout=10,
        )
        logger.info(
            "Postgres pool ready: host=%s port=%d db=%s pool=%d",
            self._host,
            self._port,
            self._database,
            self._pool_size,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        assert self._pool is not None, "PostgresStore.connect() first"
        with self._pool.connection() as conn:
            yield conn

    @contextmanager
    def cursor(self) -> Iterator[psycopg.Cursor]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                yield cur

    # ---- helpers ----
    def healthcheck(self) -> bool:
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Postgres healthcheck failed: %s", e)
            return False

    def timed_execute(self, sql: str, params: tuple | dict = ()) -> float:
        """Execute a single statement, returning latency in seconds.

        Also observes the latency in the metrics histogram
        (PATCH-020 / BUG-027). Histogram import is lazy to keep this
        module import-cheap in test contexts.
        """
        start = time.perf_counter()
        with self.cursor() as cur:
            cur.execute(sql, params)
        elapsed = time.perf_counter() - start
        try:
            from ..telemetry.metrics import REGISTRY

            REGISTRY.postgres_write_latency.observe(elapsed)
        except Exception:  # noqa: BLE001
            pass
        return elapsed

    # ---- domain writes ----
    def upsert_camera(
        self,
        camera_id: str,
        name: str,
        rtsp_url_env_key: str,
        site_id: str,
        timezone: str,
        width: int,
        height: int,
        fps_target: int,
        is_active: bool,
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO cameras (camera_id, name, rtsp_url_env_key, site_id, timezone,
                                 width, height, fps_target, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (camera_id) DO UPDATE
              SET name=EXCLUDED.name,
                  rtsp_url_env_key=EXCLUDED.rtsp_url_env_key,
                  site_id=EXCLUDED.site_id,
                  timezone=EXCLUDED.timezone,
                  width=EXCLUDED.width,
                  height=EXCLUDED.height,
                  fps_target=EXCLUDED.fps_target,
                  is_active=EXCLUDED.is_active,
                  updated_at=now();
            """,
            (
                camera_id,
                name,
                rtsp_url_env_key,
                site_id,
                timezone,
                width,
                height,
                fps_target,
                is_active,
            ),
        )

    def upsert_zone(
        self,
        zone_id: str,
        camera_id: str,
        name: str,
        polygon_json: str,
        zone_type: str,
        is_entry_zone: bool,
        is_exit_zone: bool,
        enabled: bool,
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO zones (zone_id, camera_id, name, polygon_json, zone_type,
                               is_entry_zone, is_exit_zone, enabled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (zone_id) DO UPDATE
              SET name=EXCLUDED.name, polygon_json=EXCLUDED.polygon_json,
                  zone_type=EXCLUDED.zone_type,
                  is_entry_zone=EXCLUDED.is_entry_zone,
                  is_exit_zone=EXCLUDED.is_exit_zone,
                  enabled=EXCLUDED.enabled;
            """,
            (
                zone_id,
                camera_id,
                name,
                polygon_json,
                zone_type,
                is_entry_zone,
                is_exit_zone,
                enabled,
            ),
        )

    def upsert_camera_link(
        self,
        from_camera_id: str,
        to_camera_id: str,
        min_travel_seconds: int,
        max_travel_seconds: int,
        transition_probability: float,
        enabled: bool,
        notes: str,
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO camera_links (from_camera_id, to_camera_id,
                                      min_travel_seconds, max_travel_seconds,
                                      transition_probability, enabled, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (from_camera_id, to_camera_id) DO UPDATE
              SET min_travel_seconds=EXCLUDED.min_travel_seconds,
                  max_travel_seconds=EXCLUDED.max_travel_seconds,
                  transition_probability=EXCLUDED.transition_probability,
                  enabled=EXCLUDED.enabled,
                  notes=EXCLUDED.notes;
            """,
            (
                from_camera_id,
                to_camera_id,
                min_travel_seconds,
                max_travel_seconds,
                transition_probability,
                enabled,
                notes,
            ),
        )

    def fetch_camera_links(self) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT from_camera_id, to_camera_id, min_travel_seconds,
                       max_travel_seconds, transition_probability, enabled, notes
                FROM camera_links;
                """
            )
            return list(cur.fetchall())

    def fetch_zones(self) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT zone_id, camera_id, name, polygon_json, zone_type,
                       is_entry_zone, is_exit_zone, enabled
                FROM zones WHERE enabled = TRUE;
                """
            )
            return list(cur.fetchall())

    # ---- identity writes ----
    def create_global_identity(
        self,
        global_id: str,
        session_id: str,
        first_seen_at: float,
        last_seen_at: float,
        first_camera_id: str,
        last_camera_id: str,
        confidence_state: str = "firm",
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO global_identities (global_id, session_id, first_seen_at,
                                            last_seen_at, first_camera_id, last_camera_id,
                                            confidence_state)
            VALUES (%s, %s, to_timestamp(%s), to_timestamp(%s), %s, %s, %s)
            ON CONFLICT (global_id) DO UPDATE
              SET last_seen_at=GREATEST(global_identities.last_seen_at, EXCLUDED.last_seen_at),
                  last_camera_id=EXCLUDED.last_camera_id;
            """,
            (
                global_id,
                session_id,
                first_seen_at,
                last_seen_at,
                first_camera_id,
                last_camera_id,
                confidence_state,
            ),
        )

    def update_global_identity_seen(
        self, global_id: str, last_seen_at: float, last_camera_id: str
    ) -> None:
        self.timed_execute(
            """
            UPDATE global_identities
            SET last_seen_at=GREATEST(last_seen_at, to_timestamp(%s)),
                last_camera_id=%s,
                updated_at=now()
            WHERE global_id=%s;
            """,
            (last_seen_at, last_camera_id, global_id),
        )

    def insert_tracklet(
        self,
        tracklet_id: str,
        global_id: Optional[str],
        camera_id: str,
        local_track_id: int,
        start_time: float,
        end_time: Optional[float],
        start_zone_id: Optional[str],
        end_zone_id: Optional[str],
        best_crop_uri: Optional[str],
        quality_score: Optional[float],
        frame_count: int,
        embedding_count: int,
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO tracklets (tracklet_id, global_id, camera_id, local_track_id,
                                   start_time, end_time, start_zone_id, end_zone_id,
                                   best_crop_uri, quality_score, frame_count, embedding_count)
            VALUES (%s, %s, %s, %s, to_timestamp(%s),
                    CASE WHEN %s IS NULL THEN NULL ELSE to_timestamp(%s) END,
                    %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tracklet_id) DO UPDATE
              SET end_time = COALESCE(EXCLUDED.end_time, tracklets.end_time),
                  end_zone_id = COALESCE(EXCLUDED.end_zone_id, tracklets.end_zone_id),
                  best_crop_uri = COALESCE(EXCLUDED.best_crop_uri, tracklets.best_crop_uri),
                  quality_score = COALESCE(EXCLUDED.quality_score, tracklets.quality_score),
                  embedding_count = EXCLUDED.embedding_count,
                  frame_count = EXCLUDED.frame_count;
            """,
            (
                tracklet_id,
                global_id,
                camera_id,
                local_track_id,
                start_time,
                end_time,
                end_time,
                start_zone_id,
                end_zone_id,
                best_crop_uri,
                quality_score,
                frame_count,
                embedding_count,
            ),
        )

    def update_tracklet_global_id(
        self,
        tracklet_id: str,
        global_id: Optional[str],
    ) -> None:
        """Update ``tracklets.global_id`` after the resolver assigns one.

        Fix for BUG-011 — without this, the joinable ``tracklets.global_id``
        column stays NULL even after the resolver writes
        ``identity_decisions.assigned_global_id``.
        """
        self.timed_execute(
            """
            UPDATE tracklets
            SET global_id = %s, updated_at = now()
            WHERE tracklet_id = %s;
            """,
            (global_id, tracklet_id),
        )

    def insert_tracking_event(
        self,
        tracklet_id: Optional[str],
        global_id: Optional[str],
        camera_id: str,
        ts: float,
        bbox: tuple[float, float, float, float],
        confidence: Optional[float],
        zone_id: Optional[str],
        event_type: str = "detection",
    ) -> None:
        """Persist a single tracking event (per-frame detection or update)."""
        self.timed_execute(
            """
            INSERT INTO tracking_events (
                tracklet_id, global_id, camera_id, "timestamp",
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                confidence, zone_id, event_type
            ) VALUES (%s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                tracklet_id,
                global_id,
                camera_id,
                ts,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                confidence,
                zone_id,
                event_type,
            ),
        )

    def count_tracking_events_older_than(self, cutoff_ts: float) -> int:
        """Count tracking_events older than `cutoff_ts` (used by retention)."""
        with self.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS n FROM tracking_events
                WHERE "timestamp" < to_timestamp(%s);
                """,
                (cutoff_ts,),
            )
            row = cur.fetchone()
            return int(row["n"]) if row else 0

    def delete_tracking_events_older_than(self, cutoff_ts: float) -> int:
        with self.cursor() as cur:
            cur.execute(
                """
                DELETE FROM tracking_events
                WHERE "timestamp" < to_timestamp(%s);
                """,
                (cutoff_ts,),
            )
            return cur.rowcount

    def insert_tracklet_embedding(
        self,
        tracklet_id: str,
        global_id: Optional[str],
        camera_id: str,
        model_name: str,
        model_version: str,
        vector_db_collection: str,
        vector_db_point_id: str,
        embedding_dim: int,
        quality_score: Optional[float],
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO tracklet_embeddings (tracklet_id, global_id, camera_id,
                                            model_name, model_version,
                                            vector_db_collection, vector_db_point_id,
                                            embedding_dim, quality_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (vector_db_collection, vector_db_point_id) DO NOTHING;
            """,
            (
                tracklet_id,
                global_id,
                camera_id,
                model_name,
                model_version,
                vector_db_collection,
                vector_db_point_id,
                embedding_dim,
                quality_score,
            ),
        )

    def insert_identity_decision(
        self,
        tracklet_id: str,
        source_camera_id: str,
        candidate_camera_id: Optional[str],
        assigned_global_id: Optional[str],
        decision_type: str,
        top1_global_id: Optional[str],
        top1_camera_id: Optional[str],
        top1_score: Optional[float],
        top2_global_id: Optional[str],
        top2_camera_id: Optional[str],
        top2_score: Optional[float],
        reid_similarity: Optional[float],
        temporal_score: Optional[float],
        camera_topology_score: Optional[float],
        quality_score: Optional[float],
        zone_score: Optional[float],
        final_score: Optional[float],
        reason: str,
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO identity_decisions (
                tracklet_id, source_camera_id, candidate_camera_id,
                assigned_global_id, decision_type,
                top1_global_id, top1_camera_id, top1_score,
                top2_global_id, top2_camera_id, top2_score,
                reid_similarity, temporal_score, camera_topology_score,
                quality_score, zone_score, final_score, reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                tracklet_id,
                source_camera_id,
                candidate_camera_id,
                assigned_global_id,
                decision_type,
                top1_global_id,
                top1_camera_id,
                top1_score,
                top2_global_id,
                top2_camera_id,
                top2_score,
                reid_similarity,
                temporal_score,
                camera_topology_score,
                quality_score,
                zone_score,
                final_score,
                reason,
            ),
        )

    def insert_zone_event(
        self,
        global_id: str,
        tracklet_id: str,
        camera_id: str,
        zone_id: str,
        event_type: str,
        ts: float,
        confidence: Optional[float],
    ) -> None:
        self.timed_execute(
            """
            INSERT INTO zone_events (global_id, tracklet_id, camera_id, zone_id,
                                      event_type, "timestamp", confidence)
            VALUES (%s, %s, %s, %s, %s, to_timestamp(%s), %s);
            """,
            (global_id, tracklet_id, camera_id, zone_id, event_type, ts, confidence),
        )

    def upsert_dwell(
        self,
        global_id: str,
        zone_id: str,
        camera_id: str,
        ts: float,
        event_type: str,
    ) -> None:
        """event_type is 'enter' (open dwell) or 'exit' (close it)."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                if event_type == "enter":
                    cur.execute(
                        """
                        INSERT INTO dwell_sessions (global_id, zone_id, camera_id,
                                                    entered_at, status)
                        VALUES (%s, %s, %s, to_timestamp(%s), 'open')
                        ON CONFLICT DO NOTHING;
                        """,
                        (global_id, zone_id, camera_id, ts),
                    )
                else:  # exit
                    cur.execute(
                        """
                        UPDATE dwell_sessions
                        SET exited_at = to_timestamp(%s),
                            duration_seconds = EXTRACT(EPOCH FROM (to_timestamp(%s) - entered_at))::int,
                            status = 'closed'
                        WHERE global_id = %s
                          AND zone_id = %s
                          AND status = 'open';
                        """,
                        (ts, ts, global_id, zone_id),
                    )

    def expire_old_identities(self, older_than_seconds: int) -> int:
        """Mark identities older than the window as 'expired'. Returns count."""
        with self.cursor() as cur:
            cur.execute(
                """
                UPDATE global_identities
                SET status='expired', updated_at=now()
                WHERE status='active'
                  AND last_seen_at < (now() - make_interval(secs => %s));
                """,
                (older_than_seconds,),
            )
            return cur.rowcount


def from_env() -> PostgresStore:
    return PostgresStore(
        host=os.environ.get("POSTGRES_HOST", "relation-store"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "yamaha_mtmct"),
        user=os.environ.get("POSTGRES_USER", "yamaha"),
        password=os.environ.get("POSTGRES_PASSWORD", "change_me_in_production"),
        pool_size=int(os.environ.get("POSTGRES_POOL_SIZE", "10")),
        max_overflow=int(os.environ.get("POSTGRES_POOL_MAX_OVERFLOW", "20")),
    )
