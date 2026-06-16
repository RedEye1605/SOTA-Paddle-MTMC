-- =============================================================================
-- 003_zone_tables.sql — zones, camera_links, zone_events, dwell_sessions
-- =============================================================================

-- camera_links: physical topology (also drives cross-camera gating)
CREATE TABLE IF NOT EXISTS camera_links (
    link_id                SERIAL PRIMARY KEY,
    from_camera_id         TEXT NOT NULL REFERENCES cameras(camera_id),
    to_camera_id           TEXT NOT NULL REFERENCES cameras(camera_id),
    min_travel_seconds     INTEGER NOT NULL,
    max_travel_seconds     INTEGER NOT NULL,
    transition_probability DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    enabled                BOOLEAN NOT NULL DEFAULT TRUE,
    notes                  TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (from_camera_id, to_camera_id)
);

CREATE INDEX IF NOT EXISTS idx_cl_from ON camera_links(from_camera_id);
CREATE INDEX IF NOT EXISTS idx_cl_to   ON camera_links(to_camera_id);

-- zones: polygon ROIs per camera
CREATE TABLE IF NOT EXISTS zones (
    zone_id          TEXT PRIMARY KEY,
    camera_id        TEXT NOT NULL REFERENCES cameras(camera_id),
    name             TEXT NOT NULL,
    polygon_json     TEXT NOT NULL,           -- JSON string: [[x,y], ...]
    zone_type        TEXT NOT NULL,           -- entry | exit | floor | display | vip | workshop | ...
    is_entry_zone    BOOLEAN NOT NULL DEFAULT FALSE,
    is_exit_zone     BOOLEAN NOT NULL DEFAULT FALSE,
    enabled          BOOLEAN NOT NULL DEFAULT TRUE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);

-- zone_events: every entry / exit
CREATE TABLE IF NOT EXISTS zone_events (
    zone_event_id    BIGSERIAL PRIMARY KEY,
    global_id        TEXT NOT NULL REFERENCES global_identities(global_id) ON DELETE CASCADE,
    tracklet_id      TEXT NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    camera_id        TEXT NOT NULL REFERENCES cameras(camera_id),
    zone_id          TEXT NOT NULL REFERENCES zones(zone_id),
    event_type       TEXT NOT NULL,           -- enter | exit
    "timestamp"      TIMESTAMPTZ NOT NULL,
    confidence       DOUBLE PRECISION,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ze_global_time ON zone_events(global_id, "timestamp" DESC);
CREATE INDEX IF NOT EXISTS idx_ze_zone_time   ON zone_events(zone_id, "timestamp" DESC);
CREATE INDEX IF NOT EXISTS idx_ze_camera_time ON zone_events(camera_id, "timestamp" DESC);

-- dwell_sessions: time-in-zone aggregations
CREATE TABLE IF NOT EXISTS dwell_sessions (
    dwell_id          BIGSERIAL PRIMARY KEY,
    global_id         TEXT NOT NULL REFERENCES global_identities(global_id) ON DELETE CASCADE,
    zone_id           TEXT NOT NULL REFERENCES zones(zone_id),
    camera_id         TEXT NOT NULL REFERENCES cameras(camera_id),
    entered_at        TIMESTAMPTZ NOT NULL,
    exited_at         TIMESTAMPTZ,
    duration_seconds  INTEGER,                -- NULL while open
    status            TEXT NOT NULL DEFAULT 'open',  -- open | closed
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dw_global  ON dwell_sessions(global_id);
CREATE INDEX IF NOT EXISTS idx_dw_zone    ON dwell_sessions(zone_id);
CREATE INDEX IF NOT EXISTS idx_dw_status  ON dwell_sessions(status);
CREATE INDEX IF NOT EXISTS idx_dw_open    ON dwell_sessions(global_id, status) WHERE status = 'open';
