-- =============================================================================
-- 002_identity_tables.sql — global_identities, tracklets, tracklet_embeddings,
--                            identity_decisions, identity_merge_audit
-- =============================================================================

-- global_identities: persistent 24h identity across all cameras
CREATE TABLE IF NOT EXISTS global_identities (
    global_id           TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,                 -- 24h rolling window
    first_seen_at       TIMESTAMPTZ NOT NULL,
    last_seen_at        TIMESTAMPTZ NOT NULL,
    first_camera_id     TEXT NOT NULL REFERENCES cameras(camera_id),
    last_camera_id      TEXT NOT NULL REFERENCES cameras(camera_id),
    status              TEXT NOT NULL DEFAULT 'active', -- active | merged | expired
    confidence_state    TEXT NOT NULL DEFAULT 'firm',   -- firm | ambiguous | held
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_status CHECK (status IN ('active', 'merged', 'expired')),
    CONSTRAINT chk_confidence CHECK (confidence_state IN ('firm', 'ambiguous', 'held'))
);

CREATE INDEX IF NOT EXISTS idx_gi_session_id   ON global_identities(session_id);
CREATE INDEX IF NOT EXISTS idx_gi_last_seen    ON global_identities(last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_gi_first_cam    ON global_identities(first_camera_id);
CREATE INDEX IF NOT EXISTS idx_gi_last_cam     ON global_identities(last_camera_id);
CREATE INDEX IF NOT EXISTS idx_gi_status       ON global_identities(status);

-- tracklets: stable local-track-derived groups (1 local track → 1+ tracklets
-- over time; a tracklet is closed when the local track is lost).
CREATE TABLE IF NOT EXISTS tracklets (
    tracklet_id         TEXT PRIMARY KEY,
    global_id           TEXT REFERENCES global_identities(global_id) ON DELETE SET NULL,
    camera_id           TEXT NOT NULL REFERENCES cameras(camera_id),
    local_track_id      INTEGER NOT NULL,
    start_time          TIMESTAMPTZ NOT NULL,
    end_time            TIMESTAMPTZ,
    start_zone_id       TEXT,
    end_zone_id         TEXT,
    best_crop_uri       TEXT,
    quality_score       DOUBLE PRECISION,
    frame_count         INTEGER NOT NULL DEFAULT 0,
    embedding_count     INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (camera_id, local_track_id, start_time)
);

CREATE INDEX IF NOT EXISTS idx_tracklets_global_id   ON tracklets(global_id);
CREATE INDEX IF NOT EXISTS idx_tracklets_camera_time ON tracklets(camera_id, start_time);
CREATE INDEX IF NOT EXISTS idx_tracklets_end_time    ON tracklets(end_time);

-- tracklet_embeddings: one row per upserted Qdrant point (audit trail)
CREATE TABLE IF NOT EXISTS tracklet_embeddings (
    embedding_id        BIGSERIAL PRIMARY KEY,
    tracklet_id         TEXT NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    global_id           TEXT REFERENCES global_identities(global_id) ON DELETE SET NULL,
    camera_id           TEXT NOT NULL REFERENCES cameras(camera_id),
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    vector_db_collection TEXT NOT NULL,
    vector_db_point_id  TEXT NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    quality_score       DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (vector_db_collection, vector_db_point_id)
);

CREATE INDEX IF NOT EXISTS idx_te_global_id   ON tracklet_embeddings(global_id);
CREATE INDEX IF NOT EXISTS idx_te_tracklet    ON tracklet_embeddings(tracklet_id);
CREATE INDEX IF NOT EXISTS idx_te_model       ON tracklet_embeddings(model_name, model_version);

-- tracking_events: raw per-frame bounding-box telemetry (audit)
CREATE TABLE IF NOT EXISTS tracking_events (
    event_id          BIGSERIAL PRIMARY KEY,
    tracklet_id       TEXT REFERENCES tracklets(tracklet_id) ON DELETE SET NULL,
    global_id         TEXT REFERENCES global_identities(global_id) ON DELETE SET NULL,
    camera_id         TEXT NOT NULL REFERENCES cameras(camera_id),
    "timestamp"       TIMESTAMPTZ NOT NULL,
    bbox_x1           DOUBLE PRECISION NOT NULL,
    bbox_y1           DOUBLE PRECISION NOT NULL,
    bbox_x2           DOUBLE PRECISION NOT NULL,
    bbox_y2           DOUBLE PRECISION NOT NULL,
    confidence        DOUBLE PRECISION,
    -- zone_id is informational only at this layer (no DB-level FK) so
    -- 002 can be applied before 003 creates the zones table. Application
    -- code validates the zone_id against zones() at write time.
    zone_id           TEXT,
    event_type        TEXT NOT NULL DEFAULT 'detection',  -- detection | track_update | etc.
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_te_tracklet_ts ON tracking_events(tracklet_id, "timestamp" DESC);
CREATE INDEX IF NOT EXISTS idx_te_camera_ts   ON tracking_events(camera_id, "timestamp" DESC);

-- identity_decisions: every assignment/hold/reject is logged with full
-- score breakdown (auditability + offline analysis).
CREATE TABLE IF NOT EXISTS identity_decisions (
    decision_id           BIGSERIAL PRIMARY KEY,
    tracklet_id           TEXT NOT NULL REFERENCES tracklets(tracklet_id) ON DELETE CASCADE,
    source_camera_id      TEXT NOT NULL REFERENCES cameras(camera_id),
    candidate_camera_id   TEXT REFERENCES cameras(camera_id),
    assigned_global_id    TEXT REFERENCES global_identities(global_id) ON DELETE SET NULL,
    decision_type         TEXT NOT NULL,   -- match | new | candidate | ambiguous | held
    top1_global_id        TEXT,
    top1_camera_id        TEXT,
    top1_score            DOUBLE PRECISION,
    top2_global_id        TEXT,
    top2_camera_id        TEXT,
    top2_score            DOUBLE PRECISION,
    reid_similarity       DOUBLE PRECISION,
    temporal_score        DOUBLE PRECISION,
    camera_topology_score DOUBLE PRECISION,
    quality_score         DOUBLE PRECISION,
    zone_score            DOUBLE PRECISION,
    final_score           DOUBLE PRECISION,
    reason                TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- PATCH-030 fix: enforce the decision_type enum. A typo now
    -- fails the INSERT, not silently inserts garbage.
    CONSTRAINT chk_decision_type CHECK (
        decision_type IN ('match', 'new', 'candidate', 'ambiguous', 'held')
    )
);

CREATE INDEX IF NOT EXISTS idx_id_tracklet   ON identity_decisions(tracklet_id);
CREATE INDEX IF NOT EXISTS idx_id_assigned   ON identity_decisions(assigned_global_id);
CREATE INDEX IF NOT EXISTS idx_id_decision   ON identity_decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_id_created    ON identity_decisions(created_at DESC);

-- identity_merge_audit: when an operator (or a future model) merges two
-- global_ids, the merge is recorded for audit.
CREATE TABLE IF NOT EXISTS identity_merge_audit (
    merge_id          BIGSERIAL PRIMARY KEY,
    old_global_id     TEXT NOT NULL REFERENCES global_identities(global_id),
    new_global_id     TEXT NOT NULL REFERENCES global_identities(global_id),
    operator          TEXT NOT NULL,                -- "operator:<username>" or "auto:v2"
    reason            TEXT,
    score             DOUBLE PRECISION,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ima_old ON identity_merge_audit(old_global_id);
CREATE INDEX IF NOT EXISTS idx_ima_new ON identity_merge_audit(new_global_id);
