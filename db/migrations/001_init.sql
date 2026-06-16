-- =============================================================================
-- 001_init.sql — cameras, model_versions, system_metrics
-- Source of truth: db design in Docs/database_design.md
-- =============================================================================

-- cameras: registered CCTV feeds
CREATE TABLE IF NOT EXISTS cameras (
    camera_id           TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    rtsp_url_env_key    TEXT NOT NULL,
    site_id             TEXT NOT NULL,
    timezone            TEXT NOT NULL DEFAULT 'UTC',
    width               INTEGER NOT NULL,
    height              INTEGER NOT NULL,
    fps_target          INTEGER NOT NULL DEFAULT 25,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cameras_site_id ON cameras(site_id);
CREATE INDEX IF NOT EXISTS idx_cameras_active  ON cameras(is_active);

-- model_versions: registry of all model artifacts (detector, tracker, ReID)
CREATE TABLE IF NOT EXISTS model_versions (
    model_id            SERIAL PRIMARY KEY,
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    framework           TEXT NOT NULL,
    task                TEXT NOT NULL,        -- detector | tracker | reid
    weights_uri         TEXT,                  -- local path or s3:// uri
    config_uri          TEXT,
    embedding_dim       INTEGER,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    activated_at        TIMESTAMPTZ,
    deactivated_at      TIMESTAMPTZ,
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active, task);

-- system_metrics: runtime metric snapshots (also exposed via Prometheus)
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id           BIGSERIAL PRIMARY KEY,
    metric_name         TEXT NOT NULL,
    metric_value        DOUBLE PRECISION NOT NULL,
    camera_id           TEXT,
    tags                JSONB,
    captured_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time
    ON system_metrics(metric_name, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_metrics_camera_time
    ON system_metrics(camera_id, captured_at DESC)
    WHERE camera_id IS NOT NULL;
