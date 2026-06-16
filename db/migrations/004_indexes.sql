-- =============================================================================
-- 004_indexes.sql — secondary indexes for production query patterns
-- =============================================================================

-- For dashboard "recent events by camera" queries
CREATE INDEX IF NOT EXISTS idx_ze_recent_per_camera
    ON zone_events(camera_id, "timestamp" DESC)
    INCLUDE (global_id, zone_id, event_type);

-- For "active identities right now"
CREATE INDEX IF NOT EXISTS idx_gi_active
    ON global_identities(last_seen_at DESC)
    WHERE status = 'active';

-- For "identity lookup by tracklet"
CREATE INDEX IF NOT EXISTS idx_tracklets_global_lookup
    ON tracklets(global_id, start_time DESC)
    INCLUDE (camera_id, end_time, best_crop_uri, quality_score);

-- For TTL cleanup
CREATE INDEX IF NOT EXISTS idx_gi_ttl
    ON global_identities(last_seen_at)
    WHERE status = 'active';

-- For "tracklets open now" (those without end_time)
CREATE INDEX IF NOT EXISTS idx_tracklets_open
    ON tracklets(camera_id, start_time DESC)
    WHERE end_time IS NULL;
