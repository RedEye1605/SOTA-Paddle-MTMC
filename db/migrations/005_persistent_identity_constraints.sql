-- =============================================================================
-- 005_persistent_identity_constraints.sql — idempotency for the persistent-ID
-- architecture (operator's spec, 2026-06-15). All migrations are additive
-- (CREATE ... IF NOT EXISTS / ADD CONSTRAINT IF NOT EXISTS where supported).
-- =============================================================================

-- The base schema records one row per Qdrant point:
--   UNIQUE (vector_db_collection, vector_db_point_id)
-- A previous revision added a unique constraint on
-- (tracklet_id, model_name, model_version), which conflicts with
-- per-crop embeddings from ReIDWorker and TransReIDSidecar. Drop that
-- constraint when present and keep a non-unique lookup index instead.
ALTER TABLE tracklet_embeddings
    DROP CONSTRAINT IF EXISTS tracklet_embeddings_tracklet_model_version_uniq;

CREATE INDEX IF NOT EXISTS tracklet_embeddings_tracklet_model_version_idx
    ON tracklet_embeddings (tracklet_id, model_name, model_version);

-- PATCH (2026-06-17): the column names in this migration were wrong
-- (decision_stage / last_seen_camera_id) — both columns don't exist
-- in the base tables. The correct names are decision_type and
-- last_camera_id. Renamed to fix a runtime migration failure.
--
-- Composite index (tracklet_id, decision_type) for fast audit queries
-- that filter by both tracklet and decision outcome.
CREATE INDEX IF NOT EXISTS identity_decisions_tracklet_type_idx
    ON identity_decisions (tracklet_id, decision_type);

-- Composite index (last_camera_id, last_seen_at DESC) for the
-- resolver's Stage 1 same-camera hot path: "last seen in this
-- camera, ordered by recency".
CREATE INDEX IF NOT EXISTS global_identities_last_cam_seen_idx
    ON global_identities (last_camera_id, last_seen_at DESC);

-- PATCH (2026-06-15): add updated_at to tracklets. The resolver
-- and rekey worker reference it; the column is missing from the
-- initial schema. Idempotent.
ALTER TABLE tracklets
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE
    DEFAULT now();
