#!/bin/sh
# =============================================================================
# scripts/migrate.sh — apply all SQL migrations in db/migrations/
#
# PATCH (2026-06-17): moved out of docker-compose.yaml's block-scalar
# command because compose v2's `$$` escape is broken inside `command: |`
# (the literal "$$f" was being passed to the shell, which expanded "$$"
# to its PID, breaking the for-loop variable). The script approach is
# simpler and more debuggable.
#
# Each migration runs in --single-transaction with ON_ERROR_STOP=1. A
# failed migration rolls back cleanly and the script exits non-zero.
# Idempotent migrations (CREATE TABLE IF NOT EXISTS) are non-destructive
# and safe to re-run.
# =============================================================================
set -eu

PGHOST="${PGHOST:-relation-store}"
PGUSER="${PGUSER:-${POSTGRES_USER:-yamaha}}"
PGDATABASE="${PGDATABASE:-${POSTGRES_DB:-yamaha_mtmct}}"
MIGRATIONS_DIR="${MIGRATIONS_DIR:-/migrations}"

if [ ! -d "$MIGRATIONS_DIR" ]; then
    echo "ERROR: migrations dir not found: $MIGRATIONS_DIR" >&2
    exit 1
fi

# Count migrations; exit early if none.
shopt_count=$(ls -1 "$MIGRATIONS_DIR"/*.sql 2>/dev/null | wc -l)
if [ "$shopt_count" -eq 0 ]; then
    echo "No migrations found in $MIGRATIONS_DIR"
    exit 0
fi

for f in "$MIGRATIONS_DIR"/*.sql; do
    echo "Applying $f in single transaction"
    psql -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE" \
         --single-transaction --set ON_ERROR_STOP=on -f "$f"
done

echo "All $shopt_count migrations applied successfully"
