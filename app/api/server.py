"""FastAPI server with health, identity, event, metrics, and retention
endpoints. Authentication is required for all identity endpoints (PATCH-014).

The token is loaded from ``SOTA_API_TOKEN`` (env) at startup. If the env
var is unset and ``runtime.mode`` is production, the server refuses to
start. In smoke-test mode the auth dependency is bypassed (the smoke
test never makes HTTP calls to the API; the auth check is therefore
moot for the existing test suite).

Public endpoints:
  - GET /health
  - GET /metrics

Authenticated endpoints (Bearer token):
  - GET /identity/{global_id}
  - GET /identity/decisions
  - GET /events/zone
  - GET /dwell/summary
  - GET /identity/ambiguous  (improvement-loop)

Improvement-loop endpoints (POST /admin/retention/run) live in
``scripts/retention_worker.py`` and are not exposed over HTTP — run
the script via the docker compose ``retention`` service or a cron.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.runtime_mode import RuntimeMode
from ..storage.minio_store import MinioStore
from ..storage.postgres import PostgresStore
from ..telemetry.metrics import REGISTRY

logger = logging.getLogger(__name__)


# MinIO /health probe cache. Keyed by MinioStore._endpoint so multiple
# stores (e.g. across test invocations or future multi-tenant setups)
# don't share state. The value is ``(monotonic_ts, status_str)``;
# statuses are "ok" / "down" / "timeout" — "disabled" short-circuits
# before the cache check, so it never lands in the dict.
_MINIO_CACHE_TTL_SECONDS = 30.0
_MINIO_PROBE_TIMEOUT_SECONDS = 2.0
_minio_status_cache: dict[str, tuple[float, str]] = {}
_minio_cache_lock = threading.Lock()


async def _run_blocking(func, timeout: Optional[float] = None) -> Any:
    """Run blocking storage code without Starlette's/default threadpool.

    The installed FastAPI/Starlette/httpx stack hangs on sync route
    execution and on ``asyncio.to_thread`` executor shutdown in tests.
    A daemon worker thread avoids both failure modes while preserving
    non-blocking ASGI handlers.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    def _set_result(value: Any) -> None:
        if not future.done():
            future.set_result(value)

    def _set_exception(exc: Exception) -> None:
        if not future.done():
            future.set_exception(exc)

    def _target() -> None:
        try:
            result = func()
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(_set_exception, e)
        else:
            loop.call_soon_threadsafe(_set_result, result)

    threading.Thread(target=_target, daemon=True, name="api-blocking-call").start()
    if timeout is None:
        return await future
    return await asyncio.wait_for(future, timeout=timeout)


async def _minio_status(minio: Optional[MinioStore]) -> str:
    """Return the cached minio health status, probing on miss/expired.

    Returns:
        "disabled" — minio is not wired up to this app, or the
            operator set ``MINIO_ENABLED=false`` as an explicit
            kill switch.
        "ok" / "down" / "timeout" — the result of a probe bounded to
            ``_MINIO_PROBE_TIMEOUT_SECONDS`` and cached for
            ``_MINIO_CACHE_TTL_SECONDS`` to avoid hammering the
            MinIO cluster on every /health poll.
    """
    if minio is None:
        return "disabled"
    enabled_env = os.environ.get("MINIO_ENABLED", "").strip().lower()
    if enabled_env in {"0", "false", "no", "off"}:
        return "disabled"
    endpoint = minio._endpoint
    now = time.monotonic()
    with _minio_cache_lock:
        cached = _minio_status_cache.get(endpoint)
        if cached is not None:
            ts, status_str = cached
            if (now - ts) < _MINIO_CACHE_TTL_SECONDS:
                return status_str
    # Probe outside the cache lock so two endpoints (or a second
    # caller during a slow probe) don't serialise on the same dict.
    # The daemon-thread wrapper enforces the 2s wall bound even
    # though the minio SDK's underlying urllib3 timeout is several
    # minutes by default. A timed-out SDK call may keep running in
    # that daemon thread, but /health latency stays bounded.
    try:
        ok = await _run_blocking(
            lambda: minio.is_reachable(_MINIO_PROBE_TIMEOUT_SECONDS),
            timeout=_MINIO_PROBE_TIMEOUT_SECONDS + 0.5,
        )
        status_str = "ok" if ok else "down"
    except asyncio.TimeoutError:
        status_str = "timeout"
    except Exception as e:  # noqa: BLE001
        logger.warning("minio /health probe raised: %s", e)
        status_str = "down"
    with _minio_cache_lock:
        _minio_status_cache[endpoint] = (now, status_str)
    return status_str


def _get_api_token() -> Optional[str]:
    """Read the API token from the env. Returns None if unset."""
    return os.environ.get("SOTA_API_TOKEN", "").strip() or None


def _build_auth_dependency(token: Optional[str]):
    """Build a FastAPI dependency that enforces the Bearer token.

    In production: token MUST be set; missing or wrong → 401.
    In smoke-test mode: auth is a no-op (smoke tests don't call HTTP).
    """
    bearer = HTTPBearer(auto_error=False)

    async def _verify(
        creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    ) -> Optional[str]:
        if token is None:
            return None
        if creds is None or creds.scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not _safe_compare(creds.credentials, token):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return creds.credentials

    return _verify


def _safe_compare(a: str, b: str) -> bool:
    """Constant-time string compare. Avoids timing attacks on the token."""
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def build_app(
    pg: Optional[PostgresStore] = None,
    minio: Optional[MinioStore] = None,
    mode: RuntimeMode = RuntimeMode.PRODUCTION,
    token: Optional[str] = None,
) -> FastAPI:
    """Construct the FastAPI app.

    In production mode the ``SOTA_API_TOKEN`` env var must be set; the
    server raises at startup if not. In smoke-test mode auth is a
    no-op and the env var may be absent.

    The ``minio`` argument is optional: pass the application's
    :class:`MinioStore` instance to enable the /health minio probe,
    or leave it ``None`` to report ``minio: "disabled"``.
    """
    if token is None:
        token = _get_api_token()
    if mode == RuntimeMode.PRODUCTION and not token:
        raise RuntimeError(
            "SOTA_API_TOKEN must be set in production mode (PATCH-014). "
            "Set it in the .env file or pass it via env.",
        )
    if token:
        logger.info("FastAPI auth enabled")
    else:
        logger.warning(
            "[SMOKE-TEST] FastAPI auth disabled (no SOTA_API_TOKEN). "
            "This must NOT happen in production.",
        )
    verify = _build_auth_dependency(token)

    app = FastAPI(title="SOTA-Paddle-MTMC", version="0.2.0")

    @app.get("/health")
    async def health(response: Response) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "ok", "mode": mode.value}
        if pg is not None:
            # PATCH-042: bounded DB healthcheck so a Postgres outage
            # does not block the FastAPI threadpool.
            try:
                result["postgres"] = (
                    "ok"
                    if await _run_blocking(pg.healthcheck, timeout=2.0)
                    else "down"
                )
            except asyncio.TimeoutError:
                result["postgres"] = "timeout"
        # PATCH (2026-06-17): the internal minio service was removed;
        # the api now talks to the operator's external MinIO cluster.
        # ``minio_enabled`` reflects whether the MinioStore is wired
        # up in this process; ``result["minio"]`` is the cached probe
        # status ("ok" / "down" / "timeout" / "disabled").
        result["minio_enabled"] = minio is not None
        result["minio"] = await _minio_status(minio)
        dependency_statuses = [
            v
            for k, v in result.items()
            if k not in {"status", "mode", "minio_enabled"}
        ]
        if any(v not in {"ok", "disabled"} for v in dependency_statuses):
            result["status"] = "degraded"
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    @app.get("/metrics")
    async def metrics() -> str:
        return REGISTRY.render()

    @app.get("/identity/{global_id}")
    async def get_identity(
        global_id: str,
        _token: Optional[str] = Depends(verify),
    ) -> dict[str, Any]:
        if pg is None:
            raise HTTPException(status_code=503, detail="postgres not configured")

        def _fetch_identity() -> dict[str, Any]:
            with pg.cursor() as cur:
                cur.execute(
                    """
                    SELECT global_id, session_id, first_seen_at, last_seen_at,
                           first_camera_id, last_camera_id, status, confidence_state
                    FROM global_identities WHERE global_id = %s;
                    """,
                    (global_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="not found")
                cur.execute(
                    """
                    SELECT tracklet_id, camera_id, start_time, end_time,
                           start_zone_id, end_zone_id, quality_score
                    FROM tracklets WHERE global_id = %s
                    ORDER BY start_time DESC LIMIT 200;
                    """,
                    (global_id,),
                )
                tracklets = list(cur.fetchall())
            return {"identity": row, "tracklets": tracklets}

        return await _run_blocking(_fetch_identity)

    @app.get("/identity/decisions")
    async def get_decisions(
        limit: int = Query(default=50, ge=1, le=500),
        decision_type: Optional[str] = None,
        _token: Optional[str] = Depends(verify),
    ) -> list[dict[str, Any]]:
        if pg is None:
            raise HTTPException(status_code=503, detail="postgres not configured")

        def _fetch_decisions() -> list[dict[str, Any]]:
            with pg.cursor() as cur:
                if decision_type:
                    cur.execute(
                        """
                        SELECT decision_id, tracklet_id, source_camera_id,
                               candidate_camera_id, assigned_global_id, decision_type,
                               top1_global_id, top1_score, top2_global_id, top2_score,
                               final_score, reason, created_at
                        FROM identity_decisions WHERE decision_type = %s
                        ORDER BY created_at DESC LIMIT %s;
                        """,
                        (decision_type, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT decision_id, tracklet_id, source_camera_id,
                               candidate_camera_id, assigned_global_id, decision_type,
                               top1_global_id, top1_score, top2_global_id, top2_score,
                               final_score, reason, created_at
                        FROM identity_decisions
                        ORDER BY created_at DESC LIMIT %s;
                        """,
                        (limit,),
                    )
                return list(cur.fetchall())

        return await _run_blocking(_fetch_decisions)

    @app.get("/events/zone")
    async def get_zone_events(
        limit: int = Query(default=50, ge=1, le=500),
        camera_id: Optional[str] = None,
        _token: Optional[str] = Depends(verify),
    ) -> list[dict[str, Any]]:
        if pg is None:
            raise HTTPException(status_code=503, detail="postgres not configured")

        def _fetch_zone_events() -> list[dict[str, Any]]:
            with pg.cursor() as cur:
                if camera_id:
                    cur.execute(
                        """
                        SELECT zone_event_id, global_id, tracklet_id, camera_id,
                               zone_id, event_type, "timestamp", confidence
                        FROM zone_events WHERE camera_id = %s
                        ORDER BY "timestamp" DESC LIMIT %s;
                        """,
                        (camera_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT zone_event_id, global_id, tracklet_id, camera_id,
                               zone_id, event_type, "timestamp", confidence
                        FROM zone_events
                        ORDER BY "timestamp" DESC LIMIT %s;
                        """,
                        (limit,),
                    )
                return list(cur.fetchall())

        return await _run_blocking(_fetch_zone_events)

    @app.get("/dwell/summary")
    async def get_dwell_summary(
        limit: int = Query(default=50, ge=1, le=500),
        _token: Optional[str] = Depends(verify),
    ) -> list[dict[str, Any]]:
        if pg is None:
            raise HTTPException(status_code=503, detail="postgres not configured")

        def _fetch_dwell_summary() -> list[dict[str, Any]]:
            with pg.cursor() as cur:
                cur.execute(
                    """
                    SELECT global_id, zone_id, camera_id, entered_at, exited_at,
                           duration_seconds, status
                    FROM dwell_sessions
                    ORDER BY entered_at DESC LIMIT %s;
                    """,
                    (limit,),
                )
                return list(cur.fetchall())

        return await _run_blocking(_fetch_dwell_summary)

    @app.get("/identity/ambiguous")
    async def get_ambiguous(
        limit: int = Query(default=50, ge=1, le=500),
        _token: Optional[str] = Depends(verify),
    ) -> list[dict[str, Any]]:
        """Improvement-loop endpoint (Component 12 of IMPROVEMENT_LOOP_PLAN.md).

        Returns the most recent ``decision_type='ambiguous'`` rows so
        the operator (or a future UI) can review and resolve them.
        """
        if pg is None:
            raise HTTPException(status_code=503, detail="postgres not configured")

        def _fetch_ambiguous() -> list[dict[str, Any]]:
            with pg.cursor() as cur:
                cur.execute(
                    """
                    SELECT decision_id, tracklet_id, source_camera_id,
                           candidate_camera_id, assigned_global_id,
                           top1_global_id, top1_camera_id, top1_score,
                           top2_global_id, top2_camera_id, top2_score,
                           final_score, reason, created_at
                    FROM identity_decisions
                    WHERE decision_type = 'ambiguous'
                    ORDER BY created_at DESC LIMIT %s;
                    """,
                    (limit,),
                )
                return list(cur.fetchall())

        return await _run_blocking(_fetch_ambiguous)

    return app


def serve(pg: PostgresStore, host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    app = build_app(pg=pg)
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("LOG_LEVEL", "info").lower())
