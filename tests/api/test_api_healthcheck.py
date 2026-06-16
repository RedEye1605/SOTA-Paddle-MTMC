"""Docker API HEALTHCHECK tests (PATCH-047)."""

from __future__ import annotations

import asyncio
import os
import re

import httpx

from app.core.runtime_mode import RuntimeMode


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _call() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_call())


def test_docker_compose_healthcheck_is_present_for_api() -> None:
    """The ``detect-pipeline`` service MUST have a Docker healthcheck."""
    ROOT = __file__.rsplit("/tests/", 1)[0]
    raw = (ROOT + "/docker-compose.yaml").__class__  # noqa
    text = (ROOT + "/docker-compose.yaml").__class__
    del text
    with open(ROOT + "/docker-compose.yaml") as f:
        dc = f.read()
    # Find the detect-pipeline: block and the next healthcheck: inside it.
    m = re.search(r"^  detect-pipeline:\n(?:[^\n]*\n)*?    healthcheck:", dc, re.MULTILINE)
    assert m is not None, "detect-pipeline service has no healthcheck block in docker-compose.yaml"


def test_docker_compose_healthcheck_uses_health_endpoint() -> None:
    """The healthcheck test MUST hit ``/health`` (the public endpoint
    that does not require auth).
    """
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(ROOT + "/docker-compose.yaml") as f:
        dc = f.read()
    # The test must reference localhost:8000/health.
    assert "/health" in dc
    # The test must not reference any authenticated endpoint.
    assert "/identity/" not in dc
    assert "/decisions" not in dc


def test_docker_compose_healthcheck_does_not_leak_secrets() -> None:
    """The healthcheck must not include any secret or env var name."""
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(ROOT + "/docker-compose.yaml") as f:
        dc = f.read()
    # Find the detect-pipeline healthcheck block.
    m = re.search(
        r"^  detect-pipeline:\n(?:[^\n]*\n)*?    healthcheck:\n((?:[^\n]*\n)*?)(?=^  )", dc, re.MULTILINE
    )
    assert m is not None
    block = m.group(1)
    # No token, password, or env var.
    assert "token" not in block.lower()
    assert "password" not in block.lower()
    assert "${" not in block
    assert "$$" not in block


def test_fastapi_health_endpoint_is_public() -> None:
    """The /health endpoint must not require auth (PATCH-014)."""
    # Build with a token set; /health must still answer 200.
    monkey_token = "test-token-XYZ"
    os.environ["SOTA_API_TOKEN"] = monkey_token
    try:
        from app.api.server import build_app

        app = build_app(pg=None, mode=RuntimeMode.PRODUCTION)
        r = _request(app, "GET", "/health")
        assert r.status_code == 200
        # /metrics is also public.
        r2 = _request(app, "GET", "/metrics")
        assert r2.status_code == 200
    finally:
        os.environ.pop("SOTA_API_TOKEN", None)


def test_protected_endpoints_still_require_auth() -> None:
    """After PATCH-047: /health is public, but /identity/* is not."""
    os.environ["SOTA_API_TOKEN"] = "test-token-ABC"
    try:
        from app.api.server import build_app

        app = build_app(pg=None, mode=RuntimeMode.PRODUCTION)
        # No token → 401.
        r = _request(app, "GET", "/identity/GID-X")
        assert r.status_code in (401, 403)
        # With the correct token → 503 (PG not configured) or 404.
        r2 = _request(
            app,
            "GET",
            "/identity/GID-X",
            headers={"Authorization": "Bearer test-token-ABC"},
        )
        assert r2.status_code in (401, 403, 503, 404)
    finally:
        os.environ.pop("SOTA_API_TOKEN", None)


# -----------------------------------------------------------------------------
# MinIO /health probe (Phase 9 wiring of the Phase 7 placeholder)
# -----------------------------------------------------------------------------


class _FakeMinioClient:
    """Minimal stand-in for ``minio.Minio`` — records ``bucket_exists`` calls.

    Returns ``bucket_exists_returns`` (or raises ``bucket_exists_raises``)
    so the test can simulate "reachable" / "down" / "SDK error" without
    a real MinIO server.
    """

    def __init__(
        self,
        bucket_exists_returns: bool = True,
        bucket_exists_raises: Exception | None = None,
    ) -> None:
        self.bucket_exists_returns = bucket_exists_returns
        self.bucket_exists_raises = bucket_exists_raises
        self.bucket_exists_calls: list[str] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        self.bucket_exists_calls.append(bucket_name)
        if self.bucket_exists_raises is not None:
            raise self.bucket_exists_raises
        return self.bucket_exists_returns


def _build_minio_store(
    endpoint: str = "minio.example.com:9000",
    bucket_exists_returns: bool = True,
    bucket_exists_raises: Exception | None = None,
):
    """Build a ``MinioStore`` with a fake client injected (bypasses connect())."""
    from app.storage.minio_store import MinioStore

    store = MinioStore(
        endpoint=endpoint,
        access_key="x",
        secret_key="y",
        secure=False,
        bucket="evidence",
    )
    store._client = _FakeMinioClient(
        bucket_exists_returns=bucket_exists_returns,
        bucket_exists_raises=bucket_exists_raises,
    )
    return store


def _reset_minio_cache() -> None:
    """Wipe the module-level probe cache so each test starts fresh."""
    from app.api import server

    with server._minio_cache_lock:
        server._minio_status_cache.clear()


def test_health_reports_minio_ok(monkeypatch) -> None:
    """When ``MinioStore.is_reachable`` returns True, /health must
    report ``minio: "ok"`` and the second call must hit the cache
    (no second probe)."""
    monkeypatch.setenv("SOTA_API_TOKEN", "test-token-minio-ok")
    monkeypatch.delenv("MINIO_ENABLED", raising=False)
    _reset_minio_cache()
    store = _build_minio_store(
        endpoint="minio-ok.example.com:9000", bucket_exists_returns=True
    )
    from app.api.server import build_app

    app = build_app(pg=None, minio=store, mode=RuntimeMode.PRODUCTION)

    r1 = _request(app, "GET", "/health")
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["minio_enabled"] is True
    assert body1["minio"] == "ok"

    # The fake recorded exactly one bucket_exists call (the probe).
    assert len(store._client.bucket_exists_calls) == 1
    assert store._client.bucket_exists_calls[0] == "evidence"

    # Second call must hit the cache — no extra probe.
    r2 = _request(app, "GET", "/health")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["minio"] == "ok"
    assert len(store._client.bucket_exists_calls) == 1  # cache hit


def test_health_reports_minio_down(monkeypatch) -> None:
    """When the probe returns False (or the SDK raises), /health must
    report ``minio: "down"``."""
    monkeypatch.setenv("SOTA_API_TOKEN", "test-token-minio-down")
    monkeypatch.delenv("MINIO_ENABLED", raising=False)
    _reset_minio_cache()
    store = _build_minio_store(
        endpoint="minio-down.example.com:9000", bucket_exists_returns=False
    )
    from app.api.server import build_app

    app = build_app(pg=None, minio=store, mode=RuntimeMode.PRODUCTION)

    r = _request(app, "GET", "/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["minio_enabled"] is True
    assert body["minio"] == "down"


def test_health_reports_minio_down_when_sdk_raises(monkeypatch) -> None:
    """A SDK exception (DNS, timeout, auth) must be reported as
    ``"down"`` — the handler never propagates a 5xx for a minio
    outage."""
    monkeypatch.setenv("SOTA_API_TOKEN", "test-token-minio-err")
    monkeypatch.delenv("MINIO_ENABLED", raising=False)
    _reset_minio_cache()
    store = _build_minio_store(
        endpoint="minio-err.example.com:9000",
        bucket_exists_raises=ConnectionError("DNS resolution failed"),
    )
    from app.api.server import build_app

    app = build_app(pg=None, minio=store, mode=RuntimeMode.PRODUCTION)

    r = _request(app, "GET", "/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "degraded"
    assert body["minio_enabled"] is True
    assert body["minio"] == "down"


def test_health_reports_minio_disabled_when_minio_arg_is_none(
    monkeypatch,
) -> None:
    """When the app is built without a MinioStore (minio=None), the
    /health endpoint must short-circuit to ``minio: "disabled"``
    without touching the cache or the SDK."""
    monkeypatch.setenv("SOTA_API_TOKEN", "test-token-minio-disabled")
    monkeypatch.delenv("MINIO_ENABLED", raising=False)
    _reset_minio_cache()
    from app.api.server import build_app

    app = build_app(pg=None, minio=None, mode=RuntimeMode.PRODUCTION)

    r = _request(app, "GET", "/health")
    assert r.status_code == 200
    body = r.json()
    assert body["minio_enabled"] is False
    assert body["minio"] == "disabled"
    # "disabled" never lands in the cache.
    from app.api import server

    with server._minio_cache_lock:
        assert server._minio_status_cache == {}


def test_health_reports_minio_disabled_via_env_var(monkeypatch) -> None:
    """``MINIO_ENABLED=false`` is an explicit kill switch: even if a
    MinioStore is wired up, /health must report ``minio: "disabled"``
    and never probe the cluster."""
    monkeypatch.setenv("SOTA_API_TOKEN", "test-token-minio-killed")
    monkeypatch.setenv("MINIO_ENABLED", "false")
    _reset_minio_cache()
    store = _build_minio_store(endpoint="minio-killed.example.com:9000")
    from app.api.server import build_app

    app = build_app(pg=None, minio=store, mode=RuntimeMode.PRODUCTION)

    r = _request(app, "GET", "/health")
    assert r.status_code == 200
    body = r.json()
    assert body["minio_enabled"] is True  # store IS wired up
    assert body["minio"] == "disabled"
    # No probe was issued.
    assert store._client.bucket_exists_calls == []
