"""Readiness preflight regression tests."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# _check_infra_env precedence regression
# ----------------------------------------------------------------------------


def _set_env(monkeypatch, **kwargs: str) -> None:
    for k, v in kwargs.items():
        monkeypatch.setenv(k, v)


def test_infra_env_passes_with_legitimate_non_default_credentials(monkeypatch) -> None:
    """Regression for the operator-precedence bug in
    ``_check_infra_env``. Previously the line:

        elif v == default and k.endswith("PASSWORD") or k.endswith("KEY"):

    flagged ANY env var ending in ``KEY`` as a "default credential"
    even when the value was a legitimate non-default. This caused
    ``infra_env`` to spuriously fail and blocked the readiness gate.

    The fix groups the boolean correctly:

        elif v == default and (k.endswith("PASSWORD") or k.endswith("KEY")):
    """
    from scripts.readiness_preflight import _check_infra_env

    # All vars set, none equal to the documented default. This must
    # pass — every env var ending in KEY is a legitimate non-default.
    _set_env(
        monkeypatch,
        POSTGRES_HOST="relation-store",
        POSTGRES_USER="yamaha",
        POSTGRES_PASSWORD="real_postgres_pw_dev_only_2026",
        QDRANT_HOST="vector-store",
        MINIO_ENDPOINT="minio.example.invalid",
        MINIO_ACCESS_KEY="dev_minio_access_key_2026",
        MINIO_SECRET_KEY="dev_minio_secret_key_2026",
        REDIS_HOST="message-bus",
    )

    out = _check_infra_env()
    assert out["ok"] is True, out


def test_infra_env_fails_when_default_password_used(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_infra_env

    _set_env(
        monkeypatch,
        POSTGRES_HOST="relation-store",
        POSTGRES_USER="yamaha",
        POSTGRES_PASSWORD="change_me_in_production",
        QDRANT_HOST="vector-store",
        MINIO_ENDPOINT="minio.example.invalid",
        MINIO_ACCESS_KEY="dev_minio_access_key_2026",
        MINIO_SECRET_KEY="dev_minio_secret_key_2026",
        REDIS_HOST="message-bus",
    )

    out = _check_infra_env()
    assert out["ok"] is False
    assert "POSTGRES_PASSWORD" in out["reason"]


def test_infra_env_fails_when_default_key_used(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_infra_env

    _set_env(
        monkeypatch,
        POSTGRES_HOST="relation-store",
        POSTGRES_USER="yamaha",
        POSTGRES_PASSWORD="real_postgres_pw_dev_only_2026",
        QDRANT_HOST="vector-store",
        MINIO_ENDPOINT="minio.example.invalid",
        MINIO_ACCESS_KEY="change_me_in_production",
        MINIO_SECRET_KEY="dev_minio_secret_key_2026",
        REDIS_HOST="message-bus",
    )

    out = _check_infra_env()
    assert out["ok"] is False
    assert "MINIO_ACCESS_KEY" in out["reason"]


def test_infra_env_fails_when_placeholder_key_used(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_infra_env

    _set_env(
        monkeypatch,
        POSTGRES_HOST="relation-store",
        POSTGRES_USER="yamaha",
        POSTGRES_PASSWORD="real_postgres_pw_dev_only_2026",
        QDRANT_HOST="vector-store",
        MINIO_ENDPOINT="minio.example.invalid",
        MINIO_ACCESS_KEY="<MINIO_ACCESS_KEY>",
        MINIO_SECRET_KEY="dev_minio_secret_key_2026",
        REDIS_HOST="message-bus",
    )

    out = _check_infra_env()
    assert out["ok"] is False
    assert "MINIO_ACCESS_KEY" in out["reason"]


def test_infra_env_fails_when_required_var_missing(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_infra_env

    for k in [
        "POSTGRES_HOST",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "REDIS_HOST",
    ]:
        monkeypatch.delenv(k, raising=False)

    out = _check_infra_env()
    assert out["ok"] is False
    assert "missing" in out["reason"].lower()


# ----------------------------------------------------------------------------
# _check_api_token
# ----------------------------------------------------------------------------


def test_api_token_fails_when_empty(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_api_token

    monkeypatch.delenv("SOTA_API_TOKEN", raising=False)
    out = _check_api_token()
    assert out["ok"] is False


def test_api_token_fails_when_default(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_api_token

    monkeypatch.setenv("SOTA_API_TOKEN", "change_me_in_production")
    out = _check_api_token()
    assert out["ok"] is False


def test_api_token_passes_when_real_value(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_api_token

    monkeypatch.setenv("SOTA_API_TOKEN", "a_real_long_random_token_2026_06")
    out = _check_api_token()
    assert out["ok"] is True
