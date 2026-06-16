#!/usr/bin/env python3
"""Preflight — verify the operator environment is ready to run.

Produces ``scripts/readiness_preflight.json`` consumed by
``scripts/readiness_gate.py``.

Checks:

  1. ``SOTA_API_TOKEN`` is set (production refuses to start without).
  2. The Dockerfile's GPU base image is available (informational
     on a non-GPU host; the value is recorded but not failed).
  3. The TransReID weight is present when ``reid.active_model`` is a
     TransReID profile such as ``transreid`` or ``transreid_msmt``.
  4. The PP-Human pipeline script is present.
  5. The MinIO, Postgres, Qdrant, Redis env vars are set.
  6. The benchmark report directory exists (created if missing).

Usage::

    python scripts/readiness_preflight.py
    python scripts/readiness_preflight.py --out scripts/readiness_preflight.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
)
log = logging.getLogger("readiness_preflight")

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# Check helpers
# ----------------------------------------------------------------------------


def _env_str(key: str) -> str:
    return os.environ.get(key, "").strip()


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _check(name: str, ok: bool, reason: str = "") -> dict[str, Any]:
    return {"name": name, "ok": ok, "reason": reason or ("OK" if ok else "FAIL")}


def _check_api_token() -> dict[str, Any]:
    token = _env_str("SOTA_API_TOKEN")
    if not token:
        return _check("sota_api_token", False, "SOTA_API_TOKEN env var is empty")
    if token == "change_me_in_production":
        return _check(
            "sota_api_token",
            False,
            "SOTA_API_TOKEN is the default 'change_me_in_production'",
        )
    return _check("sota_api_token", True, f"len={len(token)}")


def _check_transreid_weight() -> dict[str, Any]:
    """Verify the configured TransReID weight exists on disk."""
    cfg_path = ROOT / "configs" / "app.yaml"
    if not cfg_path.exists():
        return _check("transreid_weight", True, "configs/app.yaml not found; skipped")
    try:
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        return _check("transreid_weight", False, f"configs/app.yaml unreadable: {e}")
    reid_cfg = cfg.get("reid") or {}
    active_model = str(reid_cfg.get("active_model") or "").strip()
    if active_model not in {"transreid", "transreid_msmt"}:
        return _check(
            "transreid_weight",
            True,
            f"active_model={active_model!r}; skipped",
        )
    # Prefer TRANSREID_WEIGHT env var; fall back to YAML.
    weight = _env_str("TRANSREID_WEIGHT") or reid_cfg.get("transreid_weight", "")
    profile = reid_cfg.get("transreid_profile", reid_cfg.get("profile", ""))
    if not weight:
        return _check("transreid_weight", False, "no weight path configured")
    if not Path(weight).exists():
        return _check(
            "transreid_weight",
            False,
            f"weight {weight!r} not found (profile={profile!r})",
        )
    return _check(
        "transreid_weight",
        True,
        f"weight={weight!r} profile={profile!r}",
    )


def _check_pphuman_pipeline() -> dict[str, Any]:
    """Verify the PaddleDetection PP-Human pipeline is on disk."""
    cfg_path = ROOT / "configs" / "app.yaml"
    if not cfg_path.exists():
        return _check("pphuman_pipeline", True, "configs/app.yaml not found; skipped")
    try:
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        return _check("pphuman_pipeline", False, f"configs/app.yaml unreadable: {e}")
    det_cfg = cfg.get("detection_tracking") or {}
    pipeline = (
        _env_str("PPHUMAN_PIPELINE_PATH")
        or det_cfg.get("pphuman_pipeline_path", "")
        or "/opt/paddledetection/deploy/pipeline/pipeline.py"
    )
    if not pipeline:
        return _check("pphuman_pipeline", True, "no pipeline path configured")
    if not Path(pipeline).exists():
        return _check(
            "pphuman_pipeline",
            False,
            f"pipeline {pipeline!r} not found",
        )
    return _check("pphuman_pipeline", True, f"pipeline={pipeline!r}")


def _check_infra_env() -> dict[str, Any]:
    """Verify the infra env vars are set."""
    required = {
        "POSTGRES_HOST": "relation-store",
        "POSTGRES_USER": "yamaha",
        "POSTGRES_PASSWORD": "change_me_in_production",
        "QDRANT_HOST": "vector-store",
        "MINIO_ENDPOINT": "",
        "MINIO_ACCESS_KEY": "change_me_in_production",
        "MINIO_SECRET_KEY": "change_me_in_production",
        "REDIS_HOST": "message-bus",
    }
    # PATCH (2026-06-17): the internal minio service was removed from
    # docker-compose. An endpoint still pointing at the dead local
    # ``minio:9000`` host (the pre-Phase-7 default) is almost certainly
    # a stale .env. We refuse it here so the operator sees a clear
    # preflight failure instead of a confusing DNS error at runtime.
    stale_minio_endpoints = ("minio:9000", "minio:9000/")
    missing: list[str] = []
    defaults: list[str] = []
    placeholders: list[str] = []
    stale: list[str] = []
    for k, default in required.items():
        v = _env_str(k)
        if not v:
            missing.append(k)
        # Bug-fix: previously this was
        #   `elif v == default and k.endswith("PASSWORD") or k.endswith("KEY")`
        # which (due to ``and``/``or`` precedence) flagged ANY env var
        # ending in "KEY" as a default credential — even legitimate ones
        # like ``MINIO_ACCESS_KEY=<MINIO_ACCESS_KEY>``.  The intent is "report as
        # default ONLY if v == default AND the var is a secret".
        elif v == default and (k.endswith("PASSWORD") or k.endswith("KEY")):
            defaults.append(k)
        elif (
            k.endswith(("PASSWORD", "KEY"))
            and v.startswith("<")
            and v.endswith(">")
        ):
            placeholders.append(k)
        elif k == "MINIO_ENDPOINT" and v.startswith(stale_minio_endpoints):
            stale.append(k)
    if missing:
        return _check(
            "infra_env",
            False,
            f"missing env vars: {', '.join(missing)}",
        )
    if defaults:
        return _check(
            "infra_env",
            False,
            f"default credentials still in use: {', '.join(defaults)}",
        )
    if placeholders:
        return _check(
            "infra_env",
            False,
            f"placeholder credentials still in use: {', '.join(placeholders)}",
        )
    if stale:
        return _check(
            "infra_env",
            False,
            f"stale values still in use: {', '.join(stale)} "
            "set MINIO_ENDPOINT to the operator's external MinIO endpoint",
        )
    return _check("infra_env", True, "all env vars present")


def _check_docker_compose() -> dict[str, Any]:
    """Verify docker compose config validates."""
    if shutil.which("docker") is None:
        return _check("docker_compose", True, "docker CLI not available; skipped")
    import subprocess

    try:
        proc = subprocess.run(
            ["docker", "compose", "config", "--quiet"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001
        return _check("docker_compose", False, f"docker compose config failed: {e}")
    if proc.returncode != 0:
        return _check(
            "docker_compose",
            False,
            f"docker compose config: {proc.stderr[:200]}",
        )
    return _check("docker_compose", True, "valid")


def _check_benchmark_dir() -> dict[str, Any]:
    p = ROOT / "reports"
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return _check("benchmark_dir", True, f"path={p}")


# ----------------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------------


def run_preflight() -> dict[str, Any]:
    checks = {
        "sota_api_token": _check_api_token(),
        "transreid_weight": _check_transreid_weight(),
        "pphuman_pipeline": _check_pphuman_pipeline(),
        "infra_env": _check_infra_env(),
        "docker_compose": _check_docker_compose(),
        "benchmark_dir": _check_benchmark_dir(),
    }
    all_ok = all(c["ok"] for c in checks.values())
    return {
        "ok": all_ok,
        "checks": checks,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": (
            "production"
            if _env_str("SOTA_RUNTIME_MODE") in ("", "production", "multi_rtsp")
            else _env_str("SOTA_RUNTIME_MODE")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight — verify the operator environment is ready.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scripts/readiness_preflight.json"),
        help="Path to write the preflight JSON.",
    )
    args = parser.parse_args()
    report = run_preflight()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("Wrote preflight to %s (ok=%s)", args.out, report["ok"])
    for name, c in report["checks"].items():
        marker = "OK  " if c["ok"] else "FAIL"
        log.info("  [%s] %s: %s", marker, name, c["reason"])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
