"""Runtime mode + production safety gate.

This module is the SINGLE source of truth for whether the system is in
production mode, smoke-test mode, or benchmark mode, and whether the
synthetic/deterministic fallbacks are allowed.

Fix for PATCH-003, PATCH-040, and the audit's "production paths are
deterministic fallbacks" risk.

Rules (enforced by tests in tests/test_production_safety.py):
  * In ``production`` mode (the default for ``multi_rtsp``):
      - the synthetic detector MUST be refused;
      - the deterministic / histogram ReID MUST be refused;
      - the fake crop fabrication MUST be refused;
      - PPHUMAN_INFER_FN must resolve to a real paddle inference path
        when PP-Human is the active detector;
      - TRANSREID_MODEL_FN must resolve to a real torch model when
        TransReID is the active ReID.
  * In ``smoke_test`` mode (explicit ``--mode smoke_test`` or
    ``ALLOW_SYNTHETIC_SMOKE_TEST=true``): all of the above are allowed,
    every emitted log is prefixed ``[SMOKE-TEST]``, and the smoke output
    is never persisted to the live ``global_identities`` table.
  * In ``benchmark`` mode: real models, but ``/metrics`` includes extra
    timing histograms.

The gate is a *fail-fast* gate — once the runtime mode is set, every
component's ``load()`` method calls :func:`assert_production_safe` to
refuse to operate if the mode disallows it. This is the architectural
fix for the audit's "production paths are deterministic fallbacks" verdict.
"""

from __future__ import annotations

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class RuntimeMode(str, Enum):
    PRODUCTION = "production"
    SMOKE_TEST = "smoke_test"
    BENCHMARK = "benchmark"
    # Legacy aliases kept for backward compatibility with the old
    # ``multi_rtsp`` and ``single_cam_smoke`` modes. New code should use
    # the explicit values above.
    MULTI_RTSP = "multi_rtsp"  # → PRODUCTION
    SINGLE_CAM_SMOKE = "single_cam_smoke"  # → SMOKE_TEST

    @classmethod
    def from_string(cls, value: str | None) -> "RuntimeMode":
        if not value:
            return cls.PRODUCTION
        v = value.strip().lower()
        if v in {"production", "multi_rtsp"}:
            return cls.PRODUCTION
        if v in {"smoke_test", "smoke", "single_cam_smoke"}:
            return cls.SMOKE_TEST
        if v == "benchmark":
            return cls.BENCHMARK
        # Unknown — be strict, refuse to start
        raise ValueError(f"Unknown runtime mode: {value!r}")

    @property
    def allows_synthetic(self) -> bool:
        return self == RuntimeMode.SMOKE_TEST


def resolve_runtime_mode() -> RuntimeMode:
    """Resolve the runtime mode from env + args.

    Priority:
      1. ``SOTA_RUNTIME_MODE`` env var.
      2. ``ALLOW_SYNTHETIC_SMOKE_TEST=true`` env var → SMOKE_TEST.
      3. ``--mode`` argument is handled by main.py and passed in.
    """
    env = os.environ.get("SOTA_RUNTIME_MODE", "").strip().lower()
    if env:
        return RuntimeMode.from_string(env)
    if os.environ.get("ALLOW_SYNTHETIC_SMOKE_TEST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return RuntimeMode.SMOKE_TEST
    return RuntimeMode.PRODUCTION


class ProductionSafetyError(RuntimeError):
    """Raised when a production-unsafe code path is requested in production mode."""


def assert_production_safe(*, mode: RuntimeMode, component: str, condition: str) -> None:
    """Refuse to operate if ``mode`` disallows the requested operation.

    Args:
        mode: current runtime mode.
        component: human-readable component name (e.g. ``"PPHumanReIDAdapter"``).
        condition: what is being attempted (e.g. ``"deterministic 768-dim fallback"``).
    """
    if mode.allows_synthetic:
        return
    msg = (
        f"PRODUCTION REFUSED: {component} attempted {condition!r} but "
        f"runtime mode is {mode.value!r} which disallows synthetic / "
        f"deterministic paths. Set --mode smoke_test or "
        f"ALLOW_SYNTHETIC_SMOKE_TEST=true to allow this in dev/CI only."
    )
    logger.error(msg)
    raise ProductionSafetyError(msg)


def smoke_log(component: str, msg: str) -> None:
    """Log a smoke-test-only message with a clearly visible prefix.

    The "[SMOKE-TEST]" prefix is regex-greppable in CI for guard
    enforcement.
    """
    logger.warning("[SMOKE-TEST] %s: %s", component, msg)


# Default placeholders for the pluggable inference callables. Production
# deploys MUST set these. The system refuses to start in production mode
# if either is unset when the corresponding model is active.
PPHUMAN_INFER_FN_ENV = "PPHUMAN_INFER_FN"
TRANSREID_MODEL_FN_ENV = "TRANSREID_MODEL_FN"


def get_inference_callable(env_var: str):
    """Resolve a Python callable from an env var, e.g. ``module:function``.

    Returns ``None`` if the env var is unset, in which case the caller
    must decide whether to fail (production) or warn (smoke-test).
    """
    import importlib

    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    if ":" not in raw:
        raise ValueError(
            f"{env_var} must be in the form 'package.module:callable', got {raw!r}",
        )
    mod_path, attr = raw.split(":", 1)
    mod = importlib.import_module(mod_path)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise ValueError(f"{env_var}={raw} resolved to non-callable: {fn!r}")
    return fn


def require_inference_callable(env_var: str, *, mode: RuntimeMode) -> object:
    """Same as :func:`get_inference_callable` but refuses in production
    mode if the env var is unset.

    Used by ``ReIDAdapter.load()`` and ``PPHumanDetector.load()`` to
    fail-fast in production.
    """
    fn = get_inference_callable(env_var)
    if fn is not None:
        return fn
    assert_production_safe(
        mode=mode,
        component=env_var,
        condition="missing pluggable inference callable",
    )
    smoke_log(
        env_var,
        f"{env_var} not set; smoke-test only — real model not loaded.",
    )
    return None
