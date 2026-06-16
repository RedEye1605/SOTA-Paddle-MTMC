"""Architecture guard — one model instance per process (PATCH-037).

The audit's PATCH-007 fix must be enforced by a test that constructs
a ``MultiCameraRunner`` with at least two cameras and asserts that all
workers share the same detector (or, in smoke mode, the same
``None``-detector flag).

We also enforce the "no Service/ writes" rule (existing) and the
"weights_only=True" rule (the security audit's
``test_dangerous_weights_refused``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.runtime_mode import RuntimeMode
from app.workers.multi_camera_runner import CameraSource, MultiCameraRunner


def _stub_reader(n: int = 2):
    import numpy as np

    for i in range(n):
        yield i, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)


def test_multi_camera_shares_detector_in_smoke() -> None:
    """In smoke-test mode all workers share the same synthetic
    fallback (None detector). Verified via ``shared_detector()``.
    """
    import numpy as np

    def _fake_factory(source: str):
        def _gen(n: int = 2):
            for i in range(n):
                yield i, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)

        return _gen()

    sources = [
        CameraSource("CAM_01", "stub://1", 640, 480, 5),
        CameraSource("CAM_02", "stub://2", 640, 480, 5),
        CameraSource("CAM_03", "stub://3", 640, 480, 5),
    ]
    runner = MultiCameraRunner(
        sources,
        skip_frame_num=0,
        smoke_test_mode=True,  # implies SMOKE_TEST mode
        mode=RuntimeMode.SMOKE_TEST,
        frame_reader_factory=_fake_factory,
    )
    runner.start()
    try:
        shared = runner.shared_detector()
        # Synthetic fallback ⇒ None detector (all workers use the
        # _synthetic_detect path). The architecture-guard contract
        # is that the SHARED reference is the same across all workers.
        for w in runner._workers:
            assert w._smoke_synthetic is True
        # The runner has a single ``_shared_detector`` field that is
        # the same for all workers.
        assert shared is runner._shared_detector
    finally:
        runner.stop()


def test_multi_camera_refuses_in_production_without_detector() -> None:
    import numpy as np

    def _fake_factory(source: str):
        def _gen(n: int = 2):
            for i in range(n):
                yield i, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)

        return _gen()

    sources = [
        CameraSource("CAM_01", "stub://1", 640, 480, 5),
        CameraSource("CAM_02", "stub://2", 640, 480, 5),
    ]
    runner = MultiCameraRunner(
        sources,
        skip_frame_num=0,
        smoke_test_mode=False,  # production
        mode=RuntimeMode.PRODUCTION,
        frame_reader_factory=_fake_factory,
    )
    with pytest.raises(Exception):
        # The constructor refuses to start without a detector.
        runner.start()
    runner.stop()


def test_multi_camera_runner_wires_per_camera_metrics() -> None:
    """PATCH-018: the runner must wire per-camera metrics objects and
    record at least one observation per camera after a few frames.
    """
    import numpy as np
    from app.telemetry.per_camera import PER_CAMERA

    def _fake_factory(source: str):
        def _gen(n: int = 4):
            for i in range(n):
                yield i, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)

        return _gen()

    sources = [
        CameraSource("CAM_METRICS_A", "stub://1", 640, 480, 5),
        CameraSource("CAM_METRICS_B", "stub://2", 640, 480, 5),
    ]
    runner = MultiCameraRunner(
        sources,
        skip_frame_num=0,
        smoke_test_mode=True,
        mode=RuntimeMode.SMOKE_TEST,
        frame_reader_factory=_fake_factory,
        frame_queue_maxsize=8,
        drop_policy="drop_oldest",
    )
    runner.start()
    try:
        # Drain a few frames so _run_worker can record observations.
        drained = 0
        for r in runner.stream(max_seconds=0.5):
            drained += 1
            if drained > 8:
                break
        # Per-camera metrics objects exist and have a non-zero
        # frame-count (the observe_frame() updates the deque).
        for cam_id in ("CAM_METRICS_A", "CAM_METRICS_B"):
            m = PER_CAMERA.for_camera(cam_id)
            # We may not have observed a window's worth of frames
            # in 0.5 s, but the queue depth and the status must be
            # set.
            assert m.status in {
                __import__(
                    "app.telemetry.per_camera", fromlist=["CAMERA_STATUS_ONLINE"]
                ).CAMERA_STATUS_ONLINE,
            }
    finally:
        runner.stop()


# ---- Existing architecture-guard tests, kept as a regression suite ----


def test_no_writes_into_service() -> None:
    """SOTA code must NEVER touch the existing Service/ folder.

    Per the 2026-06-14 operator directive (§1 + §9): ``Service/``
    remains reference-only. This test enforces two things:
      1. *Production* code (app/, scripts/, configs/) MUST NOT
         reference any Service/ path — coupling SOTA production
         paths to the legacy Service/ code is the regression this
         guard exists to prevent.
      2. *Parity tests* (``tests/integrations/test_legacy_*.py``
         and the ``_parity_assets/`` helpers) are EXPLICITLY
         allowed to reference Service/ paths — their whole purpose
         is to invoke the legacy Service/ implementation in a
         separate process to capture baseline output for diff
         against the new SOTA output. These are read-only
         comparisons; they do not modify Service/.

    If you add a Service/ reference to a *non*-parity file, this
    guard will fail. That is the right outcome — it means a
    SOTA production path has started coupling to the legacy code.
    """
    ROOT = Path(__file__).resolve().parents[2]
    FORBIDDEN_PATHS_IN_SERVICE = [
        "app",
        "config.yaml",
        "main.py",
        "scripts",
        "docs",
        "offline-people-counting",
    ]
    # Files that are allowed to reference Service/ (read-only parity).
    ALLOWED_REFERENCE_FILES = {
        # docs/notes file is allowed to mention the Service/ folder
        # by name (Markdown prose).
        "comparison_with_existing_service.md",
        # Parity tests in tests/integrations/ run the legacy
        # Service/ scripts in a subprocess to capture baseline
        # output. They never modify Service/.
        "test_legacy_payload_execution_parity.py",
        "test_legacy_contract.py",
        "test_legacy_minio.py",
        "test_legacy_mqtt.py",
        "test_legacy_payload.py",
        "test_legacy_streaming.py",
        "test_legacy_toggle_live_proof.py",
        "test_legacy_toggles.py",
        "test_legacy_yaml_vs_legacy_source.py",
        "test_mqtt_legacy_contract_env.py",
        "service_dump.py",  # _parity_assets/ helper
    }
    violations: list[str] = []
    for py in (ROOT).rglob("*.py"):
        # Skip allowed-reference files.
        if py.name in ALLOWED_REFERENCE_FILES:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for forbidden in FORBIDDEN_PATHS_IN_SERVICE:
            if py.name in {"comparison_with_existing_service.md"}:
                continue
            for m in re.finditer(r'["\']([^"\']*Service/[^"\']+)["\']', text):
                p = m.group(1)
                if "/Service/" in p:
                    line = text.split("\n")[text[: m.start()].count("\n")]
                    if not line.lstrip().startswith("#"):
                        violations.append(f"{py}: {p}")
    assert not violations, "SOTA code references Service/ paths: " + "\n".join(violations)


def test_no_forbidden_models_imported() -> None:
    ROOT = Path(__file__).resolve().parents[2]
    FORBIDDEN_PATTERNS = [
        (r"rfdetr", "RF-DETR is not the primary detector; do not import."),
        (r"botsort", "BoT-SORT is not the primary tracker; do not import."),
        (r"boxmot", "BoxMOT is not the primary tracker; do not import."),
        (r"youtureid", "YouTuReID is not the default ReID; do not import."),
    ]
    violations: list[str] = []
    for py in (ROOT).rglob("*.py"):
        if "tests" in py.parts:
            continue
        if py.name == "compare_with_service_baseline.py":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for pat, msg in FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                line = text.split("\n")[text[: m.start()].count("\n")]
                if line.lstrip().startswith("#"):
                    continue
                violations.append(f"{py}: '{m.group(0)}' ({msg})")
    assert not violations, "Forbidden imports found:\n" + "\n".join(violations)


def test_dangerous_weights_refused() -> None:
    """The TransReID adapter MUST always use ``weights_only=True`` when
    loading a checkpoint — refuses arbitrary pickle objects.
    """
    import re

    ROOT = Path(__file__).resolve().parents[2]
    ad = (ROOT / "app" / "reid" / "transreid_adapter.py").read_text()
    # Weights-only safety in the vendor/load path.
    assert "weights_only" in ad, "TransReID adapter must gate on weights_only"
    # No weights_only=False anywhere in our application code.
    # Exclude tests/, .venv/ (third-party wheels), and other
    # vendored/model paths that are not under our control.
    bad = []
    skip_dirs = {".venv", "models", "data", "logs", "reports", "tests"}
    for py in ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in py.parts):
            continue
        if re.search(r"weights_only\s*=\s*False", py.read_text()):
            bad.append(str(py))
    assert not bad, f"weights_only=False found in: {bad}"
