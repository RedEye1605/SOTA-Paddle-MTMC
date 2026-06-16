"""Pinned: ``configs/pphuman/infer_cfg_pphuman_sota.yml`` carries the same
top-level keys as PaddleDetection's upstream
``deploy/pipeline/config/infer_cfg_pphuman.yml``.

Why this matters
----------------
The PaddleDetection ``Pipeline.__init__`` (pipeline.py:84) reads
``cfg['visual']`` directly. If the operator's local config is missing
any of the upstream top-level keys (``visual``, ``warmup_frame``,
``crop_thresh``, ``attr_thresh``, ``kpt_thresh``), the standalone
PP-Human run aborts at the constructor with a ``KeyError`` BEFORE it
ever reaches the cuDNN init path. That makes the cudnn 9.x bug
impossible to reproduce in isolation — and the operator's
``docker compose run --rm api python /opt/paddledetection/.../pipeline.py``
acceptance test is silently testing the wrong thing.

This test pins the operator's local config as a strict superset of
the upstream schema. We allow the operator to add MORE keys, but we
require that no upstream top-level key be missing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "configs" / "pphuman" / "infer_cfg_pphuman_sota.yml"
UPSTREAM_MARKER = Path("/opt/paddledetection")  # baked into the api image


def test_local_pphuman_cfg_has_all_upstream_top_level_keys() -> None:
    """RED: the local cfg is missing 5 keys the upstream cfg requires."""
    upstream_path = UPSTREAM_MARKER / "deploy/pipeline/config/infer_cfg_pphuman.yml"
    if not upstream_path.exists():
        import pytest

        pytest.skip("upstream cfg not present in this checkout")
    if not LOCAL.exists():
        import pytest

        pytest.skip("local cfg not present")

    up = yaml.safe_load(upstream_path.read_text(encoding="utf-8"))
    ours = yaml.safe_load(LOCAL.read_text(encoding="utf-8"))
    up_top = {k: type(v).__name__ for k, v in up.items() if not isinstance(v, dict)}
    our_top = {k: type(v).__name__ for k, v in ours.items() if not isinstance(v, dict)}
    missing = sorted(set(up_top) - set(our_top))
    assert not missing, (
        f"configs/pphuman/infer_cfg_pphuman_sota.yml is missing upstream "
        f"top-level keys: {missing}. The PaddleDetection Pipeline "
        f"constructor reads these directly (pipeline.py:84 "
        f"self.vis_result = cfg['visual']) and raises KeyError if absent, "
        f"so the cudnn-9.x acceptance test cannot run without them."
    )


def test_local_pphuman_cfg_visual_is_true() -> None:
    """RED: ``visual`` must be True (or truthy) so PP-Human's annotator
    actually draws bounding boxes into the output stream. A missing
    ``visual`` key fails the constructor; a False value passes the
    constructor but produces no bboxes in HLS."""
    if not LOCAL.exists():
        import pytest

        pytest.skip("local cfg not present")
    ours = yaml.safe_load(LOCAL.read_text(encoding="utf-8"))
    assert ours.get("visual") is True, (
        "configs/pphuman/infer_cfg_pphuman_sota.yml must have "
        "visual: True at the top level — otherwise PP-Human's annotator "
        "is disabled and the acceptance test produces a video with no "
        "bboxes (which would falsely report 'pipeline runs' when in "
        "fact no detection is visible)."
    )


def test_local_pphuman_cfg_has_mot_enabled() -> None:
    """RED: the operator's acceptance test exercises the MOT init path
    (the path that triggers the cuDNN 9.x failure). If MOT is
    disabled, the test silently skips the broken code path."""
    if not LOCAL.exists():
        import pytest

        pytest.skip("local cfg not present")
    ours = yaml.safe_load(LOCAL.read_text(encoding="utf-8"))
    mot = ours.get("MOT", {})
    assert mot.get("enable") is True, (
        "configs/pphuman/infer_cfg_pphuman_sota.yml must have "
        "MOT.enable: True — the cudnn 9.x failure manifests in the "
        "MOT init path; with MOT disabled, the failure is silent."
    )
