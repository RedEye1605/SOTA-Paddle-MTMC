"""Pinned: the api container can read the PP-Human model files from the
host's ``./models`` directory via the compose mount.

Why this matters
----------------
docker-compose.yaml must bind-mount the host's populated ``./models``
directory at ``/models`` inside the container. If this regresses to an
empty named volume, PP-Human fails at MOT init with::

    FileNotFoundError: '/models/pphuman/strongbaseline_r50_30e_pa100k/infer_cfg.yml'

even though the model files exist on the host at
``./models/pphuman/strongbaseline_r50_30e_pa100k/infer_cfg.yml``.

This test pins the contract: the host's ``./models`` directory is
bind-mounted (or copied) into the api container at ``/models``, so
PP-Human can find ``model.pdmodel`` + ``infer_cfg.yml``.

The test reads the *host* model dir and the *container's* ``/models``
dir, and asserts they expose the same files. CI runs this in the
api container (where ``/models`` is the bind-mount), so it
definitively fails on a fresh setup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOST_MODELS = ROOT / "models" / "pphuman"

REQUIRED_MODEL_DIRS = [
    "mot_ppyoloe_l_36e_pipeline",
    "strongbaseline_r50_30e_pa100k",
]
REQUIRED_FILES_PER_DIR = ["model.pdmodel", "infer_cfg.yml"]


def test_host_models_dir_has_pphuman_weights() -> None:
    """RED: the host's ./models/pphuman must have the unpacked model
    files. The Dockerfile's download script unzips the .zip into
    $MODEL_DIR/<NAME>/, producing ``model.pdmodel`` and
    ``infer_cfg.yml`` at the top of each model subdir. The host
    directory at commit time must already be populated (it is — see
    the project README's "models" section). This is a pre-condition
    for the container-side test below; if the host dir is empty,
    nothing we do to the compose file will make MOT init succeed."""
    if not HOST_MODELS.exists():
        pytest.fail(
            f"host model dir {HOST_MODELS} is missing — run scripts/download_pphuman_models.sh"
        )
    for sub in REQUIRED_MODEL_DIRS:
        sub_path = HOST_MODELS / sub
        assert sub_path.is_dir(), (
            f"host model subdir {sub_path} missing — run scripts/download_pphuman_models.sh"
        )
        for f in REQUIRED_FILES_PER_DIR:
            assert (sub_path / f).is_file(), (
                f"host model file {sub_path / f} missing — "
                f"scripts/download_pphuman_models.sh may have left a "
                f"partial download"
            )


def test_container_models_dir_exposes_host_files() -> None:
    """RED: inside the api container, ``/models/pphuman/<NAME>/`` must
    expose the same files the host has. If the compose file stops
    bind-mounting ``./models`` read-only, this test fails with the
    same FileNotFoundError the operator sees at runtime."""
    if not HOST_MODELS.exists():
        pytest.skip("host models dir missing; precondition test will fail")
    # We're already inside the api container if /opt/paddledetection
    # exists. Outside the container, the test does not apply.
    if not Path("/opt/paddledetection").exists():
        pytest.skip("not running inside the api container")
    for sub in REQUIRED_MODEL_DIRS:
        sub_path = Path("/models/pphuman") / sub
        assert sub_path.is_dir(), (
            f"container model subdir {sub_path} is missing — "
            f"docker-compose.yaml must expose the host's ./models "
            f"directory at /models:ro."
        )
        for f in REQUIRED_FILES_PER_DIR:
            assert (sub_path / f).is_file(), (
                f"container model file {sub_path / f} missing — "
                f"the named volume is shadowing the host's populated "
                f"./models."
            )
