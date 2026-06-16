"""Runtime separation architecture guard.

The multi-stage ``Dockerfile`` enforces the Paddle / torch ABI split:

  * ``base``     — nvidia/cuda + python deps + paddlepaddle-gpu==2.6.2
                   (Paddle-only venv; cuDNN 8.x pinned)
  * ``api``      — detect-pipeline image (extends ``base``; no torch)
  * ``sidecar``  — embedding-sidecar image (extends ``base``; adds torch
                   via inline curl + pip in the ``sidecar`` target; torch
                   replaces cuDNN 8.x with cuDNN 9.x — that's fine because
                   the sidecar never imports paddle)

Why this separation matters
---------------------------
PaddlePaddle 2.6.2's ``fused_conv2d_add_act_kernel`` (used by
PaddleDetection's PPYOLOE head) was compiled against cuDNN 8.6 + CUDA
11.8. If torch 2.4+ lives in the same venv, pip forces cuDNN 9.x
(torch's transitive) and the paddle kernel aborts MOT init with::

    ExternalError: CUDNN error(3000), CUDNN_STATUS_NOT_SUPPORTED.
      at phi/kernels/fusion/gpu/fused_conv2d_add_act_kernel.cu:610

The architectural fix is two separate images that share a common
``base`` layer:

  * ``yamaha-mtmct:api``       — base target only; Paddle + cuDNN 8.x
  * ``yamaha-mtmct:sidecar``   — base + torch; cuDNN 9.x

These tests pin the contract:

  1. ``requirements.txt`` is Paddle-only — no torch.
  2. The ``sidecar`` Dockerfile target installs torch inline (not via
     ``--extra`` / ``--group`` which would force ``uv lock`` to resolve
     torch + paddle together).
  3. The compose ``detect-pipeline`` service builds the ``api`` target.
  4. The compose ``embedding-sidecar`` service builds the ``sidecar``
     target.
  5. The compose ``replay-eval`` service is under profiles: [eval] so it
     does NOT come up with the default ``docker compose up``.

If any of these regress, the api image will pull torch's cuDNN 9.x
and PP-Human will re-break.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = ROOT / "requirements.txt"
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yaml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_requirements_excludes_torch() -> None:
    """RED: requirements.txt must NOT include torch / torchvision.
    This is the runtime install for the base image (shared by api and
    sidecar). The sidecar adds torch on top via the sidecar target.
    """
    text = _read(REQUIREMENTS)
    for needle in ("torch", "torchvision"):
        assert needle not in text, (
            f"requirements.txt must not include {needle} — the base "
            f"image installs only this file. {needle} transitively "
            f"pulls cuDNN 9.x, which is ABI-incompatible with Paddle "
            f"2.6.2's cuDNN 8.6 fused_conv2d_add_act kernel."
        )


def test_dockerfile_sidecar_target_installs_torch() -> None:
    """RED: the multi-stage Dockerfile's ``sidecar`` target must install
    torch inline (via curl + pip from the PyTorch cu124 index), NOT via
    ``--extra`` / ``--group`` (which would make uv lock try to resolve
    torch + paddle together)."""
    if not DOCKERFILE.exists():
        pytest.skip("Dockerfile not present")
    text = _read(DOCKERFILE)
    # The sidecar target must reference the PyTorch cu124 index URL.
    assert "download.pytorch.org/whl/cu124" in text, (
        "Dockerfile's sidecar target must install torch from the "
        "PyTorch cu124 index."
    )
    assert re.search(r"FROM\s+\S+\s+AS\s+sidecar", text), (
        "Dockerfile must define a `sidecar` stage "
        "(FROM <base> AS sidecar)."
    )
    assert not re.search(r"--group\s+eval\b", text), (
        "Dockerfile must NOT use --group eval (torch and paddle "
        "cannot coexist in one uv lock)."
    )
    assert not re.search(r"--extra\s+eval\b", text), (
        "Dockerfile must NOT use --extra eval (same reason)."
    )


def test_dockerfile_api_target_does_not_install_torch() -> None:
    """RED: the ``api`` target must NOT pull torch. If it did, the
    runtime separation would collapse (cuDNN 9.x would land in the
    api venv and break paddle's MOT init).
    """
    if not DOCKERFILE.exists():
        pytest.skip("Dockerfile not present")
    text = _read(DOCKERFILE)
    # Split the Dockerfile on stage boundaries. The api stage is
    # between `FROM <base> AS api` and the next `FROM ... AS` (or EOF).
    m = re.search(r"FROM\s+\S+\s+AS\s+api\b(.*?)(?=^FROM\s|\Z)", text, re.DOTALL | re.MULTILINE)
    assert m, "Dockerfile must define an `api` stage"
    api_block = m.group(1)
    # Only check RUN / COPY lines (comments may mention torch in prose).
    run_lines = "\n".join(
        line for line in api_block.splitlines()
        if re.match(r"^\s*(RUN|COPY|ADD)\b", line)
    )
    assert "torch" not in run_lines, (
        "Dockerfile's `api` stage must NOT install torch in any RUN / "
        "COPY / ADD line. The sidecar stage is the only place torch "
        "lives. Adding torch to the api stage would break the runtime "
        "separation (cuDNN 9.x would shadow paddle's cuDNN 8.x)."
    )


def test_compose_detect_pipeline_builds_api_target() -> None:
    """RED: the compose `detect-pipeline` service must build the `api`
    target of the multi-stage Dockerfile (so torch never lands in the
    api venv)."""
    if not COMPOSE.exists():
        pytest.skip("docker-compose.yaml not present")
    text = _read(COMPOSE)
    m = re.search(
        r"^  detect-pipeline:\s*\n((?:^    .*\n|\n)+?)(?=^  \w|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, "docker-compose.yaml must define an `detect-pipeline:` service"
    block = m.group(1)
    assert re.search(r"^\s*target:\s*api\b", block, re.MULTILINE), (
        "docker-compose.yaml `detect-pipeline:` service must declare "
        "`target: api` (the Paddle-only target of the multi-stage "
        "Dockerfile). Building the sidecar target would pull torch "
        "into the api venv."
    )


def test_compose_embedding_sidecar_builds_sidecar_target() -> None:
    """RED: the compose `embedding-sidecar` service must build the
    `sidecar` target (so torch lands only in the sidecar venv)."""
    if not COMPOSE.exists():
        pytest.skip("docker-compose.yaml not present")
    text = _read(COMPOSE)
    m = re.search(
        r"^  embedding-sidecar:\s*\n((?:^    .*\n|\n)+?)(?=^  \w|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, "docker-compose.yaml must define a `embedding-sidecar:` service"
    block = m.group(1)
    assert re.search(r"^\s*target:\s*sidecar\b", block, re.MULTILINE), (
        "docker-compose.yaml `embedding-sidecar:` service must declare "
        "`target: sidecar` (the torch-enabled target). Building the "
        "api target would lose TransReID inference."
    )
    assert re.search(r"dockerfile:\s*Dockerfile\b", block), (
        "docker-compose.yaml `embedding-sidecar:` service must declare "
        "`dockerfile: Dockerfile` (this repo's Dockerfile)."
    )


def test_compose_eval_service_is_profile_gated() -> None:
    """The compose `replay-eval` service must be profile-gated and use
    the torch-enabled image target with the same host model mount as
    the sidecar."""
    if not COMPOSE.exists():
        pytest.skip("docker-compose.yaml not present")
    text = _read(COMPOSE)
    m = re.search(
        r"^  replay-eval:\s*\n((?:^    .*\n|\n)+?)(?=^  \w|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, "docker-compose.yaml must define a `replay-eval:` service"
    block = m.group(1)
    assert re.search(r"profiles:\s*\[\s*[\"']eval[\"']\s*\]", block), (
        "docker-compose.yaml `replay-eval:` service must declare "
        '`profiles: ["eval"]` so it only comes up with '
        "`docker compose --profile eval up`."
    )
    assert re.search(r"^\s*target:\s*sidecar\b", block, re.MULTILINE), (
        "docker-compose.yaml `replay-eval:` service must build "
        "`target: sidecar`; otherwise the eval profile loses torch."
    )
    assert "./models:/models:ro" in block, (
        "docker-compose.yaml `replay-eval:` must bind-mount the host "
        "./models directory at /models:ro, matching detect-pipeline "
        "and embedding-sidecar."
    )
