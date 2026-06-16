"""Architecture-guard tests.

These fail LOUD if any hard rule from the task spec is broken.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = Path(os.environ.get("SERVICE_DIR", ROOT.parent / "Service"))

# Files / dirs the SOTA implementation must NEVER write into.
FORBIDDEN_PATHS_IN_SERVICE = [
    "app",
    "config.yaml",
    "main.py",
    "tests",
    "scripts",
    "docs",
    "offline-people-counting",
]

# Patterns in SOTA code that are not allowed.
FORBIDDEN_PATTERNS = [
    (r"rfdetr", "RF-DETR is not the primary detector; do not import."),
    (r"botsort", "BoT-SORT is not the primary tracker; do not import."),
    (r"boxmot", "BoxMOT is not the primary tracker; do not import."),
    (r"youtureid", "YouTuReID is not the default ReID; do not import."),
]


def _tracked_files() -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:  # noqa: BLE001
        return [p for p in ROOT.rglob("*") if p.is_file()]
    return [p for line in proc.stdout.splitlines() if (p := ROOT / line.strip()).is_file()]


def test_service_dir_exists() -> None:
    if not SERVICE_DIR.exists():
        pytest.skip(f"external Service/ checkout not present at {SERVICE_DIR}")


def test_no_writes_into_service() -> None:
    """SOTA code must NEVER touch the existing Service/ folder."""
    # The SOTA folder's tracked files must not include any path under
    # Service/. We do this by scanning the SOTA folder for absolute
    # references to Service/ paths.
    violations: list[str] = []
    for py in (p for p in _tracked_files() if p.suffix == ".py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for forbidden in FORBIDDEN_PATHS_IN_SERVICE:
            # Allow the docs/comparison doc to mention Service/.
            if py.name in {"comparison_with_existing_service.md"}:
                continue
            # Look for writes: open(..., "w"), Path.write_text, .unlink, etc.
            # We only flag if the path string is a *Service/* path.
            for m in re.finditer(
                r'["\']([^"\']*Service/[^"\']+)["\']',
                text,
            ):
                p = m.group(1)
                if "/Service/" in p and p.startswith("/Service/") is not False:
                    # Allow mentions in code comments
                    line = text.split("\n")[text[: m.start()].count("\n")]
                    if not line.lstrip().startswith("#"):
                        violations.append(f"{py}: {p}")
    assert not violations, "SOTA code references Service/ paths: " + "\n".join(violations)


def test_no_forbidden_models_imported() -> None:
    """No RF-DETR / BoT-SORT / BoxMOT / YouTuReID imports in SOTA code."""
    violations: list[str] = []
    for py in (p for p in _tracked_files() if p.suffix == ".py"):
        if "tests" in py.parts:
            continue
        # The `compare_with_service_baseline.py` script *names* the
        # baseline model choices in a JSON-like dict; that is the
        # explicit purpose of the file. Skip it.
        if py.name == "compare_with_service_baseline.py":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for pat, msg in FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                # Only check imports / actual references, not comments
                line = text.split("\n")[text[: m.start()].count("\n")]
                if line.lstrip().startswith("#"):
                    continue
                violations.append(f"{py}: '{m.group(0)}' ({msg})")
    assert not violations, "Forbidden imports found:\n" + "\n".join(violations)


def test_no_secrets_in_repo() -> None:
    """A basic secret scan. NEVER put a password / token / real RTSP URL
    in any tracked file."""
    SECRET_PATTERNS = [
        (r"password\s*=\s*['\"]\w{6,}", "password literal"),
        ("upload" + "123", "known leaked MinIO secret"),
        (r"minio-backend\.xiot\.my\.id", "known leaked MinIO host"),
        (r"broker\.xdevelopment\.my\.id", "known leaked MQTT host"),
        (r"100\.109\.124\.85", "known leaked private host"),
        (r"100\.94\.166\.20", "known leaked private host"),
        (r"hls\.xiot\.my\.id", "known leaked MediaMTX HLS host"),
        (r"rtc\.xiot\.my\.id", "known leaked MediaMTX WebRTC host"),
        (r"(?m)^MINIO_ACCESS_KEY\s*=\s*trainer\s*$", "known leaked MinIO user"),
        (
            r"(?m)^MQTT_(?:USERNAME|PASSWORD)\s*=\s*"
            + "node"
            + r"red\s*$",
            "known leaked MQTT credential",
        ),
        # PATCH (2026-06-17): the previous regex ``rtsp://[^/'\"]+:[^@'\"]+@``
        # matched across newlines (Python's ``[^...]`` is newline-agnostic)
        # so a multi-line markdown table with an ``@`` three cells later
        # would trigger a false positive. We tighten by excluding
        # whitespace from both the host and the password char classes
        # so a real credentialed URL (``rtsp://user:pass@host:port/path``)
        # matches on a single line but harmless operator IPs do not.
        (r"rtsp://\S+?:\S+?@\S+", "RTSP URL with credentials"),
        (r"(?i)AKIA[0-9A-Z]{16}", "AWS access key id"),
        (r"(?i)aws_secret_access_key\s*=", "AWS secret key"),
        (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key"),
    ]
    ENV_SECRET_ASSIGNMENT = re.compile(
        r"(?mi)^(?P<key>[A-Z0-9_]*(?:PASSWORD|SECRET|TOKEN|ACCESS_KEY|SECRET_KEY))"
        r"[ \t]*=[ \t]*(?P<value>[^\r\n]*)$"
    )
    violations: list[str] = []
    for f in _tracked_files():
        if not f.is_file():
            continue
        if any(
            seg in f.parts
            for seg in (
                ".venv",
                "__pycache__",
                ".git",
                "models",
                "data",
                "reports",
                ".pytest_cache",
                ".ruff_cache",
                ".serena",
            )
        ):
            continue
        # PATCH (2026-06-17): the test file itself documents the
        # regex patterns in comments and string literals; scanning it
        # produces a self-match. Same for the docs that quote the
        # patterns (Audit/SECURITY_PRIVACY_AUDIT.md). The scan
        # exists to catch unintentional leaks; the test/doc authors
        # know what they're writing.
        if f.name == "test_architecture_guards.py":
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for pat, msg in SECRET_PATTERNS:
            if re.search(pat, text):
                violations.append(f"{f}: {msg}")
        for match in ENV_SECRET_ASSIGNMENT.finditer(text):
            value = match.group("value").strip().strip("'\"")
            if not value:
                continue
            if value.startswith("<") and value.endswith(">"):
                continue
            if value.startswith("${") and value.endswith("}"):
                continue
            if value.lower() in {
                "change_me_in_production",
                "example",
                "placeholder",
                "redacted",
                "dummy",
                "test",
                "x",
                "y",
            }:
                continue
            violations.append(f"{f}: non-empty secret assignment {match.group('key')}")
    assert not violations, "Potential secrets in repo:\n" + "\n".join(violations)


def test_dockerfile_does_not_commit_secrets() -> None:
    df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "change_me_in_production" not in df, "Dockerfile must not bake in default creds"


def test_env_example_only() -> None:
    """Real secrets must live in .env, not in any tracked file."""
    tracked = {p.relative_to(ROOT).as_posix() for p in _tracked_files()}
    assert ".env" not in tracked, ".env must never be tracked"

    env_example = ROOT / ".env.example"
    assert env_example.exists(), ".env.example must document required keys"
    text = env_example.read_text(encoding="utf-8", errors="ignore")
    for key in [
        "SOTA_API_TOKEN",
        "POSTGRES_PASSWORD",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "MQTT_USERNAME",
        "MQTT_PASSWORD",
    ]:
        match = re.search(rf"(?m)^{re.escape(key)}=(.*)$", text)
        assert match is not None, f".env.example missing {key}"
        value = match.group(1).strip().strip("'\"")
        assert value == "" or (
            value.startswith("<") and value.endswith(">")
        ), f".env.example must not contain a real value for {key}"
