"""Logging setup — single source of truth for log format and level."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(
    level: str | int | None = None,
    log_file: Path | None = None,
) -> None:
    """Configure root logger.

    Idempotent: safe to call multiple times. The handler list is replaced on
    each call.
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = level.upper()

    root = logging.getLogger()
    root.setLevel(level)
    # Drop any previously attached handlers (idempotency)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stdout_h = logging.StreamHandler(stream=sys.stdout)
    stdout_h.setFormatter(formatter)
    root.addHandler(stdout_h)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.handlers.RotatingFileHandler(log_file, maxBytes=50_000_000, backupCount=5)
        file_h.setFormatter(formatter)
        root.addHandler(file_h)

    # Quieten noisy libraries
    for noisy in ("urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
