#!/usr/bin/env python3
"""Initialize Qdrant collections + payload indexes for SOTA-Paddle-MTMC.

Idempotent — safe to re-run.
"""

from __future__ import annotations

import logging
import os
import sys

from app.storage.qdrant_store import from_env

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("init_qdrant")


def main() -> int:
    store = from_env()
    store.connect()
    if not store.healthcheck():
        log.error("Qdrant healthcheck failed")
        return 1
    store.init_collections()
    cols = store.client.get_collections()
    log.info("Qdrant ready. Collections: %s", [c.name for c in cols.collections])
    return 0


if __name__ == "__main__":
    sys.exit(main())
