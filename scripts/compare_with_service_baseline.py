#!/usr/bin/env python3
"""Side-by-side comparison: SOTA-Paddle-MTMC vs `Service/`.

Skeleton — real implementation should run both pipelines against a
shared dataset and report metrics. Kept as a separate script so the
two pipelines don't share a Python entry point.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    out_dir = Path(os.environ.get("BENCHMARK_OUT_DIR", "/reports/sota_paddle_mtmct"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "sota_paddle_mtmct": {
            "framework": "paddledetection_pphuman",
            "reid_default": "pphuman_strongbaseline",
            "reid_production": "transreid",
            "vector_store": "qdrant",
            "state_cache": "redis",
            "identity_decision": "5_factor_weighted",
        },
        "service_baseline": {
            "framework": "rfdetr_botsort",
            "reid_default": "youtureid",
            "reid_production": "youtureid",
            "vector_store": "pgvector",
            "state_cache": "ram",
            "identity_decision": "single_threshold_per_camera",
        },
        "note": "Skeleton. Real comparison requires running both pipelines on a shared dataset.",
    }
    out_file = out_dir / "compare_with_service.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Written {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
