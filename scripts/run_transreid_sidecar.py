"""Entrypoint for the TransReID sidecar service.

Run with: ``python scripts/run_transreid_sidecar.py``
The sidecar:
  * Loads the real TransReID backbone (vit_base_patch16_224_TransReID)
    with the operator's MSMT17 checkpoint
    (``/models/vit_transreid_msmt.pth``).
  * Consumes ``stream:tracklets`` (the same stream the api's
    TrackletCollector emits to).
  * Writes 3840-dim L2-normalized embeddings to Qdrant
    (``person_reid_transreid_msmt``).
  * Emits a compact summary to ``stream:embeddings`` so the api's
    GlobalIdentityResolver reads the embedding with the
    ``model_name="transreid_msmt"`` discriminator.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Ensure /app is on the import path (the sidecar service's
    # working_dir in compose is /app).
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from app.reid.transreid_sidecar import run_sidecar

    run_sidecar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
