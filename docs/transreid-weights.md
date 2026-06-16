# TransReID Weight Alignment

> **How to align your TransReID weight with the configured profile.**
> This is the PATCH-011 runbook.

## TL;DR

The repo's `configs/reid/transreid.yaml` is currently set to
`profile: msmt17` with `num_class: 1041` because the on-disk
weight is `models/vit_transreid_msmt.pth` (MSMT17, 1041
identities). If you have a Market-1501 weight instead, edit
the YAML or use the inspector script.

## Inspect your checkpoint

```bash
python scripts/inspect_transreid_checkpoint.py \
    /models/transreid/transformer_120.pth --json
```

Output (Market-1501 weight):

```json
{
  "path": "/models/transreid/transformer_120.pth",
  "exists": true,
  "num_tensors": 412,
  "has_backbone": true,
  "classifier": {
    "kind": "jpm",
    "num_class": 751,
    "embedding_dim": 768
  },
  "detected_num_class": 751,
  "detected_profile": "market1501",
  "expected_num_class": 751,
  "expected_num_class_effective": 751,
  "ok": true,
  "reason": "compatible"
}
```

Output (mismatch — the weight says 1041, you configured 751):

```json
{
  "ok": false,
  "reason": "num_class_mismatch",
  "expected": 751,
  "got": 1041
}
```

## Two ways to align

### 1. Switch the profile in YAML

Edit `configs/reid/transreid.yaml`:

```yaml
profile: market1501         # was: msmt17
weight: /models/transreid/transformer_120.pth
num_class: 751
```

Then verify:

```bash
python scripts/inspect_transreid_checkpoint.py \
    /models/transreid/transformer_120.pth --profile market1501
```

Exit code 0 = compatible. Non-zero = the profile mismatches the
weight and the production-mode `load()` will raise
`ProductionSafetyError` (unless `ignore_classifier_head=true`).

### 2. Use `ignore_classifier_head=true` (recommended for
   inference)

The classifier head is training-only. For inference, you can
safely ignore the head's classifier-shape mismatch and load
only the feature extractor. Set:

```yaml
profile: msmt17
weight: /models/transreid/vit_transreid_msmt.pth
num_class: 1041
ignore_classifier_head: true
require_checkpoint_in_production: true
```

The adapter will:
1. Build the model with `num_class=1041`.
2. Load the checkpoint with `strict=False`. The classifier
   and BNNeck keys are silently dropped.
3. Log a WARNING with the expected vs detected num_class.
4. Mark the load as `ok=True` and continue.

This is the **recommended** mode for inference deployments.

## Plug-in path (alternative)

If the operator prefers the upstream `damo-cv/TransReID`
inference code, they can inject a custom callable via the
`TRANSREID_MODEL_FN` env var:

```python
# my_inference.py
import torch
from model import make_model
from config import cfg

def load_transreid():
    cfg.merge_from_file('configs/Market/vit_transreid_stride.yml')
    cfg.freeze()
    return make_model(cfg, num_class=751, camera_num=6, view_num=0)
```

```bash
export TRANSREID_MODEL_FN=my_inference:load_transreid
python -m app.main
```

The adapter takes the callable and uses it directly, bypassing
the vendored backbone and the inspector.

## Production-mode safety

In production mode (`SOTA_RUNTIME_MODE=production`), the
adapter refuses to start if:

1. The weight file is missing (or `require_checkpoint_in_production=false`
   is set explicitly to allow a plug-in).
2. `num_class` does not match the on-disk classifier head
   AND `ignore_classifier_head=false`.

In smoke-test mode (`--mode smoke_test` or
`ALLOW_SYNTHETIC_SMOKE_TEST=true`), the adapter falls back to
the histogram feature and emits a `[SMOKE-TEST]` log line.
The smoke-test fallback is NEVER available in production.

## Where to download the weights

The official damo-cv/TransReID weights live on Google Drive
(see the upstream README). The MSMT17 weight is the one
shipped in this repo's `models/` directory.

```bash
# MSMT17 — already in models/vit_transreid_msmt.pth
ls -la models/vit_transreid_msmt.pth

# Market-1501 — operator must download
# (see https://github.com/damo-cv/TransReID#testing)
```

The Paddle PP-Human MOT model and the PP-Human StrongBaseline
ReID model are also required for production. See
`Docs/official_paddle_integration.md`.
