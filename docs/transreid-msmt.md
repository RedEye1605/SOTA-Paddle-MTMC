# TransReID MSMT17 setup

> **What the project ships, why MSMT17, and the exact commands
> that produced the on-disk artifact this repo has been
> validated against.**

## Why MSMT17

| Profile     | Train set        | Cameras | Identities | Sensitivity to view change |
|-------------|------------------|---------|------------|---------------------------|
| `market1501`| Market-1501      | 6       | 1,501      | low                       |
| **`msmt17`**| MSMT17           | 15      | 4,101      | **high** (preferred for retail) |
| `duke`     | DukeMTMC-reID    | 8       | 1,812      | medium                    |
| `veri`     | VeRi-776         | n/a     | (vehicle)  | n/a                       |

The Yamaha showrooms have 4–8 cameras with overlapping viewpoints.
MSMT17 is the closest training distribution: its 15-camera training
set + heavier viewpoint variance produces ReID embeddings that
generalize across our installed cameras.

## Active config

`configs/app.yaml`:

```yaml
reid:
  active_model: pphuman_strongbaseline   # the default ships PP-Human ReID
  transreid_profile: msmt17              # used when active_model=transreid
  transreid_weight: models/vit_transreid_msmt.pth
  sie_enabled: false                     # MSMT17 deployment runs WITHOUT
                                          # the SIE camera-side embedding
```

> **SIE must stay disabled** for the MSMT17 path until the operator
> records a multi-camera benchmark that validates the
> camera-aware embeddings.  Enabling SIE silently changes the
> distance metric distribution.

## Download

```bash
# Vendored script — uses the BCE Bos mirrors:
bash scripts/download_transreid_models.sh
```

The script writes:

```text
models/vit_transreid_msmt.pth        # ~420 MB
```

## Verify the checkpoint matches the MSMT17 profile

```bash
uv run python scripts/inspect_transreid_checkpoint.py \
    models/vit_transreid_msmt.pth --profile msmt17
```

Expected output (verified on this host, 2026-06-13):

```text
path: models/vit_transreid_msmt.pth
exists: True
size_bytes: 419206362
classifier: jpm  shape=[4101, 768]
state_dict: 360 tensors
compatible_with_profile: msmt17
```

The `classifier.shape[0] == 4101` is the MSMT17 identity count.  If
this number differs, the checkpoint is not MSMT17 and the
profile must be overridden via
`reid.transreid_profile: custom` plus an explicit `num_classes`.

## Vendor forward pass

The repo ships the TransReID vendor code under
`app/reid/_transreid_native/`.  Quick smoke:

```bash
uv run python -c "
from app.reid._transreid_native import vit_base_patch16_224_TransReID
print('native transreid ok')
"
```

## Adapter health

```bash
uv run python -c "
from app.reid.transreid_adapter import TransReIDAdapter
print('transreid_adapter ok')
"
```

## Hard rules

```text
1. camera_num=0 inference is the ONLY supported MSMT17 path.
2. SIE remains disabled for MSMT17.
3. Production refuses the deterministic / histogram ReID fallback.
4. The checkpoint is NOT shipped in the docker image — it is
   mounted read-only from ./models/vit_transreid_msmt.pth to
   /models/vit_transreid_msmt.pth by docker-compose.yaml.
```
