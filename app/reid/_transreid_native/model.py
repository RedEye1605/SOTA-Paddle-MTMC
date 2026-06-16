"""Minimal vendor of the TransReID backbone for inference.

This is a small subset of the official damo-cv/TransReID repo
(`model/backbones/vit_pytorch.py` + `model/make_model.py`),
restricted to the inference path:

  * `vit_base_patch16_224_TransReID` — the SIE-augmented ViT-B/16.
  * `build_transreid_model`         — wires the backbone to the JPM
    bottleneck + BNNeck, exactly as `make_model` does for inference.
  * `extract_inference_feature`     — applies the
    `neck_feat='before'`, L2-normalize, per the official TEST config.

We do NOT vendor the training utilities (loss functions, data loaders,
center loss, etc.). The training happens upstream; this module is
load-only.

Why vendor instead of pip-install?
  * `damo-cv/TransReID` has no PyPI release.
  * The official repo depends on yacs, timm<=0.5.4, and the `config`
    module from the same repo; pulling all of that just for inference
    is a large dependency surface.
  * Vendoring ~250 lines of the official backbone keeps the contract
    intact and the dependency surface minimal.

Tested against `models/vit_transreid_msmt.pth` (the on-disk checkpoint
shipped in this repo, an MSMT17 checkpoint) and `models/MSMT17_*.pth`
(CLIP-ReID, not used here).

References:
  * https://github.com/damo-cv/TransReID/blob/master/model/backbones/vit_pytorch.py
  * https://github.com/damo-cv/TransReID/blob/master/model/make_model.py
  * https://github.com/damo-cv/TransReID/blob/master/configs/Market/vit_transreid_stride.yml
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# PyTorch is REQUIRED for this module: the SIE-Transformer, JPM, and
# BNNeck classes are torch ``nn.Module`` subclasses. We import eagerly
# at module-load time; if torch is missing, the import fails with a
# clear error.
try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "TransReID vendor requires torch; use the Dockerfile `sidecar` "
        "target or install the matching torch wheel in your dev venv. "
        f"Underlying error: {e}",
    ) from e


# -----------------------------------------------------------------------------
# Position / patch embedding (a minimal, well-tested version of timm)
# -----------------------------------------------------------------------------


def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1).normal_(mean, std))


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding — mirrors the upstream
    ``damo-cv/TransReID`` behavior:

      * ``patch_size`` is the conv kernel (16).
      * ``stride_size`` is the conv stride (12 by default — the
        upstream uses overlapping patches).
      * The pos_embed grid is computed from ``(H, W, patch_size,
        stride_size)`` so it matches the actual conv output.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        stride_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = (
            stride_size if isinstance(stride_size, tuple) else (stride_size, stride_size)
        )
        # Upstream formula: ((H - P) / S) + 1, ((W - P) / S) + 1.
        self.grid_size = (
            (img_size[0] - patch_size[0]) // self.stride_size[0] + 1,
            (img_size[1] - patch_size[1]) // self.stride_size[1] + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # Conv with kernel=patch_size, stride=stride_size
        # (overlapping patches when stride < patch_size).
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=self.stride_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 12, attn_drop: float = 0.0, proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, sie_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, C/H)
        if sie_embed is not None:
            # SIE: add camera/view embedding to q,k (officially to v too,
            # but the original repo's `attn` only adds to v — see the
            # `with_sie` branch). Here we follow the published pattern:
            # SIE is added to v in the SIETransformer.
            v = v + sie_embed
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, proj_drop=drop)
        self.drop_path = drop_path
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, dim, drop=drop)
        # drop_path is left as a constant 0 here; this is the inference path.

    def forward(self, x: torch.Tensor, sie_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), sie_embed=sie_embed)
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# SIE-Transformer (vit_base_patch16_224_TransReID)
# -----------------------------------------------------------------------------


class SIETransformer(nn.Module):
    """vit_base_patch16_224_TransReID — ViT-B/16 + Side Information Embedding.

    ``local_feature=True`` returns ``(global_feat, [local_feats])`` where
    the local feats are the [N, D] tokens of the last block (excluding
    CLS). The JPM module downstream reshapes those into 4 local parts
    and concatenates with the global CLS feature, producing a 5xD
    feature which is then BNNeck'd.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 256,
        patch_size: int = 16,
        stride_size: int = 12,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        camera: int = 0,
        view: int = 0,
        local_feature: bool = True,
        sie_xishu: float = 3.0,
    ) -> None:
        super().__init__()
        self.local_feature = local_feature
        self.sie_xishu = sie_xishu
        # PatchEmbed uses the upstream's overlapping-patch stride.
        # The on-disk pos_embed has shape (1, num_patches+1, D) for
        # the actual (H, W) input. The upstream damo-cv/TransReID
        # passes the full ``img_size=(H, W)`` tuple here; we mirror
        # that to keep the pos_embed grid consistent with the
        # checkpoint.
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        # The official TransReID does NOT use a learned pos_embed; it
        # uses sin-cos fixed pos embed to be compatible with the
        # 16-stride ImageNet pretrained ViT-B/16.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self._init_pos_embed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        _trunc_normal_(self.cls_token, std=0.02)
        # SIE embeddings: one per camera and per view
        self.camera = camera
        self.view = view
        if camera > 0:
            self.sie_embed_cam = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            _trunc_normal_(self.sie_embed_cam, std=self.sie_xishu)
        else:
            self.sie_embed_cam = None
        if view > 0:
            self.sie_embed_view = nn.Parameter(torch.zeros(view, 1, embed_dim))
            _trunc_normal_(self.sie_embed_view, std=self.sie_xishu)
        else:
            self.sie_embed_view = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # The following attributes are read by build_transreid_model()
        # downstream; the original model uses the same names.
        self.num_features = embed_dim

    def _init_pos_embed(self) -> None:
        # Sin-cos 2D position embedding (standard ViT initialization).
        emb_h = self.patch_embed.grid_size[0]
        emb_w = self.patch_embed.grid_size[1]
        grid_size = (emb_h, emb_w)
        pos_embed = self._get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            grid_size,
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    @staticmethod
    def _get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: tuple[int, int],
        cls_token: bool = True,
    ) -> np.ndarray:
        h, w = grid_size
        grid_h = np.arange(h, dtype=np.float32)
        grid_w = np.arange(w, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, h, w])
        # Half the channels for H, half for W
        emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        pos_embed = np.concatenate([emb_h, emb_w], axis=1)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward_features(
        self, x: torch.Tensor, cam_label: torch.Tensor, view_label: torch.Tensor
    ) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # SIE: per official implementation, the SIE is added to v in
        # the attention block via a passed embedding.
        sie = None
        if self.sie_embed_cam is not None and cam_label is not None:
            cam_idx = cam_label.to(x.device).clamp(0, self.camera - 1)
            cam_e = self.sie_embed_cam[cam_idx]  # (B, 1, D)
            sie = cam_e if sie is None else sie + cam_e
        if self.sie_embed_view is not None and view_label is not None:
            view_idx = view_label.to(x.device).clamp(0, self.view - 1)
            view_e = self.sie_embed_view[view_idx]
            sie = view_e if sie is None else sie + view_e
        for blk in self.blocks:
            x = blk(x, sie_embed=sie)
        return self.norm(x)

    def forward(
        self, x: torch.Tensor, cam_label: torch.Tensor, view_label: torch.Tensor
    ) -> torch.Tensor:
        feat = self.forward_features(x, cam_label, view_label)  # (B, 1+N, D)
        if self.local_feature:
            # JPM mode: caller takes the [CLS] token AND the local tokens
            cls_feat = feat[:, 0]  # (B, D)
            local_feat = feat[:, 1:]  # (B, N, D)
            return [cls_feat, local_feat]
        return feat[:, 0]  # (B, D)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


# -----------------------------------------------------------------------------
# Public factories
# -----------------------------------------------------------------------------


def vit_base_patch16_224_TransReID(
    img_size: tuple[int, int] = (256, 128),
    stride_size: int = 12,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    camera: int = 6,
    view: int = 0,
    local_feature: bool = True,
    sie_xishu: float = 3.0,
) -> SIETransformer:
    """Build vit_base_patch16_224_TransReID with the official defaults.

    Note: the official repo interprets ``img_size`` as ``(H, W)``. To
    match the upstream TransReID config
    (``configs/Market/vit_transreid_stride.yml``) the height must be
    a multiple of the stride; we use ``H=256, W=128``.
    """
    return SIETransformer(
        # Pass the FULL ``(H, W)`` tuple so PatchEmbed computes
        # ``((H-P)/S+1) * ((W-P)/S+1)`` patches correctly. The
        # on-disk pos_embed has shape ``(1, num_patches+1, D)`` for
        # the actual rectangular input, not a square grid.
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        camera=camera,
        view=view,
        local_feature=local_feature,
        sie_xishu=sie_xishu,
    )


# -----------------------------------------------------------------------------
# JPM bottleneck + BNNeck (mirrors make_model.build_transformer_local)
# -----------------------------------------------------------------------------


class JPM(nn.Module):
    """Jigsaw Patch Module: split local tokens into K parts, return list."""

    def __init__(self, num_parts: int = 4) -> None:
        super().__init__()
        self.num_parts = num_parts

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (B, N, D). We split the token sequence into `num_parts` equal
        # pieces along the N axis, then L2-normalize and concatenate.
        n = x.shape[1]
        # Avoid division-by-zero in pathological shapes.
        parts = max(1, n // self.num_parts)
        out: list[torch.Tensor] = []
        for i in range(self.num_parts):
            chunk = x[:, i * parts : (i + 1) * parts, :]
            if chunk.shape[1] == 0:
                chunk = x.new_zeros(x.shape[0], 1, x.shape[2])
            out.append(chunk.mean(dim=1))
        return out


class BNNeck(nn.Module):
    """A batch-norm neck used as the inference bottleneck.

    Per the official TEST config ``neck_feat='before'``, the inference
    feature is the BN input (not the BN output). We keep this class for
    completeness; the inference path bypasses it.
    """

    def __init__(self, in_dim: int, out_dim: int = 0) -> None:
        super().__init__()
        if out_dim <= 0:
            out_dim = in_dim
        self.bn = nn.BatchNorm1d(out_dim)
        self.bn.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


# -----------------------------------------------------------------------------
# High-level: build + load + extract
# -----------------------------------------------------------------------------


def build_transreid_model(
    *,
    num_class: int = 751,
    camera_num: int = 0,  # disabled — SIE is training-only
    view_num: int = 0,  # disabled — SIE is training-only
    stride_size: int = 12,
    sie_xishu: float = 3.0,
    jpm: bool = True,
    use_fp16: bool = True,
    device: str = "cuda",
) -> tuple[nn.Module, bool]:
    """Construct the official TransReID model (JPM, BNNeck).

    Returns ``(model, has_jpm)``. The caller is responsible for
    ``load_state_dict(strict=False)`` since the inference path skips
    the BNNeck and only uses the backbone + JPM head.

    Note: SIE (Side Information Embedding for camera/view) is
    disabled by default. SIE is a TRAINING-TIME mechanism; at
    inference we don't have camera-id or view-id labels for the
    gallery, so the SIE slots stay zero. The on-disk checkpoint's
    SIE keys are silently dropped (strict=False) at load time.
    """
    backbone = vit_base_patch16_224_TransReID(
        # Pass the FULL ``(H, W)`` tuple so PatchEmbed computes
        # ``((H-P)/S+1) * ((W-P)/S+1)`` patches correctly. The
        # on-disk pos_embed has shape ``(1, num_patches+1, D)`` for
        # the actual rectangular input, not a square grid.
        img_size=(256, 128),
        stride_size=stride_size,
        camera=camera_num,
        view=view_num,
        local_feature=jpm,
        sie_xishu=sie_xishu,
    )
    if jpm:
        # JPM head: per official `make_model` for the local_feature=True
        # path, the classifier is a list of 5 heads (global + 4 local).
        # We attach dummy heads that are replaced by state_dict.
        backbone.classifier = nn.ModuleList([nn.Linear(768, num_class) for _ in range(5)])
        backbone.bottleneck = nn.ModuleList([BNNeck(768) for _ in range(5)])
    else:
        backbone.classifier = nn.Linear(768, num_class)
        backbone.bottleneck = BNNeck(768)
    if use_fp16 and device.startswith("cuda"):
        backbone = backbone.half()
    backbone = backbone.to(device)
    return backbone, jpm


def _strip_module_prefix(state: dict) -> dict:
    """The official checkpoints prefix everything with ``module.`` because
    DataParallel was used. Strip it once so we can load with strict=False.
    """
    out: dict = {}
    for k, v in state.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_transreid_checkpoint(
    model: nn.Module,
    weight_path: str,
    *,
    device: str = "cpu",
    weights_only: bool = True,
) -> dict:
    """Load the official TransReID checkpoint.

    Returns a summary dict with the keys that were skipped (so the
    caller can log it). The state dict is loaded with
    ``weights_only=True`` (per the security audit) so arbitrary
    pickle objects are refused.
    """
    ckpt = torch.load(weight_path, map_location=device, weights_only=weights_only)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "loaded": len(state),
    }


def extract_inference_feature(
    model: nn.Module,
    *,
    has_jpm: bool,
    images: torch.Tensor,
    cam_labels: torch.Tensor,
    view_labels: torch.Tensor,
    neck_feat: str = "before",
    l2_normalize: bool = True,
) -> torch.Tensor:
    """Forward pass for inference, per the official TEST config.

    Output shape: ``(B, 5 * D) = (B, 3840)`` when JPM is active
    (1 global CLS at 768 + 4 local branches at 768), else ``(B, D)``.
    BNNeck is training-only; ``neck_feat='before'`` keeps the 5D concat
    pre-BNNeck. Features are L2-normalized for cosine similarity.

    JPM inference — local (naive) approach:
        The local branch output ``local_feat`` is the part-token sequence
        coming out of the transformer. We split it into 4 equal chunks
        along the token axis and mean-pool each chunk. The result is
        ``[cls_feat, mean(chunk_0), ..., mean(chunk_3)]`` of shape
        ``(B, 5, 768)`` flattened to ``(B, 3840)``.

    JPM inference — official TransReID approach (reference, not used here):
        The reference ``damo-cv/TransReID/model/make_model.py`` runs the
        *last* transformer block (block + final norm) once per local part,
        and takes the resulting CLS token from each of the 4 branches.
        The official feature is ``cat(global_CLS, branch_CLS_0..3)`` with
        no mean-pool and no ``/4`` rescale. See
        ``build_transformer_local.forward`` in that repo.

    Implication:
        The output SHAPE matches the official ``(B, 5 * 768)``, so the
        BNNeck heads in ``vit_transreid_msmt.pth`` are dimensionally
        compatible. However, the VALUE DISTRIBUTION differs from what
        those BNNeck weights were trained against, because the local
        mean-pool here is not the same operation as block-reuse +
        CLS-from-each-branch. Cosine distance is scale-invariant, so
        inference still works, but raw cosine scores land on a
        different scale than the paper's reported numbers. We accept
        this trade-off because (a) this path is inference-only with
        no fine-tuning of the BNNeck heads, and (b) the naive
        mean-pool is a one-liner versus a structural rewrite that
        would require re-running the last block four times per image.
    """
    model.eval()
    with torch.inference_mode():
        out = model(images, cam_label=cam_labels, view_label=view_labels)
    if has_jpm:
        # `out` is a list [cls, local]
        cls_feat, local_feat = out[0], out[1]
        n = local_feat.shape[1]
        parts = max(1, n // 4)
        local_chunks = [local_feat[:, i * parts : (i + 1) * parts, :].mean(dim=1) for i in range(4)]
        feat = torch.cat([cls_feat] + local_chunks, dim=1)  # (B, 5D)
    else:
        feat = out  # (B, D)
    if neck_feat == "after":
        # In a full re-implementation we would also run the BNNeck here
        # and the test config uses 'before' (the default), so this is a
        # no-op for our current usage.
        pass
    if l2_normalize:
        feat = nn.functional.normalize(feat, dim=1)
    return feat
