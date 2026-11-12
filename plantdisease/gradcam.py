"""Heat maps of where the models look (Sections 13, 18 and 19.4): Grad-CAM for the CNNs, gradient-
weighted attention for the ViT, and the share of the map that lies on the leaf.
"""

from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from torch import nn

from .config import TRAIN_CONFIG, safe_name
from .data.loaders import eval_tf
from .device import device
from .evaluate import sd
from .experiments import config_of
from .predictions import cached_arrays, fingerprint, load_model

# CNNs: the layer whose feature maps Grad-CAM weights (the ViT uses gradient-weighted attention)
TARGET_LAYER = {
    "cnn": lambda m: m.features,  # last conv block (256 x 14 x 14)
    "se_cnn": lambda m: m.features,
    "resnet18": lambda m: m.layer4,  # last residual stage (512 x 7 x 7)
}
# cache prefix of the per-photo leaf shares; change it when the maps of an architecture change
LEAF_SHARE_PREFIX = {"vit_b_16": "leafshare-gradattn"}


def _attention_with_grad(mha: nn.MultiheadAttention, store: list[torch.Tensor]):
    # Forward of a (batch-first) self-attention layer that computes the attention itself, so that
    # the attention tensor is part of the graph and gets a gradient. torchvision's own call returns
    # only a reshaped copy of the weights, which the output does not depend on. Same result as
    # nn.MultiheadAttention in eval mode (no dropout, no masks).
    def forward(query, key, value, **_kwargs):
        b, t, d = query.shape
        heads = mha.num_heads
        q, k, v = nn.functional.linear(
            query, mha.in_proj_weight, mha.in_proj_bias
        ).chunk(3, -1)
        q, k, v = (
            z.reshape(b, t, heads, d // heads).transpose(1, 2) for z in (q, k, v)
        )
        attention = ((q @ k.transpose(-2, -1)) / (d // heads) ** 0.5).softmax(dim=-1)
        store.append(attention)  # (B, heads, tokens, tokens)
        out = (attention @ v).transpose(1, 2).reshape(b, t, d)
        return mha.out_proj(out), None

    return forward


def gradient_attention(
    model: nn.Module, x: torch.Tensor, batch_size: int = 8
) -> torch.Tensor:
    # Gradient-weighted attention (Chefer et al., 2021) of a torchvision ViT for the predicted class
    # -> maps (B, H, W) in [0, 1]. Every block's attention is weighted by its gradient for the
    # predicted class score (positive part, mean over the heads), and the blocks are accumulated as
    # R <- R + A_bar @ R, starting from R = identity. The class token's row of R gives the relevance
    # of every patch. Unlike plain attention rollout it depends on the class and does not pile up
    # weight on large uniform (e.g. dark) regions.
    layers = [block.self_attention for block in model.encoder.layers]
    maps = []
    for chunk in x.split(batch_size):  # backward through the whole ViT: small chunks
        attentions: list[torch.Tensor] = []
        for mha in layers:
            mha.forward = _attention_with_grad(mha, attentions)
        try:
            with torch.enable_grad(), torch.autocast(device.type, enabled=False):
                logits = model(chunk)
                score = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
                grads = torch.autograd.grad(score, attentions)
        finally:
            for mha in layers:
                del mha.forward  # back to nn.MultiheadAttention.forward
        with torch.no_grad():
            relevance = torch.eye(attentions[0].shape[-1], device=chunk.device)
            relevance = relevance.expand(len(chunk), -1, -1)
            for attention, grad in zip(attentions, grads, strict=True):
                weighted = (grad * attention).clamp_min(0).mean(dim=1)
                relevance = relevance + weighted @ relevance
            maps.append(relevance[:, 0, 1:])
    # the class token's row over the 196 patch tokens, back on the 14 x 14 patch grid
    patches = torch.cat(maps)
    side = int(patches.shape[1] ** 0.5)
    up = nn.functional.interpolate(
        patches.reshape(-1, 1, side, side), size=x.shape[-2:], mode="bilinear"
    )[:, 0]
    # scale every map to [0, 1], like the Grad-CAM maps
    low = up.amin(dim=(1, 2), keepdim=True)
    high = up.amax(dim=(1, 2), keepdim=True)
    return ((up - low) / (high - low).clamp_min(1e-8)).cpu()


def cam_maps(model: nn.Module, arch: str, x: torch.Tensor) -> torch.Tensor:
    # Heat maps (B, H, W) in [0, 1] of (normalised) images: gradient-weighted attention for the ViT,
    # Grad-CAM of the predicted class for the CNNs.
    if arch == "vit_b_16":
        return gradient_attention(model, x)
    with (
        torch.autocast(device.type, enabled=False),
        GradCAM(model=model, target_layers=[TARGET_LAYER[arch](model)]) as cam,
    ):
        return torch.from_numpy(cam(input_tensor=x, targets=None))


def cam_images(name: str, seed: int, images: torch.Tensor) -> torch.Tensor:
    # Heat maps of a model for a few uint8 images.
    model = load_model(name, seed)
    return cam_maps(model, config_of(name).arch, eval_tf(images).to(device))


def leaf_attention(
    model: nn.Module,
    images: torch.Tensor,
    *,
    masks: torch.Tensor,
    arch: str,
    batch_size: int = 32,
) -> np.ndarray:
    # Share of the heat-map weight (Grad-CAM; ViT: gradient-weighted attention) on the leaf, for every image.
    shares = []
    for start in range(0, len(images), batch_size):
        x = eval_tf(images[start : start + batch_size]).to(device)
        cams = cam_maps(model, arch, x)
        on_leaf = (cams * masks[start : start + batch_size].float()).sum(dim=(1, 2))
        total = cams.sum(dim=(1, 2))
        shares.append(
            torch.where(
                total > 1e-6,
                on_leaf / total.clamp_min(1e-6),
                torch.full_like(total, float("nan")),
            )
        )
    return torch.cat(shares).numpy()


def leaf_share_table(
    model_names: Iterable[str],
    tests: dict[str, torch.Tensor],
    masks: torch.Tensor,
    variants: Sequence[str] = (
        "original",
        "background removed",
        "other-class background",
    ),
) -> pd.DataFrame:
    # Mean leaf share over the test photos, per model and test condition (mean ± sd over the seeds).
    # Per-photo shares are cached in <run>/preds/ like the predictions, keyed by images and masks.
    mask_id = fingerprint(masks)
    test_ids = {v: fingerprint(tests[v]) for v in variants}
    rows = []
    for name in model_names:
        arch = config_of(name).arch
        prefix = LEAF_SHARE_PREFIX.get(arch, "leafshare-lib")
        file_of = {
            v: f"{prefix}_{safe_name(v)}_{test_ids[v]}_{mask_id}.npy" for v in variants
        }
        files = {file_of[v]: tests[v] for v in variants}
        shares = partial(leaf_attention, masks=masks, arch=arch)
        per_seed = [
            cached_arrays(name, seed, files, shares) for seed in TRAIN_CONFIG["seeds"]
        ]
        for variant in variants:
            means = [np.nanmean(arrays[file_of[variant]]) for arrays in per_seed]
            rows.append(
                {
                    "model": name,
                    "condition": variant,
                    "leaf share": np.mean(means),
                    "sd": sd(means),
                    "leaf area share": masks.float().mean().item(),
                }
            )
    return pd.DataFrame(rows).set_index(["model", "condition"])
