"""Heat maps of where the models look (Sections 13, 18 and 19.4): Grad-CAM for the CNNs, attention
rollout for the ViT, and the share of the map that lies on the leaf.
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

# CNNs: the layer whose feature maps Grad-CAM weights (the ViT uses attention rollout instead)
TARGET_LAYER = {
    "cnn": lambda m: m.features,  # last conv block (256 x 14 x 14)
    "se_cnn": lambda m: m.features,
    "resnet18": lambda m: m.layer4,  # last residual stage (512 x 7 x 7)
}
# cache prefix of the per-photo leaf shares; change it when the maps of an architecture change
LEAF_SHARE_PREFIX = {"vit_b_16": "leafshare-rollout"}


def attention_rollout(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Attention rollout (Abnar & Zuidema, 2020) of a torchvision ViT -> maps (B, H, W) in [0, 1].
    # Every encoder block mixes the tokens by its attention plus the residual connection, so one
    # block is 0.5 * attention (mean over the heads) + 0.5 * identity. Multiplying the blocks gives
    # how much each patch flows into the class token. The map does not depend on the class.
    attentions = []

    def ask_weights(_module, args, kwargs):
        # torchvision calls the attention with need_weights=False: ask for the weights of every head
        return args, {**kwargs, "need_weights": True, "average_attn_weights": False}

    def keep_weights(_module, _inputs, output):
        attentions.append(output[1])  # (B, heads, tokens, tokens)

    attention_layers = [block.self_attention for block in model.encoder.layers]
    hooks = [
        m.register_forward_pre_hook(ask_weights, with_kwargs=True)
        for m in attention_layers
    ]
    hooks += [m.register_forward_hook(keep_weights) for m in attention_layers]
    try:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            model(x)
    finally:
        for hook in hooks:
            hook.remove()

    identity = torch.eye(attentions[0].shape[-1], device=x.device)
    rollout = identity.expand(len(x), -1, -1)
    for attention in attentions:
        mixed = 0.5 * attention.mean(dim=1) + 0.5 * identity
        mixed = mixed / mixed.sum(dim=-1, keepdim=True)
        rollout = mixed @ rollout
    # the class token's row over the 196 patch tokens, back on the 14 x 14 patch grid
    patches = rollout[:, 0, 1:]
    side = int(patches.shape[1] ** 0.5)
    maps = nn.functional.interpolate(
        patches.reshape(-1, 1, side, side), size=x.shape[-2:], mode="bilinear"
    )[:, 0]
    # scale every map to [0, 1], like the Grad-CAM maps
    low = maps.amin(dim=(1, 2), keepdim=True)
    high = maps.amax(dim=(1, 2), keepdim=True)
    return ((maps - low) / (high - low).clamp_min(1e-8)).cpu()


def cam_maps(model: nn.Module, arch: str, x: torch.Tensor) -> torch.Tensor:
    # Heat maps (B, H, W) in [0, 1] of (normalised) images: attention rollout for the ViT,
    # Grad-CAM of the predicted class for the CNNs.
    if arch == "vit_b_16":
        return attention_rollout(model, x)
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
    # Share of the heat-map weight (Grad-CAM; ViT: attention rollout) on the leaf, for every image.
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
