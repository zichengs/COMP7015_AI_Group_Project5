"""Grad-CAM and the share of the attention that lies on the leaf (Sections 13 and 17)."""

from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .config import TRAIN_CONFIG, safe_name
from .data.loaders import eval_tf
from .device import device
from .evaluate import sd
from .experiments import config_of
from .predictions import cached_arrays, fingerprint, load_model

# last convolutional feature maps of every architecture (Grad-CAM needs them, so not the ViT)
LAYER_OF = {
    "cnn": lambda m: m.features,
    "se_cnn": lambda m: m.features,
    "resnet18": lambda m: m.layer4,
}


def grad_cam(model: nn.Module, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Return Grad-CAM heat maps (B, H, W) in [0, 1] for the predicted class.
    store = {}

    def hook(
        module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        # only the layers after `layer` are recorded for the backward pass
        store["act"] = output.detach().requires_grad_()
        return store["act"]

    trainable = [p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    handle = layer.register_forward_hook(hook)
    try:
        logits = model(x)
        score = logits.gather(1, logits.argmax(1, keepdim=True)).sum()
        grad = torch.autograd.grad(score, store["act"])[0]
    finally:
        handle.remove()
        for p, t in zip(model.parameters(), trainable, strict=True):
            p.requires_grad_(t)
    # importance of each feature map, then the weighted sum of the feature maps
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"].detach()).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[:, 0]
    return (cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)).detach().cpu()


def grad_cam_images(name: str, seed: int, images: torch.Tensor) -> torch.Tensor:
    # Grad-CAM maps of a model for a few uint8 images.
    model = load_model(name, seed)
    x = eval_tf(images).to(device)
    with torch.autocast(device.type, enabled=False):
        return grad_cam(model, LAYER_OF[config_of(name).arch](model), x)


def leaf_attention(
    model: nn.Module,
    images: torch.Tensor,
    *,
    masks: torch.Tensor,
    arch: str,
    batch_size: int = 32,
) -> np.ndarray:
    # Share of the Grad-CAM weight that lies on the leaf, for every image.
    layer = LAYER_OF[arch](model)
    shares = []
    for start in range(0, len(images), batch_size):
        x = eval_tf(images[start : start + batch_size]).to(device)
        with torch.autocast(device.type, enabled=False):
            cams = grad_cam(model, layer, x)
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
    file_of = {
        v: f"leafshare_{safe_name(v)}_{fingerprint(tests[v])}_{mask_id}.npy"
        for v in variants
    }
    files = {file_of[v]: tests[v] for v in variants}
    rows = []
    for name in model_names:
        shares = partial(leaf_attention, masks=masks, arch=config_of(name).arch)
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
