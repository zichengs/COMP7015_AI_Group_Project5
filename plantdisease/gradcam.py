"""Class activation maps (Grad-CAM, Score-CAM) and the share of the attention that lies on the leaf
(Sections 13 and 19). The maps are computed with the pytorch-grad-cam library (`grad-cam` on PyPI)."""

import math
from collections.abc import Iterable, Sequence
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, score_cam
from torch import nn

from .config import TRAIN_CONFIG, safe_name
from .data.loaders import eval_tf
from .device import device
from .evaluate import sd
from .experiments import config_of
from .predictions import cached_arrays, fingerprint, load_model

# the layer whose feature maps are weighted, for every architecture
TARGET_LAYER = {
    "cnn": lambda m: m.features,  # last conv block (256 x 14 x 14)
    "se_cnn": lambda m: m.features,
    "resnet18": lambda m: m.layer4,  # last residual stage (512 x 7 x 7)
    # ViT: LayerNorm of the last encoder block; its 196 patch tokens form a 14 x 14 grid
    "vit_b_16": lambda m: m.encoder.layers[-1].ln_1,
}

METHODS = {"gradcam": GradCAM, "scorecam": ScoreCAM}
SCORECAM_BATCH = 32  # masked images per forward pass of Score-CAM
# Score-CAM shows a tqdm progress bar for every image, which floods the notebook output: turn it off
score_cam.tqdm = SimpleNamespace(tqdm=lambda iterable, **_: iterable)


def vit_reshape(tokens: torch.Tensor) -> torch.Tensor:
    # ViT tokens (B, 1 + 196, D) -> feature maps (B, D, 14, 14): drop the class token and put the
    # patch tokens back on their image grid, so that the CAM methods can treat them like conv maps.
    patches = tokens[:, 1:, :]
    side = math.isqrt(patches.shape[1])
    return patches.reshape(patches.shape[0], side, side, -1).permute(0, 3, 1, 2)


RESHAPE = {"vit_b_16": vit_reshape}


def cam_maps(
    model: nn.Module, arch: str, x: torch.Tensor, method: str = "gradcam"
) -> torch.Tensor:
    # Heat maps (B, H, W) in [0, 1] for the predicted class of every (normalised) image in x.
    # Score-CAM masks the image with every feature map at once, so it runs one image at a time.
    chunks = [x] if method == "gradcam" else list(x.split(1))
    maps = []
    with (
        torch.autocast(device.type, enabled=False),
        METHODS[method](
            model=model,
            target_layers=[TARGET_LAYER[arch](model)],
            reshape_transform=RESHAPE.get(arch),
        ) as cam,
    ):
        cam.batch_size = SCORECAM_BATCH  # used by Score-CAM only
        for chunk in chunks:
            maps.append(torch.from_numpy(cam(input_tensor=chunk, targets=None)))
    return torch.cat(maps)


def cam_images(
    name: str, seed: int, images: torch.Tensor, method: str = "gradcam"
) -> torch.Tensor:
    # Heat maps ('gradcam' or 'scorecam') of a model for a few uint8 images.
    model = load_model(name, seed)
    return cam_maps(model, config_of(name).arch, eval_tf(images).to(device), method)


def leaf_attention(
    model: nn.Module,
    images: torch.Tensor,
    *,
    masks: torch.Tensor,
    arch: str,
    batch_size: int = 32,
) -> np.ndarray:
    # Share of the Grad-CAM weight that lies on the leaf, for every image.
    shares = []
    for start in range(0, len(images), batch_size):
        x = eval_tf(images[start : start + batch_size]).to(device)
        cams = cam_maps(model, arch, x, "gradcam")
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
        v: f"leafshare-lib_{safe_name(v)}_{fingerprint(tests[v])}_{mask_id}.npy"
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
