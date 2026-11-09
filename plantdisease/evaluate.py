"""Metrics, seed statistics, compute, Grad-CAM and other evaluation tools (Sections 9-20).

Uncertainty follows Zhouyang's protocol: every model is trained with the seeds of TRAIN_CONFIG["seeds"] on one
fixed split; results are the mean ± sample standard deviation over these seeds. Two models trained
with the same seed are compared seed by seed ("higher in k of n seeds"). The standard deviation
covers the randomness of training, not the choice of test leaves.
"""

import os
import time
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import f1_score
from torch import nn

from .config import DATA_CONFIG, TRAIN_CONFIG, safe_name
from .data.loaders import eval_tf
from .device import USE_AMP, device, synchronize
from .experiments import config_of
from .models import LAYER_OF, count_params
from .predictions import cached_arrays, fingerprint, load_model
from .train import load_history


# ----------------------------------------------------------------------------
# Metrics and seed statistics
# ----------------------------------------------------------------------------
def macro_f1(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, labels: npt.ArrayLike | None = None
):
    # labels=None: the classes that occur or are predicted.
    # labels given (Sections 19, 25-28): average over these classes; a prediction into any other
    # class is an error.
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def sd(values: npt.ArrayLike) -> float | np.floating:
    # Sample standard deviation (0 for a single seed).
    values = np.asarray(values, dtype=float)
    return values.std(ddof=1) if len(values) > 1 else 0.0


def mean_sd(values: npt.ArrayLike, digits: int = 4, scale: float = 1) -> str:
    # 'mean ± sd' over the seeds, e.g. '0.9928 ± 0.0011'.
    values = np.asarray(values, dtype=float) * scale
    return f"{values.mean():.{digits}f} ± {sd(values):.{digits}f}"


def seed_scores(
    names: Iterable[str],
    preds: dict[tuple[str, int, str], np.ndarray],
    set_name: str,
    y: np.ndarray,
    labels: npt.ArrayLike | None = None,
) -> dict[str, np.ndarray]:
    # Macro-F1 of every seed: {model: array over the seeds}. preds: {(model, seed, set name): predicted classes}.
    return {
        name: np.array([macro_f1(y, preds[(name, s, set_name)], labels) for s in TRAIN_CONFIG["seeds"]])
        for name in names
    }


def compare_seeds(
    scores_a: npt.ArrayLike, scores_b: npt.ArrayLike
) -> dict[str, float | np.floating | str]:
    # Same-seed comparison of two models trained with the same seeds (same split, same seed order).
    diff = np.asarray(scores_a) - np.asarray(scores_b)
    return {
        "A - B (mean)": diff.mean(),
        "A - B (sd)": sd(diff),
        "A higher in": f"{(diff > 0).sum()} of {len(diff)} seeds",
    }


def stable_errors(
    preds_per_seed: Sequence[np.ndarray], y: np.ndarray
) -> dict[str, str | int]:
    # Errors of every seed, test photos that every seed gets wrong, and photos wrong at least once.
    wrong = np.array([p != y for p in preds_per_seed])
    return {
        "errors per seed": ", ".join(str(n) for n in wrong.sum(1)),
        "wrong in every seed": int(wrong.all(0).sum()),
        "wrong at least once": int(wrong.any(0).sum()),
    }


# ----------------------------------------------------------------------------
# Compute: inference (ours) and training cost (Zhouyang)
# ----------------------------------------------------------------------------
def measure_latency(model: nn.Module, batch_size: int, repeats: int = 20) -> float:
    # Inference time in ms per image on the current device, after a warm-up.
    x = torch.randn(batch_size, 3, DATA_CONFIG["img_size"], DATA_CONFIG["img_size"], device=device)
    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.float16, enabled=USE_AMP),
    ):
        for _ in range(3):  # warm-up
            model(x)
        synchronize()
        tic = time.perf_counter()
        for _ in range(repeats):
            model(x)
        synchronize()
    return (time.perf_counter() - tic) / repeats * 1000 / batch_size  # ms per image


def training_cost(name: str) -> dict[str, Any]:
    # Training cost over the seeds: trainable parameters (fine-tuning stage), epochs trained (early
    # stopping), and training time without validation (device-synchronised).
    histories = [load_history(name, s) for s in TRAIN_CONFIG["seeds"]]
    return {
        "trainable params": histories[0].trainable_params[-1],
        "epochs trained": mean_sd([len(h.train_time) for h in histories], 1),
        "train time (s)": mean_sd([sum(h.train_time) for h in histories], 1),
        "train s/epoch": np.mean([np.mean(h.train_time) for h in histories]),
    }


def inference_cost(name: str, checkpoint: str) -> dict[str, float]:
    # Parameters, checkpoint size and inference latency of the first seed (ours).
    model = load_model(name, TRAIN_CONFIG["seeds"][0])
    return {
        "params (M)": count_params(model) / 1e6,
        "size (MB)": os.path.getsize(checkpoint) / 1e6,
        "latency bs=1 (ms)": measure_latency(model, 1),
        "ms/image bs=64": measure_latency(model, 64),
    }


# ----------------------------------------------------------------------------
# Grad-CAM and the share of the attention on the leaf (Sections 13 and 17)
# ----------------------------------------------------------------------------
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
    weights = grad.mean(dim=(2, 3), keepdim=True)  # importance of each feature map
    cam = F.relu(
        (weights * store["act"].detach()).sum(1, keepdim=True)
    )  # weighted sum of feature maps
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[
        :, 0
    ]
    return (cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)).detach().cpu()


def grad_cam_images(name: str, seed: int, images: torch.Tensor) -> torch.Tensor:
    # Grad-CAM maps of a model for a few uint8 images.
    model = load_model(name, seed)
    x = eval_tf(images).to(device)
    with torch.autocast(device.type, enabled=False):
        return grad_cam(model, LAYER_OF[config_of(name).arch](model), x)


def leaf_attention(
    model: nn.Module,
    layer: nn.Module,
    images: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int = 32,
) -> np.ndarray:
    # Share of the Grad-CAM weight that lies on the leaf, for every image.
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
    rows = []
    mask_id = fingerprint(masks)
    files = {
        v: f"leafshare_{safe_name(v)}_{fingerprint(tests[v])}_{mask_id}.npy"
        for v in variants
    }
    for name in model_names:
        layer_of = LAYER_OF[config_of(name).arch]
        jobs = {
            files[v]: lambda model, images=tests[v], layer_of=layer_of: leaf_attention(
                model, layer_of(model), images, masks
            )
            for v in variants
        }
        arrays = [cached_arrays(name, seed, jobs) for seed in TRAIN_CONFIG["seeds"]]
        for variant in variants:
            per_seed = [np.nanmean(a[files[variant]]) for a in arrays]
            rows.append(
                {
                    "model": name,
                    "condition": variant,
                    "leaf share": np.mean(per_seed),
                    "sd": sd(per_seed),
                    "leaf area share": masks.float().mean().item(),
                }
            )
    return pd.DataFrame(rows).set_index(["model", "condition"])


# ----------------------------------------------------------------------------
# Section 19: left-out series and field photos
# ----------------------------------------------------------------------------
def shift_table(
    models: Sequence[str],
    preds: dict[tuple[str, int, str], np.ndarray],
    shift_set: str,
    *,
    y: np.ndarray,
    y_clean: np.ndarray,
    columns: tuple[str, str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Macro-F1 over the classes that occur in y: on the shifted photos and on the clean test photos
    # of the same classes (mean ± sd over the seeds). preds: {(model, seed, set name): predicted classes}.
    # columns = (clean column, shifted column).
    # Returns (table, seed scores on the clean photos, seed scores on the shifted photos).
    clean_col, shift_col = columns
    labels = np.unique(y)
    same = np.isin(y_clean, labels)  # clean test photos of the same classes
    shifted = seed_scores(models, preds, shift_set, y, labels)
    clean = {
        name: np.array(
            [
                macro_f1(y_clean[same], preds[(name, s, "test")][same], labels)
                for s in TRAIN_CONFIG["seeds"]
            ]
        )
        for name in models
    }
    table = pd.DataFrame(
        {
            clean_col: [mean_sd(clean[n], 3) for n in models],
            shift_col: [mean_sd(shifted[n], 3) for n in models],
            "drop": [clean[n].mean() - shifted[n].mean() for n in models],
        },
        index=pd.Index(models, name="model"),
    )
    return table, clean, shifted


def comparison_table(
    pairs: Iterable[tuple[str, str]], scores: dict[str, np.ndarray]
) -> pd.DataFrame:
    # Same-seed comparisons: pairs = [(new model, baseline)], scores = {model: seed scores}.
    return pd.DataFrame(
        [
            {"model": new, "baseline": base, **compare_seeds(scores[new], scores[base])}
            for new, base in pairs
        ]
    )


def per_class_recall(
    models: Iterable[str],
    preds: dict[tuple[str, int, str], np.ndarray],
    set_name: str,
    y: np.ndarray,
    classes: Iterable[int],
) -> dict[str, list[float]]:
    # Recall of every class, mean over the seeds -> {model: [recall per class]}.
    return {
        name: [
            np.mean([(preds[(name, s, set_name)][y == c] == c).mean() for s in TRAIN_CONFIG["seeds"]])
            for c in classes
        ]
        for name in models
    }


# ----------------------------------------------------------------------------
# Section 20: confidence and abstention
# ----------------------------------------------------------------------------
def risk_coverage(
    probs: np.ndarray, y: np.ndarray, coverages: np.ndarray
) -> np.ndarray:
    # Accuracy of the photos that are still answered when the model abstains on the least confident ones.
    correct = (probs.argmax(1) == y)[np.argsort(-probs.max(1), kind="stable")]
    answered = np.maximum(np.round(coverages * len(y)).astype(int), 1)
    return np.cumsum(correct)[answered - 1] / answered


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, bins: int = 10
) -> float:
    # Average gap between confidence and accuracy, weighted by the share of photos in each bin.
    confidence, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in pairwise(edges):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(
                confidence[in_bin].mean() - correct[in_bin].mean()
            )
    return ece
