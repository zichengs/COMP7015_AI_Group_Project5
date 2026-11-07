"""Metrics, bootstrap intervals, paired comparisons, Grad-CAM and other evaluation tools (Sections 10-16)."""

import time
from itertools import pairwise

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

from .config import CFG, CLASSES, SEEDS, USE_AMP, device
from .data import eval_tf
from .experiments import config_of, countermeasure_names
from .models import LAYER_OF
from .train import load_model

N_BOOT = CFG["n_bootstrap"]  # number of bootstrap resamples
K = len(CLASSES)


# ----------------------------------------------------------------------------
# Section 10.1 / 10.2: macro-F1, bootstrap over leaf groups, paired comparison
# ----------------------------------------------------------------------------
def macro_f1(y_true, y_pred, labels=None, k=K):
    # Macro-F1 from the confusion matrix. Same value as sklearn's f1_score, but fast enough for thousands of resamples.
    # labels=None: like sklearn, ignore classes that neither occur nor are predicted.
    # labels given (Section 15): average over these classes; a prediction into any other class is an error.
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if labels is None:
        labels = (cm.sum(0) + cm.sum(1)) > 0
    return f1[labels].mean()


def leaf_group_members(groups):
    # For every leaf group: the positions of its images.
    _, inverse = np.unique(groups, return_inverse=True)
    return [np.where(inverse == g)[0] for g in range(inverse.max() + 1)]


def bootstrap_resamples(members, n_boot=N_BOOT, seed=0):
    # Every resample draws as many leaf groups as the test set has (with replacement) and keeps all their images.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(members), size=(n_boot, len(members)))
    return [np.concatenate([members[g] for g in row]) for row in draws]


def bootstrap_f1(preds_per_seed, y, resamples, labels=None):
    # Macro-F1 of every resample, averaged over the training seeds.
    return np.array(
        [
            np.mean([macro_f1(y[idx], p[idx], labels) for p in preds_per_seed])
            for idx in resamples
        ]
    )


def paired_difference(scores_a, scores_b):
    # Mean difference of two bootstrap score arrays (same resamples) and its 95% interval.
    diff = scores_a - scores_b
    low, high = np.percentile(diff, [2.5, 97.5])
    return diff.mean(), low, high


def verdict(
    boot_a, boot_b, seeds_a=None, seeds_b=None, *, names=("A", "B"), tolerance=None
):
    # Paired comparison of A and B on the same bootstrap resamples -> (A - B, low, high, verdict).
    # Without seed scores (Section 10.2) only the interval decides. With seed scores (rule of Section 13.1)
    # "clearly better" also needs every seed of one model above every seed of the other.
    # With a tolerance (Section 14.3, clean photos) it asks whether a loss beyond the tolerance is excluded.
    diff, low, high = paired_difference(boot_a, boot_b)
    a, b = names
    if tolerance is not None:
        if low > -tolerance:
            result = f"no loss beyond {tolerance}"
        elif high < 0:
            result = "loss on clean photos"
        else:
            result = f"loss up to {-low:.3f} not excluded"
    elif seeds_a is None:
        if low > 0:
            result = f"{a} better"
        elif high < 0:
            result = f"{b} better"
        else:
            result = "no evidence of a difference"
    elif low > 0 and seeds_a.min() > seeds_b.max():
        result = f"{a} clearly better"
    elif high < 0 and seeds_b.min() > seeds_a.max():
        result = f"{b} clearly better"
    elif low > 0 or high < 0:
        result = f"tendency for {a if diff > 0 else b} (seed ranges overlap)"
    else:
        result = "no evidence of a difference"
    return diff, low, high, result


def seed_range(scores, digits=2):
    return f"{scores.min():.{digits}f}-{scores.max():.{digits}f}"


# ----------------------------------------------------------------------------
# Section 11: compute
# ----------------------------------------------------------------------------
def measure_latency(model, batch_size, repeats=20):
    # Inference time in ms per image on the current device, after a warm-up.
    x = torch.randn(batch_size, 3, CFG["img_size"], CFG["img_size"], device=device)
    sync = (
        torch.cuda.synchronize
        if device.type == "cuda"
        else torch.mps.synchronize
        if device.type == "mps"
        else (lambda: None)
    )
    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.float16, enabled=USE_AMP),
    ):
        for _ in range(3):  # warm-up
            model(x)
        sync()
        tic = time.perf_counter()
        for _ in range(repeats):
            model(x)
        sync()
    return (time.perf_counter() - tic) / repeats * 1000 / batch_size  # ms per image


# ----------------------------------------------------------------------------
# Sections 12.2 / 13.3: Grad-CAM and the share of the attention on the leaf
# ----------------------------------------------------------------------------
def grad_cam(model, layer, x):
    # Return Grad-CAM heat maps (B, H, W) in [0, 1] for the predicted class.
    store = {}

    def hook(module, inputs, output):
        store["act"] = output
        output.register_hook(
            lambda grad: store.update(grad=grad)
        )  # gradient of the score w.r.t. the feature maps

    handle = layer.register_forward_hook(hook)
    model.zero_grad()
    logits = model(x)
    logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    handle.remove()
    weights = store["grad"].mean(
        dim=(2, 3), keepdim=True
    )  # importance of each feature map
    cam = F.relu(
        (weights * store["act"]).sum(1, keepdim=True)
    )  # weighted sum of feature maps
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[
        :, 0
    ]
    return (cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)).detach().cpu()


def grad_cam_images(name, seed, images):
    # Grad-CAM maps of a model for a few uint8 images.
    model = load_model(name, seed)
    x = eval_tf(images).to(device)
    with torch.autocast(device.type, enabled=False):
        return grad_cam(model, LAYER_OF[config_of(name)["arch"]](model), x)


def leaf_attention(model, layer, images, masks, batch_size=64):
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
    model_names,
    tests,
    masks,
    variants=("original", "background removed", "other-class background"),
):
    # Mean leaf share over the test photos, per model and test condition (mean and range over the seeds).
    rows = []
    for name in model_names:
        layer_of = LAYER_OF[config_of(name)["arch"]]
        for variant in variants:
            per_seed = []
            for seed in SEEDS:
                model = load_model(name, seed)
                shares = leaf_attention(model, layer_of(model), tests[variant], masks)
                per_seed.append(np.nanmean(shares))
            rows.append(
                {
                    "model": name,
                    "condition": variant,
                    "leaf share": np.mean(per_seed),
                    "seed min": min(per_seed),
                    "seed max": max(per_seed),
                    "leaf area share": masks.float().mean().item(),
                }
            )
    return pd.DataFrame(rows).set_index(["model", "condition"])


# ----------------------------------------------------------------------------
# Section 15: left-out series and field photos
# ----------------------------------------------------------------------------
def shift_table(models, preds, shift_set, *, y, groups, y_clean, columns):
    # Macro-F1 over the classes that occur in y: on the shifted photos (with bootstrap interval over `groups`)
    # and on the clean test photos of the same classes. preds: {(model, seed, set name): predicted classes}.
    # columns = (clean column, shifted column, interval column). Returns (table, seed scores, bootstrap scores).
    clean_col, shift_col, interval_col = columns
    labels = np.unique(y)
    resamples = bootstrap_resamples(leaf_group_members(groups))
    same = np.isin(y_clean, labels)  # clean test photos of the same classes
    seed_scores, boot, rows = {}, {}, []
    for name in models:
        shifted = [preds[(name, seed, shift_set)] for seed in SEEDS]
        seed_scores[name] = np.array([macro_f1(y, p, labels) for p in shifted])
        boot[name] = bootstrap_f1(shifted, y, resamples, labels)
        low, high = np.percentile(boot[name], [2.5, 97.5])
        clean = np.mean(
            [
                macro_f1(y_clean[same], preds[(name, seed, "test")][same], labels)
                for seed in SEEDS
            ]
        )
        rows.append(
            {
                "model": name,
                clean_col: clean,
                shift_col: seed_scores[name].mean(),
                interval_col: f"[{low:.3f}, {high:.3f}]",
                "seeds": seed_range(seed_scores[name]),
                "drop": clean - seed_scores[name].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("model"), seed_scores, boot


def countermeasure_verdicts(bases, seed_scores, boot, interval_col):
    # Countermeasure and control vs. their baseline (rule of Section 13.1).
    rows = []
    for base in bases:
        for name in countermeasure_names(base):
            diff, low, high, result = verdict(
                boot[name],
                boot[base],
                seed_scores[name],
                seed_scores[base],
                names=("new model", "baseline"),
            )
            rows.append(
                {
                    "recipe": name,
                    "new model - baseline": diff,
                    interval_col: f"[{low:+.3f}, {high:+.3f}]",
                    "verdict": result,
                }
            )
    return pd.DataFrame(rows)


def per_class_recall(models, preds, set_name, y, classes):
    # Recall of every class, mean over the seeds -> {model: [recall per class]}.
    return {
        name: [
            np.mean([(preds[(name, s, set_name)][y == c] == c).mean() for s in SEEDS])
            for c in classes
        ]
        for name in models
    }


# ----------------------------------------------------------------------------
# Section 16: confidence and abstention
# ----------------------------------------------------------------------------
def risk_coverage(probs, y, coverages):
    # Accuracy of the photos that are still answered when the model abstains on the least confident ones.
    correct = (probs.argmax(1) == y)[np.argsort(-probs.max(1), kind="stable")]
    answered = np.maximum(np.round(coverages * len(y)).astype(int), 1)
    return np.cumsum(correct)[answered - 1] / answered


def expected_calibration_error(probs, y, bins=10):
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
