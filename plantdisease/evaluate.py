import os
import time
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from .config import DATA_CONFIG, TRAIN_CONFIG, run_path
from .device import USE_AMP, device, synchronize
from .models import count_params
from .predictions import load_model
from .train import load_history


# ----------------------------------------------------------------------------
# Metrics and seed statistics
# ----------------------------------------------------------------------------
def macro_f1(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, labels: npt.ArrayLike | None = None
):
    # labels=None: the classes that occur or are predicted.
    # labels given (field photos): average over these classes only; a prediction into
    # any other class is still an error.
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
    preds: dict[str, np.ndarray], y: np.ndarray, labels: npt.ArrayLike | None = None
) -> dict[str, np.ndarray]:
    # Macro-F1 of every seed: preds = {model: predicted classes (seeds, N)} -> {model: (seeds,)}.
    return {
        name: np.array([macro_f1(y, p, labels) for p in model_preds])
        for name, model_preds in preds.items()
    }


def per_class_f1(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    # F1 of every class, mean over the seeds. preds: (seeds, N).
    return np.mean([f1_score(y, p, average=None) for p in preds], axis=0)


def per_class_recall(
    preds: np.ndarray, y: np.ndarray, classes: Iterable[int]
) -> list[float]:
    # Recall of the given classes, mean over the seeds. preds: (seeds, N).
    return [(preds[:, y == c] == c).mean(1).mean() for c in classes]


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


def stable_errors(preds: np.ndarray, y: np.ndarray) -> dict[str, str | int]:
    # Errors of every seed, test photos that every seed gets wrong, and photos wrong at least once.
    # preds: (seeds, N).
    wrong = np.asarray(preds) != y
    return {
        "errors per seed": ", ".join(str(n) for n in wrong.sum(1)),
        "wrong in every seed": int(wrong.all(0).sum()),
        "wrong at least once": int(wrong.any(0).sum()),
    }


def field_table(
    test_preds: dict[str, np.ndarray],
    field_preds: dict[str, np.ndarray],
    y_test: np.ndarray,
    y_field: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Macro-F1 over the classes that occur in the field photos, on the clean test
    # photos of these classes and on the field photos (mean ± sd over the seeds).
    # Returns (table, seed scores on the clean test, seed scores on the field photos).
    labels = np.unique(y_field)
    same = np.isin(y_test, labels)  # clean test photos of the same classes
    clean = seed_scores(
        {name: p[:, same] for name, p in test_preds.items()}, y_test[same], labels
    )
    field = seed_scores(field_preds, y_field, labels)
    models = list(field_preds)
    table = pd.DataFrame(
        {
            "clean test": [mean_sd(clean[n], 3) for n in models],
            "field photos": [mean_sd(field[n], 3) for n in models],
            "drop": [clean[n].mean() - field[n].mean() for n in models],
        },
        index=pd.Index(models, name="model"),
    )
    return table, clean, field


# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
def measure_latency(model: nn.Module, batch_size: int, repeats: int = 20) -> float:
    # Inference time in ms per image on the current device, after a warm-up.
    x = torch.randn(
        batch_size, 3, DATA_CONFIG["img_size"], DATA_CONFIG["img_size"], device=device
    )
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


def inference_cost(name: str) -> dict[str, float]:
    # Parameters, checkpoint size and inference latency of the first seed (ours).
    seed = TRAIN_CONFIG["seeds"][0]
    model = load_model(name, seed)
    checkpoint = os.path.join(run_path(name, seed), "best.pt")
    return {
        "params (M)": count_params(model) / 1e6,
        "size (MB)": os.path.getsize(checkpoint) / 1e6,
        "latency bs=1 (ms)": measure_latency(model, 1),
        "ms/image bs=64": measure_latency(model, 64),
    }


# ----------------------------------------------------------------------------
# Confidence and abstention
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
