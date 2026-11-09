"""Using finished runs: rebuild a model from its checkpoint and predict, with an on-disk cache."""

import hashlib
import os
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TRAIN_CONFIG, run_path, safe_name
from .data.loaders import eval_loader
from .device import USE_AMP, device
from .experiments import config_of
from .models import build_model


def predict(model: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    # Return logits and labels for a whole loader.
    model.eval()
    all_logits, all_labels = [], []
    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.float16, enabled=USE_AMP),
    ):
        for x, y in loader:
            all_logits.append(model(x.to(device)).float().cpu())
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def load_model(name: str, seed: int) -> nn.Module:
    # Rebuild a model and load its best checkpoint.
    # the checkpoint holds all weights
    model = build_model(config_of(name), pretrained=False)
    state = torch.load(
        os.path.join(run_path(name, seed), "best.pt"), map_location="cpu"
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def fingerprint(images: torch.Tensor) -> str:
    return hashlib.md5(images.numpy().tobytes()).hexdigest()[:10]  # noqa: S324


def cached_arrays(
    name: str, seed: int, jobs: dict[str, Callable[[nn.Module], np.ndarray]]
) -> dict[str, np.ndarray]:
    # Results of one run, cached as <run>/preds/<file>: jobs = {file name: compute(model) -> array}.
    # The model is loaded only if a file is missing; later notebooks reuse the files.
    pred_dir = os.path.join(run_path(name, seed), "preds")
    model, results = None, {}
    for file, compute in jobs.items():
        path = os.path.join(pred_dir, file)
        if not os.path.exists(path):
            if model is None:
                model = load_model(name, seed)
            os.makedirs(pred_dir, exist_ok=True)
            np.save(path, compute(model))
        results[file] = np.load(path)
    return results


def predict_probs(
    names: Iterable[str],
    image_sets: dict[str, torch.Tensor],
    seeds: Sequence[int] = TRAIN_CONFIG["seeds"],
) -> dict[tuple[str, int, str], np.ndarray]:
    # Softmax probabilities for every (model, seed, test set): {(name, seed, set name): (N, K) array}.
    # They are cached in <run>/preds/, keyed by the image content.
    files = {s: f"{safe_name(s)}_{fingerprint(i)}.npy" for s, i in image_sets.items()}
    jobs = {
        files[s]: lambda model, images=images: predict(model, eval_loader(images))[0]
        .softmax(1)
        .numpy()
        for s, images in image_sets.items()
    }
    probs = {}
    for name in names:
        for seed in seeds:
            arrays = cached_arrays(name, seed, jobs)
            for set_name in image_sets:
                probs[(name, seed, set_name)] = arrays[files[set_name]]
    return probs


def predict_labels(
    names: Iterable[str],
    image_sets: dict[str, torch.Tensor],
    seeds: Sequence[int] = TRAIN_CONFIG["seeds"],
) -> dict[tuple[str, int, str], np.ndarray]:
    # Like predict_probs, but the predicted classes.
    return {k: p.argmax(1) for k, p in predict_probs(names, image_sets, seeds).items()}
