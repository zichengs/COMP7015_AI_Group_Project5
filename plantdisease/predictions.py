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
    # Rebuild a model and load its best checkpoint (which holds all weights).
    model = build_model(config_of(name), pretrained=False)
    state = torch.load(
        os.path.join(run_path(name, seed), "best.pt"), map_location="cpu"
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def fingerprint(images: torch.Tensor) -> str:
    # Short hash of a tensor's content: a cached result is only reused for the same images.
    return hashlib.md5(images.numpy().tobytes()).hexdigest()[:10]  # noqa: S324


def cached_arrays(
    name: str,
    seed: int,
    files: dict[str, torch.Tensor],
    compute: Callable[[nn.Module, torch.Tensor], np.ndarray],
) -> dict[str, np.ndarray]:
    # Results of one run for several image sets, cached as <run>/preds/<file>:
    # files = {file name: images}, and compute(model, images) gives the array of one file.
    # The model is loaded only if a file is missing; later notebooks reuse the files.
    pred_dir = os.path.join(run_path(name, seed), "preds")
    model, results = None, {}
    for file, images in files.items():
        path = os.path.join(pred_dir, file)
        if not os.path.exists(path):
            if model is None:
                model = load_model(name, seed)
            os.makedirs(pred_dir, exist_ok=True)
            np.save(path, compute(model, images))
        results[file] = np.load(path)
    return results


def softmax_probs(model: nn.Module, images: torch.Tensor) -> np.ndarray:
    # Softmax probabilities (N, K) of a model for uint8 images.
    logits, _ = predict(model, eval_loader(images))
    return logits.softmax(1).numpy()


def predict_probs(
    names: Iterable[str],
    image_sets: dict[str, torch.Tensor],
    seeds: Sequence[int] = TRAIN_CONFIG["seeds"],
) -> dict[str, dict[str, np.ndarray]]:
    # Softmax probabilities of every model on every image set: probs[set name][model name] is an
    # array (seeds, N, K). They are cached in <run>/preds/, keyed by the image content.
    file_of = {s: f"{safe_name(s)}_{fingerprint(i)}.npy" for s, i in image_sets.items()}
    files = {file_of[s]: images for s, images in image_sets.items()}
    probs = {s: {} for s in image_sets}
    for name in names:
        per_seed = [cached_arrays(name, seed, files, softmax_probs) for seed in seeds]
        for set_name in image_sets:
            probs[set_name][name] = np.stack([a[file_of[set_name]] for a in per_seed])
    return probs


def predict_labels(
    names: Iterable[str],
    image_sets: dict[str, torch.Tensor],
    seeds: Sequence[int] = TRAIN_CONFIG["seeds"],
) -> dict[str, dict[str, np.ndarray]]:
    # Like predict_probs, but the predicted classes: preds[set name][model name] is (seeds, N).
    probs = predict_probs(names, image_sets, seeds)
    return {
        s: {n: p.argmax(-1) for n, p in by_model.items()}
        for s, by_model in probs.items()
    }
