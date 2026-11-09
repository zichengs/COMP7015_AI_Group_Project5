"""Training engine (Section 8): run/load one configuration and seed, predictions with an on-disk cache."""

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .backgrounds import BackgroundSwapDataset
from .config import RUNS_DIR, SEEDS
from .data import CachedDataset, Data, eval_loader, eval_tf, make_loader, train_tf
from .device import USE_AMP, device, set_seed, synchronize
from .experiments import config_of
from .models import build_model, count_params, is_head, set_trainable


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


def safe_name(name: str) -> str:
    # File-system safe version of a run or test-set name, e.g. 'ViT-B/16' -> 'ViT-B_16'.
    return re.sub(r"[^\w.=+-]+", "_", name)


def run_path(name: str, seed: int) -> str:
    # Folder of one run.
    return os.path.join(RUNS_DIR, safe_name(name) + f"__seed{seed}")


def train_dataset(cfg: dict[str, Any], data: Data) -> CachedDataset:
    # Training set of a configuration: its input version, augmentation and (Section 18) background swap.
    transform = train_tf if cfg["augment"] else eval_tf
    images = data.input_images(cfg.get("input", "color"))
    if cfg.get("background_swap", 0.0) > 0:
        masks, swap_ok = data.swap_masks
        return BackgroundSwapDataset(
            images,
            data.labels,
            data.train_idx,
            transform,
            p=cfg["background_swap"],
            masks=masks,
            swap_ok=swap_ok,
            class_matched=cfg.get("class_matched_background", False),
        )
    return CachedDataset(images, data.labels, data.train_idx, transform)


def make_optimizer(model: nn.Module, cfg: dict[str, Any], stage: str) -> AdamW:
    # AdamW over the trainable parameters. In the fine-tuning stage the classifier head and the
    # backbone can have different learning rates (ViT: head 1e-4, backbone 1e-5).
    head = [
        p for n, p in model.named_parameters() if p.requires_grad and is_head(cfg, n)
    ]
    rest = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not is_head(cfg, n)
    ]
    if stage == "head":
        groups = [{"params": head, "lr": cfg["head_lr"]}]
    else:
        groups = [
            {"params": head, "lr": cfg.get("finetune_head_lr", cfg["lr"])},
            {"params": rest, "lr": cfg["lr"]},
        ]
    groups = [g for g in groups if g["params"]]
    return AdamW(groups, weight_decay=cfg["weight_decay"])


def load_finished(
    run_dir: str, cfg: dict[str, Any], data: Data
) -> dict[str, Any] | None:
    # History of a finished run, or None if it was not trained with this configuration and data.
    history_file = os.path.join(run_dir, "history.json")
    if not os.path.exists(history_file):
        return None
    with open(history_file) as f:
        saved = json.load(f)
    if saved["config"] == cfg and saved.get("data") == data.data_id:
        return saved["history"]
    return None


def start_stage(
    model: nn.Module, cfg: dict[str, Any], epoch: int, steps_per_epoch: int
) -> tuple[str, list[nn.Parameter], AdamW, CosineAnnealingLR | None]:
    # Start of a training stage ("head" or "finetune"): set the trainable parameters and
    # (re)build the optimizer and scheduler. The cosine schedule covers the stage.
    stage = "head" if epoch < cfg["head_epochs"] else "finetune"
    set_trainable(model, cfg, stage)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, cfg, stage)
    scheduler = None
    if cfg["scheduler"] == "cosine":
        stage_end = cfg["head_epochs"] if stage == "head" else cfg["epochs"]
        scheduler = CosineAnnealingLR(
            optimizer, T_max=steps_per_epoch * (stage_end - epoch)
        )
    return stage, params, optimizer, scheduler


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None,
    scaler: torch.amp.GradScaler,
    params: list[nn.Parameter],
) -> tuple[float, float, float]:
    # One training epoch. Returns loss, accuracy and the time (timed without validation, after the
    # device has finished).
    model.train()
    running_loss, running_correct = 0.0, 0
    synchronize()
    tic = time.perf_counter()
    for x_cpu, y_cpu in loader:
        x, y = (
            x_cpu.to(device, non_blocking=True),
            y_cpu.to(device, non_blocking=True),
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=USE_AMP):
            outputs = model(x)
            loss = criterion(outputs, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
        running_loss += loss.item() * x.size(0)
        running_correct += (outputs.argmax(1) == y).sum().item()
    synchronize()
    seconds = time.perf_counter() - tic
    n = len(loader.dataset)
    return running_loss / n, running_correct / n, seconds


def validate(model: nn.Module, loader: DataLoader) -> tuple[float, float, float, float]:
    # Validation loss, accuracy, macro-F1 and the time.
    tic = time.perf_counter()
    logits, labels = predict(model, loader)
    seconds = time.perf_counter() - tic
    preds = logits.argmax(1)
    return (
        F.cross_entropy(logits, labels).item(),
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="macro"),
        seconds,
    )


def run(name: str, cfg: dict[str, Any], seed: int, data: Data) -> dict[str, Any]:
    # Train one configuration with one seed (or load it if it was trained before). Returns the history.
    run_dir = run_path(name, seed)
    finished = load_finished(run_dir, cfg, data)
    if finished is not None:
        print(f"Loaded finished run: {name} (seed {seed})")
        return finished

    print(f"\n===== {name} (seed {seed}) =====")
    # stale predictions
    shutil.rmtree(os.path.join(run_dir, "preds"), ignore_errors=True)
    set_seed(seed)
    model = build_model(cfg).to(device)
    train_loader = make_loader(train_dataset(cfg, data), cfg["batch_size"], seed=seed)
    val_images = data.input_images(
        cfg.get("input", "color")
    )  # validate on the same input version
    val_loader = eval_loader(val_images, data.labels, data.val_idx)
    criterion = nn.CrossEntropyLoss(
        weight=data.class_weights.to(device) if cfg["class_weights"] else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history = {
        k: []
        for k in (
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_f1",
            "lr",
            "train_time",
            "val_time",
            "epoch_time",
            "trainable_params",
        )
    }
    best_f1, best_epoch = -1.0, -1
    # early stopping is only considered in the last training stage (after the head-only epochs)
    last_stage_start = cfg["head_epochs"] if cfg["head_epochs"] < cfg["epochs"] else 0
    patience = cfg.get("patience", 0)

    for epoch in range(cfg["epochs"]):
        if epoch in (0, cfg["head_epochs"]):
            stage, params, optimizer, scheduler = start_stage(
                model, cfg, epoch, len(train_loader)
            )

        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["trainable_params"].append(count_params(model, trainable_only=True))
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, params
        )
        val_loss, val_acc, val_f1, val_time = validate(model, val_loader)
        for key, value in {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_time": train_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_time": val_time,
            "epoch_time": train_time + val_time,
        }.items():
            history[key].append(value)
        print(
            f"Epoch {epoch + 1}/{cfg['epochs']} [{stage}] | train loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1]:.4f}"
            f" | val loss {history['val_loss'][-1]:.4f} acc {history['val_acc'][-1]:.4f} macro-F1 {history['val_f1'][-1]:.4f}"
            f" | {history['epoch_time'][-1]:.0f}s"
        )

        # checkpoint the best epoch (ties keep the earlier one)
        if history["val_f1"][-1] > best_f1:
            best_f1, best_epoch = history["val_f1"][-1], epoch
            os.makedirs(run_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

        # early stopping: `patience` epochs of the last stage without a better validation macro-F1
        epochs_without_gain = epoch - max(best_epoch, last_stage_start - 1)
        if patience and epoch + 1 < cfg["epochs"] and epochs_without_gain >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break

    history["stopped_early"] = len(history["val_f1"]) < cfg["epochs"]
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({"config": cfg, "data": data.data_id, "history": history}, f)
    print(
        f"Best val macro-F1: {best_f1:.4f} (epoch {best_epoch + 1} of {len(history['val_f1'])})"
    )
    return history


def run_seeds(
    name: str, data: Data, seeds: Sequence[int] = SEEDS
) -> list[dict[str, Any]]:
    # Train (or load) a run name with every training seed.
    return [run(name, config_of(name), seed, data) for seed in seeds]


def load_model(name: str, seed: int) -> nn.Module:
    # Rebuild a model and load its best checkpoint.
    # the checkpoint holds all weights
    model = build_model(config_of(name), pretrained=False)
    state = torch.load(
        os.path.join(run_path(name, seed), "best.pt"), map_location="cpu"
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def load_history(name: str, seed: int) -> dict[str, Any]:
    with open(os.path.join(run_path(name, seed), "history.json")) as f:
        return json.load(f)["history"]


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
    seeds: Sequence[int] = SEEDS,
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
    seeds: Sequence[int] = SEEDS,
) -> dict[tuple[str, int, str], np.ndarray]:
    # Like predict_probs, but the predicted classes.
    return {k: p.argmax(1) for k, p in predict_probs(names, image_sets, seeds).items()}
