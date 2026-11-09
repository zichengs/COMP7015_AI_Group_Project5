"""Training engine (Section 8): train one configuration with one seed, or load it if it was trained before."""

import json
import os
import shutil
import time
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import ValidationError
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import TRAIN_CONFIG, run_path
from .data.backgrounds import BackgroundSwapDataset
from .data.dataset import Data
from .data.loaders import (
    CachedDataset,
    eval_loader,
    eval_tf,
    make_loader,
    train_tf,
)
from .device import USE_AMP, device, set_seed, synchronize
from .experiments import config_of
from .models import build_model, count_params, is_head, set_trainable
from .predictions import predict
from .recipe import RunConfig, RunHistory

# ----------------------------------------------------------------------------
# Setting up a run
# ----------------------------------------------------------------------------


def train_dataset(cfg: RunConfig, data: Data) -> CachedDataset:
    # Training set of a configuration: its input version, augmentation and (Section 18) background swap.
    transform = train_tf if cfg.augment else eval_tf
    images = data.input_images(cfg.input)
    if cfg.background_swap > 0:
        masks, swap_ok = data.swap_masks
        return BackgroundSwapDataset(
            images,
            data.labels,
            data.train_idx,
            transform,
            p=cfg.background_swap,
            masks=masks,
            swap_ok=swap_ok,
            class_matched=cfg.class_matched_background,
        )
    return CachedDataset(images, data.labels, data.train_idx, transform)


def make_optimizer(model: nn.Module, cfg: RunConfig, stage: str) -> AdamW:
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
        groups = [{"params": head, "lr": cfg.head_lr}]
    else:
        groups = [
            {"params": head, "lr": cfg.finetune_head_lr or cfg.lr},
            {"params": rest, "lr": cfg.lr},
        ]
    groups = [g for g in groups if g["params"]]
    return AdamW(groups, weight_decay=cfg.weight_decay)


def start_stage(
    model: nn.Module, cfg: RunConfig, epoch: int, steps_per_epoch: int
) -> tuple[str, list[nn.Parameter], AdamW, CosineAnnealingLR | None]:
    # Start of a training stage ("head" or "finetune"): set the trainable parameters and
    # (re)build the optimizer and scheduler. The cosine schedule covers the stage.
    stage = "head" if epoch < cfg.head_epochs else "finetune"
    set_trainable(model, cfg, stage)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, cfg, stage)
    scheduler = None
    if cfg.scheduler == "cosine":
        stage_end = cfg.head_epochs if stage == "head" else cfg.epochs
        scheduler = CosineAnnealingLR(
            optimizer, T_max=steps_per_epoch * (stage_end - epoch)
        )
    return stage, params, optimizer, scheduler


# ----------------------------------------------------------------------------
# One epoch: training and validation
# ----------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None,
    scaler: torch.amp.GradScaler,  # loss scaling for mixed precision (disabled without CUDA)
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
        # mixed-precision update (see GradScaler in run()):
        scaler.scale(loss).backward()  # backward pass on the scaled loss
        scaler.unscale_(optimizer)  # gradients back to their true size, before clipping
        nn.utils.clip_grad_norm_(params, max_norm=1.0)
        scaler.step(optimizer)  # optimizer step, skipped if gradients have inf / NaN
        scaler.update()  # lower the scale after an overflow, raise it slowly while stable
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


# ----------------------------------------------------------------------------
# Saved results: <run folder>/history.json (config, data fingerprint and history)
# ----------------------------------------------------------------------------


def load_finished(run_dir: str, cfg: RunConfig, data: Data) -> RunHistory | None:
    # History of a finished run, or None if it was not trained with this configuration and data.
    history_file = os.path.join(run_dir, "history.json")
    if not os.path.exists(history_file):
        return None
    with open(history_file) as f:
        saved = json.load(f)
    try:  # settings that were not saved yet have their default value
        same_config = RunConfig.model_validate(saved["config"]) == cfg
        history = RunHistory.model_validate(saved["history"])
    except ValidationError:
        return None
    return history if same_config and saved.get("data") == data.data_id else None


def save_run(run_dir: str, cfg: RunConfig, data: Data, history: RunHistory) -> None:
    # Save what load_finished needs to recognise the run later.
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(
            {
                "config": cfg.model_dump(),
                "data": data.data_id,
                "history": history.model_dump(),
            },
            f,
        )


def load_history(name: str, seed: int) -> RunHistory:
    with open(os.path.join(run_path(name, seed), "history.json")) as f:
        return RunHistory.model_validate(json.load(f)["history"])


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------


def should_stop(cfg: RunConfig, epoch: int, best_epoch: int) -> bool:
    # Early stopping: `patience` epochs of the last training stage (after the head-only epochs)
    # without a better validation macro-F1. The last epoch never counts as an early stop.
    last_stage_start = cfg.head_epochs if cfg.head_epochs < cfg.epochs else 0
    epochs_without_gain = epoch - max(best_epoch, last_stage_start - 1)
    return (
        bool(cfg.patience)
        and epoch + 1 < cfg.epochs
        and epochs_without_gain >= cfg.patience
    )


def run(name: str, cfg: RunConfig, seed: int, data: Data) -> RunHistory:
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
    train_loader = make_loader(train_dataset(cfg, data), cfg.batch_size, seed=seed)
    val_images = data.input_images(cfg.input)  # validate on the same input version
    val_loader = eval_loader(val_images, data.labels, data.val_idx)
    criterion = nn.CrossEntropyLoss(
        weight=data.class_weights.to(device) if cfg.class_weights else None
    )
    # loss scaling for mixed precision (CUDA only): float16 gradients can underflow to zero, so the
    # loss is scaled up before backward() and the gradients are scaled back before the update. When
    # USE_AMP is False (MPS, CPU) the scaler is disabled and all its calls do nothing.
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history = RunHistory()
    best_f1, best_epoch = -1.0, -1

    for epoch in range(cfg.epochs):
        if epoch in (0, cfg.head_epochs):
            stage, params, optimizer, scheduler = start_stage(
                model, cfg, epoch, len(train_loader)
            )

        # learning rate before the epoch (the scheduler changes it during the epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, params
        )
        val_loss, val_acc, val_f1, val_time = validate(model, val_loader)
        history.add_epoch(
            lr=lr,
            trainable_params=count_params(model, trainable_only=True),
            train_loss=train_loss,
            train_acc=train_acc,
            train_time=train_time,
            val_loss=val_loss,
            val_acc=val_acc,
            val_f1=val_f1,
            val_time=val_time,
        )
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} [{stage}] | train loss {train_loss:.4f} acc {train_acc:.4f}"
            f" | val loss {val_loss:.4f} acc {val_acc:.4f} macro-F1 {val_f1:.4f}"
            f" | {train_time + val_time:.0f}s"
        )

        # checkpoint the best epoch (ties keep the earlier one)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            os.makedirs(run_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

        if should_stop(cfg, epoch, best_epoch):
            print(f"Early stopping: no improvement for {cfg.patience} epochs")
            break

    history.stopped_early = len(history.val_f1) < cfg.epochs
    save_run(run_dir, cfg, data, history)
    print(
        f"Best val macro-F1: {best_f1:.4f} (epoch {best_epoch + 1} of {len(history.val_f1)})"
    )
    return history


def run_seeds(
    name: str, data: Data, seeds: Sequence[int] = TRAIN_CONFIG["seeds"]
) -> list[RunHistory]:
    # Train (or load) a run name with every training seed.
    return [run(name, config_of(name), seed, data) for seed in seeds]
