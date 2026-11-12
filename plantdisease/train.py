import json
import os
import shutil
import time
from collections.abc import Sequence
from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
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
from .experiments import RunConfig, config_of
from .models import build_model, count_params, is_head, set_trainable
from .predictions import predict

# ----------------------------------------------------------------------------
# Setting up a run
# ----------------------------------------------------------------------------


def train_dataset(cfg: RunConfig, data: Data) -> CachedDataset:
    # Training set of a configuration: its input version, augmentation and (Section 20) background swap.
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
    head, rest = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (head if is_head(cfg, name) else rest).append(p)
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
) -> tuple[str, AdamW, CosineAnnealingLR | None]:
    # Start of a training stage ("head" or "finetune"): set the trainable parameters and
    # (re)build the optimizer and scheduler. The cosine schedule covers the stage.
    stage = "head" if epoch < cfg.head_epochs else "finetune"
    set_trainable(model, cfg, stage)
    optimizer = make_optimizer(model, cfg, stage)
    scheduler = None
    if cfg.scheduler == "cosine":
        stage_end = cfg.head_epochs if stage == "head" else cfg.epochs
        scheduler = CosineAnnealingLR(
            optimizer, T_max=steps_per_epoch * (stage_end - epoch)
        )
    return stage, optimizer, scheduler


# ----------------------------------------------------------------------------
# One epoch: training and validation
# ----------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float, float]:
    # One training epoch. Returns loss, accuracy and the time (timed without validation, after the
    # device has finished). `scaler`: loss scaling for mixed precision (see _run_seed()).
    model.train()
    # the parameters whose gradients are clipped: the trainable ones of the current stage
    params = [p for p in model.parameters() if p.requires_grad]
    running_loss, running_correct = 0.0, 0
    synchronize()
    tic = time.perf_counter()
    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=USE_AMP):
            outputs = model(x)
            loss = criterion(outputs, y)
        optimizer.zero_grad(set_to_none=True)
        # mixed-precision update (see GradScaler in _run_seed()):
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


class RunHistory(BaseModel):
    # History of one training run: one entry per trained epoch in every list.
    # Unknown fields of older files are ignored.
    model_config = ConfigDict(extra="ignore")

    train_loss: list[float] = []
    train_acc: list[float] = []
    val_loss: list[float] = []
    val_acc: list[float] = []
    val_f1: list[float] = []  # macro-F1; the checkpoint of the best epoch is kept
    train_time: list[float] = []  # seconds, training only (device synchronised)
    val_time: list[float] = []  # seconds
    trainable_params: list[int] = []
    stopped_early: bool = False  # early stopping ended the run before `epochs`

    @classmethod
    def per_epoch_fields(cls) -> list[str]:
        return [name for name in cls.model_fields if name != "stopped_early"]

    @model_validator(mode="after")
    def check_lengths(self) -> Self:
        lengths = {len(getattr(self, name)) for name in self.per_epoch_fields()}
        if len(lengths) > 1:
            raise ValueError("the per-epoch lists have different lengths")
        return self

    def add_epoch(self, **values: float) -> None:
        # Record one finished epoch: one value for every per-epoch list.
        for name in self.per_epoch_fields():
            getattr(self, name).append(values[name])


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
    # Early stopping after `patience` epochs without a better validation macro-F1. Only epochs of
    # the last training stage count: the head-only epochs never do. The last epoch is no early stop.
    # Example: head_epochs=3, patience=5, best epoch 1 (0-based, still head-only): counting starts
    # at epoch 3, so the run stops after epoch 7 unless an epoch from 3 to 7 is better.
    if not cfg.patience or epoch + 1 >= cfg.epochs:
        return False
    # the fine-tuning stage starts after the head-only epochs (a head-only run has one stage)
    last_stage_start = cfg.head_epochs if cfg.head_epochs < cfg.epochs else 0
    counting_from = max(best_epoch + 1, last_stage_start)
    epochs_without_gain = epoch + 1 - counting_from
    return epochs_without_gain >= cfg.patience


def _run_seed(name: str, seed: int, data: Data) -> RunHistory:
    # Train one run name with one seed (or load it if it was trained before). Returns the history.
    cfg = config_of(name)
    run_dir = run_path(name, seed)
    finished = load_finished(run_dir, cfg, data)
    if finished is not None:
        print(f"Loaded finished run: {name} (seed {seed})")
        return finished

    print(f"\n===== {name} (seed {seed}) =====")
    # predictions cached for an earlier version of this run are stale
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
            stage, optimizer, scheduler = start_stage(
                model, cfg, epoch, len(train_loader)
            )

        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler
        )
        val_loss, val_acc, val_f1, val_time = validate(model, val_loader)
        history.add_epoch(
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


def run(
    name: str, data: Data, seeds: Sequence[int] = TRAIN_CONFIG["seeds"]
) -> list[RunHistory]:
    # Train (or load) a run name with every seed -> one history per seed.
    return [_run_seed(name, seed, data) for seed in seeds]
