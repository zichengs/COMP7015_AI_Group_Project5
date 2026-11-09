"""Configuration, seeds and device (Section 1).

The settings are split into three groups: ``PATHS`` (where data and results live), ``DATA`` (the
dataset protocol) and ``TRAIN`` (seeds, epochs, early stopping). The model recipes are in
``experiments.py``. Paths, epochs and seeds can be overridden with environment variables, e.g.
``EPOCHS_CNN=1 SEEDS=62`` for a quick test.
"""

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # required for deterministic CUDA matmuls (set before importing torch)

import json
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

env = os.environ.get


@dataclass(frozen=True)
class Paths:
    data_root: str  # where the dataset is cloned
    cache_dir: str  # decoded + resized images
    out_dir: str  # figures, tables, checkpoints
    field_root: str  # field photos (Section 19.2)

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.out_dir, "runs")


@dataclass(frozen=True)
class DataConfig:
    classes: list[str]
    # one photo series per class, with official leaf ids (Section 3)
    train_series: dict[str, str]
    field_url: str  # PlantDoc repository (cropped version)
    field_folders: dict[str, str]  # PlantDoc folder that matches each class
    # a left-out series needs at least this many photos (stray single files are not a series)
    min_series_photos: int
    img_size: int  # ViT-B/16 requires 224
    split_seed: int  # seed of the train/val/test split
    # train, val (test = rest), applied to official leaf ids
    split_fractions: tuple[float, float]
    num_workers: int  # data loader workers


@dataclass(frozen=True)
class TrainConfig:
    seeds: list[int]  # training seeds of the final models
    epochs: dict[str, int]  # maximum epochs per architecture
    # early stopping: stop after this many epochs without a better validation macro-F1
    # (counted from the start of the last training stage; 0 = train all epochs)
    patience: dict[str, int]
    # background swap of Section 18: share of the training photos whose background is replaced
    swap_prob: float


PATHS = Paths(
    data_root=env("DATA_ROOT", "./PlantVillage-Dataset"),
    cache_dir=env("CACHE_DIR", "./cache"),
    out_dir=env("OUT_DIR", "./outputs"),
    field_root=env("FIELD_ROOT", "./PlantDoc-Dataset"),
)
DATA = DataConfig(
    classes=[
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Spider_mites Two-spotted_spider_mite",
    ],
    train_series={
        "Tomato___healthy": "rs_hl",
        "Tomato___Early_blight": "rs_erly.b",
        "Tomato___Leaf_Mold": "crnl_l.mold",
        "Tomato___Spider_mites Two-spotted_spider_mite": "com.g_spm_fl",
    },
    field_url="https://github.com/pratikkayal/PlantDoc-Dataset",
    field_folders={
        "Tomato___healthy": "Tomato leaf",
        "Tomato___Early_blight": "Tomato Early blight leaf",
        "Tomato___Leaf_Mold": "Tomato mold leaf",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato two spotted spider mites leaf",
    },
    min_series_photos=20,
    img_size=224,
    split_seed=62,
    split_fractions=(0.70, 0.15),
    num_workers=int(env("NUM_WORKERS", 2)),
)
TRAIN = TrainConfig(
    seeds=[int(s) for s in env("SEEDS", "62,63,64").split(",")],
    epochs={
        "cnn": int(env("EPOCHS_CNN", 50)),
        "resnet18": int(env("EPOCHS_RESNET", 16)),
        "vit_b_16": int(env("EPOCHS_VIT", 15)),
    },
    patience={
        "cnn": int(env("PATIENCE_CNN", 5)),
        "resnet18": int(env("PATIENCE_RESNET", 5)),
        "vit_b_16": int(env("PATIENCE_VIT", 5)),
    },
    swap_prob=0.5,
)


def config_dict() -> dict[str, dict[str, Any]]:
    # All settings, one section per group (saved to OUT_DIR/config.json).
    return {"paths": asdict(PATHS), "data": asdict(DATA), "train": asdict(TRAIN)}


CLASSES = DATA.classes
# 'Tomato___Spider_mites Two-spotted_spider_mite' -> 'Spider_mites'
SHORT_NAMES = [c.split("___")[1].split()[0] for c in CLASSES]
SEEDS = TRAIN.seeds
OUT_DIR, RUNS_DIR = PATHS.out_dir, PATHS.runs_dir
for d in [PATHS.cache_dir, OUT_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config_dict(), f, indent=2)


def set_seed(seed: int) -> None:
    # Fix all random number generators so that runs are reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(
    True, warn_only=True
)  # warn (not fail) for the few ops without a deterministic version

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
USE_AMP = (
    device.type == "cuda"
)  # mixed precision: large speed-up on NVIDIA GPUs (no gain on Apple MPS)


def synchronize() -> None:
    # Wait for the GPU (CUDA or Apple MPS) to finish, so that timings are correct.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def out_path(name: str) -> str:
    # Path of a table or figure in OUT_DIR.
    return os.path.join(OUT_DIR, name)
