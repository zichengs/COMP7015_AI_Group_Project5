import json
import os
import re
from typing import Any

env = os.environ.get

_out_dir = env("OUT_DIR", "./outputs")

PATHS_CONFIG: dict[str, str] = {
    # where the dataset is cloned
    "data_root": env("DATA_ROOT", "./PlantVillage-Dataset"),
    # decoded + resized images
    "cache_dir": env("CACHE_DIR", "./cache"),
    # figures, tables, checkpoints
    "out_dir": _out_dir,
    # one folder per trained run
    "runs_dir": os.path.join(_out_dir, "runs"),
}

DATA_CONFIG: dict[str, Any] = {
    "classes": [
        "Tomato___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
    ],
    # ViT-B/16 requires 224
    "img_size": 224,
    # seed of the train/val/test split
    "split_seed": 62,
    # train, val (test = rest), applied to the official leaf ids of every class
    "split_fractions": (0.70, 0.15),
    # data loader workers
    "num_workers": int(env("NUM_WORKERS", 2)),
    # smoke tests only (scripts/smoke_test.py): keep the first N leaves of every class; 0 = all
    "max_leaves_per_class": int(env("MAX_LEAVES", 0)),
}

# PlantDoc field photos
PLANTDOC_CONFIG: dict[str, Any] = {
    # where the dataset is downloaded
    "root": env("FIELD_ROOT", "./PlantDoc-Dataset"),
    # PlantDoc repository (cropped version)
    "url": "https://github.com/pratikkayal/PlantDoc-Dataset",
    # PlantDoc folder that matches each class
    "folders": {
        "Tomato___healthy": "Tomato leaf",
        "Tomato___Bacterial_spot": "Tomato leaf bacterial spot",
        "Tomato___Early_blight": "Tomato Early blight leaf",
        "Tomato___Late_blight": "Tomato leaf late blight",
    },
}

TRAIN_CONFIG: dict[str, Any] = {
    # training seeds of the final models
    "seeds": [int(s) for s in env("SEEDS", "62,63,64").split(",")],
    # the single seed of the tuning runs and ablations
    "tuning_seed": 62,
    # background swap: share of the training photos whose background is replaced
    "swap_prob": 0.5,
}

# 'Tomato___Bacterial_spot' -> 'Bacterial_spot'
SHORT_NAMES = [c.split("___")[1].split()[0] for c in DATA_CONFIG["classes"]]

for d in [PATHS_CONFIG["cache_dir"], PATHS_CONFIG["out_dir"], PATHS_CONFIG["runs_dir"]]:
    os.makedirs(d, exist_ok=True)


def save_config() -> None:
    # Save all settings to OUT_DIR/config.json, one section per group.
    settings = {
        "paths": PATHS_CONFIG,
        "data": DATA_CONFIG,
        "plantdoc": PLANTDOC_CONFIG,
        "train": TRAIN_CONFIG,
    }
    with open(out_path("config.json"), "w") as f:
        json.dump(settings, f, indent=2)


def out_path(name: str) -> str:
    # Path of a table or figure in the output folder.
    return os.path.join(PATHS_CONFIG["out_dir"], name)


def safe_name(name: str) -> str:
    # File-system safe version of a run or test-set name, e.g. 'ViT-B/16' -> 'ViT-B_16'.
    return re.sub(r"[^\w.=+-]+", "_", name)


def run_path(name: str, seed: int) -> str:
    # Folder of one training run: its checkpoint, history and cached predictions.
    return os.path.join(PATHS_CONFIG["runs_dir"], safe_name(name) + f"__seed{seed}")
