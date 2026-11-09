"""Configuration and seeds (Section 1).

The settings are split into three dictionaries: ``PATHS`` (where data and results live), ``DATA`` (the
dataset protocol) and ``TRAIN`` (seeds, epochs, early stopping). The model recipes are in
``experiments.py``, the device is in ``device.py``. Paths, epochs and seeds can be overridden with
environment variables, e.g. ``EPOCHS_CNN=1 SEEDS=62`` for a quick test.
"""

import json
import os
from typing import Any

env = os.environ.get

PATHS: dict[str, str] = {
    "data_root": env("DATA_ROOT", "./PlantVillage-Dataset"),  # where the dataset is cloned
    "cache_dir": env("CACHE_DIR", "./cache"),  # decoded + resized images
    "out_dir": env("OUT_DIR", "./outputs"),  # figures, tables, checkpoints
    "field_root": env("FIELD_ROOT", "./PlantDoc-Dataset"),  # field photos (Section 19.2)
}

DATA: dict[str, Any] = {
    "classes": [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Spider_mites Two-spotted_spider_mite",
    ],
    # one photo series per class, with official leaf ids (Section 3)
    "train_series": {
        "Tomato___healthy": "rs_hl",
        "Tomato___Early_blight": "rs_erly.b",
        "Tomato___Leaf_Mold": "crnl_l.mold",
        "Tomato___Spider_mites Two-spotted_spider_mite": "com.g_spm_fl",
    },
    "field_url": "https://github.com/pratikkayal/PlantDoc-Dataset",  # PlantDoc repository (cropped version)
    # PlantDoc folder that matches each class
    "field_folders": {
        "Tomato___healthy": "Tomato leaf",
        "Tomato___Early_blight": "Tomato Early blight leaf",
        "Tomato___Leaf_Mold": "Tomato mold leaf",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato two spotted spider mites leaf",
    },
    # a left-out series needs at least this many photos (stray single files are not a series)
    "min_series_photos": 20,
    "img_size": 224,  # ViT-B/16 requires 224
    "split_seed": 62,  # seed of the train/val/test split
    "split_fractions": (0.70, 0.15),  # train, val (test = rest), applied to official leaf ids
    "num_workers": int(env("NUM_WORKERS", 2)),  # data loader workers
}

TRAIN: dict[str, Any] = {
    "seeds": [int(s) for s in env("SEEDS", "62,63,64").split(",")],  # training seeds of the final models
    # maximum epochs per architecture
    "epochs": {
        "cnn": int(env("EPOCHS_CNN", 50)),
        "resnet18": int(env("EPOCHS_RESNET", 16)),
        "vit_b_16": int(env("EPOCHS_VIT", 15)),
    },
    # early stopping: stop after this many epochs without a better validation macro-F1
    # (counted from the start of the last training stage; 0 = train all epochs)
    "patience": {
        "cnn": int(env("PATIENCE_CNN", 5)),
        "resnet18": int(env("PATIENCE_RESNET", 5)),
        "vit_b_16": int(env("PATIENCE_VIT", 5)),
    },
    # background swap of Section 18: share of the training photos whose background is replaced
    "swap_prob": 0.5,
}


def config_dict() -> dict[str, dict[str, Any]]:
    # All settings, one section per group (saved to OUT_DIR/config.json).
    return {"paths": PATHS, "data": DATA, "train": TRAIN}


CLASSES = DATA["classes"]
# 'Tomato___Spider_mites Two-spotted_spider_mite' -> 'Spider_mites'
SHORT_NAMES = [c.split("___")[1].split()[0] for c in CLASSES]
SEEDS = TRAIN["seeds"]
OUT_DIR = PATHS["out_dir"]
RUNS_DIR = os.path.join(OUT_DIR, "runs")
for d in [PATHS["cache_dir"], OUT_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config_dict(), f, indent=2)


def out_path(name: str) -> str:
    # Path of a table or figure in OUT_DIR.
    return os.path.join(OUT_DIR, name)
