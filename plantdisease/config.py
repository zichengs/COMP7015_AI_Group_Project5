"""Configuration (Section 1).

The settings are split into four dictionaries: ``PATHS_CONFIG`` (where data and results live),
``DATA_CONFIG`` (the PlantVillage protocol), ``PLANTDOC_CONFIG`` (the PlantDoc field photos) and
``TRAIN_CONFIG`` (training seeds, background-swap probability). The model recipes, with their epochs
and early stopping, are defined in notebooks 02 and 03, the device is in ``device.py``. Paths, epochs and seeds
can be overridden with environment variables, e.g. ``EPOCHS_CNN=1 SEEDS=62`` for a quick test.
"""

import json
import os
from typing import Any

env = os.environ.get

_out_dir = env("OUT_DIR", "./outputs")

PATHS_CONFIG: dict[str, str] = {
    "data_root": env("DATA_ROOT", "./PlantVillage-Dataset"),  # where the dataset is cloned
    "cache_dir": env("CACHE_DIR", "./cache"),  # decoded + resized images
    "out_dir": _out_dir,  # figures, tables, checkpoints
    "runs_dir": os.path.join(_out_dir, "runs"),  # one folder per trained run
}

DATA_CONFIG: dict[str, Any] = {
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
    # a left-out series needs at least this many photos (stray single files are not a series)
    "min_series_photos": 20,
    "img_size": 224,  # ViT-B/16 requires 224
    "split_seed": 62,  # seed of the train/val/test split
    "split_fractions": (0.70, 0.15),  # train, val (test = rest), applied to official leaf ids
    "num_workers": int(env("NUM_WORKERS", 2)),  # data loader workers
}

# PlantDoc field photos (Section 19.2)
PLANTDOC_CONFIG: dict[str, Any] = {
    "root": env("FIELD_ROOT", "./PlantDoc-Dataset"),  # where the dataset is cloned
    "url": "https://github.com/pratikkayal/PlantDoc-Dataset",  # PlantDoc repository (cropped version)
    # PlantDoc folder that matches each class
    "folders": {
        "Tomato___healthy": "Tomato leaf",
        "Tomato___Early_blight": "Tomato Early blight leaf",
        "Tomato___Leaf_Mold": "Tomato mold leaf",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato two spotted spider mites leaf",
    },
}

TRAIN_CONFIG: dict[str, Any] = {
    "seeds": [int(s) for s in env("SEEDS", "62,63,64").split(",")],  # training seeds of the final models
    # background swap of Section 18: share of the training photos whose background is replaced
    "swap_prob": 0.5,
}


def config_dict() -> dict[str, dict[str, Any]]:
    # All settings, one section per group (saved to OUT_DIR/config.json).
    return {
        "paths": PATHS_CONFIG,
        "data": DATA_CONFIG,
        "plantdoc": PLANTDOC_CONFIG,
        "train": TRAIN_CONFIG,
    }


# 'Tomato___Spider_mites Two-spotted_spider_mite' -> 'Spider_mites'
SHORT_NAMES = [c.split("___")[1].split()[0] for c in DATA_CONFIG["classes"]]
for d in [PATHS_CONFIG["cache_dir"], PATHS_CONFIG["out_dir"], PATHS_CONFIG["runs_dir"]]:
    os.makedirs(d, exist_ok=True)
with open(os.path.join(PATHS_CONFIG["out_dir"], "config.json"), "w") as f:
    json.dump(config_dict(), f, indent=2)


def out_path(name: str) -> str:
    # Path of a table or figure in the output folder.
    return os.path.join(PATHS_CONFIG["out_dir"], name)
