"""Configuration, seeds and device (Section 1).

All settings live in one ``CFG`` dictionary. Paths, epochs and seeds can be overridden with
environment variables, e.g. ``EPOCHS_CNN=1 SEEDS=62`` for a quick test.
"""

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # required for deterministic CUDA matmuls (set before importing torch)

import json
import random

import numpy as np
import torch

env = os.environ.get
CFG = {
    "data_root": env(
        "DATA_ROOT", "./PlantVillage-Dataset"
    ),  # where the dataset is cloned
    "cache_dir": env("CACHE_DIR", "./cache"),  # decoded + resized images
    "out_dir": env("OUT_DIR", "./outputs"),  # figures, tables, checkpoints
    "classes": [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Spider_mites Two-spotted_spider_mite",
    ],
    "train_series": {  # one photo series per class, with official leaf ids (Section 3)
        "Tomato___healthy": "rs_hl",
        "Tomato___Early_blight": "rs_erly.b",
        "Tomato___Leaf_Mold": "crnl_l.mold",
        "Tomato___Spider_mites Two-spotted_spider_mite": "com.g_spm_fl",
    },
    "field_root": env(
        "FIELD_ROOT", "./PlantDoc-Dataset"
    ),  # field photos (Section 19.2)
    "field_photos": {  # PlantDoc (cropped version) folders that match our classes
        "url": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "folders": {
            "Tomato___healthy": "Tomato leaf",
            "Tomato___Early_blight": "Tomato Early blight leaf",
            "Tomato___Leaf_Mold": "Tomato mold leaf",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato two spotted spider mites leaf",
        },
    },
    "min_series_photos": 20,  # a left-out series needs at least this many photos (stray single files are not a series)
    "img_size": 224,  # ViT-B/16 requires 224
    "split_seed": 62,  # seed of the train/val/test split
    "split_fractions": (
        0.70,
        0.15,
    ),  # train, val (test = rest), applied to official leaf ids
    "seeds": [
        int(s) for s in env("SEEDS", "62,63,64").split(",")
    ],  # training seeds of the final models
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
    "countermeasure": {  # background augmentation of Section 18
        "swap_prob": 0.5,  # share of the training photos whose background is replaced
    },
    # "num_workers": int(
    #     env("NUM_WORKERS", 4 if platform.system() == "Linux" else 0)
    # ),  # notebook workers only work reliably on Linux
    "num_workers": int(env("NUM_WORKERS", 2)),
}
CLASSES = CFG["classes"]
# 'Tomato___Spider_mites Two-spotted_spider_mite' -> 'Spider_mites'
SHORT_NAMES = [c.split("___")[1].split()[0] for c in CLASSES]
SEEDS = CFG["seeds"]
OUT_DIR, RUNS_DIR = CFG["out_dir"], os.path.join(CFG["out_dir"], "runs")
for d in [CFG["cache_dir"], OUT_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(CFG, f, indent=2)


def set_seed(seed):
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


def synchronize():
    # Wait for the GPU (CUDA or Apple MPS) to finish, so that timings are correct.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def out_path(name):
    # Path of a table or figure in OUT_DIR.
    return os.path.join(OUT_DIR, name)
