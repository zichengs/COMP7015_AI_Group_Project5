"""Configuration, seeds and device (Section 1).

All settings live in one ``CFG`` dictionary. Paths, epochs and seeds can be overridden with
environment variables, e.g. ``EPOCHS_CNN=1 SEEDS=42`` for a quick test.
"""

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # required for deterministic CUDA matmuls (set before importing torch)

import json
import platform
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
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
    ],
    "train_series": {  # one photo series per class, with official leaf ids (Section 3)
        "Tomato___healthy": "rs_hl",
        "Tomato___Bacterial_spot": "gcrec_bact.sp",
        "Tomato___Early_blight": "rs_erly.b",
        "Tomato___Late_blight": "rs_late.b",
    },
    "field_root": env(
        "FIELD_ROOT", "./PlantDoc-Dataset"
    ),  # field photos for Section 15.2
    "field_photos": {  # PlantDoc (cropped version) folders that match our classes
        "url": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "folders": {
            "Tomato___healthy": "Tomato leaf",
            "Tomato___Bacterial_spot": "Tomato leaf bacterial spot",
            "Tomato___Early_blight": "Tomato Early blight leaf",
            "Tomato___Late_blight": "Tomato leaf late blight",
        },
    },
    "min_series_photos": 20,  # a left-out series needs at least this many photos (stray single files are not a series)
    "img_size": 224,  # ViT-B/16 requires 224
    "split_seed": 42,  # seed of the train/val/test split
    "split_fractions": (0.70, 0.15),  # train, val (test = rest), applied to leaf groups
    "dup_merge_bits": 10,  # near-duplicates (<= 10 of 256 hash bits differ) are merged into one group
    "dup_audit_bits": 20,  # looser threshold, only used to audit the final split
    "seeds": [
        int(s) for s in env("SEEDS", "42,43,44").split(",")
    ],  # training seeds of the final models
    "n_bootstrap": 1000,  # bootstrap resamples of the test leaf groups (Sections 10.1, 13.1)
    "epochs": {
        "cnn": int(env("EPOCHS_CNN", 50)),
        "resnet18": int(env("EPOCHS_RESNET", 16)),
        "vit_b_16": int(env("EPOCHS_VIT", 12)),
    },
    "vit_trainable_blocks": 4,  # ViT: number of last encoder blocks that are fine-tuned
    "countermeasure": {  # background augmentation of Section 14
        "swap_prob": 0.5,  # share of the training photos whose background is replaced
        "clean_tolerance": 0.005,  # the countermeasure may cost at most this much macro-F1 on clean photos
    },
    "num_workers": int(
        env("NUM_WORKERS", 4 if platform.system() == "Linux" else 0)
    ),  # notebook workers only work reliably on Linux
}
CLASSES = CFG["classes"]
SHORT_NAMES = [c.split("___")[1] for c in CLASSES]
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


def out_path(name):
    # Path of a table or figure in OUT_DIR.
    return os.path.join(OUT_DIR, name)
