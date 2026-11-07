"""Experiment configurations (Sections 9, 10, 15 and 18) and the selected final models."""

import json
import os

from .config import CFG, out_path

# ----------------------------------------------------------------------------
# Section 9: custom CNN and ResNet18
# ----------------------------------------------------------------------------
BASE_CNN = {
    "arch": "se_cnn",
    "lr": 1e-3,
    "batch_size": 32,
    "dropout": 0.3,
    "augment": True,
    "class_weights": True,
    "scheduler": "cosine",
    "weight_decay": 1e-4,
    "epochs": CFG["epochs"]["cnn"],
    "head_epochs": 0,
    "head_lr": None,
}
cnn_configs = {
    "SE-CNN (full recipe)": BASE_CNN,
    "SE-CNN lr=3e-4": {**BASE_CNN, "lr": 3e-4},
    "SE-CNN lr=3e-3": {**BASE_CNN, "lr": 3e-3},
    "CNN (no attention)": {**BASE_CNN, "arch": "cnn"},
    "SE-CNN no augmentation": {**BASE_CNN, "augment": False},
    "SE-CNN no dropout": {**BASE_CNN, "dropout": 0.0},
    "SE-CNN no class weights": {**BASE_CNN, "class_weights": False},
    "SE-CNN constant lr": {**BASE_CNN, "scheduler": "constant"},
    "SE-CNN batch size 64": {**BASE_CNN, "batch_size": 64},
}

BASE_RESNET = {
    "arch": "resnet18",
    "lr": 1e-4,
    "batch_size": 32,
    "dropout": 0.0,
    "augment": True,
    "class_weights": True,
    "scheduler": "cosine",
    "weight_decay": 1e-4,
    "epochs": CFG["epochs"]["resnet18"],
    "head_epochs": 1,
    "head_lr": 1e-3,
}
resnet_configs = {
    "ResNet18 frozen backbone": {
        **BASE_RESNET,
        "head_epochs": BASE_RESNET["epochs"],
        "scheduler": "constant",
    },
    "ResNet18 full FT, constant lr": {
        **BASE_RESNET,
        "head_epochs": 0,
        "scheduler": "constant",
    },
    "ResNet18 two-stage + cosine, lr=1e-4": BASE_RESNET,
    "ResNet18 two-stage + cosine, lr=3e-4": {**BASE_RESNET, "lr": 3e-4},
    "ResNet18 batch size 64": {**BASE_RESNET, "batch_size": 64},
    "ResNet18 no augmentation": {**BASE_RESNET, "augment": False},
    "ResNet18 weight decay 1e-2": {**BASE_RESNET, "weight_decay": 1e-2},
}

# ----------------------------------------------------------------------------
# Section 10: ViT-B/16 fine-tuning strategies (Zhouyang's protocol)
# 3 epochs train only the new head (lr 1e-3); then the head continues at 1e-4 and the unfrozen
# backbone parameters train at 1e-5. AdamW (weight decay 0.01), constant learning rate, batch 16,
# no class weights. The strategies differ only in which parameters are unfrozen after epoch 3.
# ----------------------------------------------------------------------------
BASE_VIT = {
    "arch": "vit_b_16",
    "finetune": "partial",
    "trainable_blocks": 2,
    "lr": 1e-5,  # unfrozen backbone parameters (after the head epochs)
    "finetune_head_lr": 1e-4,  # classifier head after the head epochs
    "head_lr": 1e-3,  # classifier head during the head epochs
    "head_epochs": 3,
    "epochs": CFG["epochs"]["vit_b_16"],
    "batch_size": 16,
    "weight_decay": 0.01,
    "scheduler": "constant",
    "class_weights": False,
    "augment": True,
    "dropout": 0.0,
}
vit_configs = {
    "ViT-B/16 head only": {**BASE_VIT, "finetune": "head"},
    "ViT-B/16 partial (last 2 blocks)": BASE_VIT,
    "ViT-B/16 full fine-tuning": {**BASE_VIT, "finetune": "full"},
    "ViT-B/16 partial, no augmentation": {**BASE_VIT, "augment": False},
}
ALL_CONFIGS = {**cnn_configs, **resnet_configs, **vit_configs}
TUNING_SEED = 42

# ----------------------------------------------------------------------------
# Derived runs: '<selected model> + <suffix>' = the selected configuration with a change
#   Section 15: the ViT retrained on another input version (trained and validated on that version)
#   Section 18: CNN / ResNet18 retrained with random backgrounds (countermeasure) or
#               class-matched backgrounds (control)
# ----------------------------------------------------------------------------
INPUT_TRAINING = {
    "grayscale training": {"input": "grayscale"},
    "background-removed training": {"input": "segmented"},
}
COUNTERMEASURES = {
    "random backgrounds": {"background_swap": CFG["countermeasure"]["swap_prob"]},
    "class-matched backgrounds": {
        "background_swap": CFG["countermeasure"]["swap_prob"],
        "class_matched_background": True,
    },
}
DERIVED = {**INPUT_TRAINING, **COUNTERMEASURES}


def input_training_names(base):
    return [f"{base} + {suffix}" for suffix in INPUT_TRAINING]


def countermeasure_names(base):
    return [f"{base} + {suffix}" for suffix in COUNTERMEASURES]


def config_of(name):
    # Configuration of a run name (tuning runs, final models and derived runs).
    for suffix, change in DERIVED.items():
        base = name.removesuffix(f" + {suffix}")
        if base != name:
            return {**ALL_CONFIGS[base], **change}
    return ALL_CONFIGS[name]


# ----------------------------------------------------------------------------
# Selected models: CNN and ResNet18 in notebook 02, ViT in notebook 03
# ----------------------------------------------------------------------------
FINAL_MODELS_FILE = out_path("final_models.json")


def save_selected(family, name):
    # Record the model selected on the validation set for a family ('cnn', 'resnet18' or 'vit_b_16').
    selected = {}
    if os.path.exists(FINAL_MODELS_FILE):
        with open(FINAL_MODELS_FILE) as f:
            selected = json.load(f)
    selected[family] = name
    with open(FINAL_MODELS_FILE, "w") as f:
        json.dump(selected, f, indent=2)


def load_final_models():
    # The selected models -> (BEST_CNN, BEST_RESNET, BEST_VIT).
    with open(FINAL_MODELS_FILE) as f:
        selected = json.load(f)
    missing = {"cnn", "resnet18", "vit_b_16"} - set(selected)
    assert not missing, f"run notebooks 02 and 03 first (missing: {sorted(missing)})"
    return selected["cnn"], selected["resnet18"], selected["vit_b_16"]
