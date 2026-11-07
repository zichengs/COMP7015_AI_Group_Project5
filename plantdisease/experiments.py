"""Experiment configurations of Sections 9 and 14 and the selected final models."""

import json

from .config import CFG, out_path

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

vit_configs = {
    "ViT-B/16 two-stage + cosine": {
        "arch": "vit_b_16",
        "lr": 5e-5,
        "batch_size": 32,
        "dropout": 0.0,
        "augment": True,
        "class_weights": True,
        "scheduler": "cosine",
        "weight_decay": 0.05,
        "epochs": CFG["epochs"]["vit_b_16"],
        "head_epochs": 1,
        "head_lr": 1e-3,
    },
}
ALL_CONFIGS = {**cnn_configs, **resnet_configs, **vit_configs}
TUNING_SEED = 42

# Section 14: every selected CNN / ResNet18 is retrained with random backgrounds (countermeasure)
# and with class-matched backgrounds (control). The run name is '<base> + <suffix>'.
COUNTERMEASURES = {
    "random backgrounds": {},
    "class-matched backgrounds": {"class_matched_background": True},
}


def countermeasure_names(base):
    return [f"{base} + {suffix}" for suffix in COUNTERMEASURES]


def config_of(name):
    # Configuration of a run name (tuning runs, final models and countermeasure runs).
    for suffix, extra in COUNTERMEASURES.items():
        base = name.removesuffix(f" + {suffix}")
        if base != name:
            swap = {
                **ALL_CONFIGS[base],
                "background_swap": CFG["countermeasure"]["swap_prob"],
            }
            return {**swap, **extra}
    return ALL_CONFIGS[name]


FINAL_MODELS_FILE = out_path("final_models.json")


def save_final_models(best_cnn, best_resnet, best_vit):
    with open(FINAL_MODELS_FILE, "w") as f:
        json.dump(
            {"cnn": best_cnn, "resnet18": best_resnet, "vit_b_16": best_vit},
            f,
            indent=2,
        )


def load_final_models():
    # The models selected on the validation set in notebook 02 -> (BEST_CNN, BEST_RESNET, BEST_VIT).
    with open(FINAL_MODELS_FILE) as f:
        final = json.load(f)
    return final["cnn"], final["resnet18"], final["vit_b_16"]
