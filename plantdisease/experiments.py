"""Run names and recipes: derived runs (Sections 15 and 18), the experiments that the notebooks
register, and the selected final models."""

import json
import os

from .config import TRAIN_CONFIG, out_path
from .recipe import RunConfig

TUNING_SEED = 62

# ----------------------------------------------------------------------------
# The experiments (tuning runs, ablations, strategies) are defined in notebooks 02 and 03 and
# registered with `register`. They are kept in OUT_DIR/experiments.json, so that every notebook finds
# the recipe of a run name without running notebooks 02 and 03 again.
# ----------------------------------------------------------------------------
EXPERIMENTS_FILE = out_path("experiments.json")
_registry: dict[str, RunConfig] = {}


def register(experiments: dict[str, RunConfig]) -> None:
    # Make the experiments known by name to this and to the other notebooks.
    _registry.update(experiments)
    saved = {}
    if os.path.exists(EXPERIMENTS_FILE):
        with open(EXPERIMENTS_FILE) as f:
            saved = json.load(f)
    saved.update({name: cfg.model_dump() for name, cfg in experiments.items()})
    with open(EXPERIMENTS_FILE, "w") as f:
        json.dump(saved, f, indent=2)


def _registered(name: str) -> RunConfig:
    if name not in _registry and os.path.exists(EXPERIMENTS_FILE):
        with open(EXPERIMENTS_FILE) as f:
            for saved_name, cfg in json.load(f).items():
                _registry.setdefault(saved_name, RunConfig.model_validate(cfg))
    if name not in _registry:
        raise KeyError(f"unknown run '{name}': run notebooks 02 and 03 first")
    return _registry[name]


# ----------------------------------------------------------------------------
# Derived runs: '<selected model> + <suffix>' = the selected configuration with a change
#   Section 15: the ViT retrained on another input version (trained and validated on that version)
#   Section 18: CNN / ResNet18 / ViT retrained with random backgrounds (countermeasure) or
#               class-matched backgrounds (control)
# ----------------------------------------------------------------------------
INPUT_TRAINING = {
    "grayscale training": {"input": "grayscale"},
    "background-removed training": {"input": "segmented"},
}
COUNTERMEASURES = {
    "random backgrounds": {"background_swap": TRAIN_CONFIG["swap_prob"]},
    "class-matched backgrounds": {
        "background_swap": TRAIN_CONFIG["swap_prob"],
        "class_matched_background": True,
    },
}
#   Notebook 07: the selected models trained with the classes 'Other disease' and 'Other crop'
OPEN_SET = {"6 classes": {"num_classes": 6}}
DERIVED = {**INPUT_TRAINING, **COUNTERMEASURES, **OPEN_SET}


def input_training_names(base: str) -> list[str]:
    return [f"{base} + {suffix}" for suffix in INPUT_TRAINING]


def countermeasure_names(base: str) -> list[str]:
    return [f"{base} + {suffix}" for suffix in COUNTERMEASURES]


def open_set_name(base: str) -> str:
    return f"{base} + {next(iter(OPEN_SET))}"


def config_of(name: str) -> RunConfig:
    # Configuration of a run name (registered experiments, final models and derived runs). Suffixes
    # can be combined, e.g. '<model> + background-removed training + 6 classes' (notebook 07).
    for suffix, change in DERIVED.items():
        base = name.removesuffix(f" + {suffix}")
        if base != name:
            return config_of(base).variant(**change)
    return _registered(name)


# ----------------------------------------------------------------------------
# Selected models: CNN and ResNet18 in notebook 02, ViT in notebook 03
# ----------------------------------------------------------------------------
FINAL_MODELS_FILE = out_path("final_models.json")


def save_selected(family: str, name: str) -> None:
    # Record the model selected on the validation set for a family ('cnn', 'resnet18' or 'vit_b_16').
    selected = {}
    if os.path.exists(FINAL_MODELS_FILE):
        with open(FINAL_MODELS_FILE) as f:
            selected = json.load(f)
    selected[family] = name
    with open(FINAL_MODELS_FILE, "w") as f:
        json.dump(selected, f, indent=2)


def load_final_models() -> tuple[str, str, str]:
    # The selected models -> (BEST_CNN, BEST_RESNET, BEST_VIT).
    with open(FINAL_MODELS_FILE) as f:
        selected = json.load(f)
    missing = {"cnn", "resnet18", "vit_b_16"} - set(selected)
    assert not missing, f"run notebooks 02 and 03 first (missing: {sorted(missing)})"
    return selected["cnn"], selected["resnet18"], selected["vit_b_16"]
