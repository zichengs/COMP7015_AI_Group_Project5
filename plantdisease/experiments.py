"""Run names and recipes: derived runs (Sections 15 and 18), the experiments that the notebooks
register, and the selected final models."""

import json
import os
from collections.abc import Iterable
from typing import Any

from .config import TRAIN_CONFIG, out_path
from .recipe import RunConfig


def _read_json(path: str) -> dict[str, Any]:
    # Contents of a JSON file ({} if it does not exist yet).
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _update_json(path: str, changes: dict[str, Any]) -> None:
    # Add or replace some entries of a JSON file and keep the others.
    contents = {**_read_json(path), **changes}
    with open(path, "w") as f:
        json.dump(contents, f, indent=2)


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
    _update_json(
        EXPERIMENTS_FILE, {name: cfg.model_dump() for name, cfg in experiments.items()}
    )


def _registered(name: str) -> RunConfig:
    # Recipe of a registered experiment (from this notebook or from experiments.json).
    if name not in _registry:
        for saved_name, cfg in _read_json(EXPERIMENTS_FILE).items():
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
DERIVED = {**INPUT_TRAINING, **COUNTERMEASURES}


def input_training_names(base: str) -> list[str]:
    return [f"{base} + {suffix}" for suffix in INPUT_TRAINING]


def countermeasure_names(base: str) -> list[str]:
    return [f"{base} + {suffix}" for suffix in COUNTERMEASURES]


def countermeasure_runs(bases: Iterable[str]) -> list[str]:
    # The countermeasure and control runs of several models, in one list.
    return [name for base in bases for name in countermeasure_names(base)]


def suffix_label(name: str, base: str) -> str:
    # Short label of a derived run: ('X + random backgrounds', 'X') -> 'random backgrounds'.
    return "baseline" if name == base else name.removeprefix(f"{base} + ")


def config_of(name: str) -> RunConfig:
    # Configuration of a run name (registered experiments, final models and derived runs). Suffixes
    # can be combined, e.g. '<model> + grayscale training + random backgrounds'.
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
    _update_json(FINAL_MODELS_FILE, {family: name})


def load_final_models() -> tuple[str, str, str]:
    # The selected models -> (BEST_CNN, BEST_RESNET, BEST_VIT).
    selected = _read_json(FINAL_MODELS_FILE)
    missing = {"cnn", "resnet18", "vit_b_16"} - set(selected)
    assert not missing, f"run notebooks 02 and 03 first (missing: {sorted(missing)})"
    return selected["cnn"], selected["resnet18"], selected["vit_b_16"]
