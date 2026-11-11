"""What is trained: the settings of one run (``RunConfig``), the recipes of every experiment of
Sections 7-10, the derived runs of Sections 15 and 18, and the selected final models.

A run is identified by its name: ``config_of(name)`` gives its recipe in every notebook.
"""

import json
import os
from collections.abc import Iterable
from typing import Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from .config import TRAIN_CONFIG, env, out_path


# ----------------------------------------------------------------------------
# The settings of one training run
# ----------------------------------------------------------------------------
class RunConfig(BaseModel):
    # Everything that defines one training run. Unknown fields are rejected, so a typo in a recipe
    # fails when the recipe is built. A recipe is immutable: `variant` makes a changed copy.
    model_config = ConfigDict(extra="forbid", frozen=True)

    # "se_cnn": custom CNN with SE attention
    arch: Literal["cnn", "se_cnn", "resnet18", "vit_b_16"]
    # learning rate of the whole network (ViT: of the unfrozen backbone parameters)
    lr: float
    batch_size: int
    # dropout before the classifier (custom CNN only)
    dropout: float = 0.0
    augment: bool
    # class-weighted loss
    class_weights: bool
    scheduler: Literal["cosine", "constant"]
    weight_decay: float
    # maximum number of epochs, including the head-only epochs
    epochs: int
    # early stopping after this many epochs without a better val macro-F1 (0 = off)
    patience: int = 0

    # Pre-trained models: head-only epochs first, then fine-tuning.
    head_epochs: int = 0
    # learning rate of the classifier head during the head-only epochs
    head_lr: float | None = None
    # learning rate of the classifier head after them (None: same as `lr`)
    finetune_head_lr: float | None = None
    # what is trained after the head-only epochs: None = the whole network, 0 = the classifier
    # head only, n = the last n ViT encoder blocks, the final LayerNorm and the classifier head
    trainable_blocks: int | None = None

    # Derived runs (Sections 15 and 18).
    input: Literal["color", "grayscale", "segmented"] = "color"
    # share of the training photos that get a random background
    background_swap: float = 0.0
    # control: the background comes from a photo of the same class
    class_matched_background: bool = False

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.head_epochs > self.epochs:
            raise ValueError("head_epochs is larger than epochs")
        if self.head_epochs > 0 and self.head_lr is None:
            raise ValueError("head_epochs > 0 needs head_lr")
        return self

    def variant(self, **changes: Any) -> Self:
        # The same recipe with some settings changed (validated again).
        return type(self).model_validate({**self.model_dump(), **changes})

    def changes_from(self, base: "RunConfig") -> dict[str, Any]:
        # The settings that differ from `base`, e.g. {"lr": 0.0003} for base.variant(lr=3e-4).
        base_values = base.model_dump()
        return {k: v for k, v in self.model_dump().items() if v != base_values[k]}


# ----------------------------------------------------------------------------
# Custom CNN and ResNet18 (Sections 7-9): one base recipe each, every experiment changes one setting
# ----------------------------------------------------------------------------
# Custom CNN (with SE attention), trained from scratch
BASE_CNN = RunConfig(
    arch="se_cnn",
    lr=1e-3,
    batch_size=32,
    dropout=0.3,
    augment=True,
    class_weights=True,
    scheduler="cosine",
    weight_decay=1e-4,
    epochs=int(env("EPOCHS_CNN", 50)),
    patience=int(env("PATIENCE_CNN", 10)),
)
# ResNet18: ImageNet-pre-trained, one head-only epoch, then fine-tuning
BASE_RESNET = RunConfig(
    arch="resnet18",
    lr=1e-4,
    batch_size=32,
    augment=True,
    class_weights=True,
    scheduler="cosine",
    weight_decay=1e-4,
    epochs=int(env("EPOCHS_RESNET", 20)),
    patience=int(env("PATIENCE_RESNET", 5)),
    head_epochs=1,
    head_lr=1e-3,
)
CNN_EXPERIMENTS = {
    "SE-CNN (full recipe)": BASE_CNN,
    "SE-CNN lr=3e-4": BASE_CNN.variant(lr=3e-4),
    "SE-CNN lr=3e-3": BASE_CNN.variant(lr=3e-3),
    "CNN (no attention)": BASE_CNN.variant(arch="cnn"),
    "SE-CNN no augmentation": BASE_CNN.variant(augment=False),
    "SE-CNN no dropout": BASE_CNN.variant(dropout=0.0),
    "SE-CNN no class weights": BASE_CNN.variant(class_weights=False),
    "SE-CNN constant lr": BASE_CNN.variant(scheduler="constant"),
    "SE-CNN batch size 64": BASE_CNN.variant(batch_size=64),
}
RESNET_EXPERIMENTS = {
    "ResNet18 frozen backbone": BASE_RESNET.variant(
        head_epochs=BASE_RESNET.epochs, scheduler="constant"
    ),
    "ResNet18 full FT, constant lr": BASE_RESNET.variant(
        head_epochs=0, scheduler="constant"
    ),
    "ResNet18 two-stage + cosine, lr=1e-4": BASE_RESNET,
    "ResNet18 two-stage + cosine, lr=3e-4": BASE_RESNET.variant(lr=3e-4),
    "ResNet18 batch size 64": BASE_RESNET.variant(batch_size=64),
    "ResNet18 no augmentation": BASE_RESNET.variant(augment=False),
    "ResNet18 weight decay 1e-2": BASE_RESNET.variant(weight_decay=1e-2),
}

# ----------------------------------------------------------------------------
# ViT-B/16 (Section 10): fine-tuning strategies
# ----------------------------------------------------------------------------
BASE_VIT = RunConfig(
    arch="vit_b_16",
    trainable_blocks=2,
    lr=1e-5,  # unfrozen backbone parameters (after the head epochs)
    finetune_head_lr=1e-4,  # classifier head after the head epochs
    head_lr=1e-3,  # classifier head during the head epochs
    head_epochs=3,
    epochs=int(env("EPOCHS_VIT", 20)),
    # counts only the epochs after the head-only epochs
    patience=int(env("PATIENCE_VIT", 5)),
    batch_size=16,
    weight_decay=0.01,
    scheduler="constant",
    class_weights=True,
    augment=True,
)
VIT_EXPERIMENTS = {
    "ViT-B/16 head only": BASE_VIT.variant(trainable_blocks=0),
    "ViT-B/16 partial (last 2 blocks)": BASE_VIT,
    "ViT-B/16 full fine-tuning": BASE_VIT.variant(trainable_blocks=None),
    "ViT-B/16 partial, no augmentation": BASE_VIT.variant(augment=False),
}

EXPERIMENTS = {**CNN_EXPERIMENTS, **RESNET_EXPERIMENTS, **VIT_EXPERIMENTS}


def changes_table(experiments: dict[str, RunConfig], base: RunConfig) -> pd.DataFrame:
    # What each experiment changes relative to its base recipe.
    changes = {
        name: ", ".join(f"{k}={v}" for k, v in cfg.changes_from(base).items())
        or "(base recipe)"
        for name, cfg in experiments.items()
    }
    return pd.Series(changes, name="changes from the base recipe").to_frame()


# ----------------------------------------------------------------------------
# Derived runs: '<selected model> + <suffix>' = the selected recipe with a change
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


def derived_names(base: str, suffixes: Iterable[str]) -> list[str]:
    # Run names of a model's derived runs, e.g. derived_names('X', COUNTERMEASURES)
    # -> ['X + random backgrounds', 'X + class-matched backgrounds'].
    return [f"{base} + {suffix}" for suffix in suffixes]


def suffix_label(name: str, base: str) -> str:
    # Short label of a derived run: ('X + random backgrounds', 'X') -> 'random backgrounds'.
    return "baseline" if name == base else name.removeprefix(f"{base} + ")


def config_of(name: str) -> RunConfig:
    # Recipe of a run name: an experiment, or a derived run ('<experiment> + <suffix>').
    for suffix, change in DERIVED.items():
        base = name.removesuffix(f" + {suffix}")
        if base != name:
            return config_of(base).variant(**change)
    return EXPERIMENTS[name]


# ----------------------------------------------------------------------------
# Selected models (OUT_DIR/final_models.json): CNN and ResNet18 in Section 9.3, ViT in Section 10.3
# ----------------------------------------------------------------------------
FINAL_MODELS_FILE = out_path("final_models.json")


def _read_selected() -> dict[str, str]:
    if not os.path.exists(FINAL_MODELS_FILE):
        return {}
    with open(FINAL_MODELS_FILE) as f:
        return json.load(f)


def save_selected(family: str, name: str) -> None:
    # Record the model selected on the validation set for a family ('cnn', 'resnet18' or 'vit_b_16').
    selected = {**_read_selected(), family: name}
    with open(FINAL_MODELS_FILE, "w") as f:
        json.dump(selected, f, indent=2)


def load_final_models() -> tuple[str, str, str]:
    # The selected models -> (BEST_CNN, BEST_RESNET, BEST_VIT).
    selected = _read_selected()
    missing = {"cnn", "resnet18", "vit_b_16"} - set(selected)
    assert not missing, (
        f"run 01_classification.ipynb up to Section 10 first (missing: {sorted(missing)})"
    )
    return selected["cnn"], selected["resnet18"], selected["vit_b_16"]
