"""Training runs as pydantic models: the recipe (settings) of a run and its history (results per epoch)."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


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
    # what is unfrozen after the head-only epochs
    finetune: Literal["head", "partial", "full"] = "full"
    # finetune="partial": number of last ViT encoder blocks that are trained
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
        if self.finetune == "partial" and self.trainable_blocks is None:
            raise ValueError('finetune="partial" needs trainable_blocks')
        return self

    def variant(self, **changes: Any) -> Self:
        # The same recipe with some settings changed (validated again).
        return type(self).model_validate({**self.model_dump(), **changes})

    def changes_from(self, base: "RunConfig") -> dict[str, Any]:
        # The settings that differ from `base`, e.g. {"lr": 0.0003} for base.variant(lr=3e-4).
        base_values = base.model_dump()
        return {k: v for k, v in self.model_dump().items() if v != base_values[k]}


class RunHistory(BaseModel):
    # History of one training run (saved in history.json): one entry per trained epoch in every list.
    # Unknown fields of older files are ignored.
    model_config = ConfigDict(extra="ignore")

    train_loss: list[float] = []
    train_acc: list[float] = []
    val_loss: list[float] = []
    val_acc: list[float] = []
    val_f1: list[float] = []  # macro-F1; the checkpoint of the best epoch is kept
    lr: list[float] = []  # learning rate at the start of each epoch
    train_time: list[float] = []  # seconds, training only (device synchronised)
    val_time: list[float] = []  # seconds
    epoch_time: list[float] = []  # train_time + val_time
    trainable_params: list[int] = []
    stopped_early: bool = False  # early stopping ended the run before `epochs`

    @model_validator(mode="after")
    def check_lengths(self) -> Self:
        lengths = {len(getattr(self, name)) for name in self.per_epoch_fields()}
        if len(lengths) > 1:
            raise ValueError("the per-epoch lists have different lengths")
        return self

    @classmethod
    def per_epoch_fields(cls) -> list[str]:
        return [name for name in cls.model_fields if name != "stopped_early"]

    def add_epoch(
        self,
        *,
        lr: float,
        trainable_params: int,
        train_loss: float,
        train_acc: float,
        train_time: float,
        val_loss: float,
        val_acc: float,
        val_f1: float,
        val_time: float,
    ) -> None:
        # Record one finished epoch: append every value to its per-epoch list.
        values = {
            "lr": lr,
            "trainable_params": trainable_params,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_time": train_time,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_time": val_time,
            "epoch_time": train_time + val_time,
        }
        for name, value in values.items():
            getattr(self, name).append(value)
