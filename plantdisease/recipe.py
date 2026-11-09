"""Training recipes (Sections 9, 10, 15 and 18): the settings of one training run, as pydantic models."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class RunConfig(BaseModel):
    # Everything that defines one training run. Unknown fields are rejected, so a typo in a recipe
    # fails when the recipe is built. A recipe is immutable: `variant` makes a changed copy.
    model_config = ConfigDict(extra="forbid", frozen=True)

    arch: Literal["cnn", "se_cnn", "resnet18", "vit_b_16"]  # "se_cnn": custom CNN with SE attention
    lr: float  # learning rate of the whole network (ViT: of the unfrozen backbone parameters)
    batch_size: int
    dropout: float = 0.0  # before the classifier (custom CNN only)
    augment: bool
    class_weights: bool  # class-weighted loss
    scheduler: Literal["cosine", "constant"]
    weight_decay: float
    epochs: int  # maximum, including the head-only epochs
    patience: int = 0  # early stopping after this many epochs without a better val macro-F1 (0 = off)

    # pre-trained models: head-only epochs first, then fine-tuning
    head_epochs: int = 0
    head_lr: float | None = None  # classifier head during the head-only epochs
    finetune_head_lr: float | None = None  # classifier head after them (None: same as `lr`)
    finetune: Literal["head", "partial", "full"] = "full"  # what is unfrozen after the head epochs
    trainable_blocks: int | None = None  # finetune="partial": number of last ViT encoder blocks

    # derived runs (Sections 15, 18 and notebook 07)
    input: Literal["color", "grayscale", "segmented"] = "color"
    background_swap: float = 0.0  # share of the training photos with a random background
    class_matched_background: bool = False  # control: the background comes from the same class
    num_classes: int | None = None  # None: the four tomato classes

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
