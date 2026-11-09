"""The Data object: everything of the data protocol in one call, `prepare()` (used by notebooks 02-06)."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import torch

from ..config import DATA_CONFIG
from .backgrounds import training_masks
from .images import build_cache, load_segmented, to_grayscale
from .records import data_fingerprint, load_records, make_split


def class_weights(df: pd.DataFrame, train_idx: np.ndarray) -> torch.Tensor:
    # Class-weighted loss: w_c = N / (K * n_c) over the training photos.
    k = len(DATA_CONFIG["classes"])
    train_counts = np.bincount(df.class_index[train_idx], minlength=k)
    return torch.tensor(len(train_idx) / (k * train_counts), dtype=torch.float32)


@dataclass
class Data:
    df: pd.DataFrame  # training series: one row per photo, with leaf id and split
    other_df: pd.DataFrame  # left-out series (Section 19.1)
    images: torch.Tensor  # cached uint8 images (N, 3, H, W), aligned with df
    labels: torch.Tensor
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    data_id: str

    @property
    def class_weights(self) -> torch.Tensor:
        return class_weights(self.df, self.train_idx)

    @property
    def y_test(self) -> np.ndarray:
        return self.labels[self.test_idx].numpy()

    @cached_property
    def segmented(self) -> tuple[torch.Tensor, np.ndarray]:
        # Background-removed version of every photo -> (images, exists), aligned with self.images.
        return load_segmented(self.df["path"], "segmented")

    @cached_property
    def swap_masks(self) -> tuple[torch.Tensor, np.ndarray]:
        # Leaf masks of all photos for the background swap of Section 18 -> (masks, usable).
        return training_masks(self)

    @cached_property
    def grayscale(self) -> torch.Tensor:
        return to_grayscale(self.images)

    def input_images(self, variant: str) -> torch.Tensor:
        # The photos in one input version: 'color', 'grayscale' or 'segmented' (background removed).
        return {
            "color": lambda: self.images,
            "grayscale": lambda: self.grayscale,
            "segmented": lambda: self.segmented[0],
        }[variant]()


def prepare() -> Data:
    # Rebuild the deterministic data protocol of notebook 01 from the image cache.
    _, _, df, other_df = load_records()
    images = build_cache(df["path"], "color")
    train_idx, val_idx, test_idx = make_split(df)
    return Data(
        df=df,
        other_df=other_df,
        images=images,
        labels=torch.tensor(df["class_index"].values),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        data_id=data_fingerprint(df, train_idx),
    )
