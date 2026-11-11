"""The Data object: everything of the data protocol in one call, `prepare()` (used by every notebook)."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import torch

from ..config import DATA_CONFIG
from .images import build_cache, has_leaf, leaf_masks, load_segmented, to_grayscale
from .records import data_fingerprint, load_records, make_split


@dataclass
class Data:
    # photos with an official leaf id: one row per photo, with leaf id and split
    df: pd.DataFrame
    images: torch.Tensor  # cached uint8 images (N, 3, H, W), aligned with df
    labels: torch.Tensor
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    # fingerprint of the training photos and the split seed: a saved run is only reused for the same data
    data_id: str

    @property
    def class_weights(self) -> torch.Tensor:
        # Class-weighted loss: w_c = N / (K * n_c) over the training photos.
        k = len(DATA_CONFIG["classes"])
        train_counts = np.bincount(self.labels[self.train_idx].numpy(), minlength=k)
        return torch.tensor(
            len(self.train_idx) / (k * train_counts), dtype=torch.float32
        )

    @property
    def y_test(self) -> np.ndarray:
        return self.labels[self.test_idx].numpy()

    @cached_property
    def segmented(self) -> tuple[torch.Tensor, np.ndarray]:
        # Background-removed version of every photo -> (images, exists), aligned with self.images.
        return load_segmented(self.df["path"])

    @cached_property
    def grayscale(self) -> torch.Tensor:
        return to_grayscale(self.images)

    @cached_property
    def swap_masks(self) -> tuple[torch.Tensor, np.ndarray]:
        # Leaf masks of all photos (aligned with self.images) and which of them can be used for the
        # background swap of Section 20 -> (masks, usable).
        seg_all, exists = self.segmented
        print(f"{(~exists).sum()} of {len(exists)} photos have no segmented version")
        # (N, H, W) bool; photos without a segmented version get an empty mask
        masks = leaf_masks(seg_all)
        swap_ok = has_leaf(masks)
        print(
            f"Leaf masks cover {masks.float().mean():.0%} of the image on average | "
            f"{(~swap_ok).sum()} of {len(swap_ok)} photos have an empty or missing mask "
            f"and are never changed"
        )
        return masks, swap_ok

    def input_images(self, variant: str) -> torch.Tensor:
        # The photos in one input version: 'color', 'grayscale' or 'segmented' (background removed).
        if variant == "color":
            return self.images
        if variant == "grayscale":
            return self.grayscale
        if variant == "segmented":
            return self.segmented[0]
        raise ValueError(variant)


def prepare() -> Data:
    # The deterministic data protocol (Sections 3-5): the photos with an official leaf id, their
    # cached images and the leaf-group split.
    all_df = load_records()
    df = all_df[all_df["source"] == "leaf map"]
    max_leaves = DATA_CONFIG["max_leaves_per_class"]
    if max_leaves > 0:
        # smoke test: the first N leaf ids of every class (sorted, so deterministic), all their photos
        kept = df.groupby("class_index")["leaf_id"].apply(
            lambda ids: sorted(ids.unique())[:max_leaves]
        )
        df = df[df["leaf_id"].isin([leaf for ids in kept for leaf in ids])]
    df = df.reset_index(drop=True)
    images = build_cache(df["path"], "color")
    train_idx, val_idx, test_idx = make_split(df)
    return Data(
        df=df,
        images=images,
        labels=torch.tensor(df["class_index"].values),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        data_id=data_fingerprint(df, train_idx),
    )
