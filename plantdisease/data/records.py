"""Photo table and split: which photos, which leaf, which split (Sections 3 and 5)."""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

from ..config import DATA_CONFIG, PATHS_CONFIG
from .leafmap import IMAGE_SUFFIXES, get_image_key, get_leaf_id


# ----------------------------------------------------------------------------
# Section 3: leaf grouping
# ----------------------------------------------------------------------------
def load_records() -> pd.DataFrame:
    # All photos of the four classes, one row per photo. `source` says where the leaf id comes
    # from; only the photos with an official leaf id (source 'leaf map') are used (see prepare()).
    data_root = PATHS_CONFIG["data_root"]
    with open(os.path.join(data_root, "leaf-map.json")) as f:
        leaf_map = json.load(f)  # image identifier -> ['<class>:::<leaf group>', ...]

    all_records = []
    # loop over each class
    for class_index, class_name in enumerate(DATA_CONFIG["classes"]):
        class_dir = os.path.join(data_root, "raw", "color", class_name)

        # loop over each file and search for images
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(IMAGE_SUFFIXES):
                image_key = get_image_key(file_name)
                leaf_id, source = get_leaf_id(leaf_map, class_name, file_name)

                all_records.append(
                    {
                        "path": os.path.join(class_dir, file_name),
                        "file_name": file_name,
                        "class_index": class_index,
                        "class_name": class_name,
                        "image_key": image_key,
                        "leaf_id": leaf_id,
                        "source": source,
                    }
                )
    return pd.DataFrame(all_records)


# ----------------------------------------------------------------------------
# Section 5: leaf-group split and leakage audit
# ----------------------------------------------------------------------------
def make_split(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    # Split the official leaf ids of every class 70/15/15 (train / val / test) and audit the result.
    # Adds df["split"]; returns (train_idx, val_idx, test_idx).
    train_frac, val_frac = DATA_CONFIG["split_fractions"]
    generator = torch.Generator().manual_seed(DATA_CONFIG["split_seed"])
    split_of_leaf = {}
    for c in range(len(DATA_CONFIG["classes"])):
        # sorted leaf ids of the class, so the split only depends on the seed
        leaves = sorted(df.loc[df.class_index == c, "leaf_id"].unique())
        n_train, n_val = round(train_frac * len(leaves)), round(val_frac * len(leaves))
        n_test = len(leaves) - n_train - n_val

        # random_split shuffles the leaf ids and cuts them into three subsets of the given sizes
        parts = random_split(leaves, [n_train, n_val, n_test], generator=generator)

        # pairs up parts with their labels
        for name, part in zip(("train", "val", "test"), parts, strict=True):
            split_of_leaf.update(dict.fromkeys((leaves[i] for i in part.indices), name))

    # Adds "split" column to the dataframe
    df["split"] = df["leaf_id"].map(split_of_leaf)

    # Audit: every official leaf id (all photos of one leaf) lies in exactly one split
    assert (df.groupby("leaf_id")["split"].nunique() == 1).all()
    return tuple(np.where(df.split == s)[0] for s in ["train", "val", "test"])


def data_fingerprint(df: pd.DataFrame, train_idx: np.ndarray) -> str:
    # Fingerprint of the training photos (file names) and the split seed:
    # a saved run is only reused for the same data.
    return hashlib.md5(  # noqa: S324
        (
            "|".join(df.loc[train_idx, "path"].map(os.path.basename))
            + str(DATA_CONFIG["split_seed"])
        ).encode()
    ).hexdigest()[:12]
