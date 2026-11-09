"""Photo table and split: which photos, which leaf, which series, which split (Sections 3 and 5)."""

import hashlib
import json
import os
from collections.abc import Collection

import numpy as np
import pandas as pd

from ..config import DATA_CONFIG, PATHS_CONFIG, out_path
from .leafmap import IMAGE_SUFFIXES, image_key, leaf_id


# ----------------------------------------------------------------------------
# Section 3: leaf grouping
# ----------------------------------------------------------------------------
def load_records() -> tuple[
    dict[str, list[str]], pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # All photos of the four classes -> (leaf_map, all_df, df: training series, other_df: left-out series).
    data_root = PATHS_CONFIG["data_root"]
    with open(os.path.join(data_root, "leaf-map.json")) as f:
        leaf_map = json.load(f)  # image identifier -> ['<class>:::<leaf group>', ...]

    records = []
    for label, class_name in enumerate(DATA_CONFIG["classes"]):
        class_dir = os.path.join(data_root, "raw", "color", class_name)
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(IMAGE_SUFFIXES):
                key = image_key(file_name)
                leaf, source = leaf_id(leaf_map, class_name, file_name)
                records.append(
                    {
                        "path": os.path.join(class_dir, file_name),
                        "file_name": file_name,
                        "label": label,
                        "class_name": class_name,
                        "key": key,
                        "series": key.split()[0],
                        "leaf_id": leaf,
                        "source": source,
                    }
                )
    all_df = pd.DataFrame(records)
    all_df["training_series"] = all_df["series"] == all_df["class_name"].map(
        DATA_CONFIG["train_series"]
    )
    all_df[all_df["leaf_id"].isna() & all_df["training_series"]].to_csv(
        out_path("excluded_images.csv"), index=False
    )
    # training photos: the training series of every class, official leaf ids only
    df = all_df[all_df["training_series"] & all_df["leaf_id"].notna()].reset_index(
        drop=True
    )
    # every other series of these classes is only used as the series test (Section 19.1)
    other_df = all_df[~all_df["training_series"]]
    series_size = other_df.groupby(["class_name", "series"])["path"].transform("size")
    other_df = other_df[series_size >= DATA_CONFIG["min_series_photos"]].reset_index(
        drop=True
    )
    return leaf_map, all_df, df, other_df


# ----------------------------------------------------------------------------
# Section 5: leaf-group split and leakage audit
# ----------------------------------------------------------------------------
def split_leaves(leaves: Collection[str]) -> dict[str, str]:
    # Shuffled leaf ids -> {leaf id: 'train' / 'val' / 'test'}, 70/15/15 in this order.
    train_frac, val_frac = DATA_CONFIG["split_fractions"]
    n_train, n_val = round(train_frac * len(leaves)), round(val_frac * len(leaves))
    return {
        leaf: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, leaf in enumerate(leaves)
    }


def make_split(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    # Split the official leaf ids of every class 70/15/15 and audit the result.
    # Adds df["split"]; returns (train_idx, val_idx, test_idx).
    rng = np.random.default_rng(DATA_CONFIG["split_seed"])
    split_of_leaf = {}
    for c in range(len(DATA_CONFIG["classes"])):
        leaves = rng.permutation(sorted(df.loc[df.label == c, "leaf_id"].unique()))
        split_of_leaf |= split_leaves(leaves)
    df["split"] = df["leaf_id"].map(split_of_leaf)

    # Audit: every official leaf id (all photos of one leaf) lies in exactly one split
    assert (df.groupby("leaf_id")["split"].nunique() == 1).all()
    return tuple(np.where(df.split == s)[0] for s in ["train", "val", "test"])


def data_fingerprint(df: pd.DataFrame, train_idx: np.ndarray) -> str:
    # Fingerprint of the training photos (file names), the series and the split seed:
    # a saved run is only reused for the same data.
    return hashlib.md5(  # noqa: S324
        (
            "|".join(df.loc[train_idx, "path"].map(os.path.basename))
            + json.dumps(DATA_CONFIG["train_series"], sort_keys=True)
            + str(DATA_CONFIG["split_seed"])
        ).encode()
    ).hexdigest()[:12]
