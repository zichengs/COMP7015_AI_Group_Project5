"""Data protocol: download, leaf grouping, image cache, near-duplicates, split, loaders (Sections 2-6)."""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .config import CFG, CLASSES, device, out_path

DATA_ROOT = CFG["data_root"]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------
# Section 2: download
# ----------------------------------------------------------------------------
def sparse_checkout(url, root, folders):
    # Clone only `folders` of a git repository (sparse checkout); returns True if all folders exist afterwards.
    git = shutil.which("git")
    if not os.path.exists(root):
        subprocess.run(  # noqa: S603
            [git, "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, root],
            check=False,
        )
    if not all(os.path.isdir(os.path.join(root, f)) for f in folders):
        subprocess.run([git, "sparse-checkout", "set", *folders], cwd=root, check=False)  # noqa: S603
    return all(os.path.isdir(os.path.join(root, f)) for f in folders)


def download_plantvillage():
    # The four tomato folders of raw/color (training images) and raw/segmented (background removed).
    folders = [f"raw/{kind}/{c}" for kind in ["color", "segmented"] for c in CLASSES]
    sparse_checkout(
        "https://github.com/spMohanty/PlantVillage-Dataset", DATA_ROOT, folders
    )
    return folders


# ----------------------------------------------------------------------------
# Section 3: leaf grouping
# ----------------------------------------------------------------------------
SERIES_PATTERN = re.compile(
    r"^(gh_hl|ghlb_ps|ghlb2es|ghlb2|ghlb) leaf (\d+)"
)  # greenhouse series with leaf numbers


def image_key(file_name):
    # '0031da2a-...___RS_HL 0105 copy.JPG' -> 'rs_hl 0105' (same logic as the repository's plant_village.py)
    key = file_name.split("___")[-1].split("copy")[0].strip()
    return re.sub(r"\.(jpe?g|png)$", "", key, flags=re.IGNORECASE).strip().lower()


def get_leaf_id(leaf_map, class_name, key):
    # Return (leaf id, source). The leaf id is None if the image has no usable leaf information.
    for entry in leaf_map.get(key, []):
        if entry.split(":::")[0] == class_name:
            return entry, "leaf map"
    match = SERIES_PATTERN.match(key)
    if (
        match and int(match.group(2)) < 1000
    ):  # numbers >= 1000 are photo counters, not leaf numbers
        return f"{class_name}:::{match.group(1)} leaf {match.group(2)}", "file name"
    return None, "excluded: no leaf information"


def load_records():
    # All photos of the four classes -> (leaf_map, all_df, df: training series, other_df: left-out series).
    with open(os.path.join(DATA_ROOT, "leaf-map.json")) as f:
        leaf_map = json.load(f)  # image identifier -> ['<class>:::<leaf group>', ...]

    records = []
    for label, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_ROOT, "raw", "color", class_name)
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(IMAGE_SUFFIXES):
                key = image_key(file_name)
                leaf_id, source = get_leaf_id(leaf_map, class_name, key)
                records.append(
                    {
                        "path": os.path.join(class_dir, file_name),
                        "file_name": file_name,
                        "label": label,
                        "class_name": class_name,
                        "key": key,
                        "series": key.split()[0],
                        "leaf_id": leaf_id,
                        "source": source,
                    }
                )
    all_df = pd.DataFrame(records)
    all_df["training_series"] = all_df["series"] == all_df["class_name"].map(
        CFG["train_series"]
    )
    all_df[all_df["leaf_id"].isna() & all_df["training_series"]].to_csv(
        out_path("excluded_images.csv"), index=False
    )
    # training photos: the training series of every class, official leaf ids only
    df = all_df[all_df["training_series"] & all_df["leaf_id"].notna()].reset_index(
        drop=True
    )
    assert (df["source"] == "leaf map").all(), "training series need official leaf ids"
    # every other series of these classes is only used as the series test (Section 15)
    other_df = all_df[~all_df["training_series"]]
    series_size = other_df.groupby(["class_name", "series"])["path"].transform("size")
    other_df = other_df[series_size >= CFG["min_series_photos"]].reset_index(drop=True)
    single = pd.Series("single " + other_df.index.astype(str), index=other_df.index)
    # photos without leaf information become a group of their own
    other_df["group"] = other_df["leaf_id"].fillna(single)
    return leaf_map, all_df, df, other_df


# ----------------------------------------------------------------------------
# Section 4: image cache, near-duplicates, background brightness
# ----------------------------------------------------------------------------
def load_resized(path, size=CFG["img_size"]):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # (3, H, W) uint8


def build_cache(paths, name):
    # Decode + resize all images once and store them as a uint8 tensor (N, 3, H, W).
    file = os.path.join(CFG["cache_dir"], f"{name}_{CFG['img_size']}.pt")
    paths = list(paths)
    if os.path.exists(file):
        cache = torch.load(file)
        if cache["paths"] == paths:
            return cache["images"]
    images = torch.stack([load_resized(p) for p in paths])
    torch.save({"paths": paths, "images": images}, file)
    return images


def chunked(fn, images, chunk=512):
    # Apply fn to a large uint8 image tensor in chunks of grayscale float images (keeps memory low).
    return torch.cat(
        [
            fn(images[i : i + chunk].float().mean(1, keepdim=True))
            for i in range(0, len(images), chunk)
        ]
    )


def dhash_bits(gray, n=16):
    # 256-bit difference hash: is each pixel brighter than its right neighbour (on a 17x16 grayscale thumbnail)?
    small = F.interpolate(gray, size=(n, n + 1), mode="area")[:, 0]
    return (small[:, :, 1:] > small[:, :, :-1]).flatten(1).float()


def hamming_pairs(bits, max_bits):
    # All index pairs (i < j) whose hashes differ in at most `max_bits` bits.
    dist = bits @ (1 - bits).T + (1 - bits) @ bits.T
    i, j = torch.where(torch.triu(dist <= max_bits, diagonal=1))
    return list(zip(i.tolist(), j.tolist(), strict=True))


def merge_near_duplicates(df, hashes):
    # Union-find: merge the leaf groups of near-duplicate images (within the same class).
    # Adds df["group"] and returns the number of merges.
    parent = {g: g for g in df["leaf_id"]}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    merges = 0
    for c in range(len(CLASSES)):
        idx = np.where(df["label"] == c)[0]
        for i, j in hamming_pairs(hashes[idx], CFG["dup_merge_bits"]):
            a, b = find(df.at[idx[i], "leaf_id"]), find(df.at[idx[j], "leaf_id"])
            if a != b:
                parent[a] = b
                merges += 1
    df["group"] = df["leaf_id"].map(find)
    return merges


def border_brightness(gray, b=12):
    # Mean gray level of a 12-pixel border around each image.
    g = gray[:, 0]
    return torch.cat(
        [
            g[:, :b].flatten(1),
            g[:, -b:].flatten(1),
            g[:, :, :b].flatten(1),
            g[:, :, -b:].flatten(1),
        ],
        1,
    ).mean(1)


# ----------------------------------------------------------------------------
# Section 5: leaf-group split and leakage audit
# ----------------------------------------------------------------------------
def make_split(df, hashes):
    # Split the leaf groups of every class 70/15/15 and audit the result.
    # Adds df["split"]; returns (train_idx, val_idx, test_idx, near-duplicate pairs across splits).
    rng = np.random.default_rng(CFG["split_seed"])
    train_frac, val_frac = CFG["split_fractions"]
    split_of_group = {}
    for c in range(len(CLASSES)):
        groups = rng.permutation(sorted(df.loc[df.label == c, "group"].unique()))
        n_train, n_val = round(train_frac * len(groups)), round(val_frac * len(groups))
        for i, g in enumerate(groups):
            split_of_group[g] = (
                "train" if i < n_train else "val" if i < n_train + n_val else "test"
            )
    df["split"] = df["group"].map(split_of_group)

    # Audit 1: every group and every official leaf id lies in exactly one split
    assert (df.groupby("group")["split"].nunique() == 1).all()
    assert (df.groupby("leaf_id")["split"].nunique() == 1).all()
    # Audit 2: near-duplicate pairs (looser threshold) that cross splits
    cross = 0
    for c in range(len(CLASSES)):
        idx = np.where(df.label == c)[0]
        cross += sum(
            df.at[idx[i], "split"] != df.at[idx[j], "split"]
            for i, j in hamming_pairs(hashes[idx], CFG["dup_audit_bits"])
        )
    train_idx, val_idx, test_idx = (
        np.where(df.split == s)[0] for s in ["train", "val", "test"]
    )
    return train_idx, val_idx, test_idx, cross


def data_fingerprint(df, train_idx):
    # Fingerprint of the training photos (file names), the series and the split seed:
    # a saved run is only reused for the same data.
    return hashlib.md5(  # noqa: S324
        (
            "|".join(df.loc[train_idx, "path"].map(os.path.basename))
            + json.dumps(CFG["train_series"], sort_keys=True)
            + str(CFG["split_seed"])
        ).encode()
    ).hexdigest()[:12]


def segmented_path(path):
    # Segmented counterpart of a photo ('<name>_final_masked.jpg').
    class_name, file_name = path.split(os.sep)[-2:]
    return os.path.join(
        DATA_ROOT,
        "raw",
        "segmented",
        class_name,
        os.path.splitext(file_name)[0] + "_final_masked.jpg",
    )


# ----------------------------------------------------------------------------
# Section 6: preprocessing, augmentation, loaders
# ----------------------------------------------------------------------------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics


class RandomRot90(nn.Module):
    # Rotate by a random multiple of 90 degrees (lossless, no black corners).
    def forward(self, x):
        return torch.rot90(x, random.randint(0, 3), dims=(-2, -1))  # noqa: S311


normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
train_tf = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        RandomRot90(),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        normalize,
    ]
)
eval_tf = normalize


class CachedDataset(Dataset):
    # Serves images from a cached uint8 tensor (indexed, not copied) and applies a transform.
    def __init__(self, images, labels, indices, transform):
        self.images, self.labels, self.indices, self.transform = (
            images,
            labels,
            indices,
            transform,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(self.images[j]), self.labels[j]


def make_loader(dataset, batch_size, seed=None):
    # Shuffled (seeded) loader for training, ordered loader for evaluation.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=seed is not None,
        num_workers=CFG["num_workers"],
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        pin_memory=device.type == "cuda",
        persistent_workers=CFG["num_workers"] > 0,
    )


def eval_loader(images, labels=None, indices=None, batch_size=64):
    # Ordered loader over (a subset of) an image tensor with the evaluation transform.
    indices = np.arange(len(images)) if indices is None else indices
    labels = torch.zeros(len(images), dtype=torch.long) if labels is None else labels
    return make_loader(
        CachedDataset(images, torch.as_tensor(labels), indices, eval_tf), batch_size
    )


def class_weights(df, train_idx):
    # Class-weighted loss: w_c = N / (K * n_c) over the training photos.
    train_counts = np.bincount(df.label[train_idx], minlength=len(CLASSES))
    return torch.tensor(
        len(train_idx) / (len(CLASSES) * train_counts), dtype=torch.float32
    )


# ----------------------------------------------------------------------------
# Everything above in one call (used by notebooks 02-05)
# ----------------------------------------------------------------------------
@dataclass
class Data:
    df: pd.DataFrame  # training series: one row per photo, with group and split
    other_df: pd.DataFrame  # left-out series (Section 15)
    images: torch.Tensor  # cached uint8 images (N, 3, H, W), aligned with df
    labels: torch.Tensor
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    data_id: str

    @property
    def class_weights(self):
        return class_weights(self.df, self.train_idx)

    @property
    def y_test(self):
        return self.labels[self.test_idx].numpy()

    @property
    def test_groups(self):
        # leaf group of every test image
        return self.df.iloc[self.test_idx]["group"].to_numpy()

    @cached_property
    def swap_masks(self):
        # Leaf masks of all photos for the background swap of Section 14 -> (masks, usable).
        # imported here: backgrounds imports this module
        from .backgrounds import training_masks  # noqa: PLC0415

        return training_masks(self)


def prepare():
    # Rebuild the deterministic data protocol of notebook 01 from the image cache.
    _, _, df, other_df = load_records()
    images = build_cache(df["path"], "color")
    hashes = chunked(dhash_bits, images)
    merge_near_duplicates(df, hashes)
    train_idx, val_idx, test_idx, _ = make_split(df, hashes)
    return Data(
        df=df,
        other_df=other_df,
        images=images,
        labels=torch.tensor(df["label"].values),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        data_id=data_fingerprint(df, train_idx),
    )
