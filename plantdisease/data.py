"""Data protocol: download, leaf grouping, image cache, split, loaders (Sections 2-6)."""

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
    # Clone only `folders` of a git repository (sparse checkout); returns the folders that exist afterwards.
    git = shutil.which("git")
    if not os.path.exists(root):
        subprocess.run(  # noqa: S603
            [git, "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, root],
            check=False,
        )
    if not all(os.path.isdir(os.path.join(root, f)) for f in folders):
        subprocess.run([git, "sparse-checkout", "set", *folders], cwd=root, check=False)  # noqa: S603
    return [f for f in folders if os.path.isdir(os.path.join(root, f))]


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
    # numbers >= 1000 are photo counters, not leaf numbers
    if match and int(match.group(2)) < 1000:
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
    # every other series of these classes is only used as the series test (Section 19.1)
    other_df = all_df[~all_df["training_series"]]
    series_size = other_df.groupby(["class_name", "series"])["path"].transform("size")
    other_df = other_df[series_size >= CFG["min_series_photos"]].reset_index(drop=True)
    return leaf_map, all_df, df, other_df


# ----------------------------------------------------------------------------
# Section 4: image cache and background brightness
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
def make_split(df):
    # Split the official leaf ids of every class 70/15/15 and audit the result.
    # Adds df["split"]; returns (train_idx, val_idx, test_idx).
    rng = np.random.default_rng(CFG["split_seed"])
    train_frac, val_frac = CFG["split_fractions"]
    split_of_leaf = {}
    for c in range(len(CLASSES)):
        leaves = rng.permutation(sorted(df.loc[df.label == c, "leaf_id"].unique()))
        n_train, n_val = round(train_frac * len(leaves)), round(val_frac * len(leaves))
        for i, leaf in enumerate(leaves):
            split_of_leaf[leaf] = (
                "train" if i < n_train else "val" if i < n_train + n_val else "test"
            )
    df["split"] = df["leaf_id"].map(split_of_leaf)

    # Audit: every official leaf id (all photos of one leaf) lies in exactly one split
    assert (df.groupby("leaf_id")["split"].nunique() == 1).all()
    return tuple(np.where(df.split == s)[0] for s in ["train", "val", "test"])


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
    # A few segmented files lack the UUID prefix ('Com.G_SpM_FL 1378_final_masked.jpg'):
    # then the original photo name after '___' is used.
    class_name, file_name = path.split(os.sep)[-2:]
    folder = os.path.join(DATA_ROOT, "raw", "segmented", class_name)
    stem = os.path.splitext(file_name)[0]
    exact = os.path.join(folder, stem + "_final_masked.jpg")
    if os.path.exists(exact):
        return exact
    return os.path.join(folder, stem.split("___")[-1] + "_final_masked.jpg")


def load_segmented(paths, name):
    # Segmented versions of `paths` (aligned; black where no segmented photo exists) and which exist.
    seg_paths = [segmented_path(p) for p in paths]
    exists = np.array([os.path.exists(p) for p in seg_paths])
    found = build_cache([p for p, e in zip(seg_paths, exists, strict=True) if e], name)
    images = torch.zeros(len(seg_paths), *found.shape[1:], dtype=torch.uint8)
    images[torch.from_numpy(exists)] = found
    return images, exists


def to_grayscale(images):
    # Grayscale version with three identical channels, so that the same (pre-trained) networks apply.
    return v2.functional.rgb_to_grayscale(images, num_output_channels=3)


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
# Everything above in one call (used by notebooks 02-06)
# ----------------------------------------------------------------------------
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
    def class_weights(self):
        return class_weights(self.df, self.train_idx)

    @property
    def y_test(self):
        return self.labels[self.test_idx].numpy()

    @cached_property
    def segmented(self):
        # Background-removed version of every photo -> (images, exists), aligned with self.images.
        return load_segmented(self.df["path"], "segmented")

    @cached_property
    def swap_masks(self):
        # Leaf masks of all photos for the background swap of Section 18 -> (masks, usable).
        # imported here: backgrounds imports this module
        from .backgrounds import training_masks  # noqa: PLC0415

        return training_masks(self)

    @cached_property
    def grayscale(self):
        return to_grayscale(self.images)

    def input_images(self, variant):
        # The photos in one input version: 'color', 'grayscale' or 'segmented' (background removed).
        return {
            "color": lambda: self.images,
            "grayscale": lambda: self.grayscale,
            "segmented": lambda: self.segmented[0],
        }[variant]()


def prepare():
    # Rebuild the deterministic data protocol of notebook 01 from the image cache.
    _, _, df, other_df = load_records()
    images = build_cache(df["path"], "color")
    train_idx, val_idx, test_idx = make_split(df)
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
