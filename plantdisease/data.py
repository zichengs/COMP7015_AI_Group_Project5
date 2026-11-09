"""Data protocol: download, leaf grouping, image cache, split, loaders (Sections 2-6)."""

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .config import CLASSES, DATA, PATHS, out_path
from .device import device
from .leafmap import image_key, leaf_id

DATA_ROOT = PATHS["data_root"]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------
# Section 2: download
# ----------------------------------------------------------------------------
def sparse_checkout(url: str, root: str, folders: list[str]) -> list[str]:
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


def field_records(folders: dict[str, int]) -> pd.DataFrame:
    # PlantDoc photos (Section 19.2, notebook 07) of `folders` ({PlantDoc folder: label}), from the
    # authors' train and test folders: all are test data here. Missing folders are downloaded;
    # folders that are not complete (download failed) are skipped.
    root = PATHS["field_root"]
    wanted = [f"{split}/{folder}" for folder in folders for split in ("train", "test")]
    available = download_github_folders(DATA["field_url"], root, wanted)
    rows = []
    for folder, label in folders.items():
        for split in ("train", "test"):
            if f"{split}/{folder}" not in available:
                continue
            folder_dir = os.path.join(root, split, folder)
            rows += [
                {"path": os.path.join(folder_dir, f), "label": label, "folder": folder}
                for f in sorted(os.listdir(folder_dir))
                if f.lower().endswith(IMAGE_SUFFIXES)
            ]
    return pd.DataFrame(rows, columns=["path", "label", "folder"])


def download_github_folders(url: str, root: str, folders: list[str]) -> list[str]:
    # Download `folders` of a GitHub repository file by file over HTTPS (no git needed); returns the
    # folders that were downloaded completely. Characters that Windows does not allow in file names
    # (PlantDoc has names like '...picture-id81781575?s=612x612.jpg') are replaced by '_'.
    repo = url.removeprefix("https://github.com/").strip("/")
    done = [os.path.join(root, f, ".complete") for f in folders]
    missing = [f for f, d in zip(folders, done, strict=True) if not os.path.exists(d)]
    if missing:
        try:
            api = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
            with urllib.request.urlopen(api, timeout=60) as response:  # noqa: S310
                tree = json.load(response)["tree"]
            for folder in missing:
                files = [
                    e["path"]
                    for e in tree
                    if e["type"] == "blob" and os.path.dirname(e["path"]) == folder
                ]
                os.makedirs(os.path.join(root, folder), exist_ok=True)
                print(f"Downloading {folder} ({len(files)} files)")
                for path in files:
                    name = re.sub(r'[<>:"/\\|?*]', "_", os.path.basename(path))
                    target = os.path.join(root, folder, name)
                    if os.path.exists(target):
                        continue
                    raw = f"https://raw.githubusercontent.com/{repo}/HEAD/{urllib.parse.quote(path)}"
                    for attempt in range(3):
                        try:
                            with urllib.request.urlopen(raw, timeout=60) as response:  # noqa: S310
                                with open(target + ".part", "wb") as f:
                                    shutil.copyfileobj(response, f)
                            break
                        except OSError:
                            if attempt == 2:
                                raise
                    os.replace(target + ".part", target)
                open(os.path.join(root, folder, ".complete"), "w").close()
        except OSError as e:
            print(f"Download from {url} stopped: {e}. Run this again to resume.")
    return [f for f, d in zip(folders, done, strict=True) if os.path.exists(d)]


def download_plantvillage(extra_folders: Iterable[str] = ()) -> list[str]:
    # The four tomato folders of raw/color (training images) and raw/segmented (background removed),
    # plus `extra_folders` (notebook 07). 'sparse-checkout set' replaces the folder list, so the four
    # tomato folders are always included.
    folders = [f"raw/{kind}/{c}" for kind in ["color", "segmented"] for c in CLASSES]
    folders += list(extra_folders)
    sparse_checkout(
        "https://github.com/spMohanty/PlantVillage-Dataset", DATA_ROOT, folders
    )
    return folders


# ----------------------------------------------------------------------------
# Section 3: leaf grouping
# ----------------------------------------------------------------------------
def load_records() -> tuple[
    dict[str, list[str]], pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # All photos of the four classes -> (leaf_map, all_df, df: training series, other_df: left-out series).
    with open(os.path.join(DATA_ROOT, "leaf-map.json")) as f:
        leaf_map = json.load(f)  # image identifier -> ['<class>:::<leaf group>', ...]

    records = []
    for label, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_ROOT, "raw", "color", class_name)
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
        DATA["train_series"]
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
    other_df = other_df[series_size >= DATA["min_series_photos"]].reset_index(drop=True)
    return leaf_map, all_df, df, other_df


# ----------------------------------------------------------------------------
# Section 4: image cache and background brightness
# ----------------------------------------------------------------------------
def load_resized(path: str, size: int = DATA["img_size"]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # (3, H, W) uint8


def build_cache(paths: Iterable[str], name: str) -> torch.Tensor:
    # Decode + resize all images once and store them as a uint8 tensor (N, 3, H, W).
    file = os.path.join(PATHS["cache_dir"], f"{name}_{DATA['img_size']}.pt")
    paths = list(paths)
    if os.path.exists(file):
        cache = torch.load(file)
        if cache["paths"] == paths:
            return cache["images"]
    images = torch.stack([load_resized(p) for p in paths])
    torch.save({"paths": paths, "images": images}, file)
    return images


# ----------------------------------------------------------------------------
# Section 5: leaf-group split and leakage audit
# ----------------------------------------------------------------------------
def split_leaves(leaves: Collection[str]) -> dict[str, str]:
    # Shuffled leaf ids -> {leaf id: 'train' / 'val' / 'test'}, 70/15/15 in this order.
    train_frac, val_frac = DATA["split_fractions"]
    n_train, n_val = round(train_frac * len(leaves)), round(val_frac * len(leaves))
    return {
        leaf: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, leaf in enumerate(leaves)
    }


def make_split(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    # Split the official leaf ids of every class 70/15/15 and audit the result.
    # Adds df["split"]; returns (train_idx, val_idx, test_idx).
    rng = np.random.default_rng(DATA["split_seed"])
    split_of_leaf = {}
    for c in range(len(CLASSES)):
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
            + json.dumps(DATA["train_series"], sort_keys=True)
            + str(DATA["split_seed"])
        ).encode()
    ).hexdigest()[:12]


@cache
def segmented_by_name(folder: str) -> dict[str, str]:
    # {original photo name: segmented file} of a segmented folder, for colour photos without the
    # UUID prefix ('RS_Rust 1563.JPG' -> '<uuid>___RS_Rust 1563_final_masked.jpg', notebook 07).
    if not os.path.isdir(folder):
        return {}
    return {
        f.split("___")[-1].removesuffix("_final_masked.jpg"): f
        for f in os.listdir(folder)
    }


def segmented_path(path: str) -> str:
    # Segmented counterpart of a photo ('<name>_final_masked.jpg').
    # A few segmented files lack the UUID prefix ('Com.G_SpM_FL 1378_final_masked.jpg'):
    # then the original photo name after '___' is used.
    class_name, file_name = path.split(os.sep)[-2:]
    folder = os.path.join(DATA_ROOT, "raw", "segmented", class_name)
    stem = os.path.splitext(file_name)[0]
    exact = os.path.join(folder, stem + "_final_masked.jpg")
    if os.path.exists(exact):
        return exact
    by_name = os.path.join(folder, stem.split("___")[-1] + "_final_masked.jpg")
    if os.path.exists(by_name) or "___" in stem:
        return by_name
    match = segmented_by_name(folder).get(stem)
    return os.path.join(folder, match) if match else by_name


def load_segmented(paths: Iterable[str], name: str) -> tuple[torch.Tensor, np.ndarray]:
    # Segmented versions of `paths` (aligned; black where no segmented photo exists) and which exist.
    seg_paths = [segmented_path(p) for p in paths]
    exists = np.array([os.path.exists(p) for p in seg_paths])
    found = build_cache([p for p, e in zip(seg_paths, exists, strict=True) if e], name)
    images = torch.zeros(len(seg_paths), *found.shape[1:], dtype=torch.uint8)
    images[torch.from_numpy(exists)] = found
    return images, exists


def to_grayscale(images: torch.Tensor) -> torch.Tensor:
    # Grayscale version with three identical channels, so that the same (pre-trained) networks apply.
    return v2.functional.rgb_to_grayscale(images, num_output_channels=3)


# ----------------------------------------------------------------------------
# Section 6: preprocessing, augmentation, loaders
# ----------------------------------------------------------------------------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics


class RandomRot90(nn.Module):
    # Rotate by a random multiple of 90 degrees (lossless, no black corners).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.images, self.labels, self.indices, self.transform = (
            images,
            labels,
            indices,
            transform,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = self.indices[i]
        return self.transform(self.images[j]), self.labels[j]


def make_loader(
    dataset: Dataset, batch_size: int, seed: int | None = None
) -> DataLoader:
    # Shuffled (seeded) loader for training, ordered loader for evaluation.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=seed is not None,
        num_workers=DATA["num_workers"],
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        pin_memory=device.type == "cuda",
        persistent_workers=DATA["num_workers"] > 0,
    )


def eval_loader(
    images: torch.Tensor,
    labels: torch.Tensor | np.ndarray | None = None,
    indices: np.ndarray | None = None,
    batch_size: int = 64,
) -> DataLoader:
    # Ordered loader over (a subset of) an image tensor with the evaluation transform.
    indices = np.arange(len(images)) if indices is None else indices
    labels = torch.zeros(len(images), dtype=torch.long) if labels is None else labels
    return make_loader(
        CachedDataset(images, torch.as_tensor(labels), indices, eval_tf), batch_size
    )


def class_weights(df: pd.DataFrame, train_idx: np.ndarray) -> torch.Tensor:
    # Class-weighted loss: w_c = N / (K * n_c) over the training photos.
    k = max(len(CLASSES), df.label.max() + 1)  # more than four classes in notebook 07
    train_counts = np.bincount(df.label[train_idx], minlength=k)
    return torch.tensor(len(train_idx) / (k * train_counts), dtype=torch.float32)


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
        # imported here: backgrounds imports this module
        from .backgrounds import training_masks  # noqa: PLC0415

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
        labels=torch.tensor(df["label"].values),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        data_id=data_fingerprint(df, train_idx),
    )
