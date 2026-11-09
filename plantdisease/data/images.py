"""Image cache and image versions: decode and resize once, segmented and grayscale versions (Section 4)."""

import os
from collections.abc import Iterable
from functools import cache

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from ..config import DATA_CONFIG, PATHS_CONFIG


def load_resized(path: str, size: int = DATA_CONFIG["img_size"]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # (3, H, W) uint8


def build_cache(paths: Iterable[str], name: str) -> torch.Tensor:
    # Decode + resize all images once and store them as a uint8 tensor (N, 3, H, W).
    file = os.path.join(
        PATHS_CONFIG["cache_dir"], f"{name}_{DATA_CONFIG['img_size']}.pt"
    )
    paths = list(paths)
    if os.path.exists(file):
        cache = torch.load(file)
        if cache["paths"] == paths:
            return cache["images"]
    images = torch.stack([load_resized(p) for p in paths])
    torch.save({"paths": paths, "images": images}, file)
    return images


@cache
def segmented_by_name(folder: str) -> dict[str, str]:
    # {original photo name: segmented file} of a segmented folder, for colour photos without the
    # UUID prefix ('RS_Rust 1563.JPG' -> '<uuid>___RS_Rust 1563_final_masked.jpg').
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
    folder = os.path.join(PATHS_CONFIG["data_root"], "raw", "segmented", class_name)
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
