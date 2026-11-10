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

    # Define save path
    file = os.path.join(
        PATHS_CONFIG["cache_dir"], f"{name}_{DATA_CONFIG['img_size']}.pt"
    )
    paths = list(paths)
    if os.path.exists(file):
        cache = torch.load(file)

        # Load data only if paths didn't change
        if cache["paths"] == paths:
            return cache["images"]

    # torch.stack combines them into one (N, 3, H, W) tensor
    images = torch.stack([load_resized(p) for p in paths])

    torch.save({"paths": paths, "images": images}, file)
    return images


SEGMENTED_SUFFIX = "_final_masked.jpg"


@cache
def segmented_files(folder: str) -> tuple[set[str], dict[str, str]]:
    # The files of a segmented folder, and {photo name without UUID: file} for looking them up.
    files = os.listdir(folder) if os.path.isdir(folder) else []
    by_name = {f.split("___")[-1].removesuffix(SEGMENTED_SUFFIX): f for f in files}
    return set(files), by_name


def segmented_path(path: str) -> str:
    # Segmented counterpart of a colour photo '<uuid>___<name>.JPG' (the file may not exist).
    # Usually '<uuid>___<name>_final_masked.jpg', but the UUID prefix can be missing on either side:
    #   segmented file without it:    'Com.G_SpM_FL 1378_final_masked.jpg'
    #   colour photo without it:      'RS_Rust 1563.JPG' -> '<uuid>___RS_Rust 1563_final_masked.jpg'
    class_name, file_name = path.split(os.sep)[-2:]
    folder = os.path.join(PATHS_CONFIG["data_root"], "raw", "segmented", class_name)
    stem = os.path.splitext(file_name)[0]
    files, by_name = segmented_files(folder)
    if stem + SEGMENTED_SUFFIX in files:  # the same name
        file = stem + SEGMENTED_SUFFIX
    elif "___" in stem:  # segmented file without the UUID prefix
        file = stem.split("___")[-1] + SEGMENTED_SUFFIX
    else:  # colour photo without the UUID prefix
        file = by_name.get(stem, stem + SEGMENTED_SUFFIX)
    return os.path.join(folder, file)


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
