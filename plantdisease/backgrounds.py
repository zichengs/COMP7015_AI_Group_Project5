"""Leaf masks and backgrounds: robustness variants (13), same-leaf background tests (13.2), countermeasure (14)."""

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.ndimage import binary_closing, binary_fill_holes
from torchvision.transforms import v2

from .data import CachedDataset, build_cache, segmented_path


def leaf_masks(segmented, threshold=10):
    # Leaf = every pixel that is not black in the segmented photo; closing and hole filling bridge small gaps.
    mask = (segmented.amax(dim=1) > threshold).numpy()
    mask = [binary_fill_holes(binary_closing(m, iterations=3)) for m in mask]
    return torch.from_numpy(np.stack(mask))


def trusted(masks):
    # A leaf mask can only be used if it contains leaf pixels at all (empty or missing masks: the photo stays unchanged)
    return masks.flatten(1).any(1).numpy()


def free_corner(image, mask, size=32):
    # The first image corner of size x size pixels that contains no leaf pixel (None if there is none).
    h, w = image.shape[-2:]
    for rows, cols in (
        (slice(0, size), slice(0, size)),
        (slice(0, size), slice(w - size, w)),
        (slice(h - size, h), slice(0, size)),
        (slice(h - size, h), slice(w - size, w)),
    ):
        if not mask[rows, cols].any():
            return image[:, rows, cols]
    return None


def background_tiles(images, masks):
    # A leaf-free corner tile of every photo (found is False if all four corners touch the leaf).
    tiles = [free_corner(i, m) for i, m in zip(images, masks, strict=True)]
    found = np.array([t is not None for t in tiles])
    empty = torch.zeros(3, 32, 32, dtype=torch.uint8)
    return torch.stack([empty if t is None else t for t in tiles]), found


def mirrored(tile, h, w):
    # Cover an h x w image with a background patch and its mirror images, so that no seams appear.
    block = torch.cat([tile, tile.flip(-1)], dim=-1)
    block = torch.cat([block, block.flip(-2)], dim=-2)
    size = block.shape[-1]
    return block.repeat(1, -(-h // size), -(-w // size))[:, :h, :w]


def stripes(n, h, w, seed=1):
    # Diagonal stripes in two random colours: a background type that never occurs in training.
    gen = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    images = []
    for _ in range(n):
        period = int(torch.randint(12, 40, (1,), generator=gen))
        colours = torch.randint(0, 256, (2, 3, 1, 1), generator=gen).to(torch.uint8)
        images.append(
            torch.where(((xx + yy) // period % 2).bool(), colours[0], colours[1])
        )
    return torch.stack(images)


# ----------------------------------------------------------------------------
# Section 13: the same test leaves under several conditions
# ----------------------------------------------------------------------------
@dataclass
class TestVariants:
    # Test photos that have a segmented version, under every test condition (all aligned).
    variants: dict  # condition -> uint8 images
    y: np.ndarray  # labels
    groups: np.ndarray  # leaf groups (for the bootstrap)
    has_seg: np.ndarray  # which test photos (of data.test_idx) are used
    background_tests: dict = field(default_factory=dict)  # Section 13.2
    donor_class: dict = field(default_factory=dict)  # class of the background photo
    masks: torch.Tensor = None  # leaf masks (N, H, W)
    keep: torch.Tensor = (
        None  # pixels kept from the photo (the leaf, or all if the mask is empty)
    )

    @property
    def original(self):
        return self.variants["original"]

    @property
    def all_tests(self):
        # every test version of the same leaves
        return {**self.variants, **self.background_tests}


def make_test_variants(data, verbose=True):
    # Section 13: original, background removed, grayscale, darker, blur and noise versions of the test leaves.
    seg_paths = [segmented_path(p) for p in data.df.loc[data.test_idx, "path"]]
    has_seg = np.array([os.path.exists(p) for p in seg_paths])
    if verbose:
        print(
            f"Test images with a segmented version: {has_seg.sum()} / {len(data.test_idx)}"
        )
    seg_images = build_cache(
        [p for p, h in zip(seg_paths, has_seg, strict=True) if h], "segmented_test"
    )
    orig = data.images[data.test_idx[has_seg]]

    gen = torch.Generator().manual_seed(0)
    variants = {
        "original": orig,
        "background removed": seg_images,
        "grayscale": v2.functional.rgb_to_grayscale(orig, num_output_channels=3),
        "darker (x0.6)": (orig.float() * 0.6).round().to(torch.uint8),
        "blur (sigma=2)": v2.functional.gaussian_blur(orig, kernel_size=13, sigma=2.0),
        "noise (sigma=20)": (orig.float() + 20 * torch.randn(orig.shape, generator=gen))
        .clamp(0, 255)
        .to(torch.uint8),
    }
    return TestVariants(
        variants=variants,
        y=data.y_test[has_seg],
        groups=data.test_groups[has_seg],
        has_seg=has_seg,
    )


def add_background_tests(tv, verbose=True):
    # Section 13.2: the leaf of every test photo on a background tile of a photo of the same / another class.
    tv.masks = leaf_masks(tv.variants["background removed"])
    has_leaf = tv.masks.flatten(1).any(1)
    tv.keep = (
        tv.masks | ~has_leaf[:, None, None]
    )  # a photo with an empty mask stays unchanged
    if verbose:
        print(
            f"{(~has_leaf).sum()} of {len(has_leaf)} test photos have an empty mask and stay unchanged"
        )

    orig, y = tv.original, tv.y
    tiles, tile_ok = background_tiles(orig, tv.masks)
    # a donor needs a leaf mask, otherwise its leaf is not marked
    tile_ok = tile_ok & has_leaf.numpy()
    donor_rng = np.random.default_rng(0)
    usable = np.flatnonzero(tile_ok)
    same_donor, other_donor = [], []
    for i in range(len(y)):
        same_pool = usable[(y[usable] == y[i]) & (usable != i)]
        other_pool = usable[y[usable] != y[i]]
        assert len(same_pool), "no leaf-free background tile found"
        assert len(other_pool), "no leaf-free background tile found"
        same_donor.append(donor_rng.choice(same_pool))
        other_donor.append(donor_rng.choice(other_pool))
    if verbose:
        print(f"{tile_ok.sum()} of {len(tile_ok)} test photos have a leaf-free corner")

    h, w = orig.shape[-2:]

    def leaf_on_tile(donors):
        # The leaf of every test photo on the (mirrored) background tile of its donor photo.
        return torch.stack(
            [
                torch.where(tv.keep[i], orig[i], mirrored(tiles[j], h, w))
                for i, j in enumerate(donors)
            ]
        )

    tv.background_tests = {
        "same-class background": leaf_on_tile(same_donor),
        "other-class background": leaf_on_tile(other_donor),
    }
    tv.donor_class = {
        "same-class background": y[same_donor],
        "other-class background": y[other_donor],
    }
    return tv


# ----------------------------------------------------------------------------
# Section 14: training with random backgrounds
# ----------------------------------------------------------------------------
def training_masks(data):
    # Leaf masks of all photos (aligned with data.images) and which of them can be used for the swap.
    seg_paths_all = [segmented_path(p) for p in data.df["path"]]
    has_segmented = np.array([os.path.exists(p) for p in seg_paths_all])
    print(
        f"{(~has_segmented).sum()} of {len(has_segmented)} photos have no segmented version"
    )
    # photos without a segmented version get an empty mask and stay unchanged
    seg_all = torch.zeros_like(data.images)
    seg_all[torch.from_numpy(has_segmented)] = build_cache(
        [p for p, h in zip(seg_paths_all, has_segmented, strict=True) if h], "segmented"
    )
    masks = leaf_masks(seg_all)  # (N, H, W) bool
    swap_ok = trusted(masks)
    print(
        f"Leaf masks cover {masks.float().mean():.0%} of the image on average | "
        f"{(~swap_ok).sum()} of {len(swap_ok)} photos have an empty or missing mask "
        f"and are never changed"
    )
    return masks, swap_ok


def corner_background(shape, pool, pool_indices, masks, swap_ok):
    # A leaf-free corner tile of a random photo of the pool, repeated over the image (black if none is found).
    _, h, w = shape
    for _ in range(10):
        j = pool_indices[random.randrange(len(pool_indices))]  # noqa: S311
        tile = free_corner(pool[j], masks[j]) if swap_ok[j] else None
        if tile is not None:
            return mirrored(tile, h, w)
    return torch.zeros(3, h, w, dtype=torch.uint8)


def random_background(shape, pool, pool_indices, masks, swap_ok):
    # One random background (3, H, W): black, solid colour, colour gradient, smooth noise or a leaf-free corner tile of another training photo.
    _, h, w = shape
    kind = random.randrange(5)  # noqa: S311
    if kind == 0:
        return torch.zeros(3, h, w, dtype=torch.uint8)
    if kind == 1:
        return torch.randint(0, 256, (3, 1, 1), dtype=torch.uint8).expand(3, h, w)
    if kind == 2:
        ramp = torch.linspace(0, 1, w).view(1, 1, w)
        if random.random() < 0.5:  # noqa: S311
            ramp = torch.linspace(0, 1, h).view(1, h, 1)
        start, end = torch.rand(3, 1, 1), torch.rand(3, 1, 1)
        return (
            ((start + (end - start) * ramp) * 255)
            .expand(3, h, w)
            .round()
            .to(torch.uint8)
        )
    if kind == 3:
        smooth = F.interpolate(torch.rand(1, 3, 6, 6), size=(h, w), mode="bilinear")
        return (smooth[0] * 255).round().to(torch.uint8)
    return corner_background(shape, pool, pool_indices, masks, swap_ok)


class BackgroundSwapDataset(CachedDataset):
    # Like CachedDataset, but with probability p everything outside the leaf is replaced by a background:
    # a random one, or (class_matched, the control) a corner tile of a training photo of the same class.
    def __init__(
        self,
        images,
        labels,
        indices,
        transform,
        *,
        p,
        masks,
        swap_ok,
        class_matched=False,
    ):
        super().__init__(images, labels, indices, transform)
        self.p, self.class_matched = p, class_matched
        self.masks, self.swap_ok = masks, swap_ok
        classes = labels[indices].numpy()
        self.pools = {c: indices[classes == c] for c in np.unique(classes)}

    def __getitem__(self, i):
        j = self.indices[i]
        image = self.images[j]
        if self.swap_ok[j] and random.random() < self.p:  # noqa: S311
            if self.class_matched:
                pool = self.pools[int(self.labels[j])]
                background = corner_background(
                    image.shape, self.images, pool, self.masks, self.swap_ok
                )
            else:
                background = random_background(
                    image.shape, self.images, self.indices, self.masks, self.swap_ok
                )
            image = torch.where(self.masks[j], image, background)
        return self.transform(image), self.labels[j]
