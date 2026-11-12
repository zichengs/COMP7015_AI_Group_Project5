import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms import v2

from .dataset import Data
from .images import has_leaf, leaf_masks, to_grayscale
from .loaders import CachedDataset

TILE_SIZE = 32  # side of a leaf-free corner tile, in pixels


# ----------------------------------------------------------------------------
# Background tiles (the leaf masks are in images.py)
# ----------------------------------------------------------------------------
def free_corner(
    image: torch.Tensor, mask: torch.Tensor, size: int = TILE_SIZE
) -> torch.Tensor | None:
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


def background_tiles(
    images: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, np.ndarray]:
    # A leaf-free corner tile of every photo (found is False if all four corners touch the leaf).
    tiles = [free_corner(i, m) for i, m in zip(images, masks, strict=True)]
    found = np.array([t is not None for t in tiles])
    empty = torch.zeros(3, TILE_SIZE, TILE_SIZE, dtype=torch.uint8)
    return torch.stack([empty if t is None else t for t in tiles]), found


def mirrored(tile: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # Cover an h x w image with a background patch and its mirror images, so that no seams appear.
    block = torch.cat([tile, tile.flip(-1)], dim=-1)
    block = torch.cat([block, block.flip(-2)], dim=-2)
    size = block.shape[-1]
    return block.repeat(1, -(-h // size), -(-w // size))[:, :h, :w]


def stripes(n: int, h: int, w: int, seed: int = 1) -> torch.Tensor:
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
# Section 16: the same test leaves under several conditions
# ----------------------------------------------------------------------------
@dataclass
class TestVariants:
    # Test photos that have a segmented version, under every test condition (all aligned).
    variants: dict[str, torch.Tensor]  # condition -> uint8 images
    y: np.ndarray  # labels
    has_seg: np.ndarray  # which test photos (of data.test_idx) are used

    @property
    def original(self) -> torch.Tensor:
        return self.variants["original"]


def image_changes(images: torch.Tensor) -> dict[str, torch.Tensor]:
    # Grayscale, darker, blurred and noisy versions of uint8 photos (Section 16).
    gen = torch.Generator().manual_seed(0)
    noisy = images.float() + 20 * torch.randn(images.shape, generator=gen)
    return {
        "grayscale": to_grayscale(images),
        "darker (x0.6)": (images.float() * 0.6).round().to(torch.uint8),
        "blur (sigma=2)": v2.functional.gaussian_blur(
            images, kernel_size=13, sigma=2.0
        ),
        "noise (sigma=20)": noisy.clamp(0, 255).to(torch.uint8),
    }


def make_test_variants(data: Data, verbose: bool = True) -> TestVariants:
    # Section 16: original, background removed, grayscale, darker, blur and noise versions of the test leaves.
    seg_images, exists = data.segmented
    has_seg = exists[data.test_idx]
    if verbose:
        print(
            f"Test images with a segmented version: {has_seg.sum()} / {len(data.test_idx)}"
        )
    used = data.test_idx[has_seg]
    orig = data.images[used]
    variants = {
        "original": orig,
        "background removed": seg_images[used],
        **image_changes(orig),
    }
    return TestVariants(variants=variants, y=data.y_test[has_seg], has_seg=has_seg)


# ----------------------------------------------------------------------------
# Section 18: the same leaf on another background
# ----------------------------------------------------------------------------
@dataclass
class BackgroundTests:
    # The test leaves of a TestVariants on other backgrounds (aligned with it).
    # leaf masks (N, H, W)
    masks: torch.Tensor
    # pixels kept from the photo: the leaf, or all of them if the mask is empty
    keep: torch.Tensor
    # 'same-class background' / 'other-class background' -> images
    tests: dict[str, torch.Tensor]
    # test -> class of the photo that the background comes from
    donor_class: dict[str, np.ndarray]
    # the leaves on diagonal stripes, a background that never occurs in training
    stripes: torch.Tensor


def make_background_tests(tv: TestVariants, verbose: bool = True) -> BackgroundTests:
    # The leaf of every test photo on a background tile of another test photo ("donor") of the same
    # class and of another class, and on stripes.
    masks = leaf_masks(tv.variants["background removed"])
    with_leaf = has_leaf(masks)
    keep = masks | ~torch.from_numpy(with_leaf)[:, None, None]
    if verbose:
        print(
            f"{(~with_leaf).sum()} of {len(with_leaf)} test photos have an empty mask and stay unchanged"
        )

    orig, y = tv.original, tv.y
    tiles, tile_ok = background_tiles(orig, masks)
    # a donor needs a leaf mask, otherwise its leaf is not marked
    tile_ok &= with_leaf
    usable = np.flatnonzero(tile_ok)
    donor_rng = np.random.default_rng(0)
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

    def leaf_on_tile(donors: list[int]) -> torch.Tensor:
        # The leaf of every test photo on the (mirrored) background tile of its donor photo.
        return torch.stack(
            [
                torch.where(keep[i], orig[i], mirrored(tiles[j], h, w))
                for i, j in enumerate(donors)
            ]
        )

    return BackgroundTests(
        masks=masks,
        keep=keep,
        tests={
            "same-class background": leaf_on_tile(same_donor),
            "other-class background": leaf_on_tile(other_donor),
        },
        donor_class={
            "same-class background": y[same_donor],
            "other-class background": y[other_donor],
        },
        stripes=torch.where(keep[:, None], orig, stripes(len(orig), h, w)),
    )


# ----------------------------------------------------------------------------
# Section 20: training with random backgrounds
# ----------------------------------------------------------------------------
class BackgroundSwapDataset(CachedDataset):
    # Like CachedDataset, but with probability p everything outside the leaf is replaced by a background:
    # a random one, or (class_matched, the control) a corner tile of a training photo of the same class.
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
        transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        p: float,
        masks: torch.Tensor,
        swap_ok: np.ndarray,
        class_matched: bool = False,
    ) -> None:
        super().__init__(images, labels, indices, transform)
        self.p = p
        self.class_matched = class_matched
        self.masks = masks
        self.swap_ok = swap_ok
        classes = labels[indices].numpy()
        # training photos of each class, for the class-matched control
        self.pools = {c: indices[classes == c] for c in np.unique(classes)}

    def corner_background(self, h: int, w: int, pool: np.ndarray) -> torch.Tensor:
        # A leaf-free corner tile of a random photo of the pool, repeated over the image (black if none is found).
        for _ in range(10):
            j = pool[random.randrange(len(pool))]  # noqa: S311
            tile = (
                free_corner(self.images[j], self.masks[j]) if self.swap_ok[j] else None
            )
            if tile is not None:
                return mirrored(tile, h, w)
        return torch.zeros(3, h, w, dtype=torch.uint8)

    def random_background(self, h: int, w: int) -> torch.Tensor:
        # One random background (3, H, W): black, solid colour, colour gradient, smooth noise or a
        # leaf-free corner tile of another training photo.
        kind = random.randrange(5)  # noqa: S311
        if kind == 0:  # black
            return torch.zeros(3, h, w, dtype=torch.uint8)
        if kind == 1:  # solid colour
            return torch.randint(0, 256, (3, 1, 1), dtype=torch.uint8).expand(3, h, w)
        if kind == 2:  # horizontal or vertical colour gradient
            ramp = torch.linspace(0, 1, w).view(1, 1, w)
            if random.random() < 0.5:  # noqa: S311
                ramp = torch.linspace(0, 1, h).view(1, h, 1)
            start, end = torch.rand(3, 1, 1), torch.rand(3, 1, 1)
            gradient = (start + (end - start) * ramp) * 255
            return gradient.expand(3, h, w).round().to(torch.uint8)
        if kind == 3:  # smooth noise
            smooth = F.interpolate(torch.rand(1, 3, 6, 6), size=(h, w), mode="bilinear")
            return (smooth[0] * 255).round().to(torch.uint8)
        return self.corner_background(h, w, self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = self.indices[i]
        image = self.images[j]
        if self.swap_ok[j] and random.random() < self.p:  # noqa: S311
            h, w = image.shape[-2:]
            if self.class_matched:
                pool = self.pools[int(self.labels[j])]
                background = self.corner_background(h, w, pool)
            else:
                background = self.random_background(h, w)
            image = torch.where(self.masks[j], image, background)
        return self.transform(image), self.labels[j]
