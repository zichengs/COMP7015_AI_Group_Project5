"""Preprocessing, augmentation and data loaders (Section 6)."""

import random
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from ..config import DATA_CONFIG
from ..device import device

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
        num_workers=DATA_CONFIG["num_workers"],
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        pin_memory=device.type == "cuda",
        persistent_workers=DATA_CONFIG["num_workers"] > 0,
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
