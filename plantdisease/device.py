"""Device, mixed precision, determinism and seeding (Section 1)."""

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # required for deterministic CUDA matmuls (set before importing torch)

import random

import numpy as np
import torch

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(
    True, warn_only=True
)  # warn (not fail) for the few ops without a deterministic version

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
USE_AMP = (
    device.type == "cuda"
)  # mixed precision: large speed-up on NVIDIA GPUs (no gain on Apple MPS)


def set_seed(seed: int) -> None:
    # Fix all random number generators so that runs are reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize() -> None:
    # Wait for the GPU (CUDA or Apple MPS) to finish, so that timings are correct.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
