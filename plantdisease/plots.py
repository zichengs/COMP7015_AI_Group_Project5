import math
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from pydantic import BaseModel

from .config import SHORT_NAMES, out_path
from .evaluate import sd


def grouped_bars(
    ax: Axes,
    means: pd.DataFrame,
    *,
    yerr: pd.DataFrame | None = None,
    seeds: pd.DataFrame | None = None,
    jitter: bool = False,
    dot_size: float = 8,
    **bar_kw: Any,
) -> None:
    # One group of bars per row of `means` and one bar per column. Optional, in the same layout:
    # `yerr` (error bars) and `seeds` (the scores of the seeds, drawn as black dots; it may hold
    # only some of the columns).
    x_pos, width = np.arange(len(means)), 0.8 / len(means.columns)
    for k, label in enumerate(means.columns):
        centre = x_pos + (k - (len(means.columns) - 1) / 2) * width
        ax.bar(
            centre,
            means[label],
            width,
            yerr=None if yerr is None else yerr[label],
            capsize=0 if yerr is None else 3,
            label=label,
            **bar_kw,
        )
        if seeds is None or label not in seeds:
            continue
        for x, scores in zip(centre, seeds[label], strict=True):
            offsets = np.linspace(-width / 4, width / 4, len(scores)) if jitter else 0
            xs = np.full(len(scores), x) + offsets
            ax.scatter(xs, scores, color="black", s=dot_size, zorder=3)
    ax.set_xticks(x_pos, means.index)


def seed_bars(
    ax: Axes, scores: pd.DataFrame, *, errorbars: bool = True, **kw: Any
) -> None:
    # grouped_bars for a table whose cells are the scores of the seeds: bars = mean over the
    # seeds, error bars = sd, dots = seeds. Rows are the bar groups, columns the bars of a group.
    grouped_bars(
        ax,
        scores.map(np.mean),
        yerr=scores.map(sd) if errorbars else None,
        seeds=scores,
        **kw,
    )


def mean_curve(
    histories: Sequence[BaseModel], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Learning curve over the seeds -> (epochs, mean, sd, values). `key` is a per-epoch list of
    # RunHistory, e.g. 'val_f1'. values: (seeds, epochs), NaN after a seed stopped early; mean and
    # sample sd ignore the NaN (sd 0 where only one seed is left).
    curves = [getattr(h, key) for h in histories]
    n_epochs = max(len(c) for c in curves)
    values = np.array([c + [np.nan] * (n_epochs - len(c)) for c in curves])
    n = np.sum(~np.isnan(values), 0)
    mean = np.nanmean(values, 0)
    spread = np.sqrt(np.nansum((values - mean) ** 2, 0) / np.maximum(n - 1, 1))
    return np.arange(1, n_epochs + 1), mean, spread, values


def show_batch(x: torch.Tensor, y: torch.Tensor, title: str, ncols: int = 8) -> None:
    # A batch of (normalised) training images in a grid, each titled with its class. The batch is
    # scaled to [0, 1] as a whole, so brightness differences between the images stay visible.
    images = (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)
    nrows = math.ceil(len(x) / ncols)
    # constrained layout keeps every row of titles clear of the images above it
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.6 * ncols, 1.9 * nrows + 0.4),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, label in zip(axes.flat, images, y, strict=False):
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(SHORT_NAMES[label], fontsize=8)
    fig.suptitle(title)
    plt.show()


def finish(file_name: str | None = None) -> None:
    # tight layout, save to the output folder (if a file name is given) and show.
    plt.tight_layout()
    if file_name:
        plt.savefig(out_path(file_name), dpi=150)
    plt.show()
