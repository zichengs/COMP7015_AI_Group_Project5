"""Small plotting helpers shared by the notebooks."""

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from pydantic import BaseModel

from .config import out_path


def grouped_bars(
    ax: Axes,
    xlabels: Sequence[str],
    series: Sequence[str],
    means: Sequence[npt.ArrayLike],
    *,
    seeds: Sequence[Any] | None = None,
    yerr: Sequence[npt.ArrayLike] | None = None,
    jitter: bool = False,
    dot_size: float = 8,
    **bar_kw: Any,
) -> None:
    # One group of bars per x label and one bar per series; optional error bars and black dots for the seeds.
    # means[k][j] / yerr[k] / seeds[k][j]: series k at x label j (seeds[k] may be None: no dots for that series).
    x_pos, width = np.arange(len(xlabels)), 0.8 / len(series)
    for k, label in enumerate(series):
        centre = x_pos + (k - (len(series) - 1) / 2) * width
        err = None if yerr is None else yerr[k]
        ax.bar(
            centre,
            means[k],
            width,
            yerr=err,
            capsize=3 if err is not None else 0,
            label=label,
            **bar_kw,
        )
        if seeds is None or seeds[k] is None:
            continue
        for j, scores in enumerate(seeds[k]):
            offsets = np.linspace(-width / 4, width / 4, len(scores)) if jitter else 0
            ax.scatter(
                np.full(len(scores), centre[j]) + offsets,
                scores,
                color="black",
                s=dot_size,
                zorder=3,
            )
    ax.set_xticks(x_pos, xlabels)


def stack_curves(histories: Sequence[BaseModel], key: str) -> np.ndarray:
    # Per-epoch values of several run histories (RunHistory) as one (runs, epochs) array; NaN after a
    # run stopped early. `key` is the name of a per-epoch list, e.g. 'val_f1'.
    curves = [getattr(h, key) for h in histories]
    n = max(len(c) for c in curves)
    return np.array([c + [np.nan] * (n - len(c)) for c in curves])


def finish(file_name: str | None = None) -> None:
    # tight layout, save to the output folder (if a file name is given) and show.
    plt.tight_layout()
    if file_name:
        plt.savefig(out_path(file_name), dpi=150)
    plt.show()
