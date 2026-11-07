"""Small plotting helpers shared by the notebooks."""

import matplotlib.pyplot as plt
import numpy as np

from .config import out_path


def grouped_bars(
    ax,
    xlabels,
    series,
    means,
    *,
    seeds=None,
    yerr=None,
    jitter=False,
    dot_size=8,
    **bar_kw,
):
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


def finish(file_name=None):
    # tight layout, save to OUT_DIR (if a file name is given) and show.
    plt.tight_layout()
    if file_name:
        plt.savefig(out_path(file_name), dpi=150)
    plt.show()
