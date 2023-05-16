import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .constants import PLOTS_DIR


def save_fig(name: str, dir: str = PLOTS_DIR, **kwargs) -> None:
    """
    Utility function to save matplotlib.pyplot plots to dir in .png format

    :param name: name of the plot
    :param dir: target directory
    :param kwargs: arguments that will be passed to matplotlib.pyplot.savefig()
    """
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, name + ".png"), **kwargs)


def finish(
    ax: plt.Axes,
    title: str,
    title_size: int = 20,
    xaxis_size: int = 13,
    yaxis_size: int = 13,
    xticks_size: int = 9,
    yticks_size: int = 9,
    plot: bool = True,
    dir: str = None,
) -> None:
    """
    Pretty finishes matplotlib plot, saves it and opens the plot.
    :param ax: The plot object
    :param title: The title to be set (also filename)
    :param title_size: Title font size
    :param xaxis_size: X axis font size
    :param yaxis_size: Y axis font size
    :param xticks_size: X ticks labels font size
    :param yticks_size: Y ticks labels font size
    :param plot: weather to call plt.show()
    :param dir: directory to save the plot to
    """
    sns.despine()
    ax.xaxis.label.set_size(xaxis_size)
    ax.yaxis.label.set_size(yaxis_size)
    ax.set_yticklabels(ax.get_yticks().astype(np.int64), size=xticks_size)
    ax.set_xticklabels(ax.get_xticklabels(), size=yticks_size)
    plt.title(title, size=title_size)
    if ax.get_legend() is not None:
        # keep legend in plot
        save_fig(
            title, dir=dir, bbox_extra_artists=(ax.get_legend(),), bbox_inches="tight"
        )
    else:
        save_fig(title, dir=dir)
    if plot:
        plt.show()
