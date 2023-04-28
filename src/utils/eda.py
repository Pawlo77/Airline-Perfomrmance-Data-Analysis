import os

import matplotlib.pyplot as plt

PLOTS_DIR = "plots"


def save_fig(
    name: str, dir: str = os.path.join(PLOTS_DIR, "eda_Pawel"), **kwargs
) -> None:
    """
    Utility function to save matplotlib.pyplot plots to dir in .png format

    :param name: name of the plot
    :param dir: target directory
    :param kwargs: arguments that will be passed to matplotlib.pyplot.savefig()
    """
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, name + ".png"), **kwargs)
