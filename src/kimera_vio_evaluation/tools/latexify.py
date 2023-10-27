"""Utilities for using latex in plots."""
# EXAMPLE TAKEN FROM: https://nipunbatra.github.io/blog/2014/latexify.html
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from math import sqrt


def latexify(fig_width=None, fig_height=None, columns=1):
    """
    Set up matplotlib's RC params for LaTeX plotting.

    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
    assert columns in [1, 2]
    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches
    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches
    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print(
            "WARNING: fig_height too large:"
            + fig_height
            + "so will reduce to"
            + MAX_HEIGHT_INCHES
            + "inches."
        )
        fig_height = MAX_HEIGHT_INCHES
    params = {
        "backend": "ps",
        "text.latex.preamble": [r"\usepackage{gensymb}"],
        "axes.labelsize": 8,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 8,
        #              'text.fontsize': 8, # was 10
        "legend.fontsize": 8,  # was 10
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }
    matplotlib.rcParams.update(params)


def format_axes(ax):
    """Set up axes ticks."""
    SPINE_COLOR = "gray"
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction="out", color=SPINE_COLOR)
    return ax


# Example of how to use latexify and the results of plotting before
# and after latexification.
if __name__ == "__main__":
    df = pd.DataFrame(np.random.randn(10, 2))
    df.columns = ["Column 1", "Column 2"]
    ax = df.plot()
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_title("Title")
    plt.tight_layout()
    plt.savefig("./image1.pdf")

    latexify()

    ax = df.plot()
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_title("Title")
    plt.tight_layout()
    format_axes(ax)
    plt.savefig("./image2.pdf")
