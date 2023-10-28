"""Show timing plots."""
import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import os


MAX_HEIGHT_INCHES = 8.0


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
    # TAKEN FROM: https://nipunbatra.github.io/blog/2014/latexify.html
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples
    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf
    assert columns in [1, 2]
    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

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
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 8,  # was 10
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def draw_timing_plot(
    filename,
    keyframe_ids,
    pipelines_times,
    ylabel="Optimization time [s]",
    display_plot=False,
    display_x_label=True,
    should_latexify=True,
    fig_width=6,
    fig_height=3,
):
    """
    Draw timing.

    Plots timing information for each pipeline contained in the list of dicts
    pipelines_times:
    - filename: where to save the figure.
    - pipelines_times: list of dicts of the form:
    [{
        'pipeline_name': pipeline_name,
        'line_color': np.random.rand(3),
        'line_style': '-',
        'times': update_times
    }, ... ]
    - keyframe_ids: corresponds to the x ticks, so update_times and keyframe_ids must
    be the same length.
    - ylabel: Y label used for the plot.
    - display_plot: whether to display the plot or not.
    - display_x_label: whether to display the x label of the plot or not.
    - latexify: whether to use latex for the generation of the plot.
    """
    if should_latexify:
        latexify(fig_width, fig_height)

    plt.figure(figsize=[fig_width, fig_height], dpi=1000)
    i = 0
    for pipeline_time in pipelines_times:
        assert len(keyframe_ids) == len(pipeline_time["times"])
        plt.plot(
            keyframe_ids,
            pipeline_time["times"],
            linestyle=pipeline_time["line_style"],
            color=pipeline_time["line_color"],
            linewidth=0.5,
            label="$t_{" + pipeline_time["pipeline_name"] + "}^{opt}$",
        )
        i = i + 1
    plt.ylabel(ylabel)
    if display_x_label:
        plt.xlabel("Keyframe Index [-]")
    plt.xlim(min(keyframe_ids), max(keyframe_ids))
    plt.ylim(bottom=0)
    plt.grid(axis="both", linestyle="--")
    plt.legend()

    # Create path to filename if it does not exist.
    filepath = pathlib.Path(filename)
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)

    plt.savefig(filename, bbox_inches="tight", transparent=True, dpi=1000)
    if display_plot:
        plt.show()


def get_pipeline_times(results_path, pipeline_names, column):
    """
    Load pipeline times.

    Args:
        results_path: path to VIO pipeline results
        pipeline_names: are the names of the pipelines for which results are available
        column: Specifies the column in the CSV file containing timing info

    Returns: keyframe indices and list of timing info
    """
    times = []
    keyframes = None
    prev_name = None
    for color_idx, name in enumerate(sorted(pipeline_names)):
        timing_path = results_path / name / "output" / "output_timingVIO.txt"
        if not timing_path.exists():
            print(f"WARNING: skipping missing {name} at '{timing_path}'... ")
            continue

        print("Parsing results for pipeline: {name} in file: '{timing_path}'")
        with timing_path.open("r") as fin:
            ret = np.loadtxt(fin, delimiter=" ", usecols=(0, column), unpack=True)

        times.append(
            {
                "pipeline_name": name,
                "line_color": color_idx,
                "line_style": "-",
                "times": ret[1],
            }
        )

        if keyframes is None:
            keyframes = ret[0]
        elif len(keyframes) == len(ret[0]):
            raise Exception(f"keyframe ids do not match between {name} and {prev_name}")

        prev_name = name

    return keyframes, times


@click.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("-p", "--pipeline-names", multiple=True, default=["S", "SP", "SPR"])
@click.option("-t", "--time-column", default=3, type=int, help="time column to display")
def main(results_path, pipeline_names, time_column):
    """
    Display timing of VIO stored in 'results_folder' parameter as a path.

    In particular 'results_folder' should point to the following filesystem structure:
        results_path/pipeline_names[1]/output/output_timingVIO.txt
        results_path/pipeline_names[2]/output/output_timingVIO.txt etc
        results_path/pipeline_names[etc]/output/output_timingVIO.txt etc

    This function will warn if there are not results available for the given pipeline.
    """
    results_path = pathlib.Path(results_path).expanduser().absolute()
    keyframes, times = get_pipeline_times(results_path, pipeline_names, time_column)
    if not len(keyframes):
        click.secho("Missing keyframe data", fg="red")
        sys.exit(os.EX_NOINPUT)

    if not len(times) > 0:
        click.secho("Missing pipeline timing information", fg="red")
        sys.exit(os.EX_NOINPUT)

    final_plot_path = results_path / "timing" / "pipeline_times.pdf"
    click.secho(f"saving timing plot to '{final_plot_path}'")
    draw_timing_plot(final_plot_path, keyframes, times)
    sys.exit(os.EX_OK)
