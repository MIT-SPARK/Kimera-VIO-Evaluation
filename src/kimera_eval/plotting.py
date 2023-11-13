"""Plot information via plotly."""
from kimera_eval.trajectory_metrics import TrajectoryResults

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import pathlib

import evo.tools.plot
import evo.core.metrics
import copy

import math

import logging
import pandas as pd
from datetime import datetime as date


DEFAULT_STAT_TYPES = [
    evo.core.metrics.StatisticsType.rmse,
    evo.core.metrics.StatisticsType.mean,
    evo.core.metrics.StatisticsType.median,
    evo.core.metrics.StatisticsType.std,
    evo.core.metrics.StatisticsType.min,
    evo.core.metrics.StatisticsType.max,
]


def draw_feature_tracking_stats(df, show_figure=False):
    """
    Draw a plotly bar plot from the csv file of the frontend.

    Args:
    - df: pandas dataframe with frontend stats
    - show_figure: optional param to show figure

    Returns:
    - fig: a plotly figure handle
    """
    colors = [
        "green" if mono_status == "VALID" else "crimson"
        for mono_status in df["mono_status"]
    ]
    x = df.index
    fig = go.Figure(
        data=[
            go.Bar(name="Keypoints Detected", x=x, y=df["nr_keypoints"]),
            go.Bar(name="Tracked Features", x=x, y=df["nrTrackerFeatures"]),
            go.Bar(
                name="Monocular Inliers (colored by Status)",
                x=x,
                y=df["nrMonoInliers"],
                marker_color=colors,
                hovertext=df["mono_status"],
                hovertemplate="Mono Inliers: %{y} <br>Mono Status: %{hovertext}",
            ),
        ]
    )
    fig.update_layout(barmode="overlay", template="plotly_white")

    if show_figure:
        fig.show()

    return fig


def draw_mono_stereo_inliers_outliers(df, show_figure=False):
    """
    Draw a plotly bar plot from the csv file of the frontend.

    showing the rate of inliers/outliers for the mono/stereo tracks.

    Args:
        - df: pandas dataframe with frontend stats
        - show_figure: optional param to show figure

    Returns:
        - fig: a plotly figure handle
    """
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    x = df.index
    fig.add_trace(
        go.Bar(name="# Mono Putatives", x=x, y=df["nrMonoPutatives"]), row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="# Mono Inliers", x=x, y=df["nrMonoInliers"]), row=1, col=1
    )

    fig.add_trace(
        go.Bar(name="# Stereo Putatives", x=x, y=df["nrStereoPutatives"]), row=2, col=1
    )
    fig.add_trace(
        go.Bar(name="# Stereo Inliers", x=x, y=df["nrStereoInliers"]), row=2, col=1
    )

    fig.add_trace(
        go.Bar(name="Mono RANSAC Iterations", x=x, y=df["monoRansacIters"]),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Bar(name="Stereo RANSAC Iterations", x=x, y=df["stereoRansacIters"]),
        row=4,
        col=1,
    )

    fig.update_layout(barmode="overlay")

    if show_figure:
        fig.show()

    return fig


def draw_frontend_timing(df, show_figure=False):
    """
    Draws a plotly bar plot from the csv file of the frontend.

    plot timing from frontend.

    Args:
        - df: pandas dataframe with frontend stats
        - show_figure: optional param to show figure

    Returns:
        - fig: a plotly figure handle
    """
    fig = go.Figure()
    x = df.index
    fig.add_trace(
        go.Bar(name="Feature Detection Time", x=x, y=df["featureDetectionTime"])
    )
    fig.add_trace(
        go.Bar(name="Feature Tracking Time", x=x, y=df["featureTrackingTime"])
    )
    fig.add_trace(go.Bar(name="Mono RANSAC Time", x=x, y=df["monoRansacTime"]))
    fig.add_trace(go.Bar(name="Stereo RANSAC Time", x=x, y=df["stereoRansacTime"]))
    fig.add_trace(
        go.Bar(name="Feature Selection Time", x=x, y=df["featureSelectionTime"])
    )
    fig.update_layout(barmode="stack", title_text="Frontend Timing")

    if show_figure:
        fig.show()

    return fig


def draw_boxplot(df):
    """Draw boxplot."""
    tidy = df.set_index(["Dataset Name"])
    tidy = (
        tidy["ATE errors"]
        .apply(lambda x: pd.Series(x))
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("ATE errors")
    )
    tidy.reset_index(level=["Dataset Name"], drop=False, inplace=True)
    tidy.sort_values("Dataset Name", inplace=True)
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=tidy["Dataset Name"], y=tidy["ATE errors"], boxpoints="all", boxmean=True
        )
    )

    fig.update_layout(
        title=go.layout.Title(text="Kimera-VIO ATE Euroc dataset " + str(date.today())),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Datasets")),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="ATE [m]"), rangemode="tozero"
        ),
        template="plotly_white",
    )
    return fig


def draw_ape_boxplots(stats, show_figure=False):
    """
    Draw boxplots.

    Simplified boxplot plotting using plotly for APE boxplots.
    See draw_ape_boxplots for the complicated version.

    Args:
        - stats: vio statistics (see 'draw_ape_boxplots' function)
        (to publish online, you need to follow the instructions here: )
        If False, it will just show the boxplot figure.
        - show_figure: whether to display the figure or not
    Returns:
        - the handle to the plotly figure
    """

    def listify_stats(stats):
        """Make a list of lists out of the stats."""
        stats_list = []
        for dataset_name in stats:
            for pipeline in stats[dataset_name]:
                result = stats[dataset_name][pipeline]
                if result is not False:
                    result = result["absolute_errors"].np_arrays["error_array"]
                    stats_list.append([dataset_name, pipeline, result])
        return stats_list

    df = pd.DataFrame()
    logging.info("Creating dataframe stats.")
    df = pd.DataFrame.from_records(listify_stats(stats))
    df.columns = ["Dataset Name", "Pipe Type", "ATE errors"]

    figure = draw_boxplot(df)

    if show_figure:
        figure.show()
    return figure


def draw_timing_plot(
    filename,
    keyframe_ids,
    pipelines_times,
    ylabel="Optimization time [s]",
    display_plot=False,
    display_x_label=True,
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


def plot_multi_line(self, df, x_id, y_ids, fig=None, row=None, col=None):
    """Plot DF rows in a line plot."""
    mode = "lines+markers"

    if fig is None:
        fig = go.Figure()

    # TODO(nathan) no asserts
    assert x_id in df
    for y_id in y_ids:
        assert y_id in df
        fig.add_trace(
            go.Scatter(x=df[x_id], y=df[y_id], mode=mode, name=y_id), row=row, col=col
        )

    return fig


def plot_3d_trajectory(df, fig=None, row=None, col=None):
    """Show a 3D trajectory."""
    args = {
        "size": 5,
        "color": df["#timestamp"],
        "colorscale": "Viridis",
        "opacity": 0.8,
    }

    trace = go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"], mode="lines+markers", marker=args
    )
    fig.add_trace(trace, row=row, col=col)
    # TODO(nathan) this is ugly
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    showarrow=False,
                    x=df["x"][0],
                    y=df["y"][0],
                    z=df["z"][0],
                    text="Start",
                    xanchor="left",
                    xshift=10,
                    opacity=0.9,
                ),
                dict(
                    showarrow=False,
                    x=df["x"].iloc[-1],
                    y=df["y"].iloc[-1],
                    z=df["z"].iloc[-1],
                    text="End",
                    xanchor="left",
                    xshift=10,
                    opacity=0.9,
                ),
            ],
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        )
    )


def plot_metric(metric, plot_title="", figsize=(8, 8), stat_types=None):
    """
    Add a metric plot to a plot collection.

    Args:
        plot_collection: a PlotCollection containing plots.
        metric: an evo.core.metric object with statistics and information.
        plot_title: a string representing the title of the plot.
        figsize: a 2-tuple representing the figure size.

    Returns:
        A plt figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    stats_to_use = DEFAULT_STAT_TYPES if stat_types is None else stat_types
    stats = {s.value: metric.get_statistic(s) for s in stats_to_use}

    evo.tools.plot.error_array(
        ax,
        metric.error,
        statistics=stats,
        title=plot_title,
        xlabel="Keyframe index [-]",
        ylabel=plot_title + " " + metric.unit.value,
    )

    return fig


def plot_traj_colormap_ape(
    ape_metric, traj_ref, traj_est1, traj_est2=None, plot_title="", figsize=(8, 8)
):
    """
    Add a trajectory colormap of ATE metrics to a plot collection.

    Args:
        ape_metric: an evo.core.metric object with statistics and information for APE.
        traj_ref: a PoseTrajectory3D object representing the reference trajectory.
        traj_est1: a PoseTrajectory3D object representing the vio-estimated trajectory.
        traj_est2: a PoseTrajectory3D object representing the pgo-estimated trajectory.
        plot_title: a string representing the title of the plot.
        figsize: a 2-tuple representing the figure size.

    Returns:
        A plt figure.
    """
    fig = plt.figure(figsize=figsize)
    plot_mode = evo.tools.plot.PlotMode.xy
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)

    ape_stats = ape_metric.get_all_statistics()

    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "reference")

    colormap_traj = traj_est1
    if traj_est2 is not None:
        evo.tools.plot.traj(
            ax, plot_mode, traj_est1, ".", "gray", "reference without pgo"
        )
        colormap_traj = traj_est2

    evo.tools.plot.traj_colormap(
        ax,
        colormap_traj,
        ape_metric.error,
        plot_mode,
        min_map=0.0,
        max_map=math.ceil(ape_stats["max"] * 10) / 10,
        title=plot_title,
    )

    return fig


def plot_traj_colormap_rpe(
    rpe_metric, traj_ref, traj_est1, traj_est2=None, plot_title="", figsize=(8, 8)
):
    """
    Add a trajectory colormap of RPE metrics to a plot collection.

    Args:
        ape_metric: an evo.core.metric object with statistics and information for RPE.
        traj_ref: a PoseTrajectory3D object representing the reference trajectory.
        traj_est1: a PoseTrajectory3D object representing the vio-estimated trajectory.
        traj_est2: a PoseTrajectory3D object representing the pgo-estimated trajectory.
        plot_title: a string representing the title of the plot.
        figsize: a 2-tuple representing the figure size.

    Returns:
        A plt figure.
    """
    fig = plt.figure(figsize=figsize)
    plot_mode = evo.tools.plot.PlotMode.xy
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)

    # We have to make deep copies to avoid altering the original data:
    traj_ref = copy.deepcopy(traj_ref)
    traj_est1 = copy.deepcopy(traj_est1)
    traj_est2 = copy.deepcopy(traj_est2)

    rpe_stats = rpe_metric.get_all_statistics()
    traj_ref.reduce_to_ids(rpe_metric.delta_ids)
    traj_est1.reduce_to_ids(rpe_metric.delta_ids)

    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "reference")

    colormap_traj = traj_est1
    if traj_est2 is not None:
        traj_est2.reduce_to_ids(rpe_metric.delta_ids)
        evo.tools.plot.traj(
            ax, plot_mode, traj_est1, ".", "gray", "reference without pgo"
        )
        colormap_traj = traj_est2

    evo.tools.plot.traj_colormap(
        ax,
        colormap_traj,
        rpe_metric.error,
        plot_mode,
        min_map=0.0,
        max_map=math.ceil(rpe_stats["max"] * 10) / 10,
        title=plot_title,
    )

    return fig


def add_results_to_collection(
    data, results: TrajectoryResults, name, plots=None, extra_trajectory=None
):
    """Add metrics to plot collection."""
    if plots is None:
        plots = evo.tools.plot.PlotCollection("Example")

    ape_name = f"{name} APE Translation"
    ape_traj_name = f"{name} ATE Mapped Onto Trajectory"
    plots.add_figure(ape_name, plot_metric(results.ape_translation, ape_name))
    plots.add_figure(
        ape_traj_name,
        plot_traj_colormap_ape(
            results.ape_translation, data.ref, data.est, extra_trajectory, ape_traj_name
        ),
    )

    rpe_tname = f"{name} RPE Translation"
    rpe_traj_tname = f"{name} RPE Translation Error Mapped Onto Trajectory"
    plots.add_figure(rpe_tname, plot_metric(results.rpe_translation, rpe_tname))
    plots.add_figure(
        rpe_traj_tname,
        plot_traj_colormap_rpe(
            results.rpe_translation,
            data.ref,
            data.est,
            extra_trajectory,
            rpe_traj_tname,
        ),
    )

    rpe_rname = f"{name} RPE Rotation"
    rpe_traj_rname = f"{name} RPE Rotation Error Mapped Onto Trajectory"
    plots.add_figure(rpe_rname, plot_metric(results.rpe_rotation, rpe_rname))
    plots.add_figure(
        rpe_traj_rname,
        plot_traj_colormap_rpe(
            results.rpe_rotation, data.ref, data.est, extra_trajectory, rpe_traj_rname
        ),
    )

    return plots
