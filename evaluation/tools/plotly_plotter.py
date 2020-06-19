#!/usr/bin/env python

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio.plotly as py

import glog as log
import pandas as pd
from datetime import datetime as date

def draw_feature_tracking_stats(df, show_figure=False):
    """ Draws a plotly bar plot from the csv file of the frontend (parsed as a pandas data frame df).

    Args:
    - df: pandas dataframe with frontend stats (expects as columns: `nr_keypoints`, `nrTrackerFeatures`, `nrMonoInliers`)
    - show_figure: optional param to show figure

    Returns:
    - fig: a plotly figure handle
    """
    colors = ['green' if mono_status == "VALID" else 'crimson' for mono_status in df["mono_status"]]
    x = df.index
    fig = go.Figure(data=[
        go.Bar(name='Keypoints Detected', x=x, y=df["nr_keypoints"]),
        go.Bar(name='Tracked Features', x=x, y=df["nrTrackerFeatures"]),
        go.Bar(name='Monocular Inliers (colored by Status)', x=x, y=df["nrMonoInliers"], marker_color=colors,
               hovertext=df["mono_status"], hovertemplate="Mono Inliers: %{y} <br>Mono Status: %{hovertext}")
    ])
    fig.update_layout(barmode='overlay', template='plotly_white')

    if show_figure:
        fig.show()

    return fig

def draw_mono_stereo_inliers_outliers(df, show_figure=False):
    """ Draws a plotly bar plot from the csv file of the frontend (parsed as a pandas data frame df)
    showing the rate of inliers/outliers for the mono/stereo tracks.

    Args:
        - df: pandas dataframe with frontend stats (expects as columns: `nr_keypoints`, `nrTrackerFeatures`, `nrMonoInliers`)
        - show_figure: optional param to show figure

    Returns:
        - fig: a plotly figure handle
    """

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
    x = df.index
    fig.add_trace(go.Bar(name='# Mono Putatives', x=x, y=df["nrMonoPutatives"]), row=1, col=1)
    fig.add_trace(go.Bar(name='# Mono Inliers', x=x, y=df["nrMonoInliers"]), row=1, col=1)

    fig.add_trace(go.Bar(name='# Stereo Putatives', x=x, y=df["nrStereoPutatives"]), row=2, col=1)
    fig.add_trace(go.Bar(name='# Stereo Inliers', x=x, y=df["nrStereoInliers"]), row=2, col=1)

    fig.add_trace(go.Bar(name='Mono RANSAC Iterations', x=x, y=df["monoRansacIters"]), row=3, col=1)

    fig.add_trace(go.Bar(name='Stereo RANSAC Iterations', x=x, y=df["stereoRansacIters"]), row=4, col=1)

    fig.update_layout(barmode='overlay')

    if show_figure:
        fig.show()

    return fig

def draw_frontend_timing(df, show_figure=False):
    """ Draws a plotly bar plot from the csv file of the frontend (parsed as a pandas data frame df):
    plot timing from frontend.

    Args:
        - df: pandas dataframe with frontend stats (expects as columns: `nr_keypoints`, `nrTrackerFeatures`, `nrMonoInliers`)
        - show_figure: optional param to show figure

    Returns:
        - fig: a plotly figure handle
    """
    fig = go.Figure()
    x = df.index
    fig.add_trace(go.Bar(name='Feature Detection Time', x=x, y=df["featureDetectionTime"]))
    fig.add_trace(go.Bar(name='Feature Tracking Time', x=x, y=df["featureTrackingTime"]))
    fig.add_trace(go.Bar(name='Mono RANSAC Time', x=x, y=df["monoRansacTime"]))
    fig.add_trace(go.Bar(name='Stereo RANSAC Time', x=x, y=df["stereoRansacTime"]))
    fig.add_trace(go.Bar(name='Feature Selection Time', x=x, y=df["featureSelectionTime"]))
    fig.update_layout(barmode='stack', title_text='Frontend Timing')

    if show_figure:
        fig.show()

    return fig

def draw_boxplot_plotly(df):
    tidy = df.set_index(['Dataset Name'])
    tidy = tidy['ATE errors'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('ATE errors')
    tidy.reset_index(level=['Dataset Name'], drop=False, inplace=True)
    tidy.sort_values('Dataset Name', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Box(x=tidy['Dataset Name'], y=tidy['ATE errors'], boxpoints='all', boxmean=True))

    fig.update_layout(
    title=go.layout.Title(
        text="Kimera-VIO ATE Euroc dataset " + str(date.today())
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='Datasets'
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="ATE [m]"
            ),
        rangemode='tozero'
        ),
    template='plotly_white'
    )
    return fig

def draw_ape_boxplots_plotly(stats, upload_plots_online = False, show_figure = False):
    """ Simplified boxplot plotting using plotly for APE boxplots:
    See draw_ape_boxplots for the complicated version.

    Args:
        - stats: vio statistics (see 'draw_ape_boxplots' function)
        - upload_plots_online: if set to True it will publish plots online to plotly server.
        (to publish online, you need to follow the instructions here: )
        If False, it will just show the boxplot figure.
        - show_figure: whether to display the figure or not
    Returns:
        - the handle to the plotly figure
    """

    def listify_stats(stats):
        """ Makes a list of lists out of the stats (for easy conversion into pandas dataframe) """
        stats_list = []
        for dataset_name in stats:
            for pipeline in stats[dataset_name]:
                result = stats[dataset_name][pipeline]
                if result != False:
                    result = result['absolute_errors'].np_arrays['error_array']
                    stats_list.append([dataset_name, pipeline, result])
        return stats_list

    df = pd.DataFrame()
    log.info("Creating dataframe stats.")
    df = pd.DataFrame.from_records(listify_stats(stats))
    df.columns = ['Dataset Name', 'Pipe Type', 'ATE errors']

    figure = draw_boxplot_plotly(df)

    if upload_plots_online:
        py.iplot(figure, filename=figure.layout.title.text + '.html', world_readable=True, auto_open=False)
    if show_figure:
        figure.show()
    return figure
