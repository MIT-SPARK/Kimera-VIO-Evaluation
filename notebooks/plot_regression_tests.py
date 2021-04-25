# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import yaml
import os
import logging
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    log.addHandler(ch)

# %%
# Parse experiment yaml file
experiments_path = "../experiments/regression_test.yaml"

# Get experiment information from yaml file.
experiment_params = yaml.load(open(experiments_path))

regression_tests_dir = os.path.expandvars(experiment_params["regression_tests_dir"])

datasets_to_run = experiment_params["datasets_to_run"]
regression_params = experiment_params["regression_parameters"]


# %%
# Retrieve stats, if they are not there, try to collect them:
def collect_stats(
    full_stats_path, regression_params, regression_tests_dir, datasets_to_run
):
    # TODO(Toni): recollection of results should be automatic by looking for results.yaml files in the
    # regression_tests_dir file system.
    # Collect all yaml results for a given parameter name:
    stats = dict()
    for regression_param in regression_params:
        # Redirect to param_name_value dir param_name = regression_param['name']
        param_name = regression_param["name"]
        stats[param_name] = dict()
        for param_value in regression_param["values"]:
            results_dir = os.path.join(
                regression_tests_dir, param_name, str(param_value)
            )
            # Redirect to modified params_dir
            params_dir = os.path.join(results_dir, "params")
            stats[param_name][param_value] = dict()
            for dataset in datasets_to_run:
                dataset_name = dataset["name"]
                pipelines_to_run = dataset["pipelines"]
                stats[param_name][param_value][dataset_name] = dict()
                for pipeline in pipelines_to_run:
                    results_file = os.path.join(
                        results_dir, dataset_name, pipeline, "results.yaml"
                    )
                    if os.path.isfile(results_file):
                        stats[param_name][param_value][dataset_name][
                            pipeline
                        ] = yaml.load(open(results_file, "r"))
                    else:
                        log.warning(
                            "Could not find results file: {}. Adding cross to boxplot...".format(
                                results_file
                            )
                        )
                        stats[param_name][param_value][dataset_name][pipeline] = False

    # Save all stats in regression tests root directory for future usage.
    with open(full_stats_path, "w") as outfile:
        outfile.write(yaml.dump(stats))
    return stats


full_stats_path = os.path.join(regression_tests_dir, "all_stats.yaml")
stats = dict()
if os.path.isfile(full_stats_path):
    log.info("Found existent stats. Opening full stats from:" + full_stats_path)
    stats = yaml.load(open(full_stats_path))
else:
    log.info("Collecting full stats.")
    stats = collect_stats(
        full_stats_path, regression_params, regression_tests_dir, datasets_to_run
    )

# Push to the cloud?!

# %%
# Store stats in a tidy Pandas DataFrame # TODO(Toni): this should be done in the evaluation_lib.py script...
def listify_regression_stats(stats):
    """ Makes a list of lists out of the stats (for easy conversion into pandas dataframe) """
    stats_list = []
    for param_name in stats:
        for param_value in stats[param_name]:
            for dataset_name in stats[param_name][param_value]:
                for pipeline in stats[param_name][param_value][dataset_name]:
                    result = stats[param_name][param_value][dataset_name][pipeline]
                    if result != False:
                        result = result["absolute_errors"].np_arrays["error_array"]
                        stats_list.append(
                            [param_name, param_value, dataset_name, pipeline, result]
                        )
    return stats_list


# Create or load Pandas DataFrame
df = pd.DataFrame()
all_stats_pickle_dir = os.path.join(regression_tests_dir, "all_stats.pkl")
if os.path.isfile(all_stats_pickle_dir):
    log.info(
        "Found existent pickle file. Opening pickled stats from:" + all_stats_pickle_dir
    )
    df = pd.read_pickle(all_stats_pickle_dir)
else:
    log.info("Creating dataframe stats.")
    df = pd.DataFrame.from_records(listify_regression_stats(stats))
    df.columns = [
        "Param Name",
        "Param Value",
        "Dataset Name",
        "Pipe Type",
        "ATE errors",
    ]
    df.set_index(["Param Name", "Dataset Name"], inplace=True)

    # Save dataframe as pickle for future use
    # df.to_pickle(all_stats_pickle_dir)

# Print df
df


# %%
def regression_boxplot(param_name, dataset_name, tidy):
    tidy.set_index(["Param Value", "Pipe Type"], inplace=True)
    tidy_2 = (
        tidy["ATE errors"]
        .apply(lambda x: pd.Series(x))
        .stack()
        .reset_index(level=2, drop=True)
        .to_frame("ATE errors")
    )
    tidy_2.reset_index(level=["Pipe Type", "Param Value"], drop=False, inplace=True)
    fig = px.box(
        tidy_2, x="Param Value", y="ATE errors", points="all", color="Pipe Type"
    )

    fig.update_layout(
        title=go.layout.Title(text="Dataset: " + dataset_name),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=param_name)),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(text="ATE [m]"), rangemode="tozero"
        ),
        template="plotly_white",
    )
    return fig


# %%
# Generate figures
figures = [
    regression_boxplot(x, y, df.loc[x].loc[[y]])
    for x in df.index.levels[0]
    for y in df.index.levels[1]
]

# %%
# Show figures
for figure in figures:
    figure.show()

# %%
import plotly.io as pio

pio.orca.status
plotly.io.orca.config.executable = "venv/bin/orca-server"

# %%
# Save figures
if not os.path.exists("figures"):
    os.mkdir("figures")
for fig in figures:
    plotly.offline.plot(
        fig,
        filename="figures/regression_test_"
        + fig.layout.title.text
        + "_"
        + fig.layout.xaxis.title.text
        + ".html",
    )

# for figure in figures:
#    figure.write_image("figures/"+ figure.layout.title.text + ".svg")

# %%
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
import plotly.graph_objects as go
from chart_studio.grid_objs import Column, Grid

from datetime import datetime as dt
import numpy as np
from IPython.display import IFrame

upload_plots_online = True
if upload_plots_online:
    for fig in figures:
        py.iplot(
            fig,
            filename="regression_test_"
            + fig.layout.title.text
            + "_"
            + fig.layout.xaxis.title.text
            + ".html",
            world_readable=True,
            auto_open=True,
        )


# %%
def url_to_iframe(url, text=True):
    html = ""
    # style
    html += """<head>
    <style>
    div.textbox {
        margin: 30px;
        font-weight: bold;
    }
    </style>
    </head>'
    """
    # iframe
    html += (
        "<iframe src="
        + url
        + '.embed#{} width=750 height=400 frameBorder="0"></iframe>'
    )
    if text:
        html += """<body>
        <div class="textbox">
            <p>Click on the presentation above and use left/right arrow keys to flip through the slides.</p>
        </div>
        </body>
        """
    return html


# %%
