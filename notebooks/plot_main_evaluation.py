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

import plotly
import plotly.express as px
import plotly.graph_objects as go

import glog as log

import evaluation.tools as evt
from evaluation.evaluation_lib import aggregate_all_results

# %%
# Parse experiment yaml file
experiments_path = "../experiments/full_euroc.yaml"

# Get experiment information from yaml file.
experiment_params = yaml.load(open(experiments_path))

# Get directory where all results are stored
results_dir = os.path.expandvars(experiment_params["results_dir"])

# Collect results
stats = aggregate_all_results(results_dir)
# evt.check_stats(stats)

# %%
figure = evt.draw_ape_boxplots_plotly(stats, False)
figure.show()
