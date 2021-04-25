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
import os.path
from os import path

from evaluation.metric_semantic_evaluation import MeshEvaluator

# %% [markdown]
# # uHumans1

# %%
visualize = True
only_geometric_eval = False
gt_mesh_path = "/home/tonirv/datasets/uHumans1/gt_mesh/office_toni2.ply"
semantic_labels_csv_path="/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/cfg/tesse_multiscene_office1_segmentation_mapping.csv"

# %% [markdown]
# ## Office 06h

# %%
# FILL PATHS BELOW
est_mesh_path = "/home/tonirv/Documents/uHumans1_PGMO_vxblx/office_scene/uHumans1_06h/mesh_pgmo.ply"

mesh_eval = MeshEvaluator(est_mesh_path, gt_mesh_path, semantic_labels_csv_path)
mesh_eval.compare_meshes(only_geometric = only_geometric_eval)

# %% [markdown]
# ## Office 12h

# %%
# FILL PATHS BELOW
est_mesh_path = "/home/tonirv/Documents/uHumans1_PGMO_vxblx/office_scene/uHumans1_12h/mesh_pgmo.ply"

mesh_eval = MeshEvaluator(est_mesh_path, gt_mesh_path, semantic_labels_csv_path)
mesh_eval.compare_meshes(only_geometric = only_geometric_eval)


# %% [markdown]
# ## Office 30h

# %%
# FILL PATHS BELOW
est_mesh_path = "/home/tonirv/Documents/uHumans1_PGMO_vxblx/office_scene/uHumans1_30h/mesh_pgmo.ply"

mesh_eval = MeshEvaluator(est_mesh_path, gt_mesh_path, semantic_labels_csv_path)
mesh_eval.compare_meshes(only_geometric = only_geometric_eval)

