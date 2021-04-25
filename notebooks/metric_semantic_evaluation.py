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

# %%
# FILL PATHS BELOW
est_mesh_path = "/home/tonirv/Documents/uHumans1_PGMO_vxblx/office_scene/uHumans1_06h/mesh_pgmo.ply"
gt_mesh_path = "/home/tonirv/datasets/uHumans1/gt_mesh/office_toni2.ply"
visualize = True
semantic_labels_csv_path="/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/cfg/tesse_multiscene_office2_segmentation_mapping.csv"

mesh_eval = MeshEvaluator(est_mesh_path, gt_mesh_path, semantic_labels_csv_path)
mesh_eval.compare_meshes(only_geometric = True)

