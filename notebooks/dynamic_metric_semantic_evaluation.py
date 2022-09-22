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
import numpy as np
import os
import glog as log
import copy

from __future__ import division

import open3d as o3d
import pandas as pd

from evaluation.tools.mesh import Mesh
from evaluation.tools.mesh_evaluator import MeshEvaluator

# Rotation matrices:
# East North Up (ENU) frame to Unity's world frame of reference
# fmt: off
enu_R_unity = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]])
# fmt: on
unity_R_enu = np.transpose(enu_R_unity)

# Right Handed frame to Unity's Left Handed frame of reference
# fmt: off
righthand_R_lefthand = np.array([[1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, 1]])
# fmt: on
lefthand_R_righthand = np.transpose(righthand_R_lefthand)


visualize = False

# %%
# FILL PATHS BELOW
# gt_mesh_path = "/home/tonirv/Downloads/tesse_multiscene_office1_3d_semantic_v5.ply"
# est_mesh_path = "/home/tonirv/Downloads/tesse_semantics_2.ply"

# gt_mesh_path = "/home/tonirv/Code/ROS/flight_goggles_ws/src/voxblox/voxblox_ros/mesh_results/semantic_mesh_tonirv_ld_9118_6487309760727328010.ply"
# est_mesh_path = "/home/tonirv/Code/ROS/flight_goggles_ws/src/voxblox/voxblox_ros/mesh_results/semantic_mesh_tonirv_ld_9118_6487309760727328010.ply"

# gt_mesh_path = "/home/tonirv/Downloads/tesse_multiscene_office1_3d_semantic_v5.ply"
# est_mesh_path = "/home/tonirv/Code/ROS/flight_goggles_ws/src/voxblox/voxblox_ros/mesh_results/tesse_semantics_3.ply"

gt_mesh_path = "/home/tonirv/Downloads/office1_tony.ply"

est_base_path = "/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/mesh_results/"
# est_mesh_names = ["humans_6_long_gt_dyn.ply",
#                  "humans_6_long_gt_no_dyn.ply",
#                  "humans_12_long_gt_dyn.ply",
#                  "humans_12_long_gt_no_dyn.ply",
#                  "humans_30_long_gt_dyn.ply",
#                  "humans_30_long_gt_no_dyn.ply"
#                 ]
est_mesh_names = [
    "humans_6_longvio_dyn.ply",
    "humans_6_longvio_no_dyn.ply",
    # "humans_12_long_vio_dyn.ply",
    # "humans_12_long_vio_no_dyn.ply",
    # "humans_30_long_vio_dyn.ply",
    # "humans_30_long_vio_no_dyn.ply"
]

est_mesh_paths = []
for est_mesh_name in est_mesh_names:
    est_mesh_path = est_base_path + est_mesh_name
    est_mesh_paths.append(est_mesh_path)
    print(est_mesh_path)


# est_mesh_path = "/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/mesh_results/humans_6_long_gt_no_dyn.ply"

# est_mesh_path = "/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/mesh_results/humans_12_long_gt_dyn.ply"
# est_mesh_path = "/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/mesh_results/humans_12_long_gt_no_dyn.ply"

# est_mesh_path = "/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/mesh_results/humans_12_long_gt_dyn.ply"
# est_mesh_path = "/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/mesh_results/humans_12_long_gt_no_dyn.ply"

# %%
print("Loading Ground-truth mesh...")
gt_mesh_original = Mesh(gt_mesh_path)

# Transform Meshes to same frame of reference
gt_mesh = copy.deepcopy(gt_mesh_original)

# Align Pointclouds Manually:
# est_mesh.mesh_o3d.translate([0, -5, 0])
# gt_mesh.transform_left(righthand_R_lefthand)
gt_mesh.transform_left(enu_R_unity)

if visualize:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True
    vis.add_geometry(est_mesh.mesh_o3d)
    vis.add_geometry(gt_mesh.mesh_o3d)
    vis.add_geometry(o3d.geometry.create_mesh_coordinate_frame(size=4))
    vis.run()
    vis.destroy_window()

NUMBER_OF_SAMPLES = 1000000
gt_pcl = o3d.geometry.sample_points_uniformly(gt_mesh.mesh_o3d, NUMBER_OF_SAMPLES)

# %%
print("Loading Estimated meshes...")
dict_est_pcls = dict()
for est_mesh_path in est_mesh_paths:
    est_mesh_original = Mesh(est_mesh_path)
    est_mesh = copy.deepcopy(est_mesh_original)
    # Don't sample estimated mesh, just pick vertices, otw you'll be mixing colors...
    # est_pcl = o3d.geometry.sample_points_uniformly(est_mesh.mesh_o3d, NUMBER_OF_SAMPLES)
    est_pcl = o3d.io.read_point_cloud(est_mesh_path)
    dict_est_pcls[est_mesh_path] = est_pcl


# %%
# ICP
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_correspondences(source, target, correspondences):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, correspondences])  # target_temp,


# %%
# ICP params
ICP_THRESHOLD = 1.5
trans_init = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# %%
def visualize_two_pcls(gt_pcl, est_pcl):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().mesh_show_back_face = True
    vis.add_geometry(gt_pcl)
    vis.add_geometry(est_pcl)
    vis.run()
    vis.destroy_window()


# %%
from tqdm import tqdm, trange
from random import random, randint
from time import sleep


def evaluate_icp_for_clouds(gt_pcl, dict_est_pcls):
    with trange(len(dict_est_pcls)) as t:
        i = 0
        for est_pcl_key in dict_est_pcls:
            # Description will be displayed on the left
            t.set_description("GEN %i" % i)
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype
            t.set_postfix(loss=random(), gen=randint(1, 999), str="h", lst=[1, 2])

            est_pcl = dict_est_pcls[est_pcl_key]
            if visualize:
                visualize_two_pcls(gt_pcl, est_pcl)
                draw_registration_result(est_pcl, gt_pcl, trans_init)

            evaluation = o3d.registration.evaluate_registration(
                est_pcl, gt_pcl, ICP_THRESHOLD, trans_init
            )
            reg_p2p = o3d.registration.registration_icp(
                est_pcl,
                gt_pcl,
                ICP_THRESHOLD,
                trans_init,
                o3d.registration.TransformationEstimationPointToPoint(),
                o3d.registration.ICPConvergenceCriteria(max_iteration=2),
            )
            correspondences = reg_p2p.correspondence_set

            if visualize:
                # Draw Registration Result
                draw_registration_result(est_pcl, gt_pcl, reg_p2p.transformation)

            print("# # # # REGISTRATION INLIER RMSE for: %s " % est_pcl_key)
            print(reg_p2p.inlier_rmse)
            print("")
            i = i + 1


# %%
# RUN FULL EVALUATION
evaluate_icp_for_clouds(gt_pcl, dict_est_pcls)

# %%
# Visualize initial registration problem
if visualize:
    draw_registration_result(est_pcl, gt_pcl, trans_init)

# %%
# Evaluate current fit between pointclouds
evaluation = o3d.registration.evaluate_registration(
    est_pcl, gt_pcl, ICP_THRESHOLD, trans_init
)

# %%
print("Initial registration")
print(evaluation)

# %%
print("Apply point-to-point ICP")
reg_p2p = o3d.registration.registration_icp(
    est_pcl,
    gt_pcl,
    ICP_THRESHOLD,
    trans_init,
    o3d.registration.TransformationEstimationPointToPoint(),
    o3d.registration.ICPConvergenceCriteria(max_iteration=2000),
)
correspondences = reg_p2p.correspondence_set

# %%
print(reg_p2p)
print("")

print("Transformation is:")
print(reg_p2p.transformation)
print("")

print("Correspondence Set:")
print(reg_p2p.correspondence_set)
print("")

print("# # # # REGISTRATION INLIER RMSE: # # # # ")
print(reg_p2p.inlier_rmse)
print("")
