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
artifacts_path = "/home/tonirv/Documents/uHumans2_VIO_vxblx/"
gt_meshes_path = "/home/tonirv/datasets/uHumans2/uHumans dataset V2.0 GT Meshes/"

semantic_labels_csvs_path = "/home/tonirv/Code/ROS/kimera_ws/src/Kimera-Semantics/kimera_semantics_ros/cfg/"
visualize = False


# %%
def run_mesh_evaluation(scene_type, human_size, number_of_mesh_samples):
    est_mesh_base_path = artifacts_path + "/{}_scene/uHumans2_{}_s1_{}h/".format(scene_type, scene_type, human_size)
    est_mesh_names = [
      #'mesh_DVIO.ply', 
      #'mesh_gt.ply', 
      #'mesh_DVIO_wo_DM.ply', 
      #'mesh_gt_wo_DM.ply',
        'mesh_pgmo.ply'
    ]
    
    # Parallelize! Write output to file?
    for est_mesh_name in est_mesh_names:
        est_mesh_path = est_mesh_base_path + est_mesh_name
        print("EVAL: {} in {} scene".format(est_mesh_name, scene_type))
        if not os.path.exists(est_mesh_path):
            print("Path to {} doesn't exist: {}".format(est_mesh_name, est_mesh_path))
            continue
            
        mesh_eval = MeshEvaluator(est_mesh_path, gt_mesh_path, semantic_labels_csv_path, visualize)
        
        only_geometric_eval = False
        #if "_" in est_mesh_name:
            # Only compute geometric erros if comparing with vs wo DM.
        #    only_geometric_eval = True
        
        inlier_rmse, semantic_accuracy = mesh_eval.compare_meshes(number_of_mesh_samples, only_geometric_eval)
        
        print("Inlier RMSE [m]: ", inlier_rmse)
        print("Semantic Accuracy [%]: ", semantic_accuracy)


# %% [markdown]
# # Apartment Scene

# %%
gt_mesh_path = gt_meshes_path + "apartment.ply"
semantic_labels_csv_path = semantic_labels_csvs_path + "tesse_multiscene_archviz1_segmentation_mapping.csv"

# %% [markdown]
# ## Apartment S1 00h
#

# %%
run_mesh_evaluation("apartment", "00", 10000000)


# %% [markdown]
# ## Apartment S1 01h
#

# %%
run_mesh_evaluation("apartment", "01", 10000000)


# %% [markdown]
# ## Apartment S1 02h
#

# %%
run_mesh_evaluation("apartment", "02", 10000000)


# %% [markdown]
# # Office Scene

# %%
gt_mesh_path = gt_meshes_path + "office.ply"
semantic_labels_csv_path = semantic_labels_csvs_path + "tesse_multiscene_office2_segmentation_mapping.csv"

# %% [markdown]
# ## Office Scene 00h

# %%
run_mesh_evaluation("office", "00", 10000000)

# %% [markdown]
# ## Office Scene 06h

# %%
run_mesh_evaluation("office", "06", 10000000)

# %% [markdown]
# ## Office Scene 12h

# %%
run_mesh_evaluation("office", "12", 10000000)

# %% [markdown]
# # Nieghborhood Scene

# %%
gt_mesh_path = gt_meshes_path + "neighborhood.ply"
semantic_labels_csv_path = semantic_labels_csvs_path + "tesse_multiscene_neighborhood1_segmentation_mapping.csv"

# %% [markdown]
# ## Nieghborhood Scene 00h

# %%
run_mesh_evaluation("neighborhood", "00", 10000000)

# %% [markdown]
# ## Nieghborhood Scene 24h

# %%
run_mesh_evaluation("neighborhood", "24", 10000000)

# %% [markdown]
# ## Nieghborhood Scene 36h

# %%
run_mesh_evaluation("neighborhood", "36", 10000000)

# %% [markdown]
# # Subway Scene

# %%
gt_mesh_path = gt_meshes_path + "subway.ply"
semantic_labels_csv_path = semantic_labels_csvs_path + "tesse_multiscene_underground1_segmentation_mapping.csv"

# %% [markdown]
# ## Subway Scene 00h

# %%
run_mesh_evaluation("subway", "00", 10000000)

# %% [markdown]
# ## Subway Scene 24h

# %%
run_mesh_evaluation("subway", "24", 10000000)

# %% [markdown]
# ## Subway Scene 36h

# %%
run_mesh_evaluation("subway", "36", 1000000)
