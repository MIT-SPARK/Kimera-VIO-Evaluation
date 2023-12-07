# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Plot Backend
#
# Plots the APE, RPE and trajectory against ground truth for the final backend
# output trajectory.

# %%
import os
import copy
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

import kimera_eval
import kimera_eval.notebook_utilities as neval

import pathlib

import evo.tools.plot
from evo.core import sync

# #%matplotlib widget
# #%matplotlib notebook
# %matplotlib inline
import matplotlib.pyplot as plt

neval.setup_logging(__name__)


# %% [markdown]
# ## Data Locations
#
# Make sure to set the following paths.
#
# `vio_output_dir` is the path to the directory containing `output_*.csv` files obtained
# from logging a run of SparkVio.
#
# `gt_data_file` is the absolute path to the `csv` file containing ground truth data for
# the absolute pose at each timestamp of the dataset.

# %%
# Define directory to VIO output csv files as well as ground truth absolute poses.
vio_output_dir = "~/test_output/MH_01_easy/Euroc"
gt_data_file = None

vio_output_dir = pathlib.Path(vio_output_dir).expanduser()
if gt_data_file is None:
    gt_data_file = vio_output_dir / "traj_gt.csv"


# %% [markdown]
# ## Backend Trajectory
#
# Associate, align and process the trajectory as determined by the backend.
# Note that this does not include loop closure factors or other optimizations.
# This is pure VIO.


# %%
# Load ground truth and estimated data as csv DataFrames.
gt_df = pd.read_csv(gt_data_file, sep=",", index_col=0)
gt_df = gt_df[~gt_df.index.duplicated()]

output_poses_filename = vio_output_dir / "traj_vio.csv"
output_poses_df = pd.read_csv(output_poses_filename, sep=",", index_col=0)


# %%
# Convert the gt relative-pose DataFrame to a trajectory object.
traj_ref_complete = kimera_eval.df_to_trajectory(gt_df)

# Use the backend poses as trajectory.
traj_est_unaligned = kimera_eval.df_to_trajectory(output_poses_df)
discard_n_start_poses = 0
discard_n_end_poses = 0

# Associate the data.
traj_est = copy.deepcopy(traj_est_unaligned)
traj_ref, traj_est = sync.associate_trajectories(traj_ref_complete, traj_est)
traj_est = kimera_eval.align_trajectory(
    traj_est,
    traj_ref,
    correct_scale=False,
    discard_n_start_poses=int(discard_n_start_poses),
    discard_n_end_poses=int(discard_n_end_poses),
)

print("traj_ref: ", str(traj_ref))
print("traj_est: ", str(traj_est))

# %%
# plot ground truth trajectory with pose
plot_mode = evo.tools.plot.PlotMode.xy
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)
neval.draw_trajectory_poses(
    ax, traj_ref_complete, marker_scale=0.2, plot_mode=plot_mode, downsample=100
)
neval.draw_start_and_end(ax, traj_ref_complete, plot_mode)
evo.tools.plot.traj(ax, plot_mode, traj_ref_complete, "--", "gray", "reference")
plt.title("Reference trajectory with pose")
plt.show()

# plot unaligned trajectory with pose
plot_mode = evo.tools.plot.PlotMode.xy
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)
neval.draw_trajectory_poses(
    ax, traj_est_unaligned, marker_scale=0.3, plot_mode=plot_mode
)
neval.draw_start_and_end(ax, traj_est_unaligned, plot_mode)
evo.tools.plot.traj(ax, plot_mode, traj_est_unaligned, "--", "gray", "reference")
plt.title("Estimated trajectory with pose")
plt.show()


# %%
plot_mode = evo.tools.plot.PlotMode.xyz
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)

gt_df_downsampled = gt_df.iloc[:1200:100]

# reference trajectory
traj_ref_downsampled = kimera_eval.df_to_trajectory(gt_df_downsampled)
neval.draw_trajectory_poses(ax, traj_ref, plot_mode=plot_mode, marker_scale=3)
neval.draw_trajectory_poses(ax, traj_est, plot_mode=plot_mode, marker_scale=3)
evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "reference")
evo.tools.plot.traj(ax, plot_mode, traj_est, "--", "green", "estimate (aligned)")

plt.title("Trajectory with pose")
plt.show()


# %% [markdown]
# ## Absolute-Pose-Error Plotting
#
# Plot absolute-pose-error along the entire trajectory.
# APE gives a good sense of overall VIO performance across the entire trajectory.

# %% [markdown]
# ### Absolute Translation Errors
#
# The following two plots show 1) VIO's absolute translation errors (ATE) in meters with
# respect to time, and 2) estimated trajectory color coded by magnitudes of the ATE.

# %%
# Plot APE of trajectory rotation and translation parts.
num_of_poses = traj_est.num_poses
traj_est.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)
traj_ref.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)

seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]

ape_tran = kimera_eval.get_ape_trans((traj_ref, traj_est))
fig1 = kimera_eval.plot_metric(ape_tran, "VIO ATE in Meters")
plt.show()

# %%
# Plot the ground truth and estimated trajectories against each other with APE overlaid.
fig = kimera_eval.plot_traj_colormap_ape(
    ape_tran,
    traj_ref,
    traj_est,
    plot_title="VIO Trajectory Tracking - Color Coded by ATE",
)
plt.show()

# %% [markdown]
# ### Absolute Rotation Errors
#
# The following two plots show 1) VIO's absolute rotation errors (ARE) in meters with
# respect to time, and 2) estimated trajectory color coded by magnitudes of the ARE.
# Note that the estimated trajectory used here, unlike ATE, is the unaligned,
# original estimated trajectory.

# %%
# Plot ARE
traj_est_unaligned.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)

ape_rot = kimera_eval.get_ape_rot((traj_ref, traj_est_unaligned))
fig2 = kimera_eval.plot_metric(ape_rot, "VIO ARE in Degrees")
plt.show()

# %%
# Plot the ground truth and estimated trajectories against each other with APE overlaid.
fig2 = kimera_eval.plot_traj_colormap_ape(
    ape_rot,
    traj_ref,
    traj_est_unaligned,
    plot_title="VIO Trajectory Tracking - Color Coded by ARE",
)
plt.show()

# %% [markdown]
# ## Velocity Estimate vs GT Plotting
#
# The two lines for each axis should match fairly well.

# %%
gt_vel = np.linalg.norm(np.column_stack((gt_df.vx, gt_df.vy, gt_df.vz)), axis=1)
est_vel = np.linalg.norm(
    np.column_stack((output_poses_df.vx, output_poses_df.vy, output_poses_df.vz)),
    axis=1,
)

fig = plt.figure()
plt.plot(gt_df.index, gt_vel, label="Ground Truth")
plt.plot(output_poses_df.index, est_vel, label="VIO Estimate")
plt.title("Velocity Norm")
plt.legend()
plt.show()

# %% [markdown]
# ## Relative-Pose-Error Plotting
#
# Plot relative-pose-error along the entire trajectory.
# RPE gives a good sense of overall VIO performance from one frame to the next.

# %%
# Get RPE for entire relative trajectory.
rpe_rot = kimera_eval.get_rpe_rot((traj_ref, traj_est))
rpe_tran = kimera_eval.get_rpe_trans((traj_ref, traj_est))

# %% [markdown]
# ### Relative Translation Errors

# %%
# Plot RPE of trajectory rotation and translation parts.
seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps[1:]]

fig1 = kimera_eval.plot_metric(rpe_tran, "VIO RTE in Meters")
fig2 = kimera_eval.plot_traj_colormap_rpe(
    rpe_tran,
    traj_ref,
    traj_est,
    plot_title="VIO Trajectory Tracking - Color Coded by RTE",
)
plt.show()

# %% [markdown]
# ### Relative Rotation Errors

# %%
fig1 = kimera_eval.plot_metric(rpe_rot, "VIO RRE in Degrees")
fig2 = kimera_eval.plot_traj_colormap_rpe(
    rpe_rot,
    traj_ref,
    traj_est,
    plot_title="VIO Trajectory Tracking - Color Coded by RRE",
)
plt.show()

# %%
fig = kimera_eval.plot_traj_colormap_rpe(
    rpe_rot,
    traj_ref,
    traj_est,
    plot_title="VIO Trajectory Tracking - Color Coded by RRE",
)
plt.show()

# %% [markdown]
# ## Trajectory Plotting
#
# Plot the aligned and associated estimated trajectory against ground truth.

# %%
# Plot the trajectories for quick error visualization.

fig = plt.figure()
traj_by_label = {
    "estimate (unaligned)": traj_est_unaligned,
    "estimate (aligned)": traj_est,
    "reference": traj_ref,
}
evo.tools.plot.trajectories(
    fig,
    traj_by_label,
    evo.tools.plot.PlotMode.xyz,
    title="PIM Trajectory Tracking in 3D",
)
plt.show()

# %% [markdown]
# ## PIM Plotting
#
# Plot preintegrated-imu-measurement estimates of current state over time.
# This comes in as a trajectory.
# The plots of error serve to help visualize the error in pim values over time.
#
# Note that these pim values are built off the backend's estimation,
# not off of ground truth.

# %%
pim_filename = os.path.join(
    os.path.expandvars(vio_output_dir), "output_pim_navstates.csv"
)
pim_df = pd.read_csv(pim_filename, sep=",", index_col=0)
pim_df = pim_df.iloc[1:]  # first row is logged as zero, remove
neval.rename_pim_df(pim_df)

gt_df = pd.read_csv(gt_data_file, sep=",", index_col=0)
gt_df = gt_df[~gt_df.index.duplicated()]
# TODO: Add relative angle comparison plot between IMU and mono RANSAC

# %%
# Convert the gt relative-pose DataFrame to a trajectory object.
traj_ref = kimera_eval.df_to_trajectory(gt_df)

# Use the mono ransac file as estimated trajectory.
# traj_est_unassociated = file_interface.read_swe_csv_trajectory(ransac_mono_filename)
traj_est_unaligned = kimera_eval.df_to_trajectory(pim_df)

# Associate the data.
traj_est = copy.deepcopy(traj_est_unaligned)
traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
traj_est = kimera_eval.align_trajectory(traj_est, traj_ref, correct_scale=False)

print("traj_ref: ", str(traj_ref))
print("traj_est: ", str(traj_est))

# %%
gt_vel = np.linalg.norm(np.column_stack((gt_df.vx, gt_df.vy, gt_df.vz)), axis=1)
est_vel = np.linalg.norm(np.column_stack((pim_df.vx, pim_df.vy, pim_df.vz)), axis=1)

fig = plt.figure()
plt.plot(gt_df.index, gt_vel, label="Ground Truth")
plt.plot(pim_df.index, est_vel, label="PIM Estimate")
plt.title("Velocity Norm")
plt.legend()
plt.show()

# %% [markdown]
# ### Relative Angles
# This plot compares the relative angles between GT and PIM outputs.

# %%
# Convert to relative traj
traj_ref_rel = kimera_eval.convert_abs_traj_to_rel_traj(traj_ref, up_to_scale=False)
traj_est_rel = kimera_eval.convert_abs_traj_to_rel_traj(traj_est, up_to_scale=False)

# Plot the PIM angles
PIM_angles = []
PIM_angles_timestamps = []
for i in range(len(traj_est_rel._poses_se3)):
    PIM_angles_timestamps.append(traj_est_rel.timestamps[i])
    # quaternion to axisangle
    rotm = traj_est_rel._poses_se3[i][0:3, 0:3]
    r = R.from_matrix(rotm)

    rot_vec = r.as_rotvec()
    PIM_angles.append(np.linalg.norm(rot_vec))


# Plot the GT angles
gt_angles = []
gt_angles_timestamps = []
for i in range(len(traj_ref_rel._poses_se3)):
    gt_angles_timestamps.append(traj_ref_rel.timestamps[i])
    # rotation matrix to axisangle
    rotm = traj_ref_rel._poses_se3[i][0:3, 0:3]
    r = R.from_matrix(rotm)

    rot_vec = r.as_rotvec()
    gt_angles.append(np.linalg.norm(rot_vec))


plt.figure()
plt.plot(PIM_angles_timestamps, PIM_angles, "r", label="PIM")
plt.plot(gt_angles_timestamps, gt_angles, "b", label="GT")
plt.legend(loc="upper right")
ax = plt.gca()
ax.set_xlabel("Timestamps")
ax.set_ylabel("Relative Angles [rad]")

plt.show()

# %%
# Get RPE for entire relative trajectory.
rpe_rot = kimera_eval.get_rpe_rot((traj_ref, traj_est))
rpe_tran = kimera_eval.get_rpe_trans((traj_ref, traj_est))

# %%
# Plot RPE of trajectory rotation and translation parts.
seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps[1:]]

kimera_eval.plot_metric(rpe_rot, "PIM RRE in Degrees")
kimera_eval.plot_metric(rpe_tran, "PIM RTE in Meters")
plt.show()

# %%
fig = kimera_eval.plot_traj_colormap_rpe(
    rpe_rot,
    traj_ref,
    traj_est,
    plot_title="PIM Trajectory Tracking - Color Coded by RRE",
)
plt.show()

# %% [markdown]
# # Smart Factors
# Plot smart factors vs. time

# %%
output_sf_filename = os.path.join(
    os.path.expandvars(vio_output_dir), "output_smartFactors.csv"
)
output_sf_df = pd.read_csv(output_sf_filename, sep=",", index_col=0)
fig = plt.figure()

# plt.plot(output_sf_df.timestamp_kf / 1e9, output_sf_df.numSF, label='numSF')
plt.plot(output_sf_df.timestamp_kf / 1e9, output_sf_df.numValid, label="numValid")

if np.any(output_sf_df.numDegenerate > 0):
    plt.plot(
        output_sf_df.timestamp_kf / 1e9,
        output_sf_df.numDegenerate,
        label="numDegenerate",
    )
if np.any(output_sf_df.numFarPoints > 0):
    plt.plot(
        output_sf_df.timestamp_kf / 1e9, output_sf_df.numFarPoints, label="numFarPoints"
    )
if np.any(output_sf_df.numNonInitialized > 0):
    plt.plot(
        output_sf_df.timestamp_kf / 1e9,
        output_sf_df.numNonInitialized,
        label="numNonInitialized",
    )
if np.any(output_sf_df.numCheirality > 0):
    plt.plot(
        output_sf_df.timestamp_kf / 1e9,
        output_sf_df.numCheirality,
        label="numCheirality",
    )
if np.any(output_sf_df.numOutliers > 0):
    plt.plot(
        output_sf_df.timestamp_kf / 1e9, output_sf_df.numOutliers, label="numOutliers"
    )

plt.ylabel("Valid Smart Factors")
plt.xlabel("Timestamps")
plt.legend(ncol=6)
plt.gcf().set_size_inches([9, 6])
plt.tight_layout()
plt.show()


# %% [markdown]
# # Biases
# Plot biases of gyro and accelerometer

# %%
fig1 = plt.figure()
plt.plot(output_poses_df.index, output_poses_df.bgx, label="bgx")
plt.plot(output_poses_df.index, output_poses_df.bgy, label="bgy")
plt.plot(output_poses_df.index, output_poses_df.bgz, label="bgz")
plt.ylabel("Gyro Biases")
plt.xlabel("Timestamps")
plt.legend()

fig2 = plt.figure()
plt.plot(output_poses_df.index, output_poses_df.bax, label="bax")
plt.plot(output_poses_df.index, output_poses_df.bay, label="bay")
plt.plot(output_poses_df.index, output_poses_df.baz, label="baz")
plt.ylabel("Acceleration Biases")
plt.xlabel("Timestamps")
plt.legend()
plt.show()


# %% [markdown]
# # External Odometry
# Plot external odometry trajectory against ground truth.

# %%
gt_df = pd.read_csv(gt_data_file, sep=",", index_col=0)
gt_df = gt_df[~gt_df.index.duplicated()]

output_external_odom_filename = os.path.join(
    os.path.expandvars(vio_output_dir), "output_backend_external_odometry.csv"
)
output_external_odom_df = pd.read_csv(
    output_external_odom_filename, sep=",", index_col=0
)

# Convert the gt relative-pose DataFrame to a trajectory object.
traj_ref_complete = kimera_eval.df_to_trajectory(gt_df)

# Use the backend poses as trajectory.
traj_est_unaligned_rel = kimera_eval.df_to_trajectory(output_external_odom_df)

# Convert to absolute trajectory by concatenating all between poses
traj_est_unaligned = kimera_eval.convert_rel_traj_to_abs_traj(traj_est_unaligned_rel)
discard_n_start_poses = 0
discard_n_end_poses = 0

# Associate the data.
traj_est = copy.deepcopy(traj_est_unaligned)
traj_ref, traj_est = sync.associate_trajectories(traj_ref_complete, traj_est)
traj_est = kimera_eval.align_trajectory(
    traj_est,
    traj_ref,
    correct_scale=False,
    discard_n_start_poses=int(discard_n_start_poses),
    discard_n_end_poses=int(discard_n_end_poses),
)

print("traj_ref: ", str(traj_ref))
print("traj_est: ", str(traj_est))

# %%
# plot ground truth trajectory with pose
plot_mode = evo.tools.plot.PlotMode.xy
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)
neval.draw_trajectory_poses(ax, traj_ref_complete, marker_scale=2, plot_mode=plot_mode)
neval.draw_start_and_end(ax, traj_ref_complete, plot_mode)
evo.tools.plot.traj(ax, plot_mode, traj_ref_complete, "--", "gray", "reference")
plt.title("Reference trajectory with pose")
plt.show()

# plot unaligned trajectory with pose
plot_mode = evo.tools.plot.PlotMode.xy
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)
neval.draw_trajectory_poses(
    ax, traj_est_unaligned, marker_scale=0.3, plot_mode=plot_mode
)
neval.draw_start_and_end(ax, traj_est_unaligned, plot_mode)
evo.tools.plot.traj(ax, plot_mode, traj_est_unaligned, "--", "gray", "reference")
plt.title("External odometry estimated trajectory with pose")
plt.show()

plot_mode = evo.tools.plot.PlotMode.xyz
fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)

gt_df_downsampled = gt_df.iloc[:1200:100]

# reference trajectory
traj_ref_downsampled = kimera_eval.df_to_trajectory(gt_df_downsampled)
neval.draw_trajectory_poses(ax, traj_ref, plot_mode=plot_mode, marker_scale=3)
neval.draw_trajectory_poses(ax, traj_est, plot_mode=plot_mode, marker_scale=3)
evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "reference")
evo.tools.plot.traj(ax, plot_mode, traj_est, "--", "green", "estimate (aligned)")

plt.title("Trajectory with pose")
plt.show()


# %%
# Plot APE of trajectory rotation and translation parts.
num_of_poses = traj_est.num_poses
traj_est.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)
traj_ref.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)

seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]

ape_tran = kimera_eval.get_ape_trans((traj_ref, traj_est))
fig1 = kimera_eval.plot_metric(ape_tran, "External Odometry ATE in Meters")
plt.show()

# Plot the ground truth and estimated trajectories against each other with APE overlaid.
fig2 = kimera_eval.plot_traj_colormap_ape(
    ape_tran,
    traj_ref,
    traj_est,
    plot_title="External Odometry Trajectory Tracking - Color Coded by ATE",
)
plt.show()

# Plot ARE

ape_rot = kimera_eval.get_ape_rot((traj_ref, traj_est))
fig3 = kimera_eval.plot_metric(ape_rot, "External Odometry ARE in Degrees")
plt.show()

# Plot the ground truth and estimated trajectories against each other with APE overlaid.
fig4 = kimera_eval.plot_traj_colormap_ape(
    ape_rot,
    traj_ref,
    traj_est,
    plot_title="External Odometry Trajectory Tracking - Color Coded by ARE",
)
plt.show()
