# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.8.1
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# %% [markdown]
# # Plot Loop-Closure-Detection
#
# Plots statistics on loop closure detection as well as optimized trajectory RPE, APE and trajectory against ground truth.

# %%
import yaml
import os
import copy
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    log.addHandler(ch)

from evo.tools import file_interface
from evo.tools import plot
from evo.tools import pandas_bridge
from evo.tools.settings import SETTINGS

from evo.core import sync
from evo.core import trajectory
from evo.core import metrics
from evo.core import transformations
from evo.core import lie_algebra as lie

from evaluation.evaluation_lib import (
    get_ape_trans,
    get_ape_rot,
    get_rpe_trans,
    get_rpe_rot,
    plot_metric,
    convert_rel_traj_from_body_to_cam,
)

# %matplotlib inline
# # %matplotlib notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %% [markdown]
# ## Data Locations
#
# Make sure to set the following paths.
#
# `vio_output_dir` is the path to the directory containing `output_*.csv` files obtained from logging a run of SparkVio.
#
# `gt_data_file` is the absolute path to the `csv` file containing ground truth data for the absolute pose at each timestamp of the dataset.

# %%
# Define directory to VIO output csv files as well as ground truth absolute poses.
vio_output_dir = ""
gt_data_file = vio_output_dir + "traj_gt.csv"
left_cam_calibration_file = ""

# %%
# Load calibration data
with open(left_cam_calibration_file) as f:
    f.readline()  # skip first line
    left_calibration_data = yaml.safe_load(f)
    body_T_leftCam = np.reshape(np.array(left_calibration_data["T_BS"]["data"]), (4, 4))
    print("Left cam calibration matrix: ")
    print(body_T_leftCam)


# %%
def downsize_lc_df(df):
    """Remove all entries from a pandas DataFrame object that have '0' for the timestamp, which
    includes all entries that do not have loop closures. Returns this cleaned DataFrame.

    Args:
        df: A pandas.DataFrame object representing loop-closure detections, indexed by timestamp.

    Returns:
        A pandas.DataFrame object with only loop closure entries.
    """
    df = df[~df.index.duplicated()]
    ts = np.array(df.index.tolist())
    good_ts = ts[np.where(ts > 0)]
    res = df.reindex(index=good_ts)
    return res


def downsize_lc_result_df(df):
    """Same as downsize_lc_df but checks the `isloop` field of the DataFrame for the LCD result 
    DataFrame, instead of the timestamp.
    """
    df = df[~df.index.duplicated()]
    ts = np.array(df.index.tolist())
    isloop = np.array(df.isLoop.tolist())
    good_ts = ts[np.where(isloop == 1)]
    res = df.reindex(index=good_ts)
    return res


def closest_num(ls, query):
    ls_len = len(ls)
    return ls[min(range(ls_len), key = lambda i: abs(ls[i]-query))]


def get_gt_rel_pose(gt_df, match_ts, query_ts, to_scale=True):
    """Returns the relative pose from match to query for given timestamps.
    
    Args:
        gt_df: A pandas.DataFrame object with timestamps as indices containing, at a minimum,
            columns representing the xyz position and wxyz quaternion-rotation at each
            timestamp, corresponding to the absolute pose at that time.
        match_ts: An integer representing the match frame timestamp.
        query_ts: An integer representing the query frame timestamp.
        to_scale: A boolean. If set to False, relative poses will have their translation
            part normalized.
    Returns:
        A 4x4 numpy array representing the relative pose from match to query frame.
    """
    w_T_bmatch = None
    w_T_bquery = None
    
    try:
        closest_ts = closest_num(gt_df.index, match_ts)
        if closest_ts != match_ts:
#             print("using closest match for timestamps")
            pass

        w_t_bmatch = np.array([gt_df.at[closest_ts, idx] for idx in ["x", "y", "z"]])
        w_q_bmatch = np.array(
            [gt_df.at[closest_ts, idx] for idx in ["qw", "qx", "qy", "qz"]]
        )
        w_T_bmatch = transformations.quaternion_matrix(w_q_bmatch)
        w_T_bmatch[:3, 3] = w_t_bmatch
    except:
        print(
            "Failed to convert an abs pose to a rel pose. Timestamp ",
            match_ts,
            " is not available in ground truth df.",
        )
        return None
        
    try:
        closest_ts = closest_num(gt_df.index, query_ts)
        if closest_ts != query_ts:
#             print("using closest match for timestamps")
            pass

        w_t_bquery = np.array([gt_df.at[closest_ts, idx] for idx in ["x", "y", "z"]])
        w_q_bquery = np.array(
            [gt_df.at[closest_ts, idx] for idx in ["qw", "qx", "qy", "qz"]]
        )
        w_T_bquery = transformations.quaternion_matrix(w_q_bquery)
        w_T_bquery[:3, 3] = w_t_bquery
    except:
        print(
            "Failed to convert an abs pose to a rel pose. Timestamp ",
            query_ts,
            " is not available in ground truth df.",
        )
        return None
        
    bmatch_T_bquery = lie.relative_se3(w_T_bmatch, w_T_bquery)
    bmatch_t_bquery = bmatch_T_bquery[:3, 3]

    if not to_scale:
        norm = np.linalg.norm(bmatch_t_bquery)
        if norm > 1e-6:
            bmatch_t_bquery = bmatch_t_bquery / np.linalg.norm(bmatch_t_bquery)
            
    bmatch_T_bquery[:3, 3] = bmatch_t_bquery
    
    return bmatch_T_bquery


def convert_abs_traj_to_rel_traj_lcd(df, lcd_df, to_scale=True):
    """Converts an absolute-pose trajectory to a relative-pose trajectory.

    The incoming DataFrame df is processed element-wise. At each kf timestamp (which is the
    index of the DataFrame row) starting from the second (index 1), the relative pose
    from the match timestamp to the query stamp is calculated (in the match-
    timestamp's coordinate frame). This relative pose is then appended to the
    resulting DataFrame.
    The resulting DataFrame has timestamp indices corresponding to poses that represent
    the relative transformation between the match timestamp and the query one.

    Args:
        df: A pandas.DataFrame object with timestamps as indices containing, at a minimum,
            columns representing the xyz position and wxyz quaternion-rotation at each
            timestamp, corresponding to the absolute pose at that time.
        lcd_df: A pandas.DataFrame object with timestamps as indices containing, at a minimum,
            columns representing the timestamp of query frames and the timestamps of the
            match frames.
        to_scale: A boolean. If set to False, relative poses will have their translation
            part normalized.

    Returns:
        A pandas.DataFrame object with xyz position and wxyz quaternion fields for the
        relative pose trajectory corresponding to the absolute one given in 'df', and
        relative by the given match and query timestamps.
    """
    rows_list = []
    index_list = []

    for i in range(len(lcd_df.index)):
        match_ts = lcd_df.timestamp_match[lcd_df.index[i]]
        query_ts = lcd_df.timestamp_query[lcd_df.index[i]]
        
        if match_ts == 0 and query_ts == 0:
            continue

        bi_T_bidelta = get_gt_rel_pose(df, match_ts, query_ts, to_scale)
        
        if bi_T_bidelta is not None:
            bi_R_bidelta = copy.deepcopy(bi_T_bidelta)
            bi_R_bidelta[:, 3] = np.array([0, 0, 0, 1])
            bi_q_bidelta = transformations.quaternion_from_matrix(bi_R_bidelta)
            bi_t_bidelta = bi_T_bidelta[:3, 3]

            new_row = {
                "x": bi_t_bidelta[0],
                "y": bi_t_bidelta[1],
                "z": bi_t_bidelta[2],
                "qw": bi_q_bidelta[0],
                "qx": bi_q_bidelta[1],
                "qy": bi_q_bidelta[2],
                "qz": bi_q_bidelta[3],
            }
            rows_list.append(new_row)
            index_list.append(lcd_df.index[i])

    return pd.DataFrame(data=rows_list, index=index_list)


def rename_euroc_gt_df(df):
    """Renames a DataFrame built from a EuRoC ground-truth data csv file to be easier to read.

    Column labels are changed to be more readable and to be identical to the generic pose
    trajectory format used with other csv files. Note that '#timestamp' will not actually
    be renamed if it is the index of the DataFrame (which it should be). It will be
    appropriately renamed if it is the index name.
    This operation is 'inplace': It does not return a new DataFrame but simply changes
    the existing one.

    Args:
        df: A pandas.DataFrame object.
    """
    df.index.names = ["timestamp"]
    df.rename(
        columns={
            " p_RS_R_x [m]": "x",
            " p_RS_R_y [m]": "y",
            " p_RS_R_z [m]": "z",
            " q_RS_w []": "qw",
            " q_RS_x []": "qx",
            " q_RS_y []": "qy",
            " q_RS_z []": "qz",
            " v_RS_R_x [m s^-1]": "vx",
            " v_RS_R_y [m s^-1]": "vy",
            " v_RS_R_z [m s^-1]": "vz",
            " b_w_RS_S_x [rad s^-1]": "bgx",
            " b_w_RS_S_y [rad s^-1]": "bgy",
            " b_w_RS_S_z [rad s^-1]": "bgz",
            " b_a_RS_S_x [m s^-2]": "bax",
            " b_a_RS_S_y [m s^-2]": "bay",
            " b_a_RS_S_z [m s^-2]": "baz",
        },
        inplace=True,
    )


def rename_lcd_result_df(df):
    """Renames a DataFrame built from an LCD results measurements csv file to be converted to a trajectory.

    This is an 'inplace' argument and returns nothing.

    Args:
        df: A pandas.DataFrame object.
    """
    df.index.names = ["timestamp"]
    df.rename(columns={
        "#timestamp_match": "timestamp_match",
        "px": "x",
        "py": "y",
        "pz": "z",
    }, inplace=True)


# %% [markdown]
# ## LoopClosureDetector Statistics Plotting
#
# Gather and plot various statistics on LCD module performance, including RANSAC information, keyframe status (w.r.t. loop closure detection), and loop closure events and the quality of their relative poses.

# %% [markdown]
# ### LCD Status Frequency Chart
#
# Each keyframe is processed for potential loop closures. During this process, the loop-closure detector can either identify a loop closure or not. There are several reasons why a loop closure would not be detected. This plot helps to identify why loop closures are not detected between keyframes.

# %%
output_lcd_status_filename = os.path.join(
    os.path.expandvars(vio_output_dir), "output_lcd_status.csv"
)
lcd_debuginfo_df = pd.read_csv(output_lcd_status_filename, sep=",", index_col=0)

status_freq_map = {}

for status in lcd_debuginfo_df.lcd_status:
    if status not in status_freq_map:
        status_freq_map[status] = 1
    else:
        status_freq_map[status] += 1


print(
    "Full Size of PGO:                       ", lcd_debuginfo_df.pgo_size.tolist()[-1]
)

# Print the overall number of loop closures detected over all time.
if "LOOP_DETECTED" in status_freq_map:
    print("Loop Closures Detected:                 ", status_freq_map["LOOP_DETECTED"])
else:
    print("Loop Closures Detected:                 ", 0)

print(
    "Loop Closures Registered by PGO by End: ",
    lcd_debuginfo_df.pgo_lc_count.tolist()[-1],
)
print(
    "Loop Closures Accepted by PGO at End:   ",
    lcd_debuginfo_df.pgo_lc_inliers.tolist()[-1],
)

# Plot failure modes as a histogram.
fig = plt.figure(figsize=(18, 10))
plt.bar(status_freq_map.keys(), status_freq_map.values(), width=1.0)

plt.xticks(status_freq_map.keys(), list(status_freq_map.keys()))
plt.ylabel("Status Frequency")
plt.title("LoopClosureDetector Status Histogram")

plt.show()

# %% [markdown]
# ### LCD RANSAC Performance Charts
#
# Plot the performance of the geometric-verification and pose-recovery steps. These are handled by Nister (5pt) RANSAC and Arun (3pt) RANSAC respectively.
#
# inlier percentages and iterations are plotted for both methods.

# %%
lcd_debuginfo_small_df = downsize_lc_df(lcd_debuginfo_df)

# Helper functions for processing data summary.
def get_mean(attrib):
    ls = lcd_debuginfo_small_df[attrib].tolist()
    return float(sum(ls)) / len(ls)


def get_min(attrib):
    return min(lcd_debuginfo_small_df[attrib])


def get_max(attrib):
    return max(lcd_debuginfo_small_df[attrib])


# Construct and visualize summary. TODO(marcus): use a LaTeX table.
summary_stats = [
    ("Average number of mono ransac inliers", get_mean("mono_inliers")),
    ("Average size of mono ransac input", get_mean("mono_input_size")),
    ("Average number of stereo ransac inliers", get_mean("stereo_inliers")),
    ("Average size of stereo ransac input", get_mean("stereo_input_size")),
    ("Maximum mono ransac iterations", get_max("mono_iters")),
    ("Maximum stereo ransac iterations", get_max("stereo_iters")),
]

attrib_len = [len(attrib[0]) for attrib in summary_stats]
max_attrib_len = max(attrib_len)

print("\nRANSAC Statistic Summary for Loop Closures ONLY:\n")
for entry in summary_stats:
    attrib = entry[0]
    value = entry[1]
    spacing = max_attrib_len - len(attrib)
    print(attrib + " " * spacing + ": " + str(value))


# Plot ransac inlier and iteration statistics.
fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), squeeze=False)

lcd_debuginfo_small_df.plot(kind="line", y="mono_inliers", ax=axes1[0, 0])
lcd_debuginfo_small_df.plot(kind="line", y="mono_input_size", ax=axes1[0, 0])
lcd_debuginfo_small_df.plot(kind="line", y="mono_iters", ax=axes1[0, 0])

lcd_debuginfo_small_df.plot(kind="line", y="stereo_inliers", ax=axes1[0, 1])
lcd_debuginfo_small_df.plot(kind="line", y="stereo_input_size", ax=axes1[0, 1])
lcd_debuginfo_small_df.plot(kind="line", y="stereo_iters", ax=axes1[0, 1])

plt.show()

# %% [markdown]
# ### Geometric Verification (2d2d RANSAC) Error Plotting
#
# Calculate error statistics for the 2d2d RANSAC pose estimate for all loop closure candidates that make it to the geometric verification check step.
#
# The first plot is the relative angles for the 2d2d ransac and the ground truth. You want the two plots to be very similar; any disparity is error.
#
# The second plot only has one line and is the norm of the error in position from the 2d2d ransac result to the ground truth. You want this as close to zero as possible all the way through.

# %%
gt_df = pd.read_csv(gt_data_file, sep=",", index_col=0)  # Absolute gt in body frame
rename_euroc_gt_df(gt_df)  # some pre-processing for euroc only (doesn't affect non-euroc)

# Get 2d2d ransac results (camera frame, relative, only for LC candidates) as dataframe
output_loop_closures_filename = os.path.join(os.path.expandvars(vio_output_dir), "output_lcd_geom_verif.csv")
lcd_2d2d_df = pd.read_csv(output_loop_closures_filename, sep=",")
rename_lcd_result_df(lcd_2d2d_df)

# Build trajectory objects
traj_est_rel = pandas_bridge.df_to_trajectory(lcd_2d2d_df)
ref_rel_df = convert_abs_traj_to_rel_traj_lcd(gt_df, lcd_2d2d_df, True)  # keep scale and normalize later
traj_ref_rel = pandas_bridge.df_to_trajectory(ref_rel_df)
traj_ref_cam_rel = convert_rel_traj_from_body_to_cam(traj_ref_rel, body_T_leftCam)

print("traj_ref_rel: ", str(traj_ref_rel))
print("traj_ref_cam_rel: ", str(traj_ref_cam_rel))
print("traj_est_rel: ", str(traj_est_rel))

# %%
# Plot rotation part error
est_angles = []
gt_angles = []
gt_angles_timestamps = []
rot_errors = []

assert(len(traj_est_rel.poses_se3) == len(traj_ref_cam_rel.poses_se3))
for i in range(len(traj_est_rel.poses_se3)):
    est_rot = R.from_dcm(traj_est_rel.poses_se3[i][:3,:3])
    gt_rot = R.from_dcm(traj_ref_cam_rel.poses_se3[i][:3,:3])
    
    est_angles.append(np.linalg.norm(est_rot.as_rotvec()))
    gt_angles.append(np.linalg.norm(gt_rot.as_rotvec()))
    gt_angles_timestamps.append(traj_ref_cam_rel.timestamps[i])
    
    error = gt_rot.inv() * est_rot
    error_angle = np.linalg.norm(error.as_rotvec())
    rot_errors.append(error_angle)

plt.figure(figsize=(18, 10))
# plt.plot(gt_angles_timestamps, np.rad2deg(gt_angles), "b", label="GT")
# plt.plot(gt_angles_timestamps, np.rad2deg(est_angles), "g", label="Est")
plt.plot(gt_angles_timestamps, np.rad2deg(rot_errors), "r", label="Error")
plt.legend(loc="upper right")
ax = plt.gca()
ax.set_xlabel("Timestamps [-]")
ax.set_ylabel("Rotation Part Error [deg]")
ax.set_title("Geometric Verification (2d2d Ransac) Rotation Part Error")

plt.show()

# %%
# calculate the translation errors up-to-scale
trans_errors = []
for i in range(len(traj_ref_cam_rel.timestamps)):

    # normalized translation vector from gt
    t_ref = traj_ref_cam_rel.poses_se3[i][0:3, 3]
    if np.linalg.norm(t_ref) > 1e-6:
        t_ref /= np.linalg.norm(t_ref)

    # normalized translation vector from mono ransac
    t_est = traj_est_rel.poses_se3[i][0:3, 3]
    if np.linalg.norm(t_est) > 1e-6:
        t_est /= np.linalg.norm(t_est)

    # calculate error (up to scale, equivalent to the angle between the two translation vectors)
    trans_errors.append(np.linalg.norm(t_ref - t_est))

plt.figure(figsize=(18, 10))
plt.plot(traj_ref_cam_rel.timestamps, trans_errors, "r", label="Error")
plt.legend(loc="upper right")
ax = plt.gca()
ax.set_xlabel("Timestamps [-]")
ax.set_ylabel("Translation Part Error (up-to-scale)")
ax.set_title("Geometric Verification (2d2d Ransac) Translation Part Errors")

plt.show()

# %% [markdown]
# ### Pose Recovery (3d3d or 2d3d RANSAC) Error Plotting
#
# Same as the previous section, but for final pose recovery. These are the loop closure relative poses that are passed to the PGO, if they pass the check. They're obtained via 3d3d ransac or PnP (2d3d) depending on user selection.

# %%
gt_df = pd.read_csv(gt_data_file, sep=",", index_col=0)
rename_euroc_gt_df(gt_df)

# Get 3d3d or 2d3d ransac results (camera frame, relative, only for LC candidates) as dataframe
output_loop_closures_filename = os.path.join(os.path.expandvars(vio_output_dir), "output_lcd_pose_recovery.csv")
lcd_3d3d_df = pd.read_csv(output_loop_closures_filename, sep=",")
rename_lcd_result_df(lcd_3d3d_df)

# Build trajectory objects
traj_est_rel = pandas_bridge.df_to_trajectory(lcd_3d3d_df)
ref_rel_df = convert_abs_traj_to_rel_traj_lcd(gt_df, lcd_3d3d_df, True)
traj_ref_rel = pandas_bridge.df_to_trajectory(ref_rel_df)
traj_ref_cam_rel = convert_rel_traj_from_body_to_cam(traj_ref_rel, body_T_leftCam)

print("traj_ref_rel: ", str(traj_ref_rel))
print("traj_ref_cam_rel: ", str(traj_ref_cam_rel))
print("traj_est_rel: ", str(traj_est_rel))

# %%
# Plot rotation part error
est_angles = []
gt_angles = []
gt_angles_timestamps = []
rot_errors = []

assert(len(traj_est_rel.poses_se3) == len(traj_ref_cam_rel.poses_se3))
for i in range(len(traj_est_rel.poses_se3)):
    est_rot = R.from_dcm(traj_est_rel.poses_se3[i][:3,:3])
    gt_rot = R.from_dcm(traj_ref_cam_rel.poses_se3[i][:3,:3])
    
    est_angles.append(np.linalg.norm(est_rot.as_rotvec()))
    gt_angles.append(np.linalg.norm(gt_rot.as_rotvec()))
    gt_angles_timestamps.append(traj_ref_cam_rel.timestamps[i])
    
    error = gt_rot.inv() * est_rot
    error_angle = np.linalg.norm(error.as_rotvec())
    rot_errors.append(error_angle)

plt.figure(figsize=(18, 10))
# plt.plot(gt_angles_timestamps, np.rad2deg(gt_angles), "b", label="GT")
# plt.plot(gt_angles_timestamps, np.rad2deg(est_angles), "g", label="Est")
plt.plot(gt_angles_timestamps, np.rad2deg(rot_errors), "r", label="Error")
plt.legend(loc="upper right")
ax = plt.gca()
ax.set_xlabel("Timestamps [-]")
ax.set_ylabel("Rotation Part Error [deg]")
ax.set_title("Pose Recovery (3d3d or 2d3d Ransac) Rotation Part Error")

plt.show()

# %%
# Get RPE for entire relative trajectory.
ape_rot = get_ape_rot((traj_ref_cam_rel, traj_est_rel))
ape_tran = get_ape_trans((traj_ref_cam_rel, traj_est_rel))

# calculate the translation errors
trans_errors = []
for i in range(len(traj_ref_cam_rel.timestamps)):
    t_ref = traj_ref_cam_rel.poses_se3[i][:3,3]
    t_est = traj_est_rel.poses_se3[i][:3,3]
    
    trans_errors.append(np.linalg.norm(t_ref - t_est))

plt.figure(figsize=(18, 10))
plt.plot(traj_ref_cam_rel.timestamps, trans_errors, "r", label="Error")
plt.legend(loc="upper right")
ax = plt.gca()
ax.set_xlabel("Timestamps [-]")
ax.set_ylabel("Translation Part Error [m]")
ax.set_title("Pose Recovery (3d3d or 2d3d Ransac) Translation Part Errors")

plt.show()

# %% [markdown]
# ## Loop Closure Error Plotting on Trajectory
#
# Visualize the loop closures directly on the GT and VIO trajectories, and color-code by error.
#
# We use the pose-recovery data because pose-recovery is the final step in the LCD process, before RPGO uses PCM and/or GNC to perform outlier-rejection. The poses obtained at this step are the final between-poses passed to the RPGO backend. We re-use the errors calculated in the last section for color-coding.

# %%
# Load ground truth and estimated data as csv DataFrames.
gt_df = pd.read_csv(gt_data_file, sep=",", index_col=0)
gt_df = gt_df[~gt_df.index.duplicated()]
rename_euroc_gt_df(gt_df)

# Load VIO trajectory
output_poses_filename = os.path.join(os.path.expandvars(vio_output_dir), "traj_vio.csv")
output_poses_df = pd.read_csv(output_poses_filename, sep=",", index_col=0)
traj_ref = pandas_bridge.df_to_trajectory(gt_df)

# %%
# Get coordinates for all LC lines to plot
xs = []
ys = []
for i in range(len(lcd_3d3d_df)):
    match_ts = lcd_3d3d_df.timestamp_match[i]
    query_ts = lcd_3d3d_df.timestamp_query[i]
    
    closest_ts = closest_num(gt_df.index, match_ts)
    w_t_bmatch_gt = np.array([gt_df.at[closest_ts, idx] for idx in ["x", "y", "z"]])
    
    closest_ts = closest_num(gt_df.index, query_ts)
    w_t_bquery_gt = np.array([gt_df.at[closest_ts, idx] for idx in ["x", "y", "z"]])
    
    xs.append([w_t_bquery_gt[0], w_t_bmatch_gt[0]])
    ys.append([w_t_bquery_gt[1], w_t_bmatch_gt[1]])

plot_mode = plot.PlotMode.xy
fig = plt.figure(figsize=(18, 10))
ax = plot.prepare_axis(fig, plot_mode)

# Get colors based on rotation error
err = np.rad2deg(rot_errors)
norm = mpl.colors.Normalize(vmin=min(err), vmax=max(err), clip=True)
mapper = cm.ScalarMappable(
    norm=norm,
    cmap=SETTINGS.plot_trajectory_cmap)  # cm.*_r is reversed cmap
mapper.set_array(err)
colors = [mapper.to_rgba(a) for a in err]
cbar = fig.colorbar(
    mapper, ticks=[min(err), (max(err) - (max(err) - min(err)) / 2), max(err)])
cbar.ax.set_yticklabels([
    "{0:0.3f}".format(min(err)),
    "{0:0.3f}".format(max(err) - (max(err) - min(err)) / 2),
    "{0:0.3f}".format(max(err))
])

# Plot the ground truth and estimated trajectories against each other with APE overlaid.
ax.set_title("Ground-Truth Trajectory with Loop Closures (Colored-Coded by Rotation Error in Deg)")
plot.traj(ax, plot_mode, traj_ref, "--", "gray", "reference")

# Plot accepted loop closures
assert(len(xs) == len(ys))
for i in range(len(xs)):
    x = xs[i]
    y = ys[i]
    color = colors[i]
    plt.plot(x, y, color=color)

ax.legend()
plt.show()

# %% [markdown]
# ## LoopClosureDetector PGO-Optimized Trajectory Plotting
#
# Plot the APE, RPE, and trajectory of the Pose-graph-optimized trajectory, including loop closures on top of regular odometry updates.
#
# The results are visualized against both ground truth and the odometry-estimate alone to show the performance gain from loop closure detection.

# %%
# Load ground truth and estimated data as csv DataFrames.
gt_df = pd.read_csv(gt_data_file, sep=",", index_col=0)

output_poses_filename = os.path.join(os.path.expandvars(vio_output_dir), "traj_vio.csv")
output_poses_df = pd.read_csv(output_poses_filename, sep=",", index_col=0)

output_pgo_poses_filename = os.path.join(os.path.expandvars(vio_output_dir), "traj_pgo.csv")
output_pgo_poses_df = pd.read_csv(output_pgo_poses_filename, sep=",", index_col=0)

# %%
gt_df = gt_df[~gt_df.index.duplicated()]
rename_euroc_gt_df(gt_df)

# %%
discard_n_start_poses = 10
discard_n_end_poses = 10

# Convert the gt relative-pose DataFrame to a trajectory object.
traj_ref = pandas_bridge.df_to_trajectory(gt_df)

# Compare against the VIO without PGO.
traj_ref_cp = copy.deepcopy(traj_ref)
traj_vio = pandas_bridge.df_to_trajectory(output_poses_df)
traj_ref_cp, traj_vio = sync.associate_trajectories(traj_ref_cp, traj_vio)
traj_vio = trajectory.align_trajectory(
    traj_vio,
    traj_ref_cp,
    correct_scale=False,
    discard_n_start_poses=int(discard_n_start_poses),
    discard_n_end_poses=int(discard_n_end_poses),
)

# Use the PGO output as estimated trajectory.
traj_est = pandas_bridge.df_to_trajectory(output_pgo_poses_df)

# Associate the data.
traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
traj_est = trajectory.align_trajectory(
    traj_est,
    traj_ref,
    correct_scale=False,
    discard_n_start_poses=int(discard_n_start_poses),
    discard_n_end_poses=int(discard_n_end_poses),
)

print("traj_ref: ", str(traj_ref))
print("traj_vio: ", str(traj_vio))
print("traj_est: ", str(traj_est))

# %% [markdown]
# ## Absolute-Pose-Error Plotting
#
# Plot absolute-pose-error along the entire trajectory. APE gives a good sense of overall VIO performance across the entire trajectory.

# %%
# Plot APE of trajectory rotation and translation parts.
num_of_poses = traj_est.num_poses

traj_ref.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)
traj_ref_cp.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)
traj_vio.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)
traj_est.reduce_to_ids(
    range(int(discard_n_start_poses), int(num_of_poses - discard_n_end_poses), 1)
)

# %%
ape_rot_vio = get_ape_rot((traj_ref_cp, traj_vio))
ape_tran_vio = get_ape_trans((traj_ref_cp, traj_vio))
plot_metric(ape_rot_vio, "VIO ARE in Degrees", figsize=(18, 10))
plot_metric(ape_tran_vio, "VIO ATE in Meters", figsize=(18, 10))
plt.show()

# %%
ape_rot_pgo = get_ape_rot((traj_ref, traj_est))
ape_tran_pgo = get_ape_trans((traj_ref, traj_est))
plot_metric(ape_rot_pgo, "VIO+PGO ARE in Degrees", figsize=(18, 10))
plot_metric(ape_tran_pgo, "VIO+PGO ATE in Meters", figsize=(18, 10))
plt.show()

# %%
# Plot the ground truth and estimated trajectories against each other with APE overlaid.
plot_mode = plot.PlotMode.xy
fig = plt.figure(figsize=(18, 10))
ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref, "--", "gray", "reference")
# plot.traj(ax, plot_mode, traj_vio, ".", "gray", "vio without pgo")
plot.traj_colormap(
    ax,
    traj_est,
    ape_tran_pgo.error,
    plot_mode,
    min_map=ape_tran_pgo.get_all_statistics()["min"],
    max_map=ape_tran_pgo.get_all_statistics()["max"],
    title="VIO+PGO Trajectory Tracking - Color Coded by ATE",
)
ax.legend()
plt.show()

# %% [markdown]
# ## Relative-Pose-Error Plotting
#
# Plot relative-pose-error along the entire trajectory. RPE gives a good sense of overall VIO performance from one frame to the next.

# %%
# Get RPE for entire relative trajectory.
rpe_rot_vio = get_rpe_rot((traj_ref_cp, traj_vio))
rpe_tran_vio = get_rpe_trans((traj_ref_cp, traj_vio))

rpe_rot_pgo = get_rpe_rot((traj_ref, traj_est))
rpe_tran_pgo = get_rpe_trans((traj_ref, traj_est))

# %%
# Plot RPE of trajectory rotation and translation parts.
plot_metric(rpe_rot_vio, "VIO RRE in Degrees", figsize=(18, 10))
plot_metric(rpe_tran_vio, "VIO RTE in Meters", figsize=(18, 10))
plt.show()

# %%
# Plot RPE of trajectory rotation and translation parts.
plot_metric(rpe_rot_pgo, "VIO+PGO RRE in Degrees", figsize=(18, 10))
plot_metric(rpe_tran_pgo, "VIO+PGO RTE in Meters", figsize=(18, 10))
plt.show()

# %%
# important: restrict data to delta ids for plot.
traj_ref_plot = copy.deepcopy(traj_ref)
traj_est_plot = copy.deepcopy(traj_est)
traj_ref_plot.reduce_to_ids(rpe_rot_pgo.delta_ids)
traj_est_plot.reduce_to_ids(rpe_rot_pgo.delta_ids)

# Plot the ground truth and estimated trajectories against each other with RPE overlaid.

plot_mode = plot.PlotMode.xy
fig = plt.figure(figsize=(18, 10))
ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref_plot, "--", "gray", "reference")
plot.traj_colormap(
    ax,
    traj_est_plot,
    rpe_rot_pgo.error,
    plot_mode,
    min_map=rpe_rot_pgo.get_all_statistics()["min"],
    max_map=rpe_rot_pgo.get_all_statistics()["max"],
    title="VIO+PGO Trajectory Tracking - Color Coded by RRE",
)
ax.legend()
plt.show()

# %%
traj_vio = pandas_bridge.df_to_trajectory(output_poses_df)
traj_ref, traj_vio = sync.associate_trajectories(traj_ref, traj_est)
traj_vio = trajectory.align_trajectory(traj_vio, traj_ref, correct_scale=False)

# Plot the trajectories for quick error visualization.

fig = plt.figure(figsize=(18, 10))
traj_by_label = {"VIO only": traj_vio, "VIO + PGO": traj_est, "reference": traj_ref}
plot.trajectories(
    fig, traj_by_label, plot.PlotMode.xyz, title="PIM Trajectory Tracking in 3D"
)
plt.show()
