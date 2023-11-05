"""Main library for evaluation."""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

import evo.core.trajectory
import evo.core.lie_algebra as lie
import evo.core.metrics
import evo.tools.plot


DEFAULT_STAT_TYPES = [
    evo.core.metrics.StatisticsType.rmse,
    evo.core.metrics.StatisticsType.mean,
    evo.core.metrics.StatisticsType.median,
    evo.core.metrics.StatisticsType.std,
    evo.core.metrics.StatisticsType.min,
    evo.core.metrics.StatisticsType.max,
]


def get_ape_rot(data):
    """Return APE rotation metric for input data.

    Args:
        data: A 2-tuple containing the reference trajectory and the
            estimated trajectory as PoseTrajectory3D objects.

    Returns:
        A metrics object containing the desired results.
    """
    ape_rot = evo.core.metrics.APE(evo.core.metrics.PoseRelation.rotation_angle_deg)
    ape_rot.process_data(data)

    return ape_rot


def get_ape_trans(data):
    """Return APE translation metric for input data.

    Args:
        data: A 2-tuple containing the reference trajectory and the
            estimated trajectory as PoseTrajectory3D objects.

    Returns:
        A metrics object containing the desired results.
    """
    ape_trans = evo.core.metrics.APE(evo.core.metrics.PoseRelation.translation_part)
    ape_trans.process_data(data)

    return ape_trans


def get_rpe_rot(data):
    """Return RPE rotation metric for input data.

    Args:
        data: A 2-tuple containing the reference trajectory and the
            estimated trajectory as PoseTrajectory3D objects.

    Returns:
        A metrics object containing the desired results.
    """
    rpe_rot = evo.core.metrics.RPE(
        evo.core.metrics.PoseRelation.rotation_angle_deg,
        1.0,
        evo.core.metrics.Unit.frames,
        1.0,
        False,
    )
    rpe_rot.process_data(data)

    return rpe_rot


def get_rpe_trans(data):
    """Return RPE translation metric for input data.

    Args:
        data: A 2-tuple containing the reference trajectory and the
            estimated trajectory as PoseTrajectory3D objects.

    Returns:
        A metrics object containing the desired results.
    """
    rpe_trans = evo.core.metrics.RPE(
        evo.core.metrics.PoseRelation.translation_part,
        1.0,
        evo.core.metrics.Unit.frames,
        0.0,
        False,
    )
    rpe_trans.process_data(data)

    return rpe_trans


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


def df_to_trajectory(
    df,
    pos_names=("x", "y", "z"),
    quaternion_names=("qw", "qx", "qy", "qz"),
    time_column=None,
    time_conversion_factor=1.0e-9,
):
    """
    Convert a provided dataframe to an evo trajectory.

    Args:
        df (pd.DataFrame): dataframe to convert
        pos_names (Tuple[str, str, str]): names of columns to use for position
        quaternion_names (Tuple[str, str, str, str]): names of columns to use
            for orientation
        time_column (Optional[str]): column to use for timestamps. If not provided,
            index of the dataframe is used
        time_conversion_factor (float): scale factor to convert from native timestamp
            units to seconds (nanoseconds assumed)

    Returns:
        evo.core.trajectory.PoseTrajectory3D: trajectory generated from the dataframe
    """
    pos_xyz = df.loc[:, pos_names].to_numpy()
    quaternion_wxyz = df.loc[:, quaternion_names].to_numpy()
    if time_column is not None:
        timestamps = df[time_column].to_numpy()
    else:
        timestamps = df.index.to_numpy()
    return evo.core.trajectory.PoseTrajectory3D(
        positions_xyz=pos_xyz,
        orientations_quat_wxyz=quaternion_wxyz,
        timestamps=timestamps * time_conversion_factor,
    )


def align_trajectory(
    traj,
    traj_ref,
    correct_scale: bool = False,
    correct_only_scale: bool = False,
    n: int = -1,
    discard_n_start_poses=0,
    discard_n_end_poses=0,
):
    """
    Align two trajectories, optionally discarding starting and ending poses.

    Args:
        traj: Trajectory to align
        traj_ref: Reference trajectory to align to
        correct_scale: Correct the scale of traj
        correct_only_scale: Only correct the scale of traj
        n: Number of poses to use to compute alignment
        discard_n_start_poses: number of poses to discard at the start of the trajectory
        discard_n_end_poses: number of poses to discard at the end of the trajectory

    Returns:
        Copy of traj aligned to traj_ref
    """
    est_end = traj.num_poses - discard_n_end_poses
    traj_est = copy.deepcopy(traj)
    traj_est.reduce_to_ids(np.arange(discard_n_start_poses, est_end))

    ref_end = traj.num_poses - discard_n_end_poses
    traj_ref_reduced = copy.deepcopy(traj_ref)
    traj_ref_reduced.reduce_to_ids(np.arange(discard_n_start_poses, ref_end))

    r_a, t_a, s = traj_est.align(
        traj_ref_reduced,
        correct_scale=correct_scale,
        correct_only_scale=correct_only_scale,
        n=n,
    )

    traj_aligned = copy.deepcopy(traj)
    if correct_only_scale:
        traj_aligned.scale(s)
    elif correct_scale:
        traj_aligned.scale(s)
        traj_aligned.transform(lie.se3(r_a, t_a))
    else:
        traj_aligned.transform(lie.se3(r_a, t_a))

    return traj_aligned


def convert_abs_traj_to_rel_traj(traj, up_to_scale=False):
    """Convert an absolute-pose trajectory to a relative-pose trajectory.

    The incoming trajectory is processed element-wise. At each timestamp
    starting from the second (index 1), the relative pose
    from the previous timestamp to the current one is calculated (in the previous-
    timestamp's coordinate frame). This relative pose is then appended to the
    resulting trajectory.
    The resulting trajectory has timestamp indices corresponding to poses that represent
    the relative transformation between that timestamp and the **next** one.

    Args:
        traj: A PoseTrajectory3D object
        up_to_scale: If set to True, relative poses will have their translation
            part normalized.

    Returns:
        A PoseTrajectory3D object with xyz position and wxyz quaternion fields for the
        relative pose trajectory corresponding to the absolute one given in `traj`.
    """
    new_poses = []

    for i in range(1, len(traj.timestamps)):
        rel_pose = lie.relative_se3(traj.poses_se3[i - 1], traj.poses_se3[i])

        if up_to_scale:
            bim1_t_bi = rel_pose[:3, 3]
            norm = np.linalg.norm(bim1_t_bi)
            if norm > 1e-6:
                bim1_t_bi = bim1_t_bi / norm
                rel_pose[:3, 3] = bim1_t_bi

        new_poses.append(rel_pose)

    return evo.core.trajectory.PoseTrajectory3D(
        timestamps=traj.timestamps[1:], poses_se3=new_poses
    )


def convert_rel_traj_to_abs_traj(traj):
    """
    Convert a relative pose trajectory to an absolute-pose trajectory.

    The incoming trajectory is processed elemente-wise. Poses at each
    timestamp are appended to the absolute pose from the previous timestamp.

    Args:
        traj: A PoseTrajectory3D object

    Returns:
        A PoseTrajectory3D object with xyz position and wxyz quaternion fields for the
        relative pose trajectory corresponding to the relative one given in `traj`.
    """
    new_poses = [lie.se3()]  # origin at identity

    for i in range(0, len(traj.timestamps)):
        abs_pose = np.dot(new_poses[-1], traj.poses_se3[i])
        new_poses.append(abs_pose)

    return evo.core.trajectory.PoseTrajectory3D(
        timestamps=traj.timestamps[1:], poses_se3=new_poses
    )


def convert_rel_traj_from_body_to_cam(rel_traj, body_T_cam):
    """
    Convert a relative pose trajectory from body frame to camera frame.

    Args:
        rel_traj: Relative trajectory, a PoseTrajectory3D object containing timestamps
            and relative poses at each timestamp. It has to have the poses_se3 field.

        body_T_cam: The SE(3) transformation from camera from to body frame. Also known
            as camera extrinsics matrix.

    Returns:
        A PoseTrajectory3D object in camera frame
    """

    def assert_so3(R):
        assert np.isclose(np.linalg.det(R), 1, atol=1e-06)
        assert np.allclose(np.matmul(R, R.transpose()), np.eye(3), atol=1e-06)

    assert_so3(body_T_cam[0:3, 0:3])

    new_poses = []
    for i in range(len(rel_traj.timestamps)):
        im1_body_T_body_i = rel_traj.poses_se3[i]
        assert_so3(im1_body_T_body_i[0:3, 0:3])

        im1_cam_T_cam_i = np.matmul(
            np.matmul(np.linalg.inv(body_T_cam), im1_body_T_body_i), body_T_cam
        )

        assert_so3(np.linalg.inv(body_T_cam)[0:3, 0:3])
        assert_so3(im1_cam_T_cam_i[0:3, 0:3])

        new_poses.append(im1_cam_T_cam_i)

    return evo.core.trajectory.PoseTrajectory3D(
        timestamps=rel_traj.timestamps, poses_se3=new_poses
    )
