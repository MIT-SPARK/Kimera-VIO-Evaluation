"""Utilites for creating evo trajectories from Kimera-VIO data."""
import evo.core.trajectory
import evo.core.lie_algebra as lie
import numpy as np
import copy


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
