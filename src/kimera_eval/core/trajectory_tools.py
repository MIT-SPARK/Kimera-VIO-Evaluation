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
