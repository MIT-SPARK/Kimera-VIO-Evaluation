"""Main library for evaluation."""
from kimera_eval.experiment_config import AnalysisConfig
from dataclasses import dataclass
from typing import Optional, List
import copy
import logging
import pathlib
import pickle
import numpy as np
import pandas as pd

import evo.core.trajectory
import evo.core.lie_algebra as lie
import evo.core.metrics
import evo.core.sync
from evo.core.metrics import PoseRelation, Unit


def _get_default_rpe_args(
    delta=1.0, delta_unit=Unit.frames, rel_delta_tol=1.0, all_pairs=False
):
    return copy.deepcopy(locals())


def get_ape_rot(data):
    """Return APE rotation metric for input trajectories."""
    ape_rot = evo.core.metrics.APE(PoseRelation.rotation_angle_deg)
    ape_rot.process_data(data)
    return ape_rot


def get_ape_trans(data):
    """Return APE translation metric for input trajectories."""
    ape_trans = evo.core.metrics.APE(PoseRelation.translation_part)
    ape_trans.process_data(data)
    return ape_trans


def get_rpe_rot(data):
    """Return RPE rotation metric for input trajectories."""
    rpe_args = _get_default_rpe_args(rel_delta_tol=1.0)
    metric = evo.core.metrics.RPE(PoseRelation.rotation_angle_deg, **rpe_args)
    metric.process_data(data)
    return metric


def get_rpe_trans(data):
    """Return RPE translation metric for input trajectories."""
    rpe_args = _get_default_rpe_args(rel_delta_tol=0.0)
    metric = evo.core.metrics.RPE(PoseRelation.translation_part, **rpe_args)
    metric.process_data(data)
    return metric


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


@dataclass
class TrajectoryPair:
    """Reference and estimated trajectory."""

    est: evo.core.trajectory.PoseTrajectory3D
    ref: evo.core.trajectory.PoseTrajectory3D

    @classmethod
    def load(cls, est_path: pathlib.Path, ref_path: pathlib.Path, sync=True):
        """Load trajectories."""
        if not est_path.exists():
            raise ValueError(f"estimated path '{est_path}' does not exist")

        if not ref_path.exists():
            raise ValueError(f"ground-truth path '{ref_path}' does not exist")

        est = df_to_trajectory(pd.read_csv(est_path, sep=",", index_col=0))
        ref = df_to_trajectory(pd.read_csv(ref_path, sep=",", index_col=0))
        if sync:
            ref, est = evo.core.sync.associate_trajectories(ref, est)

        return cls(ref, est)

    def clone(self):
        """Create a trajectory copy."""
        return TrajectoryPair(copy.deepcopy(self.est), copy.deepcopy(self.ref))

    def align(self, config: AnalysisConfig):
        """Get an aligned version of the original trajectory group."""
        args = {
            "correct_scale": False,
            "discard_n_start_poses": config.discard_n_start_poses,
            "discard_n_end_poses": config.discard_n_end_poses,
        }
        est = align_trajectory(self.est, self.ref, **args)
        return TrajectoryPair(est, copy.deepcopy(self.ref))

    def reduce(self, start, end):
        """Get a clipped trajectory."""
        new_pair = self.clone()
        new_pair.est.reduce_to_ids(range(start, end))
        new_pair.ref.reduce_to_ids(range(start, end))
        return new_pair

    @property
    def num_poses(self):
        """Get number of trajectory poses."""
        return self.est.num_poses

    @property
    def data(self):
        """Get tuple (ref, est) of trajectories."""
        return (self.ref, self.est)

    @property
    def length_m(self):
        """Get trajectory length in meters."""
        return self.est.path_length


@dataclass
class RpeResult:
    """Class holding a specific trajectory result."""

    delta: float
    rotation: evo.core.metrics.RPE
    translation: evo.core.metrics.RPE

    @classmethod
    def from_trajectory(cls, delta, trajectory: TrajectoryPair):
        """Compute RPE from a specific delta in meters."""
        logging.debug(f"RPE analysis of segment length {delta} [m]")
        rpe_args = {
            "delta": delta,
            "delta_unit": evo.core.metrics.Unit.meters,
            "rel_delta_tol": 0.01,
            "all_pairs": True,
        }
        trans = evo.core.metrics.RPE(PoseRelation.translation_part, **rpe_args)
        trans.process_data(trajectory.data)
        rot = evo.core.metrics.RPE(PoseRelation.rotation_angle_deg, **rpe_args)
        rot.process_data(trajectory.data)
        return cls(delta, rotation=rot, translation=trans)


@dataclass
class TrajectoryResults:
    """Class holding trajectory results."""

    ape_translation: evo.core.metrics.APE
    rpe_translation: evo.core.metrics.RPE
    rpe_rotation: evo.core.metrics.RPE
    relative_errors: List[RpeResult]
    trajectory_length_m: float

    @classmethod
    def analyze(cls, config: AnalysisConfig, trajectory: TrajectoryPair):
        """Compute metrics for trajectory."""
        rpe_results = [
            RpeResult.from_trajectory(x, trajectory) for x in config.segments
        ]
        return cls(
            ape_translation=get_ape_trans(trajectory.data),
            rpe_translation=get_rpe_trans(trajectory.data),
            rpe_rotation=get_rpe_rot(trajectory.data),
            relative_errors=rpe_results,
            trajectory_length_m=trajectory.length_m,
        )

    @staticmethod
    def load(result_path: pathlib.Path):
        """Load results from file."""
        if not result_path.exists():
            return None

        with result_path.open("rb") as fin:
            return pickle.load(fin)

    def save(self, result_path: pathlib.Path):
        """Save results to file."""
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("wb") as fout:
            pickle.dump(self, fout)


@dataclass
class TrajectoryGroup:
    """Group of trajectories."""

    vio: TrajectoryPair
    pgo: Optional[TrajectoryPair] = None

    @classmethod
    def load(
        cls,
        result_path: pathlib.Path,
        ref_name="traj_gt.csv",
        vio_name="traj_vio.csv",
        pgo_name="traj_pgo.csv",
        gt_path: Optional[pathlib.Path] = None,
    ):
        """Load trajectories."""
        traj_gt_path = (result_path if gt_path is None else gt_path) / ref_name
        traj_vio_path = result_path / vio_name
        traj_pgo_path = result_path / pgo_name

        if not traj_gt_path.exists():
            raise ValueError(f"GT path '{traj_gt_path}' does not exist")

        if not traj_vio_path.exists():
            raise ValueError(f"VIO result path '{traj_vio_path}' does not exist")

        vio_pair = TrajectoryPair.load(traj_vio_path, traj_gt_path)

        if not traj_pgo_path.exists():
            logging.debug(f"No PGO results found at '{traj_pgo_path}'")
            pgo_pair = None
        else:
            pgo_pair = TrajectoryPair.load(traj_pgo_path, traj_gt_path)

        return cls(vio_pair, pgo_pair)

    def align(self, config: AnalysisConfig):
        """Get an aligned version of the original trajectory group."""
        vio_aligned = self.vio.align(config)
        pgo_aligned = None if self.pgo is None else self.pgo.align(config)
        return TrajectoryGroup(vio_aligned, pgo_aligned)

    def reduce(self, config: AnalysisConfig):
        """Reduce trajectories to input window."""
        num_poses = self.vio.num_poses
        if self.pgo is not None:
            num_poses = min(num_poses, self.pgo.num_poses)

        start = config.discard_n_start_poses
        end = num_poses - config.discard_n_end_poses
        pgo_reduced = None if self.pgo is None else self.pgo.reduce(start, end)
        return TrajectoryGroup(self.vio.reduce(start, end), pgo_reduced)
