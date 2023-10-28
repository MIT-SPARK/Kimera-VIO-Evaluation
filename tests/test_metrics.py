"""Test evaluation code."""
import pytest
import random
import numpy as np

from evo.core import trajectory, metrics
from kimera_eval.core.trajectory_metrics import (
    get_ape_rot,
    get_ape_trans,
    get_rpe_rot,
    get_rpe_trans,
    convert_abs_traj_to_rel_traj,
)


def test_get_ape_rot_0():
    """Test equivalent trajectories for zero ARE."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    ape_metric_rot = get_ape_rot((traj_1, traj_2))

    assert ape_metric_rot.pose_relation == metrics.PoseRelation.rotation_angle_deg
    assert ape_metric_rot.unit == metrics.Unit.degrees
    assert np.allclose(
        ape_metric_rot.error,
        [0.0 for i in range(len(ape_metric_rot.error))],
        atol=1e-5,
    )


def test_get_ape_rot_1():
    """Test a 90-deg ARE on the first pose only."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    quat_wxyz[0] = [1, 0, 0, 0]
    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    quat_wxyz[0] = [
        0.7071068,
        0.7071068,
        0,
        0,
    ]
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    ape_metric_rot = get_ape_rot((traj_1, traj_2))

    assert ape_metric_rot.error[0] == pytest.approx(90.0)
    assert np.allclose(
        ape_metric_rot.error[1:],
        [0.0 for i in range(len(ape_metric_rot.error[1:]))],
        atol=1e-5,
    )


def test_get_ape_trans_0():
    """Test equivalent trajectories for zero ATE."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    ape_metric_trans = get_ape_trans((traj_1, traj_2))

    assert ape_metric_trans.pose_relation == metrics.PoseRelation.translation_part
    assert ape_metric_trans.unit == metrics.Unit.meters
    assert np.allclose(
        ape_metric_trans.error,
        [0.0 for i in range(len(ape_metric_trans.error))],
        atol=1e-5,
    )


def test_get_ape_trans_1():
    """Test a 1-meter ATE on the first pose only."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    pos_xyz[0] = [0, 0, 0]
    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    pos_xyz[0] = [
        1,
        0,
        0,
    ]
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    ape_metric_trans = get_ape_trans((traj_1, traj_2))

    assert ape_metric_trans.error[0] == pytest.approx(1.0)
    assert np.allclose(
        ape_metric_trans.error[1:],
        [0.0 for i in range(len(ape_metric_trans.error[1:]))],
        atol=1e-5,
    )


def test_get_rpe_rot_0():
    """Test equivalent trajectories for zero RRE."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    rpe_metric_rot = get_rpe_rot((traj_1, traj_2))

    assert rpe_metric_rot.pose_relation == metrics.PoseRelation.rotation_angle_deg
    assert rpe_metric_rot.unit == metrics.Unit.degrees
    assert np.allclose(
        rpe_metric_rot.error,
        [0.0 for i in range(len(rpe_metric_rot.error))],
        atol=1e-5,
    )


def test_get_rpe_rot_1():
    """Test a 90-deg RRE on the first pose only."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    quat_wxyz[0] = [1, 0, 0, 0]
    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    quat_wxyz[0] = [
        0.7071068,
        0.7071068,
        0,
        0,
    ]
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    rpe_metric_rot = get_rpe_rot((traj_1, traj_2))

    assert rpe_metric_rot.error[0] == pytest.approx(90.0)
    assert np.allclose(
        rpe_metric_rot.error[1:],
        [0.0 for i in range(len(rpe_metric_rot.error[1:]))],
        atol=1e-5,
    )


def test_get_rpe_trans_0():
    """Test equivalent trajectories for zero RTE."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    rpe_metric_trans = get_rpe_trans((traj_1, traj_2))

    assert rpe_metric_trans.pose_relation == metrics.PoseRelation.translation_part
    assert rpe_metric_trans.unit == metrics.Unit.meters
    assert np.allclose(
        rpe_metric_trans.error,
        [0.0 for i in range(len(rpe_metric_trans.error))],
        atol=1e-5,
    )


def test_get_rpe_trans_1():
    """Test a 1-meter RTE on the first pose only."""
    pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
    quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
    timestamps = [i for i in range(15)]

    pos_xyz[0] = [0, 0, 0]
    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    pos_xyz[0] = [
        1,
        0,
        0,
    ]
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    rpe_metric_trans = get_rpe_trans((traj_1, traj_2))

    assert rpe_metric_trans.error[0] == pytest.approx(1.0)
    assert np.allclose(
        rpe_metric_trans.error[1:],
        [0.0 for i in range(len(rpe_metric_trans.error[1:]))],
        atol=1e-5,
    )


def test_convert_abs_traj_to_rel_traj_0():
    """Test translation-only relative trajectory."""
    pos_xyz = [[0, 0, 0], [1, 0, 0]]
    quat_wxyz = [
        [
            0.7071068,
            0.7071068,
            0,
            0,
        ],
        [
            0.7071068,
            0.7071068,
            0,
            0,
        ],
    ]
    timestamps = [0, 1]

    traj_abs = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
    traj_rel = convert_abs_traj_to_rel_traj(traj_abs, False)

    assert len(traj_rel.positions_xyz) == 1
    assert len(traj_rel.orientations_quat_wxyz) == 1
    assert len(traj_rel.timestamps) == 1
    assert traj_rel.timestamps[0] == timestamps[1]

    assert np.allclose(traj_rel.positions_xyz[0], pos_xyz[1], atol=1e-6)
    assert np.allclose(traj_rel.orientations_quat_wxyz[0], [1, 0, 0, 0], atol=1e-6)


def test_convert_abs_traj_to_rel_traj_1():
    """Test equivalent trajectories for zero relative ATE (not to-scale)."""
    pos_xyz = np.array([[random.random() for _ in range(3)] for _ in range(15)])
    quat_wxyz = np.array([[random.random() for _ in range(4)] for _ in range(15)])
    timestamps = np.array([i for i in range(15)])

    traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
    pos_xyz = [pos * 2 for pos in pos_xyz]
    traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

    traj_1 = convert_abs_traj_to_rel_traj(traj_1, True)
    traj_2 = convert_abs_traj_to_rel_traj(traj_2, True)
    ape_metric_trans = get_ape_trans((traj_1, traj_2))

    assert np.allclose(
        ape_metric_trans.error,
        [0.0 for i in range(len(ape_metric_trans.error))],
        atol=1e-5,
    )
