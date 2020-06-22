#!/usr/bin/env python

import unittest
import random
import numpy as np

from evo.core import trajectory, metrics

from evaluation.evaluation_lib import get_ape_rot, get_ape_trans, \
    get_rpe_rot, get_rpe_trans, convert_abs_traj_to_rel_traj


class TestEvaluationMisc(unittest.TestCase):

    def test_get_ape_rot(self):
        # Test equivalent trajectories for zero ARE
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_rot = get_ape_rot((traj_1, traj_2))

        self.assertEqual(ape_metric_rot.pose_relation, metrics.PoseRelation.rotation_angle_deg)
        self.assertEqual(ape_metric_rot.unit, metrics.Unit.degrees)
        self.assertTrue(np.allclose(ape_metric_rot.error, [0. for i in range(15)], atol=1e-5))

        # Test a 90-deg ARE on the first pose only
        quat_wxyz[0] = [1, 0, 0, 0]
        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        quat_wxyz[0] = [0.7071068, 0.7071068, 0, 0,]
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_rot = get_ape_rot((traj_1, traj_2))

        self.assertAlmostEqual(ape_metric_rot.error[0], 90.)
        self.assertTrue(np.allclose(ape_metric_rot.error[1:], [0. for i in range(14)], atol=1e-5))

    def test_get_ape_trans(self):
        # Test equivalent trajectories for zero ATE
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_trans = get_ape_trans((traj_1, traj_2))

        self.assertEqual(ape_metric_trans.pose_relation, metrics.PoseRelation.translation_part)
        self.assertEqual(ape_metric_trans.unit, metrics.Unit.meters)
        self.assertTrue(np.allclose(ape_metric_trans.error, [0. for i in range(15)], atol=1e-5))

        # Test a 1-meter ATE on the first pose only
        pos_xyz[0] = [0, 0, 0]
        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        pos_xyz[0] = [1, 0, 0,]
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_trans = get_ape_trans((traj_1, traj_2))

        self.assertAlmostEqual(ape_metric_trans.error[0], 1.)
        self.assertTrue(np.allclose(ape_metric_trans.error[1:], [0. for i in range(14)], atol=1e-5))

    def test_get_rpe_rot(self):
        pass

    def test_get_rpe_trans(self):
        pass

    def test_convert_abs_traj_to_rel_traj(self):
        pos_xyz = [[0, 0, 0], [1, 0, 0]]
        # quat_wxyz = [[1, 0, 0, 0], [0.7071068, 0.7071068, 0, 0,]]
        quat_wxyz = [[1, 0, 0, 0], [1, 0, 0, 0,]]
        timestamps = [0, 1]
        
        traj_abs = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_rel = convert_abs_traj_to_rel_traj(traj_abs, False)


        self.assertEqual(len(traj_rel.positions_xyz), 1)
        self.assertEqual(len(traj_rel.orientations_quat_wxyz), 1)
        self.assertEqual(len(traj_rel.timestamps), 1)
        self.assertEqual(traj_rel.timestamps[0], timestamps[1])

        self.assertTrue(np.allclose(traj_rel.positions_xyz[0], pos_xyz[1], atol=1e-6))
        self.assertTrue(np.allclose(traj_rel.orientations_quat_wxyz[0], quat_wxyz[1], atol=1e-6))


if __name__ == '__main__':
    unittest.main()
