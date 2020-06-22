#!/usr/bin/env python

from unittest import TestCase
import random
import numpy as np
import os

from evo.core import trajectory, metrics

from evaluation.evaluation_lib import get_ape_rot, get_ape_trans, \
    get_rpe_rot, get_rpe_trans, convert_abs_traj_to_rel_traj

from .. import main_evaluation as ev

class TestMainEvaluation(TestCase):
    def test_default_functionality(self):
        parser = ev.parser()
        args = parser.parse_args([os.path.join(os.getcwd(), 
                                               'evaluation/tests/test_experiments/test_euroc.yaml'),
                                  '-a', '--save_results', '--save_plots', '--save_boxplots'])
        ev.run(args)

        test_output_dir = os.path.join(os.getcwd(), 'evaluation/tests/test_results/V1_01_easy/Euroc/')
        # Check that we have generated a results file.
        results_file = os.path.join(test_output_dir, 'results_vio.yaml')
        self.assertTrue(os.path.isfile(results_file))
        # Remove file so that we do not re-test and get a false negative...
        os.remove(results_file)

        results_file = os.path.join(test_output_dir, 'results_pgo.yaml')
        self.assertTrue(os.path.isfile(results_file))
        # Remove file so that we do not re-test and get a false negative...
        os.remove(results_file)

        # Check that we have generated boxplots.
        boxplots_file = os.path.join(test_output_dir, '../../datasets_ape_boxplots.pdf')
        self.assertTrue(os.path.isfile(boxplots_file))
        # Remove file so that we do not re-test and get a false negative...
        os.remove(boxplots_file)

        # Check that we have generated APE table.
        ape_table_file = os.path.join(test_output_dir, '../../APE_table.tex')
        self.assertTrue(os.path.isfile(ape_table_file))
        # Remove file so that we do not re-test and get a false negative...
        os.remove(ape_table_file)

        # Check that we have generated plots.
        plot_filename = "plots.pdf"
        plot_filepath = os.path.join(test_output_dir, plot_filename)
        print("Checking plot with filename: %s \n At path: %s" % (plot_filename, plot_filepath))
        self.assertTrue(os.path.isfile(plot_filepath))
        try:
            # Remove file so that we do not re-test and get a false negative...
            os.remove(plot_filepath)
        except:
            raise Exception("Error while deleting file : ", plot_filepath)

class TestEvaluationMisc(TestCase):

    def test_get_ape_rot_0(self):
        """ Test equivalent trajectories for zero ARE """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_rot = get_ape_rot((traj_1, traj_2))

        self.assertEqual(ape_metric_rot.pose_relation, metrics.PoseRelation.rotation_angle_deg)
        self.assertEqual(ape_metric_rot.unit, metrics.Unit.degrees)
        self.assertTrue(
            np.allclose(ape_metric_rot.error, 
                        [0. for i in range(len(ape_metric_rot.error))], 
                        atol=1e-5)
        )

    def test_get_ape_rot_1(self):
        """ Test a 90-deg ARE on the first pose only """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        quat_wxyz[0] = [1, 0, 0, 0]
        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        quat_wxyz[0] = [0.7071068, 0.7071068, 0, 0,]
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_rot = get_ape_rot((traj_1, traj_2))

        self.assertAlmostEqual(ape_metric_rot.error[0], 90.)
        self.assertTrue(
            np.allclose(ape_metric_rot.error[1:],
                        [0. for i in range(len(ape_metric_rot.error[1:]))],
                        atol=1e-5)
        )

    def test_get_ape_trans_0(self):
        """ Test equivalent trajectories for zero ATE """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_trans = get_ape_trans((traj_1, traj_2))

        self.assertEqual(ape_metric_trans.pose_relation, metrics.PoseRelation.translation_part)
        self.assertEqual(ape_metric_trans.unit, metrics.Unit.meters)
        self.assertTrue(
            np.allclose(ape_metric_trans.error, 
                        [0. for i in range(len(ape_metric_trans.error))], 
                        atol=1e-5)
        )

    def test_get_ape_trans_1(self):
        """ Test a 1-meter ATE on the first pose only """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        pos_xyz[0] = [0, 0, 0]
        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        pos_xyz[0] = [1, 0, 0,]
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        ape_metric_trans = get_ape_trans((traj_1, traj_2))

        self.assertAlmostEqual(ape_metric_trans.error[0], 1.)
        self.assertTrue(
            np.allclose(ape_metric_trans.error[1:],
                        [0. for i in range(len(ape_metric_trans.error[1:]))],
                        atol=1e-5)
        )

    def test_get_rpe_rot_0(self):
        """ Test equivalent trajectories for zero RRE """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        rpe_metric_rot = get_rpe_rot((traj_1, traj_2))

        self.assertEqual(rpe_metric_rot.pose_relation, metrics.PoseRelation.rotation_angle_deg)
        self.assertEqual(rpe_metric_rot.unit, metrics.Unit.degrees)
        self.assertTrue(
            np.allclose(rpe_metric_rot.error,
                        [0. for i in range(len(rpe_metric_rot.error))], 
                        atol=1e-5)
        )

    def test_get_rpe_rot_1(self):
        """ Test a 90-deg RRE on the first pose only """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        quat_wxyz[0] = [1, 0, 0, 0]
        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        quat_wxyz[0] = [0.7071068, 0.7071068, 0, 0,]
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        rpe_metric_rot = get_rpe_rot((traj_1, traj_2))

        self.assertAlmostEqual(rpe_metric_rot.error[0], 90.)
        self.assertTrue(
            np.allclose(rpe_metric_rot.error[1:],
                        [0. for i in range(len(rpe_metric_rot.error[1:]))],
                        atol=1e-5)
        )

    def test_get_rpe_trans_0(self):
        """ Test equivalent trajectories for zero RTE """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        rpe_metric_trans = get_rpe_trans((traj_1, traj_2))

        self.assertEqual(rpe_metric_trans.pose_relation, metrics.PoseRelation.translation_part)
        self.assertEqual(rpe_metric_trans.unit, metrics.Unit.meters)
        self.assertTrue(
            np.allclose(rpe_metric_trans.error,
                        [0. for i in range(len(rpe_metric_trans.error))], 
                        atol=1e-5)
        )

    def test_get_rpe_trans_1(self):
        """ Test a 1-meter RTE on the first pose only """
        pos_xyz = [[random.random() for _ in range(3)] for _ in range(15)]
        quat_wxyz = [[random.random() for _ in range(4)] for _ in range(15)]
        timestamps = [i for i in range(15)]

        pos_xyz[0] = [0, 0, 0]
        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        pos_xyz[0] = [1, 0, 0,]
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        rpe_metric_trans = get_rpe_trans((traj_1, traj_2))

        self.assertAlmostEqual(rpe_metric_trans.error[0], 1.)
        self.assertTrue(
            np.allclose(rpe_metric_trans.error[1:],
                        [0. for i in range(len(rpe_metric_trans.error[1:]))],
                        atol=1e-5)
        )

    def test_convert_abs_traj_to_rel_traj_0(self):
        """ Test translation-only relative trajectory """
        pos_xyz = [[0, 0, 0], [1, 0, 0]]
        quat_wxyz = [[0.7071068, 0.7071068, 0, 0,], [0.7071068, 0.7071068, 0, 0,]]
        timestamps = [0, 1]
        
        traj_abs = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        traj_rel = convert_abs_traj_to_rel_traj(traj_abs, False)

        self.assertEqual(len(traj_rel.positions_xyz), 1)
        self.assertEqual(len(traj_rel.orientations_quat_wxyz), 1)
        self.assertEqual(len(traj_rel.timestamps), 1)
        self.assertEqual(traj_rel.timestamps[0], timestamps[1])

        self.assertTrue(np.allclose(traj_rel.positions_xyz[0],
                                    pos_xyz[1],
                                    atol=1e-6))
        self.assertTrue(np.allclose(traj_rel.orientations_quat_wxyz[0],
                                    [1, 0, 0, 0],
                                    atol=1e-6))

    def test_convert_abs_traj_to_rel_traj_1(self):
        """ Test equivalent trajectories for zero relative ATE (not to-scale) """
        pos_xyz = np.array([[random.random() for _ in range(3)] for _ in range(15)])
        quat_wxyz = np.array([[random.random() for _ in range(4)] for _ in range(15)])
        timestamps = np.array([i for i in range(15)])

        traj_1 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)
        pos_xyz = [pos*2 for pos in pos_xyz]
        traj_2 = trajectory.PoseTrajectory3D(pos_xyz, quat_wxyz, timestamps)

        traj_1 = convert_abs_traj_to_rel_traj(traj_1, True)
        traj_2 = convert_abs_traj_to_rel_traj(traj_2, True)
        ape_metric_trans = get_ape_trans((traj_1, traj_2))

        self.assertTrue(
            np.allclose(ape_metric_trans.error,
                        [0. for i in range(len(ape_metric_trans.error))],
                        atol=1e-5)
        )