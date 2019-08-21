#!/usr/bin/env python

from unittest import TestCase
import os.path

import evaluation as ev

class TestMainEvaluation(TestCase):
    def test_default_functionality(self):
        parser = ev.parser()
        args = parser.parse_args(['evaluation/tests/test_experiments/test_euroc.yaml', '-a',
                                  '--save_results', '--save_plots', '--save_boxplots'])
        ev.run(args)

        test_output_dir = 'evaluation/tests/test_results/V1_01_easy/S/'
        # Check that we have generated a results file.
        results_file = os.path.join(test_output_dir, 'results.yaml')
        self.assertTrue(os.path.isfile(results_file))

        # Check that we have generated boxplots.
        boxplots_file = os.path.join(test_output_dir, '../traj_relative_errors_boxplots.eps')
        self.assertTrue(os.path.isfile(boxplots_file))
        # Remove file so that we do not re-test and get a false negative...
        os.remove(boxplots_file)

        # Check that we have generated plots.
        plots_list = ['plots_APE_translation.eps',
                      'plots_APE_translation_trajectory_error.eps',
                      'plots_RPE_rotation.eps',
                      'plots_RPE_rotation_trajectory_error.eps',
                      'plots_RPE_translation.eps',
                      'plots_RPE_translation_trajectory_error.eps']
        for plot_filename in plots_list:
            print("Checking plot with filename: %s" % plot_filename)
            plot_filepath = os.path.join(test_output_dir, plot_filename)
            self.assertTrue(os.path.isfile(plot_filepath))
            try:
                # Remove file so that we do not re-test and get a false negative...
                os.remove(plot_filepath)
            except:
                raise Exception("Error while deleting file : ", plots_filepath)

        # Remove file so that we do not re-test and get a false negative...
        os.remove(results_file)
