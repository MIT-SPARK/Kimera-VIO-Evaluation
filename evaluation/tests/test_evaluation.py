from unittest import TestCase

import evaluation as ev

import os.path

class TestMainEvaluation(TestCase):
    def test_default_functionality(self):
        parser = ev.parser()
        args = parser.parse_args(['evaluation/tests/test_experiments/test_euroc.yaml', '-a', '--save_results'])
        ev.run(args)

        # Check that we have generated a results file.
        results_file = 'evaluation/tests/test_results/V1_01_easy/S/results.yaml'
        self.assertTrue(os.path.isfile(results_file))

        # Remove file so that we do not re-test and get a false negative...
        os.remove(results_file)


