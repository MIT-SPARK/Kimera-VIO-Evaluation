"""Test that analysis works as expected for results."""
import kimera_eval


def test_basic_evaluation(tmp_path, resources):
    """Test that the evaluator works."""
    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(config_path)
    evaluator = kimera_eval.DatasetEvaluator(config, tmp_path)
    evaluator.evaluate()
