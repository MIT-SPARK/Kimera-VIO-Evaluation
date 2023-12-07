"""Test that analysis works as expected for results."""
import kimera_eval
import shutil


def test_basic_evaluation(tmp_path, resources):
    """Test that the evaluator works."""
    kimera_eval.configure_logging(level="INFO")
    shutil.copytree(resources / "test_results" / "V1_01_easy", tmp_path / "V1_01_easy")

    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(config_path)
    # add a config that will fail
    config.pipelines.append(
        kimera_eval.PipelineConfig(
            name="FakePipeline", param_name="FakePipeline", param_path=""
        )
    )
    evaluator = kimera_eval.DatasetEvaluator(config)
    failed = evaluator.evaluate(tmp_path)
    assert failed == ["V1_01_easy:FakePipeline"]
