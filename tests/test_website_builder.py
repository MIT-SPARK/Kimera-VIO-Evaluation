"""Test website builder."""
import kimera_eval
import shutil


def test_basic_evaluation(tmp_path, resources):
    """Test that the evaluator works."""
    kimera_eval.configure_logging(level="INFO")
    shutil.copytree(resources / "test_results" / "V1_01_easy", tmp_path / "V1_01_easy")

    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(config_path)
    evaluator = kimera_eval.DatasetEvaluator(config)
    failed = evaluator.evaluate(tmp_path)
    assert len(failed) == 0

    builder = kimera_eval.website.WebsiteBuilder()
    builder.write(tmp_path, tmp_path / "website")
    assert (tmp_path / "website" / "vio_ape_euroc.html").exists()
    assert (tmp_path / "website" / "detailed_performance.html").exists()
    assert (tmp_path / "website" / "frontend.html").exists()
    assert (tmp_path / "website" / "datasets.html").exists()
