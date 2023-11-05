"""Test experiment config loading."""
import kimera_eval


def test_load_config(resources):
    """Test that config loading works."""
    config_path = resources / "test_experiments" / "test_euroc.yaml"
    kimera_eval.ExperimentConfig.load(config_path)
