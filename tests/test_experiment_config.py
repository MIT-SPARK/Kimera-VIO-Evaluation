"""Test experiment config loading."""
import kimera_eval
import pathlib
import pytest


def test_read_named_config():
    """Test that supplying information without a name works."""
    with pytest.raises(ValueError):
        kimera_eval.read_named_config(kimera_eval.PipelineConfig, param_name="test")


def test_load_config(resources):
    """Test that config loading works."""
    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(config_path)

    this_path = pathlib.Path().absolute()
    assert config.executable_path == this_path
    assert config.param_path == this_path
    expected_datapath = this_path / "evaluation" / "tests" / "test_results"
    assert config.dataset_path == expected_datapath

    assert len(config.pipelines) == 1
    pipeline = config.pipelines[0]
    assert pipeline.name == "Euroc"
    assert pipeline.param_name == "EurocParams"
    assert pipeline.extra_flags_path == pathlib.Path("/some/flag/file/path.flag")

    flag_files = [x for x in pipeline.flag_files]
    expected_flag_files = [
        this_path / "stereoVIOEuroc.flags",
        this_path / "Mesher.flags",
        this_path / "VioBackend.flags",
        this_path / "RegularVioBackend.flags",
        this_path / "Visualizer3D.flags",
        pathlib.Path("/some/flag/file/path.flag"),
    ]
    assert flag_files == expected_flag_files

    assert len(config.sequences) == 1
    sequence = config.sequences[0]
    assert sequence.name == "V1_01_easy"
    assert sequence.use_lcd
    assert sequence.initial_frame == 10
    assert sequence.final_frame == 3500
