"""Test experiment config loading."""
import kimera_eval
import pathlib
import pytest


def test_read_named_config():
    """Test that supplying information without a name works."""
    with pytest.raises(ValueError):
        kimera_eval.read_named_config(kimera_eval.PipelineConfig, param_name="test")


def test_load_config(resources, pkgpath):
    """Test that config loading works."""
    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(config_path)

    this_path = pathlib.Path().absolute()
    assert config.executable_path == this_path
    assert config.param_path == this_path
    expected_datapath = this_path / "evaluation" / "tests" / "test_results"
    assert config.dataset_path == expected_datapath
    assert config.vocabulary_path == (this_path / "vocabulary" / "ORBvoc.yml")

    assert len(config.pipelines) == 2
    pipeline = config.pipelines[0]
    assert pipeline.name == "Euroc"
    assert pipeline.param_name == "EurocParams"
    assert pipeline.extra_flags_path == pathlib.Path("/some/flag/file/path.flag")

    flag_files = [x for x in pipeline.flag_files]
    flag_path = this_path / "flags"
    expected_flag_files = [
        flag_path / "stereoVIOEuroc.flags",
        flag_path / "Mesher.flags",
        flag_path / "VioBackend.flags",
        flag_path / "RegularVioBackend.flags",
        flag_path / "Visualizer3D.flags",
        pathlib.Path("/some/flag/file/path.flag"),
        pkgpath / "config" / "NoVisualizer.flag",
    ]
    assert flag_files == expected_flag_files
    expected_args = [f"--flagfile={x}" for x in expected_flag_files] + [
        f"--params_folder_path={this_path / 'EurocParams'}"
    ]
    assert pipeline.args == expected_args

    assert len(config.sequences) == 3
    sequence = config.sequences[0]
    assert sequence.name == "V1_01_easy"
    assert sequence.use_lcd
    assert sequence.initial_frame == 10
    assert sequence.final_frame == 3500
    assert sequence.args == ["--initial_k=10", "--final_k=3500", "--use_lcd=true"]

    assert config.sequences[1].name == "V1_02_medium"
    assert config.sequences[1].args == ["--initial_k=20", "--use_lcd=false"]

    assert config.sequences[2].name == "V1_03_hard"
    assert config.sequences[2].args == [
        "--initial_k=0",
        "--final_k=20",
        "--use_lcd=false",
    ]


def test_load_config_overrides(resources, pkgpath):
    """Test that config loading works."""
    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(
        config_path,
        executable_path="/foo/foo",
        dataset_path="foo/bar",
        param_path="/bar/foo",
        vocabulary_path="/some/vocab/path",
    )

    this_path = pathlib.Path().absolute()
    assert config.executable_path == pathlib.Path("/foo/foo")
    assert config.param_path == pathlib.Path("/bar/foo")
    assert config.dataset_path == pathlib.Path(this_path / "foo/bar")
    assert config.vocabulary_path == pathlib.Path("/some/vocab/path")

    assert len(config.pipelines) == 2
    pipeline = config.pipelines[0]

    flag_files = [x for x in pipeline.flag_files]
    flag_path = pathlib.Path("/bar/foo") / "flags"
    expected_flag_files = [
        flag_path / "stereoVIOEuroc.flags",
        flag_path / "Mesher.flags",
        flag_path / "VioBackend.flags",
        flag_path / "RegularVioBackend.flags",
        flag_path / "Visualizer3D.flags",
        pathlib.Path("/some/flag/file/path.flag"),
        pkgpath / "config" / "NoVisualizer.flag",
    ]
    assert flag_files == expected_flag_files


def test_load_missing(resources):
    """Test that we require certain paths to be specified."""
    config_path = resources / "test_experiments" / "invalid_experiment.yaml"

    # no required paths specified
    config = kimera_eval.ExperimentConfig.load(config_path)
    assert config is None

    # one required path specified
    config = kimera_eval.ExperimentConfig.load(config_path, executable_path="/foo/bar")
    assert config is None

    # two required paths specified
    config = kimera_eval.ExperimentConfig.load(
        config_path, executable_path="/foo/bar", param_path="/bar/foo"
    )
    assert config is None

    # all required paths specified
    config = kimera_eval.ExperimentConfig.load(
        config_path,
        executable_path="/foo/bar",
        param_path="/bar/foo",
        dataset_path="/foo/foo",
    )
    assert config is not None
    assert config.args == [
        "/foo/bar",
        "--vocabulary_path=/bar/foo/vocabulary/ORBvoc.yml",
    ]
