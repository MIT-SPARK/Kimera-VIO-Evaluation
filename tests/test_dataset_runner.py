"""Test that datset runner behaves correctly."""
import kimera_eval


def _get_flags(status):
    return {k: {x: y[0] for x, y in v.items()} for k, v in status.items()}


def test_run_normal(tmp_path, resources):
    """Test that config loading works."""
    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(
        config_path,
        executable_path="/foo/foo",
        dataset_path="foo/bar",
        param_path="/bar/foo",
        vocabulary_path="/some/vocab/path",
    )

    runner = kimera_eval.DatasetRunner(config, tmp_path)
    status = runner.run_all()

    status_flags = _get_flags(status)
    expected = {
        "V1_01_easy": {"Euroc": False, "Euroc2": False},
        "V1_02_medium": {"Euroc": False, "Euroc2": False},
        "V1_03_hard": {"Euroc": False, "Euroc2": False},
    }
    assert status_flags == expected


def test_run_mocked(tmp_path, resources):
    """Test that config loading works."""
    config_path = resources / "test_experiments" / "test_euroc.yaml"
    config = kimera_eval.ExperimentConfig.load(
        config_path,
        executable_path=resources / "fake_executable.py",
        dataset_path="foo/bar",
        param_path="/bar/foo",
        vocabulary_path="/some/vocab/path",
    )

    runner = kimera_eval.DatasetRunner(config, tmp_path)
    status = runner.run_all()

    status_flags = _get_flags(status)
    expected = {
        "V1_01_easy": {"Euroc": True, "Euroc2": True},
        "V1_02_medium": {"Euroc": True, "Euroc2": True},
        "V1_03_hard": {"Euroc": True, "Euroc2": True},
    }
    assert status_flags == expected

    status = runner.run_all()

    status_flags = _get_flags(status)
    expected = {
        "V1_01_easy": {"Euroc": False, "Euroc2": False},
        "V1_02_medium": {"Euroc": False, "Euroc2": False},
        "V1_03_hard": {"Euroc": False, "Euroc2": False},
    }
    assert status_flags == expected

    status = runner.run_all(allow_removal=True)

    status_flags = _get_flags(status)
    expected = {
        "V1_01_easy": {"Euroc": True, "Euroc2": True},
        "V1_02_medium": {"Euroc": True, "Euroc2": True},
        "V1_03_hard": {"Euroc": True, "Euroc2": True},
    }
    assert status_flags == expected
