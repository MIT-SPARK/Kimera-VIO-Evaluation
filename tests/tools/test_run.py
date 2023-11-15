"""Test that datset runner behaves correctly."""


def test_run_jenkins(cli, tmp_path, resources):
    """Test that config loading from package works."""
    ret = cli(
        [
            "run",
            "-n",
            "jenkins_euroc",
            "-e",
            str(resources / "fake_executable.py"),
            str(tmp_path),
        ]
    )
    assert ret.exit_code == 0, f"out: {ret.stdout}"


def test_run_custom_experiment(cli, tmp_path, resources):
    """Test that config loading from custom dir works."""
    ret = cli(
        [
            "run",
            "--experiments-dir",
            str(resources / "test_experiments"),
            "-n",
            "test_experiment",
            "-e",
            str(resources / "fake_executable.py"),
            str(tmp_path),
        ]
    )
    assert ret.exit_code == 0, f"out: {ret.stdout}"


def test_run_missing(cli, tmp_path, resources):
    """Test that an invalid configuration exists gracefully."""
    ret = cli(
        [
            "run",
            "--experiments-dir",
            str(resources / "test_experiments"),
            "-n",
            "some_fake_experiment",
            "-e",
            str(resources / "fake_executable.py"),
            str(tmp_path),
        ]
    )
    assert ret.exit_code != 0
    assert "Could not find" in ret.stdout


def test_run_invalid(cli, tmp_path, resources):
    """Test that an invalid configuration exists gracefully."""
    ret = cli(
        [
            "run",
            "--experiments-dir",
            str(resources / "test_experiments"),
            "-n",
            "invalid_experiment",
            "-e",
            str(resources / "fake_executable.py"),
            str(tmp_path),
        ]
    )
    assert ret.exit_code != 0
    assert "Failed to load" in ret.stdout
