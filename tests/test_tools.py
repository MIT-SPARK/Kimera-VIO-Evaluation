"""Test that datset runner behaves correctly."""
import shutil


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


def test_evaluate_normal(cli, tmp_path, resources):
    """Test that the evaluator works via command line."""
    shutil.copytree(resources / "test_results" / "V1_01_easy", tmp_path / "V1_01_easy")

    ret = cli(
        [
            "evaluate",
            "--experiments-dir",
            str(resources / "test_experiments"),
            "-n",
            "test_euroc",
            str(tmp_path),
        ]
    )
    assert ret.exit_code == 0, f"out: {ret.stdout}"


def test_website_normal(cli, tmp_path, resources):
    """Test that the evaluator works via command line."""
    result_path = tmp_path / "results"
    result_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        resources / "test_results" / "V1_01_easy", result_path / "V1_01_easy"
    )

    ret = cli(
        [
            "evaluate",
            "--experiments-dir",
            str(resources / "test_experiments"),
            "-n",
            "test_euroc",
            str(result_path),
        ]
    )
    assert ret.exit_code == 0, f"out: {ret.stdout}"

    ret = cli(["website", str(tmp_path / "website"), str(result_path)])
    assert ret.exit_code == 0, f"out: {ret.stdout}"


def test_summary_normal(cli, tmp_path, resources):
    """Test that the evaluator works via command line."""
    result_path = tmp_path / "results"
    result_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        resources / "test_results" / "V1_01_easy", result_path / "V1_01_easy"
    )

    ret = cli(
        [
            "evaluate",
            "--experiments-dir",
            str(resources / "test_experiments"),
            "-n",
            "test_euroc",
            str(result_path),
        ]
    )
    assert ret.exit_code == 0, f"out: {ret.stdout}"

    ret = cli(
        [
            "summary",
            str(result_path / "V1_01_easy" / "Euroc" / "results_vio.pickle"),
            "-o",
            str(tmp_path / "summary.csv"),
        ]
    )
    assert ret.exit_code == 0, f"out: {ret.stdout}"
