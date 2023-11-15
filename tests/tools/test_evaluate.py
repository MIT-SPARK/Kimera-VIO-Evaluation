"""Test that analysis works as expected for results."""
import kimera_eval
import shutil


def test_evaluation_command(cli, tmp_path, resources):
    """Test that the evaluator works via command line."""
    kimera_eval.configure_logging(level="INFO")
    shutil.copytree(resources / "test_results" / "V1_01_easy", tmp_path / "V1_01_easy")

    ret = cli(
        [
            "evaluate",
            "--experiments-dir",
            str(resources / "test_experiments"),
            "-n",
            "test_experiment",
            str(tmp_path),
        ]
    )
    assert ret.exit_code == 0, f"out: {ret.stdout}"
