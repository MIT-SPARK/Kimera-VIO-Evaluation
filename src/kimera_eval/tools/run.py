"""Main entry point for running evaluations."""
from kimera_eval.dataset_runner import DatasetRunner
from kimera_eval.experiment_config import ExperimentConfig
import click
import pathlib
import sys
import logging


def _get_experiment_directory():
    return pathlib.Path(__file__).absolute().parent / "experiments"


def _normalize_path(input_path):
    return pathlib.Path(input_path).expanduser().absolute()


@click.command(name="run")
@click.argument("output-path", type=click.Path())
@click.option("-e", "--executable-path", type=click.Path(exists=True), default=None)
@click.option("-p", "--param-path", type=click.Path(exists=True), default=None)
@click.option("-d", "--dataset-path", type=click.Path(exists=True), default=None)
@click.option("-n", "--name", default="example_euroc", help="experiment to run")
@click.option("-v", "--vocab-path", type=click.Path(exists=True), default=None)
@click.option("--minloglevel", default=2, help="set VIO minloglevel")
@click.option(
    "--experiments-dir",
    type=click.Path(exists=True),
    default=None,
    help="directory containing experiment files",
)
def run(
    output_path,
    executable_path,
    param_path,
    dataset_path,
    name,
    vocab_path,
    minloglevel,
    force_removal,
    experiments_dir,
):
    """Run evaluation on datasets."""
    dataset_path = _normalize_path(dataset_path)
    if experiments_dir:
        experiments_dir = _normalize_path(experiments_dir)
    else:
        experiments_dir = _get_experiment_directory()

    experiment_path = experiments_dir / f"{name}.yaml"
    if not experiment_path.exists():
        logging.fatal(f"Could not find experiment '{experiment_path}'", fg="red")
        sys.exit(1)

    config = ExperimentConfig.load(
        experiment_path,
        dataset_path=dataset_path,
        param_path=param_path,
        executable_path=executable_path,
        vocabluary_path=vocab_path,
    )

    output_path = _normalize_path(output_path)
    runner = DatasetRunner(config, output_path)
    runner.run_all(minloglevel=minloglevel, allow_removal=force_removal)
