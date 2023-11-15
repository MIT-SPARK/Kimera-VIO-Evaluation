"""Main entry point for running evaluations."""
import kimera_eval.paths
from kimera_eval.dataset_runner import DatasetRunner
from kimera_eval.experiment_config import ExperimentConfig
import click
import sys
import logging


@click.command(name="run")
@click.argument("output-path", type=click.Path())
@click.option("-e", "--executable-path", type=click.Path(exists=True), default=None)
@click.option("-p", "--param-path", type=click.Path(exists=True), default=None)
@click.option("-d", "--dataset-path", type=click.Path(exists=True), default=None)
@click.option("-n", "--name", default="example_euroc", help="experiment to run")
@click.option("-v", "--vocab-path", type=click.Path(exists=True), default=None)
@click.option("--minloglevel", default=2, help="set VIO minloglevel")
@click.option("-f", "--force-removal", is_flag=True, help="remove prior results")
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
    if experiments_dir:
        experiments_dir = kimera_eval.paths.normalize_path(experiments_dir)
    else:
        experiments_dir = kimera_eval.paths.experiment_directory()

    experiment_path = experiments_dir / f"{name}.yaml"
    if not experiment_path.exists():
        logging.fatal(f"Could not find experiment '{experiment_path}'")
        sys.exit(1)

    config = ExperimentConfig.load(
        experiment_path,
        dataset_path=dataset_path,
        param_path=param_path,
        executable_path=executable_path,
        vocabluary_path=vocab_path,
    )
    if config is None:
        logging.fatal(f"Failed to load experiment config from '{experiment_path}'")
        sys.exit(1)

    output_path = kimera_eval.paths.normalize_path(output_path)
    runner = DatasetRunner(config, output_path)
    runner.run_all(minloglevel=minloglevel, allow_removal=force_removal)
