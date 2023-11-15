"""Main entry point for running evaluations."""
import kimera_eval.paths
from kimera_eval.dataset_evaluator import DatasetEvaluator
from kimera_eval.experiment_config import ExperimentConfig
import logging
import click
import sys


@click.command(name="evaluate")
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--save_plots", is_flag=True, help="Save plots?")
@click.option("--write_website", is_flag=True, help="Write website with results?")
@click.option("-n", "--name", default="example_euroc", help="experiment to run")
@click.option(
    "--experiments-dir",
    type=click.Path(exists=True),
    default=None,
    help="directory containing experiment files",
)
def run(results_path, write_website, name, experiments_dir):
    """
    Perform full evaluation of SPARK VIO pipeline.

    Reports the following metrics:
      - APE translation
      - RPE translation
      - RPE rotation
    """
    if experiments_dir:
        experiments_dir = kimera_eval.paths.normalize_path(experiments_dir)
    else:
        experiments_dir = kimera_eval.paths.experiment_directory()

    experiment_path = experiments_dir / f"{name}.yaml"
    if not experiment_path.exists():
        logging.fatal(f"Could not find experiment '{experiment_path}'", fg="red")
        sys.exit(1)

    config = ExperimentConfig.load(experiment_path)

    dataset_evaluator = DatasetEvaluator(config)
    failed = dataset_evaluator.evaluate(results_path)
    if len(failed) != 0:
        sys.exit(1)
