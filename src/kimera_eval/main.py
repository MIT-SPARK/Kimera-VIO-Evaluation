"""Main entry point for running evaluations."""
import click
import glog
import os
import pathlib
import sys
import yaml
from evaluation.evaluation_lib import DatasetEvaluator, aggregate_ape_results


def _get_experiment_directory():
    return pathlib.Path(__file__).absolute().parent / "experiments"


@click.command()
@click.option(
    "-e", "--experiment", default="experiments.yaml", help="experiment to run"
)
@click.option(
    "-d",
    "--experiment-directory",
    default=None,
    type=click.Path(exists=True),
    help="directory containing experiment files",
)
@click.option("-l", "--log-level", default="INFO", help="log level")
@click.option("-r", "--run-pipeline", is_flag=True, help="Run vio?")
@click.option("-a", "--analyze", is_flag=True, help="Analyze vio, compute APE and RPE")
@click.option("-p", "--plot", is_flag=True, help="show plot window")
@click.option("--save_plots", is_flag=True, help="Save plots?")
@click.option("--write_website", is_flag=True, help="Write website with results?")
@click.option("--save_boxplots", action="store_true", help="Save boxplots?")
@click.option("--save_results", action="store_true", help="Save results?")
@click.option("-v", "--verbose", is_flag=True, help="log kimera-vio output to console")
def main(experiment, experiment_directory, log_level):
    """
    Perform full evaluation of SPARK VIO pipeline.

    Reports the following metrics:
      - APE translation
      - RPE translation
      - RPE rotation
    """
    glog.setLevel(log_level)
    if experiment_directory:
        experiment_directory = (
            pathlib.Path(experiment_directory).expanduser().absolute()
        )
    else:
        experiment_directory = _get_experiment_directory()

    experiment_path = experiment_directory / experiment
    if not experiment_path.exists():
        click.secho(f"Could not find experiment '{experiment_path}'", fg="red")
        sys.exit(1)

    with experiment_path.open("r") as fin:
        experiment_params = yaml.safe_load(fin.read())

    # TODO(marcus): parse from experiments
    extra_flagfile_path = ""
    # TODO(marcus): choose which of the following based on -r -a flags
    dataset_evaluator = DatasetEvaluator(experiment_params, args, extra_flagfile_path)
    dataset_evaluator.evaluate()
    # Aggregate results in results directory
    aggregate_ape_results(os.path.expandvars(experiment_params["results_dir"]))
    sys.exit(os.EX_OK)
