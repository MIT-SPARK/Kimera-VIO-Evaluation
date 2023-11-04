"""Main entry point for running evaluations."""
import click
import os
import pathlib
import sys
import yaml
import logging


def _get_experiment_directory():
    return pathlib.Path(__file__).absolute().parent / "experiments"


@click.command(name="run")
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
@click.option("--minloglevel", is_flag=True, help="set VIO minloglevel")
def run(experiment, experiment_directory, minloglevel):
    """Run evaluation on datasets."""
    logging.debug(f"experiment: {experiment}")
    logging.debug(f"experiment_directory: {experiment_directory}")
    logging.debug(f"minloglevel: {minloglevel}")


@click.command()
@click.option("-a", "--analyze", is_flag=True, help="Analyze vio, compute APE and RPE")
@click.option("-p", "--plot", is_flag=True, help="show plot window")
@click.option("--save_plots", is_flag=True, help="Save plots?")
@click.option("--write_website", is_flag=True, help="Write website with results?")
@click.option("--save_boxplots", is_flag=True, help="Save boxplots?")
@click.option("--save_results", is_flag=True, help="Save results?")
@click.option("-v", "--verbose", is_flag=True, help="log kimera-vio output to console")
def main(experiment, experiment_directory, log_level):
    """
    Perform full evaluation of SPARK VIO pipeline.

    Reports the following metrics:
      - APE translation
      - RPE translation
      - RPE rotation
    """
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
