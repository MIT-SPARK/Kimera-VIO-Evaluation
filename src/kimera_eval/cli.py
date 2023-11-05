"""Main entry point for running evaluations."""
from kimera_eval.core.dataset_runner import DatasetRunner, ExperimentConfig
import click
import os
import pathlib
import sys
import yaml
import logging


def _get_experiment_directory():
    return pathlib.Path(__file__).absolute().parent / "experiments"


def _normalize_path(input_path):
    return pathlib.Path(input_path).expanduser().absolute()


@click.command(name="run")
@click.argument("executable_path", type=click.Path(exists=True))
@click.argument("param_path", type=click.Path(exists=True))
@click.argument("dataset_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "-e", "--experiment", default="example_euroc.yaml", help="experiment to run"
)
@click.option(
    "-d",
    "--experiments-dir",
    default=None,
    type=click.Path(exists=True),
    help="directory containing experiment files",
)
@click.option("--minloglevel", default=2, help="set VIO minloglevel")
def run(
    executable_path,
    param_path,
    dataset_path,
    output_path,
    experiment,
    experiments_dir,
    minloglevel,
):
    """Run evaluation on datasets."""
    dataset_path = _normalize_path(dataset_path)
    if experiments_dir:
        experiments_dir = _normalize_path(experiments_dir)
    else:
        experiments_dir = _get_experiment_directory()

    experiment_path = experiments_dir / experiment
    if not experiment_path.exists():
        click.secho(f"Could not find experiment '{experiment_path}'", fg="red")
        sys.exit(1)

    config = ExperimentConfig(
        experiment_path, dataset_path, param_path, executable_path
    )
    output_path = _normalize_path(output_path)
    runner = DatasetRunner(config, output_path)
    runner.run_all(minloglevel=minloglevel)


@click.command()
@click.option("-a", "--analyze", is_flag=True, help="Analyze vio, compute APE and RPE")
@click.option("-p", "--plot", is_flag=True, help="show plot window")
@click.option("--save_plots", is_flag=True, help="Save plots?")
@click.option("--write_website", is_flag=True, help="Write website with results?")
@click.option("--save_boxplots", is_flag=True, help="Save boxplots?")
@click.option("--save_results", is_flag=True, help="Save results?")
@click.option("-v", "--verbose", is_flag=True, help="log kimera-vio output to console")
def main():
    """
    Perform full evaluation of SPARK VIO pipeline.

    Reports the following metrics:
      - APE translation
      - RPE translation
      - RPE rotation
    """
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
