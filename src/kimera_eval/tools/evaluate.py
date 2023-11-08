"""Main entry point for running evaluations."""
from kimera_eval.dataset_evaluator import DatasetEvaluator
import click
import yaml


@click.command(name="evaluate")
@click.option("-a", "--analyze", is_flag=True, help="Analyze vio, compute APE and RPE")
@click.option("-p", "--plot", is_flag=True, help="show plot window")
@click.option("--save_plots", is_flag=True, help="Save plots?")
@click.option("--write_website", is_flag=True, help="Write website with results?")
@click.option("--save_boxplots", is_flag=True, help="Save boxplots?")
@click.option("--save_results", is_flag=True, help="Save results?")
def run(analyze, plot, save_plots, write_website, save_boxplots, save_results):
    """
    Perform full evaluation of SPARK VIO pipeline.

    Reports the following metrics:
      - APE translation
      - RPE translation
      - RPE rotation
    """
    with experiment_path.open("r") as fin:
        experiment_params = yaml.safe_load(fin.read())

    dataset_evaluator = DatasetEvaluator(
        experiment_params, results_path, extra_flagfile_path
    )
    dataset_evaluator.evaluate()
    dataset_evaluator.aggregate(results_path)
