"""Write a performance summary to file."""
import os
import sys
import click
import csv
import yaml
import pathlib


@click.command()
@click.argument("result_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None)
def main(result_path, output):
    """Plot summary of performance results for VIO pipeline."""
    result_path = pathlib.Path(result_path).expanduser().absolute()
    click.secho(f"Reading VIO results from: '{result_path}'")

    with result_path.open("r") as fin:
        results = yaml.safe_load(fin)

    ATE_mean = results["absolute_errors"].stats["mean"]
    ATE_rmse = results["absolute_errors"].stats["rmse"]

    if output:
        output = pathlib.Path(output).expanduser().absolute()
    else:
        output = result_path.parent / "results_summary.csv"

    click.secho(f"Writing VIO summary results to '{output}'")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fout:
        writer = csv.DictWriter(fout, fieldnames=["ATE_mean", "ATE_rmse"])
        writer.writeheader()
        writer.writerow({"ATE_mean": ATE_mean, "ATE_rmse": ATE_rmse})

    sys.exit(os.EX_OK)
