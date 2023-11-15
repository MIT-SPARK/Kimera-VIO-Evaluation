"""Write a performance summary to file."""
from kimera_eval.trajectory_metrics import TrajectoryResults
import click
import csv
import pathlib


@click.command(name="summary")
@click.argument("result_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None)
def run(result_path, output):
    """Plot summary of performance results for VIO pipeline."""
    result_path = pathlib.Path(result_path).expanduser().absolute()
    click.secho(f"Reading VIO results from: '{result_path}'")

    results = TrajectoryResults.load(result_path)
    stats = results.ape_translation.get_all_statistics()

    if output:
        output = pathlib.Path(output).expanduser().absolute()
    else:
        output = result_path.parent / "results_summary.csv"

    click.secho(f"Writing VIO summary results to '{output}'")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fout:
        writer = csv.DictWriter(fout, fieldnames=["ATE_mean", "ATE_rmse"])
        writer.writeheader()
        writer.writerow({"ATE_mean": stats["mean"], "ATE_rmse": stats["rmse"]})
