"""Show timing plots."""
from kimera_eval.result_aggregation import get_pipeline_times
from kimera_eval.plotting import draw_timing_plot
import click
import pathlib
import sys
import os


@click.command("timing")
@click.argument("results_path", type=click.Path(exists=True))
@click.option("-p", "--pipeline-names", multiple=True, default=None)
@click.option("-t", "--time-column", default=3, type=int, help="time column to display")
def run(results_path, pipeline_names, time_column):
    """Display timing of all VIO results stored under RESULTS_PATH."""
    results_path = pathlib.Path(results_path).expanduser().absolute()
    keyframes, times = get_pipeline_times(results_path, pipeline_names, time_column)
    if not len(keyframes):
        click.secho("Missing keyframe data", fg="red")
        sys.exit(os.EX_NOINPUT)

    if not len(times) > 0:
        click.secho("Missing pipeline timing information", fg="red")
        sys.exit(os.EX_NOINPUT)

    final_plot_path = results_path / "timing" / "pipeline_times.pdf"
    click.secho(f"saving timing plot to '{final_plot_path}'")
    draw_timing_plot(final_plot_path, keyframes, times)
