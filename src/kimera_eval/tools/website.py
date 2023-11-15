"""Command for constructing website."""
import kimera_eval.paths
from kimera_eval.website import WebsiteBuilder
import logging
import click


@click.command(name="website")
@click.argument("output_path", type=click.Path())
@click.argument("result_paths", type=click.Path(), nargs=-1)
def run(output_path, result_paths):
    """
    Generate website pages for all piplines and datasets.

    Generates a collection of pages for every experiment result
    path added.
    """
    output_path = kimera_eval.paths.normalize_path(output_path)
    result_paths = [kimera_eval.paths.normalize_path(x) for x in result_paths]

    builder = WebsiteBuilder()
    for result_path in result_paths:
        curr_output = output_path / result_path.stem
        curr_output.mkdir(parents=True, exists_ok=True)
        logging.info(f"Writing website for '{result_path}' to '{curr_output}'")
        builder.write(result_path, curr_output)
