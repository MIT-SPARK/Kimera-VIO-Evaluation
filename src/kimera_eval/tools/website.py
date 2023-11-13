"""Command for constructing website."""
from kimera_eval.website import WebsiteBuilder
import logging
import pathlib
import click


def _normalize_path(input_path):
    return pathlib.Path(input_path).expanduser().absolute()


@click.command(name="website")
@click.argument("output_path", type=click.Path())
@click.argument("result_paths", type=click.Path(), nargs=-1)
def run(output_path, result_paths):
    """
    Generate website pages for all piplines and datasets.

    Generates a collection of pages for every experiment result
    path added.
    """
    output_path = _normalize_path(output_path)
    result_paths = [_normalize_path(x) for x in result_paths]

    builder = WebsiteBuilder()
    for result_path in result_paths:
        curr_output = output_path / result_path.stem
        curr_output.mkdir(parents=True, exists_ok=True)
        logging.info(f"Writing website for '{result_path}' to '{curr_output}'")
        builder.write(result_path, curr_output)
