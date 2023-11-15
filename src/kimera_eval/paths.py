"""Main entry point for running evaluations."""
import pathlib


def experiment_directory():
    """Get location of experiment files."""
    return pathlib.Path(__file__).absolute().parent / "experiments"


def normalize_path(input_path):
    """Normalize path."""
    return pathlib.Path(input_path).expanduser().absolute()
