"""Test fixtures for package."""
import pytest
import pathlib
import kimera_eval
import kimera_eval.__main__
import functools
import click.testing


@pytest.fixture
def resources():
    """Get the test resources path."""
    return pathlib.Path(__file__).resolve().parent / "resources"


@pytest.fixture
def pkgpath():
    """Get path to the package."""
    return pathlib.Path(kimera_eval.__file__).absolute().parent


@pytest.fixture
def cli():
    """Get CLI test runner."""
    kimera_eval.configure_logging(level="INFO")
    runner = click.testing.CliRunner()
    return functools.partial(runner.invoke, kimera_eval.__main__.main)
