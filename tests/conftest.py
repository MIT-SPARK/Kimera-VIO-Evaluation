"""Test fixtures for package."""
import pytest
import pathlib
import kimera_eval


@pytest.fixture
def resources():
    """Get the test resources path."""
    return pathlib.Path(__file__).resolve().parent / "resources"


@pytest.fixture
def pkgpath():
    """Get path to the package."""
    return pathlib.Path(kimera_eval.__file__).absolute().parent
