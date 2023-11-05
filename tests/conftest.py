"""Test fixtures for package."""
import pytest
import pathlib


@pytest.fixture
def resources():
    """Get the test resources path."""
    return pathlib.Path(__file__).resolve().parent / "resources"
