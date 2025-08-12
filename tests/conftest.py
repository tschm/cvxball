"""Pytest configuration file for the tests package."""

from pathlib import Path

import pytest


@pytest.fixture(name="root_dir")
def root_fixture() -> Path:
    """Fixture that provides the root directory path for the project.

    This fixture is useful for obtaining the root directory of the project
    relative to the current file's location.

    Parameters:
        None

    Returns:
        Path: A Path object representing the root directory of the project.
    """
    return Path(__file__).parent.parent
