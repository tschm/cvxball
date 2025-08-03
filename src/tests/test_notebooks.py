"""Tests for the Python notebook files.

This module tests that all Python notebook files in the notebooks directory
can be executed without errors.
"""

import subprocess
import sys
from pathlib import Path
from security import safe_command


def test_notebooks(root_dir: Path) -> None:
    """Test that all Python notebook files can be executed successfully.

    This test finds all .py files in the notebooks directory and attempts to
    execute them using the Python interpreter. It prints the results of each
    execution but does not assert any specific outcomes.

    Args:
        root_dir: The root directory of the project.

    Note:
        This test is primarily for diagnostic purposes and does not
        fail if a notebook execution is unsuccessful. It only prints
        the results.
    """
    # Get the path to the notebooks directory
    path = root_dir / "notebooks"

    # List all .py files in the directory using glob
    py_files: list[Path] = list(path.glob("*.py"))

    # Loop over the files and run them
    for py_file in py_files:
        print(f"Running {py_file.name}...")
        result = safe_command.run(subprocess.run, [sys.executable, str(py_file)], capture_output=True, text=True)

        # Print the result of running the Python file
        if result.returncode == 0:
            print(f"{py_file.name} ran successfully.")
            print(f"Output: {result.stdout}")
        else:
            print(f"Error running {py_file.name}:")
            print(f"stderr: {result.stderr}")
