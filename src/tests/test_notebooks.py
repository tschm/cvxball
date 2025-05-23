import subprocess
import sys


def test_notebooks(root_dir):
    # loop over all notebooks
    path = root_dir / "notebooks"

    # List all .py files in the directory using glob
    py_files = list(path.glob("*.py"))

    # Loop over the files and run them
    for py_file in py_files:
        print(f"Running {py_file.name}...")
        result = subprocess.run([sys.executable, str(py_file)], capture_output=True, text=True)

        # Print the result of running the Python file
        if result.returncode == 0:
            print(f"{py_file.name} ran successfully.")
            print(f"Output: {result.stdout}")
        else:
            print(f"Error running {py_file.name}:")
            print(f"stderr: {result.stderr}")
