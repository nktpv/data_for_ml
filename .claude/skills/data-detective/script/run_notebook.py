"""Execute a Jupyter notebook in-place and save with outputs.

Usage:
    python .claude/skills/data-detective/scripts/run_notebook.py notebooks/data_quality.ipynb

The notebook is executed with its own directory as the working directory,
so relative paths like ../data/raw/combined.parquet resolve correctly.
"""

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run(notebook_path: str, timeout: int = 600) -> None:
    path = Path(notebook_path).resolve()
    if not path.exists():
        print(f"Notebook not found: {path}")
        sys.exit(1)

    print(f"Executing {path} ...")

    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Saved with outputs: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook.ipynb>")
        sys.exit(1)
    run(sys.argv[1])