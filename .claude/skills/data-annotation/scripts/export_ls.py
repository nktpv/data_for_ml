"""Export annotation tasks to Label Studio JSON format.

Usage::

    python export_ls.py "pos,neg,neutral,mixed"
    python export_ls.py "pos,neg,neutral,mixed" data/labeled/labeled.parquet 0.7
    python export_ls.py "pos,neg,neutral,mixed" data/labeled/labeled.parquet 0.7 --all

Exports low-confidence examples by default.
Pass --all to export every row.
Generates both a tasks JSON and a project config XML.
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.annotation_agent import AnnotationAgent


def main(
    labels_str: str,
    parquet_path: str = "data/labeled/labeled.parquet",
    threshold: float = 0.7,
    export_all: bool = False,
) -> None:
    labels = [lbl.strip() for lbl in labels_str.split(",")]
    df = pd.read_parquet(ROOT / parquet_path)

    agent = AnnotationAgent()

    tasks_path = agent.export_to_labelstudio(
        df,
        confidence_threshold=threshold,
        export_all=export_all,
    )
    config_path = agent.generate_ls_config(labels)

    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    result = {
        "tasks_exported": len(tasks),
        "total_rows": len(df),
        "threshold": threshold,
        "export_all": export_all,
        "tasks_file": str(tasks_path),
        "config_file": str(config_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_ls.py 'Label1,Label2,...' [parquet] [threshold] [--all]")
        sys.exit(1)
    labels_arg = sys.argv[1]
    parquet_arg = sys.argv[2] if len(sys.argv) > 2 else "data/labeled/labeled.parquet"
    threshold_arg = 0.7
    export_all_arg = False
    for arg in sys.argv[3:]:
        if arg == "--all":
            export_all_arg = True
        else:
            try:
                threshold_arg = float(arg)
            except ValueError:
                pass
    main(labels_arg, parquet_arg, threshold_arg, export_all_arg)
