"""Detect data quality issues in a parquet dataset.

Usage:
    python .claude/skills/data-detective/scripts/detect.py
    python .claude/skills/data-detective/scripts/detect.py data/raw/combined.parquet

Prints a JSON report to stdout.
Saves data/detective/problems.json.
"""

import json
import sys
from pathlib import Path

# Resolve project root (skills/data-detective/scripts/ → root)
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.data_quality_agent import DataQualityAgent


def main(parquet_path: str = "data/raw/combined.parquet") -> None:
    df = pd.read_parquet(ROOT / parquet_path)
    print(f"Loaded {len(df):,} rows from {parquet_path}", file=sys.stderr)

    agent = DataQualityAgent()
    report = agent.detect_issues(df)

    out_dir = ROOT / "data" / "detective"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_dict = report.to_dict()
    problems_path = out_dir / "problems.json"
    with open(problems_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved {problems_path}", file=sys.stderr)
    print(json.dumps(report_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/combined.parquet"
    main(path)