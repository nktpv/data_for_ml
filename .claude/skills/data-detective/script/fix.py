"""Apply a cleaning strategy to a parquet dataset.

Usage:
    python .claude/skills/data-detective/scripts/fix.py balanced
    python .claude/skills/data-detective/scripts/fix.py aggressive data/raw/combined.parquet
    python .claude/skills/data-detective/scripts/fix.py conservative

Saves cleaned dataset to data/cleaned/cleaned.parquet.
Prints a JSON summary to stdout.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.data_quality_agent import DataQualityAgent


def main(strategy: str = "balanced", parquet_path: str = "data/raw/combined.parquet") -> None:
    df = pd.read_parquet(ROOT / parquet_path)
    print(f"Loaded {len(df):,} rows", file=sys.stderr)

    agent = DataQualityAgent()
    df_clean = agent.fix(df, strategy=strategy)
    saved = agent.save(df_clean)

    result = {
        "strategy":     strategy,
        "rows_before":  len(df),
        "rows_after":   len(df_clean),
        "rows_removed": len(df) - len(df_clean),
        "saved_to":     str(saved),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    parquet  = sys.argv[2] if len(sys.argv) > 2 else "data/raw/combined.parquet"
    main(strategy, parquet)