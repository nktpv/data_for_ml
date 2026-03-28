"""Compare raw vs cleaned dataset and write QUALITY_REPORT.md.

Usage:
    python .claude/skills/data-detective/scripts/compare.py
    python .claude/skills/data-detective/scripts/compare.py data/raw/combined.parquet data/cleaned/cleaned.parquet
    python .claude/skills/data-detective/scripts/compare.py raw.parquet cleaned.parquet balanced

Saves data/detective/QUALITY_REPORT.md.
Prints comparison JSON to stdout.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.data_quality_agent import DataQualityAgent


def main(
    raw_path: str = "data/raw/combined.parquet",
    cleaned_path: str = "data/cleaned/cleaned.parquet",
    strategy_name: str = "balanced",
) -> None:
    df_raw     = pd.read_parquet(ROOT / raw_path)
    df_cleaned = pd.read_parquet(ROOT / cleaned_path)

    agent      = DataQualityAgent()
    comparison = agent.compare(df_raw, df_cleaned)

    out_dir = ROOT / "data" / "detective"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Quality Report",
        "",
        "## Task: Sentiment Analysis (IMDB reviews)",
        "",
        f"## Cleaning strategy applied: `{strategy_name}`",
        "",
        "## Before → After",
        "",
        "| Metric | Before | After | Change |",
        "|--------|--------|-------|--------|",
    ]
    for metric, vals in comparison.metrics.items():
        lines.append(
            f"| {metric} | {vals.get('before', '—')} "
            f"| {vals.get('after', '—')} "
            f"| {vals.get('change', '—')} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- **Exact duplicates** were removed to prevent data leakage between train/test splits.",
        "- **HTML entities** (`&amp;`, `&#39;`, `<br>`) were decoded — raw markup confuses subword tokenisers.",
        "- **IQR outliers** (multiplier 3×) were trimmed; moderate-length reviews were kept because "
          "longer texts often carry richer sentiment signal.",
        "- **Class balance** was not altered — the IMDB dataset is inherently ~50/50 (pos/neg), "
          "which is ideal for binary classification.",
    ]

    report_path = out_dir / "QUALITY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    result = {
        "strategy":     strategy_name,
        "metrics":      comparison.metrics,
        "report_saved": str(report_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raw      = sys.argv[1] if len(sys.argv) > 1 else "data/raw/combined.parquet"
    cleaned  = sys.argv[2] if len(sys.argv) > 2 else "data/cleaned/cleaned.parquet"
    strategy = sys.argv[3] if len(sys.argv) > 3 else "balanced"
    main(raw, cleaned, strategy)