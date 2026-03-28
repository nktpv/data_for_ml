"""Merge manual Label Studio annotations back into the labeled dataset.

Usage::

    python import_ls.py
    python import_ls.py data/annotation/ls_export.json data/labeled/labeled.parquet

Overwrites auto_label/confidence for manually reviewed rows (confidence set to 1.0).
Saves updated labeled.parquet.
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from agents.annotation_agent import AnnotationAgent


def main(
    ls_export_path: str = "data/annotation/ls_export.json",
    parquet_path: str = "data/labeled/labeled.parquet",
) -> None:
    df = pd.read_parquet(ROOT / parquet_path)
    agent = AnnotationAgent()

    before_disputed = int(df["is_disputed"].sum()) if "is_disputed" in df.columns else 0
    before_manual = int((df["confidence"] == 1.0).sum()) if "confidence" in df.columns else 0

    print(f"Before: {len(df)} rows, disputed: {before_disputed}")

    df_merged = agent.import_from_labelstudio(df, ROOT / ls_export_path)
    saved = agent.save(df_merged)

    after_disputed = int(df_merged["is_disputed"].sum()) if "is_disputed" in df_merged.columns else 0
    after_manual = int((df_merged["confidence"] == 1.0).sum()) if "confidence" in df_merged.columns else 0
    newly_reviewed = max(0, after_manual - before_manual)

    print(f"Manual labels merged: {newly_reviewed}")
    print(f"After:  disputed: {after_disputed}")
    print(f"Saved:  {saved}")


if __name__ == "__main__":
    ls_path = sys.argv[1] if len(sys.argv) > 1 else "data/annotation/ls_export.json"
    parquet = sys.argv[2] if len(sys.argv) > 2 else "data/labeled/labeled.parquet"
    main(ls_path, parquet)
