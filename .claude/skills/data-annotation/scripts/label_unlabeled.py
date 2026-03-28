"""Annotate only unlabeled rows in the dataset, preserving existing labels.

Usage::

    python label_unlabeled.py "pos,neg,neutral,mixed" "sentiment analysis" data/cleaned/cleaned.parquet 10
    python label_unlabeled.py "pos,neg" "sentiment analysis" data/cleaned/cleaned.parquet 10 --allow-new-labels
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from agents.annotation_agent import AnnotationAgent


def main(
    labels_str: str,
    task: str = "sentiment analysis",
    parquet_path: str = "data/cleaned/cleaned.parquet",
    batch_size: int = 10,
    allow_new_labels: bool | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    labels = [lbl.strip() for lbl in labels_str.split(",")]
    df = pd.read_parquet(ROOT / parquet_path)

    if allow_new_labels is None:
        try:
            with open(ROOT / "config.yaml") as f:
                config = yaml.safe_load(f)
            allow_new_labels = config.get("task", {}).get("allow_new_labels", False)
        except Exception:
            allow_new_labels = False

    unlabeled_mask = df["label"].isna()
    df_unlabeled = df[unlabeled_mask].copy()
    df_already_labeled = df[~unlabeled_mask].copy()

    print(f"Dataset:          {len(df)} rows total")
    print(f"  Already labeled: {len(df_already_labeled)}")
    print(f"  To annotate:     {len(df_unlabeled)}")
    print(f"Labels:           {labels}")
    print(f"Task:             {task}")
    print(f"Batch size:       {batch_size}")
    print(f"Allow new labels: {allow_new_labels}")
    print()
    print("Annotating unlabeled rows via OpenRouter API...")

    agent = AnnotationAgent()

    df_newly_labeled = agent.auto_label(
        df_unlabeled,
        labels=labels,
        task_description=task,
        batch_size=batch_size,
        allow_new_labels=allow_new_labels,
    )

    df_already_labeled["auto_label"] = df_already_labeled["label"]
    df_already_labeled["confidence"] = 1.0
    df_already_labeled["is_disputed"] = False

    df_full = pd.concat([df_already_labeled, df_newly_labeled]).sort_index()

    saved = agent.save(df_full)
    print(f"\nSaved: {saved}")

    dist = df_newly_labeled["auto_label"].value_counts()
    print("\nNew label distribution (unlabeled rows):")
    for label, count in dist.items():
        pct = count / len(df_newly_labeled) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    conf = df_newly_labeled["confidence"]
    print(f"\nConfidence (new labels): mean={conf.mean():.3f}, median={conf.median():.3f}")
    disputed = int(df_newly_labeled["is_disputed"].sum())
    print(f"Disputed (confidence < 0.7): {disputed} ({disputed / len(df_newly_labeled) * 100:.1f}%)")

    return df_full, labels


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python label_unlabeled.py 'Label1,Label2,...' [task] [parquet_path] [batch_size] [--allow-new-labels]"
        )
        sys.exit(1)
    allow_new = "--allow-new-labels" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--allow-new-labels"]
    labels_arg = args[0]
    task_arg = args[1] if len(args) > 1 else "sentiment analysis"
    parquet_arg = args[2] if len(args) > 2 else "data/cleaned/cleaned.parquet"
    batch_arg = int(args[3]) if len(args) > 3 else 10
    main(labels_arg, task_arg, parquet_arg, batch_arg, allow_new_labels=True if allow_new else None)
