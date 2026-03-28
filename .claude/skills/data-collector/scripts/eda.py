"""EDA script for unified text-classification datasets.

Usage:
    python scripts/eda.py data/raw/combined.parquet data/eda

Generates in <eda_dir>:
    class_distribution.png
    text_length_distribution.png
    top_words.png
    source_distribution.png
    REPORT.md
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "that", "this", "it", "its",
    "not", "as", "if", "so", "he", "she", "they", "we", "you", "i",
    # Russian
    "и", "в", "на", "с", "что", "как", "по", "это", "из", "за",
    "не", "но", "к", "от", "до", "то", "а", "о", "же", "он", "она",
}


def _clean(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]


def plot_class_distribution(df: pd.DataFrame, out: Path) -> dict:
    if df["label"].isna().all():
        return {}
    counts = df["label"].dropna().value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.bar(ax=ax, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_title("Class distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)
    return counts.to_dict()


def plot_text_lengths(df: pd.DataFrame, out: Path) -> dict:
    lengths = df["text"].astype(str).str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=60, color="#4C72B0", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(lengths.median(), color="red", linestyle="--", label=f"Median: {lengths.median():.0f}")
    ax.set_title("Text length distribution (characters)")
    ax.set_xlabel("Length")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "text_length_distribution.png", dpi=150)
    plt.close(fig)
    return {
        "mean": round(lengths.mean(), 1),
        "median": round(lengths.median(), 1),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "std": round(lengths.std(), 1),
    }


def plot_top_words(df: pd.DataFrame, out: Path, top_n: int = 20) -> list:
    all_words: list[str] = []
    for text in df["text"].astype(str):
        all_words.extend(_clean(text))
    most_common = Counter(all_words).most_common(top_n)
    if not most_common:
        return []
    words, freqs = zip(*most_common)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(words)), freqs, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(f"Top-{top_n} most frequent words")
    ax.set_xlabel("Frequency")
    plt.tight_layout()
    fig.savefig(out / "top_words.png", dpi=150)
    plt.close(fig)
    return list(most_common)


def plot_source_distribution(df: pd.DataFrame, out: Path) -> dict:
    counts = df["source"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("Source distribution")
    plt.tight_layout()
    fig.savefig(out / "source_distribution.png", dpi=150)
    plt.close(fig)
    return counts.to_dict()


def generate_report(
    df: pd.DataFrame,
    out: Path,
    class_dist: dict,
    length_stats: dict,
    top_words: list,
    source_dist: dict,
) -> None:
    total = sum(class_dist.values()) if class_dist else 0
    class_table = ""
    if class_dist:
        class_table = "| Label | Count | % |\n|-------|-------|---|\n"
        for label, count in sorted(class_dist.items(), key=lambda x: -x[1]):
            class_table += f"| {label} | {count:,} | {count/total*100:.1f}% |\n"
    else:
        class_table = "_No labels in this dataset._\n"

    word_table = "| Word | Frequency |\n|------|----------|\n"
    for word, freq in top_words:
        word_table += f"| {word} | {freq:,} |\n"

    source_table = "| Source | Count |\n|--------|-------|\n"
    for source, count in sorted(source_dist.items(), key=lambda x: -x[1]):
        source_table += f"| {source} | {count:,} |\n"

    report = f"""# EDA Report

## Dataset overview

- **Total samples**: {len(df):,}
- **Columns**: {', '.join(df.columns)}
- **Unique labels**: {df['label'].nunique()} ({df['label'].isna().sum()} unlabelled)
- **Unique sources**: {df['source'].nunique()}
- **Missing values**: {df.isnull().sum().to_dict()}

## Class distribution

![Class distribution](class_distribution.png)

{class_table}

## Text length statistics

![Text length distribution](text_length_distribution.png)

| Metric | Value |
|--------|-------|
| Mean | {length_stats.get('mean', 'n/a')} |
| Median | {length_stats.get('median', 'n/a')} |
| Min | {length_stats.get('min', 'n/a')} |
| Max | {length_stats.get('max', 'n/a')} |
| Std | {length_stats.get('std', 'n/a')} |

## Top-{len(top_words)} words

![Top words](top_words.png)

{word_table}

## Source distribution

![Source distribution](source_distribution.png)

{source_table}
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")


def main(parquet_path: str, eda_dir: str) -> None:
    out = Path(eda_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows from {parquet_path}")

    class_dist = plot_class_distribution(df, out)
    print("  ✓ class_distribution.png")

    length_stats = plot_text_lengths(df, out)
    print("  ✓ text_length_distribution.png")

    top_words = plot_top_words(df, out)
    print("  ✓ top_words.png")

    source_dist = plot_source_distribution(df, out)
    print("  ✓ source_distribution.png")

    generate_report(df, out, class_dist, length_stats, top_words, source_dist)
    print("  ✓ REPORT.md")
    print(f"EDA done → {out}/")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eda.py <parquet_path> <eda_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])