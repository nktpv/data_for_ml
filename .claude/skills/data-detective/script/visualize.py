"""Generate data quality charts and save them to data/detective/.

Usage:
    python .claude/skills/data-detective/scripts/visualize.py
    python .claude/skills/data-detective/scripts/visualize.py data/raw/combined.parquet
    python .claude/skills/data-detective/scripts/visualize.py data/raw/combined.parquet data/cleaned/cleaned.parquet

When only raw_path is given: produces issue-detection charts (missing_values, outliers, class_balance).
When cleaned_path is also given: additionally produces before/after charts.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = ROOT / "data" / "detective"
PALETTE = {"raw": "#e05c5c", "clean": "#4caf7d", "neutral": "#4c8abf"}


# ---------------------------------------------------------------------------
# Issue-detection charts (raw data only)
# ---------------------------------------------------------------------------

def plot_missing(df: pd.DataFrame, out: Path) -> None:
    skip = {"label"}
    missing = df.drop(columns=[c for c in skip if c in df.columns]).isnull().sum()
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values detected",
                ha="center", va="center", fontsize=14, color="#555")
        ax.set_axis_off()
    else:
        missing.plot.bar(ax=ax, color=PALETTE["raw"], edgecolor="white")
        ax.set_title("Missing values per column (label column excluded)")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(out / "missing_values.png", dpi=120)
    plt.close(fig)
    print("  ✓ missing_values.png")


def plot_outliers(df: pd.DataFrame, out: Path, text_col: str = "text") -> None:
    if text_col not in df.columns:
        return
    lengths = df[text_col].astype(str).str.len()
    q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
    iqr     = q3 - q1
    lower   = q1 - 1.5 * iqr
    upper   = q3 + 1.5 * iqr
    n_out   = int(((lengths < lower) | (lengths > upper)).sum())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=60, color=PALETTE["neutral"], edgecolor="white", alpha=0.85)
    ax.axvline(lower, color=PALETTE["raw"], linestyle="--", linewidth=1.2,
               label=f"IQR lower bound ({lower:.0f})")
    ax.axvline(upper, color=PALETTE["raw"], linestyle="--", linewidth=1.2,
               label=f"IQR upper bound ({upper:.0f})")
    ax.axvline(lengths.median(), color="#888", linestyle=":", linewidth=1,
               label=f"Median ({lengths.median():.0f})")
    ax.set_title(f"Review length distribution — {n_out} outliers outside IQR bounds")
    ax.set_xlabel("Character length")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "outliers.png", dpi=120)
    plt.close(fig)
    print("  ✓ outliers.png")


def plot_class_balance(df: pd.DataFrame, out: Path, label_col: str = "label") -> None:
    if label_col not in df.columns:
        return
    counts = df[label_col].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE["clean"], PALETTE["raw"]][:len(counts)],
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + counts.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Sentiment class distribution (pos / neg)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=120)
    plt.close(fig)
    print("  ✓ class_balance.png")


# ---------------------------------------------------------------------------
# Before / after comparison charts
# ---------------------------------------------------------------------------

def plot_row_counts(df_raw: pd.DataFrame, df_clean: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [len(df_raw), len(df_clean)]
    bars   = ax.bar(["Raw", "Cleaned"], values,
                    color=[PALETTE["raw"], PALETTE["clean"]], edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Row count: raw vs cleaned")
    ax.set_ylabel("Rows")
    ax.set_ylim(0, max(values) * 1.12)
    plt.tight_layout()
    fig.savefig(out / "before_after_rows.png", dpi=120)
    plt.close(fig)
    print("  ✓ before_after_rows.png")


def plot_class_comparison(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                          out: Path, label_col: str = "label") -> None:
    if label_col not in df_raw.columns:
        return
    counts_raw   = df_raw[label_col].dropna().value_counts()
    counts_clean = df_clean[label_col].dropna().value_counts()
    labels       = sorted(set(counts_raw.index) | set(counts_clean.index))

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, [counts_raw.get(l, 0) for l in labels],   w,
           label="Raw",     color=PALETTE["raw"],   alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, [counts_clean.get(l, 0) for l in labels], w,
           label="Cleaned", color=PALETTE["clean"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title("Class distribution: raw vs cleaned")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "before_after_class_balance.png", dpi=120)
    plt.close(fig)
    print("  ✓ before_after_class_balance.png")


def plot_length_comparison(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                           out: Path, text_col: str = "text") -> None:
    if text_col not in df_raw.columns:
        return
    len_raw   = df_raw[text_col].astype(str).str.len()
    len_clean = df_clean[text_col].astype(str).str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(len_raw,   bins=60, alpha=0.55, label="Raw",     color=PALETTE["raw"])
    ax.hist(len_clean, bins=60, alpha=0.55, label="Cleaned", color=PALETTE["clean"])
    ax.set_title("Review length distribution: raw vs cleaned")
    ax.set_xlabel("Character length")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "before_after_text_lengths.png", dpi=120)
    plt.close(fig)
    print("  ✓ before_after_text_lengths.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(raw_path: str = "data/raw/combined.parquet",
         cleaned_path: str | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_parquet(ROOT / raw_path)
    print(f"Loaded raw: {len(df_raw):,} rows")

    plot_missing(df_raw, OUT_DIR)
    plot_outliers(df_raw, OUT_DIR)
    plot_class_balance(df_raw, OUT_DIR)

    if cleaned_path:
        df_clean = pd.read_parquet(ROOT / cleaned_path)
        print(f"Loaded cleaned: {len(df_clean):,} rows")
        plot_row_counts(df_raw, df_clean, OUT_DIR)
        plot_class_comparison(df_raw, df_clean, OUT_DIR)
        plot_length_comparison(df_raw, df_clean, OUT_DIR)

    print(f"\nAll charts saved to {OUT_DIR}/")


if __name__ == "__main__":
    raw     = sys.argv[1] if len(sys.argv) > 1 else "data/raw/combined.parquet"
    cleaned = sys.argv[2] if len(sys.argv) > 2 else None
    main(raw, cleaned)