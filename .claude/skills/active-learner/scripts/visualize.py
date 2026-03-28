"""Regenerate learning curve chart from saved history JSONs.

Usage::

    python visualize.py
    python visualize.py data/active_learning
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(data_dir: str = "data/active_learning") -> None:
    d = ROOT / data_dir

    # Load all history files
    results = {}
    for p in sorted(d.glob("history_*.json")):
        strategy = p.stem.replace("history_", "")
        with open(p, "r", encoding="utf-8") as f:
            results[strategy] = json.load(f)

    if not results:
        print(f"No history files found in {d}")
        sys.exit(1)

    print(f"Loaded strategies: {list(results.keys())}")

    # Plot
    markers = {"entropy": "o", "margin": "s", "random": "^"}
    colors = {"entropy": "#2196F3", "margin": "#FF9800", "random": "#9E9E9E"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for strategy, history in results.items():
        n_labeled = [h["n_labeled"] for h in history]
        acc = [h["accuracy"] for h in history]
        f1 = [h["f1"] for h in history]
        marker = markers.get(strategy, "o")
        color = colors.get(strategy, "#000000")

        axes[0].plot(n_labeled, acc, marker=marker, color=color, label=strategy, linewidth=2)
        axes[1].plot(n_labeled, f1, marker=marker, color=color, label=strategy, linewidth=2)

    axes[0].set_xlabel("Number of labeled examples")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Learning Curve — Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of labeled examples")
    axes[1].set_ylabel("F1 (macro)")
    axes[1].set_title("Learning Curve — F1 Macro")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = d / "learning_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/active_learning"
    main(data_dir)
