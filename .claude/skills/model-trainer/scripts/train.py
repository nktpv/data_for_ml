"""Train final model: LLM selects model, trains on all data, saves artifacts.

Usage::

    python train.py
    python train.py data/labeled/labeled.parquet
    python train.py data/labeled/labeled.parquet --no-llm

Pass --no-llm to skip LLM selection and use logreg_balanced.
OpenRouter API (model from OPENROUTER_MODEL env) analyzes AL results and dataset
to choose the best model.
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from agents.model_trainer_agent import ModelTrainerAgent, MODELS
from agents.openrouter_client import DEFAULT_MODEL


def main(
    parquet_path: str = "data/labeled/labeled.parquet",
    use_llm: bool = True,
) -> None:
    # Read task type
    task_type = "sentiment analysis"
    try:
        with open(ROOT / "config.yaml") as f:
            config = yaml.safe_load(f)
        task_type = config.get("task", {}).get("type", task_type)
    except Exception:
        pass

    df = pd.read_parquet(ROOT / parquet_path)
    label_col = "auto_label"
    df_labeled = df[df[label_col].notna()].copy()

    print(f"Dataset: {len(df_labeled)} labeled rows")
    print(f"Labels: {df_labeled[label_col].nunique()} classes")
    print(f"Task: {task_type}")
    print()

    agent = ModelTrainerAgent()

    # Step 1: Model selection
    if use_llm:
        print(f"Asking OpenRouter API ({DEFAULT_MODEL}) to select model...")
        recommendation = agent.select_model(df_labeled, task_type=task_type)
        model_key = recommendation["model"]
        print(f"  Model: {MODELS[model_key]['name']} ({model_key})")
        print(f"  Reasoning: {recommendation.get('reasoning', '')}")
    else:
        model_key = "logreg_balanced"
        recommendation = {"model": model_key, "reasoning": "Default (--no-llm)"}
        print(f"Using default: {model_key}")
    print()

    # Step 2: Split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df_labeled, test_size=0.2, random_state=42, stratify=df_labeled[label_col],
    )
    print(f"Split: {len(train_df)} train / {len(test_df)} test")

    # Step 3: Train
    print(f"Training {MODELS[model_key]['name']}...")
    agent.train(train_df, model_key=model_key)
    print("  Done.")

    # Step 4: Evaluate
    print("Evaluating...")
    metrics = agent.evaluate(test_df)
    metrics["train_size"] = len(train_df)

    print(f"\n  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  Precision:       {metrics['precision_macro']:.4f}")
    print(f"  Recall:          {metrics['recall_macro']:.4f}")

    # Step 5: Save model
    print("\nSaving model...")
    model_paths = agent.save_model()
    for name, p in model_paths.items():
        print(f"  {name}: {p}")

    # Step 6: Report
    print("\nGenerating report...")
    report_path = agent.report(recommendation, metrics, train_size=len(train_df))
    print(f"  Report: {report_path}")

    # Per-class summary
    print("\nPer-class F1:")
    for cls, vals in metrics.get("per_class", {}).items():
        if isinstance(vals, dict):
            print(f"  {cls}: {vals.get('f1-score', 0):.4f} (support={int(vals.get('support', 0))})")


if __name__ == "__main__":
    no_llm = "--no-llm" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-llm"]
    parquet = args[0] if len(args) > 0 else "data/labeled/labeled.parquet"
    main(parquet, use_llm=not no_llm)
