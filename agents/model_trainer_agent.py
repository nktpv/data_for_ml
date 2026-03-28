"""ModelTrainerAgent — final model selection, training, and evaluation.

Reads AL experiment results, uses OpenRouter API (model from OPENROUTER_MODEL env) to choose
the best model, trains it on all labeled data, and saves artifacts (model, metrics, charts, report).

Skills:
    - select_model(al_results, df) → recommendation (via OpenRouter API)
    - train(df, model_key) → fitted model
    - evaluate(test_df) → metrics + charts
    - save_model(path) → saved .pkl
    - report() → MODEL_REPORT.md

Usage::

    from agents.model_trainer_agent import ModelTrainerAgent

    agent = ModelTrainerAgent()
    recommendation = agent.select_model(al_results, df)
    agent.train(df_train, recommendation["model"])
    metrics = agent.evaluate(df_test)
    agent.save_model()
    agent.report(recommendation, metrics)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from .openrouter_client import DEFAULT_MODEL  # noqa: E402

MODELS = {
    "logreg": {
        "name": "Logistic Regression",
        "factory": lambda rs: LogisticRegression(max_iter=1000, random_state=rs, solver="lbfgs"),
    },
    "logreg_balanced": {
        "name": "Logistic Regression (balanced)",
        "factory": lambda rs: LogisticRegression(
            max_iter=1000, random_state=rs, solver="lbfgs", class_weight="balanced",
        ),
    },
    "svm": {
        "name": "Linear SVM (calibrated)",
        "factory": lambda rs: CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=rs), cv=3,
        ),
    },
    "nb": {
        "name": "Multinomial Naive Bayes",
        "factory": lambda rs: MultinomialNB(alpha=1.0),
    },
}


class ModelTrainerAgent:
    """Final model trainer with LLM-powered model selection.

    Skills:
        - ``select_model(al_results, df)`` — OpenRouter LLM picks best model
        - ``train(df, model_key)`` — train on all data
        - ``evaluate(test_df)`` — metrics + confusion matrix
        - ``save_model()`` — save to models/
        - ``report(recommendation, metrics)`` — MODEL_REPORT.md
    """

    def __init__(
        self,
        text_col: str = "text",
        label_col: str = "auto_label",
        random_state: int = 42,
    ):
        self.text_col = text_col
        self.label_col = label_col
        self.random_state = random_state
        self._vectorizer: TfidfVectorizer | None = None
        self._model = None
        self._is_fitted = False
        self._model_key: str = "logreg"
        self._classes: list[str] = []

    # ── Skill: select_model (LLM-powered) ─────────────────────────

    def select_model(
        self,
        df: pd.DataFrame,
        al_dir: str | Path = "data/active_learning",
        task_type: str = "sentiment analysis",
    ) -> dict:
        """Use OpenRouter API to choose the best model based on AL results and data.

        Args:
            df: Full labeled dataset.
            al_dir: Directory with AL experiment results.
            task_type: Task description from config.

        Returns:
            Dict with 'model', 'reasoning' keys.
        """
        import requests

        api_key = os.getenv("OPENROUTER_API_KEY")

        al_path = ROOT_DIR / al_dir

        # Load AL histories
        al_summary = {}
        for p in sorted(al_path.glob("history_*.json")):
            strategy = p.stem.replace("history_", "")
            with open(p, "r", encoding="utf-8") as f:
                history = json.load(f)
            al_summary[strategy] = {
                "final_accuracy": history[-1]["accuracy"],
                "final_f1": history[-1]["f1"],
                "n_labeled": history[-1]["n_labeled"],
            }

        # Load AL report for LLM recommendation
        al_recommendation = ""
        report_path = al_path / "REPORT.md"
        if report_path.exists():
            al_recommendation = report_path.read_text(encoding="utf-8")

        # Dataset stats
        label_counts = df[self.label_col].value_counts()
        n_classes = len(label_counts)
        total = len(df)
        imbalance = round(int(label_counts.max()) / int(label_counts.min()), 1)
        avg_text_len = int(df[self.text_col].astype(str).str.len().mean())

        models_desc = "\n".join(
            f"  - {key}: {info['name']}" for key, info in MODELS.items()
        )

        prompt = f"""You are an ML engineer choosing the best model to train on the FULL dataset.

Task: {task_type}

Dataset:
- Total samples: {total}
- Classes: {n_classes}
- Distribution: {dict(label_counts)}
- Imbalance ratio: {imbalance}x
- Avg text length: {avg_text_len} chars

Active Learning experiment results (on small subsets):
{json.dumps(al_summary, indent=2)}

AL Report excerpt:
{al_recommendation[:1500]}

Available models:
{models_desc}

Now we will train on ALL {total} samples (80% train, 20% test).
Choose the best model considering:
1. AL experiment showed which model type works best for this data
2. Class imbalance — does the model need balanced weights?
3. Dataset size — all models can handle {total} samples easily
4. We need predict_proba for confidence scores

Respond with ONLY a JSON object:
{{
  "model": "<model_key>",
  "reasoning": "<2-3 sentences>"
}}"""

        response_text = None
        if api_key:
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": DEFAULT_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                response_text = resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                response_text = None

        if response_text:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    if result.get("model") not in MODELS:
                        result["model"] = "logreg_balanced"
                    return result
                except json.JSONDecodeError:
                    pass

        return {
            "model": "logreg_balanced" if imbalance > 3 else "logreg",
            "reasoning": f"Fallback: imbalance={imbalance}x, {n_classes} classes.",
        }

    # ── Skill: train ──────────────────────────────────────────────

    def train(
        self,
        train_df: pd.DataFrame,
        model_key: str = "logreg",
    ) -> "ModelTrainerAgent":
        """Train TF-IDF + model on training data.

        Args:
            train_df: Training DataFrame.
            model_key: Key from MODELS dict.

        Returns:
            self (fitted).
        """
        self._model_key = model_key
        self._vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
        )
        self._model = MODELS[model_key]["factory"](self.random_state)

        texts = train_df[self.text_col].astype(str).tolist()
        labels = train_df[self.label_col].astype(str).tolist()
        self._classes = sorted(set(labels))

        X = self._vectorizer.fit_transform(texts)
        self._model.fit(X, labels)
        self._is_fitted = True
        return self

    # ── Skill: evaluate ───────────────────────────────────────────

    def evaluate(
        self,
        test_df: pd.DataFrame,
        output_dir: str | Path = "data/model",
    ) -> dict:
        """Evaluate model and save metrics + charts.

        Args:
            test_df: Test DataFrame.
            output_dir: Where to save artifacts.

        Returns:
            Dict with accuracy, f1, precision, recall, per-class metrics.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        out = ROOT_DIR / output_dir
        out.mkdir(parents=True, exist_ok=True)

        texts = test_df[self.text_col].astype(str).tolist()
        y_true = test_df[self.label_col].astype(str).tolist()

        X = self._vectorizer.transform(texts)
        y_pred = self._model.predict(X)

        # Overall metrics
        metrics = {
            "model": self._model_key,
            "model_name": MODELS[self._model_key]["name"],
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
            "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "train_size": None,  # filled by caller
            "test_size": len(test_df),
            "n_classes": len(self._classes),
            "classes": self._classes,
        }

        # Classification report (text)
        report_text = classification_report(y_true, y_pred, zero_division=0)
        (out / "classification_report.txt").write_text(report_text, encoding="utf-8")

        # Per-class metrics
        per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["per_class"] = {
            k: v for k, v in per_class.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        }

        # Save metrics JSON
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Confusion matrix
        self._plot_confusion_matrix(y_true, y_pred, out / "confusion_matrix.png")

        # Per-class F1 bar chart
        self._plot_per_class_f1(per_class, out / "per_class_f1.png")

        return metrics

    def _plot_confusion_matrix(
        self, y_true: list, y_pred: list, path: Path,
    ) -> None:
        """Plot and save confusion matrix."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Text annotations
        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_per_class_f1(self, report_dict: dict, path: Path) -> None:
        """Plot per-class F1 scores."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = []
        f1_scores = []
        for k, v in report_dict.items():
            if k not in ("accuracy", "macro avg", "weighted avg") and isinstance(v, dict):
                classes.append(k)
                f1_scores.append(v.get("f1-score", 0))

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(classes, f1_scores, color="#2196F3")
        ax.set_xlabel("F1 Score")
        ax.set_title("Per-class F1 Score")
        ax.set_xlim(0, 1.05)

        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # ── Skill: save_model ─────────────────────────────────────────

    def save_model(
        self,
        model_dir: str | Path = "models",
    ) -> dict[str, Path]:
        """Save trained model and vectorizer.

        Args:
            model_dir: Directory to save model files.

        Returns:
            Dict of saved file paths.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        out = ROOT_DIR / model_dir
        out.mkdir(parents=True, exist_ok=True)

        # Save model + vectorizer together
        model_path = out / "final_model.pkl"
        joblib.dump({
            "vectorizer": self._vectorizer,
            "model": self._model,
            "model_key": self._model_key,
            "classes": self._classes,
        }, model_path)

        # Save config
        config = {
            "model_key": self._model_key,
            "model_name": MODELS[self._model_key]["name"],
            "classes": self._classes,
            "n_classes": len(self._classes),
            "vectorizer": {
                "max_features": self._vectorizer.max_features,
                "ngram_range": list(self._vectorizer.ngram_range),
            },
        }
        config_path = out / "model_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return {"model": model_path, "config": config_path}

    # ── Skill: report ─────────────────────────────────────────────

    def report(
        self,
        recommendation: dict,
        metrics: dict,
        train_size: int = 0,
        output_dir: str | Path = "data/model",
    ) -> Path:
        """Generate MODEL_REPORT.md.

        Args:
            recommendation: Output of select_model().
            metrics: Output of evaluate().
            train_size: Number of training examples.
            output_dir: Where to save report.

        Returns:
            Path to MODEL_REPORT.md.
        """
        out = ROOT_DIR / output_dir
        out.mkdir(parents=True, exist_ok=True)

        model_name = MODELS.get(metrics["model"], {}).get("name", metrics["model"])

        lines = [
            "# Model Training Report",
            "",
            "## Model Selection (OpenRouter API)",
            "",
            f"Model chosen by OpenRouter API ({DEFAULT_MODEL}): **{model_name}** (`{metrics['model']}`)",
            "",
            f"**Reasoning**: {recommendation.get('reasoning', 'N/A')}",
            "",
            "## Training",
            "",
            f"- **Model**: TF-IDF + {model_name}",
            f"- **Train size**: {train_size}",
            f"- **Test size**: {metrics['test_size']}",
            f"- **Classes**: {metrics['n_classes']} ({', '.join(metrics['classes'])})",
            "",
            "## Overall Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {metrics['accuracy']:.4f} |",
            f"| F1 (macro) | {metrics['f1_macro']:.4f} |",
            f"| F1 (weighted) | {metrics['f1_weighted']:.4f} |",
            f"| Precision (macro) | {metrics['precision_macro']:.4f} |",
            f"| Recall (macro) | {metrics['recall_macro']:.4f} |",
            "",
            "## Per-class Metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|-------|-----------|--------|----|---------|",
        ]

        for cls, vals in metrics.get("per_class", {}).items():
            if isinstance(vals, dict):
                lines.append(
                    f"| {cls} | {vals.get('precision', 0):.4f} | "
                    f"{vals.get('recall', 0):.4f} | "
                    f"{vals.get('f1-score', 0):.4f} | "
                    f"{int(vals.get('support', 0))} |"
                )

        lines.extend([
            "",
            "## Artifacts",
            "",
            "- `models/final_model.pkl` — trained model (joblib)",
            "- `models/model_config.json` — model configuration",
            "- `data/model/confusion_matrix.png` — confusion matrix",
            "- `data/model/per_class_f1.png` — per-class F1 chart",
            "- `data/model/classification_report.txt` — full classification report",
            "- `data/model/metrics.json` — metrics in JSON format",
            "",
        ])

        path = out / "MODEL_REPORT.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ── Full pipeline ─────────────────────────────────────────────

    def run(
        self,
        parquet_path: str | Path = "data/labeled/labeled.parquet",
        task_type: str = "sentiment analysis",
        test_size: float = 0.2,
    ) -> dict:
        """Run full training pipeline: select → train → evaluate → save → report.

        Args:
            parquet_path: Path to labeled dataset.
            task_type: Task type from config.
            test_size: Fraction for test split.

        Returns:
            Dict with recommendation, metrics, and paths.
        """
        df = pd.read_parquet(ROOT_DIR / parquet_path)
        df_labeled = df[df[self.label_col].notna()].copy()

        # Step 1: LLM selects model
        recommendation = self.select_model(df_labeled, task_type=task_type)

        # Step 2: Split data
        train_df, test_df = train_test_split(
            df_labeled,
            test_size=test_size,
            random_state=self.random_state,
            stratify=df_labeled[self.label_col],
        )

        # Step 3: Train
        self.train(train_df, model_key=recommendation["model"])

        # Step 4: Evaluate
        metrics = self.evaluate(test_df)
        metrics["train_size"] = len(train_df)

        # Step 5: Save model
        model_paths = self.save_model()

        # Step 6: Generate report
        report_path = self.report(recommendation, metrics, train_size=len(train_df))

        return {
            "recommendation": recommendation,
            "metrics": metrics,
            "model_paths": model_paths,
            "report_path": report_path,
        }
