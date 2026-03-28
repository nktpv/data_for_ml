"""ActiveLearningAgent — smart data selection for labeling.

Demonstrates that intelligent sampling (entropy, margin) achieves the same
model quality with fewer labeled examples compared to random sampling.

LLM-powered model selection: OpenRouter API (model from OPENROUTER_MODEL env) analyzes
the task and data to choose the best model and seed size for the experiment.

Skills:
    - select_model(df, task_type) → model name + seed size (via OpenRouter API)
    - fit(labeled_df) → fitted model
    - query(pool_df, strategy, batch_size) → indices of most informative examples
    - evaluate(test_df) → Metrics (accuracy, F1-macro)
    - run_cycle(...) → history of metrics per iteration
    - compare_strategies(...) → dict of histories for each strategy
    - report(results) → learning_curve.png + REPORT.md

Usage::

    from agents.active_learning_agent import ActiveLearningAgent

    agent = ActiveLearningAgent()
    recommendation = agent.select_model(df, task_type="sentiment analysis")
    # → {"model": "logreg_balanced", "seed_size": 50, "reasoning": "..."}

    results = agent.compare_strategies(df, seed_size=recommendation["seed_size"])
    agent.report(results)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from .openrouter_client import DEFAULT_MODEL  # noqa: E402

Strategy = Literal["entropy", "margin", "random"]

# Available models for AL experiments
MODELS = {
    "logreg": {
        "name": "Logistic Regression",
        "description": "Linear model with calibrated probabilities, good default for text classification",
        "factory": lambda rs: LogisticRegression(max_iter=1000, random_state=rs, solver="lbfgs"),
    },
    "logreg_balanced": {
        "name": "Logistic Regression (balanced)",
        "description": "LogReg with class_weight='balanced', handles class imbalance",
        "factory": lambda rs: LogisticRegression(
            max_iter=1000, random_state=rs, solver="lbfgs", class_weight="balanced",
        ),
    },
    "svm": {
        "name": "Linear SVM (calibrated)",
        "description": "Fast linear SVM with Platt scaling for probabilities, strong on high-dimensional text",
        "factory": lambda rs: CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=rs), cv=3,
        ),
    },
    "nb": {
        "name": "Multinomial Naive Bayes",
        "description": "Fast probabilistic model, works well with small datasets and bag-of-words",
        "factory": lambda rs: MultinomialNB(alpha=1.0),
    },
}


@dataclass
class Metrics:
    """Evaluation metrics for one iteration."""
    iteration: int
    n_labeled: int
    accuracy: float
    f1: float

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "n_labeled": self.n_labeled,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
        }


class ActiveLearningAgent:
    """Active Learning agent with LLM-powered model selection.

    Skills:
        - ``select_model(df, task_type)`` — OpenRouter LLM chooses model + seed size
        - ``fit(labeled_df)`` — train the model
        - ``query(pool_df, strategy, batch_size)`` — select informative examples
        - ``evaluate(test_df)`` — compute accuracy and F1
        - ``run_cycle(...)`` — full AL loop
        - ``compare_strategies(...)`` — run multiple strategies
        - ``report(results)`` — generate charts and report
    """

    def __init__(
        self,
        model: str = "logreg",
        text_col: str = "text",
        label_col: str = "auto_label",
        random_state: int = 42,
    ):
        self.model_key = model
        self.text_col = text_col
        self.label_col = label_col
        self.random_state = random_state
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
        )
        self._model = self._create_model(model)
        self._is_fitted = False
        self._llm_recommendation: dict | None = None

    def _create_model(self, model_key: str):
        """Create a sklearn model instance from key."""
        if model_key not in MODELS:
            raise ValueError(
                f"Unknown model: {model_key}. Available: {list(MODELS.keys())}"
            )
        return MODELS[model_key]["factory"](self.random_state)

    # ── Skill: select_model (LLM-powered) ─────────────────────────

    def select_model(
        self,
        df: pd.DataFrame,
        task_type: str = "sentiment analysis",
    ) -> dict:
        """Use OpenRouter API to analyze data and recommend model + seed size.

        Args:
            df: Labeled dataset to analyze.
            task_type: Task description from config.yaml.

        Returns:
            Dict with 'model', 'seed_size', 'reasoning' keys.
        """
        import re
        import requests

        api_key = os.getenv("OPENROUTER_API_KEY")

        # Gather dataset stats for LLM
        label_counts = df[self.label_col].value_counts()
        n_classes = len(label_counts)
        min_class_size = int(label_counts.min())
        max_class_size = int(label_counts.max())
        imbalance_ratio = round(max_class_size / min_class_size, 1)
        total = len(df)
        avg_text_len = int(df[self.text_col].astype(str).str.len().mean())
        sample_texts = df[self.text_col].astype(str).head(5).tolist()

        models_desc = "\n".join(
            f"  - {key}: {info['description']}"
            for key, info in MODELS.items()
        )

        prompt = f"""You are an ML engineer choosing the best model and seed size for an Active Learning experiment.

Task: {task_type}

Dataset stats:
- Total samples: {total}
- Number of classes: {n_classes}
- Class distribution: {dict(label_counts)}
- Imbalance ratio (max/min): {imbalance_ratio}x
- Min class size: {min_class_size}
- Average text length: {avg_text_len} chars

Sample texts:
{chr(10).join(f'  "{t[:150]}"' for t in sample_texts)}

Available models:
{models_desc}

Choose:
1. The best model for this AL experiment (one of: {', '.join(MODELS.keys())})
2. The optimal seed size (initial labeled set). Rules:
   - Must have at least 2-3 examples per class in the seed
   - Smaller seed = more dramatic AL learning curve
   - Too small = model can't learn at all
   - Consider the min class size and imbalance

Respond with ONLY a JSON object:
{{
  "model": "<model_key>",
  "seed_size": <int>,
  "reasoning": "<2-3 sentences explaining your choice>"
}}"""

        response_text = None
        if api_key:
            for attempt in range(4):
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
                            "max_tokens": 700,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    response_text = (msg.get("content") or "").strip()
                    if response_text:
                        break
                except Exception:
                    response_text = None
                    if attempt < 3:
                        time.sleep(2 ** attempt)

        if response_text:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                    if result.get("model") not in MODELS:
                        result["model"] = "logreg"
                    if not isinstance(result.get("seed_size"), int) or result["seed_size"] < n_classes * 2:
                        result["seed_size"] = max(n_classes * 5, 50)
                    self._llm_recommendation = result
                    return result
                except json.JSONDecodeError:
                    pass

        # Fallback
        fallback = {
            "model": "logreg_balanced" if imbalance_ratio > 3 else "logreg",
            "seed_size": max(n_classes * 5, 50),
            "reasoning": f"Fallback: chose based on imbalance ratio {imbalance_ratio}x and {n_classes} classes.",
        }
        self._llm_recommendation = fallback
        return fallback

    # ── Skill: fit ─────────────────────────────────────────────────

    def fit(self, labeled_df: pd.DataFrame) -> "ActiveLearningAgent":
        """Train TF-IDF + LogReg on labeled data.

        Args:
            labeled_df: DataFrame with text and label columns.

        Returns:
            self (fitted agent).
        """
        texts = labeled_df[self.text_col].astype(str).tolist()
        labels = labeled_df[self.label_col].astype(str).tolist()

        X = self._vectorizer.fit_transform(texts)
        self._model.fit(X, labels)
        self._is_fitted = True
        return self

    # ── Skill: query ───────────────────────────────────────────────

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: Strategy = "entropy",
        batch_size: int = 20,
    ) -> np.ndarray:
        """Select the most informative examples from the pool.

        Args:
            pool_df: Unlabeled pool DataFrame.
            strategy: Selection strategy ('entropy', 'margin', 'random').
            batch_size: Number of examples to select.

        Returns:
            Array of indices (from pool_df.index) to label next.
        """
        if len(pool_df) <= batch_size:
            return pool_df.index.values

        if strategy == "random":
            rng = np.random.RandomState(self.random_state)
            return rng.choice(pool_df.index.values, size=batch_size, replace=False)

        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        texts = pool_df[self.text_col].astype(str).tolist()
        X = self._vectorizer.transform(texts)
        proba = self._model.predict_proba(X)

        if strategy == "entropy":
            scores = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            # Higher entropy = more uncertain = more informative
            top_idx = np.argsort(scores)[-batch_size:]
        elif strategy == "margin":
            sorted_proba = np.sort(proba, axis=1)
            margins = sorted_proba[:, -1] - sorted_proba[:, -2]
            # Smaller margin = more uncertain = more informative
            top_idx = np.argsort(margins)[:batch_size]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return pool_df.index.values[top_idx]

    # ── Skill: evaluate ───────────────────────────────────────────

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluate model on test set.

        Args:
            test_df: Test DataFrame with text and label columns.

        Returns:
            Dict with 'accuracy' and 'f1' keys.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        texts = test_df[self.text_col].astype(str).tolist()
        y_true = test_df[self.label_col].astype(str).tolist()

        X = self._vectorizer.transform(texts)
        y_pred = self._model.predict(X)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    # ── Skill: run_cycle ──────────────────────────────────────────

    def run_cycle(
        self,
        seed_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: Strategy = "entropy",
        n_iterations: int = 5,
        batch_size: int = 20,
    ) -> list[dict]:
        """Run a full active learning cycle.

        Args:
            seed_df: Initial labeled seed set.
            pool_df: Pool of "unlabeled" examples (labels hidden).
            test_df: Fixed test set for evaluation.
            strategy: Query strategy.
            n_iterations: Number of AL iterations.
            batch_size: Examples to add per iteration.

        Returns:
            List of Metrics dicts, one per iteration (including iteration 0).
        """
        labeled = seed_df.copy()
        pool = pool_df.copy()
        history: list[dict] = []

        for i in range(n_iterations + 1):
            # Reset and train a fresh model each iteration
            self._vectorizer = TfidfVectorizer(
                max_features=10000, ngram_range=(1, 2),
            )
            self._model = self._create_model(self.model_key)

            self.fit(labeled)
            metrics = self.evaluate(test_df)

            m = Metrics(
                iteration=i,
                n_labeled=len(labeled),
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
            )
            history.append(m.to_dict())

            if i < n_iterations and len(pool) > 0:
                selected_idx = self.query(pool, strategy=strategy, batch_size=batch_size)
                new_examples = pool.loc[selected_idx]
                labeled = pd.concat([labeled, new_examples])
                pool = pool.drop(selected_idx)

        return history

    # ── Skill: compare_strategies ─────────────────────────────────

    def compare_strategies(
        self,
        full_df: pd.DataFrame,
        strategies: list[Strategy] | None = None,
        seed_size: int = 50,
        test_size: float = 0.2,
        n_iterations: int = 5,
        batch_size: int = 20,
    ) -> dict[str, list[dict]]:
        """Run AL cycles for multiple strategies with identical splits.

        Args:
            full_df: Full labeled dataset.
            strategies: List of strategies to compare.
            seed_size: Number of initial labeled examples.
            test_size: Fraction for test set.
            n_iterations: AL iterations per strategy.
            batch_size: Examples per iteration.

        Returns:
            Dict mapping strategy name to its history.
        """
        if strategies is None:
            strategies = ["entropy", "margin", "random"]

        # Fixed split: test set
        train_pool, test_df = train_test_split(
            full_df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=full_df[self.label_col],
        )

        # Fixed split: seed from train_pool
        seed_df, pool_df = train_test_split(
            train_pool,
            train_size=seed_size,
            random_state=self.random_state,
            stratify=train_pool[self.label_col],
        )

        results = {}
        for strategy in strategies:
            history = self.run_cycle(
                seed_df=seed_df.copy(),
                pool_df=pool_df.copy(),
                test_df=test_df,
                strategy=strategy,
                n_iterations=n_iterations,
                batch_size=batch_size,
            )
            results[strategy] = history

        return results

    # ── Skill: report ─────────────────────────────────────────────

    def report(
        self,
        results: dict[str, list[dict]],
        output_dir: str | Path = "data/active_learning",
    ) -> dict[str, Path]:
        """Generate learning curve chart and markdown report.

        Args:
            results: Output of compare_strategies().
            output_dir: Where to save artifacts.

        Returns:
            Dict of artifact name to path.
        """
        out = ROOT_DIR / output_dir
        out.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save histories
        for strategy, history in results.items():
            p = out / f"history_{strategy}.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            paths[f"history_{strategy}"] = p

        # Learning curve plot
        chart_path = self._plot_learning_curves(results, out / "learning_curve.png")
        paths["learning_curve"] = chart_path

        # Markdown report
        report_path = self._generate_report(results, out / "REPORT.md")
        paths["report"] = report_path

        return paths

    def _plot_learning_curves(
        self, results: dict[str, list[dict]], path: Path,
    ) -> Path:
        """Plot all strategies on one chart."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        markers = {"entropy": "o", "margin": "s", "random": "^"}
        colors = {"entropy": "#2196F3", "margin": "#FF9800", "random": "#9E9E9E"}

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
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        return path

    def _generate_report(
        self, results: dict[str, list[dict]], path: Path,
    ) -> Path:
        """Generate REPORT.md with comparison and savings analysis."""
        lines = [
            "# Active Learning Report",
            "",
            "## Experiment Setup",
            "",
        ]

        # Get setup info from first strategy
        first_strategy = list(results.keys())[0]
        first_history = results[first_strategy]
        seed_size = first_history[0]["n_labeled"]
        final_size = first_history[-1]["n_labeled"]
        n_iterations = len(first_history) - 1
        batch_size = (final_size - seed_size) // n_iterations if n_iterations > 0 else 0

        model_name = MODELS.get(self.model_key, {}).get("name", self.model_key)

        lines.extend([
            f"- **Seed size**: {seed_size}",
            f"- **Batch size**: {batch_size}",
            f"- **Iterations**: {n_iterations}",
            f"- **Final labeled**: {final_size}",
            f"- **Strategies**: {', '.join(results.keys())}",
            f"- **Model**: TF-IDF + {model_name}",
            "",
            "## Results",
            "",
            "### Final Metrics",
            "",
            "| Strategy | Accuracy | F1 (macro) |",
            "|----------|----------|------------|",
        ])

        final_metrics = {}
        for strategy, history in results.items():
            final = history[-1]
            final_metrics[strategy] = final
            lines.append(
                f"| {strategy} | {final['accuracy']:.4f} | {final['f1']:.4f} |"
            )

        lines.extend(["", "### Learning Curve Data", ""])

        # Full table
        lines.append("| Iteration | N labeled |" + "".join(
            f" {s} Acc | {s} F1 |" for s in results.keys()
        ))
        lines.append("|-----------|-----------|" + "".join(
            "---------|--------|" for _ in results.keys()
        ))

        n_rows = len(first_history)
        for i in range(n_rows):
            row = f"| {i} | {first_history[i]['n_labeled']} |"
            for strategy in results:
                h = results[strategy][i]
                row += f" {h['accuracy']:.4f} | {h['f1']:.4f} |"
            lines.append(row)

        # Savings analysis
        lines.extend(["", "## Savings Analysis", ""])

        if "random" in results and len(results) > 1:
            random_final_f1 = results["random"][-1]["f1"]
            random_final_n = results["random"][-1]["n_labeled"]

            for strategy in results:
                if strategy == "random":
                    continue

                # Find at which n_labeled this strategy reaches random's final F1
                target_f1 = random_final_f1
                reached_at = None
                for h in results[strategy]:
                    if h["f1"] >= target_f1:
                        reached_at = h["n_labeled"]
                        break

                if reached_at is not None:
                    saved = random_final_n - reached_at
                    pct = saved / random_final_n * 100
                    lines.extend([
                        f"**{strategy}** reaches Random's final F1 ({target_f1:.4f}) "
                        f"at **{reached_at}** examples.",
                        f"Savings: **{saved} examples ({pct:.1f}%)** vs random baseline.",
                        "",
                    ])
                else:
                    lines.extend([
                        f"**{strategy}** did not reach Random's final F1 ({target_f1:.4f}) "
                        f"within {n_iterations} iterations.",
                        "",
                    ])

        # Best strategy
        best = max(final_metrics.items(), key=lambda x: x[1]["f1"])
        lines.extend(["## Conclusion", "", f"Best strategy: **{best[0]}** (F1 = {best[1]['f1']:.4f})", ""])
        if best[0] == "random":
            lines.extend([
                "On this single run, random sampling won — common when the initial seed is very small "
                "and metrics are noisy. Repeat with several `random_state` values or a larger seed "
                "before drawing conclusions about AL.",
                "",
            ])
        else:
            lines.extend([
                "Active learning can reach a given quality with fewer labeled examples by prioritizing "
                "the most informative points; confirm with multiple random seeds.",
                "",
            ])

        # LLM model selection reasoning
        if self._llm_recommendation:
            rec = self._llm_recommendation
            lines.extend([
                "## Model Selection (OpenRouter API)",
                "",
                f"Model chosen by OpenRouter API ({DEFAULT_MODEL}): **{MODELS.get(rec['model'], {}).get('name', rec['model'])}** (`{rec['model']}`)",
                f"Seed size: **{rec['seed_size']}**",
                "",
                f"**Reasoning**: {rec.get('reasoning', 'N/A')}",
                "",
            ])

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ── Prepare data splits ───────────────────────────────────────

    @staticmethod
    def prepare_splits(
        df: pd.DataFrame,
        label_col: str = "auto_label",
        seed_size: int = 50,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into seed, pool, and test sets.

        Args:
            df: Full labeled dataset.
            label_col: Label column name.
            seed_size: Number of seed examples.
            test_size: Fraction for test set.
            random_state: Random seed.

        Returns:
            (seed_df, pool_df, test_df)
        """
        train_pool, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df[label_col],
        )
        seed_df, pool_df = train_test_split(
            train_pool, train_size=seed_size, random_state=random_state,
            stratify=train_pool[label_col],
        )
        return seed_df, pool_df, test_df
