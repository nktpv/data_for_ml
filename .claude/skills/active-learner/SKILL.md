---
name: active-learner
description: "Active Learning — LLM picks model+seed, strategy comparison (entropy/margin/random), learning curves"
---

# Active Learner

Runs Active Learning experiments with LLM-powered model selection.
OpenRouter API analyzes the dataset and task to choose the best model and seed size.
Then compares sampling strategies (entropy, margin, random) and shows how many
labeled examples can be saved.

## Trigger

User asks to run active learning, compare sampling strategies, or optimize labeling budget.
Examples:
- "запусти active learning"
- "сравни стратегии отбора данных"
- "run active learning experiment"
- "сколько примеров реально нужно для разметки?"

## Autonomy rules

- **ALL steps are FULLY AUTONOMOUS** — no human-in-the-loop checkpoints.
- Run everything silently, show results at the end.
- If something fails — fix it or skip it, move on.

## Workflow

### Step 1 — Load data (SILENT)

Load labeled dataset and task type from config:

```python
import pandas as pd
import yaml
from agents.active_learning_agent import ActiveLearningAgent

df = pd.read_parquet("data/labeled/labeled.parquet")
with open("config.yaml") as f:
    config = yaml.safe_load(f)
task_type = config["task"]["type"]
df_labeled = df[df["auto_label"].notna()].copy()
```

### Step 2 — LLM selects model and seed size (SILENT)

OpenRouter model analyzes the dataset (class distribution, imbalance, text lengths, sample texts)
and chooses the best model and seed size:

```python
agent = ActiveLearningAgent()
recommendation = agent.select_model(df_labeled, task_type=task_type)
# → {"model": "logreg_balanced", "seed_size": 50, "reasoning": "..."}
```

Available models:
- `logreg` — Logistic Regression (good default)
- `logreg_balanced` — LogReg with class_weight='balanced' (handles imbalance)
- `svm` — Linear SVM with Platt scaling (strong on high-dimensional text)

### Step 3 — Run AL cycles for all strategies (SILENT)

Run the full experiment via script:

```bash
python .claude/skills/active-learner/scripts/run_al.py data/labeled/labeled.parquet 20 5
```

The script calls `select_model()` automatically, then runs AL cycles for entropy, margin, and random.

Or via Python:
```python
agent = ActiveLearningAgent(model=recommendation["model"])
results = agent.compare_strategies(
    df_labeled,
    strategies=["entropy", "margin", "random"],
    seed_size=recommendation["seed_size"],
    n_iterations=5,
    batch_size=20,
)
```

### Step 4 — Generate charts and report (SILENT)

```python
paths = agent.report(results)
```

Or regenerate chart from saved histories:
```bash
python .claude/skills/active-learner/scripts/visualize.py
```

Generates:
- `data/active_learning/history_entropy.json`
- `data/active_learning/history_margin.json`
- `data/active_learning/history_random.json`
- `data/active_learning/learning_curve.png`
- `data/active_learning/REPORT.md` (includes LLM model selection reasoning)

### Step 5 — Generate notebook (SILENT)

Create `notebooks/al_experiment.ipynb` with:
1. LLM model selection result and reasoning
2. Data loading and split info
3. Learning curves (all strategies on one chart)
4. Metrics table per iteration
5. Savings analysis
6. **Markdown cell with conclusions**: which strategy wins, how many examples saved, why this model

Execute:
```bash
python .claude/skills/active-learner/scripts/run_notebook.py notebooks/al_experiment.ipynb
```

### Step 6 — Summary

```
Active Learning experiment complete!
   LLM selected: Logistic Regression (balanced), seed=50
     Reason: "4 sentiment classes with moderate imbalance, balanced weights prevent minority class collapse"

   Dataset: 5,062 rows (seed=50, pool=3,999, test=1,013)
   Strategies: entropy, margin, random
   Iterations: 5 (batch_size=20)

   Final metrics (at 150 labeled):
     entropy: accuracy=0.87, F1=0.85
     margin:  accuracy=0.86, F1=0.84
     random:  accuracy=0.82, F1=0.79

   Savings:
     entropy reaches random's F1 at 100 examples (33% fewer)
     margin reaches random's F1 at 120 examples (20% fewer)

   Saved:
     data/active_learning/learning_curve.png
     data/active_learning/REPORT.md
     data/active_learning/history_*.json
     notebooks/al_experiment.ipynb (with results)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_al.py` | Full experiment: LLM model selection + all strategies + report. Pass `--no-llm` to skip LLM. |
| `scripts/visualize.py` | Regenerate learning_curve.png from saved histories |

## Config

Input dataset: `data/labeled/labeled.parquet` (output of Label Master)
Output: `data/active_learning/` (histories, chart, report)
Notebook: `notebooks/al_experiment.ipynb`
API key: `OPENROUTER_API_KEY` from `.env` (for model selection)
