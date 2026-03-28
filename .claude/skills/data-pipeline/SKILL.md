---
name: data-pipeline
description: "Full ML data pipeline — collect → clean → label → active learning → train. One command, human-in-the-loop at key checkpoints."
user-invocable: true
---

# Data Pipeline

Meta-skill that orchestrates all agents into a single end-to-end pipeline.
One command produces a ready-to-use labeled dataset, trained model, and full report.

## Trigger

User asks to run the full pipeline, or provides a dataset task description.
Examples:
- "/data-pipeline запусти пайплайн для sentiment analysis"
- "запусти полный пайплайн для sentiment analysis"
- "run the full data pipeline for sentiment analysis"

## Autonomy rules

- Each step follows the autonomy rules of its own skill (SKILL.md).
- **Human-in-the-loop checkpoints** exist at steps 1, 2, 3 — the pipeline STOPS and waits.
- Steps 4 and 5 are fully autonomous.
- Between steps, the pipeline automatically passes data via parquet files.
- If a step fails — try to fix it. If unfixable — stop and report what happened.

## Pipeline Overview

```
User request: "запусти пайплайн для sentiment analysis отзывов"
         │
         ▼
┌─────────────────────────────────┐
│  Step 1: Dataset Collector      │  → data/raw/combined.parquet
│  ⏸ CHECKPOINT: выбор источников │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 2: Data Detective         │  → data/cleaned/cleaned.parquet
│  ⏸ CHECKPOINT: стратегия чистки │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 3: Label Master           │  → data/labeled/labeled.parquet
│  ⏸ CHECKPOINT: таксономия       │
│  ⏸ CHECKPOINT: Label Studio     │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 4: Active Learner         │  → data/active_learning/
│  (fully autonomous)             │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 5: Model Trainer          │  → models/final_model.pkl
│  (fully autonomous)             │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 6: Pipeline Report        │  → PIPELINE_REPORT.md
│  (fully autonomous)             │
└─────────────────────────────────┘
```

## Workflow

### Step 1 — Dataset Collection

Follow the **data-collector** skill (`.claude/skills/data-collector/SKILL.md`).

Input: user's task description (from `/data-pipeline` argument).
Output: `data/raw/combined.parquet`

The skill will:
1. Save task type to `config.yaml`
2. Search for sources (HuggingFace, Kaggle, scraping, API, RSS)
3. Validate sources
4. **⏸ CHECKPOINT**: show validated sources table → wait for user to pick
5. Collect selected sources
6. Generate EDA notebook
7. Show summary

**After Step 1 completes, verify** `data/raw/combined.parquet` exists before proceeding.

### Step 2 — Data Quality Check

Follow the **data-detective** skill (`.claude/skills/data-detective/SKILL.md`).

Input: `data/raw/combined.parquet`
Output: `data/cleaned/cleaned.parquet`

The skill will:
1. Deep issue discovery (all possible problems)
2. Apply all cleaning strategies and compare
3. **⏸ CHECKPOINT**: show all findings + strategy comparison → wait for user to pick
4. Save cleaned dataset
5. Generate quality report and notebook

**After Step 2 completes, verify** `data/cleaned/cleaned.parquet` exists before proceeding.

### Step 3 — Auto-labeling

Follow the **data-annotation** skill (`.claude/skills/data-annotation/SKILL.md`).

Input: `data/cleaned/cleaned.parquet`
Output: `data/labeled/labeled.parquet`

The skill will:
1. Analyze data, propose taxonomy
2. **⏸ CHECKPOINT**: show taxonomy + 4 options → wait for user to choose
3. Auto-label via OpenRouter API
4. Generate spec + quality metrics
5. Export disputed examples to Label Studio
6. **⏸ CHECKPOINT**: "Open Label Studio?" → wait for user
7. If yes: start LS, wait for user to finish, merge back
8. Generate annotation notebook

**After Step 3 completes, verify** `data/labeled/labeled.parquet` exists before proceeding.

### Step 4 — Active Learning Experiment

Follow the **active-learner** skill (`.claude/skills/active-learner/SKILL.md`).

Input: `data/labeled/labeled.parquet`
Output: `data/active_learning/` (histories, learning curve, report)

The skill will:
1. LLM selects model and seed size
2. Run AL cycles for entropy, margin, random
3. Generate charts and report
4. Generate AL notebook

Fully autonomous — no checkpoints.

Run via script:
```bash
python .claude/skills/active-learner/scripts/run_al.py data/labeled/labeled.parquet 20 5
```

**After Step 4 completes, verify** `data/active_learning/REPORT.md` exists before proceeding.

### Step 5 — Final Model Training

Follow the **model-trainer** skill (`.claude/skills/model-trainer/SKILL.md`).

Input: `data/labeled/labeled.parquet` + `data/active_learning/` results
Output: `models/final_model.pkl` + `data/model/` artifacts

The skill will:
1. LLM selects model based on AL results
2. Train on 80% of all labeled data
3. Evaluate on 20% test set
4. Save model + metrics + charts + report

Fully autonomous — no checkpoints.

Run via script:
```bash
python .claude/skills/model-trainer/scripts/train.py data/labeled/labeled.parquet
```

**After Step 5 completes, verify** `models/final_model.pkl` exists before proceeding.

### Step 6 — Pipeline Report

Generate `PIPELINE_REPORT.md` (project root) — a comprehensive summary of the entire pipeline run.

Collect information from all previous steps:

```python
# Read all reports and metrics
import json, yaml
from pathlib import Path

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Step 1 — collection stats
raw_df = pd.read_parquet("data/raw/combined.parquet")

# Step 2 — cleaning stats
cleaned_df = pd.read_parquet("data/cleaned/cleaned.parquet")

# Step 3 — labeling stats
labeled_df = pd.read_parquet("data/labeled/labeled.parquet")
with open("data/annotation/quality_metrics.json") as f:
    annotation_metrics = json.load(f)

# Step 4 — AL stats
with open("data/active_learning/REPORT.md") as f:
    al_report = f.read()

# Step 5 — model stats
with open("data/model/metrics.json") as f:
    model_metrics = json.load(f)
```

Write `PIPELINE_REPORT.md`:

```markdown
# ML Data Pipeline Report

## Task
{task_type from config.yaml}

## Pipeline Summary

| Step | Agent | Input | Output | Key Result |
|------|-------|-------|--------|------------|
| 1 | Dataset Collector | User request | combined.parquet | N sources, M rows |
| 2 | Data Detective | combined.parquet | cleaned.parquet | K issues fixed |
| 3 | Label Master | cleaned.parquet | labeled.parquet | N classes, M disputed |
| 4 | Active Learner | labeled.parquet | AL report | entropy saves X% |
| 5 | Model Trainer | labeled.parquet | final_model.pkl | accuracy=Y, F1=Z |

## Step 1: Data Collection
[summary from EDA report]

## Step 2: Data Cleaning
[summary from QUALITY_REPORT.md]

## Step 3: Labeling
[summary from annotation metrics]

## Step 4: Active Learning
[summary from AL report]

## Step 5: Model Training
[summary from model metrics]

## All Artifacts

data/raw/combined.parquet
data/cleaned/cleaned.parquet
data/labeled/labeled.parquet
data/active_learning/learning_curve.png
models/final_model.pkl
data/model/confusion_matrix.png
notebooks/eda.ipynb
notebooks/data_quality.ipynb
notebooks/annotation.ipynb
notebooks/al_experiment.ipynb
notebooks/model_training.ipynb

## Recommendations
[what to improve, how to collect more data, which classes need more examples]
```

### Final Summary

After all steps complete, show the user:

```
Pipeline complete!

   Task: Sentiment analysis
   Sources: 3 (HuggingFace, Kaggle, RSS)
   Raw data: 5,062 rows
   Cleaned: 5,032 rows (30 removed)
   Labeled: 5,032 rows (4 classes: pos/neg/neutral/mixed, 9 disputed)
   AL: entropy saves 33% vs random
   Final model: accuracy=0.88, F1=0.85

   Artifacts:
     PIPELINE_REPORT.md
     models/final_model.pkl
     notebooks/ (5 notebooks with results)
     data/ (all intermediate datasets + reports + charts)

   The pipeline is reproducible:
     python scripts/run_pipeline.py "sentiment analysis"
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Standalone reproducible pipeline (no Claude Code needed) |

## Data Flow

```
config.yaml (task.type) ──────────────────────────────────────────────┐
                                                                       │
data/raw/combined.parquet ──→ data/cleaned/cleaned.parquet             │
                               ──→ data/labeled/labeled.parquet        │
                                    ──→ data/active_learning/          │
                                    ──→ models/final_model.pkl         │
                                                                       │
PIPELINE_REPORT.md ◄──────────────────────────────────────────────────┘
```
