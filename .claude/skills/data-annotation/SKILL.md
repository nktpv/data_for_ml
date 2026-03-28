---
name: data-annotation
description: "Auto-annotate text data with OpenRouter API, generate annotation spec, assess quality, export uncertain examples to Label Studio for manual review"
---

# Data Annotator

Auto-annotates text data using the OpenRouter API, generates a human-readable annotation specification, checks labeling quality, and exports low-confidence examples to Label Studio for manual review.

## Trigger

User asks to label, annotate, or classify data. Can also be called automatically after the data-detective skill finishes.
Examples:
- "разметь данные"
- "запусти авторазметку"
- "label the dataset"
- "annotate texts"

## Autonomy rules

- **Step 1**: FULLY AUTONOMOUS — analyze data. No questions.
- **Step 2 is MANDATORY and BLOCKING**: show analysis, ask user what to do (re-label / label unlabeled / keep as-is).
- **Steps 3, 4, 5**: FULLY AUTONOMOUS — label, assess quality, export.
- **Step 6 is MANDATORY and BLOCKING**: ask user if they want to open Label Studio for disputed examples.
- **Step 7**: If user said yes — start Label Studio. If no — save and finish.

## Workflow

### Step 1 — Analyze dataset and propose taxonomy (SILENT)

Load the cleaned dataset and task context from config:

```python
import pandas as pd
import yaml

df = pd.read_parquet("data/cleaned/cleaned.parquet")
with open("config.yaml") as f:
    config = yaml.safe_load(f)
task_type = config["task"]["type"]  # e.g. "sentiment analysis"
```

Analyze:
- What labels exist in the `label` column
- Number of unique labels, their distribution
- How many rows are labeled vs missing labels
- Label consistency (casing, naming variants)
- Extract 2–3 representative example texts per existing label

**Propose a new taxonomy.** Based on `task_type` from config AND the actual texts, the LLM should:
1. Read a sample of texts (50–100) to understand what categories/tones actually appear
2. Assess whether existing labels are sufficient or need refinement
3. Propose NEW classes that better fit the task — the taxonomy does NOT have to match original labels
4. For example, for "sentiment analysis" the model might propose splitting the binary `pos/neg` into a 4-class scale, or adding `neutral` and `mixed` for ambiguous texts
5. Each proposed class must have a clear, concise definition

### Step 2 — Confirm task with user (HUMAN-IN-THE-LOOP)

Present the findings, proposed taxonomy, and ask what to do:

```
Task: Sentiment analysis (from config.yaml)

Dataset analysis:
   Total rows: 5,062
   Labeled: 4,556 (90.0%)
   Unlabeled: 506 (10.0%)
   Existing labels: 2 (pos, neg)

Proposed taxonomy (based on data analysis):

| # | Label    | Definition                                              | Existing match        |
|---|----------|---------------------------------------------------------|-----------------------|
| 1 | pos      | Clearly positive tone: praise, satisfaction, approval   | pos                   |
| 2 | neg      | Clearly negative tone: criticism, dissatisfaction       | neg                   |
| 3 | neutral  | NEW — factual/objective text, no strong sentiment       | —                     |
| 4 | mixed    | NEW — conflicting positive and negative signals in one  | —                     |

Why this taxonomy:
  - Added neutral (~180 factual texts currently forced into pos or neg)
  - Added mixed (~90 reviews containing contradictory sentiment cues)

What would you like to do?
  1. Re-annotate ALL texts with the proposed taxonomy
  2. Re-annotate ALL texts but adjust the taxonomy first
  3. Annotate only UNLABELED texts (506 rows) with the proposed taxonomy
  4. Keep labels as-is (just fill in the missing labels programmatically)
```

**STOP HERE. Wait for user's response.**

User may:
- Choose option 1, 2, 3, or 4
- Adjust labels (add / remove / rename / merge)
- Change confidence threshold

### Step 3 — Auto-annotate with OpenRouter API (SILENT)

Based on user's choice:

**Option 1 (re-annotate all with proposed taxonomy):** Run LLM classification on every row using the confirmed taxonomy.
**Option 2 (re-annotate all with adjusted taxonomy):** Same as 1 but user modified the classes first.
**Option 3 (annotate unlabeled only):** Only classify rows where `label` is null.
**Option 4 (keep as-is):** Copy `label` → `auto_label`, set `confidence=1.0`, fill nulls with most common label.

For options 1 and 2, use the auto_label script:

```bash
python .claude/skills/data-annotation/scripts/auto_label.py "pos,neg,neutral,mixed" "sentiment analysis" data/cleaned/cleaned.parquet 10
```

For option 3 (unlabeled only), use the label_unlabeled script:

```bash
python .claude/skills/data-annotation/scripts/label_unlabeled.py "pos,neg,neutral,mixed" "sentiment analysis" data/cleaned/cleaned.parquet 10
```

Both scripts read `allow_new_labels` from `config.yaml` automatically. Override with `--allow-new-labels`.

Each row receives:
- `auto_label` — the predicted label
- `confidence` — 0.0 to 1.0 (model's self-assessed certainty)
- `is_disputed` — True if confidence < threshold (default 0.7)

**IMPORTANT:** This calls the OpenRouter API. With 5k texts at batch_size=10, that is ~500 API calls. Show progress.

### Step 4 — Generate annotation spec and check quality (SILENT)

Generate the annotation specification:

```python
from agents.annotation_agent import AnnotationAgent
agent = AnnotationAgent()
agent.generate_spec(df_labeled, task="sentiment_analysis", labels=confirmed_labels)
```

Saves `data/annotation/annotation_spec.md` with:
- Task description and annotation goal
- Class definitions with distinguishing criteria
- 3+ high-confidence examples per class (taken from the actual dataset)
- Edge cases and annotation guidelines for ambiguous texts

Check quality:

```bash
python .claude/skills/data-annotation/scripts/check_quality.py data/labeled/labeled.parquet 0.7
```

### Step 5 — Export disputed examples (SILENT)

Export low-confidence examples to Label Studio format:

```bash
python .claude/skills/data-annotation/scripts/export_ls.py "pos,neg,neutral,mixed" data/labeled/labeled.parquet 0.7
```

Generates:
- `data/annotation/labelstudio_tasks.json` — tasks with model pre-annotations
- `data/annotation/labelstudio_config.xml` — project labeling config

### Step 6 — Ask about Label Studio (HUMAN-IN-THE-LOOP)

Show results and ask:

```
Annotation complete!
   Total annotated: 5,062
   Confidence: mean=0.88, median=0.93
   Disputed (confidence < 0.7): 153 (3%)
   Label distribution: pos=48%, neg=44%, neutral=6%, mixed=2%

   Disputed examples exported to: data/annotation/labelstudio_tasks.json (153 tasks)

Would you like to manually review disputed examples in Label Studio now? (yes/no)
```

**STOP HERE. Wait for user's response.**

### Step 7a — If YES: Start Label Studio

```bash
python .claude/skills/data-annotation/scripts/start_ls.py "Sentiment Review" "pos,neg,neutral,mixed"
```

This will:
1. Start Label Studio on localhost:8080 (if not already running)
2. Create a project with the confirmed label taxonomy
3. Import disputed tasks with model pre-annotations
4. Open the browser to the project

Tell the user:

```
Label Studio is open at http://localhost:8080

Instructions:
  1. Review the pre-annotated examples
  2. Correct any wrong labels
  3. When done, export your annotations:
     Project → Export → JSON
     Save to: data/annotation/ls_export.json

Then say "ready" or "done" and I will merge the manual labels back.
```

**STOP HERE. Wait for user to finish reviewing.**

When user says they are done:

```bash
python .claude/skills/data-annotation/scripts/import_ls.py data/annotation/ls_export.json data/labeled/labeled.parquet
```

Merges manual labels into `labeled.parquet` (confidence=1.0 for manually reviewed rows).

### Step 7b — If NO: Save and finish

Save as-is. Disputed rows keep their auto_label with low confidence.

### Step 8 — Generate notebook and summary (SILENT)

Create `notebooks/annotation.ipynb` with:
1. Label distribution bar chart
2. Confidence distribution histogram
3. Disputed examples table (text, auto_label, confidence)
4. If original labels exist: confusion matrix (auto vs original), Cohen's kappa
5. **Markdown cell with justification**: annotation approach, what the model was confident about and where it was uncertain

Execute:
```bash
python .claude/skills/data-annotation/scripts/run_notebook.py notebooks/annotation.ipynb
```

### Step 9 — Final summary

```
Annotation complete!
   Model: OpenRouter API
   Labels: pos, neg, neutral, mixed
   Total annotated: 5,062
   Manually reviewed: 153 (in Label Studio)
   Confidence: mean=0.91
   Disputed remaining: 0

   Saved:
     data/labeled/labeled.parquet
     data/annotation/annotation_spec.md
     data/annotation/quality_metrics.json
     data/annotation/labelstudio_tasks.json
     notebooks/annotation.ipynb (with outputs)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/auto_label.py` | Annotate ALL texts via OpenRouter API, save labeled.parquet |
| `scripts/label_unlabeled.py` | Annotate only UNLABELED rows (Option 3) via OpenRouter API, save labeled.parquet |
| `scripts/check_quality.py` | Compute Cohen's kappa, confidence stats, save metrics JSON |
| `scripts/export_ls.py` | Export disputed examples to Label Studio JSON + config XML |
| `scripts/start_ls.py` | Start Label Studio, create project, import tasks, open browser |
| `scripts/import_ls.py` | Merge manual labels from LS export back into labeled.parquet |
| `scripts/run_notebook.py` | Execute a Jupyter notebook in-place and save with outputs |

## Config

Input dataset: `data/cleaned/cleaned.parquet` (output of data-detective)
Output dataset: `data/labeled/labeled.parquet` (input for the next pipeline stage)
Artifacts: `data/annotation/` (spec, metrics, LS tasks, LS config, LS export)
Label Studio: `pip install label-studio label-studio-sdk`, runs on localhost:8080
API key: `OPENROUTER_API_KEY` in `.env`
