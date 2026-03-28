---
name: data-detective
description: "Detect and fix data quality issues: missing values, duplicates, outliers, class imbalance, label normalization. Generates cleaned dataset + quality notebook."
---

# Data Detective

Scans the collected dataset for ALL quality issues, proposes cleaning strategies, waits for the user to choose one, then saves the cleaned data and generates a before/after quality report with a notebook.

## Trigger

User asks to check, clean, fix, or analyze data quality. Can also be called automatically after the data-collector skill finishes.

Examples:
- "проверь качество данных"
- "почисти датасет"
- "find data quality issues"
- "run data detective on combined.parquet"

## Autonomy rules

- **Steps 1, 2** — fully autonomous, no output, no questions.
- **Step 3 is MANDATORY and BLOCKING** — show all findings, wait for the user to pick a strategy. Never proceed without a reply.
- **Steps 4, 5, 6** — fully autonomous after the user responds.

---

## Workflow

### Step 1 — Deep issue scan (silent)

Load dataset and task context from config:

```python
import pandas as pd
import yaml
from agents.data_quality_agent import DataQualityAgent

df = pd.read_parquet("data/raw/combined.parquet")
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
task_type = cfg["task"]["type"]   # e.g. "sentiment analysis"

agent = DataQualityAgent()
report = agent.detect_issues(df)
```

Use `task_type` to frame the analysis — for sentiment analysis the key quality concerns are: label consistency (pos/neg must be uniform), text length extremes (very short reviews carry little signal), and duplicate reviews that would leak between train/test splits.

**Critical rule: missing labels are NOT a defect.** A downstream annotation step will handle unlabelled rows. Therefore:
- Nulls in the `label` column → ignore, do not count as missing values
- Class imbalance → report for visibility only, do NOT mark as something to fix
- Unlabelled rows → keep, never drop

Run the built-in detection, then go further as an LLM and look for:

1. **Missing values** — nulls in `text` and metadata columns (not `label`)
2. **Exact duplicates** — identical `text` values
3. **Length outliers** — IQR-based detection on character count
4. **Empty / whitespace-only texts** — rows with no usable content
5. **Inconsistent labels** — same sentiment expressed differently across sources (`pos` vs `positive` vs `1`, `neg` vs `negative` vs `0`)
6. **Near-duplicates** — reviews that differ only in whitespace, punctuation, or a trailing sentence
7. **Encoding noise** — mojibake, HTML entities (`&amp;`, `&#39;`), leftover markup tags
8. **Extreme lengths** — reviews under ~20 characters (not enough signal) or over ~5 000 characters (possible data entry error or concatenation artifact)
9. **Language mixing** — non-English reviews in a dataset expected to be English
10. **Source quality** — does one source contribute disproportionately short or noisy texts?

Run detection and persist findings:

```bash
python .claude/skills/data-detective/scripts/detect.py data/raw/combined.parquet
```

Saves `data/detective/problems.json`.

---

### Step 2 — Compute all three strategies and compare (silent)

**Label normalisation is applied first**, regardless of which strategy the user picks. Map any inconsistent variants to the canonical form:

```python
label_map = {
    "positive": "pos",
    "1":        "pos",
    "negative": "neg",
    "0":        "neg",
}
df["label"] = df["label"].map(lambda x: label_map.get(str(x).lower(), x))
```

Then run all three strategies and collect stats (do NOT save any file yet):

```python
results = {}
for strategy in ["aggressive", "conservative", "balanced"]:
    df_fixed = agent.fix(df, strategy=strategy)
    results[strategy] = {
        "df":         df_fixed,
        "comparison": agent.compare(df, df_fixed),
        "report":     agent.detect_issues(df_fixed),
    }
```

---

### Step 3 — Present findings and ask the user to choose (HUMAN-IN-THE-LOOP ⛔ STOP)

Show everything found, then the strategy comparison table:

```
Problems detected:

| #  | Problem                  | Count   | Severity | Examples                                   |
|----|--------------------------|---------|----------|--------------------------------------------|
| 1  | Inconsistent labels      | 4 pairs | HIGH     | positive↔pos, negative↔neg                |
| 2  | Text length outliers     | 43      | MEDIUM   | 8 chars min, 6 420 chars max               |
| 3  | Exact duplicates         | 8       | MEDIUM   | "Great movie!" appears 5 times             |
| 4  | Near-duplicates          | 3       | LOW      | Same review with minor punctuation diffs   |
| 5  | HTML entities in text    | 18      | LOW      | &amp; &#39; <br> found in 18 rows          |
| 6  | Class imbalance          | 1.02x   | INFO     | pos: 50.5%, neg: 49.5% (well balanced)    |

Label normalisation (applied before any strategy):
  positive → pos,  1 → pos,  negative → neg,  0 → neg

How each strategy handles the problems:

| Metric               | Original | Aggressive | Conservative | Balanced  |
|----------------------|----------|------------|--------------|-----------|
| Total rows           | 5 100    | 5 030      | 5 100        | 5 062     |
| Exact duplicates     | 8        | 0          | 0            | 0         |
| Length outliers      | 43       | 0          | 43           | 9         |
| HTML entities fixed  | 18       | 18         | 0            | 18        |
| Rows removed         | —        | 70 (1.4%)  | 8 (0.2%)     | 38 (0.7%) |

Recommendation:
  For sentiment analysis, "balanced" is the best fit:
  - Duplicates are removed (they would inflate accuracy metrics on the test split)
  - HTML entities are cleaned (noise that confuses the tokeniser)
  - Only the most extreme length outliers are dropped — preserving long reviews
    that may contain strong sentiment signal
  - The near-50/50 class balance is untouched (no upsampling or downsampling needed)

Pick a strategy — type 1 (aggressive), 2 (conservative), or 3 (balanced),
or describe a custom approach:
```

**STOP. Do not write any files until the user replies.**

---

### Step 4 — Apply the chosen strategy (silent)

```bash
python .claude/skills/data-detective/scripts/fix.py balanced data/raw/combined.parquet
```

Saves only `data/cleaned/cleaned.parquet`.

---

### Step 5 — Generate comparison report and quality notebook (silent)

**Charts** (raw + before/after):

```bash
python .claude/skills/data-detective/scripts/visualize.py \
    data/raw/combined.parquet \
    data/cleaned/cleaned.parquet
```

Saves to `data/detective/`: `missing_values.png`, `outliers.png`, `class_balance.png`, `before_after_rows.png`, `before_after_class_balance.png`, `before_after_text_lengths.png`.

**Comparison report**:

```bash
python .claude/skills/data-detective/scripts/compare.py \
    data/raw/combined.parquet \
    data/cleaned/cleaned.parquet \
    balanced
```

Saves `data/detective/QUALITY_REPORT.md`.

**Quality notebook** — generate `notebooks/data_quality.ipynb` with the following cells, tailored to the actual data:

1. **Markdown title** — "Data Quality Report: Sentiment Analysis (IMDB)"
2. **Load data** — read `data/raw/combined.parquet` and `data/cleaned/cleaned.parquet`
3. **Detected problems** — one code cell + chart per issue:
   - Missing values bar chart
   - Text length histogram with IQR bounds
   - Exact duplicate count + example rows
   - Class balance bar chart (before label normalisation)
   - HTML entity examples if any were found
4. **Label normalisation** — show the mapping, then updated class balance chart
5. **Raw vs Cleaned comparison** — pandas DataFrame with Before / After / Change columns; embed the `before_after_*.png` charts
6. **Justification (markdown cell)** — THIS IS THE KEY DELIVERABLE:
   - List each problem found and its severity
   - Explain what was fixed and why it matters for sentiment analysis specifically
     (e.g. "Duplicate reviews would leak between train/test splits, inflating accuracy")
   - Explain what was left untouched and why
     (e.g. "Class balance is 50/50 — no resampling needed")
   - Reference actual numbers from the data, not generic boilerplate
7. **Summary** — key numbers and recommended next steps

Execute the notebook in-place:

```bash
python .claude/skills/data-detective/scripts/run_notebook.py notebooks/data_quality.ipynb
```

Verify these files exist and are non-empty before proceeding:
- `notebooks/data_quality.ipynb` (with cell outputs)
- `data/detective/QUALITY_REPORT.md`
- `data/detective/outliers.png`
- `data/detective/class_balance.png`
- `data/detective/before_after_class_balance.png`
- `data/detective/before_after_text_lengths.png`

If anything is missing — fix the notebook and re-run.

---

### Step 6 — Summary

```
✅ Data quality check complete!

   Label normalisation: positive→pos, negative→neg, 1→pos, 0→neg
   Problems found: 6 (2 HIGH, 2 MEDIUM, 2 LOW/INFO)
   Strategy applied: balanced

   Raw → Cleaned:
     Rows:          5 100   → 5 062  (-38)
     Duplicates:    8       → 0
     Length outliers: 43   → 9
     HTML entities:  18    → 0

   Saved:    data/cleaned/cleaned.parquet
   Report:   data/detective/QUALITY_REPORT.md
   Charts:   data/detective/*.png
   Notebook: notebooks/data_quality.ipynb
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/detect.py` | Run issue detection, save `data/detective/problems.json` |
| `scripts/fix.py` | Apply a cleaning strategy, save `data/cleaned/cleaned.parquet` |
| `scripts/compare.py` | Diff raw vs cleaned, write `data/detective/QUALITY_REPORT.md` |
| `scripts/visualize.py` | Generate all PNG charts to `data/detective/` |
| `scripts/run_notebook.py` | Execute a notebook in-place and save with outputs |

## Paths (relative to project root)

```
Input:   data/raw/combined.parquet          ← from data-collector skill
Output:  data/cleaned/cleaned.parquet       ← input for next pipeline stage
Reports: data/detective/
         ├── problems.json
         ├── QUALITY_REPORT.md
         ├── missing_values.png
         ├── outliers.png
         ├── class_balance.png
         ├── before_after_rows.png
         ├── before_after_class_balance.png
         └── before_after_text_lengths.png
Notebook: notebooks/data_quality.ipynb
```