---
name: data-collector
description: "Universal dataset search and collection agent. Given a topic, searches multiple sources, validates candidates, asks the user to pick, then collects and unifies data into a single parquet file + EDA notebook."
---

# Data Collector Agent

Universal dataset search and collection agent. Given a topic, it searches multiple sources, validates candidates, asks the user to pick, then collects and unifies the data into a single parquet file.

---

## Trigger

User asks to find / collect / gather data for any ML task.

Examples:
- "find data for spam detection"
- "нужны данные для анализа тональности текстов"
- "collect --topic sentiment analysis"

---

## Autonomy rules

- **Steps 1, 2, 3** — fully autonomous. Silent. No questions.
- **Step 4** — MANDATORY STOP. Show table, wait for user reply. Never proceed without it.
- **Steps 5, 6, 7** — fully autonomous after user selection.
- On any failure: log, skip, move on. Do not ask the user.

---

## Workflow

### Step 1 — Parse the task

Extract from user input:
- `topic` — what the data is about (news, reviews, tweets, emails…)
- `task` — ML goal (classification, sentiment, NER, clustering…)
- `language` — default: English
- `classes` — if mentioned; otherwise leave open

Save to `config.yaml`:
```python
import yaml
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["task"]["type"] = "sentiment analysis "   # derived from input
cfg["task"]["language"] = "en"
with open("config.yaml", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True)
```

---

### Step 2 — Search sources (silent, parallel)

Goal: up to **20 validated candidates** across at least **3 source types**.

**Required source types (must have all):**
- `hf_dataset` — HuggingFace Hub
- `kaggle_dataset` — Kaggle (if `KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`)
- `scrape` or `api` — web scraping or public JSON/RSS API

You MUST always find type of sources like:
`hf_dataset`/`kaggle_dataset` and `scrape`/`api`.
You MUST provide at least 2 data sources (one is an open dataset from HuggingFace/Kaggle, and one is scraping or an API)!

**How to search:**

A) HuggingFace — run with multiple synonyms:
```bash
python .claude/skills/dataset-collector/scripts/search_hf.py "sentiment analysis" 5
python .claude/skills/dataset-collector/scripts/search_hf.py "topic categorization" 5
```

B) Kaggle — if credentials exist:
```bash
python .claude/skills/dataset-collector/scripts/search_kaggle.py "sentiment analysis" 5
```

C) Web search — find relevant APIs, RSS feeds, open datasets. Use your knowledge of well-known datasets:  
`ag_news`, `imdb`, `yelp_review_full`, `dbpedia_14`, `yahoo_answers`, `20newsgroups`, `SetFit/20_newsgroups`, etc.

D) Use built-in knowledge of well-known public sources.

---

### Step 3 — Validate (silent)

**IMPORTANT: validation must be fast — use metadata-only checks, never download datasets.**

For each candidate by source type:

- **hf_dataset** → call `DataCollectionAgent._validate_hf_metadata(name)` — uses `huggingface_hub.dataset_info()`, zero download. Row count and license come from the HF Hub API (`info.card_data`).
- **kaggle_dataset** → call `DataCollectionAgent._validate_kaggle_metadata(name)` — uses `api.dataset_list_files()`, zero download.
- **scrape / api / rss** → one quick HTTP request (3–5 items), confirm it returns text.

**Never call `load_dataset()` or download any files during validation.**
Full data download happens only in Step 5, after the user selects sources.

Drop any source where validation returns False.
If fewer than 2 source **types** survive → search more.

---

### Step 4 — Present table (HUMAN-IN-THE-LOOP ⛔ STOP)

Show a markdown table of validated sources, ranked by relevance.

**Relevance** (1–5 ★): domain match + task match + label quality + language match  
**License**: include if known (e.g. MIT, CC BY 4.0, Apache 2.0, unknown)

```
| #  | Name                  | Type          | Relevance | License     | Description                        | Est. size |
|----|-----------------------|---------------|-----------|-------------|------------------------------------|-----------|
| 1  | fancyzhx/ag_news      | hf_dataset    | ★★★★★     | CC BY 4.0   | 4 news topics, 120k rows           | 120k      |
| 2  | SetFit/20_newsgroups  | hf_dataset    | ★★★★☆     | unknown     | 20 newsgroup topics                | 18k       |
| 3  | BBC RSS (4 cats)      | rss           | ★★★★☆     | © BBC       | Live BBC feed, needs label map     | ~170      |
| 4  | news-scrape.com       | scrape        | ★★★☆☆     | unknown     | Raw articles, no labels            | ~500      |
```

Rules:
- At least 2 different source types in the table
- Sort: relevance desc
- Max 20 rows
- User picks by numbers: `1, 3` or `all` / `все`
- **STOP. Wait. Do not run any collection code until the user replies.**

---

### Step 5 — Collect (silent)

For each selected source, use `DataCollectionAgent`:

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config='config.yaml')
df = agent.run(sources=[
    {'type': 'hf_dataset', 'name': 'imdb'},
    {'type': 'scrape', 'url': '...', 'selector': '...'},
])
```

**Unified schema — every row must have:**

| Column       | Type | Notes                    |
|--------------|------|--------------------------|
| `text`       | str  | Main text content        |
| `label`      | str  | Class label (nullable)   |
| `source`     | str  | Source identifier        |
| `collected_at` | str | ISO 8601 timestamp     |

**Labels are optional.** Collect text-only sources too — a downstream annotation agent handles labelling.

Save:
- `data/raw/combined.parquet`
- `data/raw/<source_name>.parquet` (per source)

---

### Step 6 — EDA (silent)

#### 6.1 — Read the dataset

```python
import pandas as pd
df = pd.read_parquet("data/raw/combined.parquet")
```

Understand what's in the data: column names and types, number of rows, unique labels, unique sources, sample rows.

#### 6.2 — Generate EDA notebook

Create `notebooks/eda.ipynb` with cells **tailored to the specific task and data**. The notebook MUST include:

1. **Title + description** (markdown) — mention the specific task (e.g., "EDA: News Topic Classification")
2. **Load data** — read combined.parquet
3. **Dataset overview** — shape, dtypes, nulls, sample rows
4. **Class distribution** — bar chart + pie chart + table with counts and percentages
5. **Text length analysis** — histogram of char lengths, histogram of word counts, stats table (mean/median/min/max/std)
6. **Text length by class** — overlapping histograms or box plots per label
7. **Top-20 words** — horizontal bar chart (exclude stopwords if possible)
8. **Top words per class** — separate top-10 for each label
9. **Source distribution** — pie chart showing data from each source
10. **Summary** — key findings as text

Important:
- Use `matplotlib` for plots (works headless)
- Save every plot to `data/eda/` as PNG (e.g., `plt.savefig("../data/eda/class_distribution.png")`)
- Also generate `data/eda/REPORT.md` with stats embedded as a markdown table
- Adapt analysis to the actual data — if labels are in Russian, handle encoding; if there are many classes, adjust chart layout

#### 6.3 — Execute the notebook

```bash
python .claude/skills/data-collector/scripts/run_notebook.py notebooks/eda.ipynb
```

This executes every cell in-place and saves the notebook with all outputs (plots, tables, print statements).

#### 6.4 — Verify

Check that these files exist and are non-empty:
- `notebooks/eda.ipynb` (with outputs)
- `data/eda/class_distribution.png`
- `data/eda/text_length_distribution.png`
- `data/eda/top_words.png`
- `data/eda/REPORT.md`

If any are missing, fix the notebook and re-run.

---

### Step 7 — Summary

```
✅ Dataset collected!
   Topic: news topic classification
   Sources: 2 selected (hf_dataset: imdb, rss: BBC News)
   Total rows: 5,500
   Labels: 4 (World, Sports, Business, Sci/Tech)
   Saved: data/raw/combined.parquet
   EDA: data/eda/REPORT.md
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/search_hf.py` | Search HuggingFace Hub, returns JSON |
| `scripts/search_kaggle.py` | Search Kaggle, returns JSON |
| `scripts/eda.py` | Generate EDA charts + REPORT.md |
| `scripts/run_notebook.py` | Execute a notebook in-place and save with outputs |
| `agents/data_collection_agent.py` | Core agent class |

---

## Config

Reads `config.yaml`:
- `general.max_samples_per_source` — cap per source (default 5000)
- `general.validation_sample_size` — rows for validation (default 10)
- `huggingface.enabled` / `kaggle.enabled` / `scraping.enabled` / `rss.enabled`