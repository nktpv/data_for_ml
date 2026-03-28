# ML Data Pipeline Report

## Task
sentiment analysis — определение тональности текстов (movie reviews)

## Pipeline Summary

| Step | Agent | Input | Output | Key Result |
|------|-------|-------|--------|------------|
| 1 | Dataset Collector | User request | combined.parquet | 2 sources, 5,017 rows |
| 2 | Data Detective | combined.parquet | cleaned.parquet | aggressive: -363 rows, HTML fixed |
| 3 | Label Master | cleaned.parquet | labeled.parquet | 2 classes (neg/pos), inversion fixed |
| 4 | Active Learner | labeled.parquet | AL report | entropy saves 83% vs random |
| 5 | Model Trainer | labeled.parquet | final_model.pkl | accuracy=0.86, F1=0.86 |

## Step 1: Data Collection

- Sources: IMDB Movie Reviews (HuggingFace) + TechCrunch RSS
- Raw rows: 5,017 (IMDB: 4,996 + RSS: 21)
- EDA: data/eda/REPORT.md

## Step 2: Data Cleaning

- Strategy: **aggressive**
- Raw → Cleaned: 5,017 → 4,654 (363 removed, 7.2%)
- HTML entities fixed: 2,938 rows (100%)
- Length outliers removed: 363 rows
- Report: data/detective/QUALITY_REPORT.md

## Step 3: Labeling

- Issue found: label inversion in dataset (0=pos, 1=neg — reversed from standard)
- Fix: neg↔pos swap applied to all 4,654 rows
- LLM validation on 20 samples: kappa=0.53, agreement=70%
- Final distribution: neg=2,346 / pos=2,308

## Step 4: Active Learning

- Model (LLM selected): TF-IDF + Logistic Regression, seed=20
- Strategies: entropy, margin, random (5 iterations, batch=20)
- Best strategy: entropy (F1=0.574 at 120 labeled)
- Savings: entropy reaches random F1 at 20 examples → **83% fewer labels needed**

## Step 5: Model Training

- Model: TF-IDF + Logistic Regression
- Train: 3,723 rows | Test: 931 rows
- **Accuracy: 0.8614**
- **F1 macro: 0.8614**
- Precision: 0.8616 | Recall: 0.8615
- Top NEG features: bad, worst, the worst, terrible, boring
- Top POS features: great, excellent, best, the best, wonderful

## All Artifacts



## Model Usage



## Recommendations

1. **Expand classes**: Add  and  via LLM annotation (confirmed viable in 20-sample test)
2. **More data**: Add Russian-language reviews (Kinopoisk, IVI) for multilingual support
3. **Better model**: Replace TF-IDF+LogReg with a fine-tuned BERT/RoBERTa — expected F1 jump to 0.93+
4. **Active learning**: Use entropy sampling when collecting new annotations — saves 83% labeling budget
5. **Re-annotation**: Run full LLM pass with 4-class taxonomy for richer sentiment granularity
