# Model Training Report

## Model Selection (OpenRouter API)

Model chosen by OpenRouter API (nvidia/nemotron-3-super-120b-a12b:free): **Logistic Regression** (`logreg`)

**Reasoning**: The active learning experiments used TF-IDF + Logistic Regression and showed it achieves the highest F1 (0.574) with entropy/margin sampling, indicating this model type is well-suited to the sentiment data. The class distribution is nearly balanced (imbalance ratio 1.0x), so standard logistic regression suffices without needing class‑balanced weighting, and it naturally provides predict_proba for confidence scores.

## Training

- **Model**: TF-IDF + Logistic Regression
- **Train size**: 3723
- **Test size**: 931
- **Classes**: 2 (neg, pos)

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.8614 |
| F1 (macro) | 0.8614 |
| F1 (weighted) | 0.8614 |
| Precision (macro) | 0.8616 |
| Recall (macro) | 0.8615 |

## Per-class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| neg | 0.8712 | 0.8507 | 0.8608 | 469 |
| pos | 0.8520 | 0.8723 | 0.8620 | 462 |

## Artifacts

- `models/final_model.pkl` — trained model (joblib)
- `models/model_config.json` — model configuration
- `data/model/confusion_matrix.png` — confusion matrix
- `data/model/per_class_f1.png` — per-class F1 chart
- `data/model/classification_report.txt` — full classification report
- `data/model/metrics.json` — metrics in JSON format
