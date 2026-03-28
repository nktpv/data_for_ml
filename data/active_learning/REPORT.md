# Active Learning Report

## Experiment Setup

- **Seed size**: 20
- **Batch size**: 20
- **Iterations**: 5
- **Final labeled**: 120
- **Strategies**: entropy, margin, random
- **Model**: TF-IDF + Logistic Regression

## Results

### Final Metrics

| Strategy | Accuracy | F1 (macro) |
|----------|----------|------------|
| entropy | 0.6112 | 0.5740 |
| margin | 0.6112 | 0.5740 |
| random | 0.5865 | 0.5128 |

### Learning Curve Data

| Iteration | N labeled | entropy Acc | entropy F1 | margin Acc | margin F1 | random Acc | random F1 |
|-----------|-----------|---------|--------|---------|--------|---------|--------|
| 0 | 20 | 0.5435 | 0.5320 | 0.5435 | 0.5320 | 0.5435 | 0.5320 |
| 1 | 40 | 0.5038 | 0.3350 | 0.5038 | 0.3350 | 0.5038 | 0.3350 |
| 2 | 60 | 0.5081 | 0.3723 | 0.5081 | 0.3723 | 0.5038 | 0.3350 |
| 3 | 80 | 0.5639 | 0.4867 | 0.5639 | 0.4867 | 0.5564 | 0.4560 |
| 4 | 100 | 0.6198 | 0.6012 | 0.6198 | 0.6012 | 0.5843 | 0.5141 |
| 5 | 120 | 0.6112 | 0.5740 | 0.6112 | 0.5740 | 0.5865 | 0.5128 |

## Savings Analysis

**entropy** reaches Random's final F1 (0.5128) at **20** examples.
Savings: **100 examples (83.3%)** vs random baseline.

**margin** reaches Random's final F1 (0.5128) at **20** examples.
Savings: **100 examples (83.3%)** vs random baseline.

## Conclusion

Best strategy: **entropy** (F1 = 0.5740)

Active learning can reach a given quality with fewer labeled examples by prioritizing the most informative points; confirm with multiple random seeds.

## Model Selection (OpenRouter API)

Model chosen by OpenRouter API (nvidia/nemotron-3-super-120b-a12b:free): **Logistic Regression** (`logreg`)
Seed size: **20**

**Reasoning**: The dataset is nearly balanced, so a standard logistic regression with calibrated probabilities provides strong baseline performance and reliable uncertainty estimates for active learning. A seed of 20 samples (10 per class) satisfies the minimum per‑class requirement while being small enough to produce a pronounced learning curve.
