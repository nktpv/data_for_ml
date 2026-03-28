# Data Quality Report

## Task: Sentiment Analysis (IMDB reviews)

## Cleaning strategy applied: `aggressive`

## Before → After

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total rows | 5017 | 4654 | -363 (-7.2%) |
| Exact duplicates | 0 | 0 | 0 (0.0%) |
| Near-duplicates | 0 | 0 | 0 (0.0%) |
| Length outliers | 356 | 183 | -173 (-48.6%) |
| Empty texts | 0 | 0 | 0 (0.0%) |
| Encoding issues | 2938 | 0 | -2938 (-100.0%) |
| Short texts | 0 | 0 | 0 (0.0%) |
| Imbalance ratio | 1.0016 | 1.0165 | +0.014899999999999913 (+1.5%) |

## Notes

- **Exact duplicates** were removed to prevent data leakage between train/test splits.
- **HTML entities** (`&amp;`, `&#39;`, `<br>`) were decoded — raw markup confuses subword tokenisers.
- **IQR outliers** (multiplier 3×) were trimmed; moderate-length reviews were kept because longer texts often carry richer sentiment signal.
- **Class balance** was not altered — the IMDB dataset is inherently ~50/50 (pos/neg), which is ideal for binary classification.