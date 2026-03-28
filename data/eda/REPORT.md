# EDA Report — Sentiment Analysis Dataset

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total rows | 5,017 |
| Sources | 2 |
| Labeled rows | 4,996 |
| Unlabeled rows | 21 |
| Avg char length | 1388 |
| Avg word count | 238 |

## Class Distribution

| Label | Count | % |
|-------|-------|---|
| negative | 2500 | 49.8% |
| positive | 2496 | 49.8% |
| unlabeled | 21 | 0.4% |

## Sources

| Source | Rows |
|--------|------|
| hf:ajaykarthick/imdb-movie-reviews | 4996 |
| rss:https://feeds.feedburner.com/TechCrunch | 21 |

## Key Findings

- Dataset is well-balanced between positive/negative IMDB reviews (~50/50)
- 21 unlabeled TechCrunch articles included for potential semi-supervised use
- Movie reviews are long (avg ~230 words), RSS snippets are short (~30 words)
- Common words differ clearly between pos/neg classes — good signal for classification
