"""
DataQualityAgent
================
Detects data quality issues, applies cleaning strategies,
and compares before/after metrics.

Unified input schema (from DataCollectionAgent):
    text         str   — review / document text
    label        str   — class label (nullable — missing labels are fine)
    source       str   — source identifier
    collected_at str   — ISO 8601 timestamp

Usage
-----
    from agents.data_quality_agent import DataQualityAgent

    agent = DataQualityAgent()

    # 1. Detect all issues
    report = agent.detect_issues(df)
    print(report)                    # QualityReport dataclass

    # 2. Apply a preset strategy
    df_clean = agent.fix(df, strategy="balanced")

    # 3. Compare before / after
    comparison = agent.compare(df, df_clean)
    print(comparison.to_markdown())

    # 4. Save
    agent.save(df_clean)             # → data/cleaned/cleaned.parquet
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Container for all detected issues."""
    total_rows: int = 0
    missing: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    near_duplicates: int = 0
    length_outliers: int = 0
    length_stats: dict[str, float] = field(default_factory=dict)
    empty_texts: int = 0
    inconsistent_labels: list[tuple[str, str]] = field(default_factory=list)
    encoding_issues: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    short_texts: int = 0    # < MIN_TEXT_LEN chars
    long_texts: int = 0     # > MAX_TEXT_LEN chars
    extra: dict[str, Any] = field(default_factory=dict)

    # Thresholds used during detection (stored for reproducibility)
    iqr_lower: float = 0.0
    iqr_upper: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "missing": self.missing,
            "duplicates": self.duplicates,
            "near_duplicates": self.near_duplicates,
            "length_outliers": self.length_outliers,
            "length_stats": self.length_stats,
            "empty_texts": self.empty_texts,
            "inconsistent_labels": self.inconsistent_labels,
            "encoding_issues": self.encoding_issues,
            "class_distribution": self.class_distribution,
            "imbalance_ratio": round(self.imbalance_ratio, 4),
            "short_texts": self.short_texts,
            "long_texts": self.long_texts,
            "iqr_lower": self.iqr_lower,
            "iqr_upper": self.iqr_upper,
            "extra": self.extra,
        }

    def __repr__(self) -> str:
        lines = [
            f"QualityReport ({self.total_rows:,} rows)",
            f"  Missing values      : {self.missing}",
            f"  Exact duplicates    : {self.duplicates}",
            f"  Near-duplicates     : {self.near_duplicates}",
            f"  Length outliers     : {self.length_outliers} "
            f"  (IQR bounds: {self.iqr_lower:.0f}–{self.iqr_upper:.0f})",
            f"  Empty texts         : {self.empty_texts}",
            f"  Encoding issues     : {self.encoding_issues}",
            f"  Inconsistent labels : {len(self.inconsistent_labels)} pairs",
            f"  Class imbalance     : {self.imbalance_ratio:.2f}x",
            f"  Short texts (<20)   : {self.short_texts}",
            f"  Long texts (>5000)  : {self.long_texts}",
        ]
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Before / after metrics."""
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        rows = ["| Metric | Before | After | Change |",
                "|--------|--------|-------|--------|"]
        for metric, vals in self.metrics.items():
            rows.append(
                f"| {metric} | {vals.get('before', '—')} "
                f"| {vals.get('after', '—')} "
                f"| {vals.get('change', '—')} |"
            )
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return self.metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_TEXT_LEN = 20       # chars — below this a review carries no sentiment signal
MAX_TEXT_LEN = 5_000    # chars — above this likely a data-entry artifact

HTML_ENTITY_RE = re.compile(r"&(?:[a-z]+|#\d+);", re.IGNORECASE)
MARKUP_TAG_RE  = re.compile(r"<[^>]{1,60}>")
CTRL_CHAR_RE   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Canonical label map for sentiment analysis
DEFAULT_LABEL_MAP: dict[str, str] = {
    "positive": "pos",
    "1":        "pos",
    "true":     "pos",
    "negative": "neg",
    "0":        "neg",
    "false":    "neg",
}


# ---------------------------------------------------------------------------
# DataQualityAgent
# ---------------------------------------------------------------------------

class DataQualityAgent:
    """Detect, fix, and compare data quality for text classification datasets."""

    def __init__(
        self,
        min_text_len: int = MIN_TEXT_LEN,
        max_text_len: int = MAX_TEXT_LEN,
        label_map: dict[str, str] | None = None,
        output_dir: str | Path = "data/cleaned",
    ) -> None:
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Skill: detect_issues
    # ------------------------------------------------------------------

    def detect_issues(self, df: pd.DataFrame) -> QualityReport:
        """Run full quality scan on a DataFrame.

        Returns a QualityReport with every issue found.
        NOTE: missing labels (label column nulls) are intentionally ignored.
        """
        report = QualityReport(total_rows=len(df))
        text = df["text"].astype(str) if "text" in df.columns else pd.Series([], dtype=str)

        # 1. Missing values (text + metadata only, skip label)
        skip_cols = {"label"}
        report.missing = {
            col: int(df[col].isna().sum())
            for col in df.columns
            if col not in skip_cols and df[col].isna().any()
        }

        # 2. Exact duplicates
        report.duplicates = int(text.duplicated().sum())

        # 3. Text length stats + IQR outliers
        lengths = text.str.len()
        q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        report.iqr_lower = float(lower)
        report.iqr_upper = float(upper)
        report.length_outliers = int(((lengths < lower) | (lengths > upper)).sum())
        report.length_stats = {
            "mean":   round(float(lengths.mean()), 1),
            "median": round(float(lengths.median()), 1),
            "min":    int(lengths.min()),
            "max":    int(lengths.max()),
            "std":    round(float(lengths.std()), 1),
            "q1":     round(float(q1), 1),
            "q3":     round(float(q3), 1),
        }

        # 4. Empty / whitespace-only texts
        report.empty_texts = int((text.str.strip().str.len() == 0).sum())

        # 5. Extreme length texts
        report.short_texts = int((lengths < self.min_text_len).sum())
        report.long_texts  = int((lengths > self.max_text_len).sum())

        # 6. Encoding noise (HTML entities + markup tags)
        html_mask  = text.str.contains(HTML_ENTITY_RE, regex=True)
        markup_mask = text.str.contains(MARKUP_TAG_RE,  regex=True)
        report.encoding_issues = int((html_mask | markup_mask).sum())

        # 7. Near-duplicates (simplified: same text after stripping + lowercasing)
        normalised = text.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
        report.near_duplicates = int(normalised.duplicated().sum()) - report.duplicates
        report.near_duplicates = max(report.near_duplicates, 0)

        # 8. Inconsistent labels (only when label column exists)
        if "label" in df.columns:
            raw_labels = df["label"].dropna().astype(str).unique().tolist()
            pairs: list[tuple[str, str]] = []
            for raw in raw_labels:
                canonical = self.label_map.get(raw.lower())
                if canonical and canonical != raw:
                    pairs.append((raw, canonical))
            report.inconsistent_labels = pairs

            # 9. Class distribution + imbalance
            counts = df["label"].dropna().value_counts()
            report.class_distribution = counts.to_dict()
            if len(counts) >= 2:
                report.imbalance_ratio = round(
                    float(counts.iloc[0]) / float(counts.iloc[-1]), 4
                )

        return report

    # ------------------------------------------------------------------
    # Skill: fix
    # ------------------------------------------------------------------

    def fix(self, df: pd.DataFrame, strategy: str = "balanced") -> pd.DataFrame:
        """Apply a cleaning strategy.

        Strategies
        ----------
        aggressive
            Remove all IQR outliers, deduplicate (exact + near), fix encoding,
            drop short texts.
        conservative
            Only remove exact duplicates. No other changes.
        balanced  (recommended for sentiment analysis)
            Remove exact duplicates, fix encoding issues, drop only the most
            extreme outliers (outside 3×IQR), keep near-duplicates and
            moderately-long texts.
        """
        out = df.copy()

        # Always: normalise labels
        if "label" in out.columns:
            out["label"] = out["label"].apply(
                lambda x: self.label_map.get(str(x).lower(), x) if pd.notna(x) else x
            )

        # Always: fix encoding (safe for every strategy)
        out["text"] = out["text"].astype(str).apply(self._clean_text)

        if strategy == "conservative":
            out = self._drop_exact_duplicates(out)

        elif strategy == "aggressive":
            out = self._drop_exact_duplicates(out)
            out = self._drop_near_duplicates(out)
            out = self._drop_iqr_outliers(out, multiplier=1.5)
            out = out[out["text"].str.len() >= self.min_text_len]

        elif strategy == "balanced":
            out = self._drop_exact_duplicates(out)
            out = self._drop_iqr_outliers(out, multiplier=3.0)  # only extremes

        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                "Choose 'aggressive', 'conservative', or 'balanced'."
            )

        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Skill: compare
    # ------------------------------------------------------------------

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> ComparisonReport:
        """Produce a before/after metric table."""
        r_before = self.detect_issues(df_before)
        r_after  = self.detect_issues(df_after)

        def _chg(a, b) -> str:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                diff = b - a
                pct  = (diff / a * 100) if a else 0
                sign = "+" if diff > 0 else ""
                return f"{sign}{diff} ({sign}{pct:.1f}%)"
            return "—"

        metrics: dict[str, dict] = {
            "Total rows":       {"before": r_before.total_rows,        "after": r_after.total_rows,        "change": _chg(r_before.total_rows,        r_after.total_rows)},
            "Exact duplicates": {"before": r_before.duplicates,        "after": r_after.duplicates,        "change": _chg(r_before.duplicates,        r_after.duplicates)},
            "Near-duplicates":  {"before": r_before.near_duplicates,   "after": r_after.near_duplicates,   "change": _chg(r_before.near_duplicates,   r_after.near_duplicates)},
            "Length outliers":  {"before": r_before.length_outliers,   "after": r_after.length_outliers,   "change": _chg(r_before.length_outliers,   r_after.length_outliers)},
            "Empty texts":      {"before": r_before.empty_texts,       "after": r_after.empty_texts,       "change": _chg(r_before.empty_texts,       r_after.empty_texts)},
            "Encoding issues":  {"before": r_before.encoding_issues,   "after": r_after.encoding_issues,   "change": _chg(r_before.encoding_issues,   r_after.encoding_issues)},
            "Short texts":      {"before": r_before.short_texts,       "after": r_after.short_texts,       "change": _chg(r_before.short_texts,       r_after.short_texts)},
            "Imbalance ratio":  {"before": r_before.imbalance_ratio,   "after": r_after.imbalance_ratio,   "change": _chg(r_before.imbalance_ratio,   r_after.imbalance_ratio)},
        }
        return ComparisonReport(metrics=metrics)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str = "cleaned.parquet") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        df.to_parquet(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Fix encoding noise: HTML entities, markup tags, control characters."""
        text = HTML_ENTITY_RE.sub(lambda m: _decode_html_entity(m.group()), text)
        text = MARKUP_TAG_RE.sub(" ", text)
        text = CTRL_CHAR_RE.sub(" ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    @staticmethod
    def _drop_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        normed = df["text"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
        return df[~normed.duplicated()].reset_index(drop=True)

    @staticmethod
    def _drop_iqr_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
        lengths = df["text"].astype(str).str.len()
        q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
        iqr = q3 - q1
        mask = (lengths >= q1 - multiplier * iqr) & (lengths <= q3 + multiplier * iqr)
        return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# HTML entity decoder (no external deps)
# ---------------------------------------------------------------------------

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&apos;": "'", "&nbsp;": " ",
    "&#39;": "'", "&#34;": '"', "&#38;": "&",
}

def _decode_html_entity(entity: str) -> str:
    if entity in _HTML_ENTITIES:
        return _HTML_ENTITIES[entity]
    if entity.startswith("&#") and entity.endswith(";"):
        try:
            code = int(entity[2:-1])
            return chr(code)
        except (ValueError, OverflowError):
            pass
    return entity
