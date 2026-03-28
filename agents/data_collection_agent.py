"""
DataCollectionAgent
===================
Core agent for collecting text datasets from multiple sources and
returning a unified DataFrame with a fixed schema.

Unified schema (every source):
  text         str   — main text content
  label        str   — class label (nullable)
  source       str   — source identifier
  collected_at str   — ISO 8601 timestamp

Usage
-----
    from agents.data_collection_agent import DataCollectionAgent

    agent = DataCollectionAgent(config="config.yaml")

    agent = DataCollectionAgent(config='config.yaml')
    df = agent.run(sources=[
        {'type': 'hf_dataset', 'name': 'imdb'},
        {'type': 'scrape', 'url': '...', 'selector': '...'},
    ])
    df.to_parquet("data/raw/combined.parquet", index=False)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

SCHEMA = ["text", "label", "source", "collected_at"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ensure df has all required columns in the right order."""
    if "text" not in df.columns:
        raise ValueError(f"Source '{source}' produced no 'text' column.")
    if "label" not in df.columns:
        df["label"] = None
    df["source"] = source
    df["collected_at"] = _now()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df[SCHEMA].copy()


# ---------------------------------------------------------------------------
# DataCollectionAgent
# ---------------------------------------------------------------------------

class DataCollectionAgent:
    """Collect text data from multiple sources and unify into one DataFrame."""

    def __init__(self, config: str | Path = "config.yaml") -> None:
        with open(config, encoding="utf-8") as f:
            self.cfg: dict[str, Any] = yaml.safe_load(f)

        self.max_samples: int = (
            self.cfg.get("general", {}).get("max_samples_per_source", 5000)
        )
        self.validation_n: int = (
            self.cfg.get("general", {}).get("validation_sample_size", 10)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, sources: list[dict]) -> pd.DataFrame:
        """Collect all sources and return a merged, deduplicated DataFrame."""
        frames: list[pd.DataFrame] = []

        for spec in sources:
            src_type = spec.get("type", "").lower()
            try:
                logger.info("Collecting: %s", spec)
                df = self._dispatch(src_type, spec)
                if df is not None and not df.empty:
                    frames.append(df)
                    logger.info("  → %d rows from %s", len(df), spec)
            except Exception as exc:
                logger.warning("  ✗ Failed (%s): %s", src_type, exc)

        if not frames:
            raise RuntimeError("All sources failed — no data collected.")

        combined = self.merge(frames)
        return combined

    def validate(self, spec: dict) -> bool:
        """Check if source is usable — metadata-only for HF/Kaggle (no download)."""
        src_type = spec.get("type", "").lower()
        if src_type == "hf_dataset":
            return self._validate_hf_metadata(spec["name"])
        elif src_type == "kaggle_dataset":
            return self._validate_kaggle_metadata(spec["name"])
        else:
            # For scrape/api/rss — quick fetch of a few rows is fine
            try:
                df = self._dispatch(src_type, spec, limit=self.validation_n)
                return df is not None and not df.empty
            except Exception:
                return False

    @staticmethod
    def _validate_hf_metadata(name: str) -> bool:
        """Confirm dataset exists on HF Hub via API — zero download."""
        try:
            from huggingface_hub import dataset_info
            from huggingface_hub.utils import RepositoryNotFoundError
            dataset_info(name)
            return True
        except Exception:
            return False

    @staticmethod
    def _validate_kaggle_metadata(name: str) -> bool:
        """Confirm Kaggle dataset exists and has files — zero download."""
        import os
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            return False
        try:
            api = KaggleApi()
            api.authenticate()
            files = api.dataset_list_files(name)
            return bool(files and files.files)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        name: str,
        source: str = "hf",
        split: str = "train",
        text_col: str | None = None,
        label_col: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load a dataset from HuggingFace Hub or Kaggle.

        Args:
            name:      Dataset id, e.g. "fancyzhx/ag_news" or "user/slug".
            source:    "hf" | "kaggle"
            split:     Dataset split (HF only).
            text_col:  Override auto-detected text column.
            label_col: Override auto-detected label column.
            limit:     Max rows to return.
        """
        if source == "hf":
            return self._load_hf(name, split, text_col, label_col, limit)
        elif source == "kaggle":
            return self._load_kaggle(name, text_col, label_col, limit)
        else:
            raise ValueError(f"Unknown source: {source!r}. Use 'hf' or 'kaggle'.")

    def scrape(
        self,
        url: str,
        selector: str,
        label: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Scrape text from a web page using a CSS selector.

        Args:
            url:      Page URL.
            selector: CSS selector that targets text-containing elements.
            label:    Optional fixed label for all scraped rows.
            limit:    Max rows.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError("Install requests and beautifulsoup4: pip install requests beautifulsoup4") from e

        headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetCollector/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        elements = soup.select(selector)
        texts = [el.get_text(strip=True) for el in elements if el.get_text(strip=True)]

        if limit:
            texts = texts[:limit]

        df = pd.DataFrame({"text": texts, "label": label})
        return _to_schema(df, f"scrape:{url}")

    def fetch_api(
        self,
        endpoint: str,
        params: dict | None = None,
        text_field: str = "text",
        label_field: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch JSON data from a public API.

        Args:
            endpoint:    API URL.
            params:      Query parameters.
            text_field:  JSON field containing the text.
            label_field: JSON field containing the label (optional).
            limit:       Max rows.
        """
        try:
            import requests
        except ImportError as e:
            raise ImportError("Install requests: pip install requests") from e

        resp = requests.get(endpoint, params=params or {}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Handle both list and {"items": [...]} style responses
        if isinstance(data, dict):
            data = data.get("items") or data.get("results") or data.get("data") or []
        if not isinstance(data, list):
            raise ValueError(f"Cannot parse API response as a list: {type(data)}")

        rows = []
        for item in data:
            text = item.get(text_field, "")
            label = item.get(label_field) if label_field else None
            rows.append({"text": text, "label": label})

        df = pd.DataFrame(rows)
        if limit:
            df = df.head(limit)
        return _to_schema(df, f"api:{endpoint}")

    def fetch_rss(
        self,
        url: str,
        label_map: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch articles from an RSS/Atom feed.

        Args:
            url:       Feed URL.
            label_map: Map feed category names to label strings.
            limit:     Max rows.
        """
        try:
            import feedparser
        except ImportError as e:
            raise ImportError("Install feedparser: pip install feedparser") from e

        feed = feedparser.parse(url)
        rows = []
        for entry in feed.entries:
            text = getattr(entry, "summary", "") or getattr(entry, "title", "")
            category = getattr(entry, "tags", [{}])[0].get("term") if hasattr(entry, "tags") else None
            label = (label_map or {}).get(category) if category else None
            rows.append({"text": text, "label": label})

        df = pd.DataFrame(rows)
        if limit:
            df = df.head(limit)
        return _to_schema(df, f"rss:{url}")

    def merge(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate multiple DataFrames, drop duplicates on 'text'."""
        combined = pd.concat(frames, ignore_index=True)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["text"])
        after = len(combined)
        if before != after:
            logger.info("Deduplication: %d → %d rows", before, after)
        return combined.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dispatch(self, src_type: str, spec: dict, limit: int | None = None) -> pd.DataFrame | None:
        lim = limit or self.max_samples
        if src_type == "hf_dataset":
            return self.load_dataset(
                spec["name"], source="hf",
                split=spec.get("split", "train"),
                text_col=spec.get("text_col"),
                label_col=spec.get("label_col"),
                limit=lim,
            )
        elif src_type == "kaggle_dataset":
            return self.load_dataset(
                spec["name"], source="kaggle",
                text_col=spec.get("text_col"),
                label_col=spec.get("label_col"),
                limit=lim,
            )
        elif src_type == "scrape":
            return self.scrape(
                spec["url"], spec.get("selector", "p"),
                label=spec.get("label"),
                limit=lim,
            )
        elif src_type == "api":
            return self.fetch_api(
                spec["endpoint"],
                params=spec.get("params"),
                text_field=spec.get("text_field", "text"),
                label_field=spec.get("label_field"),
                limit=lim,
            )
        elif src_type == "rss":
            return self.fetch_rss(
                spec["url"],
                label_map=spec.get("label_map"),
                limit=lim,
            )
        else:
            raise ValueError(f"Unknown source type: {src_type!r}")

    def _load_hf(
        self,
        name: str,
        split: str,
        text_col: str | None,
        label_col: str | None,
        limit: int | None,
    ) -> pd.DataFrame:
        try:
            from datasets import load_dataset as hf_load
        except ImportError as e:
            raise ImportError("Install datasets: pip install datasets") from e

        # Use streaming for small limits — avoids downloading the full dataset.
        # Full download is used only for large collections (> 5000 rows).
        STREAMING_THRESHOLD = 5000
        use_streaming = limit is not None and limit <= STREAMING_THRESHOLD

        if use_streaming:
            ds_stream = hf_load(name, split=split, streaming=True)
            take_n = min(limit * 2, 10000)  # 2x for stratification headroom
            rows = list(ds_stream.take(take_n))
            df = pd.DataFrame(rows)
            # Feature names for integer label mapping
            label_names = None
            if hasattr(ds_stream, "features") and ds_stream.features:
                feats = ds_stream.features
                candidate = next(
                    (c for c in ["label", "category", "class", "target", "sentiment"] if c in feats),
                    None,
                )
                if candidate and hasattr(feats[candidate], "names"):
                    label_names = {i: v for i, v in enumerate(feats[candidate].names)}
        else:
            ds = hf_load(name, split=split)
            ds = ds.shuffle(seed=42)
            if limit:
                ds = ds.select(range(min(limit * 2, len(ds))))
            df = ds.to_pandas()
            label_names = None
            if hasattr(ds, "features"):
                candidate = next(
                    (c for c in ["label", "category", "class", "target", "sentiment"] if c in ds.features),
                    None,
                )
                if candidate and hasattr(ds.features[candidate], "names"):
                    label_names = {i: v for i, v in enumerate(ds.features[candidate].names)}

        text_col = text_col or self._detect_col(df, ["text", "sentence", "content", "review", "article", "body"])
        label_col = label_col or self._detect_col(df, ["label", "category", "class", "target", "sentiment"])

        result = pd.DataFrame()
        result["text"] = df[text_col]
        result["label"] = df[label_col].astype(str) if label_col else None

        # Map integer labels to string names when available
        if label_col and label_names:
            result["label"] = df[label_col].map(label_names)

        # Stratified sampling: equal rows per class (when labels exist and limit set)
        if limit and label_col and result["label"].notna().any():
            result = self._stratified_sample(result, limit)
        elif limit:
            result = result.head(limit)

        return _to_schema(result, f"hf:{name}")

    @staticmethod
    def _stratified_sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Sample up to `limit` rows with equal representation per label class."""
        labeled = df[df["label"].notna()]
        unlabeled = df[df["label"].isna()]
        classes = labeled["label"].unique()
        n_classes = len(classes)
        if n_classes == 0:
            return df.head(limit)
        per_class = limit // n_classes
        frames = [
            labeled[labeled["label"] == cls].head(per_class)
            for cls in classes
        ]
        sampled = pd.concat(frames, ignore_index=True)
        # Fill remaining slots with unlabeled rows if any
        remaining = limit - len(sampled)
        if remaining > 0 and not unlabeled.empty:
            sampled = pd.concat([sampled, unlabeled.head(remaining)], ignore_index=True)
        return sampled

    def _load_kaggle(
        self,
        name: str,
        text_col: str | None,
        label_col: str | None,
        limit: int | None,
    ) -> pd.DataFrame:
        import tempfile
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise ImportError("Install kaggle: pip install kaggle") from e

        api = KaggleApi()
        api.authenticate()

        # Kaggle always requires downloading the zip, but we read only `limit` rows.
        # For large limits, download once and read fully; for small limits, nrows caps it.
        with tempfile.TemporaryDirectory() as tmpdir:
            api.dataset_download_files(name, path=tmpdir, unzip=True)
            csvs = sorted(Path(tmpdir).glob("**/*.csv"), key=lambda p: p.stat().st_size, reverse=True)
            if not csvs:
                raise FileNotFoundError(f"No CSV files found in Kaggle dataset '{name}'")
            df = pd.read_csv(csvs[0], nrows=limit)  # nrows=None reads all

        text_col = text_col or self._detect_col(df, ["text", "sentence", "content", "review", "article"])
        label_col = label_col or self._detect_col(df, ["label", "category", "class", "target"])

        result = pd.DataFrame()
        result["text"] = df[text_col]
        result["label"] = df[label_col].astype(str) if label_col else None

        # Stratified sampling when labels exist
        if limit and label_col and result["label"].notna().any():
            result = self._stratified_sample(result, limit)
        elif limit:
            result = result.head(limit)

        return _to_schema(result, f"kaggle:{name}")

    @staticmethod
    def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        # fuzzy: first string column
        str_cols = [c for c in df.columns if df[c].dtype == object]
        return str_cols[0] if str_cols else None