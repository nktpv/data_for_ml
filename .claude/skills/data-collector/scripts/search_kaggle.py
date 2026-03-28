"""Search Kaggle for datasets.

Usage:
    python scripts/search_kaggle.py "news classification" 10

Requires KAGGLE_USERNAME and KAGGLE_KEY in environment or ~/.kaggle/kaggle.json.
Prints a JSON array of dataset descriptors to stdout.
"""

import json
import os
import sys
from pathlib import Path


def _load_env() -> None:
    """Load .env from project root (walk up from this file)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            break


def search(query: str, max_results: int = 10) -> list[dict]:
    _load_env()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(json.dumps({"error": "kaggle package not installed: pip install kaggle"}))
        sys.exit(1)

    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        print(json.dumps({"error": "KAGGLE_USERNAME and KAGGLE_KEY not set in environment"}))
        sys.exit(1)

    api = KaggleApi()
    # Suppress verbose output from Kaggle library
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api.authenticate()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        datasets = api.dataset_list(
            search=query,
            sort_by="hottest",
            file_type="csv",
            max_size=100 * 1024 * 1024,
        )

    results = []
    for ds in list(datasets)[:max_results]:
        downloads = ds.download_count or 0
        score = 2
        if downloads >= 1_000:
            score += 1
        if downloads >= 10_000:
            score += 1
        if query.lower().split()[0] in (ds.title or "").lower():
            score = min(score + 1, 5)

        results.append({
            "name": ds.ref,
            "type": "kaggle_dataset",
            "description": (ds.title or "")[:200],
            "size_bytes": ds.total_bytes,
            "downloads": downloads,
            "relevance": min(score, 5),
        })

    return results


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "text classification"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(json.dumps(search(query, n), ensure_ascii=False, indent=2))