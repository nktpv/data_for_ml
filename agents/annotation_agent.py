"""Annotation agent: auto-label text data via OpenRouter API, assess quality, export to Label Studio."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from .openrouter_client import DEFAULT_MODEL


@dataclass
class QualityMetrics:
    kappa: float | None
    agreement_pct: float
    label_distribution: dict[str, int]
    confidence_mean: float
    confidence_median: float
    disputed_count: int
    disputed_pct: float
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kappa": self.kappa,
            "agreement_pct": round(self.agreement_pct, 4),
            "label_distribution": self.label_distribution,
            "confidence_mean": round(self.confidence_mean, 4),
            "confidence_median": round(self.confidence_median, 4),
            "disputed_count": self.disputed_count,
            "disputed_pct": round(self.disputed_pct, 4),
            "total": self.total,
        }


class AnnotationAgent:
    """Auto-label text data with OpenRouter API, assess annotation quality, export to Label Studio."""

    LABELED_PATH = ROOT / "data" / "labeled" / "labeled.parquet"
    ANNOTATION_DIR = ROOT / "data" / "annotation"

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold
        self._client = None

    # ------------------------------------------------------------------
    # OpenRouter client (lazy init)
    # ------------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            from agents.openrouter_client import OpenRouterClient

            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENROUTER_API_KEY not set in environment")
            self._client = OpenRouterClient(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Auto-label
    # ------------------------------------------------------------------

    def auto_label(
        self,
        df: pd.DataFrame,
        labels: list[str],
        task_description: str = "text classification",
        batch_size: int = 10,
        allow_new_labels: bool = False,
        model: str = DEFAULT_MODEL,
    ) -> pd.DataFrame:
        """Classify every row in df via OpenRouter API.

        Adds columns: auto_label, confidence, is_disputed.
        """
        df = df.copy()
        texts = df["text"].tolist()
        total = len(texts)
        results: list[dict] = []

        n_batches = (total + batch_size - 1) // batch_size
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(
                f"  Annotating batch {batch_num}/{n_batches} "
                f"({min(i + batch_size, total)}/{total})",
                flush=True,
            )
            batch_results = self._classify_batch(
                batch, labels, task_description, allow_new_labels, model
            )
            results.extend(batch_results)

        df["auto_label"] = [r["label"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["is_disputed"] = df["confidence"] < self.confidence_threshold
        return df

    def _classify_batch(
        self,
        texts: list[str],
        labels: list[str],
        task_description: str,
        allow_new_labels: bool,
        model: str,
    ) -> list[dict]:
        labels_str = ", ".join(labels)
        new_label_note = (
            "You may also suggest a NEW label if none fits — write it in UPPERCASE."
            if allow_new_labels
            else "Use ONLY the labels listed above. Do not invent new ones."
        )
        numbered = "\n".join(f"{idx + 1}. {t[:500]}" for idx, t in enumerate(texts))

        prompt = (
            f"You are an expert text annotator for {task_description}.\n"
            f"Labels: {labels_str}\n"
            f"{new_label_note}\n\n"
            f"For each text below respond with a JSON array (same order).\n"
            f'Each element: {{"label": "<label>", "confidence": <0.0-1.0>}}\n'
            f"Confidence = how certain you are (1.0 = very certain, 0.0 = completely uncertain).\n\n"
            f"Texts:\n{numbered}\n\n"
            f"Respond ONLY with a valid JSON array, no extra text."
        )

        for attempt in range(3):
            try:
                msg = self.client.messages.create(
                    model=model,
                    max_tokens=1024,  # needs ≥1024: ~500 for reasoning + ~200 for JSON
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = msg.content[0].text.strip()
                return self._parse_json_array(raw, len(texts), labels, allow_new_labels)
            except Exception as exc:
                if attempt == 2:
                    print(f"  WARNING: batch failed after 3 attempts: {exc}")
                    return [{"label": labels[0], "confidence": 0.0}] * len(texts)
                time.sleep(2**attempt)

        return [{"label": labels[0], "confidence": 0.0}] * len(texts)

    def _parse_json_array(
        self,
        raw: str,
        expected: int,
        labels: list[str],
        allow_new_labels: bool,
    ) -> list[dict]:
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw.strip())

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
            else:
                raise

        lower_map = {lbl.lower(): lbl for lbl in labels}
        results = []
        for item in data[:expected]:
            lbl = str(item.get("label", labels[0]))
            conf = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            if lbl.lower() in lower_map:
                lbl = lower_map[lbl.lower()]
            elif not allow_new_labels:
                lbl = labels[0]
            results.append({"label": lbl, "confidence": conf})

        while len(results) < expected:
            results.append({"label": labels[0], "confidence": 0.0})

        return results

    # ------------------------------------------------------------------
    # Generate annotation spec
    # ------------------------------------------------------------------

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str,
        labels: list[str],
        model: str = DEFAULT_MODEL,
    ) -> Path:
        """Generate annotation_spec.md with task definition, label descriptions, examples, edge cases."""
        self.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        out_path = self.ANNOTATION_DIR / "annotation_spec.md"

        examples_per_label: dict[str, list[str]] = {}
        if "auto_label" in df.columns and "confidence" in df.columns:
            for lbl in labels:
                mask = (df["auto_label"] == lbl) & (df["confidence"] >= 0.85)
                examples_per_label[lbl] = df[mask]["text"].head(5).tolist()

        examples_json = json.dumps(examples_per_label, ensure_ascii=False, indent=2)

        prompt = (
            f"Generate a detailed annotation specification (Markdown) for the following task.\n\n"
            f"Task: {task}\n"
            f"Labels: {', '.join(labels)}\n\n"
            f"For each label include:\n"
            f"1. A clear definition (2-3 sentences)\n"
            f"2. At least 3 representative examples from the dataset\n"
            f"3. Key distinguishing features vs. other labels\n\n"
            f"Also include:\n"
            f"- General annotation guidelines (how to handle ambiguous cases)\n"
            f"- At least 5 concrete edge cases with the correct label and rationale\n\n"
            f"High-confidence example texts per label (from the actual dataset):\n"
            f"{examples_json}\n\n"
            f"Write a professional annotation specification in Markdown.\n"
            f"Start with: # Annotation Specification: {task}"
        )

        msg = self.client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        spec_text = msg.content[0].text.strip()
        out_path.write_text(spec_text, encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------
    # Check quality
    # ------------------------------------------------------------------

    def check_quality(
        self,
        df: pd.DataFrame,
        confidence_threshold: float | None = None,
    ) -> QualityMetrics:
        """Compute Cohen's kappa, confidence stats, label distribution, disputed count."""
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        total = len(df)
        label_dist = df["auto_label"].value_counts().to_dict()
        conf = df["confidence"]

        disputed = int((conf < threshold).sum())

        kappa: float | None = None
        agreement_pct: float = 0.0
        if "label" in df.columns:
            both = df.dropna(subset=["label", "auto_label"])
            if len(both) > 0:
                agreement_pct = float((both["label"] == both["auto_label"]).mean())
                try:
                    from sklearn.metrics import cohen_kappa_score

                    kappa = float(cohen_kappa_score(both["label"], both["auto_label"]))
                except Exception:
                    kappa = None

        return QualityMetrics(
            kappa=kappa,
            agreement_pct=agreement_pct,
            label_distribution=label_dist,
            confidence_mean=float(conf.mean()),
            confidence_median=float(conf.median()),
            disputed_count=disputed,
            disputed_pct=disputed / total if total else 0.0,
            total=total,
        )

    # ------------------------------------------------------------------
    # Export to Label Studio
    # ------------------------------------------------------------------

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        confidence_threshold: float | None = None,
        export_all: bool = False,
    ) -> Path:
        """Export low-confidence (or all) rows to Label Studio import JSON."""
        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        self.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

        subset = df if export_all else df[df["confidence"] < threshold]

        tasks = []
        for idx, row in subset.iterrows():
            task: dict[str, Any] = {
                "id": int(idx),
                "data": {"text": str(row["text"])},
            }
            if "auto_label" in row and pd.notna(row.get("auto_label")):
                task["predictions"] = [
                    {
                        "model_version": "openrouter-api",
                        "score": float(row.get("confidence", 0.5)),
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [str(row["auto_label"])]},
                            }
                        ],
                    }
                ]
            tasks.append(task)

        out_path = self.ANNOTATION_DIR / "labelstudio_tasks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        return out_path

    def generate_ls_config(self, labels: list[str]) -> Path:
        """Generate Label Studio labeling config XML for text classification."""
        self.ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
        choices_xml = "\n".join(f'    <Choice value="{lbl}" />' for lbl in labels)
        config_xml = (
            "<View>\n"
            '  <Text name="text" value="$text" />\n'
            '  <Choices name="label" toName="text" choice="single" showInLine="true">\n'
            f"{choices_xml}\n"
            "  </Choices>\n"
            "</View>"
        )
        out_path = self.ANNOTATION_DIR / "labelstudio_config.xml"
        out_path.write_text(config_xml, encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------
    # Import from Label Studio
    # ------------------------------------------------------------------

    def import_from_labelstudio(
        self,
        df: pd.DataFrame,
        ls_export_path: str | Path,
    ) -> pd.DataFrame:
        """Merge manual Label Studio annotations back into df.

        Overwrites auto_label / confidence / is_disputed for manually reviewed rows.
        """
        with open(ls_export_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        df = df.copy()
        for ann in annotations:
            completions = ann.get("annotations", ann.get("completions", []))
            if not completions:
                continue
            result = completions[-1].get("result", [])
            choices: list[str] | None = None
            for r in result:
                if r.get("type") == "choices":
                    ch = r.get("value", {}).get("choices", [])
                    if ch:
                        choices = ch
                        break
            if not choices:
                continue

            task_id = ann.get("id")
            text = (ann.get("data") or {}).get("text")
            row_idx = None
            # Label Studio may replace task ids on import; only trust task_id if it points
            # at the same text we exported.
            if (
                task_id is not None
                and task_id in df.index
                and text is not None
                and "text" in df.columns
                and df.at[task_id, "text"] == text
            ):
                row_idx = task_id
            elif (
                task_id is not None
                and task_id in df.index
                and (text is None or "text" not in df.columns)
            ):
                row_idx = task_id
            elif text is not None and "text" in df.columns:
                matches = df.index[df["text"] == text].tolist()
                if len(matches) >= 1:
                    row_idx = matches[0]

            if row_idx is None:
                continue

            lbl = choices[0]
            df.at[row_idx, "auto_label"] = lbl
            df.at[row_idx, "confidence"] = 1.0
            df.at[row_idx, "is_disputed"] = False
            if "label" in df.columns and pd.isna(df.at[row_idx, "label"]):
                df.at[row_idx, "label"] = lbl

        return df

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame) -> Path:
        """Persist labeled DataFrame to data/labeled/labeled.parquet."""
        out_path = self.LABELED_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=True)
        return out_path
