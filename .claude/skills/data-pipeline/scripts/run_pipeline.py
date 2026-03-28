"""Standalone reproducible pipeline: collect → clean → label → AL → train.

Usage::

    python run_pipeline.py "sentiment analysis"
    python run_pipeline.py "sentiment analysis of product reviews"

This script runs all 5 agents sequentially, from scratch.
Human-in-the-loop checkpoints use input() prompts in the terminal.

For the full interactive experience, use /data-pipeline in Cursor AI instead.
"""

import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd


def step_banner(step: int, name: str) -> None:
    print()
    print("=" * 60)
    print(f"  STEP {step}: {name}")
    print("=" * 60)
    print()


def check_file(path: str, step_name: str) -> bool:
    p = ROOT / path
    if p.exists():
        df = pd.read_parquet(p) if path.endswith(".parquet") else None
        size = f" ({len(df)} rows)" if df is not None else ""
        print(f"  [OK] {path}{size}")
        return True
    else:
        print(f"  [FAIL] {path} not found after {step_name}")
        return False


# ─────────────────────────────────────────────────────────────
# Step 1: Data Collection
# ─────────────────────────────────────────────────────────────
def run_step1_collection(task_description: str) -> pd.DataFrame:
    """Search sources, let user pick, collect data."""
    from agents.data_collection_agent import DataCollectionAgent

    agent = DataCollectionAgent(config=ROOT / "config.yaml")

    # Check for existing data
    raw_path = ROOT / "data/raw/combined.parquet"
    if raw_path.exists():
        df_existing = pd.read_parquet(raw_path)
        print(f"Found existing data/raw/combined.parquet ({len(df_existing)} rows)")
        skip = input("Skip collection and use existing data? [Y/n]: ").strip().lower()
        if skip != "n":
            print("  Using existing data.")
            return df_existing

    # Search for sources
    print(f"Searching datasets for: {task_description}")
    print()
    sources = agent.search_sources(task_description)

    if not sources:
        print("  No sources found. Please check your query or API keys.")
        sys.exit(1)

    # Validate sources
    print(f"Found {len(sources)} candidate sources. Validating...")
    valid_sources = []
    for s in sources:
        try:
            result = agent.validate_source(s)
            ok = result["ok"]
            estimated = result.get("estimated_size", 0)
            reason = result.get("reason", "")
            if ok:
                s["estimated_size"] = estimated
                valid_sources.append(s)
                status = f"OK ~{estimated} rows"
            else:
                status = f"SKIP ({reason})"
        except Exception as e:
            status = f"ERROR: {e}"
        name = s.get("name", s.get("source", "?"))
        print(f"  [{status}] {name}")

    if not valid_sources:
        print("  No valid sources found (min 500 records per source).")
        sys.exit(1)

    # Show validated sources with relevance
    print()
    print("Validated sources:")
    for i, s in enumerate(valid_sources, 1):
        display = s.get("display_name", s.get("name", s.get("source", "?")))
        src_type = s.get("type", "?")
        relevance = s.get("relevance", "?")
        estimated = s.get("estimated_size", "?")
        print(f"  {i}. [{src_type}] {display}  (relevance: {relevance}/5, ~{estimated} rows)")

    # ⏸ CHECKPOINT: let user pick sources
    print()
    print("Enter source numbers to collect (comma-separated), or 'all':")
    choice = input("> ").strip().lower()

    if choice == "all":
        selected = valid_sources
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = [valid_sources[i] for i in indices if 0 <= i < len(valid_sources)]
        except (ValueError, IndexError):
            print("  Invalid selection, using all sources.")
            selected = valid_sources

    if not selected:
        print("  No sources selected.")
        sys.exit(1)

    print(f"\nCollecting {len(selected)} sources...")
    df = agent.run(selected)
    print(f"  Collected {len(df)} rows total.")
    return df


# ─────────────────────────────────────────────────────────────
# Step 2: Data Quality Check
# ─────────────────────────────────────────────────────────────
def run_step2_cleaning() -> pd.DataFrame:
    """Detect issues, show strategies, let user pick, clean."""
    from agents.data_quality_agent import DataQualityAgent

    # Check for existing data
    cleaned_path = ROOT / "data/cleaned/cleaned.parquet"
    if cleaned_path.exists():
        df_existing = pd.read_parquet(cleaned_path)
        print(f"Found existing data/cleaned/cleaned.parquet ({len(df_existing)} rows)")
        skip = input("Skip cleaning and use existing data? [Y/n]: ").strip().lower()
        if skip != "n":
            print("  Using existing data.")
            return df_existing

    agent = DataQualityAgent()
    df_raw = pd.read_parquet(ROOT / "data/raw/combined.parquet")
    print(f"Loaded {len(df_raw)} rows from data/raw/combined.parquet")

    # Detect issues
    print("\nDetecting issues...")
    report = agent.detect_issues(df_raw)
    print(f"  Missing values: {report.missing.total}")
    print(f"  Duplicates: {report.duplicates.total} ({report.duplicates.pct:.1f}%)")
    print(f"  Outliers: {report.outliers.total}")
    print(f"  Empty texts: {report.empty_texts} ({report.empty_texts_pct:.1f}%)")
    if report.imbalance.is_imbalanced:
        print(f"  Class imbalance detected (ratio: {report.imbalance.imbalance_ratio:.1f})")

    # Compare strategies
    strategies = agent.list_strategies()
    print("\nAvailable cleaning strategies:")
    for name, info in strategies.items():
        df_fixed = agent.fix(df_raw, strategy=name)
        removed = len(df_raw) - len(df_fixed)
        print(f"  {name}: {info.get('description', '')} → removes {removed} rows ({removed/len(df_raw)*100:.1f}%)")

    # ⏸ CHECKPOINT: let user pick strategy
    print()
    print("Choose strategy (aggressive / balanced / conservative):")
    choice = input("> ").strip().lower()
    if choice not in strategies:
        print(f"  Unknown strategy '{choice}', using 'balanced'.")
        choice = "balanced"

    df_cleaned = agent.fix(df_raw, strategy=choice)
    print(f"\n  Applied '{choice}': {len(df_raw)} → {len(df_cleaned)} rows")

    # Save
    saved_path = agent.save(df_cleaned)
    print(f"  Saved: {saved_path}")

    # Generate quality report
    comparison = agent.compare(df_raw, df_cleaned)
    report_dir = ROOT / "data/detective"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Data Quality Report",
        "",
        f"**Strategy**: {choice}",
        f"**Before**: {len(df_raw)} rows",
        f"**After**: {len(df_cleaned)} rows",
        f"**Removed**: {len(df_raw) - len(df_cleaned)} rows",
        "",
        "## Metrics Comparison",
        "",
    ]
    for metric, vals in comparison.metrics.items():
        report_lines.append(f"- **{metric}**: {vals.get('before', '?')} → {vals.get('after', '?')}")
    report_lines.append("")

    report_file = report_dir / "QUALITY_REPORT.md"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Report: {report_file}")

    return df_cleaned


# ─────────────────────────────────────────────────────────────
# Step 3: Auto-labeling
# ─────────────────────────────────────────────────────────────
def run_step3_labeling(task_description: str) -> pd.DataFrame:
    """Analyze data, propose taxonomy, let user pick, label."""
    from agents.annotation_agent import AnnotationAgent

    # Check for existing data
    labeled_path = ROOT / "data/labeled/labeled.parquet"
    if labeled_path.exists():
        df_existing = pd.read_parquet(labeled_path)
        print(f"Found existing data/labeled/labeled.parquet ({len(df_existing)} rows)")
        skip = input("Skip labeling and use existing data? [Y/n]: ").strip().lower()
        if skip != "n":
            print("  Using existing data.")
            return df_existing

    agent = AnnotationAgent()
    df_cleaned = pd.read_parquet(ROOT / "data/cleaned/cleaned.parquet")
    print(f"Loaded {len(df_cleaned)} rows from data/cleaned/cleaned.parquet")

    # Read config for allow_new_labels
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    allow_new = config.get("task", {}).get("allow_new_labels", False)

    # Determine labels from existing data or ask user
    existing_labels = []
    if "label" in df_cleaned.columns:
        non_empty = df_cleaned["label"].dropna()
        non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
        if len(non_empty) > 0:
            existing_labels = sorted(non_empty.unique().tolist())

    if existing_labels:
        print(f"\nExisting labels found: {existing_labels}")
        print("\nOptions:")
        print("  1. Use existing labels as taxonomy")
        print("  2. Enter custom labels")
        print("  3. Use existing + allow new labels from LLM")
        choice = input("\nChoose option [1/2/3]: ").strip()

        if choice == "2":
            labels_input = input("Enter labels (comma-separated): ").strip()
            labels = [l.strip() for l in labels_input.split(",") if l.strip()]
        elif choice == "3":
            labels = existing_labels
            allow_new = True
        else:
            labels = existing_labels
    else:
        print("\nNo existing labels found.")
        labels_input = input("Enter labels (comma-separated): ").strip()
        if not labels_input:
            print("  No labels provided. Cannot proceed.")
            sys.exit(1)
        labels = [l.strip() for l in labels_input.split(",") if l.strip()]

    # ⏸ CHECKPOINT: confirm taxonomy
    print(f"\nTaxonomy: {labels}")
    print(f"Allow new labels: {allow_new}")
    confirm = input("Proceed with labeling? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("  Aborted.")
        sys.exit(0)

    # Auto-label
    print(f"\nAuto-labeling {len(df_cleaned)} rows...")
    df_labeled = agent.auto_label(
        df_cleaned,
        labels=labels,
        task_description=task_description,
        allow_new_labels=allow_new,
    )

    # Quality metrics
    metrics = agent.check_quality(df_labeled)
    print(f"\n  Confidence: mean={metrics.confidence_mean:.4f}, median={metrics.confidence_median:.4f}")
    print(f"  Disputed: {metrics.low_confidence_count} ({metrics.low_confidence_pct:.2f}%)")
    print(f"  Label distribution:")
    for label, cnt in sorted(metrics.label_distribution.items(), key=lambda x: -x[1]):
        pct = metrics.label_distribution_pct.get(label, 0)
        print(f"    {label}: {cnt} ({pct:.1f}%)")

    # Save quality metrics
    ann_dir = ROOT / "data/annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    metrics_dict = {
        "confidence_mean": round(metrics.confidence_mean, 4),
        "confidence_median": round(metrics.confidence_median, 4),
        "confidence_std": round(metrics.confidence_std, 4),
        "low_confidence_count": metrics.low_confidence_count,
        "low_confidence_pct": round(metrics.low_confidence_pct, 2),
        "total": metrics.total,
        "label_distribution": metrics.label_distribution,
    }
    with open(ann_dir / "quality_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    # Generate annotation spec
    agent.generate_spec(df_labeled, task=task_description, labels=labels)

    # Save
    saved_path = agent.save(df_labeled)
    print(f"  Saved: {saved_path}")

    # Export disputed to Label Studio
    if metrics.low_confidence_count > 0:
        agent.export_to_labelstudio(df_labeled)
        agent.generate_ls_config(labels)
        print(f"  Exported {metrics.low_confidence_count} disputed examples to Label Studio")

        ls_choice = input("\nStart Label Studio for manual review? [y/N]: ").strip().lower()
        if ls_choice == "y":
            print("  Starting Label Studio...")
            try:
                ls_script = ROOT / ".claude/skills/data-annotation/scripts/start_ls.py"
                subprocess.run(
                    [sys.executable, str(ls_script)],
                    cwd=str(ROOT), timeout=120,
                )
                # After LS, try to import
                import_choice = input("Import Label Studio corrections? [y/N]: ").strip().lower()
                if import_choice == "y":
                    df_labeled = agent.import_from_labelstudio(df_labeled)
                    agent.save(df_labeled)
                    print("  Imported and saved.")
            except Exception as e:
                print(f"  Label Studio error: {e}")
                print("  Continuing with auto-labels.")

    return df_labeled


# ─────────────────────────────────────────────────────────────
# Steps 4-6: Autonomous
# ─────────────────────────────────────────────────────────────
def run_step4_active_learning() -> None:
    """Run AL experiment (fully autonomous)."""
    print("Running AL experiment (fully autonomous)...")
    result = subprocess.run(
        [sys.executable, str(ROOT / ".claude/skills/active-learner/scripts/run_al.py"),
         "data/labeled/labeled.parquet", "20", "5"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  AL failed: {result.stderr[:500]}")
        print("  Continuing anyway...")
    else:
        lines = result.stdout.strip().split("\n")
        for line in lines[-10:]:
            print(f"  {line}")
    check_file("data/active_learning/REPORT.md", "active learning")


def run_step5_model_training() -> None:
    """Train final model (fully autonomous)."""
    print("Training final model (fully autonomous)...")
    result = subprocess.run(
        [sys.executable, str(ROOT / ".claude/skills/model-trainer/scripts/train.py"),
         "data/labeled/labeled.parquet"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  Training failed: {result.stderr[:500]}")
        print("  Continuing anyway...")
    else:
        lines = result.stdout.strip().split("\n")
        for line in lines[-15:]:
            print(f"  {line}")
    check_file("models/final_model.pkl", "model training")
    check_file("data/model/MODEL_REPORT.md", "model training")


def generate_pipeline_report(task_description: str) -> None:
    """Generate PIPELINE_REPORT.md in project root from all step artifacts."""
    lines = [
        "# ML Data Pipeline Report",
        "",
        f"**Task**: {task_description}",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
    ]

    # Step 1 — Collection
    lines.extend(["## Step 1: Data Collection", ""])
    raw_path = ROOT / "data/raw/combined.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        sources = df["source"].nunique() if "source" in df.columns else "?"
        lines.append(f"- **Rows**: {len(df)}")
        lines.append(f"- **Sources**: {sources}")
        if "source" in df.columns:
            for src, cnt in df["source"].value_counts().items():
                lines.append(f"  - {src}: {cnt} rows")
    else:
        lines.append("_Data not found_")
    lines.append("")

    # Step 2 — Cleaning
    lines.extend(["## Step 2: Data Cleaning", ""])
    cleaned_path = ROOT / "data/cleaned/cleaned.parquet"
    if cleaned_path.exists() and raw_path.exists():
        df_raw = pd.read_parquet(raw_path)
        df_clean = pd.read_parquet(cleaned_path)
        removed = len(df_raw) - len(df_clean)
        lines.append(f"- **Before**: {len(df_raw)} rows")
        lines.append(f"- **After**: {len(df_clean)} rows")
        lines.append(f"- **Removed**: {removed} ({removed/len(df_raw)*100:.1f}%)")
    else:
        lines.append("_Data not found_")

    report_path = ROOT / "data/detective/QUALITY_REPORT.md"
    if report_path.exists():
        lines.append(f"- **Report**: data/detective/QUALITY_REPORT.md")
    lines.append("")

    # Step 3 — Labeling
    lines.extend(["## Step 3: Auto-labeling", ""])
    labeled_path = ROOT / "data/labeled/labeled.parquet"
    if labeled_path.exists():
        df_lab = pd.read_parquet(labeled_path)
        n_classes = df_lab["auto_label"].nunique() if "auto_label" in df_lab.columns else "?"
        lines.append(f"- **Rows**: {len(df_lab)}")
        lines.append(f"- **Classes**: {n_classes}")
        if "auto_label" in df_lab.columns:
            for label, cnt in df_lab["auto_label"].value_counts().items():
                pct = cnt / len(df_lab) * 100
                lines.append(f"  - {label}: {cnt} ({pct:.1f}%)")

    metrics_path = ROOT / "data/annotation/quality_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        lines.append(f"- **Confidence mean**: {m.get('confidence_mean', '?')}")
        lines.append(f"- **Disputed**: {m.get('low_confidence_count', '?')} ({m.get('low_confidence_pct', '?')}%)")
    lines.append("")

    # Step 4 — Active Learning
    lines.extend(["## Step 4: Active Learning", ""])
    al_report = ROOT / "data/active_learning/REPORT.md"
    if al_report.exists():
        for hist_file in sorted((ROOT / "data/active_learning").glob("history_*.json")):
            strategy = hist_file.stem.replace("history_", "")
            with open(hist_file) as f:
                h = json.load(f)
            final = h[-1]
            lines.append(f"- **{strategy}**: F1={final['f1']:.4f} (at {final['n_labeled']} examples)")
    else:
        lines.append("_AL results not found_")
    lines.append("")

    # Step 5 — Model Training
    lines.extend(["## Step 5: Model Training", ""])
    model_metrics_path = ROOT / "data/model/metrics.json"
    if model_metrics_path.exists():
        with open(model_metrics_path) as f:
            mm = json.load(f)
        lines.append(f"- **Model**: {mm.get('model_name', '?')}")
        lines.append(f"- **Accuracy**: {mm.get('accuracy', '?')}")
        lines.append(f"- **F1 (macro)**: {mm.get('f1_macro', '?')}")
        lines.append(f"- **F1 (weighted)**: {mm.get('f1_weighted', '?')}")
    else:
        lines.append("_Model metrics not found_")
    lines.append("")

    # All artifacts
    lines.extend(["## All Artifacts", ""])
    artifact_groups = {
        "Data": [
            "data/raw/combined.parquet",
            "data/cleaned/cleaned.parquet",
            "data/labeled/labeled.parquet",
        ],
        "Reports": [
            "data/detective/QUALITY_REPORT.md",
            "data/active_learning/REPORT.md",
            "data/model/MODEL_REPORT.md",
        ],
        "Charts": [
            "data/active_learning/learning_curve.png",
            "data/model/confusion_matrix.png",
            "data/model/per_class_f1.png",
        ],
        "Model": [
            "models/final_model.pkl",
            "models/model_config.json",
        ],
    }

    for group, files in artifact_groups.items():
        lines.append(f"### {group}")
        lines.append("")
        for f in files:
            exists = "x" if (ROOT / f).exists() else " "
            lines.append(f"- [{exists}] `{f}`")
        lines.append("")

    out = ROOT / "PIPELINE_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main(task_description: str) -> None:
    start_time = datetime.now()
    print(f"ML Data Pipeline")
    print(f"Task: {task_description}")
    print(f"Started: {start_time.isoformat()}")

    # Save task to config
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["task"]["type"] = task_description
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Saved task type to config.yaml")

    # ── Step 1: Data Collection ──────────────────────────────
    step_banner(1, "DATA COLLECTION")
    run_step1_collection(task_description)

    if not check_file("data/raw/combined.parquet", "collection"):
        sys.exit(1)

    # ── Step 2: Data Cleaning ────────────────────────────────
    step_banner(2, "DATA QUALITY CHECK")
    run_step2_cleaning()

    if not check_file("data/cleaned/cleaned.parquet", "cleaning"):
        sys.exit(1)

    # ── Step 3: Labeling ─────────────────────────────────────
    step_banner(3, "AUTO-LABELING")
    run_step3_labeling(task_description)

    if not check_file("data/labeled/labeled.parquet", "labeling"):
        sys.exit(1)

    # ── Step 4: Active Learning ──────────────────────────────
    step_banner(4, "ACTIVE LEARNING")
    run_step4_active_learning()

    # ── Step 5: Model Training ───────────────────────────────
    step_banner(5, "MODEL TRAINING")
    run_step5_model_training()

    # ── Step 6: Pipeline Report ──────────────────────────────
    step_banner(6, "PIPELINE REPORT")
    print("Generating pipeline report...")
    generate_pipeline_report(task_description)

    # ── Final Summary ────────────────────────────────────────
    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print()
    print(f"  Task: {task_description}")
    print(f"  Duration: {duration}")
    print()

    artifacts = [
        "data/raw/combined.parquet",
        "data/cleaned/cleaned.parquet",
        "data/labeled/labeled.parquet",
        "data/active_learning/REPORT.md",
        "data/active_learning/learning_curve.png",
        "models/final_model.pkl",
        "models/model_config.json",
        "data/model/MODEL_REPORT.md",
        "data/model/confusion_matrix.png",
        "PIPELINE_REPORT.md",
    ]

    print("  Artifacts:")
    for a in artifacts:
        exists = "OK" if (ROOT / a).exists() else "MISSING"
        print(f"    [{exists}] {a}")

    print()
    print(f"  Full report: PIPELINE_REPORT.md")
    print()
    print("  Reproducible with:")
    print(f'    python run_pipeline.py "{task_description}"')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python run_pipeline.py "Task description"')
        print('Example: python run_pipeline.py "sentiment analysis"')
        sys.exit(1)
    main(sys.argv[1])
