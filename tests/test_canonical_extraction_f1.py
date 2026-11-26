"""The canonical real-paper extraction must meet the micro-F1 target."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_micro_f1_exceeds_target():
    summary = pd.read_csv(ROOT / "analysis" / "real_paper_extraction_summary.csv")
    row = summary.loc[summary["condition"].eq("llm")].iloc[0]
    assert row["f1_micro"] >= 0.5
    assert row["precision_micro"] >= 0.5
    assert row["recall_micro"] >= 0.85
    assert row["parse_failures"] == 0


def test_caller_schema_does_not_use_gold_labels():
    manifest = json.loads(
        (ROOT / "results" / "real_paper_extraction_schema_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "no gold labels used" in manifest["schema_source"]
    assert manifest["calls"] == 21
