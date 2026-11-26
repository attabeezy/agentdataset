"""Contract for the schema-guided extraction study."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def test_schema_guided_micro_f1_exceeds_target():
    totals = pd.read_csv(RESULTS / "schema_guided_extraction_totals.csv")
    assert len(totals) == 1
    row = totals.iloc[0]
    assert row["arm"] == "schema_guided"
    assert row["false_positives"] <= 2
    assert row["precision"] >= 0.9
    assert row["recall"] >= 0.85
    assert row["f1"] >= 0.5


def test_schema_guided_per_source_artifacts_exist():
    frame = pd.read_csv(RESULTS / "schema_guided_extraction.csv")
    assert set(frame["source_id"]) == {
        "cdc_nhanes_rural_urban", "calerie_rct", "student_mental_health"
    }
    assert (frame["arm"] == "schema_guided").all()
    assert (RESULTS / "schema_guided_extraction_raw.csv").exists()
    assert (RESULTS / "schema_guided_extraction_manifest.json").exists()
