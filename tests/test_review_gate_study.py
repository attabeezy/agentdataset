"""Contracts for the agentic review-gate and reconciliation study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def test_review_gate_study_artifacts_exist():
    for name in [
        "review_gate_study_raw.csv",
        "review_gate_study_summary.csv",
        "review_gate_study_decisions.csv",
        "review_gate_study_manifest.json",
    ]:
        assert (RESULTS / name).exists(), name


def test_review_gate_study_has_three_conditions_per_source():
    raw = _read("review_gate_study_raw.csv")
    assert set(raw["condition"]) == {"gold", "unreviewed", "reviewed"}
    assert set(raw["source_id"]) == {
        "cdc_nhanes_rural_urban", "calerie_rct", "student_mental_health"
    }
    assert raw.groupby(["source_id", "condition"]).size().eq(30).all()
    assert set(raw["noise"].unique()) == {0.02}


def test_review_decisions_cover_verified_and_unsupported_claims():
    decisions = _read("review_gate_study_decisions.csv")
    assert set(decisions["group"]) == {"verified", "unsupported"}
    assert set(decisions["decision"]) <= {"accepted", "rejected", "deferred", "edited"}
    assert decisions.groupby("source_id")["claim_id"].nunique().min() > 0


def test_reviewed_parameters_drop_unsupported_claims():
    summary = _read("review_gate_study_summary.csv")
    for source_id in summary["source_id"].unique():
        reviewed = summary.loc[
            summary["source_id"].eq(source_id) & summary["condition"].eq("reviewed")
        ].iloc[0]
        unreviewed = summary.loc[
            summary["source_id"].eq(source_id) & summary["condition"].eq("unreviewed")
        ].iloc[0]
        assert reviewed["n_variables_mean"] <= unreviewed["n_variables_mean"]


def test_manifest_records_provenance_and_reconciliation():
    payload = json.loads(
        (RESULTS / "review_gate_study_manifest.json").read_text(encoding="utf-8")
    )
    provenance = payload.get("provenance", payload)
    assert provenance["command"]
    assert provenance["git_commit"]
    assert payload["model"]
    assert payload["seeds"][0] == 0 and len(payload["seeds"]) == 30
    assert payload["review_calls"] > 0
    assert "pooled_average" in payload["reconciliation_methods"] or "kept_separate" in payload["reconciliation_methods"]
