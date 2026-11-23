"""Guards for the §9.7 correlation-matrix validity audit."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import corr_validity_audit as cva  # noqa: E402


def _by_id(specs):
    return {s["source_id"]: s for s in specs}


def test_stress_specs_span_the_validity_spectrum():
    """The audit must contain known-invalid matrices, or a 100% PD rate is vacuous."""
    specs = _by_id(cva._stress_specs())
    assert set(specs) == {
        "stress_valid",
        "stress_slightly_invalid",
        "stress_severely_invalid",
        "stress_singular",
        "stress_dense_invalid",
    }

    results = {name: cva.audit_spec(spec)[0] for name, spec in specs.items()}

    assert results["stress_valid"]["is_pd"] is True
    assert results["stress_valid"]["action"] == "cholesky_ok"

    for name in ("stress_slightly_invalid", "stress_severely_invalid",
                 "stress_singular", "stress_dense_invalid"):
        assert results[name]["is_pd"] is False, name
        assert results[name]["action"] == "dropped_all", name
        assert results[name]["repair_action"] == "repaired_psd", name
        assert results[name]["min_eigenvalue_after"] >= -1e-10, name

    # Severity ordering must be reflected in how far the repair has to move.
    assert (
        results["stress_slightly_invalid"]["frobenius_delta"]
        < results["stress_severely_invalid"]["frobenius_delta"]
    )


def test_repair_beats_dropping_on_correlation_fidelity():
    """The whole point of the repair arm: it should retain more structure."""
    spec = _by_id(cva._stress_specs())["stress_slightly_invalid"]
    _, scores = cva.audit_spec(spec)
    assert scores["corr_score_psd"] > scores["corr_score_drop"]
    assert scores["corr_score_delta_psd_minus_drop"] > 0


def test_singular_matrix_is_classified_invalid_not_borderline():
    """r=1.0 is exactly singular; Cholesky rejects it, so we must too."""
    spec = _by_id(cva._stress_specs())["stress_singular"]
    diag, _ = cva.audit_spec(spec)
    assert diag["min_eigenvalue"] == pytest.approx(0.0, abs=1e-12)
    assert diag["is_pd"] is False
