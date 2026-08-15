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


def test_regex_extraction_produces_vacuous_matrices():
    """Regression guard on the §9.7 headline finding.

    The regex extractor names variables positionally (var_1, var_2, ...) but
    emits correlations under the names found in the text, so no declared pair
    can ever resolve. Every regex correlation is silently discarded before the
    positive-definiteness question is even reached. If this test starts failing,
    the extractor was fixed and the paper's numbers must be regenerated.
    """
    import benchmark
    from agentdataset.core.extractor import Extractor

    text = benchmark._build_ablation_texts()[0]
    params = Extractor(api_key="", env_var="_TEST_NO_KEY").extract_parameters(
        text["text"], "t"
    )

    assert params.correlations, "the text does declare a correlation"
    declared_vars = set(params.variables)
    named = {
        name
        for corr in params.correlations.values()
        for name in (corr.var1, corr.var2)
    }
    assert not (named & declared_vars), "no correlation variable resolves"

    spec = cva._spec("regex_probe", "auto_extracted", params, "regex")
    diag, _ = cva.audit_spec(spec)
    assert diag["n_pairs_applied"] == 0
    assert diag["n_pairs_skipped_unknown_var"] == len(params.correlations)
    # The matrix is the identity, so it passes the PD check while imposing nothing.
    assert diag["is_pd"] is True
    assert diag["min_eigenvalue"] == pytest.approx(1.0)


def test_collect_specs_covers_every_input_class():
    specs, _notes = cva.collect_specs()
    classes = {s["source_class"] for s in specs}
    assert {"templated", "auto_extracted", "stress"} <= classes
    assert all(s["params"].correlations for s in specs)
