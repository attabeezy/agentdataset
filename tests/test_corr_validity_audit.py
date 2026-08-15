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


def test_regex_correlations_resolve_against_extracted_variables():
    """Regression guard on the §9.7 fix.

    The regex extractor used to name continuous variables positionally
    (var_1, var_2, ...) while emitting correlations under the names found in the
    text, so no declared pair could ever resolve: every regex correlation was
    discarded before positive-definiteness was even considered, leaving a valid
    but vacuous identity matrix. Names are now recovered from the surrounding
    prose and normalized on both sides.

    If this fails, dependence is being silently dropped again.
    """
    import benchmark
    from agentdataset.core.extractor import Extractor

    text = benchmark._build_ablation_texts()[0]
    params = Extractor(api_key="", env_var="_TEST_NO_KEY").extract_parameters(
        text["text"], "t"
    )

    assert params.correlations, "the text does declare a correlation"
    declared_vars = set(params.variables)
    for corr in params.correlations.values():
        assert corr.var1 in declared_vars, f"{corr.var1} does not resolve"
        assert corr.var2 in declared_vars, f"{corr.var2} does not resolve"

    spec = cva._spec("regex_probe", "auto_extracted", params, "regex")
    diag, _ = cva.audit_spec(spec)
    assert diag["n_pairs_applied"] == len(params.correlations)
    assert diag["n_pairs_skipped_unknown_var"] == 0
    # A matrix that actually carries a correlation is no longer the identity.
    assert diag["min_eigenvalue"] < 1.0
    assert diag["action"] == "cholesky_ok"


def test_unnamed_statistics_still_fall_back_to_positional_names():
    """The fix must not invent names when the prose does not supply one.

    A wrong name is worse than no name: it mismatches silently, where var_N is
    at least visibly unresolved.
    """
    from agentdataset.core.extractor import Extractor

    variables, _ = Extractor(api_key="", env_var="_TEST_NO_KEY")._extract_with_regex(
        "The mean is 10.5 and the standard deviation is 2.1."
    )
    assert list(variables) == ["var_1"]


def test_collect_specs_covers_every_input_class():
    specs, _notes = cva.collect_specs()
    classes = {s["source_class"] for s in specs}
    assert {"templated", "auto_extracted", "stress"} <= classes
    assert all(s["params"].correlations for s in specs)
