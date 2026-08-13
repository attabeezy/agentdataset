import inspect

import numpy as np
import pandas as pd
import pytest

from agentdataset.evaluation import RAW_RESULT_COLUMNS, validate_raw_results
from analysis.adult_calibration import (
    ORACLE_STRATEGIES,
    _base_row,
    _fit_calibrator,
    _select_f1_threshold,
    evaluate_calibration_strategies,
)


def _synthetic_probabilities(n_val: int = 600, n_test: int = 400, seed: int = 0):
    """Well-separated-ish probabilities whose labels are drawn from those probabilities."""
    rng = np.random.default_rng(seed)
    val_probabilities = rng.uniform(0.0, 1.0, n_val)
    y_val = (rng.uniform(size=n_val) < val_probabilities).astype(int)
    test_probabilities = rng.uniform(0.0, 1.0, n_test)
    y_test = (rng.uniform(size=n_test) < test_probabilities).astype(int)
    return val_probabilities, y_val, test_probabilities, y_test


def _by_strategy(rows):
    return {row["calibration_strategy"]: row for row in rows}


def test_threshold_selection_never_sees_real_test_labels():
    """Flipping every real test label must not move any non-oracle decision.

    The oracle strategy is asserted to move, so this test fails loudly if the strategies
    stop depending on labels at all rather than passing vacuously.
    """
    val_probabilities, y_val, test_probabilities, y_test = _synthetic_probabilities()

    baseline = _by_strategy(
        evaluate_calibration_strategies(val_probabilities, y_val, test_probabilities, y_test)
    )
    flipped = _by_strategy(
        evaluate_calibration_strategies(val_probabilities, y_val, test_probabilities, 1 - y_test)
    )

    honest = [name for name in baseline if name not in ORACLE_STRATEGIES]
    assert len(honest) >= 5
    for name in honest:
        assert baseline[name]["status"] == flipped[name]["status"], name
        if baseline[name]["status"] != "ok":
            continue
        # The threshold and the resulting predictions on the real test fold are the two
        # things leakage would change.
        assert baseline[name]["metrics"]["decision_threshold"] == pytest.approx(
            flipped[name]["metrics"]["decision_threshold"]
        ), name
        assert baseline[name]["metrics"]["pred_pos_rate"] == pytest.approx(
            flipped[name]["metrics"]["pred_pos_rate"]
        ), name

    oracle = "oracle_threshold_real_test"
    assert baseline[oracle]["metrics"]["decision_threshold"] != pytest.approx(
        flipped[oracle]["metrics"]["decision_threshold"]
    )


def test_calibration_helpers_do_not_accept_test_arguments():
    """Leakage is prevented by the signatures, so guard them against refactors."""
    assert list(inspect.signature(_select_f1_threshold).parameters) == ["y_true", "probabilities"]
    assert list(inspect.signature(_fit_calibrator).parameters) == [
        "method", "y_true", "probabilities",
    ]


def test_threshold_selection_maximises_f1_on_validation():
    y_val = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = _select_f1_threshold(y_val, probabilities)
    assert threshold == pytest.approx(0.8)
    predictions = (probabilities >= threshold).astype(int)
    assert predictions.tolist() == y_val.tolist()


def test_isotonic_is_skipped_on_tiny_validation_split():
    val_probabilities, y_val, test_probabilities, y_test = _synthetic_probabilities(n_val=80)
    rows = _by_strategy(
        evaluate_calibration_strategies(val_probabilities, y_val, test_probabilities, y_test)
    )
    for name in ("isotonic", "isotonic_threshold"):
        assert rows[name]["status"] == "skipped"
        assert rows[name]["failure_stage"] == "insufficient_validation_sample"
        assert rows[name]["metrics"] == {}
    # Platt is parametric and still reportable at this size.
    assert rows["platt"]["status"] == "ok"
    assert _fit_calibrator("isotonic", y_val, val_probabilities) is None


def test_single_class_validation_split_skips_every_calibrated_strategy():
    val_probabilities, _, test_probabilities, y_test = _synthetic_probabilities()
    y_val = np.zeros(len(val_probabilities), dtype=int)
    rows = _by_strategy(
        evaluate_calibration_strategies(val_probabilities, y_val, test_probabilities, y_test)
    )
    for name in ("threshold_synth_val", "platt", "platt_threshold", "isotonic", "isotonic_threshold"):
        assert rows[name]["status"] == "skipped"
        assert rows[name]["failure_stage"] == "degenerate_val_split"
    assert rows["uncalibrated"]["status"] == "ok"
    assert rows["oracle_threshold_real_test"]["status"] == "ok"


def test_monotone_calibration_preserves_ranking_metrics():
    """Platt scaling and threshold selection must not change ROC-AUC."""
    val_probabilities, y_val, test_probabilities, y_test = _synthetic_probabilities()
    rows = _by_strategy(
        evaluate_calibration_strategies(val_probabilities, y_val, test_probabilities, y_test)
    )
    reference = rows["uncalibrated"]["metrics"]["roc_auc"]
    for name in ("threshold_synth_val", "platt", "platt_threshold"):
        assert rows[name]["metrics"]["roc_auc"] == pytest.approx(reference)


def test_calibration_raw_rows_match_shared_schema():
    val_probabilities, y_val, test_probabilities, y_test = _synthetic_probabilities()
    results = evaluate_calibration_strategies(
        val_probabilities, y_val, test_probabilities, y_test, test_probabilities
    )
    rows = []
    for result in results:
        strategy = result["calibration_strategy"]
        rows.append({
            **_base_row("adult_income", "agentdataset", "logistic_regression", 0, strategy),
            "calibration_strategy": strategy,
            "uses_real_test_labels": result["uses_real_test_labels"],
            "metric": "minority_f1",
            "value": result["metrics"].get("minority_f1", np.nan),
            "status": result["status"],
            "failure_stage": result["failure_stage"],
        })
    frame = pd.DataFrame(rows)

    validate_raw_results(frame)
    assert not set(RAW_RESULT_COLUMNS) - set(frame.columns)
    flagged = frame.loc[frame["uses_real_test_labels"] == 1, "calibration_strategy"].tolist()
    assert flagged == list(ORACLE_STRATEGIES)
    assert frame.loc[frame["synthesizer"] == "agentdataset", "condition"].eq("tstr").all()


def test_real_source_rows_are_labelled_trtr():
    row = _base_row("adult_income", "real", "logistic_regression", 3, "platt")
    assert row["condition"] == "trtr"
    assert row["extraction_method"] == "none"
    assert row["synthesis_seed"] is None
