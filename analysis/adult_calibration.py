"""P1 Adult calibration study: how much of the minority-F1 gap is a threshold artifact?

The 30-seed utility benchmark (analysis/p0_utility_benchmark.py) shows that on Adult a
logistic regression trained on AgentDataset output matches TRTR ranking quality
(ROC-AUC .850 vs .850) while losing minority-class F1 (.381 vs .542) — with *higher*
precision and collapsed recall. That is the signature of a decision-threshold artifact
rather than a discrimination deficit. Histogram gradient boosting, by contrast, loses
.025 AUC, which no threshold can recover.

This module separates the two. Every calibration decision is made on a validation split
carved out of the *synthetic* training frame; the real held-out test fold is only ever
scored against, never fit or tuned on. The one exception is the explicitly labelled
`oracle_threshold_real_test` diagnostic, which reports the ceiling that thresholding
alone could reach and carries uses_real_test_labels=1 so it can never be mistaken for a
usable method.

Outputs (P0 raw/summary/manifest convention):
  results/adult_calibration_raw.csv           seed-level long format
  results/adult_calibration.csv               bootstrap summary across seeds
  results/adult_calibration_run_manifest.json provenance

Runs fully locally: the optimization loop only calls the synthesizer and validator, so
no LLM API key is required (unlike analysis/adult_f1_gap.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.evaluation import collect_provenance, summarize_long_results, write_json
from analysis.p0_utility_benchmark import (
    _encode_classification,
    _independent_marginals,
    _sdv_baseline,
    parameters_from_training_data,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
EXPERIMENT_ID = "adult_calibration_v1"

# Isotonic regression is nonparametric and overfits small validation splits badly, so it
# is skipped rather than reported when the split cannot support it. Adult clears both
# bounds comfortably; the guard exists for reuse on smaller datasets.
MIN_ISOTONIC_VALIDATION_ROWS = 500
MIN_ISOTONIC_CLASS_COUNT = 50

VALIDATION_FRACTION = 0.2
DEFAULT_THRESHOLD = 0.5

ORACLE_STRATEGIES = ("oracle_threshold_real_test",)
STRATEGIES = (
    "uncalibrated",
    "uncalibrated_full_fit",
    "threshold_synth_val",
    "platt",
    "platt_threshold",
    "isotonic",
    "isotonic_threshold",
    "oracle_threshold_real_test",
)

_EPSILON = 1e-12


# --------------------------------------------------------------------------------------
# Calibration primitives.
#
# These take validation arrays only. They have no parameter through which real test data
# could be passed, which is what tests/test_adult_calibration.py asserts on: leakage is
# prevented by the signature, not by discipline at the call site.
# --------------------------------------------------------------------------------------


def _select_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the probability threshold maximising minority-class F1 on `y_true`."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD
    # precision_recall_curve returns one more point than thresholds; the trailing point
    # is the degenerate (recall=0) end and has no corresponding threshold.
    precision, recall = precision[:-1], recall[:-1]
    scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(np.argmax(scores))])


def _fit_calibrator(
    method: str, y_true: np.ndarray, probabilities: np.ndarray
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Fit a probability calibrator on validation data.

    Returns None when the validation split cannot support the method, so the caller can
    record an explicit `skipped` row instead of a fabricated value.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None

    if method == "platt":
        # Classical Platt scaling: a sigmoid fitted on the model's log-odds. Working in
        # logit space keeps this equivalent for models whose scores are not already
        # probabilities (e.g. gradient boosting).
        clipped = np.clip(probabilities, _EPSILON, 1 - _EPSILON)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        sigmoid = LogisticRegression(max_iter=1000).fit(logits, y_true)

        def transform(values: np.ndarray) -> np.ndarray:
            values = np.clip(np.asarray(values, dtype=float), _EPSILON, 1 - _EPSILON)
            return sigmoid.predict_proba(
                np.log(values / (1 - values)).reshape(-1, 1)
            )[:, 1]

        return transform

    if method == "isotonic":
        counts = np.bincount(y_true.astype(int), minlength=2)
        if len(y_true) < MIN_ISOTONIC_VALIDATION_ROWS or counts.min() < MIN_ISOTONIC_CLASS_COUNT:
            return None
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(probabilities, y_true)

        def transform(values: np.ndarray) -> np.ndarray:
            return np.asarray(isotonic.predict(np.asarray(values, dtype=float)), dtype=float)

        return transform

    raise ValueError(f"Unknown calibration method: {method}")


def _isotonic_skip_reason(y_true: np.ndarray) -> str:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return "degenerate_val_split"
    return "insufficient_validation_sample"


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def _threshold_metrics(
    y_test: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Score calibrated probabilities against the real held-out test fold."""
    y_test = np.asarray(y_test)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    single_class = len(np.unique(y_test)) < 2
    return {
        "minority_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "minority_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "minority_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "pred_pos_rate": float(predictions.mean()),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_test, probabilities)),
        "decision_threshold": float(threshold),
    }


def evaluate_calibration_strategies(
    val_probabilities: np.ndarray,
    y_val: np.ndarray,
    test_probabilities: np.ndarray,
    y_test: np.ndarray,
    full_fit_test_probabilities: np.ndarray | None = None,
) -> list[dict]:
    """Apply every calibration strategy and score each on the real test fold.

    Every strategy except `oracle_threshold_real_test` derives its calibrator and its
    threshold from (`val_probabilities`, `y_val`) alone. `y_test` is used only to score.
    """
    y_val = np.asarray(y_val)
    val_probabilities = np.asarray(val_probabilities, dtype=float)
    test_probabilities = np.asarray(test_probabilities, dtype=float)
    rows: list[dict] = []

    def ok(strategy: str, metrics: dict[str, float], oracle: bool = False) -> None:
        rows.append({
            "calibration_strategy": strategy,
            "uses_real_test_labels": int(oracle),
            "status": "ok",
            "failure_stage": None,
            "metrics": metrics,
        })

    def skip(strategy: str, reason: str) -> None:
        rows.append({
            "calibration_strategy": strategy,
            "uses_real_test_labels": 0,
            "status": "skipped",
            "failure_stage": reason,
            "metrics": {},
        })

    ok("uncalibrated", _threshold_metrics(y_test, test_probabilities, DEFAULT_THRESHOLD))

    if full_fit_test_probabilities is not None:
        ok(
            "uncalibrated_full_fit",
            _threshold_metrics(
                y_test, np.asarray(full_fit_test_probabilities, dtype=float), DEFAULT_THRESHOLD
            ),
        )

    degenerate_validation = len(np.unique(y_val)) < 2

    if degenerate_validation:
        skip("threshold_synth_val", "degenerate_val_split")
    else:
        threshold = _select_f1_threshold(y_val, val_probabilities)
        ok("threshold_synth_val", _threshold_metrics(y_test, test_probabilities, threshold))

    for method in ("platt", "isotonic"):
        calibrator = _fit_calibrator(method, y_val, val_probabilities)
        if calibrator is None:
            reason = (
                "degenerate_val_split" if degenerate_validation else _isotonic_skip_reason(y_val)
            )
            skip(method, reason)
            skip(f"{method}_threshold", reason)
            continue
        calibrated_test = calibrator(test_probabilities)
        ok(method, _threshold_metrics(y_test, calibrated_test, DEFAULT_THRESHOLD))
        calibrated_val = calibrator(val_probabilities)
        threshold = _select_f1_threshold(y_val, calibrated_val)
        ok(f"{method}_threshold", _threshold_metrics(y_test, calibrated_test, threshold))

    # Diagnostic upper bound only. Tuned directly on the real test labels, so it is not a
    # usable method: it bounds what any thresholding rule could achieve, which makes the
    # residual gap beneath it structural rather than a threshold artifact.
    oracle_threshold = _select_f1_threshold(y_test, test_probabilities)
    ok(
        "oracle_threshold_real_test",
        _threshold_metrics(y_test, test_probabilities, oracle_threshold),
        oracle=True,
    )

    return rows


# --------------------------------------------------------------------------------------
# Study driver
# --------------------------------------------------------------------------------------


def _adult_dataset() -> dict:
    # Keep the heavyweight optional SDV stack out of module import and unit tests.
    from benchmark import _load_adult

    dataset = _load_adult()
    dataset["task_type"] = "classification"
    dataset["positive_label"] = ">50K"
    return dataset


def _split_fit_validation(frame: pd.DataFrame, target: str, seed: int):
    """Carve a validation split out of a training frame, stratified where possible."""
    try:
        return train_test_split(
            frame, test_size=VALIDATION_FRACTION, random_state=seed, stratify=frame[target]
        )
    except ValueError:
        # Too few members of some class to stratify; an unstratified split still yields a
        # usable (possibly degenerate) validation fold, which downstream code handles.
        return train_test_split(frame, test_size=VALIDATION_FRACTION, random_state=seed)


def _model_factories(seed: int) -> dict[str, Callable]:
    return {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=seed),
    }


def _base_row(dataset_name: str, synthesizer: str, model_name: str, seed: int, strategy: str) -> dict:
    is_real = synthesizer == "real"
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{dataset_name}:{synthesizer}:{model_name}:{strategy}:{seed}",
        "experiment_type": "calibration",
        "source_id": None,
        "dataset": dataset_name,
        "task_type": "classification",
        "condition": "trtr" if is_real else "tstr",
        "model_family": model_name,
        "synthesizer": synthesizer,
        "seed": seed,
        "split_seed": seed,
        "synthesis_seed": None if is_real else seed,
        "extraction_method": "none" if is_real else "gold_train_summary",
        "provider": "local",
        "model": None,
    }


def run_calibration_study(n_seeds: int = 30, dataset: dict | None = None) -> pd.DataFrame:
    """Run the seed loop and return seed-level rows in the shared raw schema."""
    if dataset is None:
        dataset = _adult_dataset()
    name, target = dataset["name"], dataset["target"]
    feature_columns = dataset["continuous"] + dataset.get("categorical_features", [])
    rows: list[dict] = []

    for seed in range(n_seeds):
        # Identical split to analysis/p0_utility_benchmark.py so the two studies pair.
        train, test = train_test_split(
            dataset["df"], test_size=0.2, random_state=seed, stratify=dataset["df"][target]
        )
        params = parameters_from_training_data(dataset, train)
        orchestrator = Orchestrator(
            session_id=f"adultcal_{name}_{seed}",
            base_dir=str(ROOT / ".agentdataset_cache" / "adult_calibration"),
        )
        orchestrator.synthesizer = Synthesizer(n_rows=len(train), seed=seed)
        _, agent_frame = orchestrator.run_optimization_loop(params, iterations=3)

        source_frames = {
            "real": train,
            "agentdataset": agent_frame,
            "independent": _independent_marginals(params, len(train), seed),
            "sdv_gaussian_copula": _sdv_baseline(train, feature_columns, target, len(train), seed),
        }

        # Keyed by (model_family, strategy, metric) so TSTR deltas pair exactly.
        trtr_values: dict[tuple[str, str, str], float] = {}
        pending_deltas: list[tuple[dict, dict[str, float]]] = []

        for synthesizer, frame in source_frames.items():
            if frame is None:
                rows.append({
                    **_base_row(name, synthesizer, None, seed, "none"),
                    "metric": "run_failure",
                    "value": np.nan,
                    "status": "failed",
                    "failure_stage": "synthesis_unavailable",
                    "calibration_strategy": None,
                    "uses_real_test_labels": 0,
                })
                continue
            try:
                fit_frame, val_frame = _split_fit_validation(frame, target, seed)
                # The ordinal encoder is always fitted on the real training frame, as in
                # the P0 benchmark, so encodings are comparable across synthesizers.
                (x_fit, y_fit), (x_val, y_val), (x_test, y_test) = _encode_classification(
                    dataset, train, fit_frame, val_frame, test
                )
                (x_full, y_full), = _encode_classification(dataset, train, frame)

                for model_name, factory in _model_factories(seed).items():
                    model = factory().fit(x_fit, y_fit)
                    val_probabilities = model.predict_proba(x_val)[:, 1]
                    test_probabilities = model.predict_proba(x_test)[:, 1]
                    full_model = factory().fit(x_full, y_full)
                    full_probabilities = full_model.predict_proba(x_test)[:, 1]

                    for result in evaluate_calibration_strategies(
                        val_probabilities, y_val, test_probabilities, y_test, full_probabilities
                    ):
                        strategy = result["calibration_strategy"]
                        base = {
                            **_base_row(name, synthesizer, model_name, seed, strategy),
                            "calibration_strategy": strategy,
                            "uses_real_test_labels": result["uses_real_test_labels"],
                        }
                        if result["status"] != "ok":
                            rows.append({
                                **base,
                                "metric": "run_skipped",
                                "value": np.nan,
                                "status": result["status"],
                                "failure_stage": result["failure_stage"],
                            })
                            continue
                        for metric, value in result["metrics"].items():
                            rows.append({
                                **base, "metric": metric, "value": float(value),
                                "status": "ok", "failure_stage": None,
                            })
                            if synthesizer == "real":
                                trtr_values[(model_name, strategy, metric)] = float(value)
                        if synthesizer != "real":
                            pending_deltas.append((base, result["metrics"]))
            except Exception as exc:  # noqa: BLE001 - failures stay visible as rows
                rows.append({
                    **_base_row(name, synthesizer, None, seed, "none"),
                    "metric": "run_failure",
                    "value": np.nan,
                    "status": "failed",
                    "failure_stage": type(exc).__name__,
                    "calibration_strategy": None,
                    "uses_real_test_labels": 0,
                })

        # Paired TSTR - TRTR deltas, matched within (model_family, strategy, metric).
        for base, metrics in pending_deltas:
            key_prefix = (base["model_family"], base["calibration_strategy"])
            for metric, value in metrics.items():
                reference = trtr_values.get((*key_prefix, metric))
                if reference is None:
                    continue
                rows.append({
                    **base,
                    "run_id": base["run_id"] + ":delta",
                    "condition": "tstr_minus_trtr",
                    "metric": metric,
                    "value": float(value) - reference,
                    "status": "ok",
                    "failure_stage": None,
                })

    return pd.DataFrame(rows)


def main(n_seeds: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = run_calibration_study(n_seeds=n_seeds)
    raw.to_csv(RESULTS_DIR / "adult_calibration_raw.csv", index=False)
    summary = summarize_long_results(
        raw,
        [
            "dataset", "task_type", "condition", "model_family", "synthesizer",
            "calibration_strategy", "extraction_method", "uses_real_test_labels",
        ],
    )
    summary.to_csv(RESULTS_DIR / "adult_calibration.csv", index=False)
    write_json(RESULTS_DIR / "adult_calibration_run_manifest.json", {
        **collect_provenance(f"python analysis/adult_calibration.py --seeds {n_seeds}", range(n_seeds)),
        "primary_metrics": {"classification": ["minority_f1", "roc_auc", "pred_pos_rate"]},
        "bootstrap_resamples": 10_000,
        "summary_parameter_scope": "training split only",
        "calibration_fit_scope": "synthetic validation split only",
        "validation_fraction": VALIDATION_FRACTION,
        "isotonic_guard": {
            "min_validation_rows": MIN_ISOTONIC_VALIDATION_ROWS,
            "min_class_count": MIN_ISOTONIC_CLASS_COUNT,
        },
        "oracle_strategies": list(ORACLE_STRATEGIES),
        "strategies": list(STRATEGIES),
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    main(args.seeds)
