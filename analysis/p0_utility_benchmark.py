"""P0 utility benchmark: nonlinear classification, regression, and intervals."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.evaluation import collect_provenance, summarize_long_results, write_json
from agentdataset.models.schemas import CorrelationParams, MetaParams, Parameters, VariableParams

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODEL = "openrouter/openai/gpt-4o"


def _load_concrete() -> dict:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=165)
    frame = dataset.data.features.join(dataset.data.targets)
    frame.columns = [re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_") for column in frame.columns]
    target = next(column for column in frame.columns if "compressive_strength" in column)
    features = [column for column in frame.columns if column != target]
    frame = frame[[*features, target]].apply(pd.to_numeric, errors="coerce").dropna()
    return {
        "name": "concrete_compressive_strength", "domain": "engineering",
        "task_type": "regression", "df": frame, "continuous": features,
        "categorical_features": [], "target": target, "target_units": "MPa",
    }


def _classification_datasets() -> list[dict]:
    # Keep the heavyweight optional SDV stack out of module import and unit tests.
    from benchmark import _load_adult, _load_german_credit, _load_pima

    datasets = [_load_adult(), _load_pima(), _load_german_credit()]
    positive = {
        "adult_income": ">50K", "pima_diabetes": "diabetes", "german_credit": "bad",
    }
    for dataset in datasets:
        dataset["task_type"] = "classification"
        dataset["positive_label"] = positive[dataset["name"]]
    return datasets


def _independent_marginals(params: Parameters, n_rows: int, seed: int) -> pd.DataFrame:
    independent = params.model_copy(deep=True)
    independent.correlations = {}
    return Synthesizer(n_rows=n_rows, seed=seed).synthesize(independent)


def _sdv_baseline(train: pd.DataFrame, features: list[str], target: str, n_rows: int, seed: int):
    from benchmark import synthesize_sdv_gaussian_copula

    return synthesize_sdv_gaussian_copula(train, features, target, n_rows, seed)


def parameters_from_training_data(dataset: dict, train: pd.DataFrame) -> Parameters:
    """Create gold summary parameters from a training split only."""
    columns = dataset["continuous"] + dataset.get("categorical_features", []) + [dataset["target"]]
    variables = {}
    for name in columns:
        if name in dataset.get("categorical_features", []) or dataset["task_type"] == "classification" and name == dataset["target"]:
            probabilities = train[name].value_counts(normalize=True).sort_index()
            variables[name] = VariableParams(
                name=name, distribution="categorical",
                categories={str(label): float(value) for label, value in probabilities.items()},
            )
        else:
            mean, std = float(train[name].mean()), float(train[name].std())
            variables[name] = VariableParams(
                name=name, distribution="normal", mean=mean, std=std,
                min=mean - 3 * std, max=mean + 3 * std,
            )

    numeric = train[columns].copy()
    for name in dataset.get("categorical_features", []):
        labels = list(variables[name].categories)
        numeric[name] = numeric[name].map({label: index for index, label in enumerate(labels)})
    if dataset["task_type"] == "classification":
        target = dataset["target"]
        # Correlations must use the exact category-code order consumed by the
        # synthesizer. Positive-class orientation is an evaluation concern and
        # is applied separately in _encode_classification.
        labels = list(variables[target].categories)
        numeric[target] = numeric[target].astype(str).map(
            {label: index for index, label in enumerate(labels)}
        )
    correlations = {}
    matrix = numeric.corr()
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            value = float(matrix.loc[left, right])
            if np.isfinite(value):
                key = f"{left}__{right}"
                correlations[key] = CorrelationParams(
                    var1=left, var2=right, correlation=value,
                    direction="positive" if value >= 0 else "negative",
                )
    return Parameters(
        variables=variables, correlations=correlations,
        meta=MetaParams(
            source=f"train_only_summary:{dataset['name']}",
            extracted_at=datetime.now(timezone.utc).isoformat(),
            extraction_method="gold_train_summary",
        ),
    )


def _encode_classification(dataset: dict, train: pd.DataFrame, *frames: pd.DataFrame):
    feature_columns = dataset["continuous"] + dataset["categorical_features"]
    encoder = None
    if dataset["categorical_features"]:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train[dataset["categorical_features"]].astype(str))

    def transform(frame: pd.DataFrame):
        features = frame[feature_columns].copy()
        if encoder is not None:
            features[dataset["categorical_features"]] = encoder.transform(
                features[dataset["categorical_features"]].astype(str)
            )
        target = (frame[dataset["target"]].astype(str) == dataset["positive_label"]).astype(int)
        return features.astype(float), target.to_numpy()

    return [transform(frame) for frame in frames]


def _classification_metrics(model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "minority_precision": precision_score(y_test, predictions, zero_division=0),
        "minority_recall": recall_score(y_test, predictions, zero_division=0),
        "minority_f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
    }


def _regression_metrics(model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": r2_score(y_test, predictions),
    }


def _add_metrics(rows: list[dict], base: dict, metrics: dict) -> None:
    for metric, value in metrics.items():
        rows.append({**base, "metric": metric, "value": float(value), "status": "ok", "failure_stage": None})


def run_utility_benchmark(n_seeds: int = 30, datasets: list[dict] | None = None) -> pd.DataFrame:
    if datasets is None:
        datasets = [*_classification_datasets(), _load_concrete()]
    rows = []
    for dataset in datasets:
        frame, target = dataset["df"], dataset["target"]
        feature_columns = dataset["continuous"] + dataset.get("categorical_features", [])
        stratify = frame[target] if dataset["task_type"] == "classification" else None
        for seed in range(n_seeds):
            train, test = train_test_split(frame, test_size=0.2, random_state=seed, stratify=stratify)
            params = parameters_from_training_data(dataset, train)
            orchestrator = Orchestrator(
                session_id=f"p0_{dataset['name']}_{seed}",
                base_dir=str(ROOT / ".agentdataset_cache" / "p0_utility"),
            )
            orchestrator.synthesizer = Synthesizer(n_rows=len(train), seed=seed)
            _, agent_frame = orchestrator.run_optimization_loop(params, iterations=3)
            synth_frames = {
                "agentdataset": agent_frame,
                "independent": _independent_marginals(params, len(train), seed),
                "sdv_gaussian_copula": _sdv_baseline(train, feature_columns, target, len(train), seed),
            }

            if dataset["task_type"] == "classification":
                model_factories = {
                    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
                    "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=seed),
                }
                encoded = _encode_classification(dataset, train, train, test)
                (x_real, y_real), (x_test, y_test) = encoded
                scorer = _classification_metrics
            else:
                model_factories = {
                    "linear_regression": LinearRegression,
                    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(random_state=seed),
                }
                x_real, y_real = train[feature_columns].astype(float), train[target].astype(float)
                x_test, y_test = test[feature_columns].astype(float), test[target].astype(float)
                scorer = _regression_metrics

            trtr_by_model = {}
            for model_name, factory in model_factories.items():
                trtr = scorer(factory(), x_real, y_real, x_test, y_test)
                trtr_by_model[model_name] = trtr
                base = {
                    "experiment_id": "p0_utility_v1", "run_id": f"{dataset['name']}:trtr:{model_name}:{seed}",
                    "experiment_type": "downstream_utility", "source_id": None,
                    "dataset": dataset["name"], "task_type": dataset["task_type"],
                    "condition": "trtr", "model_family": model_name, "synthesizer": "real",
                    "seed": seed, "split_seed": seed, "synthesis_seed": None,
                    "extraction_method": "none", "provider": "local", "model": None,
                }
                _add_metrics(rows, base, trtr)

            for synthesizer, synthetic in synth_frames.items():
                if synthetic is None:
                    continue
                try:
                    if dataset["task_type"] == "classification":
                        (x_synth, y_synth), = _encode_classification(dataset, train, synthetic)
                    else:
                        x_synth = synthetic[feature_columns].astype(float)
                        y_synth = synthetic[target].astype(float)
                    for model_name, factory in model_factories.items():
                        metrics = scorer(factory(), x_synth, y_synth, x_test, y_test)
                        base = {
                            "experiment_id": "p0_utility_v1", "run_id": f"{dataset['name']}:{synthesizer}:{model_name}:{seed}",
                            "experiment_type": "downstream_utility", "source_id": None,
                            "dataset": dataset["name"], "task_type": dataset["task_type"],
                            "condition": "tstr", "model_family": model_name, "synthesizer": synthesizer,
                            "seed": seed, "split_seed": seed, "synthesis_seed": seed,
                            "extraction_method": "gold_train_summary", "provider": "local", "model": None,
                        }
                        _add_metrics(rows, base, metrics)
                        delta = {name: value - trtr_by_model[model_name][name] for name, value in metrics.items()}
                        _add_metrics(rows, {**base, "run_id": base["run_id"] + ":delta", "condition": "tstr_minus_trtr"}, delta)
                except Exception as exc:
                    rows.append({
                        "experiment_id": "p0_utility_v1", "run_id": f"{dataset['name']}:{synthesizer}:failed:{seed}",
                        "experiment_type": "downstream_utility", "source_id": None,
                        "dataset": dataset["name"], "task_type": dataset["task_type"],
                        "condition": "tstr", "model_family": None, "synthesizer": synthesizer,
                        "seed": seed, "split_seed": seed, "synthesis_seed": seed,
                        "metric": "run_failure", "value": np.nan, "status": "failed",
                        "failure_stage": type(exc).__name__, "extraction_method": "gold_train_summary",
                        "provider": "local", "model": None,
                    })
    return pd.DataFrame(rows)


def main(n_seeds: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Captured before any output is written: recording provenance afterwards
    # makes git_dirty structurally always True, because this run's own CSVs are
    # in the working tree by then.
    provenance = collect_provenance(
        f"python analysis/p0_utility_benchmark.py --seeds {n_seeds}", range(n_seeds)
    )
    raw = run_utility_benchmark(n_seeds=n_seeds)
    raw.to_csv(RESULTS_DIR / "p0_utility_raw.csv", index=False)
    summary = summarize_long_results(
        raw, ["dataset", "task_type", "condition", "model_family", "synthesizer", "extraction_method"]
    )
    summary.to_csv(RESULTS_DIR / "p0_utility_summary.csv", index=False)
    write_json(RESULTS_DIR / "p0_utility_run_manifest.json", {
        **provenance,
        "primary_metrics": {
            "classification": ["minority_f1", "roc_auc"],
            "regression": ["rmse", "r2"],
        },
        "bootstrap_resamples": 10_000,
        "summary_parameter_scope": "training split only",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    main(args.seeds)
