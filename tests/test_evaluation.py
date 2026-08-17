import numpy as np
import pandas as pd
import pytest

from agentdataset.evaluation import (
    RAW_RESULT_COLUMNS,
    bootstrap_ci,
    summarize_long_results,
    validate_raw_results,
)
from agentdataset.models.schemas import MetaParams, Parameters, VariableParams
from analysis import p0_utility_benchmark as utility
from analysis.p0_utility_benchmark import _encode_classification, parameters_from_training_data
from analysis.real_paper_eval import flatten_parameters, score_extraction


def _raw_frame(values):
    rows = []
    for seed, value in enumerate(values):
        row = {column: None for column in RAW_RESULT_COLUMNS}
        row.update({
            "experiment_id": "e", "run_id": f"r{seed}", "experiment_type": "test",
            "dataset": "d", "task_type": "classification", "condition": "tstr",
            "model_family": "m", "synthesizer": "s", "seed": seed,
            "metric": "f1", "value": value, "status": "ok",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def test_bootstrap_and_summary_are_deterministic():
    assert bootstrap_ci([1, 2, 3], seed=7) == bootstrap_ci([1, 2, 3], seed=7)
    raw = _raw_frame([0.2, 0.4, 0.6])
    summary = summarize_long_results(raw, ["dataset", "condition"])
    assert summary.iloc[0]["n"] == 3
    assert summary.iloc[0]["mean"] == pytest.approx(0.4)


def test_raw_schema_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        validate_raw_results(pd.DataFrame({"run_id": ["x"]}))


def test_training_parameters_do_not_use_test_rows():
    train = pd.DataFrame({"x": [0.0, 2.0, 4.0], "target": ["no", "yes", "no"]})
    dataset = {
        "name": "toy", "task_type": "classification", "continuous": ["x"],
        "categorical_features": [], "target": "target", "positive_label": "yes",
    }
    params = parameters_from_training_data(dataset, train)
    assert params.variables["x"].mean == pytest.approx(2.0)
    assert params.variables["target"].categories == {"no": pytest.approx(2 / 3), "yes": pytest.approx(1 / 3)}


def test_positive_class_is_explicit_not_encoder_order():
    train = pd.DataFrame({"x": [1, 2, 3, 4], "target": ["disease", "healthy", "healthy", "disease"]})
    dataset = {
        "continuous": ["x"], "categorical_features": [], "target": "target",
        "positive_label": "disease",
    }
    (_, target), = _encode_classification(dataset, train, train)
    assert target.tolist() == [1, 0, 0, 1]


def test_target_correlation_uses_synthesizer_category_order():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": ["disease", "disease", "healthy", "healthy"]})
    dataset = {
        "name": "toy", "task_type": "classification", "continuous": ["x"],
        "categorical_features": [], "target": "target", "positive_label": "disease",
    }
    params = parameters_from_training_data(dataset, train)
    assert list(params.variables["target"].categories) == ["disease", "healthy"]
    assert params.correlations["x__target"].correlation > 0


def test_one_shot_copula_is_exact_first_loop_iteration(tmp_path):
    train = pd.DataFrame({
        "x": np.linspace(0.0, 9.0, 10),
        "y": [1.0, 2.5, 4.0, 3.5, 8.0, 7.0, 11.0, 10.0, 15.0, 17.0],
    })
    dataset = {
        "name": "toy", "task_type": "regression", "continuous": ["x"],
        "categorical_features": [], "target": "y",
    }
    params = parameters_from_training_data(dataset, train)
    expected = utility._one_shot_copula(params, len(train), seed=7)

    orchestrator = utility.Orchestrator(
        session_id="one_shot_equivalence", base_dir=str(tmp_path)
    )
    orchestrator.synthesizer = utility.Synthesizer(n_rows=len(train), seed=7)
    _, actual = orchestrator.run_optimization_loop(params, iterations=1)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_utility_arms_share_gold_summaries_and_paired_seeds(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "x": np.linspace(0.0, 49.0, 50),
        "target": np.linspace(0.0, 49.0, 50) ** 2 + 2.0,
    })
    dataset = {
        "name": "toy_regression", "task_type": "regression", "df": frame,
        "continuous": ["x"], "categorical_features": [], "target": "target",
    }
    monkeypatch.setattr(utility, "ROOT", tmp_path)
    monkeypatch.setattr(
        utility,
        "_sdv_baseline",
        lambda train, features, target, n_rows, seed: train.copy(),
    )

    raw = utility.run_utility_benchmark(n_seeds=1, datasets=[dataset])
    tstr = raw.loc[raw["condition"].eq("tstr")]

    assert set(tstr["synthesizer"]) == {
        "agentdataset", "copula_no_loop", "independent", "sdv_gaussian_copula",
    }
    assert set(tstr["extraction_method"]) == {"gold_train_summary"}
    assert set(tstr["provider"]) == {"local"}
    assert set(tstr["seed"]) == {0}
    assert np.array_equal(
        tstr["split_seed"].to_numpy(), tstr["synthesis_seed"].to_numpy()
    )


def test_baseline_contrasts_are_paired_with_declared_direction():
    rows = []
    values = {
        "independent": 0.10,
        "copula_no_loop": 0.40,
        "agentdataset": 0.55,
        "sdv_gaussian_copula": 0.70,
    }
    for synthesizer, value in values.items():
        rows.append({
            "dataset": "toy", "task_type": "classification",
            "condition": "tstr", "model_family": "lr", "seed": 3,
            "metric": "minority_f1", "synthesizer": synthesizer,
            "value": value, "status": "ok",
        })

    contrasts = utility.build_baseline_contrasts(pd.DataFrame(rows))
    observed = dict(zip(contrasts["synthesizer"], contrasts["value"]))

    assert observed == pytest.approx({
        "copula_no_loop_minus_independent": 0.30,
        "agentdataset_minus_copula_no_loop": 0.15,
        "sdv_gaussian_copula_minus_agentdataset": 0.15,
    })
    assert contrasts["run_id"].is_unique
    assert set(contrasts["extraction_method"]) == {"gold_train_summary"}


def test_extraction_scoring_separates_numeric_and_name_accuracy():
    params = Parameters(
        variables={"var_1": VariableParams(name="var_1", mean=10.0, std=2.0)},
        correlations={},
        meta=MetaParams(source="s", extracted_at="now", extraction_method="regex_fallback"),
    )
    gold = pd.DataFrame([
        {"gold_id": "g1", "source_id": "s", "variable_name": "age", "source_label": "Age", "aliases": "age", "statistic_type": "mean", "value": 10.0, "decimals": 1},
        {"gold_id": "g2", "source_id": "s", "variable_name": "age", "source_label": "Age", "aliases": "age", "statistic_type": "std", "value": 2.0, "decimals": 1},
    ])
    metrics, errors = score_extraction("s", params, gold)
    assert metrics["f1"] == 1.0
    assert metrics["numeric_accuracy"] == 1.0
    assert metrics["variable_name_accuracy"] == 0.0
    assert set(errors.error_category) == {"variable"}


def test_flatten_parameters_emits_mean_and_std():
    params = Parameters(
        variables={"age": VariableParams(name="age", mean=10, std=2)},
        correlations={}, meta=MetaParams(source="s", extracted_at="now"),
    )
    flattened = flatten_parameters(params)
    assert set(flattened.statistic_type) == {"mean", "std"}
