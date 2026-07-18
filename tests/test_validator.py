import pytest
import pandas as pd
import numpy as np
from agentdataset.core.validator import Validator
from agentdataset.models.schemas import Parameters, VariableParams, CorrelationParams, MetaParams

def test_validator_init():
    val = Validator(thresholds={"fidelity_score": 80.0})
    assert val.thresholds["fidelity_score"] == 80.0

def test_compute_ks_test():
    val = Validator()
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"v1": rng.normal(0, 1, 1000)})
    params = Parameters(
        variables={"v1": VariableParams(name="v1", mean=0, std=1)},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    p_values = val.compute_ks_test(df, params)
    assert p_values["v1"] > 0.05

def test_validate():
    val = Validator()
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "v1": rng.normal(0, 1, 1000),
        "v2": rng.normal(10, 2, 1000)
    })
    params = Parameters(
        variables={
            "v1": VariableParams(name="v1", mean=0, std=1),
            "v2": VariableParams(name="v2", mean=10, std=2)
        },
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    report = val.validate(df, params)
    assert report.overall_score > 0
    assert report.approved in [True, False]
    assert "avg_min_dist" in report.privacy_details
    assert "privacy_score" in report.privacy_details


def test_correlation_similarity_rewards_match():
    """Off-diagonal metric: matching synthetic correlation scores higher than a mismatch."""
    val = Validator()
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 2000)
    # b strongly correlated with a; c independent of a
    b = a * 0.9 + rng.normal(0, 0.2, 2000)
    c = rng.normal(0, 1, 2000)
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    params_match = Parameters(
        variables={n: VariableParams(name=n) for n in ("a", "b", "c")},
        correlations={"ab": CorrelationParams(var1="a", var2="b", correlation=0.9)},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    params_wrong = Parameters(
        variables={n: VariableParams(name=n) for n in ("a", "b", "c")},
        correlations={"ac": CorrelationParams(var1="a", var2="c", correlation=0.9)},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    good = val.compute_correlation_similarity(df, params_match)
    bad = val.compute_correlation_similarity(df, params_wrong)
    assert good > bad
    assert 0.0 <= bad <= 1.0 and 0.0 <= good <= 1.0


def test_compute_ks_test_categorical_uses_chi_square():
    val = Validator()
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"sex": rng.choice(["Male", "Female"], size=1000, p=[0.6, 0.4])})
    params = Parameters(
        variables={"sex": VariableParams(name="sex", distribution="categorical", categories={"Male": 0.6, "Female": 0.4})},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    p_values = val.compute_ks_test(df, params)
    assert p_values["sex"] > 0.05


def test_compute_ks_test_categorical_mismatch_low_pvalue():
    val = Validator()
    df = pd.DataFrame({"sex": ["Male"] * 950 + ["Female"] * 50})
    params = Parameters(
        variables={"sex": VariableParams(name="sex", distribution="categorical", categories={"Male": 0.5, "Female": 0.5})},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    p_values = val.compute_ks_test(df, params)
    assert p_values["sex"] < 0.05


def test_correlation_similarity_with_categorical_column():
    """Categorical columns must be numerically encoded, not dropped, by the
    correlation-similarity computation."""
    val = Validator()
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 2000)
    cat = np.where(a > 0, "high", "low")
    df = pd.DataFrame({"a": a, "cat": cat})
    params = Parameters(
        variables={
            "a": VariableParams(name="a"),
            "cat": VariableParams(name="cat", distribution="categorical", categories={"high": 0.5, "low": 0.5}),
        },
        correlations={"ac": CorrelationParams(var1="a", var2="cat", correlation=0.9)},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    score = val.compute_correlation_similarity(df, params)
    assert 0.0 <= score <= 1.0


def test_validate_categorical_bias_score():
    val = Validator()
    df = pd.DataFrame({"sex": ["Male"] * 500 + ["Female"] * 500})
    params = Parameters(
        variables={"sex": VariableParams(name="sex", distribution="categorical", categories={"Male": 0.5, "Female": 0.5})},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    report = val.validate(df, params)
    assert report.bias_score == 100.0


def test_ks_gamma_non_positive_mean_no_crash():
    val = Validator()
    df = pd.DataFrame({"g": np.random.default_rng(1).normal(0, 1, 200)})
    params = Parameters(
        variables={"g": VariableParams(name="g", distribution="gamma", mean=-1.0, std=2.0)},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    pvals = val.compute_ks_test(df, params)  # must not raise
    assert "g" in pvals


def test_privacy_score_spread_vs_clustered():
    """Spread-out data should score higher than tightly clustered data."""
    val = Validator()
    spread = pd.DataFrame({"x": np.linspace(0, 100, 200), "y": np.linspace(0, 100, 200)})
    clustered = pd.DataFrame({"x": np.ones(200) * 50, "y": np.ones(200) * 50})
    spread_score = val.compute_privacy_score(spread)["privacy_score"]
    clustered_score = val.compute_privacy_score(clustered)["privacy_score"]
    assert spread_score > clustered_score
