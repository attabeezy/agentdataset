import pytest
import pandas as pd
import numpy as np
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.models.schemas import Parameters, VariableParams, CorrelationParams, MetaParams

def test_synthesizer_init():
    syn = Synthesizer(n_rows=500, seed=123)
    assert syn.n_rows == 500
    assert syn.seed == 123

def test_generate_variable():
    syn = Synthesizer(n_rows=100)
    params = VariableParams(name="v1", distribution="normal", mean=10.0, std=1.0)
    data = syn.generate_variable(params, noise_level=0.0)
    assert len(data) == 100
    assert abs(np.mean(data) - 10.0) < 0.5

def test_synthesize_no_vars():
    syn = Synthesizer()
    params = Parameters(variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N"))
    df = syn.synthesize(params)
    assert df.empty

def test_synthesize_with_vars():
    syn = Synthesizer(n_rows=100)
    v1 = VariableParams(name="v1", mean=0, std=1)
    v2 = VariableParams(name="v2", mean=10, std=2)
    params = Parameters(
        variables={"v1": v1, "v2": v2},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    df = syn.synthesize(params)
    assert df.shape == (100, 2)
    assert "v1" in df.columns
    assert "v2" in df.columns


def test_generate_variable_categorical():
    syn = Synthesizer(n_rows=1000)
    params = VariableParams(name="v1", distribution="categorical", categories={"yes": 0.7, "no": 0.3})
    data = syn.generate_variable(params, noise_level=0.0)
    assert len(data) == 1000
    # Integer codes (0/1), mapped to labels later in synthesize().
    assert set(np.unique(data)).issubset({0, 1})


def test_synthesize_categorical_variable_produces_labels():
    syn = Synthesizer(n_rows=1000, seed=0)
    params = Parameters(
        variables={
            "sex": VariableParams(name="sex", distribution="categorical", categories={"Female": 0.4, "Male": 0.6}),
        },
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    assert set(df["sex"].unique()) <= {"Female", "Male"}
    freqs = df["sex"].value_counts(normalize=True)
    assert abs(freqs["Male"] - 0.6) < 0.1


def test_synthesize_categorical_correlated_with_continuous():
    """A continuous variable strongly correlated with a binary categorical
    variable should show a real difference in means across categories."""
    syn = Synthesizer(n_rows=2000, seed=0)
    params = Parameters(
        variables={
            "score": VariableParams(name="score", distribution="normal", mean=0.0, std=1.0),
            "outcome": VariableParams(name="outcome", distribution="categorical", categories={"no": 0.5, "yes": 0.5}),
        },
        correlations={
            "c": CorrelationParams(var1="score", var2="outcome", correlation=0.8),
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    mean_yes = df.loc[df["outcome"] == "yes", "score"].mean()
    mean_no = df.loc[df["outcome"] == "no", "score"].mean()
    assert mean_yes > mean_no


def test_generate_variable_categorical_three_categories():
    syn = Synthesizer(n_rows=3000)
    params = VariableParams(
        name="v1", distribution="categorical", categories={"a": 0.5, "b": 0.3, "c": 0.2}
    )
    data = syn.generate_variable(params, noise_level=0.0)
    assert set(np.unique(data)).issubset({0, 1, 2})
    freqs = pd.Series(data).value_counts(normalize=True)
    assert abs(freqs[0] - 0.5) < 0.05
    assert abs(freqs[1] - 0.3) < 0.05
    assert abs(freqs[2] - 0.2) < 0.05


def test_synthesize_three_category_variable_produces_labels():
    syn = Synthesizer(n_rows=3000, seed=0)
    params = Parameters(
        variables={
            "status": VariableParams(
                name="status",
                distribution="categorical",
                categories={"married": 0.5, "never_married": 0.3, "prev_married": 0.2},
            ),
        },
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    assert set(df["status"].unique()) == {"married", "never_married", "prev_married"}
    freqs = df["status"].value_counts(normalize=True)
    assert abs(freqs["married"] - 0.5) < 0.05
    assert abs(freqs["never_married"] - 0.3) < 0.05
    assert abs(freqs["prev_married"] - 0.2) < 0.05


def test_synthesize_three_category_correlated_with_continuous():
    """The copula treats category codes as ordinal (insertion order of the
    categories dict), so a strong positive correlation should produce
    monotonically increasing group means across the three categories."""
    syn = Synthesizer(n_rows=3000, seed=0)
    params = Parameters(
        variables={
            "score": VariableParams(name="score", distribution="normal", mean=0.0, std=1.0),
            "level": VariableParams(
                name="level",
                distribution="categorical",
                categories={"low": 0.4, "mid": 0.3, "high": 0.3},
            ),
        },
        correlations={
            "c": CorrelationParams(var1="score", var2="level", correlation=0.8),
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    mean_low = df.loc[df["level"] == "low", "score"].mean()
    mean_mid = df.loc[df["level"] == "mid", "score"].mean()
    mean_high = df.loc[df["level"] == "high", "score"].mean()
    assert mean_low < mean_mid < mean_high


def test_gamma_with_non_positive_mean_does_not_crash():
    """Gamma needs mean>0; non-positive mean must fall back to normal, not crash."""
    syn = Synthesizer(n_rows=100)
    params = VariableParams(name="g", distribution="gamma", mean=-3.0, std=2.0)
    data = syn.generate_variable(params, noise_level=0.1)
    assert len(data) == 100
    assert np.all(np.isfinite(data))


def test_non_positive_definite_corr_warns_and_falls_back():
    """An impossible correlation matrix triggers a RuntimeWarning, not a crash."""
    syn = Synthesizer(n_rows=100)
    params = Parameters(
        variables={
            "a": VariableParams(name="a", mean=0, std=1),
            "b": VariableParams(name="b", mean=0, std=1),
            "c": VariableParams(name="c", mean=0, std=1),
        },
        # Mutually contradictory correlations → not positive-definite
        correlations={
            "ab": CorrelationParams(var1="a", var2="b", correlation=0.9),
            "ac": CorrelationParams(var1="a", var2="c", correlation=0.9),
            "bc": CorrelationParams(var1="b", var2="c", correlation=-0.9),
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )
    with pytest.warns(RuntimeWarning):
        df = syn.synthesize(params)
    assert df.shape == (100, 3)
