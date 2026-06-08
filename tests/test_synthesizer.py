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
