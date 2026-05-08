import pytest
import pandas as pd
import numpy as np
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.models.schemas import Parameters, VariableParams, MetaParams

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
