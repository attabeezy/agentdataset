import pytest
import pandas as pd
import numpy as np
from agentdataset.core.validator import Validator
from agentdataset.models.schemas import Parameters, VariableParams, MetaParams

def test_validator_init():
    val = Validator(thresholds={"fidelity_score": 80.0})
    assert val.thresholds["fidelity_score"] == 80.0

def test_compute_ks_test():
    val = Validator()
    df = pd.DataFrame({"v1": np.random.normal(0, 1, 1000)})
    params = Parameters(
        variables={"v1": VariableParams(name="v1", mean=0, std=1)},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    p_values = val.compute_ks_test(df, params)
    assert p_values["v1"] > 0.05

def test_validate():
    val = Validator()
    df = pd.DataFrame({
        "v1": np.random.normal(0, 1, 1000),
        "v2": np.random.normal(10, 2, 1000)
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
