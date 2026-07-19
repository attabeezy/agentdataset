import pytest
from datetime import datetime
from agentdataset.models.schemas import (
    VariableParams, CorrelationParams, MetaParams, 
    Parameters, FidelityReport, SessionContext, DiscoveryResult
)

def test_variable_params():
    var = VariableParams(name="test_var", mean=10.0, std=2.0)
    assert var.name == "test_var"
    assert var.distribution == "normal"
    assert var.mean == 10.0
    assert var.std == 2.0
    assert var.min is None

def test_variable_params_categorical():
    var = VariableParams(name="sex", distribution="categorical", categories={"Male": 0.6, "Female": 0.4})
    assert var.distribution == "categorical"
    assert var.categories == {"Male": 0.6, "Female": 0.4}

def test_variable_params_categorical_probabilities_normalized():
    var = VariableParams(name="sex", distribution="categorical", categories={"Male": 3, "Female": 1})
    assert var.categories == {"Male": 0.75, "Female": 0.25}

def test_variable_params_three_categories_normalized():
    var = VariableParams(name="status", distribution="categorical", categories={"a": 5, "b": 3, "c": 2})
    assert var.categories == {"a": 0.5, "b": 0.3, "c": 0.2}

def test_variable_params_categorical_without_categories_demotes_to_normal():
    var = VariableParams(name="sex", distribution="categorical")
    assert var.distribution == "normal"

def test_correlation_params():
    corr = CorrelationParams(var1="v1", var2="v2", correlation=0.5)
    assert corr.var1 == "v1"
    assert corr.correlation == 0.5

def test_meta_params():
    meta = MetaParams(source="test_source", extracted_at="now")
    assert meta.source == "test_source"

def test_parameters():
    var = VariableParams(name="v1")
    meta = MetaParams(source="src", extracted_at="now")
    params = Parameters(
        variables={"v1": var},
        correlations={},
        meta=meta
    )
    assert "v1" in params.variables

def test_fidelity_report():
    report = FidelityReport(
        overall_score=95.0,
        ks_score=90.0,
        corr_score=90.0,
        bias_score=100.0,
        ks_pvalues={"v1": 0.5},
        bias_details={},
        privacy_details={"avg_min_dist": 0.1},
        approved=True
    )
    assert report.approved is True

def test_session_context():
    ctx = SessionContext(session_id="id1", path="/tmp/path")
    assert ctx.session_id == "id1"
    assert isinstance(ctx.created_at, datetime)

def test_discovery_result():
    res = DiscoveryResult(title="T", url="U", source_type="pdf", relevance_score=1.0)
    assert res.source_type == "pdf"
