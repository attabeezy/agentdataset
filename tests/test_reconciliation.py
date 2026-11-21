from agentdataset.core.reconciliation import reconcile_parameters
from agentdataset.models.schemas import Parameters, VariableParams, CorrelationParams, MetaParams


def _params(source, vars_, corrs=None):
    return Parameters(
        variables={n: VariableParams(name=n, mean=v[0], std=v[1]) for n, v in vars_.items()},
        correlations={k: CorrelationParams(var1=c[0], var2=c[1], correlation=c[2]) for k, c in (corrs or {}).items()},
        meta=MetaParams(source=source, extracted_at="t", extraction_method="llm"),
    )


def test_reconcile_pools_compatible():
    p1 = _params("s1", {"age": (45.0, 3.0)}, {"age__income": ("age", "income", 0.6)})
    p2 = _params("s2", {"age": (46.0, 3.2)}, {"age__income": ("age", "income", 0.62)})
    merged, decs = reconcile_parameters([p1, p2])
    assert "age" in merged.variables
    assert merged.variables["age"].mean == (45.0 + 46.0) / 2
    assert any(d.method == "pooled_average" for d in decs)


def test_reconcile_keeps_separate_on_conflict():
    p1 = _params("s1", {"age": (45.0, 3.0)})
    p2 = _params("s2", {"age": (80.0, 3.0)})
    merged, decs = reconcile_parameters([p1, p2])
    # not pooled
    assert not any(d.method == "pooled_average" and d.target_variable == "age" for d in decs)
    assert any(d.method == "kept_separate" for d in decs)
    # per-source entries exist
    assert any("__src" in k for k in merged.variables.keys())