import json
import pytest
from unittest.mock import MagicMock, patch
from agentdataset.core.extractor import Extractor, _merge_chunk_results
from agentdataset.models.schemas import VariableParams, CorrelationParams


def test_extractor_init():
    ext = Extractor(model="test-model", api_key="sk-test")
    assert ext.model == "test-model"
    assert ext.api_key == "sk-test"


@patch('agentdataset.core.extractor.fitz')
def test_pdf_to_markdown(mock_fitz):
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "page text"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz.open.return_value = mock_doc

    ext = Extractor()
    md = ext.pdf_to_markdown("test.pdf")
    assert md == "page text"


@patch('agentdataset.core.extractor.completion')
def test_extract_parameters_llm_path(mock_completion):
    """LLM path: completion returns valid JSON, variables populated from it."""
    llm_json = {
        "variables": {
            "income": {"distribution": "normal", "mean": 50000.0, "std": 12000.0, "min": None, "max": None}
        },
        "correlations": {}
    }
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(llm_json)
    mock_completion.return_value = mock_response

    ext = Extractor(model="gpt-4o", api_key="sk-test")
    params = ext.extract_parameters("Some research text.", "test_source")

    assert params.meta.extraction_method == "llm"
    assert "income" in params.variables
    assert params.variables["income"].mean == 50000.0
    assert params.variables["income"].std == 12000.0
    assert params.meta.source == "test_source"


@patch('agentdataset.core.extractor.completion')
def test_extract_parameters_llm_fallback(mock_completion):
    """LLM failure (all chunks) raises RuntimeError; no regex fallback."""
    mock_completion.side_effect = Exception("API error")

    ext = Extractor(model="gpt-4o", api_key="sk-test")
    text = "The mean is 10.5 and the standard deviation is 2.1."
    with pytest.raises(RuntimeError):
        ext.extract_parameters(text, "test_source")



@patch('agentdataset.core.extractor.completion')
def test_extract_parameters_malformed_json_falls_back(mock_completion):
    """LLM returns non-JSON text → all chunks fail → RuntimeError."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Here is the data, not JSON at all."
    mock_completion.return_value = mock_response

    ext = Extractor(model="gpt-4o", api_key="sk-test")
    text = "The mean is 10.5 and the standard deviation is 2.1."
    with pytest.raises(RuntimeError):
        ext.extract_parameters(text, "src")


def test_parse_llm_result_wellformed():
    ext = Extractor()
    data = {
        "variables": {"x": {"distribution": "normal", "mean": 5.0, "std": 2.0}},
        "correlations": {"x__y": {"var1": "x", "var2": "y", "correlation": 0.5}},
    }
    variables, correlations = ext._parse_llm_result(data)
    assert variables["x"].mean == 5.0
    assert variables["x"].min == 5.0 - 3 * 2.0  # default min = mean - 3*std
    assert correlations["x__y"].correlation == 0.5


def test_parse_llm_result_missing_and_bad_fields():
    """Missing variables key, missing mean/std, and clamped correlation are handled."""
    ext = Extractor()
    data = {
        "variables": {"v": {"distribution": "weird"}},  # no mean/std, unknown dist
        "correlations": {"c": {"var1": "a", "var2": "b", "correlation": 5.0}},  # out of range
    }
    variables, correlations = ext._parse_llm_result(data)
    assert variables["v"].mean == 0.0 and variables["v"].std == 1.0
    assert variables["v"].distribution == "normal"      # coerced from "weird"
    assert correlations["c"].correlation == 1.0          # clamped from 5.0

    # Completely empty dict → no variables, no crash
    variables, correlations = ext._parse_llm_result({})
    assert variables == {} and correlations == {}






def test_parse_llm_result_categorical():
    ext = Extractor()
    data = {
        "variables": {
            "sex": {"distribution": "categorical", "categories": {"Female": 0.4, "Male": 0.6}}
        },
        "correlations": {},
    }
    variables, _ = ext._parse_llm_result(data)
    assert variables["sex"].distribution == "categorical"
    assert variables["sex"].categories == {"Female": 0.4, "Male": 0.6}



@patch('agentdataset.core.extractor.completion')
def test_extracted_at_is_iso_format(mock_completion):
    import json as _json
    from datetime import datetime
    from unittest.mock import MagicMock
    llm_json = {"variables": {"v": {"distribution": "normal", "mean": 1.0, "std": 1.0}}, "correlations": {}}
    mock_response = MagicMock()
    mock_response.choices[0].message.content = _json.dumps(llm_json)
    mock_completion.return_value = mock_response
    ext = Extractor(api_key="sk-test")
    params = ext.extract_parameters("mean = 1.0 std = 1.0", "src")
    datetime.fromisoformat(params.meta.extracted_at)


def test_parse_llm_result_tolerates_explicit_null_numbers():
    ext = Extractor()
    variables, correlations = ext._parse_llm_result({
        "variables": {
            "group": {
                "distribution": "categorical", "mean": None, "std": None,
                "categories": {"a": 0.5, "bad": None, "b": 0.5},
            }
        },
        "correlations": {
            "x__y": {"var1": "x", "var2": "y", "correlation": None}
        },
    })

    assert variables["group"].categories == {"a": 0.5, "b": 0.5}
    assert correlations["x__y"].correlation == 0.0


# --- §9.7 fix: regex variable naming ------------------------------------------








def test_normalize_variable_name():
    from agentdataset.core.extractor import _normalize_variable_name

    assert _normalize_variable_name("Heart Rate") == "heart_rate"
    assert _normalize_variable_name("education_num") == "education_num"
    assert _normalize_variable_name("  follow-up  ") == "follow_up"
    assert _normalize_variable_name("BMI") == "bmi"



def _llm_payload(variables, correlations):
    return {"variables": variables, "correlations": correlations}


def test_llm_exact_names_pass_through_unchanged():
    """The guarantee that makes this change safe: exact matches are untouched.

    If this breaks, previously reported extraction numbers could move.
    """
    ext = Extractor()
    data = _llm_payload(
        {
            "heart_rate": {"distribution": "normal", "mean": 72.0, "std": 8.0},
            "blood_pressure": {"distribution": "normal", "mean": 120.0, "std": 15.0},
        },
        {"c1": {"var1": "heart_rate", "var2": "blood_pressure", "correlation": 0.35}},
    )
    _, correlations = ext._parse_llm_result(data)
    corr = correlations["c1"]
    assert corr.var1 == "heart_rate"
    assert corr.var2 == "blood_pressure"


def test_llm_correlation_name_resolves_by_normalization():
    """A response that spells the variable differently in its correlations."""
    ext = Extractor()
    data = _llm_payload(
        {
            "heart_rate": {"distribution": "normal", "mean": 72.0, "std": 8.0},
            "blood_pressure": {"distribution": "normal", "mean": 120.0, "std": 15.0},
        },
        {"c1": {"var1": "Heart Rate", "var2": "blood pressure", "correlation": 0.35}},
    )
    variables, correlations = ext._parse_llm_result(data)
    corr = correlations["c1"]
    assert corr.var1 == "heart_rate"
    assert corr.var2 == "blood_pressure"
    assert corr.var1 in variables and corr.var2 in variables


def test_llm_ambiguous_name_is_left_unresolved():
    """Two variables differing only by case are ambiguous; guessing would merge them."""
    ext = Extractor()
    data = _llm_payload(
        {
            "BMI": {"distribution": "normal", "mean": 27.0, "std": 5.0},
            "bmi": {"distribution": "normal", "mean": 31.0, "std": 6.0},
            "age": {"distribution": "normal", "mean": 40.0, "std": 10.0},
        },
        {"c1": {"var1": "b m i", "var2": "age", "correlation": 0.2}},
    )
    variables, correlations = ext._parse_llm_result(data)
    corr = correlations["c1"]
    # Left exactly as the model wrote it, so the synthesizer counts it as skipped
    # rather than silently attaching it to whichever variable sorted first.
    assert corr.var1 == "b m i"
    assert corr.var1 not in variables


def test_llm_unmatched_variable_is_preserved_not_dropped():
    """A correlation naming an unreported variable stays visible in diagnostics."""
    ext = Extractor()
    data = _llm_payload(
        {"age": {"distribution": "normal", "mean": 40.0, "std": 10.0}},
        {"c1": {"var1": "age", "var2": "income", "correlation": 0.3}},
    )
    variables, correlations = ext._parse_llm_result(data)
    corr = correlations["c1"]
    assert corr.var2 == "income"
    assert corr.var2 not in variables


def test_llm_resolution_makes_correlation_apply_in_synthesis():
    """End to end: the resolved pair actually reaches the correlation matrix."""
    from agentdataset.core.synthesizer import Synthesizer
    from agentdataset.models.schemas import MetaParams, Parameters

    ext = Extractor()
    variables, correlations = ext._parse_llm_result(_llm_payload(
        {
            "heart_rate": {"distribution": "normal", "mean": 72.0, "std": 8.0},
            "blood_pressure": {"distribution": "normal", "mean": 120.0, "std": 15.0},
        },
        {"c1": {"var1": "Heart Rate", "var2": "Blood Pressure", "correlation": 0.6}},
    ))
    params = Parameters(
        variables=variables, correlations=correlations,
        meta=MetaParams(source="s", extracted_at="n"),
    )
    synth = Synthesizer(n_rows=2000, seed=0)
    synth.synthesize(params)
    assert synth.last_diagnostics.n_pairs_applied == 1
    assert synth.last_diagnostics.n_pairs_skipped_unknown_var == 0

def test_record_completion_falls_back_to_usage_cost():
    """Unmapped OpenRouter IDs raise in completion_cost; upstream usage.cost kept."""
    from unittest.mock import MagicMock, patch
    ext = Extractor(model="openrouter/google/gemini-3.8-flash", api_key="sk-test")
    response = MagicMock()
    response.usage.prompt_tokens = 400
    response.usage.completion_tokens = 150
    response.usage.total_tokens = 550
    response.usage.cost = 0.0009
    with patch("agentdataset.core.extractor.litellm.completion_cost", side_effect=Exception("not mapped")):
        ext._record_completion(response, started_at=0.0, purpose="test")
    assert ext.call_history[-1]["cost_usd"] == 0.0009


# ---------------------------------------------------------------------------
# _merge_chunk_results (chunk-merge dedup)
# ---------------------------------------------------------------------------

def _vp(name, mean=1.0, std=2.0):
    return VariableParams(name=name, distribution="normal", mean=mean, std=std)


def _cp(var1, var2, corr=0.5):
    return CorrelationParams(var1=var1, var2=var2, correlation=corr)


def test_merge_empty_returns_empty():
    assert _merge_chunk_results([]) == ({}, {})


def test_merge_single_chunk_variables_passthrough():
    vars_in = {"age": _vp("age", 30.0, 5.0)}
    corrs_in = {"c1": _cp("age", "income", 0.4)}
    merged_vars, merged_corrs = _merge_chunk_results([(vars_in, corrs_in)])
    assert "age" in merged_vars
    assert merged_vars["age"].mean == pytest.approx(30.0)
    # Correlation preserved even when only one chunk.
    assert len(merged_corrs) == 1


def test_merge_dedups_variables_by_normalized_name():
    r1 = ({"Age": _vp("Age", 30.0, 5.0)}, {})
    r2 = ({"age": _vp("age", 40.0, 7.0)}, {})
    merged_vars, _ = _merge_chunk_results([r1, r2])
    assert list(merged_vars.keys()) == ["Age"]
    assert merged_vars["Age"].mean == pytest.approx(35.0)
    assert merged_vars["Age"].std == pytest.approx(6.0)


def test_merge_dedups_variables_with_punctuation_differences():
    r1 = ({"heart rate": _vp("heart rate", 70.0, 8.0)}, {})
    r2 = ({"heart_rate": _vp("heart_rate", 72.0, 9.0)}, {})
    merged_vars, _ = _merge_chunk_results([r1, r2])
    assert len(merged_vars) == 1


def test_merge_dedups_correlations_by_pair():
    # Same pair reported under different chunk-local keys must collapse.
    r1 = ({}, {"c1": _cp("a", "b", 0.6)})
    r2 = ({}, {"c1": _cp("a", "b", 0.8)})
    _, merged_corrs = _merge_chunk_results([r1, r2])
    assert len(merged_corrs) == 1
    assert list(merged_corrs.values())[0].correlation == pytest.approx(0.7)


def test_merge_keeps_distinct_correlations():
    # Two different pairs that both happened to be "c1" in their chunks.
    r1 = ({}, {"c1": _cp("a", "b", 0.6)})
    r2 = ({}, {"c1": _cp("c", "d", 0.8)})
    _, merged_corrs = _merge_chunk_results([r1, r2])
    assert len(merged_corrs) == 2


def test_merge_correlation_pair_order_insensitive():
    r1 = ({}, {"c1": _cp("a", "b", 0.6)})
    r2 = ({}, {"c2": _cp("b", "a", 0.8)})
    _, merged_corrs = _merge_chunk_results([r1, r2])
    assert len(merged_corrs) == 1


def test_merge_skips_self_correlations():
    r1 = ({}, {"c1": _cp("a", "a", 1.0)})
    _, merged_corrs = _merge_chunk_results([r1])
    assert merged_corrs == {}


def test_merge_preserves_unmatched_correlation():
    r1 = ({"a": _vp("a")}, {"c1": _cp("a", "income", 0.3)})
    merged_vars, merged_corrs = _merge_chunk_results([r1])
    assert "a" in merged_vars
    cp = list(merged_corrs.values())[0]
    assert cp.var2 == "income"
    assert cp.var2 not in merged_vars


def test_extractor_last_chunk_results_populated():
    import json as _json
    from unittest.mock import MagicMock, patch
    llm_json = {
        "variables": {"income": {"distribution": "normal", "mean": 50000.0, "std": 12000.0}},
        "correlations": {},
    }
    mock_response = MagicMock()
    mock_response.choices[0].message.content = _json.dumps(llm_json)
    with patch("agentdataset.core.extractor.completion", return_value=mock_response):
        ext = Extractor(model="gpt-4o", api_key="sk-test")
        ext.extract_parameters("Some research text.", "test_source")
        assert isinstance(ext.last_chunk_results, list)
        assert len(ext.last_chunk_results) == 1
