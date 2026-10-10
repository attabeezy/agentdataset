import json
import pytest
from unittest.mock import MagicMock, patch
from agentdataset.core.extractor import Extractor


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
    """LLM failure falls back to regex and labels method correctly."""
    mock_completion.side_effect = Exception("API error")

    ext = Extractor(model="gpt-4o", api_key="sk-test")
    text = "The mean is 10.5 and the standard deviation is 2.1."
    params = ext.extract_parameters(text, "test_source")

    assert params.meta.extraction_method == "regex_fallback"
    assert len(params.variables) == 1
    var = list(params.variables.values())[0]
    assert var.mean == 10.5
    assert var.std == 2.1


def test_extract_parameters_regex_no_key():
    """No API key → skips LLM entirely, uses regex."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)

    ext = Extractor()  # no api_key
    text = "The mean is 10.5 and the standard deviation is 2.1."
    params = ext.extract_parameters(text, "test_source")

    assert params.meta.extraction_method == "regex_fallback"
    assert len(params.variables) == 1
    var = params.variables["var_1"]
    assert var.mean == 10.5
    assert var.std == 2.1
    assert params.meta.source == "test_source"


@patch('agentdataset.core.extractor.completion')
def test_extract_parameters_malformed_json_falls_back(mock_completion):
    """LLM returns non-JSON text → real json.loads raises → regex fallback runs."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Here is the data, not JSON at all."
    mock_completion.return_value = mock_response

    ext = Extractor(model="gpt-4o", api_key="sk-test")
    text = "The mean is 10.5 and the standard deviation is 2.1."
    params = ext.extract_parameters(text, "src")

    # json.loads failed inside _extract_with_llm; method must report the fallback.
    assert params.meta.extraction_method == "regex_fallback"
    assert params.variables["var_1"].mean == 10.5


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


def test_regex_negative_and_scientific_numbers():
    ext = Extractor()
    text = "mean = -5.3 and std = 1.2e-1 in the sample."
    variables, _ = ext._extract_with_regex(text)
    var = variables["var_1"]
    assert var.mean == -5.3
    assert var.std == 0.12


def test_regex_extracts_categorical():
    ext = Extractor()
    text = "The categorical variable sex takes value 'Female' with probability 0.4 and 'Male' with probability 0.6."
    variables, _ = ext._extract_with_regex(text)
    assert "sex" in variables
    var = variables["sex"]
    assert var.distribution == "categorical"
    assert var.categories == {"Female": 0.4, "Male": 0.6}


def test_regex_extracts_three_category_variable():
    ext = Extractor()
    text = (
        "The categorical variable marital_status takes value 'married' with probability 0.5, "
        "'never_married' with probability 0.3, and 'prev_married' with probability 0.2."
    )
    variables, _ = ext._extract_with_regex(text)
    assert "marital_status" in variables
    var = variables["marital_status"]
    assert var.distribution == "categorical"
    assert var.categories == {"married": 0.5, "never_married": 0.3, "prev_married": 0.2}


def test_regex_ignores_single_category_fragment():
    """A lone label/probability pair would normalize to probability 1.0 — it is
    treated as a parse fragment, not a categorical variable."""
    ext = Extractor()
    text = "The categorical variable status takes value 'active' with probability 0.9."
    variables, _ = ext._extract_with_regex(text)
    assert "status" not in variables


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


def test_regex_extracts_correlation():
    ext = Extractor()
    text = "The correlation between income and age is 0.65 overall."
    _, correlations = ext._extract_with_regex(text)
    assert len(correlations) == 1
    c = list(correlations.values())[0]
    assert c.var1 == "income" and c.var2 == "age"
    assert c.correlation == 0.65 and c.direction == "positive"


def test_extracted_at_is_iso_format():
    from datetime import datetime
    ext = Extractor()
    params = ext.extract_parameters("mean = 1.0 std = 1.0", "src")
    # Should parse as ISO 8601 without raising.
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


def test_regex_names_variable_from_variable_clause():
    """"The variable X has mean ... and std ..." -- the UCI prose form."""
    ext = Extractor()
    text = "The variable education_num has mean 10.08 and std 2.57."
    variables, _ = ext._extract_with_regex(text)
    assert "education_num" in variables
    assert variables["education_num"].mean == 10.08
    assert variables["education_num"].std == 2.57


def test_regex_names_variable_from_distribution_clause():
    """Multi-word names in prose normalize to the underscored form."""
    ext = Extractor()
    text = (
        "We observed a heart rate distribution with mean 72.0 and std 8.0.\n"
        "Blood pressure was normally distributed with mean 120.0 and std 15.0."
    )
    variables, _ = ext._extract_with_regex(text)
    assert set(variables) == {"heart_rate", "blood_pressure"}
    assert variables["heart_rate"].mean == 72.0
    assert variables["blood_pressure"].std == 15.0


def test_regex_correlation_resolves_against_named_variables():
    """The whole point of the fix: the declared pair must reference real variables."""
    ext = Extractor()
    text = (
        "We observed a heart rate distribution with mean 72.0 and std 8.0.\n"
        "Blood pressure was normally distributed with mean 120.0 and std 15.0.\n"
        "The correlation between heart_rate and blood_pressure is 0.35."
    )
    variables, correlations = ext._extract_with_regex(text)
    corr = list(correlations.values())[0]
    assert corr.var1 in variables and corr.var2 in variables


def test_regex_does_not_capture_articles_or_filler_as_names():
    """A wrong name mismatches silently, so the capture is deliberately narrow."""
    ext = Extractor()
    variables, _ = ext._extract_with_regex(
        "In the sample, the mean is 10.5 and the standard deviation is 2.1."
    )
    assert list(variables) == ["var_1"]
    assert not any(name in ("the", "a", "sample", "in_the_sample") for name in variables)


def test_regex_falls_back_when_name_already_taken():
    """Two variables must never collapse into one because they share a name."""
    ext = Extractor()
    text = (
        "The variable age has mean 38.0 and std 13.0.\n"
        "The variable age has mean 51.0 and std 9.0."
    )
    variables, _ = ext._extract_with_regex(text)
    assert len(variables) == 2, "distinct statistics must stay distinct"
    assert "age" in variables
    assert any(name.startswith("var_") for name in variables)


def test_regex_name_capture_does_not_reach_across_sentences():
    """A name must be adjacent to its statistic, not merely somewhere earlier."""
    ext = Extractor()
    text = (
        "The variable age was collected for every participant. "
        "Some other text intervenes here to push the clause well out of range, "
        "padding the gap further still. The mean is 10.5 and the std is 2.1."
    )
    variables, _ = ext._extract_with_regex(text)
    assert list(variables) == ["var_1"]


def test_normalize_variable_name():
    from agentdataset.core.extractor import _normalize_variable_name

    assert _normalize_variable_name("Heart Rate") == "heart_rate"
    assert _normalize_variable_name("education_num") == "education_num"
    assert _normalize_variable_name("  follow-up  ") == "follow_up"
    assert _normalize_variable_name("BMI") == "bmi"


def test_regex_categorical_name_is_normalized_for_matching():
    """Categorical names normalize too, so they can resolve in a correlation."""
    ext = Extractor()
    text = (
        "The categorical variable Sex takes value 'Female' with probability 0.4 "
        "and 'Male' with probability 0.6.\n"
        "The variable age has mean 38.0 and std 13.0.\n"
        "The correlation between sex and age is 0.12."
    )
    variables, correlations = ext._extract_with_regex(text)
    assert "sex" in variables
    # Labels keep their original casing; only the variable name is normalized.
    assert variables["sex"].categories == {"Female": 0.4, "Male": 0.6}
    corr = list(correlations.values())[0]
    assert corr.var1 in variables and corr.var2 in variables


# --- LLM path: fallback-only correlation name resolution ----------------------


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
