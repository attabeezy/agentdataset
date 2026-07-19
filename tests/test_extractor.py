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
