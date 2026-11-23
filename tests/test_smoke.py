"""Smoke tests for the redesigned CLI-first, LLM-only, agentic pipeline."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentdataset.core.extractor import Extractor
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.cache import ContentCache
from agentdataset.models.schemas import Parameters, VariableParams, MetaParams


def test_extractor_requires_api_key():
    os.environ.pop("OPENROUTER_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    ext = Extractor()
    with pytest.raises(ValueError):
        ext.extract_parameters("mean 1 std 2", "src")


@patch("agentdataset.core.extractor.completion")
def test_extractor_llm_path(mock_completion):
    import json
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "variables": {"x": {"distribution": "normal", "mean": 1.0, "std": 2.0}},
        "correlations": {},
    })
    mock_completion.return_value = mock_response
    ext = Extractor(api_key="sk-test", fallback_models=["openrouter/auto"])
    params = ext.extract_parameters("text", "src")
    assert params.meta.extraction_method == "llm"
    assert "x" in params.variables
    assert ext._model_chain[0] == "openrouter/auto"


@patch("agentdataset.core.extractor.completion")
def test_extractor_fallback_tries_next_model(mock_completion):
    mock_completion.side_effect = [
        Exception("primary down"),
        MagicMock(choices=[MagicMock(message=MagicMock(content='{"variables":{},"correlations":{}}'))]),
    ]
    ext = Extractor(model="openrouter/auto", api_key="sk-test",
                    fallback_models=["openrouter/auto", "openrouter/free"])
    # force model chain to include a second
    ext.fallback_models = ["openrouter/auto", "openrouter/free"]
    params = ext.extract_parameters("text", "src")
    assert params.meta.extraction_method == "llm"
    assert mock_completion.call_count == 2


def test_cache_round_trip(tmp_path):
    c = ContentCache(base_dir=str(tmp_path / "cache"))
    key = c.key("a", "b", 1)
    assert c.get(key) is None
    c.set(key, {"hello": 42})
    assert c.get(key) == {"hello": 42}
    c.invalidate(key)
    assert c.get(key) is None


def test_process_source_uses_cache(tmp_path):
    from unittest.mock import patch, MagicMock
    orch = Orchestrator(session_id="s", base_dir=str(tmp_path / "sessions"),
                        api_key="sk-test", env_var="OPENROUTER_API_KEY")
    # prime cache
    cached_params = Parameters(
        variables={"v": VariableParams(name="v", mean=1.0, std=2.0)},
        correlations={},
        meta=MetaParams(source="cached", extracted_at="now", extraction_method="llm"),
    )
    key = orch.cache.key("process_source", "http://example.com/x", orch.extractor.model,
                         orch.extractor.env_var, "v1")
    orch.cache.set(key, cached_params.model_dump())
    # discovery result mock
    res = MagicMock()
    res.url = "http://example.com/x"
    res.title = "t"
    out = orch.process_source(res)
    assert out.variables["v"].mean == 1.0


@patch("agentdataset.core.extractor.completion")
def test_agent_loop_accepts(mock_completion):
    import json
    # extraction
    mock_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(
            content=json.dumps({"variables": {"v": {"distribution": "normal", "mean": 1.0, "std": 2.0}},
                               "correlations": {}})))]),
        # critique
        MagicMock(choices=[MagicMock(message=MagicMock(content="looks good"))]),
        # agent decision: accept
        MagicMock(choices=[MagicMock(message=MagicMock(content='{"action":"accept","noise_level":0.02,"corr_repair":null,"reason":"ok"}'))]),
    ]
    orch = Orchestrator(session_id="s", base_dir=tempfile.mkdtemp(), api_key="sk-test",
                        env_var="OPENROUTER_API_KEY")
    params = orch.extractor.extract_parameters("text", "src")
    score, df = orch.run_agent_loop(params, iterations=2, source_text="text")
    assert df is not None and not df.empty
    assert orch.optimization_trace[0]["action"] == "accept"


def test_doctor_command_runs():
    from agentdataset.cli import main
    # doctor should at least not crash when run with --help
    try:
        main(["doctor", "--help"])
    except SystemExit as e:
        assert e.code in (0, None)


def test_no_regex_attribute():
    ext = Extractor(api_key="sk-test")
    assert not hasattr(ext, "_extract_with_regex")


def test_no_noise_pivot_in_orchestrator():
    import agentdataset.core.orchestrator as o
    assert not hasattr(o, "NoisePivotConfig")
    assert not hasattr(o, "next_noise")
    assert not hasattr(o, "DEFAULT_PIVOT")
