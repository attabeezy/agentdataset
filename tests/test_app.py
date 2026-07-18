import pandas as pd
import pytest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agentdataset.core.discovery import SearchError
from agentdataset.models.schemas import (
    DiscoveryResult,
    Parameters,
    VariableParams,
    MetaParams,
)


def _click(at, label):
    for b in at.button:
        if b.label == label:
            b.click()
            return
    raise AssertionError(f"No button found with label {label!r}")


def _make_results():
    return [
        DiscoveryResult(
            title="Paper A", url="http://a.com", source_type="pdf",
            relevance_score=1.0, snippet="s",
        ),
        DiscoveryResult(
            title="Paper B", url="http://b.com", source_type="html",
            relevance_score=0.8, snippet="s",
        ),
    ]


def _make_params(source="s", variables=None, method="llm"):
    return Parameters(
        variables=variables or {},
        correlations={},
        meta=MetaParams(
            source=source, extracted_at="2026-01-01 00:00:00", extraction_method=method
        ),
    )


def _run_search(at, query="SME lending"):
    at.text_input(key="search_query").input(query)
    at.run()
    _click(at, "Search Knowledge Sources")
    at.run()


def test_empty_query_shows_warning():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        at = AppTest.from_file("app.py")
        at.run()
        _click(at, "Search Knowledge Sources")
        at.run()

        assert any("Please enter a research query" in w.value for w in at.warning)
        instance.run_discovery.assert_not_called()


def test_successful_search_shows_sources_and_suggestions():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_discovery.return_value = _make_results()
        instance.suggest_sources.return_value = [1]

        at = AppTest.from_file("app.py")
        at.run()
        _run_search(at)

        assert any("Found 2 potential sources" in s.value for s in at.success)
        markdown_text = " ".join(m.value for m in at.markdown)
        assert "Paper A" in markdown_text and "Paper B" in markdown_text
        assert "Suggested" in markdown_text


def test_no_sources_found_shows_info():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_discovery.return_value = []
        instance.suggest_sources.return_value = []

        at = AppTest.from_file("app.py")
        at.run()
        _run_search(at)

        assert any("No sources found" in i.value for i in at.info)


def test_search_error_shows_error_message():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_discovery.side_effect = SearchError("backend down")

        at = AppTest.from_file("app.py")
        at.run()
        _run_search(at)

        assert any("Search failed" in e.value for e in at.error)
        assert at.session_state["discovery_results"] == []
        assert at.session_state["suggested_indices"] == []


def test_zero_variable_extraction_shows_error():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_discovery.return_value = _make_results()
        instance.suggest_sources.return_value = []
        empty_params = _make_params(method="regex_fallback")
        instance.process_source.return_value = empty_params
        instance.merge_parameters.return_value = empty_params

        at = AppTest.from_file("app.py")
        at.run()
        _run_search(at)

        _click(at, "Generate Dataset from Selected")
        at.run()

        assert any(
            "No statistical parameters could be extracted" in e.value
            for e in at.error
        )
        assert any("regex_fallback" in e.value for e in at.error)
        instance.run_optimization_loop.assert_not_called()


def test_multi_source_merge_calls_optimization_with_merged_params():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_discovery.return_value = _make_results()
        instance.suggest_sources.return_value = []

        params_a = _make_params(source="Paper A", variables={
            "age": VariableParams(name="age", mean=30.0, std=5.0)
        })
        params_b = _make_params(source="Paper B", variables={
            "income": VariableParams(name="income", mean=50000.0, std=1000.0)
        })
        merged = _make_params(source="merged(Paper A, Paper B)", variables={
            "age": VariableParams(name="age", mean=30.0, std=5.0),
            "income": VariableParams(name="income", mean=50000.0, std=1000.0),
        })
        instance.process_source.side_effect = [params_a, params_b]
        instance.merge_parameters.return_value = merged
        instance.run_optimization_loop.return_value = (
            0.9,
            pd.DataFrame({"age": [30], "income": [50000]}),
        )

        at = AppTest.from_file("app.py")
        at.run()
        _run_search(at)

        at.checkbox(key="check_1").check()
        at.run()

        _click(at, "Generate Dataset from Selected")
        at.run()

        assert instance.process_source.call_count == 2
        instance.merge_parameters.assert_called_once_with([params_a, params_b])
        instance.run_optimization_loop.assert_called_once()
        called_params = instance.run_optimization_loop.call_args[0][0]
        assert called_params is merged
        # The merge status line is written to a transient `st.empty()` placeholder
        # that gets overwritten later in the same run (by "Running Synthesis-..."),
        # so it isn't observable in the final tree — the call assertions above are
        # the meaningful signal that the merge path actually ran.
        assert any("Fidelity Score: 0.9" in s.value for s in at.success)


def test_successful_generation_shows_results_panel():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_discovery.return_value = _make_results()
        instance.suggest_sources.return_value = []
        params = _make_params(variables={
            "age": VariableParams(name="age", mean=30.0, std=5.0)
        })
        instance.process_source.return_value = params
        instance.merge_parameters.return_value = params
        df = pd.DataFrame({"age": [29, 31, 30]})
        instance.run_optimization_loop.return_value = (0.87, df)

        at = AppTest.from_file("app.py")
        at.run()
        _run_search(at)

        _click(at, "Generate Dataset from Selected")
        at.run()

        assert any("Fidelity Score: 0.87" in s.value for s in at.success)
        assert at.session_state["best_data"] is not None
        assert any(
            "Final Synthetic Dataset" in h.value for h in at.subheader
        )


def test_no_sources_selected_shows_warning():
    with patch("agentdataset.core.orchestrator.Orchestrator") as MockOrch:
        instance = MockOrch.return_value
        instance.run_discovery.return_value = _make_results()
        instance.suggest_sources.return_value = []

        at = AppTest.from_file("app.py")
        at.run()
        _run_search(at)

        at.checkbox(key="check_0").uncheck()
        at.run()

        _click(at, "Generate Dataset from Selected")
        at.run()

        assert any(
            "Please select at least one source" in w.value for w in at.warning
        )
        instance.process_source.assert_not_called()
