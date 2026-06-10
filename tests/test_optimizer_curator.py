import pytest
from unittest.mock import MagicMock, patch
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.models.schemas import DiscoveryResult


@pytest.fixture
def mock_orchestrator(tmp_path):
    with (
        patch("agentdataset.core.orchestrator.DiscoveryAgent"),
        patch("agentdataset.core.orchestrator.Extractor"),
        patch("agentdataset.core.orchestrator.Synthesizer"),
        patch("agentdataset.core.orchestrator.Validator"),
    ):
        orc = Orchestrator(session_id="test_session", base_dir=str(tmp_path))
        return orc


def test_optimize_query(mock_orchestrator):
    # Mock LLM response with 3 queries on separate lines.
    mock_orchestrator.extractor.llm_call.return_value = "Query 1\nQuery 2\nQuery 3"

    original_query = "SME lending in Kenya"
    optimized = mock_orchestrator.optimize_query(original_query)

    assert original_query in optimized
    assert len(optimized) == 4  # Original + 3 optimized
    assert "Query 1" in optimized
    mock_orchestrator.extractor.llm_call.assert_called_once()


def test_optimize_query_failure(mock_orchestrator):
    # Simulate LLM failure
    mock_orchestrator.extractor.llm_call.side_effect = Exception("API Error")

    original_query = "SME lending in Kenya"
    optimized = mock_orchestrator.optimize_query(original_query)

    assert optimized == [original_query]


def test_run_discovery_with_optimization(mock_orchestrator):
    # Setup: 2 optimized queries, each returning 1 unique result
    with patch.object(mock_orchestrator, "optimize_query", return_value=["q1", "q2"]):
        res1 = DiscoveryResult(
            title="T1", url="U1", source_type="pdf", relevance_score=1.0, snippet="S1"
        )
        res2 = DiscoveryResult(
            title="T2", url="U2", source_type="pdf", relevance_score=1.0, snippet="S2"
        )

        # search() is called for each optimized query
        mock_orchestrator.discovery.search.side_effect = [[res1], [res2]]

        results = mock_orchestrator.run_discovery("some query")

        assert len(results) == 2
        assert results[0].url == "U1"
        assert results[1].url == "U2"
        assert mock_orchestrator.discovery.search.call_count == 2


def test_run_discovery_deduplication(mock_orchestrator):
    with patch.object(mock_orchestrator, "optimize_query", return_value=["q1", "q2"]):
        res1 = DiscoveryResult(
            title="T1", url="U1", source_type="pdf", relevance_score=1.0, snippet="S1"
        )

        # Both queries return the same result
        mock_orchestrator.discovery.search.side_effect = [[res1], [res1]]

        results = mock_orchestrator.run_discovery("some query")

        assert len(results) == 1
        assert results[0].url == "U1"


def test_suggest_sources_success(mock_orchestrator):
    results = [
        DiscoveryResult(
            title="General News",
            url="U1",
            source_type="html",
            relevance_score=0.8,
            snippet="Just some news about economy.",
        ),
        DiscoveryResult(
            title="Research Paper",
            url="U2",
            source_type="pdf",
            relevance_score=1.0,
            snippet="Mean was 5.2, SD was 1.1.",
        ),
        DiscoveryResult(
            title="Blog Post",
            url="U3",
            source_type="html",
            relevance_score=0.7,
            snippet="I think lending is hard.",
        ),
        DiscoveryResult(
            title="Annual Report",
            url="U4",
            source_type="pdf",
            relevance_score=1.0,
            snippet="Correlation of 0.8 between income and loan.",
        ),
    ]

    # Mock LLM to suggest IDs 1 and 3 (Research Paper and Annual Report)
    mock_orchestrator.extractor.llm_call.return_value = "1, 3"

    suggestions = mock_orchestrator.suggest_sources(results)

    assert suggestions == [1, 3]


def test_suggest_sources_none(mock_orchestrator):
    results = [
        DiscoveryResult(
            title="T1",
            url="U1",
            source_type="html",
            relevance_score=0.8,
            snippet="No stats here.",
        )
    ]
    mock_orchestrator.extractor.llm_call.return_value = "None"

    suggestions = mock_orchestrator.suggest_sources(results)
    assert suggestions == []


def test_suggest_sources_malformed_response(mock_orchestrator):
    results = [
        DiscoveryResult(
            title="T1", url="U1", source_type="html", relevance_score=0.8, snippet="..."
        )
    ]
    # LLM returns a sentence instead of just IDs
    mock_orchestrator.extractor.llm_call.return_value = (
        "I suggest that source 0 is the best one."
    )

    suggestions = mock_orchestrator.suggest_sources(results)
    assert suggestions == [0]
