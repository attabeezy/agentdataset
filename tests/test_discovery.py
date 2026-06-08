import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from agentdataset.core.discovery import DiscoveryAgent, PDF_PATH_PREFIX, SearchError, _HTTP_HEADERS
from agentdataset.models.schemas import DiscoveryResult


def test_discovery_agent_init():
    agent = DiscoveryAgent(max_results=10)
    assert agent.max_results == 10


@patch('agentdataset.core.discovery.DDGS')
def test_discovery_agent_search(mock_ddgs):
    mock_instance = mock_ddgs.return_value.__enter__.return_value
    mock_instance.text.return_value = [
        {'title': 'Result 1', 'href': 'http://test.com/file.pdf', 'body': 'Snippet 1'},
        {'title': 'Result 2', 'href': 'http://test.com/page', 'body': 'Snippet 2'}
    ]

    agent = DiscoveryAgent(max_results=2)
    results = agent.search("test query")

    assert len(results) > 0
    assert any(r.source_type == "pdf" for r in results)
    assert any(r.source_type == "html" for r in results)


@patch('agentdataset.core.discovery.trafilatura')
def test_discovery_agent_fetch_html(mock_traf):
    mock_traf.fetch_url.return_value = "<html>test</html>"
    mock_traf.extract.return_value = "extracted content"

    agent = DiscoveryAgent()
    res = DiscoveryResult(title="T", url="http://test.com", source_type="html", relevance_score=1.0)
    content = agent.fetch_content(res)

    assert content == "extracted content"
    mock_traf.fetch_url.assert_called_once_with("http://test.com")


@patch('agentdataset.core.discovery.requests')
@patch('agentdataset.core.discovery.tempfile.NamedTemporaryFile')
def test_discovery_agent_fetch_pdf_downloads(mock_ntf, mock_requests):
    """Successful PDF download returns pdf://<path> prefix."""
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"PDF bytes"]
    mock_requests.get.return_value.__enter__ = lambda s: s
    mock_requests.get.return_value = mock_response

    mock_tmp = MagicMock()
    mock_tmp.name = "/tmp/fake.pdf"
    mock_ntf.return_value = mock_tmp

    agent = DiscoveryAgent()
    res = DiscoveryResult(title="T", url="http://test.com/a.pdf", source_type="pdf", relevance_score=1.0, snippet="fallback")
    content = agent.fetch_content(res)

    assert content.startswith(PDF_PATH_PREFIX)
    assert content == PDF_PATH_PREFIX + "/tmp/fake.pdf"


@patch('agentdataset.core.discovery.requests')
def test_discovery_agent_fetch_pdf_fallback_on_error(mock_requests):
    """Failed PDF download falls back to snippet."""
    mock_requests.get.side_effect = Exception("network error")

    agent = DiscoveryAgent()
    res = DiscoveryResult(title="T", url="http://test.com/a.pdf", source_type="pdf", relevance_score=1.0, snippet="pdf snippet text")
    content = agent.fetch_content(res)

    assert content == "pdf snippet text"


@patch('agentdataset.core.discovery.requests')
@patch('agentdataset.core.discovery.tempfile.NamedTemporaryFile')
def test_pdf_download_sends_browser_headers(mock_ntf, mock_requests):
    """PDF download must send a browser User-Agent to avoid 403s."""
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"PDF"]
    mock_requests.get.return_value = mock_response
    mock_tmp = MagicMock(); mock_tmp.name = "/tmp/x.pdf"
    mock_ntf.return_value = mock_tmp

    agent = DiscoveryAgent()
    res = DiscoveryResult(title="T", url="http://x/a.pdf", source_type="pdf", relevance_score=1.0)
    agent.fetch_content(res)

    _, kwargs = mock_requests.get.call_args
    assert kwargs.get("headers") == _HTTP_HEADERS
    assert "User-Agent" in kwargs["headers"]


@patch('agentdataset.core.discovery.trafilatura')
def test_html_fetch_falls_back_to_snippet(mock_traf):
    """HTML extraction returning None falls back to the search snippet."""
    mock_traf.fetch_url.return_value = "<html></html>"
    mock_traf.extract.return_value = None  # nothing extractable

    agent = DiscoveryAgent()
    res = DiscoveryResult(title="T", url="http://x", source_type="html", relevance_score=1.0, snippet="snip")
    assert agent.fetch_content(res) == "snip"


@patch('agentdataset.core.discovery.DDGS')
def test_search_raises_on_backend_failure(mock_ddgs):
    """A backend failure with no results raises SearchError (not silent empty)."""
    mock_ddgs.side_effect = RuntimeError("backend down")
    agent = DiscoveryAgent()
    with pytest.raises(SearchError):
        agent.search("anything")
