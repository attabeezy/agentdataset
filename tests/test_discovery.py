import pytest
from unittest.mock import MagicMock, patch
from agentdataset.core.discovery import DiscoveryAgent
from agentdataset.models.schemas import DiscoveryResult

def test_discovery_agent_init():
    agent = DiscoveryAgent(max_results=10)
    assert agent.max_results == 10

@patch('agentdataset.core.discovery.DDGS')
def test_discovery_agent_search(mock_ddgs):
    # Mock DDGS context manager
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

def test_discovery_agent_fetch_pdf():
    agent = DiscoveryAgent()
    # PDF fetch returns the search snippet (not the raw URL) until full PDF download is implemented
    res = DiscoveryResult(title="T", url="http://test.com/a.pdf", source_type="pdf", relevance_score=1.0, snippet="pdf snippet text")
    content = agent.fetch_content(res)
    assert content == "pdf snippet text"
