import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.models.schemas import Parameters, VariableParams, MetaParams, DiscoveryResult

@pytest.fixture
def mock_orchestrator(tmp_path):
    with patch('agentdataset.core.orchestrator.DiscoveryAgent'), \
         patch('agentdataset.core.orchestrator.Extractor'), \
         patch('agentdataset.core.orchestrator.Synthesizer'), \
         patch('agentdataset.core.orchestrator.Validator'):
        orc = Orchestrator(session_id="test_session", base_dir=str(tmp_path))
        return orc

def test_orchestrator_init(mock_orchestrator):
    assert mock_orchestrator.context.session_id == "test_session"

def test_run_discovery(mock_orchestrator):
    mock_orchestrator.discovery.search.return_value = [DiscoveryResult(title="T", url="U", source_type="pdf", relevance_score=1.0)]
    results = mock_orchestrator.run_discovery("query")
    assert len(results) == 1
    mock_orchestrator.discovery.search.assert_called_once_with("query")

def test_process_source(mock_orchestrator):
    res = DiscoveryResult(title="T", url="U", source_type="pdf", relevance_score=1.0)
    mock_orchestrator.discovery.fetch_content.return_value = "content"
    mock_orchestrator.extractor.extract_parameters.return_value = Parameters(
        variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N")
    )
    params = mock_orchestrator.process_source(res)
    assert isinstance(params, Parameters)

def test_run_optimization_loop(mock_orchestrator):
    params = Parameters(
        variables={"v1": VariableParams(name="v1")}, 
        correlations={}, 
        meta=MetaParams(source="S", extracted_at="N")
    )
    df = pd.DataFrame({"v1": [1, 2, 3]})
    mock_orchestrator.synthesizer.synthesize.return_value = df
    
    # Mock validator report
    report = MagicMock()
    report.overall_score = 95.0
    mock_orchestrator.validator.validate.return_value = report
    mock_orchestrator.validator.generate_datacard.return_value = "mock datacard content"
    
    score, data = mock_orchestrator.run_optimization_loop(params, iterations=1)
    
    assert score == 95.0
    assert data.equals(df)
    assert mock_orchestrator.best_score == 95.0
