import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.discovery import PDF_PATH_PREFIX
from agentdataset.models.schemas import Parameters, VariableParams, CorrelationParams, MetaParams, DiscoveryResult

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


def _make_params(source, variables, correlations=None):
    return Parameters(
        variables=variables,
        correlations=correlations or {},
        meta=MetaParams(source=source, extracted_at="2026-01-01 00:00:00"),
    )


def test_merge_parameters_single(mock_orchestrator):
    p = _make_params("s1", {"age": VariableParams(name="age", mean=30.0, std=5.0)})
    assert mock_orchestrator.merge_parameters([p]) is p


def test_merge_parameters_averages_same_variable(mock_orchestrator):
    p1 = _make_params("s1", {"age": VariableParams(name="age", mean=30.0, std=5.0)})
    p2 = _make_params("s2", {"age": VariableParams(name="age", mean=40.0, std=7.0)})
    merged = mock_orchestrator.merge_parameters([p1, p2])
    assert merged.variables["age"].mean == pytest.approx(35.0)
    assert merged.variables["age"].std == pytest.approx(6.0)
    assert "s1" in merged.meta.source and "s2" in merged.meta.source


def test_merge_parameters_unions_different_variables(mock_orchestrator):
    p1 = _make_params("s1", {"age": VariableParams(name="age", mean=30.0, std=5.0)})
    p2 = _make_params("s2", {"income": VariableParams(name="income", mean=5000.0, std=1000.0)})
    merged = mock_orchestrator.merge_parameters([p1, p2])
    assert "age" in merged.variables
    assert "income" in merged.variables


def test_merge_parameters_averages_correlations(mock_orchestrator):
    corr1 = {"c1": CorrelationParams(var1="a", var2="b", correlation=0.6)}
    corr2 = {"c1": CorrelationParams(var1="a", var2="b", correlation=0.8)}
    p1 = _make_params("s1", {"a": VariableParams(name="a"), "b": VariableParams(name="b")}, corr1)
    p2 = _make_params("s2", {"a": VariableParams(name="a"), "b": VariableParams(name="b")}, corr2)
    merged = mock_orchestrator.merge_parameters([p1, p2])
    assert merged.correlations["c1"].correlation == pytest.approx(0.7)

def test_run_discovery(mock_orchestrator):
    mock_orchestrator.discovery.search.return_value = [DiscoveryResult(title="T", url="U", source_type="pdf", relevance_score=1.0)]
    results = mock_orchestrator.run_discovery("query")
    assert len(results) == 1
    mock_orchestrator.discovery.search.assert_called_once_with("query")

def test_process_source_plain_text(mock_orchestrator):
    res = DiscoveryResult(title="T", url="U", source_type="html", relevance_score=1.0)
    mock_orchestrator.discovery.fetch_content.return_value = "plain text content"
    mock_orchestrator.extractor.extract_parameters.return_value = Parameters(
        variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N")
    )
    params = mock_orchestrator.process_source(res)
    assert isinstance(params, Parameters)
    mock_orchestrator.extractor.pdf_to_markdown.assert_not_called()
    mock_orchestrator.extractor.extract_parameters.assert_called_once_with("plain text content", "T")


def test_process_source_pdf_path(mock_orchestrator, tmp_path):
    """pdf:// prefix triggers pdf_to_markdown then cleans up the temp file."""
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"PDF")

    res = DiscoveryResult(title="T", url="U", source_type="pdf", relevance_score=1.0)
    mock_orchestrator.discovery.fetch_content.return_value = PDF_PATH_PREFIX + str(fake_pdf)
    mock_orchestrator.extractor.pdf_to_markdown.return_value = "parsed markdown"
    mock_orchestrator.extractor.extract_parameters.return_value = Parameters(
        variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N")
    )

    params = mock_orchestrator.process_source(res)

    assert isinstance(params, Parameters)
    mock_orchestrator.extractor.pdf_to_markdown.assert_called_once_with(str(fake_pdf))
    mock_orchestrator.extractor.extract_parameters.assert_called_once_with("parsed markdown", "T")
    assert not fake_pdf.exists(), "Temp PDF should be deleted after processing"

def test_run_optimization_loop(mock_orchestrator):
    params = Parameters(
        variables={"v1": VariableParams(name="v1")},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    df = pd.DataFrame({"v1": [1, 2, 3]})
    mock_orchestrator.synthesizer.synthesize.return_value = df

    report = MagicMock()
    report.overall_score = 95.0
    mock_orchestrator.validator.validate.return_value = report
    mock_orchestrator.validator.generate_datacard.return_value = "mock datacard content"

    score, data = mock_orchestrator.run_optimization_loop(params, iterations=1)

    assert score == 95.0
    assert data.equals(df)
    assert mock_orchestrator.best_score == 95.0


def test_noise_pivot_strategy(mock_orchestrator):
    """Streak counter drives explore → exploit → reset transitions."""
    from agentdataset.core.orchestrator import PATIENCE, MAX_NOISE, MIN_NOISE

    params = Parameters(
        variables={"v1": VariableParams(name="v1")},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    df = pd.DataFrame({"v1": [1, 2, 3]})
    mock_orchestrator.synthesizer.synthesize.return_value = df

    # Scores: first improves (streak reset), then 4 consecutive non-improvements
    # to exercise explore (streak 1), exploit (streak 2), explore (streak 3), reset (streak 4)
    scores = [95.0, 80.0, 79.0, 78.0, 77.0]
    reports = [MagicMock(overall_score=s) for s in scores]
    mock_orchestrator.validator.validate.side_effect = reports
    mock_orchestrator.validator.generate_datacard.return_value = ""

    # Capture noise_level passed to synthesize on each call
    noise_calls = []
    original_synthesize = mock_orchestrator.synthesizer.synthesize
    def capture_noise(params, noise_level):
        noise_calls.append(noise_level)
        return df
    mock_orchestrator.synthesizer.synthesize.side_effect = capture_noise

    mock_orchestrator.run_optimization_loop(params, iterations=5)

    initial = 0.1
    # iter 0: noise = 0.1, score 95 → keep, streak resets to 0
    assert noise_calls[0] == pytest.approx(initial)
    # iter 1: streak=1 (explore) → noise *= 1.1
    assert noise_calls[1] == pytest.approx(initial)
    explore_noise = initial * 1.1
    # iter 2: streak=2 (exploit, streak % PATIENCE == 0) → noise *= 0.5
    assert noise_calls[2] == pytest.approx(explore_noise)
    exploit_noise = max(explore_noise * 0.5, MIN_NOISE)
    # iter 3: streak=3 (explore again)
    assert noise_calls[3] == pytest.approx(exploit_noise)
    # iter 4: streak=4 (full cycle, streak % (PATIENCE*2) == 0) → reset
    assert noise_calls[4] == pytest.approx(min(exploit_noise * 1.1, MAX_NOISE))
