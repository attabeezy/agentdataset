import pytest
from pathlib import Path
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


def test_prune_old_sessions(tmp_path):
    from agentdataset.core.orchestrator import _prune_old_sessions
    # Create 5 fake session dirs (names sort lexicographically)
    for i in range(5):
        (tmp_path / f"run_00{i}").mkdir()
    _prune_old_sessions(str(tmp_path), keep=3)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["run_002", "run_003", "run_004"]


def test_prune_does_nothing_when_under_limit(tmp_path):
    from agentdataset.core.orchestrator import _prune_old_sessions
    for i in range(2):
        (tmp_path / f"run_00{i}").mkdir()
    _prune_old_sessions(str(tmp_path), keep=3)
    assert len(list(tmp_path.iterdir())) == 2


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


def test_run_optimization_loop_rejects_empty_params(mock_orchestrator):
    """No variables → loop refuses rather than synthesizing an empty frame."""
    empty = Parameters(
        variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N")
    )
    with pytest.raises(ValueError):
        mock_orchestrator.run_optimization_loop(empty, iterations=1)
    mock_orchestrator.synthesizer.synthesize.assert_not_called()


def test_optimization_artifacts_support_unicode(mock_orchestrator):
    params = Parameters(
        variables={"score_α": VariableParams(name="score_α")},
        correlations={},
        meta=MetaParams(source="study α", extracted_at="now"),
    )
    frame = pd.DataFrame({"score_α": [1.0, 2.0, 3.0]})
    mock_orchestrator.synthesizer.synthesize.return_value = frame
    mock_orchestrator.validator.validate.return_value = MagicMock(
        overall_score=95.0, ks_score=100.0, corr_score=100.0, bias_score=100.0
    )
    mock_orchestrator.validator.generate_datacard.return_value = "diagnostic α"

    mock_orchestrator.run_optimization_loop(params, iterations=1)

    session = Path(mock_orchestrator.context.path)
    assert "α" in (session / "parameters.json").read_text(encoding="utf-8")
    assert "α" in (session / "DATACARD.md").read_text(encoding="utf-8")


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


# --- §9.7: the pivot schedule as configurable, testable surface ---------------


def _noise_sequence(cfg, orchestrator, n_iterations, scores):
    """Drive the loop with fixed scores and return the noise used per iteration."""
    params = Parameters(
        variables={"v1": VariableParams(name="v1")},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = pd.DataFrame({"v1": [1, 2, 3]})
    orchestrator.validator.validate.side_effect = [
        MagicMock(overall_score=s) for s in scores
    ]
    orchestrator.validator.generate_datacard.return_value = ""

    calls = []

    def capture(params, noise_level):
        calls.append(noise_level)
        return df

    orchestrator.synthesizer.synthesize.side_effect = capture
    orchestrator.run_optimization_loop(params, iterations=n_iterations, pivot=cfg)
    return calls


def test_next_noise_explore_exploit_reset():
    """The pure schedule function reproduces each documented branch."""
    from agentdataset.core.orchestrator import NoisePivotConfig, next_noise

    cfg = NoisePivotConfig()  # patience 2, reset_multiple 2 → full cycle 4

    noise, phase = next_noise(1, 0.1, cfg)
    assert phase == "explore" and noise == pytest.approx(0.11)

    noise, phase = next_noise(2, 0.11, cfg)
    assert phase == "exploit" and noise == pytest.approx(0.055)

    noise, phase = next_noise(3, 0.055, cfg)
    assert phase == "explore" and noise == pytest.approx(0.0605)

    noise, phase = next_noise(4, 0.0605, cfg)
    assert phase == "reset" and noise == pytest.approx(cfg.initial_noise)


def test_next_noise_respects_bounds():
    from agentdataset.core.orchestrator import NoisePivotConfig, next_noise

    cfg = NoisePivotConfig()
    # Exploit cannot go below min_noise.
    noise, phase = next_noise(2, cfg.min_noise, cfg)
    assert phase == "exploit" and noise == pytest.approx(cfg.min_noise)
    # Explore cannot exceed max_noise.
    noise, phase = next_noise(1, cfg.max_noise, cfg)
    assert phase == "explore" and noise == pytest.approx(cfg.max_noise)


def test_next_noise_reset_can_be_disabled():
    """reset_multiple=None is the study arm with no reset branch."""
    from agentdataset.core.orchestrator import NoisePivotConfig, next_noise

    cfg = NoisePivotConfig(reset_multiple=None)
    phases = [next_noise(s, 0.1, cfg)[1] for s in range(1, 9)]
    assert "reset" not in phases
    assert phases == ["explore", "exploit"] * 4


def test_reset_takes_precedence_over_exploit():
    """A full-cycle streak also satisfies the exploit condition; reset wins."""
    from agentdataset.core.orchestrator import NoisePivotConfig, next_noise

    cfg = NoisePivotConfig(patience=2, reset_multiple=2)
    assert next_noise(4, 0.5, cfg)[1] == "reset"


def test_invalid_pivot_config_rejected():
    from agentdataset.core.orchestrator import NoisePivotConfig

    with pytest.raises(ValueError, match="patience"):
        NoisePivotConfig(patience=0)
    with pytest.raises(ValueError, match="reset_multiple"):
        NoisePivotConfig(reset_multiple=0)
    with pytest.raises(ValueError, match="min_noise"):
        NoisePivotConfig(min_noise=1.0, max_noise=0.5)


def test_default_pivot_matches_omitting_the_argument(mock_orchestrator, tmp_path):
    """Passing DEFAULT_PIVOT explicitly must change nothing."""
    from agentdataset.core.orchestrator import DEFAULT_PIVOT

    scores = [95.0, 80.0, 79.0, 78.0, 77.0]
    explicit = _noise_sequence(DEFAULT_PIVOT, mock_orchestrator, 5, scores)

    with patch('agentdataset.core.orchestrator.DiscoveryAgent'), \
         patch('agentdataset.core.orchestrator.Extractor'), \
         patch('agentdataset.core.orchestrator.Synthesizer'), \
         patch('agentdataset.core.orchestrator.Validator'):
        fresh = Orchestrator(session_id="test_default", base_dir=str(tmp_path))
    implicit = _noise_sequence(None, fresh, 5, scores)

    assert explicit == pytest.approx(implicit)


def test_custom_pivot_changes_the_schedule(mock_orchestrator):
    """A non-default config must actually take effect end to end."""
    from agentdataset.core.orchestrator import NoisePivotConfig

    cfg = NoisePivotConfig(initial_noise=0.4, explore_mult=2.0, patience=3)
    calls = _noise_sequence(cfg, mock_orchestrator, 3, [95.0, 80.0, 79.0])

    assert calls[0] == pytest.approx(0.4)
    # streak 1 → explore at the custom multiplier
    assert calls[1] == pytest.approx(0.4)
    assert calls[2] == pytest.approx(0.8)


def test_optimization_trace_records_phase(mock_orchestrator):
    """The trace carries the pivot phase so the ablation can count resets."""
    scores = [95.0, 80.0, 79.0, 78.0, 77.0]
    _noise_sequence(None, mock_orchestrator, 5, scores)

    phases = [row["phase"] for row in mock_orchestrator.optimization_trace]
    assert phases == ["keep", "explore", "exploit", "explore", "reset"]
    assert all("noise_after" in row for row in mock_orchestrator.optimization_trace)
    timing_fields = {
        "synthesis_seconds",
        "validation_seconds",
        "artifact_write_seconds",
        "iteration_seconds",
        "refinement_overhead_seconds",
    }
    assert all(
        timing_fields <= row.keys() for row in mock_orchestrator.optimization_trace
    )
    for row in mock_orchestrator.optimization_trace:
        assert all(row[field] >= 0 for field in timing_fields)
        accounted = (
            row["synthesis_seconds"]
            + row["validation_seconds"]
            + row["artifact_write_seconds"]
            + row["refinement_overhead_seconds"]
        )
        assert accounted == pytest.approx(row["iteration_seconds"])
