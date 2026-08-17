from types import SimpleNamespace

import pandas as pd
import pytest

from agentdataset.models.schemas import MetaParams, Parameters, VariableParams
from analysis import end_to_end_cost as cost


def _response(content="ok", prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def test_spend_guard_stops_on_cost_and_attempt_limits():
    cost_guard = cost.SpendGuard(cap_usd=0.05, max_attempts=10)
    assert cost_guard.before_attempt() == 1
    cost_guard.record_known_cost(0.05)
    with pytest.raises(cost.BudgetStopped, match="reached cap"):
        cost_guard.before_attempt()

    attempt_guard = cost.SpendGuard(cap_usd=1.0, max_attempts=1)
    assert attempt_guard.before_attempt() == 1
    with pytest.raises(cost.BudgetStopped, match="attempt cap"):
        attempt_guard.before_attempt()


def test_accounting_extractor_records_failed_attempt_and_retry(monkeypatch):
    responses = iter([RuntimeError("temporary"), _response()])

    def fake_completion(**kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("agentdataset.core.extractor.completion", fake_completion)
    monkeypatch.setattr("agentdataset.core.extractor.litellm.completion_cost", lambda **kwargs: 0.012)
    guard = cost.SpendGuard(cap_usd=0.25, max_attempts=12)
    extractor = cost.AccountingExtractor(model="test/model", guard=guard)

    assert extractor.llm_call("prompt", purpose="query_expansion") == "ok"
    assert [row["status"] for row in extractor.attempt_history] == ["failed", "ok"]
    assert extractor.attempt_history[0]["cost_known"] is False
    assert extractor.attempt_history[1]["retry_index"] == 1
    assert extractor.attempt_history[1]["prompt_tokens"] == 10
    assert guard.attempts == 2
    assert guard.known_cost_usd == pytest.approx(0.012)


def test_summary_averages_timing_repetitions_before_totaling():
    rows = [
        cost._base_row(
            "s", "r", "query_expansion", provider="openrouter",
            llm_calls=1, prompt_tokens=10, completion_tokens=5, total_tokens=15,
            api_cost_usd=0.01, api_seconds=1.0, wall_seconds=1.0,
        ),
        cost._base_row(
            "s", "r", "web_search", network_seconds=5.0, wall_seconds=5.0,
        ),
    ]
    for repetition, seconds in enumerate((1.0, 2.0, 3.0)):
        rows.append(cost._base_row(
            "s", "r", "synthesis", timing_repetition=repetition,
            local_seconds=seconds, wall_seconds=seconds,
            optimization_iterations=3,
        ))

    summary = cost._summarize(pd.DataFrame(rows))
    synthesis = summary.loc[
        summary["source_id"].eq("s") & summary["stage"].eq("synthesis")
    ].iloc[0]
    total = summary.loc[
        summary["source_id"].eq("s") & summary["stage"].eq("total")
    ].iloc[0]

    assert synthesis.local_seconds == pytest.approx(2.0)
    assert total.api_cost_usd == pytest.approx(0.01)
    assert total.api_seconds == pytest.approx(1.0)
    assert total.network_seconds == pytest.approx(5.0)
    assert total.local_seconds == pytest.approx(2.0)
    assert total.wall_seconds == pytest.approx(8.0)
    mean_synthesis = summary.loc[
        summary["source_id"].eq("mean_one_dataset")
        & summary["stage"].eq("synthesis")
    ].iloc[0]
    assert mean_synthesis.local_seconds == pytest.approx(2.0)


def test_target_selection_records_prespecified_fallback():
    source = pd.Series({
        "title": "Chosen paper",
        "source_url": "https://example.test/paper",
        "direct_url": "https://example.test/paper.pdf",
        "source_type": "pdf",
    })
    selected, route = cost._select_target([], source)
    assert selected.url == source.direct_url
    assert route == "prespecified_direct_url_fallback"


def test_target_selection_prefers_source_domain_over_loose_title_match():
    source = pd.Series({
        "title": "Rural Urban Differences in Objective and Subjective Measures of Physical Activity",
        "source_url": "https://www.cdc.gov/pcd/issues/2014/14_0189.htm",
        "direct_url": "https://www.cdc.gov/pcd/issues/2014/14_0189.htm",
        "source_type": "html",
    })
    loose = cost.DiscoveryResult(
        title="Rural urban differences in objective measures of physical activity",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4149321/",
        source_type="pdf",
        relevance_score=1.0,
    )
    exact = cost.DiscoveryResult(
        title=source.title,
        url=source.direct_url,
        source_type="html",
        relevance_score=1.0,
    )
    selected, route = cost._select_target([loose, exact], source)
    assert selected.url == source.direct_url
    assert route == "discovered_target"


def test_cached_mode_constructs_no_accounting_extractor(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    gold = tmp_path / "gold.csv"
    cache = tmp_path / "cache"
    parsed = tmp_path / ".agentdataset_cache" / "real_papers"
    parsed.mkdir(parents=True)
    pd.DataFrame([{"source_id": "s", "sample_size": 10}]).to_csv(manifest, index=False)
    pd.DataFrame([{"unused": 1}]).to_csv(gold, index=False)
    (parsed / "s.txt").write_text("cached text", encoding="utf-8")
    params = Parameters(
        variables={"x": VariableParams(name="x", mean=0, std=1)},
        correlations={},
        meta=MetaParams(source="s", extracted_at="now", extraction_method="gold"),
    )

    monkeypatch.setattr(cost, "ROOT", tmp_path)
    monkeypatch.setattr(cost, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(cost, "GOLD_PATH", gold)
    monkeypatch.setattr(cost, "CACHE_DIR", cache)
    monkeypatch.setattr(cost, "gold_to_parameters", lambda source_id, frame: params)
    monkeypatch.setattr(cost, "_time_local_pipeline", lambda source, params, run_id: [])
    monkeypatch.setattr(
        cost,
        "AccountingExtractor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("paid path used")),
    )

    raw, _, metadata = cost.run("cached")
    assert not raw["record_type"].eq("llm_attempt").any()
    assert raw.loc[raw["stage"].eq("query_expansion"), "status"].item() == "skipped_cached"
    assert metadata["known_cost_usd"] == 0.0
    assert metadata["llm_attempts"] == 0
