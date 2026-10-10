"""Tests for the P1 9.6 extraction challenge set, scorer and prompt ablation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import benchmark
from agentdataset.core.extractor import CAVEMAN_PROMPT, Extractor
from agentdataset.models.schemas import CorrelationParams, VariableParams

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_SET_PATH = ROOT / "analysis" / "extraction_challenge_set.json"
CACHE_DIR = ROOT / ".agentdataset_cache" / "real_papers"
MANIFEST_PATH = ROOT / "analysis" / "real_paper_manifest.csv"


def _payload() -> dict:
    return json.loads(CHALLENGE_SET_PATH.read_text(encoding="utf-8"))


def _cases() -> list[dict]:
    return _payload()["cases"]


def _var(name, mean, std, distribution="normal"):
    return VariableParams(
        name=name, distribution=distribution, mean=mean, std=std,
        min=mean - 3 * std, max=mean + 3 * std,
    )


def _case(gold_variables=(), gold_correlations=(), must_not_extract=(), text=""):
    return {
        "case_id": "synthetic",
        "category": "test",
        "provenance": "constructed",
        "text": text,
        "gold_variables": list(gold_variables),
        "gold_correlations": list(gold_correlations),
        "must_not_extract": list(must_not_extract),
    }


def _gold(name, mean, std, group=None, aliases=()):
    return {
        "name": name, "aliases": list(aliases), "distribution": "normal",
        "mean": mean, "std": std, "units": None, "population": None,
        "variant_group": group or name,
    }


# --------------------------------------------------------------------------
# Fixture integrity
# --------------------------------------------------------------------------

def test_challenge_set_covers_every_category():
    payload = _payload()
    present = {c["category"] for c in payload["cases"]}
    assert present == set(payload["categories"])
    for category in payload["categories"]:
        n = sum(c["category"] == category for c in payload["cases"])
        assert n >= 3, f"{category} has only {n} case(s)"


def test_case_ids_unique_and_schema_complete():
    cases = _cases()
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids))
    required = {"case_id", "category", "provenance", "source_id", "text",
                "gold_variables", "gold_correlations", "must_not_extract", "notes"}
    for case in cases:
        assert required <= set(case), f"{case['case_id']} missing {required - set(case)}"
        assert case["provenance"] in {"real", "constructed"}
        assert case["text"].strip(), f"{case['case_id']} has empty text"
        for gold in case["gold_variables"]:
            assert {"mean", "std", "variant_group"} <= set(gold)
        for entry in case["must_not_extract"]:
            assert {"value", "reason"} <= set(entry)
            assert entry["reason"].strip(), "every distractor needs a stated reason"


def test_constructed_cases_are_labelled_as_such():
    """A constructed case must never read as a real quotation."""
    for case in _cases():
        if case["provenance"] == "constructed":
            assert case["source_id"] is None
            assert "constructed" in case["text"].lower(), case["case_id"]


def test_real_cases_resolve_to_the_source_manifest():
    manifest = pd.read_csv(MANIFEST_PATH)
    known = set(manifest["source_id"])
    for case in _cases():
        if case["provenance"] == "real":
            assert case["source_id"] in known, case["case_id"]
            assert case["spans"], f"{case['case_id']} declares no spans"
            assert case["source_sha256"]


@pytest.mark.skipif(not CACHE_DIR.exists(),
                    reason="cached real papers are gitignored; run real_paper_eval.py --mode live")
def test_real_excerpts_are_verbatim_at_their_declared_offsets():
    payload = _payload()
    join = payload["span_join"]
    for case in payload["cases"]:
        if case["provenance"] != "real":
            continue
        text = (CACHE_DIR / f"{case['source_id']}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        pieces = [text[s["start"]:s["end"]] for s in case["spans"]]
        for piece in pieces:
            assert piece in text
        assert join.join(pieces) == case["text"], case["case_id"]


@pytest.mark.skipif(not CACHE_DIR.exists(), reason="cached real papers are gitignored")
def test_fixture_matches_its_case_definitions():
    """The committed JSON must be regenerable from the annotations."""
    import analysis.build_extraction_challenge_set as builder
    rebuilt = builder.build()
    assert rebuilt == _cases()


def test_gold_and_distractor_values_do_not_collide():
    """A value cannot be both required and forbidden in the same case."""
    for case in _cases():
        forbidden = [d["value"] for d in case["must_not_extract"]]
        for gold in case["gold_variables"]:
            for value in forbidden:
                assert not benchmark._close(gold["mean"], value), \
                    f"{case['case_id']}: gold mean {gold['mean']} is also a distractor"
        for gold in case["gold_correlations"]:
            for value in forbidden:
                assert not benchmark._close(gold["correlation"], value), \
                    f"{case['case_id']}: gold correlation is also a distractor"


@pytest.mark.skipif(not CACHE_DIR.exists(), reason="cached real papers are gitignored")
def test_distractors_actually_appear_in_their_case_text():
    """A 'must not extract' value the model could never see is not a real test."""
    for case in _cases():
        numbers = benchmark._numbers_in_text(case["text"])
        for entry in case["must_not_extract"]:
            assert any(benchmark._close(entry["value"], n) for n in numbers), \
                f"{case['case_id']}: distractor {entry['value']} is absent from the text"


@pytest.mark.skipif(not CACHE_DIR.exists(), reason="cached real papers are gitignored")
def test_gold_values_appear_in_their_case_text():
    for case in _cases():
        numbers = benchmark._numbers_in_text(case["text"])
        for gold in case["gold_variables"]:
            assert any(benchmark._close(gold["mean"], n) for n in numbers), \
                f"{case['case_id']}: gold mean {gold['mean']} is absent from the text"


# --------------------------------------------------------------------------
# Number parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("mean 3,410 g", 3410.0),
    ("1.2e-3 mol/L", 0.0012),
    ("2.5E+04 counts", 25000.0),
    ("6.02 x 10^5 per mL", 602000.0),
    ("438.3 ± 35.3min/wk", 35.3),
    ("201.6 (26.3)b", 26.3),
])
def test_numbers_in_text_parses_real_world_forms(text, expected):
    numbers = benchmark._numbers_in_text(text)
    assert any(benchmark._close(expected, n) for n in numbers), numbers


def test_range_hyphen_is_not_read_as_a_minus_sign():
    numbers = benchmark._numbers_in_text("95% CI 1.08-1.87")
    assert any(benchmark._close(1.08, n) for n in numbers)
    assert any(benchmark._close(1.87, n) for n in numbers)
    assert not any(benchmark._close(-1.87, n) for n in numbers)


def test_genuine_negatives_still_parse():
    numbers = benchmark._numbers_in_text("the correlation was r = -0.30")
    assert any(benchmark._close(-0.30, n) for n in numbers)


def test_percentages_yield_both_readings():
    numbers = benchmark._numbers_in_text("111 (77.6%)")
    assert any(benchmark._close(77.6, n) for n in numbers)
    assert any(benchmark._close(0.776, n) for n in numbers)


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------

def test_perfect_extraction_scores_f1_one():
    case = _case(gold_variables=[_gold("bmi", 27.3, 4.9)], text="mean BMI 27.3 (4.9)")
    score = benchmark._score_challenge_case({"bmi": _var("bmi", 27.3, 4.9)}, {}, case)
    assert score["tp"] == 1 and score["fp"] == 0 and score["fn"] == 0
    assert score["f1"] == pytest.approx(1.0)
    assert score["name_accuracy"] == pytest.approx(1.0)


def test_regex_style_names_still_match_by_value():
    """var_N naming must not be penalised as a miss — only as low name_accuracy."""
    case = _case(gold_variables=[_gold("bmi", 27.3, 4.9)], text="mean BMI 27.3 (4.9)")
    score = benchmark._score_challenge_case({"var_1": _var("var_1", 27.3, 4.9)}, {}, case)
    assert score["tp"] == 1
    assert score["f1"] == pytest.approx(1.0)
    assert score["name_accuracy"] == pytest.approx(0.0)


def test_alias_counts_as_a_name_hit():
    case = _case(gold_variables=[_gold("social_support", 30.83, 7.03, aliases=("DUFSSQ-8",))],
                 text="DUFSSQ-8 30.83 (7.03)")
    score = benchmark._score_challenge_case(
        {"DUFSSQ 8": _var("DUFSSQ 8", 30.83, 7.03)}, {}, case
    )
    assert score["name_accuracy"] == pytest.approx(1.0)


def test_extracting_a_distractor_is_a_context_error():
    case = _case(
        gold_variables=[_gold("ferritin", 88.4, 21.7)],
        must_not_extract=[{"value": 0.62, "reason": "p-value"}],
        text="ferritin 88.4 (21.7), p = 0.62",
    )
    score = benchmark._score_challenge_case({"p": _var("p", 0.62, 1.0)}, {}, case)
    assert score["fp_distractor"] == 1
    assert score["fp_fabricated"] == 0
    assert score["context_error_rate"] == pytest.approx(1.0)


def test_inventing_a_value_is_a_hallucination():
    case = _case(gold_variables=[_gold("ferritin", 88.4, 21.7)], text="ferritin 88.4 (21.7)")
    score = benchmark._score_challenge_case({"x": _var("x", 512.7, 3.3)}, {}, case)
    assert score["fp_fabricated"] == 1
    assert score["fp_distractor"] == 0
    assert score["hallucination_rate"] == pytest.approx(1.0)


def test_value_present_in_text_but_unannotated_is_fp_other():
    case = _case(gold_variables=[_gold("a", 10.0, 2.0)], text="a 10.0 (2.0) and 99.5 (1.0)")
    score = benchmark._score_challenge_case({"b": _var("b", 99.5, 1.0)}, {}, case)
    assert score["fp_other"] == 1
    assert score["fp_fabricated"] == 0


def test_mixing_populations_is_a_conflation_error():
    case = _case(
        gold_variables=[_gold("age", 38.0, 7.2, group="age"),
                        _gold("age", 37.9, 6.9, group="age")],
        text="Age 38.0 (7.2) vs 37.9 (6.9)",
    )
    score = benchmark._score_challenge_case({"age": _var("age", 38.0, 6.9)}, {}, case)
    assert score["fp_conflated"] == 1
    assert score["tp"] == 0


def test_reporting_one_population_satisfies_the_group():
    case = _case(
        gold_variables=[_gold("age", 38.0, 7.2, group="age"),
                        _gold("age", 37.9, 6.9, group="age")],
        text="Age 38.0 (7.2) vs 37.9 (6.9)",
    )
    score = benchmark._score_challenge_case({"age": _var("age", 38.0, 7.2)}, {}, case)
    assert score["n_gold_groups"] == 1
    assert score["recall"] == pytest.approx(1.0)
    assert score["f1"] == pytest.approx(1.0)


def test_correct_abstention_scores_f1_one():
    case = _case(must_not_extract=[{"value": 0.14, "reason": "trend only"}],
                 text="a trend for a correlation (r=0.14, p=0.07)")
    score = benchmark._score_challenge_case({}, {}, case)
    assert score["correct_abstention"] is True
    assert score["f1"] == pytest.approx(1.0)
    assert score["fp"] == 0


def test_speaking_when_it_should_abstain_is_penalised():
    case = _case(must_not_extract=[{"value": 0.14, "reason": "trend only"}],
                 text="a trend for a correlation (r=0.14, p=0.07)")
    corr = CorrelationParams(var1="bmi", var2="wbc", correlation=0.14, direction="positive")
    score = benchmark._score_challenge_case({}, {"bmi__wbc": corr}, case)
    assert score["fp_distractor"] == 1
    assert score["precision"] == pytest.approx(0.0)


def test_silence_when_gold_exists_scores_zero_f1_without_false_positives():
    case = _case(gold_variables=[_gold("a", 10.0, 2.0)], text="a 10.0 (2.0)")
    score = benchmark._score_challenge_case({}, {}, case)
    assert score["precision"] == pytest.approx(1.0)  # nothing emitted, nothing wrong
    assert score["recall"] == pytest.approx(0.0)
    assert score["f1"] == pytest.approx(0.0)
    assert score["correct_abstention"] is False


def test_correlations_are_scored():
    case = _case(gold_correlations=[{"var1": "a", "var2": "b", "correlation": 0.58,
                                     "variant_group": "a__b"}],
                 text="r = 0.58")
    corr = CorrelationParams(var1="a", var2="b", correlation=0.58, direction="positive")
    score = benchmark._score_challenge_case({}, {"a__b": corr}, case)
    assert score["tp"] == 1 and score["f1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Prompt plumbing
# --------------------------------------------------------------------------

def test_prompt_schema_parity():
    """Caveman and natural-language prompts must request the same schema.

    A verbose baseline that asks for fewer fields is a strawman: it inflates the
    measured token reduction and invalidates the quality comparison.
    """
    from agentdataset.core.extractor import NATURAL_LANGUAGE_PROMPT
    caveman = CAVEMAN_PROMPT.lower()
    verbose = NATURAL_LANGUAGE_PROMPT.lower()
    for token in ["normal", "uniform", "gamma", "categorical", "categories",
                  "variables", "correlations", "var1", "var2", "correlation",
                  "direction", "positive", "negative", "mean", "std", "min", "max",
                  "distribution", "json"]:
        assert token in caveman, f"{token!r} missing from CAVEMAN_PROMPT"
        assert token in verbose, f"{token!r} missing from NATURAL_LANGUAGE_PROMPT"


def test_caveman_prompt_is_shorter_than_the_natural_language_prompt():
    from agentdataset.core.extractor import NATURAL_LANGUAGE_PROMPT
    assert len(CAVEMAN_PROMPT) < len(NATURAL_LANGUAGE_PROMPT)
    # The benchmark alias must keep pointing at the same text, or the recorded
    # token-savings comparison silently changes meaning.
    assert benchmark._VERBOSE_PROMPT is NATURAL_LANGUAGE_PROMPT


def test_production_prompt_is_the_natural_language_one():
    """Switched on the strength of the §9.6 ablation: the Caveman prompt scored
    F1 .710 against .778 (paired p=.0001) for a 3.5% cost saving."""
    from agentdataset.core.extractor import EXTRACTION_PROMPT, NATURAL_LANGUAGE_PROMPT
    assert EXTRACTION_PROMPT is NATURAL_LANGUAGE_PROMPT


@patch("agentdataset.core.extractor.completion")
def test_extract_with_llm_defaults_to_the_production_prompt(mock_completion):
    from agentdataset.core.extractor import EXTRACTION_PROMPT
    response = MagicMock()
    response.choices[0].message.content = json.dumps({"variables": {}, "correlations": {}})
    mock_completion.return_value = response

    Extractor(model="m", api_key="sk-test")._extract_with_llm("text")
    messages = mock_completion.call_args.kwargs["messages"]
    assert messages[0]["content"] == EXTRACTION_PROMPT


@patch("agentdataset.core.extractor.completion")
def test_extract_with_llm_honours_prompt_and_model_overrides(mock_completion):
    response = MagicMock()
    response.choices[0].message.content = json.dumps({"variables": {}, "correlations": {}})
    mock_completion.return_value = response

    extractor = Extractor(model="default-model", api_key="sk-test")
    extractor._extract_with_llm(
        "text", system_prompt="OVERRIDE", model="openrouter/other/model",
        num_retries=0, purpose="challenge:test",
    )
    kwargs = mock_completion.call_args.kwargs
    assert kwargs["messages"][0]["content"] == "OVERRIDE"
    assert kwargs["model"] == "openrouter/other/model"
    assert kwargs["num_retries"] == 0
    assert kwargs["response_format"] == {"type": "json_object"}  # production setting
    assert extractor.call_history[-1]["purpose"] == "challenge:test"
    assert extractor.call_history[-1]["model"] == "openrouter/other/model"


@pytest.mark.parametrize("wrapped,expected", [
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('{"a": 1}', '{"a": 1}'),
])
def test_strip_json_fence(wrapped, expected):
    from agentdataset.core.extractor import strip_json_fence
    assert strip_json_fence(wrapped) == expected


@patch("agentdataset.core.extractor.completion")
def test_fenced_json_still_parses_and_is_flagged(mock_completion):
    """Anthropic via OpenRouter ignores response_format and fences its JSON.

    Without the strip, the production parser fails and extract_parameters
    silently falls back to regex for every such call.
    """
    response = MagicMock()
    response.choices[0].message.content = (
        '```json\n{"variables": {"bmi": {"distribution": "normal", '
        '"mean": 27.3, "std": 4.9}}, "correlations": {}}\n```'
    )
    mock_completion.return_value = response

    extractor = Extractor(model="m", api_key="sk-test")
    data = extractor._extract_with_llm("text")
    assert data["variables"]["bmi"]["mean"] == 27.3
    assert extractor.call_history[-1]["fence_stripped"] is True


@patch("agentdataset.core.extractor.completion")
def test_unfenced_json_is_not_flagged(mock_completion):
    response = MagicMock()
    response.choices[0].message.content = '{"variables": {}, "correlations": {}}'
    mock_completion.return_value = response

    extractor = Extractor(model="m", api_key="sk-test")
    extractor._extract_with_llm("text")
    assert extractor.call_history[-1]["fence_stripped"] is False


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def _mock_llm_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = json.dumps(payload)
    response.usage.prompt_tokens = 200
    response.usage.completion_tokens = 40
    response.usage.total_tokens = 240
    return response


@patch("agentdataset.core.extractor.completion")
def test_run_extraction_challenge_produces_all_arms(mock_completion):
    mock_completion.return_value = _mock_llm_response({
        "variables": {"bmi": {"distribution": "normal", "mean": 27.3, "std": 4.9}},
        "correlations": {},
    })
    cases = [_case(gold_variables=[_gold("bmi", 27.3, 4.9)],
                   text="In this constructed text, mean BMI was 27.3 (SD 4.9).")]
    cases[0]["case_id"] = "t1"
    models = [{"name": "OpenAI", "model": "openrouter/openai/gpt-4o",
               "key_env": "OPENROUTER_API_KEY"}]

    df = benchmark.run_extraction_challenge(
        Extractor(model=models[0]["model"], api_key="sk-test"),
        challenge_set=cases, models=models, n_llm_repeats=2,
    )
    assert set(df["arm"]) == {"regex", "caveman", "natural_language"}
    # 1 regex + 2 arms x 1 model x 2 reps
    assert len(df) == 5
    assert not df["api_error"].any()
    llm = df[df["arm"] != "regex"]
    assert (llm["f1"] == 1.0).all()
    assert (llm["prompt_tokens"] == 200).all()


@patch("agentdataset.core.extractor.completion")
def test_api_errors_are_recorded_not_dropped(mock_completion):
    mock_completion.side_effect = RuntimeError("rate limited")
    cases = [_case(gold_variables=[_gold("bmi", 27.3, 4.9)],
                   text="In this constructed text, mean BMI was 27.3 (SD 4.9).")]
    cases[0]["case_id"] = "t1"
    models = [{"name": "OpenAI", "model": "openrouter/openai/gpt-4o",
               "key_env": "OPENROUTER_API_KEY"}]

    df = benchmark.run_extraction_challenge(
        Extractor(model=models[0]["model"], api_key="sk-test"),
        challenge_set=cases, models=models, n_llm_repeats=1,
    )
    failed = df[df["api_error"]]
    assert len(failed) == 2, "one failure row per LLM arm, none silently dropped"
    assert set(failed["error_type"]) == {"RuntimeError"}
    assert failed["f1"].isna().all()


@patch("agentdataset.core.extractor.completion")
def test_non_json_output_is_malformed_not_an_api_error(mock_completion):
    """A model emitting prose is a quality failure, not an infrastructure failure."""
    response = MagicMock()
    response.choices[0].message.content = "Sure! Here are the statistics you asked for."
    mock_completion.return_value = response
    cases = [_case(gold_variables=[_gold("bmi", 27.3, 4.9)],
                   text="In this constructed text, mean BMI was 27.3 (SD 4.9).")]
    cases[0]["case_id"] = "t1"
    models = [{"name": "OpenAI", "model": "openrouter/openai/gpt-4o",
               "key_env": "OPENROUTER_API_KEY"}]

    df = benchmark.run_extraction_challenge(
        Extractor(model=models[0]["model"], api_key="sk-test"),
        challenge_set=cases, models=models, n_llm_repeats=1,
    )
    llm = df[df["arm"] != "regex"]
    assert llm["malformed"].all()
    assert not llm["api_error"].any()
    assert (llm["f1"] == 0.0).all()


@patch("agentdataset.core.extractor.completion")
def test_schema_violation_counts_as_malformed(mock_completion):
    mock_completion.return_value = _mock_llm_response({"result": "no schema keys here"})
    cases = [_case(gold_variables=[_gold("bmi", 27.3, 4.9)],
                   text="In this constructed text, mean BMI was 27.3 (SD 4.9).")]
    cases[0]["case_id"] = "t1"
    models = [{"name": "OpenAI", "model": "openrouter/openai/gpt-4o",
               "key_env": "OPENROUTER_API_KEY"}]

    df = benchmark.run_extraction_challenge(
        Extractor(model=models[0]["model"], api_key="sk-test"),
        challenge_set=cases, models=models, n_llm_repeats=1,
    )
    llm = df[df["arm"] != "regex"]
    assert llm["malformed"].all()
    assert (llm["f1"] == 0.0).all()


def test_regex_only_run_needs_no_api_key():
    cases = benchmark.load_challenge_set(CHALLENGE_SET_PATH)
    df = benchmark.run_extraction_challenge(
        Extractor(model="openrouter/openai/gpt-4o", env_var="AGENTDATASET_ABSENT_KEY"),
        challenge_set=cases, arms=(),
    )
    assert set(df["arm"]) == {"regex"}
    assert len(df) == len(cases)
    assert (df["cost_usd"] == 0.0).all()
    assert (df["prompt_tokens"] == 0).all()


def test_load_challenge_set_filters_by_category():
    cases = benchmark.load_challenge_set(CHALLENGE_SET_PATH, categories=["negated_finding"])
    assert cases and all(c["category"] == "negated_finding" for c in cases)
    with pytest.raises(ValueError):
        benchmark.load_challenge_set(CHALLENGE_SET_PATH, categories=["no_such_category"])


def test_legacy_ablation_signature_still_works():
    """run_ablation_study() must keep producing ablation_extraction_method.csv."""
    extractor = Extractor(model="openrouter/openai/gpt-4o", env_var="AGENTDATASET_ABSENT_KEY")
    df = benchmark.run_ablation_extraction_method(extractor, n_llm_repeats=0)
    assert isinstance(df, pd.DataFrame)
    assert set(df["method"]) == {"regex"}
    assert "success_rate" in df.columns

# --------------------------------------------------------------------------
# Five-lab harness (docs/experiment.md): additive only, no frozen behavior moves
# --------------------------------------------------------------------------

def test_fivelab_providers_shape():
    assert len(benchmark._FIVELAB_PROVIDERS) == 5
    names = [p["name"] for p in benchmark._FIVELAB_PROVIDERS]
    assert names == ["DeepSeek", "GLM", "GPT", "Claude", "Gemini"]
    models = [p["model"] for p in benchmark._FIVELAB_PROVIDERS]
    assert len(set(models)) == 5
    assert all(p["key_env"] == "OPENROUTER_API_KEY" for p in benchmark._FIVELAB_PROVIDERS)
    # Legacy list untouched: frozen cost/latency artifact keeps its provider set.
    assert [p["name"] for p in benchmark._COST_LATENCY_PROVIDERS] == ["OpenAI", "Anthropic", "Google"]
    assert len(benchmark._ALL_PROVIDERS) == 8


def test_ablation_tagged_filename():
    from analysis.extraction_ablation import tagged_filename
    assert tagged_filename("extraction_challenge_raw", "") == "extraction_challenge_raw"
    assert tagged_filename("extraction_challenge_raw", "fivelab_gpt") == "extraction_challenge_raw_fivelab_gpt"


def test_real_paper_tagged_filename():
    from analysis.real_paper_eval import _tagged
    assert _tagged("real_paper_extraction.csv", "") == "real_paper_extraction.csv"
    assert _tagged("real_paper_extraction.csv", "fivelab_gpt") == "real_paper_extraction_fivelab_gpt.csv"
