"""Offline tests for the benchmark's literature-style source-text template.

The template feeds the LLM-only extractor, so these tests check the generated
prose (means/stds, category probabilities, correlation sentences) without any
network access or API keys.
"""

import numpy as np
import pandas as pd

from benchmark import _dataset_to_source_text


def _toy_dataset() -> dict:
    """Small in-memory dataset with a 3-category feature and a binary target."""
    rng = np.random.default_rng(0)
    n = 1000
    status = rng.choice(
        ["married", "never_married", "prev_married"], size=n, p=[0.5, 0.3, 0.2]
    )
    age = rng.normal(40, 10, n)
    outcome = np.where(rng.random(n) < 0.4, "yes", "no")
    df = pd.DataFrame({"age": age, "status": status, "outcome": outcome})
    return {
        "name": "toy",
        "domain": "test",
        "df": df,
        "continuous": ["age"],
        "categorical_features": ["status"],
        "target": "outcome",
    }


def test_source_text_reports_continuous_statistics():
    text = _dataset_to_source_text(_toy_dataset())
    assert "The variable age has mean" in text
    assert "and std" in text


def test_source_text_lists_every_category_and_target_value():
    text = _dataset_to_source_text(_toy_dataset())
    for label in ("'married'", "'never_married'", "'prev_married'", "'yes'", "'no'"):
        assert label in text
    assert "The categorical variable status takes value" in text
    assert "The categorical variable outcome takes value" in text


def test_source_text_includes_continuous_and_categorical_correlations():
    text = _dataset_to_source_text(_toy_dataset())
    assert "The correlation between age and outcome is" in text
    assert "The correlation between status and outcome is" in text


def test_ablation_resume_skips_completed_cells(tmp_path, monkeypatch):
    """A checkpointed cell must not be recomputed or overwritten."""
    import benchmark

    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path)
    tiny_text = {
        "id": "d0",
        "text": "mean 1 std 0.5; mean 2 std 0.5; correlation 0.1",
        "ground_truth": {"x": {"mean": 1.0, "std": 0.5}, "y": {"mean": 2.0, "std": 0.5}},
        "corr_pair": ("x", "y"),
        "corr_val": 0.1,
    }
    monkeypatch.setattr(benchmark, "_build_ablation_texts", lambda: [tiny_text])
    pd.DataFrame(
        [{"text_id": "d0", "seed": 0, "iterations": 0, "score": 99.0}]
    ).to_csv(tmp_path / "ablation_noise_pivot.csv", index=False)

    df = benchmark.run_ablation_noise_pivot(
        iterations_grid=(0,), n_seeds=1, resume=True
    )
    assert len(df) == 1
    assert df.iloc[0]["score"] == 99.0


def test_ablation_uses_a_seed_scoped_synthesizer_per_cell(tmp_path, monkeypatch):
    """Each cell must draw from its own seeded RNG, not a shared stream."""
    import benchmark

    monkeypatch.setattr(benchmark, "RESULTS_DIR", tmp_path)
    tiny_text = {
        "id": "d0",
        "text": "mean 1 std 0.5; mean 2 std 0.5; correlation 0.1",
        "ground_truth": {"x": {"mean": 1.0, "std": 0.5}, "y": {"mean": 2.0, "std": 0.5}},
        "corr_pair": ("x", "y"),
        "corr_val": 0.1,
    }
    monkeypatch.setattr(benchmark, "_build_ablation_texts", lambda: [tiny_text])

    seeds_seen = []

    class _FakeSynthesizer:
        def __init__(self, seed: int = 42, **_: object):
            seeds_seen.append(seed)

        def synthesize(self, params, noise_level=0.1):
            return pd.DataFrame({"x": [1.0], "y": [2.0]})

    class _FakeReport:
        overall_score = 99.0

    class _FakeValidator:
        def validate(self, df, params):
            return _FakeReport()

    monkeypatch.setattr(benchmark, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(benchmark, "Validator", _FakeValidator)

    df = benchmark.run_ablation_noise_pivot(
        iterations_grid=(0,), n_seeds=3, resume=False
    )
    assert seeds_seen == [0, 1, 2]
    assert list(df["seed"]) == [0, 1, 2]
