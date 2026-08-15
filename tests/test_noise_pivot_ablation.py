"""Guards for the §9.7 noise-pivot sensitivity study."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import noise_pivot_ablation as npa  # noqa: E402
from agentdataset.core.orchestrator import (  # noqa: E402
    DEFAULT_PIVOT,
    NoisePivotConfig,
    Orchestrator,
)
from agentdataset.core.synthesizer import Synthesizer  # noqa: E402
from agentdataset.core.validator import Validator  # noqa: E402


def _spec():
    text = npa.benchmark._build_ablation_texts()[0]
    return npa.benchmark._params_from_ground_truth(text, "parity_test")


def test_trial_loop_matches_the_real_orchestrator(tmp_path):
    """The ablation reimplements the ratchet; it must not drift from production.

    If this fails, the study is measuring the ablation's loop rather than the
    optimizer the paper describes.
    """
    params = _spec()

    with patch("agentdataset.core.orchestrator.DiscoveryAgent"), \
         patch("agentdataset.core.orchestrator.Extractor"), \
         patch("agentdataset.core.orchestrator.Synthesizer"), \
         patch("agentdataset.core.orchestrator.Validator"):
        orc = Orchestrator(session_id="parity", base_dir=str(tmp_path))
    orc.synthesizer = Synthesizer(n_rows=npa.N_ROWS, seed=7)
    orc.validator = Validator()

    orchestrator_best, _ = orc.run_optimization_loop(params, iterations=6)
    trial = npa.run_pivot_trial(params, iterations=6, cfg=DEFAULT_PIVOT, seed=7)

    assert trial["best_score"] == pytest.approx(orchestrator_best)

    orchestrator_noise = [row["noise_level"] for row in orc.optimization_trace]
    trial_noise = [row["noise_level"] for row in trial["trace"]]
    assert trial_noise == pytest.approx(orchestrator_noise)

    orchestrator_phases = [row["phase"] for row in orc.optimization_trace]
    assert [row["phase"] for row in trial["trace"]] == orchestrator_phases


def test_fixed_noise_config_never_moves_the_noise():
    """The fixed-noise baseline must actually be fixed."""
    cfg = npa.fixed_noise_config(0.25)
    trial = npa.run_pivot_trial(_spec(), iterations=8, cfg=cfg, seed=1)

    noises = [row["noise_level"] for row in trial["trace"]]
    assert noises == pytest.approx([0.25] * len(noises))
    assert trial["reset_count"] == 0


def test_no_loop_baseline_runs_exactly_one_synthesis():
    trial = npa.run_pivot_trial(_spec(), iterations=0, cfg=DEFAULT_PIVOT, seed=1)
    assert len(trial["trace"]) == 1
    assert trial["best_score"] == pytest.approx(trial["final_score"])


def test_random_search_varies_noise_and_matches_budget():
    trial = npa.run_random_search_trial(_spec(), iterations=6, seed=3)
    noises = [row["noise_level"] for row in trial["trace"]]
    assert len(noises) == 6, "budget must match the pivot arm"
    assert len(set(noises)) == 6, "every step should redraw"
    assert all(DEFAULT_PIVOT.min_noise <= n <= DEFAULT_PIVOT.max_noise for n in noises)


def test_best_score_is_monotone_in_budget():
    """The ratchet only ever keeps improvements, so a longer budget cannot lose.

    This is the premise behind the 10-iteration variance analysis: any spread at
    a larger budget comes from the schedule, not from the ratchet regressing.
    """
    short = npa.run_pivot_trial(_spec(), iterations=3, cfg=DEFAULT_PIVOT, seed=5)
    long = npa.run_pivot_trial(_spec(), iterations=10, cfg=DEFAULT_PIVOT, seed=5)
    assert long["best_score"] >= short["best_score"] - 1e-9


def test_sources_carry_a_predeclared_tune_report_split():
    sources = npa.load_sources()
    splits = {s["split"] for s in sources}
    assert splits == {"tune", "report"}
    assert all(len(s["params"].variables) >= 2 for s in sources)


def test_condition_grid_covers_every_declared_factor():
    conditions = npa.build_conditions(quick=False)
    factors = {c["factor"] for c in conditions}
    for factor in npa.OFAT_LEVELS:
        assert factor in factors, f"{factor} missing from the grid"
    families = {c["family"] for c in conditions}
    assert {"baseline_fixed", "baseline_random", "grid", "ofat"} <= families


def test_incumbent_condition_is_the_shipped_default():
    """The 'incumbent' arm must be the configuration the code actually ships."""
    conditions = npa.build_conditions(quick=False)
    incumbent = next(c for c in conditions if c["condition"] == "incumbent")
    assert incumbent["config"] == NoisePivotConfig()
    assert incumbent["iterations"] == npa.DEFAULT_ITERATIONS
