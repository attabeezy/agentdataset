"""Noise-pivot sensitivity study (§9.7, P1).

Varies every knob of the noise-pivot schedule on a predeclared grid and compares
the shipped schedule against fixed-noise, random-search, and no-loop baselines.
Answers three questions the reviewer raised:

  1. Are the shipped defaults empirically justified, or arbitrary?
  2. Does the adaptive schedule beat trivial alternatives at equal budget?
  3. Why is the 10-iteration arm so much noisier than the 5-iteration arm?

Hyperparameters are selected on a *tune* subset of sources and the chosen
configuration is re-reported on a held-out *report* subset, so the answer to
"were these tuned on the same data you report on?" is recorded rather than
dodged.

No API calls: every source spec comes from embedded ground truth or from the
annotated real-paper gold table.

Usage (PowerShell):
  uv run python analysis/noise_pivot_ablation.py --quick
  uv run python analysis/noise_pivot_ablation.py --seeds 20
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark  # noqa: E402
from agentdataset.core.orchestrator import (  # noqa: E402
    DEFAULT_PIVOT,
    NoisePivotConfig,
    next_noise,
)
from agentdataset.core.synthesizer import Synthesizer  # noqa: E402
from agentdataset.core.validator import Validator  # noqa: E402
from agentdataset.evaluation import (  # noqa: E402
    RAW_RESULT_COLUMNS,
    collect_provenance,
    summarize_long_results,
    validate_raw_results,
    write_json,
)

RESULTS_DIR = ROOT / "results"
EXPERIMENT_ID = "noise_pivot_ablation"

# Rows are 10k by default in production; 2k keeps the grid tractable while
# preserving the ranking between configurations (fidelity is a distributional
# score, not a small-sample statistic).
N_ROWS = 2000

# --- The predeclared grid -----------------------------------------------------
# One-factor-at-a-time around the shipped configuration, plus a full
# patience x iteration-budget grid. A full factorial over all six factors is
# ~4k cells before seeds and answers a question nobody asked; OFAT answers the
# question that was actually asked, which is whether each shipped default is
# locally justified.

DEFAULT_ITERATIONS = 5

OFAT_LEVELS = {
    "patience": [1, 2, 3, 4],
    "initial_noise": [0.05, 0.1, 0.2, 0.4],
    "explore_mult": [1.0, 1.05, 1.1, 1.25, 1.5],
    "exploit_factor": [0.25, 0.5, 0.75, 0.9],
    "reset_multiple": [2, 3, 4, None],
}

ITERATION_LEVELS = [0, 1, 2, 3, 5, 10, 20]
GRID_PATIENCE = [1, 2, 3, 4]
GRID_ITERATIONS = [1, 2, 3, 5, 10, 20]
FIXED_NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.5]

QUICK_SEEDS = 3
QUICK_ITERATIONS = [0, 3, 5]


def fixed_noise_config(level: float) -> NoisePivotConfig:
    """A non-adaptive schedule, expressed in the same config object.

    explore and exploit both multiply by 1 and the reset branch is disabled, so
    the noise never moves. This deliberately runs through the same orchestrator
    code path as the adaptive arm — a separately written baseline loop would
    make any difference between the arms partly an artefact of the reimplementation.
    """
    return NoisePivotConfig(
        initial_noise=level,
        explore_mult=1.0,
        exploit_factor=1.0,
        reset_multiple=None,
        min_noise=min(level, 0.01),
        max_noise=max(level, 2.0),
    )


# --- Sources ------------------------------------------------------------------


def load_sources() -> list[dict]:
    """Templated ground-truth specs plus real-paper gold specs.

    The tune/report split is by position and is fixed here, in code, before any
    result is looked at.
    """
    sources: list[dict] = []

    for index, text in enumerate(benchmark._build_ablation_texts()):
        sources.append({
            "source_id": text["id"],
            "source_class": "templated",
            "extraction_method": "ground_truth",
            "params": benchmark._params_from_ground_truth(text, "noise_pivot_ablation"),
            "split": "tune" if index < 4 else "report",
        })

    gold_path = ROOT / "analysis" / "real_paper_ground_truth.csv"
    if gold_path.exists():
        from analysis.real_paper_eval import gold_to_parameters

        gold = pd.read_csv(gold_path)
        for index, source_id in enumerate(sorted(gold.source_id.unique())):
            params = gold_to_parameters(source_id, gold)
            if len(params.variables) < 2:
                # A single-variable spec cannot exercise the correlation term of
                # the fidelity score, so it cannot discriminate between schedules.
                continue
            sources.append({
                "source_id": source_id,
                "source_class": "gold",
                "extraction_method": "gold",
                "params": params,
                "split": "tune" if index % 2 else "report",
            })

    return sources


# --- Trial execution ----------------------------------------------------------


def run_pivot_trial(
    params, iterations: int, cfg: NoisePivotConfig, seed: int
) -> dict:
    """Run the ratchet loop under `cfg` and return per-trial metrics.

    Mirrors Orchestrator.run_optimization_loop exactly (see the parity test in
    tests/test_noise_pivot_ablation.py) but without session I/O, which dominates
    runtime at this cell count and is irrelevant to the schedule's behaviour.
    """
    synthesizer = Synthesizer(n_rows=N_ROWS, seed=seed)
    validator = Validator()

    best_score = -1.0
    noise = cfg.initial_noise
    streak = 0
    trace = []
    started = time.perf_counter()

    # iterations=0 is the no-loop baseline: one synthesis at the initial noise,
    # no ratchet, which is what a user gets without the optimizer at all.
    for i in range(max(iterations, 1)):
        df = synthesizer.synthesize(params, noise_level=noise)
        report = validator.validate(df, params)
        improved = report.overall_score > best_score
        row = {
            "iteration": i + 1,
            "noise_level": noise,
            "overall_score": report.overall_score,
            "corr_score": report.corr_score,
            "improved": improved,
            "phase": "keep",
        }
        if improved:
            best_score = report.overall_score
            streak = 0
        else:
            streak += 1
            noise, phase = next_noise(streak, noise, cfg)
            row["phase"] = phase
        trace.append(row)

    runtime = time.perf_counter() - started
    scores = [r["overall_score"] for r in trace]
    phases = [r["phase"] for r in trace]

    return {
        "best_score": best_score,
        "final_score": scores[-1],
        # 1-indexed iteration that first attained the best score.
        "iterations_to_best": int(np.argmax(scores)) + 1,
        "runtime_s": runtime,
        "reset_count": phases.count("reset"),
        "explore_count": phases.count("explore"),
        "exploit_count": phases.count("exploit"),
        "n_improvements": sum(r["improved"] for r in trace),
        "terminal_noise": trace[-1]["noise_level"],
        "mean_noise": float(np.mean([r["noise_level"] for r in trace])),
        "trace": trace,
    }


def run_random_search_trial(params, iterations: int, seed: int) -> dict:
    """Budget-matched random search: noise ~ LogUniform(MIN, MAX) each step."""
    synthesizer = Synthesizer(n_rows=N_ROWS, seed=seed)
    validator = Validator()
    rng = np.random.default_rng(seed)

    best_score = -1.0
    trace = []
    started = time.perf_counter()

    for i in range(max(iterations, 1)):
        noise = float(
            np.exp(rng.uniform(np.log(DEFAULT_PIVOT.min_noise), np.log(DEFAULT_PIVOT.max_noise)))
        )
        df = synthesizer.synthesize(params, noise_level=noise)
        report = validator.validate(df, params)
        improved = report.overall_score > best_score
        if improved:
            best_score = report.overall_score
        trace.append({
            "iteration": i + 1,
            "noise_level": noise,
            "overall_score": report.overall_score,
            "corr_score": report.corr_score,
            "improved": improved,
            "phase": "random",
        })

    runtime = time.perf_counter() - started
    scores = [r["overall_score"] for r in trace]
    return {
        "best_score": best_score,
        "final_score": scores[-1],
        "iterations_to_best": int(np.argmax(scores)) + 1,
        "runtime_s": runtime,
        "reset_count": 0,
        "explore_count": 0,
        "exploit_count": 0,
        "n_improvements": sum(r["improved"] for r in trace),
        "terminal_noise": trace[-1]["noise_level"],
        "mean_noise": float(np.mean([r["noise_level"] for r in trace])),
        "trace": trace,
    }


# --- Grid construction --------------------------------------------------------


def build_conditions(quick: bool) -> list[dict]:
    """Every (label, family, factor, level, iterations, config) cell to run."""
    iteration_levels = QUICK_ITERATIONS if quick else ITERATION_LEVELS
    conditions: list[dict] = []

    def add(label, family, factor, level, iterations, cfg, kind="pivot"):
        conditions.append({
            "condition": label,
            "family": family,
            "factor": factor,
            "level": str(level),
            "iterations": iterations,
            "config": cfg,
            "kind": kind,
        })

    # The shipped configuration, at the shipped budget.
    add("incumbent", "incumbent", "none", "default", DEFAULT_ITERATIONS, DEFAULT_PIVOT)

    # Iteration budget, holding the schedule fixed.
    for iterations in iteration_levels:
        add(f"iterations={iterations}", "iterations", "iterations", iterations,
            iterations, DEFAULT_PIVOT)

    # One factor at a time, at the shipped budget.
    if not quick:
        for factor, levels in OFAT_LEVELS.items():
            for level in levels:
                cfg = NoisePivotConfig(**{factor: level})
                add(f"{factor}={level}", "ofat", factor, level, DEFAULT_ITERATIONS, cfg)

        # patience x budget, the two factors §9.7 names explicitly.
        for patience, iterations in itertools.product(GRID_PATIENCE, GRID_ITERATIONS):
            add(f"grid:patience={patience},iterations={iterations}", "grid",
                "patience_x_iterations", f"{patience}x{iterations}", iterations,
                NoisePivotConfig(patience=patience))

    # Baselines, budget-matched by identical synthesize+validate call counts.
    for level in (FIXED_NOISE_LEVELS if not quick else [0.1]):
        for iterations in iteration_levels:
            add(f"fixed_noise={level},iterations={iterations}", "baseline_fixed",
                "fixed_noise", level, iterations, fixed_noise_config(level))

    for iterations in iteration_levels:
        add(f"random_search,iterations={iterations}", "baseline_random",
            "random_search", "loguniform", iterations, None, kind="random")

    return conditions


# --- Main ---------------------------------------------------------------------


def main(seeds: int, quick: bool) -> None:
    sources = load_sources()
    conditions = build_conditions(quick)
    seed_list = list(range(seeds))

    total = len(sources) * len(conditions) * len(seed_list)
    print(
        f"{len(sources)} sources x {len(conditions)} conditions x {len(seed_list)} "
        f"seeds = {total} trials"
    )

    # Provenance must be captured before anything is written, otherwise
    # git_dirty is structurally always True.
    provenance = collect_provenance(
        command=f"uv run python analysis/noise_pivot_ablation.py --seeds {seeds}"
        + (" --quick" if quick else ""),
        seeds=seed_list,
    )

    raw_rows: list[dict] = []
    trace_rows: list[dict] = []
    done = 0

    for source in sources:
        for condition in conditions:
            for seed in seed_list:
                base = {
                    "experiment_id": EXPERIMENT_ID,
                    "experiment_type": "noise_pivot_sensitivity",
                    "source_id": source["source_id"],
                    "dataset": source["source_class"],
                    "task_type": "synthesis",
                    "condition": condition["condition"],
                    "model_family": "none",
                    "synthesizer": "agentdataset",
                    "seed": seed,
                    "split_seed": seed,
                    "synthesis_seed": seed,
                    "extraction_method": source["extraction_method"],
                    "provider": "none",
                    "model": "none",
                    "family": condition["family"],
                    "factor": condition["factor"],
                    "level": condition["level"],
                    "iterations": condition["iterations"],
                    "split": source["split"],
                }
                run_id = (
                    f"{EXPERIMENT_ID}:{source['source_id']}:"
                    f"{condition['condition']}:{seed}"
                )
                try:
                    if condition["kind"] == "random":
                        outcome = run_random_search_trial(
                            source["params"], condition["iterations"], seed
                        )
                    else:
                        outcome = run_pivot_trial(
                            source["params"],
                            condition["iterations"],
                            condition["config"],
                            seed,
                        )
                except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                    raw_rows.append({
                        **base, "run_id": run_id, "metric": "best_score",
                        "value": float("nan"), "status": "failed",
                        "failure_stage": f"{type(exc).__name__}: {exc}"[:200],
                    })
                    done += 1
                    continue

                for row in outcome.pop("trace"):
                    trace_rows.append({
                        "source_id": source["source_id"],
                        "condition": condition["condition"],
                        "family": condition["family"],
                        "iterations": condition["iterations"],
                        "seed": seed,
                        **row,
                    })

                for metric, value in outcome.items():
                    raw_rows.append({
                        **base,
                        "run_id": f"{run_id}:{metric}",
                        "metric": metric,
                        "value": float(value),
                        "status": "ok",
                        "failure_stage": "",
                    })
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{total} trials")

    raw = pd.DataFrame(raw_rows)
    for column in RAW_RESULT_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""
    validate_raw_results(raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # The grid produces ~209k raw rows and ~125k trace rows; uncompressed these
    # are ~62 MB and ~12 MB, an order of magnitude past anything else in
    # results/. pandas reads .csv.gz transparently, so store them compressed.
    raw.to_csv(RESULTS_DIR / f"{EXPERIMENT_ID}_raw.csv.gz", index=False)

    summary = summarize_long_results(
        raw, ["condition", "family", "factor", "level", "iterations", "split"]
    )
    summary.to_csv(RESULTS_DIR / f"{EXPERIMENT_ID}_summary.csv", index=False)

    traces = pd.DataFrame(trace_rows)
    traces.to_csv(RESULTS_DIR / "noise_pivot_traces.csv.gz", index=False)

    write_json(
        RESULTS_DIR / f"{EXPERIMENT_ID}_run_manifest.json",
        {
            **provenance,
            "n_sources": len(sources),
            "n_conditions": len(conditions),
            "n_seeds": len(seed_list),
            "n_trials": total,
            "n_rows_per_dataset": N_ROWS,
            "quick_mode": quick,
            "tune_sources": [s["source_id"] for s in sources if s["split"] == "tune"],
            "report_sources": [s["source_id"] for s in sources if s["split"] == "report"],
            "ofat_levels": {k: [str(v) for v in vs] for k, vs in OFAT_LEVELS.items()},
            "iteration_levels": ITERATION_LEVELS,
            "fixed_noise_levels": FIXED_NOISE_LEVELS,
        },
    )
    print(f"Wrote {len(raw)} raw rows and {len(traces)} trace rows to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--quick", action="store_true", help="small smoke grid")
    args = parser.parse_args()
    main(seeds=QUICK_SEEDS if args.quick else args.seeds, quick=args.quick)
