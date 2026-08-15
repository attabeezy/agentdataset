"""Turn the §9.7 noise-pivot grid into the numbers the paper reports.

Separate from the grid runner so the statistics can be re-derived without
re-running ~20k trials. Produces:

  results/noise_pivot_tests.csv        paired incumbent-vs-alternative tests
  results/noise_pivot_budget.csv       score and variance by iteration budget
  results/noise_pivot_variance.csv     the 10-iteration variance decomposition

Usage (PowerShell):
  uv run python analysis/noise_pivot_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RAW = RESULTS_DIR / "noise_pivot_ablation_raw.csv.gz"
TRACES = RESULTS_DIR / "noise_pivot_traces.csv.gz"

PRIMARY_METRIC = "best_score"


def load_metric(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Wide view of one metric, indexed by the pairing unit."""
    subset = raw.loc[raw.metric.eq(metric) & raw.status.eq("ok")]
    return subset.pivot_table(
        index=["source_id", "seed", "split"],
        columns="condition",
        values="value",
        aggfunc="first",
    )


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = (n - rank) * pvalues[index]
        running = max(running, value)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare every arm against the incumbent, paired on (source, seed).

    Tests are run on the tune split only. The report split is summarised
    separately so the held-out margin is never used to choose a default.
    """
    wide = load_metric(raw, PRIMARY_METRIC)
    if "incumbent" not in wide.columns:
        raise SystemExit("no incumbent arm in the raw results")

    meta = (
        raw[["condition", "family", "factor", "level", "iterations"]]
        .drop_duplicates("condition")
        .set_index("condition")
    )

    rows = []
    for split in ("tune", "report"):
        block = wide.xs(split, level="split")
        baseline = block["incumbent"]
        for condition in block.columns:
            if condition == "incumbent":
                continue
            paired = pd.concat(
                [baseline, block[condition]], axis=1, keys=["incumbent", "arm"]
            ).dropna()
            if len(paired) < 5:
                continue
            difference = paired["arm"] - paired["incumbent"]
            if np.allclose(difference, 0):
                pvalue, statistic = 1.0, 0.0
            else:
                statistic, pvalue = stats.wilcoxon(
                    paired["arm"], paired["incumbent"], zero_method="zsplit"
                )
            rows.append({
                "split": split,
                "condition": condition,
                "family": meta.loc[condition, "family"],
                "factor": meta.loc[condition, "factor"],
                "level": meta.loc[condition, "level"],
                "iterations": meta.loc[condition, "iterations"],
                "n_pairs": len(paired),
                "incumbent_mean": float(paired["incumbent"].mean()),
                "arm_mean": float(paired["arm"].mean()),
                "mean_delta": float(difference.mean()),
                "median_delta": float(difference.median()),
                # Paired Cohen's d.
                "effect_size": float(
                    difference.mean() / difference.std(ddof=1)
                ) if difference.std(ddof=1) > 0 else 0.0,
                "wilcoxon_stat": float(statistic),
                "p_value": float(pvalue),
            })

    frame = pd.DataFrame(rows)
    # Holm correction within each (split, factor) family, which is the unit the
    # paper draws conclusions over.
    frame["p_holm"] = np.nan
    for _, index in frame.groupby(["split", "factor"]).groups.items():
        frame.loc[index, "p_holm"] = holm(frame.loc[index, "p_value"].tolist())
    frame["significant_holm_05"] = frame["p_holm"] < 0.05
    return frame.sort_values(["split", "family", "factor", "level"])


def budget_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Score, spread, and cost as a function of the iteration budget."""
    subset = raw.loc[
        raw.status.eq("ok")
        & raw.family.isin(["iterations", "baseline_fixed", "baseline_random"])
    ]
    wanted = ["best_score", "final_score", "runtime_s", "reset_count",
              "iterations_to_best", "n_improvements"]
    subset = subset.loc[subset.metric.isin(wanted)]

    grouped = (
        subset.groupby(["family", "condition", "iterations", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return grouped.sort_values(["family", "iterations", "metric"])


def variance_analysis(raw: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    """Decompose where the spread at long budgets comes from.

    The ratchet is monotone, so best_score cannot get worse with more budget.
    Any *increase* in spread at 10 iterations must therefore come from the
    schedule visiting different noise levels across seeds, not from the
    optimizer regressing. These rows let the paper say which it is.
    """
    best = raw.loc[raw.metric.eq("best_score") & raw.status.eq("ok")]

    rows = []
    for (family, condition, iterations), group in best.groupby(
        ["family", "condition", "iterations"]
    ):
        # Spread across seeds, averaged over sources: the quantity a user feels
        # when they rerun the tool.
        per_source = group.groupby("source_id")["value"]
        rows.append({
            "family": family,
            "condition": condition,
            "iterations": iterations,
            "mean_best": float(group["value"].mean()),
            "across_seed_sd": float(per_source.std(ddof=1).mean()),
            "across_source_sd": float(per_source.mean().std(ddof=1)),
            "n": len(group),
        })

    frame = pd.DataFrame(rows)

    if not traces.empty:
        noise_stats = (
            traces.groupby(["condition", "iterations"])
            .agg(
                mean_noise=("noise_level", "mean"),
                max_noise=("noise_level", "max"),
                sd_noise=("noise_level", "std"),
                score_sd_within_run=("overall_score", "std"),
                reset_share=("phase", lambda s: float((s == "reset").mean())),
                explore_share=("phase", lambda s: float((s == "explore").mean())),
            )
            .reset_index()
        )
        frame = frame.merge(noise_stats, on=["condition", "iterations"], how="left")

    return frame.sort_values(["family", "iterations"])


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW}; run analysis/noise_pivot_ablation.py first")

    raw = pd.read_csv(RAW)
    traces = pd.read_csv(TRACES) if TRACES.exists() else pd.DataFrame()

    tests = paired_tests(raw)
    tests.to_csv(RESULTS_DIR / "noise_pivot_tests.csv", index=False)

    budget = budget_table(raw)
    budget.to_csv(RESULTS_DIR / "noise_pivot_budget.csv", index=False)

    variance = variance_analysis(raw, traces)
    variance.to_csv(RESULTS_DIR / "noise_pivot_variance.csv", index=False)

    # --- console summary, i.e. the numbers that go into the paper -------------
    print("\n=== Iteration budget (pivot schedule, best_score) ===")
    budget_best = budget.loc[
        budget.family.eq("iterations") & budget.metric.eq("best_score")
    ]
    print(budget_best[["iterations", "mean", "std", "count"]].to_string(index=False))

    print("\n=== Across-seed spread by budget ===")
    spread = variance.loc[variance.family.eq("iterations")]
    columns = [c for c in ["iterations", "mean_best", "across_seed_sd",
                           "mean_noise", "max_noise", "reset_share"]
               if c in spread.columns]
    print(spread[columns].to_string(index=False))

    print("\n=== Baselines at the shipped budget ===")
    base = variance.loc[variance.family.isin(["baseline_fixed", "baseline_random"])]
    print(
        base.loc[base.iterations.eq(5), ["condition", "mean_best", "across_seed_sd"]]
        .to_string(index=False)
    )

    print("\n=== Arms beating the incumbent on the tune split (Holm p<.05) ===")
    winners = tests.loc[
        tests.split.eq("tune") & tests.significant_holm_05 & (tests.mean_delta > 0)
    ]
    if winners.empty:
        print("none — the shipped configuration is not beaten by any single-factor change")
    else:
        print(
            winners[["condition", "mean_delta", "effect_size", "p_holm"]]
            .sort_values("mean_delta", ascending=False)
            .to_string(index=False)
        )

    print(f"\nWrote tests/budget/variance tables to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
