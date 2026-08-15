"""Does the fidelity score detect wrong parameters? (§9.7 follow-up)

The fidelity score's KS term compares each synthesized column against the same
parametric CDF the synthesizer drew it from, using the same declared mean and
standard deviation. If the extracted parameters are wrong, the synthesizer
reproduces the wrong parameters faithfully and the KS term cannot notice.

This probe measures that directly. For each real dataset we build parameters
from the true data, corrupt them by a controlled amount, synthesize from the
corrupted parameters, and then record two things:

  reported_*  what AgentDataset reports, scoring the synthetic data against the
              corrupted parameters it believes -- i.e. what a user sees
  external_*  how far the synthetic data actually is from the real data, which
              is the thing the user cares about and the score never sees

If reported fidelity stays flat while the external divergence grows, the score
is a self-consistency check rather than a measure of source fidelity.

Usage (PowerShell):
  uv run python analysis/fidelity_validity_probe.py
  uv run python analysis/fidelity_validity_probe.py --seeds 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark  # noqa: E402
from agentdataset.core.synthesizer import Synthesizer  # noqa: E402
from agentdataset.core.validator import Validator  # noqa: E402
from agentdataset.evaluation import (  # noqa: E402
    RAW_RESULT_COLUMNS,
    collect_provenance,
    summarize_long_results,
    validate_raw_results,
    write_json,
)
from agentdataset.models.schemas import (  # noqa: E402
    CorrelationParams,
    MetaParams,
    Parameters,
    VariableParams,
)

RESULTS_DIR = ROOT / "results"
EXPERIMENT_ID = "fidelity_validity_probe"
N_ROWS = 2000

# Corruption magnitudes, in units of the variable's true standard deviation.
EPSILONS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def true_parameters(dataset: dict) -> tuple[Parameters, pd.DataFrame]:
    """Parameters computed from the real data -- the best case extraction could do."""
    df = dataset["df"]
    continuous = dataset["continuous"]
    real = df[continuous].astype(float)

    variables = {}
    for name in continuous:
        mean, std = float(real[name].mean()), float(real[name].std())
        variables[name] = VariableParams(
            name=name, distribution="normal", mean=mean, std=std,
            min=mean - 3 * std, max=mean + 3 * std,
        )

    correlations = {}
    for i, a in enumerate(continuous):
        for b in continuous[i + 1:]:
            value = float(real[a].corr(real[b]))
            correlations[f"{a}__{b}"] = CorrelationParams(
                var1=a, var2=b, correlation=value,
                direction="positive" if value >= 0 else "negative",
            )

    params = Parameters(
        variables=variables, correlations=correlations,
        meta=MetaParams(source=dataset["name"], extracted_at="probe",
                        extraction_method="true_from_data"),
    )
    return params, real


def corrupt(params: Parameters, kind: str, epsilon: float, rng) -> Parameters:
    """Return a copy of `params` with a controlled error introduced."""
    out = params.model_copy(deep=True)
    names = list(out.variables)

    if kind == "shift_mean":
        for name in names:
            v = out.variables[name]
            v.mean = v.mean + epsilon * v.std
    elif kind == "scale_std":
        for name in names:
            v = out.variables[name]
            v.std = v.std * (1.0 + epsilon)
    elif kind == "swap_stats":
        # Structural error: attach each variable's statistics to another
        # variable. epsilon controls only whether it happens at all.
        if epsilon > 0 and len(names) > 1:
            shuffled = names[1:] + names[:1]
            originals = {n: (out.variables[n].mean, out.variables[n].std) for n in names}
            for target, donor in zip(names, shuffled):
                out.variables[target].mean, out.variables[target].std = originals[donor]
    else:
        raise ValueError(kind)

    return out


def external_divergence(synthetic: pd.DataFrame, real: pd.DataFrame) -> dict:
    """How far the synthetic data actually is from the real data."""
    ks_stats, mean_errors, std_errors = [], [], []
    for name in real.columns:
        if name not in synthetic.columns:
            continue
        s = pd.to_numeric(synthetic[name], errors="coerce").dropna().to_numpy()
        r = real[name].to_numpy(dtype=float)
        if len(s) == 0:
            continue
        # Two-sample KS against the real empirical distribution -- the check the
        # internal score never performs.
        ks_stats.append(float(stats.ks_2samp(s, r).statistic))
        scale = r.std() or 1.0
        mean_errors.append(abs(s.mean() - r.mean()) / scale)
        std_errors.append(abs(s.std() - r.std()) / scale)

    return {
        "external_ks_vs_real": float(np.mean(ks_stats)) if ks_stats else float("nan"),
        "external_mean_error_sd": float(np.mean(mean_errors)) if mean_errors else float("nan"),
        "external_std_error_sd": float(np.mean(std_errors)) if std_errors else float("nan"),
    }


def main(seeds: int) -> None:
    provenance = collect_provenance(
        command=f"uv run python analysis/fidelity_validity_probe.py --seeds {seeds}",
        seeds=list(range(seeds)),
    )

    validator = Validator()
    raw_rows: list[dict] = []
    notes: list[str] = []

    for loader_name in ("_load_adult", "_load_pima", "_load_german_credit"):
        try:
            dataset = getattr(benchmark, loader_name)()
            params, real = true_parameters(dataset)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{loader_name}: {type(exc).__name__}: {exc}")
            continue

        print(f"{dataset['name']}: {len(real)} real rows, {len(params.variables)} continuous vars")

        for kind in ("shift_mean", "scale_std", "swap_stats"):
            epsilons = EPSILONS if kind != "swap_stats" else [0.0, 1.0]
            for epsilon in epsilons:
                for seed in range(seeds):
                    rng = np.random.default_rng(seed)
                    corrupted = corrupt(params, kind, epsilon, rng)
                    synthetic = Synthesizer(n_rows=N_ROWS, seed=seed).synthesize(
                        corrupted, noise_level=0.0
                    )
                    # What the system reports: scored against the parameters it
                    # believes, which are the corrupted ones.
                    report = validator.validate(synthetic, corrupted)
                    metrics = {
                        "reported_fidelity": report.overall_score,
                        "reported_ks": report.ks_score,
                        "reported_corr": report.corr_score,
                        "reported_bias": report.bias_score,
                        **external_divergence(synthetic, real),
                    }

                    base = {
                        "experiment_id": EXPERIMENT_ID,
                        "experiment_type": "fidelity_validity",
                        "source_id": dataset["name"],
                        "dataset": dataset["name"],
                        "task_type": "synthesis",
                        "condition": f"{kind}:{epsilon}",
                        "model_family": "none",
                        "synthesizer": "agentdataset",
                        "seed": seed,
                        "split_seed": seed,
                        "synthesis_seed": seed,
                        "extraction_method": "true_from_data",
                        "provider": "none",
                        "model": "none",
                        "corruption": kind,
                        "epsilon": epsilon,
                    }
                    for metric, value in metrics.items():
                        raw_rows.append({
                            **base,
                            "run_id": f"{EXPERIMENT_ID}:{dataset['name']}:{kind}:{epsilon}:{seed}:{metric}",
                            "metric": metric,
                            "value": float(value),
                            "status": "ok",
                            "failure_stage": "",
                        })

    raw = pd.DataFrame(raw_rows)
    for column in RAW_RESULT_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""
    validate_raw_results(raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RESULTS_DIR / f"{EXPERIMENT_ID}_raw.csv", index=False)
    summary = summarize_long_results(raw, ["dataset", "corruption", "epsilon"])
    summary.to_csv(RESULTS_DIR / f"{EXPERIMENT_ID}_summary.csv", index=False)

    write_json(
        RESULTS_DIR / f"{EXPERIMENT_ID}_run_manifest.json",
        {**provenance, "n_rows_per_dataset": N_ROWS, "epsilons": EPSILONS,
         "collection_notes": notes},
    )

    # --- the headline table --------------------------------------------------
    pivot = (
        raw.loc[raw.status.eq("ok")]
        .pivot_table(index=["corruption", "epsilon"], columns="metric",
                     values="value", aggfunc="mean")
    )
    columns = ["reported_fidelity", "reported_ks", "external_ks_vs_real",
               "external_mean_error_sd"]
    print("\n=== reported vs actual, averaged over datasets and seeds ===")
    print(pivot[columns].round(3).to_string())
    print(f"\nWrote {len(raw)} rows to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    main(seeds=args.seeds)
