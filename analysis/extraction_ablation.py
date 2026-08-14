"""Extraction robustness and prompt ablation over the challenge set (P1 9.6).

Runs three arms over the same annotated challenge cases:

  caveman           — the Caveman Protocol system prompt (compressed)
  natural_language  — a verbose prompt requesting the identical schema; this is
                      the production prompt as of the §9.6 ablation
  regex             — the regex fallback, run once and reported separately

and reports accuracy, precision/recall/F1, malformed-output rate, hallucination
rate, tokens, latency and cost per arm, with multiple repetitions per LLM arm to
expose nondeterminism.

Every model is routed through OpenRouter, so one OPENROUTER_API_KEY covers all
vendors in the sweep.

Usage (PowerShell):
    .venv\\Scripts\\python.exe analysis\\extraction_ablation.py --dry-run
    .venv\\Scripts\\python.exe analysis\\extraction_ablation.py --models openai --reps 1
    .venv\\Scripts\\python.exe analysis\\extraction_ablation.py --reps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark  # noqa: E402
from agentdataset.core.extractor import Extractor  # noqa: E402
from agentdataset.evaluation import (  # noqa: E402
    collect_provenance,
    sha256_file,
    summarize_long_results,
    write_json,
)

RESULTS_DIR = ROOT / "results"
CHALLENGE_SET = ROOT / "analysis" / "extraction_challenge_set.json"
ENV_VAR = "OPENROUTER_API_KEY"

# Metrics melted into the shared long-format results schema for summarisation.
METRIC_COLUMNS = [
    "f1", "precision", "recall",
    "hallucination_rate", "context_error_rate", "conflation_rate",
    "malformed", "fence_stripped", "abstained", "correct_abstention",
    "n_extracted", "std_rel_err", "name_accuracy",
    "prompt_tokens", "completion_tokens", "cost_usd", "latency_s",
]


def load_env() -> None:
    """Load .env without printing or logging any secret."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    import os
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide per-extraction frame into the shared RAW_RESULT_COLUMNS schema."""
    frames = []
    for metric in METRIC_COLUMNS:
        if metric not in raw.columns:
            continue
        frame = pd.DataFrame({
            "experiment_id": "extraction_challenge",
            "run_id": raw["run_id"],
            "experiment_type": "ablation",
            "source_id": raw["case_id"],
            "dataset": raw["category"],
            "task_type": "extraction",
            "condition": raw["arm"],
            "model_family": raw["model"],
            "synthesizer": "not_applicable",
            "seed": raw["rep"],
            "split_seed": np.nan,
            "synthesis_seed": np.nan,
            "metric": metric,
            "value": pd.to_numeric(raw[metric], errors="coerce"),
            "status": np.where(raw["api_error"], "failed", "ok"),
            "failure_stage": raw["error_type"],
            "extraction_method": raw["arm"],
            "provider": raw["model"],
            "model": raw["model"],
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main(reps: int, model_names: list[str] | None, categories: list[str] | None,
         dry_run: bool) -> None:
    load_env()

    providers = benchmark._COST_LATENCY_PROVIDERS
    if model_names:
        wanted = {name.lower() for name in model_names}
        providers = [p for p in providers if p["name"].lower() in wanted]
        if not providers:
            available = ", ".join(p["name"] for p in benchmark._COST_LATENCY_PROVIDERS)
            raise SystemExit(f"No providers matched {sorted(wanted)}. Available: {available}")

    cases = benchmark.load_challenge_set(CHALLENGE_SET, categories=categories)
    extractor = Extractor(model=providers[0]["model"], env_var=ENV_VAR)

    # Captured BEFORE any output is written. Recording provenance afterwards
    # makes git_dirty structurally always True, because this run's own CSVs are
    # sitting in the working tree by then. Provenance must describe the code
    # state that produced the results, not the tree state after writing them.
    provenance = collect_provenance(
        "python analysis/extraction_ablation.py " + " ".join(sys.argv[1:]),
        seeds=list(range(reps)),
    )

    raw = benchmark.run_extraction_challenge(
        extractor,
        challenge_set=cases,
        models=providers,
        n_llm_repeats=reps,
        arms=() if dry_run else benchmark.LLM_ARMS,
    )
    if raw.empty:
        raise SystemExit("No results collected.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "extraction_challenge_raw.csv"
    raw.to_csv(raw_path, index=False)

    long = to_long(raw)
    summary = summarize_long_results(long, ["condition", "model_family"])
    summary_path = RESULTS_DIR / "extraction_challenge_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_category = summarize_long_results(long, ["condition", "dataset"])
    by_category = by_category[by_category["metric"].isin(
        ["f1", "precision", "recall", "hallucination_rate", "context_error_rate"]
    )]
    category_path = RESULTS_DIR / "extraction_challenge_by_category.csv"
    by_category.to_csv(category_path, index=False)

    manifest = {
        "provenance": provenance,
        "challenge_set": {
            "path": str(CHALLENGE_SET.relative_to(ROOT)),
            "sha256": sha256_file(CHALLENGE_SET),
            "n_cases": len(cases),
            "n_real": sum(c["provenance"] == "real" for c in cases),
            "n_constructed": sum(c["provenance"] == "constructed" for c in cases),
            "categories": sorted({c["category"] for c in cases}),
        },
        "arms": ["regex"] if dry_run else ["regex", *benchmark.LLM_ARMS],
        "models": [p["model"] for p in providers],
        "gateway": "openrouter",
        "reps_per_llm_arm": reps,
        "response_format": "json_object (production setting; malformed rate is schema-level)",
        "num_retries": 0,
        "n_extractions": int(len(raw)),
        "n_api_errors": int(raw["api_error"].sum()),
        "total_cost_usd": float(pd.to_numeric(raw["cost_usd"], errors="coerce").sum()),
        "llm_calls": extractor.call_history,
    }
    manifest_path = RESULTS_DIR / "extraction_challenge_run_manifest.json"
    write_json(manifest_path, manifest)

    for path in (raw_path, summary_path, category_path, manifest_path):
        print(f"Saved {path.relative_to(ROOT)}")
    print(f"Total measured cost: ${manifest['total_cost_usd']:.4f} "
          f"over {manifest['n_extractions']} extractions "
          f"({manifest['n_api_errors']} API errors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=5,
                        help="Repetitions per (case, LLM arm, model). Default 5.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Provider names to include (openai, anthropic, google). "
                             "Default: all, all routed through OpenRouter.")
    parser.add_argument("--cases", nargs="*", default=None, dest="categories",
                        help="Restrict to these taxonomy categories (cheap smoke runs).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Regex arm only; makes no API calls and needs no key.")
    args = parser.parse_args()
    main(args.reps, args.models, args.categories, args.dry_run)
