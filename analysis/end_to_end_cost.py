"""End-to-end cost and latency accounting for one finished dataset (P1 9.8).

Every cost figure in the paper so far is *per extraction call*. This script answers
the different question a practitioner actually asks: what does it cost to produce
one final dataset from a real paper? It runs the whole pipeline over each real
source and records, per stage, the paid API cost and the local compute time:

    query_expansion      LLM   expand the user's topic into search queries
    search               local DuckDuckGo, once per expanded query
    fetch                local network retrieval of the selected document
    parse                local PDF/HTML/XML -> text
    extraction           LLM   parsed text -> Parameters
    synthesis            local per optimization iteration
    validation           local per optimization iteration
    refinement_overhead  local loop bookkeeping + session artifact writes

Deliberately *not* a re-run of `real_paper_eval.py`: that script owns the frozen
discovery/extraction/fidelity results behind Table 1, and re-running it live would
regenerate them. This script writes only its own files and reuses that module's
stage functions so both measure the same code path.

Network and LLM stages run once per source (they do not depend on the synthesis
seed); the synthesis/validation/refinement block repeats over `--seeds` seeds so
local compute gets a spread rather than a single sample.

Usage (PowerShell):
    .venv\\Scripts\\python.exe analysis\\end_to_end_cost.py --dry-run
    .venv\\Scripts\\python.exe analysis\\end_to_end_cost.py --sources calerie_rct
    .venv\\Scripts\\python.exe analysis\\end_to_end_cost.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from agentdataset.core.extractor import Extractor  # noqa: E402
from agentdataset.core.orchestrator import Orchestrator  # noqa: E402
from agentdataset.core.synthesizer import Synthesizer  # noqa: E402
from agentdataset.evaluation import (  # noqa: E402
    RAW_RESULT_COLUMNS,
    collect_provenance,
    stage_timer,
    summarize_long_results,
    validate_raw_results,
    write_json,
)

# Reused rather than reimplemented so this script and the frozen real-paper study
# exercise the same fetch/parse path and the same model configuration.
from real_paper_eval import (  # noqa: E402
    CACHE_DIR,
    ENV_VAR,
    MANIFEST_PATH,
    MODEL,
    fetch_parsed_text,
    load_env_safely,
)

RESULTS_DIR = ROOT / "results"
EXPERIMENT_ID = "end_to_end_cost_v1"
REFINEMENT_ITERATIONS = 3

# Stages whose wall time is paid to a provider rather than spent locally. Used to
# decide which side of the API/local split a stage's seconds land on.
API_STAGES = ("query_expansion", "extraction")

STAGE_ORDER = (
    "query_expansion", "search", "fetch", "parse", "extraction",
    "synthesis", "validation", "refinement_overhead",
)


def _row(
    source_id: str,
    stage: str,
    metric: str,
    value,
    *,
    seed: int | None = None,
    status: str = "ok",
    failure_stage: str | None = None,
    extraction_method: str | None = None,
) -> dict:
    """One observation in the shared long-format results schema.

    `condition` carries the pipeline stage, so the same summarisation helper used
    by every other experiment groups these rows without special-casing.
    """
    run_id = f"{source_id}:{stage}" + (f":{seed}" if seed is not None else "")
    return {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "experiment_type": "end_to_end_cost",
        "source_id": source_id,
        "dataset": None,
        "task_type": "pipeline_stage",
        "condition": stage,
        "model_family": "openai" if stage in API_STAGES else None,
        "synthesizer": "agentdataset",
        "seed": seed,
        "split_seed": None,
        "synthesis_seed": seed,
        "metric": metric,
        "value": value,
        "status": status,
        "failure_stage": failure_stage,
        "extraction_method": extraction_method,
        "provider": "openrouter" if stage in API_STAGES else "local",
        "model": MODEL if stage in API_STAGES else None,
    }


def _calls_since(extractor: Extractor, mark: int) -> list[dict]:
    """LLM calls recorded since `mark`, the length of call_history beforehand."""
    return extractor.call_history[mark:]


def _api_totals(calls: list[dict]) -> dict:
    """Aggregate a slice of call history into per-stage API totals."""
    def _sum(key: str) -> float:
        values = [c.get(key) for c in calls]
        return float(np.nansum([v if v is not None else np.nan for v in values])) if values else 0.0

    return {
        "api_cost_usd": _sum("cost_usd"),
        "api_latency_seconds": _sum("latency_seconds"),
        "prompt_tokens": _sum("prompt_tokens"),
        "completion_tokens": _sum("completion_tokens"),
        "llm_calls": float(len(calls)),
        "llm_call_failures": float(sum(c.get("status") == "failed" for c in calls)),
    }


def _emit_stage(
    rows: list[dict],
    source_id: str,
    stage: str,
    timing: dict,
    calls: list[dict],
    *,
    seed: int | None = None,
) -> None:
    """Write the standard metric set for one stage into `rows`.

    Local compute is wall time minus the time the stage spent waiting on a
    provider. A stage that made no LLM call is entirely local; an LLM stage still
    has a local share (prompt assembly, JSON parsing, Pydantic validation), and
    that share is what a practitioner can optimise without changing their bill.
    """
    totals = _api_totals(calls)
    local_seconds = max(0.0, timing["wall_seconds"] - totals["api_latency_seconds"])
    status = "ok" if timing["status"] == "ok" else "failed"
    failure_stage = None if status == "ok" else stage
    metrics = {
        "wall_seconds": timing["wall_seconds"],
        "cpu_seconds": timing["cpu_seconds"],
        "local_compute_seconds": local_seconds,
        **totals,
    }
    for metric, value in metrics.items():
        rows.append(_row(source_id, stage, metric, value, seed=seed,
                         status=status, failure_stage=failure_stage))


def run_source(
    source: pd.Series,
    seeds: tuple[int, ...],
    dry_run: bool,
) -> tuple[list[dict], Extractor]:
    """Run the full pipeline once for one real paper, timing every stage."""
    source_id = source.source_id
    rows: list[dict] = []
    extractor = Extractor(model=MODEL, env_var=ENV_VAR)
    orchestrator = Orchestrator(
        session_id=f"cost_{source_id}", model=MODEL, env_var=ENV_VAR,
        base_dir=str(CACHE_DIR / "sessions"),
    )
    orchestrator.extractor = extractor

    # --- Discovery: query expansion (paid) then search (local network) ---
    queries = [source.discovery_query]
    if dry_run:
        for stage in ("query_expansion", "search"):
            for metric in ("wall_seconds", "cpu_seconds", "local_compute_seconds",
                           "api_cost_usd", "api_latency_seconds", "prompt_tokens",
                           "completion_tokens", "llm_calls", "llm_call_failures"):
                rows.append(_row(source_id, stage, metric, 0.0, status="skipped"))
    else:
        mark = len(extractor.call_history)
        with stage_timer("query_expansion") as timing:
            queries = orchestrator.optimize_query(source.discovery_query)
        _emit_stage(rows, source_id, "query_expansion", timing,
                    _calls_since(extractor, mark))

        with stage_timer("search") as timing:
            n_results = 0
            for query in queries:
                try:
                    n_results += len(orchestrator.discovery.search(query))
                except Exception:
                    # A failed search still costs the wall time already spent, and
                    # the timer records it; the pipeline continues on the manifest's
                    # direct URL exactly as the real-paper study does.
                    pass
        _emit_stage(rows, source_id, "search", timing, [])
        rows.append(_row(source_id, "search", "search_results", float(n_results)))
        rows.append(_row(source_id, "search", "queries_issued", float(len(queries))))

    # --- Fetch and parse (local network + local CPU) ---
    # fetch_parsed_text times both internally and reports them separately, so the
    # split here comes from its own measurements rather than a second stopwatch.
    with stage_timer("fetch_parse") as fetch_parse_timing:
        text, fetch_meta = fetch_parsed_text(source)
    fetch_seconds = float(fetch_meta.get("fetch_seconds", float("nan")))
    total_fetch_parse = float(fetch_meta.get("fetch_parse_seconds", fetch_parse_timing["wall_seconds"]))
    parse_seconds = max(0.0, total_fetch_parse - fetch_seconds)
    fetch_ok = fetch_meta.get("fetch_status") == "ok"
    parse_ok = fetch_meta.get("parse_status") == "ok"

    _emit_stage(rows, source_id, "fetch",
                {"wall_seconds": fetch_seconds, "cpu_seconds": float("nan"),
                 "status": "ok" if fetch_ok else "failed"}, [])
    _emit_stage(rows, source_id, "parse",
                {"wall_seconds": parse_seconds, "cpu_seconds": float("nan"),
                 "status": "ok" if parse_ok else "failed"}, [])
    rows.append(_row(source_id, "parse", "parsed_characters",
                     float(fetch_meta.get("parsed_characters", 0))))
    rows.append(_row(source_id, "parse", "documents_processed", 1.0))

    # --- Extraction (paid) ---
    mark = len(extractor.call_history)
    with stage_timer("extraction") as timing:
        if dry_run:
            variables, correlations = extractor._extract_with_regex(text)
            from datetime import datetime, timezone

            from agentdataset.models.schemas import MetaParams, Parameters
            params = Parameters(
                variables=variables, correlations=correlations,
                meta=MetaParams(source=source.title,
                                extracted_at=datetime.now(timezone.utc).isoformat(),
                                extraction_method="regex_fallback"),
            )
        else:
            with extractor.stage_scope("extraction"):
                params = extractor.extract_parameters(text, source.title)
    extraction_calls = _calls_since(extractor, mark)
    _emit_stage(rows, source_id, "extraction", timing, extraction_calls)
    rows.append(_row(source_id, "extraction", "variables_extracted",
                     float(len(params.variables)),
                     extraction_method=params.meta.extraction_method))
    rows.append(_row(source_id, "extraction", "correlations_extracted",
                     float(len(params.correlations)),
                     extraction_method=params.meta.extraction_method))

    if not params.variables:
        # Nothing to synthesize. Recorded rather than crashed: a source that
        # yields no variables still consumed the discovery and extraction budget,
        # and that is exactly the failure overhead §9.8 asks to be reported.
        for stage in ("synthesis", "validation", "refinement_overhead"):
            rows.append(_row(source_id, stage, "wall_seconds", float("nan"),
                             status="failed", failure_stage="extraction"))
        return rows, extractor

    # --- Synthesis / validation / refinement (local), repeated over seeds ---
    for seed in seeds:
        loop_orchestrator = Orchestrator(
            session_id=f"cost_{source_id}_{seed}", model=MODEL, env_var=ENV_VAR,
            base_dir=str(CACHE_DIR / "sessions"),
        )
        loop_orchestrator.synthesizer = Synthesizer(n_rows=int(source.sample_size), seed=seed)
        with stage_timer("refinement") as loop_timing:
            loop_orchestrator.run_optimization_loop(params, iterations=REFINEMENT_ITERATIONS)
        trace = pd.DataFrame(loop_orchestrator.optimization_trace)
        synthesis_seconds = float(trace["synthesis_seconds"].sum())
        validation_seconds = float(trace["validation_seconds"].sum())
        # Whatever the loop spent that was neither synthesis nor validation:
        # ratchet bookkeeping, datacard generation, and session artifact writes.
        overhead_seconds = max(0.0, loop_timing["wall_seconds"] - synthesis_seconds - validation_seconds)

        for stage, seconds in (
            ("synthesis", synthesis_seconds),
            ("validation", validation_seconds),
            ("refinement_overhead", overhead_seconds),
        ):
            _emit_stage(rows, source_id, stage,
                        {"wall_seconds": seconds, "cpu_seconds": float("nan"), "status": "ok"},
                        [], seed=seed)
        rows.append(_row(source_id, "refinement_overhead", "optimization_iterations",
                         float(len(trace)), seed=seed))
        rows.append(_row(source_id, "refinement_overhead", "rows_synthesized",
                         float(source.sample_size), seed=seed))

    return rows, extractor


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Per source x stage means, plus per-source and grand TOTAL rows.

    Totals sum the per-stage *means* rather than the raw observations, so a stage
    measured over three seeds does not count three times toward the cost of the
    one dataset the totals describe.
    """
    stage_summary = summarize_long_results(raw, ["source_id", "condition"])
    stage_summary = stage_summary.rename(columns={"condition": "stage"})

    totalled = ("wall_seconds", "local_compute_seconds", "api_cost_usd",
                "api_latency_seconds", "prompt_tokens", "completion_tokens",
                "llm_calls", "llm_call_failures")
    totals = []
    per_source = stage_summary[stage_summary["metric"].isin(totalled)]
    for source_id, group in per_source.groupby("source_id"):
        for metric, metric_group in group.groupby("metric"):
            totals.append({"source_id": source_id, "stage": "TOTAL", "metric": metric,
                           "n": int(metric_group["n"].sum()),
                           "mean": float(metric_group["mean"].sum()),
                           "std": float("nan"), "ci95_low": float("nan"),
                           "ci95_high": float("nan")})
    grand = pd.DataFrame(totals)
    if not grand.empty:
        for metric, metric_group in grand.groupby("metric"):
            totals.append({"source_id": "ALL_SOURCES", "stage": "TOTAL", "metric": metric,
                           "n": int(metric_group["n"].sum()),
                           "mean": float(metric_group["mean"].sum()),
                           "std": float("nan"), "ci95_low": float("nan"),
                           "ci95_high": float("nan")})

    summary = pd.concat([stage_summary, pd.DataFrame(totals)], ignore_index=True)
    order = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    summary["_order"] = summary["stage"].map(lambda s: order.get(s, len(order)))
    return summary.sort_values(["source_id", "_order", "metric"]).drop(columns="_order")


def main(seeds: int, sources: list[str] | None, dry_run: bool) -> None:
    load_env_safely()
    manifest = pd.read_csv(MANIFEST_PATH)
    if sources:
        manifest = manifest[manifest.source_id.isin(sources)]
    seed_tuple = tuple(range(seeds))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Captured before any output is written; otherwise git_dirty is structurally
    # always True, because this run's own CSVs are already in the working tree.
    provenance = collect_provenance(
        "python analysis/end_to_end_cost.py"
        + f" --seeds {seeds}"
        + (f" --sources {' '.join(sources)}" if sources else "")
        + (" --dry-run" if dry_run else ""),
        seed_tuple,
    )

    all_rows: list[dict] = []
    all_calls: list[dict] = []
    for _, source in manifest.iterrows():
        print(f"[{source.source_id}] running full pipeline...")
        rows, extractor = run_source(source, seed_tuple, dry_run)
        all_rows.extend(rows)
        all_calls.extend(extractor.call_history)

    raw = pd.DataFrame(all_rows, columns=RAW_RESULT_COLUMNS)
    validate_raw_results(raw)
    raw_path = RESULTS_DIR / "end_to_end_cost_latency.csv"
    raw.to_csv(raw_path, index=False)

    summary = _summarize(raw)
    summary_path = RESULTS_DIR / "end_to_end_cost_latency_summary.csv"
    summary.to_csv(summary_path, index=False)

    grand = summary[summary.source_id.eq("ALL_SOURCES") & summary.stage.eq("TOTAL")]
    def _grand(metric: str) -> float:
        row = grand[grand.metric.eq(metric)]
        return float(row["mean"].iloc[0]) if len(row) else float("nan")

    manifest_payload = {
        "provenance": provenance,
        "provider": "openrouter", "model": MODEL,
        "gateway": "openrouter",
        "mode": "dry_run (regex extraction, no discovery)" if dry_run else "live",
        "sources": manifest.source_id.tolist(),
        "seeds": list(seed_tuple),
        "refinement_iterations": REFINEMENT_ITERATIONS,
        "scope": "cost of producing one final dataset per source; "
                 "network and LLM stages run once, local stages averaged over seeds",
        "stages": list(STAGE_ORDER),
        "totals": {metric: _grand(metric) for metric in
                   ("api_cost_usd", "api_latency_seconds", "local_compute_seconds",
                    "wall_seconds", "llm_calls", "llm_call_failures",
                    "prompt_tokens", "completion_tokens")},
        "llm_calls": all_calls,
    }
    manifest_path = RESULTS_DIR / "end_to_end_cost_latency_run_manifest.json"
    write_json(manifest_path, manifest_payload)

    for path in (raw_path, summary_path, manifest_path):
        print(f"Saved {path.relative_to(ROOT)}")
    print(f"Grand total across {len(manifest)} sources: "
          f"${_grand('api_cost_usd'):.4f} API cost, "
          f"{_grand('api_latency_seconds'):.1f}s API latency, "
          f"{_grand('local_compute_seconds'):.1f}s local compute, "
          f"{_grand('llm_calls'):.0f} LLM calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of synthesis seeds for the local stages. Default 3.")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Restrict to these manifest source_ids.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip discovery and use regex extraction; needs no API key.")
    args = parser.parse_args()
    main(args.seeds, args.sources, args.dry_run)
