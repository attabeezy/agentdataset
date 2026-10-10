"""Reproducible real-paper discovery, extraction, and synthesis study.

Live mode uses the configured OpenRouter credential. Scoring and synthesis can
be rerun offline from cached parsed text in ``.agentdataset_cache/real_papers``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from agentdataset.core.discovery import DiscoveryAgent, PDF_PATH_PREFIX
from agentdataset.core.extractor import Extractor
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.core.validator import Validator
from agentdataset.evaluation import collect_provenance, summarize_long_results, write_json
from agentdataset.models.schemas import (
    CorrelationParams,
    DiscoveryResult,
    MetaParams,
    Parameters,
    VariableParams,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "analysis" / "real_paper_manifest.csv"
GOLD_PATH = ROOT / "analysis" / "real_paper_ground_truth.csv"
CACHE_DIR = ROOT / ".agentdataset_cache" / "real_papers"
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = ROOT / "analysis"
MODEL = "openrouter/openai/gpt-4o"
ENV_VAR = "OPENROUTER_API_KEY"
SEEDS = tuple(range(30))


def load_env_safely() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _source_token(row: pd.Series) -> str:
    for url in (row["source_url"], row["direct_url"]):
        match = re.search(r"PMC\d+", url, flags=re.I)
        if match:
            return match.group(0).lower()
    return Path(urlparse(row["source_url"]).path).stem.lower()


def _is_target(result: DiscoveryResult, row: pd.Series) -> bool:
    token = _source_token(row)
    haystack = f"{result.url} {result.title}".lower().replace("-", "_")
    title_terms = [term for term in re.findall(r"[a-z0-9]+", row["title"].lower()) if len(term) > 5]
    return token in haystack or sum(term in haystack for term in title_terms) >= 4


def fetch_parsed_text(row: pd.Series) -> tuple[str, dict]:
    result = DiscoveryResult(
        title=row["title"], url=row["direct_url"],
        source_type=row["source_type"], relevance_score=1.0,
    )
    discovery = DiscoveryAgent()
    extractor = Extractor()
    started = time.perf_counter()
    if str(row["direct_url"]).endswith("fullTextXML"):
        try:
            response = requests.get(row["direct_url"], timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            content = "\n".join(text.strip() for text in root.itertext() if text.strip())
        except Exception:
            content = ""
    else:
        content = discovery.fetch_content(result)
    fetch_seconds = time.perf_counter() - started
    fetch_status = "ok"
    parse_status = "ok"
    parsed = ""
    try:
        if content.startswith(PDF_PATH_PREFIX):
            pdf_path = content[len(PDF_PATH_PREFIX):]
            try:
                parsed = extractor.pdf_to_markdown(pdf_path)
            finally:
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
        else:
            parsed = content
        if not parsed.strip():
            parse_status = "empty"
    except Exception as exc:
        parse_status = f"failed:{type(exc).__name__}"
    if not content:
        fetch_status = "empty"
    return parsed, {
        "fetch_status": fetch_status,
        "parse_status": parse_status,
        "fetch_parse_seconds": time.perf_counter() - started,
        "fetch_seconds": fetch_seconds,
        "parsed_characters": len(parsed),
    }


def run_discovery_study(manifest: pd.DataFrame, extractor: Extractor) -> pd.DataFrame:
    rows = []
    for _, source in manifest.iterrows():
        orchestrator = Orchestrator(
            session_id=f"discovery_{source.source_id}", model=MODEL,
            env_var=ENV_VAR, base_dir=str(CACHE_DIR / "sessions"),
        )
        queries = orchestrator.optimize_query(source.discovery_query)
        seen: set[str] = set()
        global_rank = 0
        search_failed = False
        for query_index, query in enumerate(queries):
            started = time.perf_counter()
            try:
                results = orchestrator.discovery.search(query)
                search_status = "ok"
            except Exception as exc:
                results = []
                search_status = f"failed:{type(exc).__name__}"
                search_failed = True
            elapsed = time.perf_counter() - started
            for query_rank, result in enumerate(results, start=1):
                duplicate = result.url in seen
                if not duplicate:
                    seen.add(result.url)
                    global_rank += 1
                rows.append({
                    "source_id": source.source_id,
                    "search_date": datetime.now(timezone.utc).date().isoformat(),
                    "query_index": query_index,
                    "query": query,
                    "query_rank": query_rank,
                    "deduplicated_rank": None if duplicate else global_rank,
                    "title": result.title,
                    "url": result.url,
                    "source_type": result.source_type,
                    "snippet": result.snippet,
                    "duplicate": duplicate,
                    "is_target": _is_target(result, source),
                    "search_status": search_status,
                    "search_seconds": elapsed,
                })
        if not rows or not any(r["source_id"] == source.source_id for r in rows):
            rows.append({
                "source_id": source.source_id, "search_date": datetime.now(timezone.utc).date().isoformat(),
                "query_index": 0, "query": source.discovery_query, "query_rank": None,
                "deduplicated_rank": None, "title": None, "url": None,
                "source_type": None, "snippet": None, "duplicate": False,
                "is_target": False, "search_status": "failed" if search_failed else "empty",
                "search_seconds": None,
            })
    return pd.DataFrame(rows)


def flatten_parameters(params: Parameters) -> pd.DataFrame:
    rows = []
    for name, variable in params.variables.items():
        if variable.distribution == "categorical" and variable.categories:
            for label, probability in variable.categories.items():
                rows.append({"variable_name": name, "statistic_type": "probability", "value": probability, "label": label})
        else:
            rows.extend([
                {"variable_name": name, "statistic_type": "mean", "value": variable.mean, "label": None},
                {"variable_name": name, "statistic_type": "std", "value": variable.std, "label": None},
            ])
    for correlation in params.correlations.values():
        rows.append({
            "variable_name": f"{correlation.var1}__{correlation.var2}",
            "statistic_type": "correlation", "value": correlation.correlation, "label": None,
        })
    return pd.DataFrame(rows)


def _name_matches(predicted: str, gold: pd.Series) -> bool:
    normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    candidates = [gold.variable_name, gold.source_label, *str(gold.aliases).split("|")]
    pred = normalize(predicted)
    return any(pred == normalize(candidate) or pred in normalize(candidate) for candidate in candidates)


def score_extraction(source_id: str, params: Parameters, gold: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    expected = gold.loc[gold.source_id.eq(source_id)].reset_index(drop=True)
    predicted = flatten_parameters(params).reset_index(drop=True)
    used: set[int] = set()
    matches = []
    errors = []
    for pred_index, pred in predicted.iterrows():
        named_candidates = []
        numeric_candidates = []
        for gold_index, item in expected.iterrows():
            if gold_index in used or pred.statistic_type != item.statistic_type:
                continue
            tolerance = 0.5 * (10 ** -int(item.decimals))
            difference = abs(float(pred.value) - float(item.value))
            if _name_matches(pred.variable_name, item):
                named_candidates.append((difference, gold_index, item, tolerance))
            if difference <= tolerance + 1e-12:
                numeric_candidates.append((difference, gold_index, item, tolerance))
        candidates = named_candidates or numeric_candidates
        if candidates:
            difference, gold_index, item, tolerance = sorted(candidates, key=lambda entry: entry[0])[0]
            used.add(gold_index)
            name_ok = _name_matches(pred.variable_name, item)
            numeric_ok = difference <= tolerance + 1e-12
            matches.append({
                "gold_id": item.gold_id, "predicted_variable": pred.variable_name,
                "statistic_type": pred.statistic_type, "gold_value": item.value,
                "predicted_value": pred.value, "numeric_error": difference,
                "numeric_correct": numeric_ok,
                "variable_name_correct": name_ok,
                "statistic_type_correct": True,
                "context_correct": False,
            })
            if not name_ok:
                errors.append({"source_id": source_id, "gold_id": item.gold_id, "error_category": "variable", "detail": pred.variable_name})
            elif not numeric_ok:
                errors.append({"source_id": source_id, "gold_id": item.gold_id, "error_category": "numeric_value", "detail": f"gold={item.value};predicted={pred.value}"})
        else:
            errors.append({"source_id": source_id, "gold_id": None, "error_category": "unsupported_or_out_of_scope", "detail": f"{pred.variable_name}:{pred.statistic_type}={pred.value}"})
    for gold_index, item in expected.iterrows():
        if gold_index not in used:
            errors.append({"source_id": source_id, "gold_id": item.gold_id, "error_category": "no_extraction", "detail": item.source_label})

    tp, fp, fn = len(matches), len(predicted) - len(matches), len(expected) - len(matches)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "source_id": source_id,
        "extraction_method": params.meta.extraction_method,
        "gold_items": len(expected), "predicted_items": len(predicted),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "numeric_accuracy": np.mean([m["numeric_correct"] for m in matches]) if matches else 0.0,
        "variable_name_accuracy": np.mean([m["variable_name_correct"] for m in matches]) if matches else 0.0,
        "statistic_type_accuracy": np.mean([m["statistic_type_correct"] for m in matches]) if matches else 0.0,
        "context_scope_accuracy": np.mean([m["context_correct"] for m in matches]) if matches else 0.0,
        "unsupported_or_out_of_scope_rate": fp / len(predicted) if len(predicted) else 0.0,
    }
    return metrics, pd.DataFrame(errors)


def gold_to_parameters(source_id: str, gold: pd.DataFrame) -> Parameters:
    subset = gold.loc[gold.source_id.eq(source_id)]
    variables = {}
    correlations = {}
    for name, group in subset.groupby("variable_name"):
        if "__" in name:
            row = group.loc[group.statistic_type.eq("correlation")]
            if not row.empty:
                var1, var2 = name.split("__", 1)
                value = float(row.iloc[0].value)
                correlations[name] = CorrelationParams(var1=var1, var2=var2, correlation=value, direction="positive" if value >= 0 else "negative")
            continue
        probs = group.loc[group.statistic_type.eq("probability")]
        if not probs.empty:
            categories = {
                re.sub(r"^.*?\b", "", str(row.source_label), count=1).strip() or str(index): float(row.value)
                for index, row in probs.iterrows()
            }
            variables[name] = VariableParams(name=name, distribution="categorical", categories=categories)
            continue
        means = group.loc[group.statistic_type.eq("mean")]
        stds = group.loc[group.statistic_type.eq("std")]
        if not means.empty and not stds.empty:
            mean, std = float(means.iloc[0].value), float(stds.iloc[0].value)
            variables[name] = VariableParams(name=name, mean=mean, std=std, min=mean - 3 * std, max=mean + 3 * std)
    return Parameters(
        variables=variables, correlations=correlations,
        meta=MetaParams(source=source_id, extracted_at=datetime.now(timezone.utc).isoformat(), extraction_method="gold"),
    )


def synthesize_conditions(source: pd.Series, conditions: dict[str, Parameters]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows, trace_rows = [], []
    for condition, params in conditions.items():
        if not params.variables:
            continue
        for seed in SEEDS:
            orchestrator = Orchestrator(
                session_id=f"paper_{source.source_id}_{condition}_{seed}",
                base_dir=str(CACHE_DIR / "sessions"), model=MODEL, env_var=ENV_VAR,
            )
            orchestrator.synthesizer = Synthesizer(n_rows=int(source.sample_size), seed=seed)
            best_score, frame = orchestrator.run_optimization_loop(params, iterations=3)
            report = Validator().validate(frame, params)
            # §9.7: read the structured diagnostics the synthesizer now records
            # instead of string-matching a RuntimeWarning message. `pairs_applied`
            # matters as much as the PD status — a matrix can pass the
            # positive-definite check and still impose nothing, if the declared
            # pairs name variables that were never extracted.
            diagnostics = orchestrator.synthesizer.diagnostics
            fallback_count = sum(d.action == "dropped_all" for d in diagnostics)
            vacuous_count = sum(d.n_pairs_applied == 0 for d in diagnostics)
            for metric, value in {
                "fidelity_score": best_score, "ks_score": report.ks_score,
                "correlation_score": report.corr_score, "bias_score": report.bias_score,
                "normalized_row_dispersion": report.dispersion_details["normalized_row_dispersion"],
                "fallback_warning_count": fallback_count,
                "vacuous_matrix_count": vacuous_count,
                "correlation_matrices_built": len(diagnostics),
            }.items():
                raw_rows.append({
                    "experiment_id": "real_paper_pipeline_v1", "run_id": f"{source.source_id}:{condition}:{seed}",
                    "experiment_type": "real_paper_synthesis", "source_id": source.source_id,
                    "dataset": None, "task_type": "synthesis", "condition": condition,
                    "model_family": None, "synthesizer": "agentdataset", "seed": seed,
                    "split_seed": None, "synthesis_seed": seed, "metric": metric,
                    "value": value, "status": "ok", "failure_stage": None,
                    "extraction_method": params.meta.extraction_method,
                    "provider": "openrouter" if condition == "llm" else "local", "model": MODEL if condition == "llm" else None,
                })
            for item in orchestrator.optimization_trace:
                trace_rows.append({"source_id": source.source_id, "condition": condition, "seed": seed, **item})
    return pd.DataFrame(raw_rows), pd.DataFrame(trace_rows)


def _summarize_discovery(discovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_id, group in discovery.groupby("source_id"):
        unique = group.loc[~group["duplicate"].astype(bool)]
        target_ranks = pd.to_numeric(
            unique.loc[unique["is_target"].astype(bool), "deduplicated_rank"], errors="coerce"
        ).dropna()
        first_rank = float(target_ranks.min()) if len(target_ranks) else np.nan
        rows.append({
            "source_id": source_id,
            "target_found": bool(len(target_ranks)),
            "first_relevant_rank": first_rank,
            "recall_at_5": float(bool(len(target_ranks) and first_rank <= 5)),
            "reciprocal_rank": 1.0 / first_rank if np.isfinite(first_rank) else 0.0,
            "search_failed": bool(group["search_status"].astype(str).str.startswith("failed").any()),
            "unique_results": len(unique),
        })
    return pd.DataFrame(rows)


def _summarize_extraction(extraction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in extraction.groupby("condition"):
        tp, fp, fn = group[["true_positives", "false_positives", "false_negatives"]].sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "condition": condition, "documents": len(group),
            "precision_micro": precision, "recall_micro": recall,
            "f1_micro": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "f1_macro": group["f1"].mean(),
            "unsupported_or_out_of_scope_rate_macro": group["unsupported_or_out_of_scope_rate"].mean(),
            "variable_name_accuracy_macro": group["variable_name_accuracy"].mean(),
            "context_scope_accuracy_macro": group["context_scope_accuracy"].mean(),
            "parse_failures": int((~group["parse_status"].eq("ok")).sum()),
        })
    return pd.DataFrame(rows)


def _tagged(filename: str, tag: str) -> str:
    """Insert a run tag before the extension: ("a.csv", "t") -> "a_t.csv"."""
    if not tag:
        return filename
    stem, ext = filename.rsplit(".", 1)
    return f"{stem}_{tag}.{ext}"


def run(mode: str, refresh_sources: set[str] | None = None,
        model: str = MODEL, tag: str = "") -> None:
    load_env_safely()
    manifest = pd.read_csv(MANIFEST_PATH)
    gold = pd.read_csv(GOLD_PATH)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Captured BEFORE any output is written. Recording provenance afterwards
    # makes git_dirty structurally always True, because this run's own CSVs are
    # sitting in the working tree by then.
    provenance = collect_provenance(
        f"python analysis/real_paper_eval.py --mode {mode}" +
        "".join(f" --refresh-source {source}" for source in sorted(refresh_sources or set())) +
        (f" --model {model}" if model != MODEL else "") +
        (f" --tag {tag}" if tag else ""),
        SEEDS,
    )

    extractor = Extractor(model=model, env_var=ENV_VAR)
    if mode == "live" and not tag:
        discovery = run_discovery_study(manifest, extractor)
        discovery.to_csv(ANALYSIS_DIR / "real_paper_discovery.csv", index=False)
        _summarize_discovery(discovery).to_csv(ANALYSIS_DIR / "real_paper_discovery_summary.csv", index=False)
    elif (ANALYSIS_DIR / "real_paper_discovery.csv").exists():
        # Shared discovery input: tagged runs read it but never rewrite it,
        # so committed discovery files cannot be clobbered by a model sweep.
        discovery = pd.read_csv(ANALYSIS_DIR / "real_paper_discovery.csv")
        if not tag:
            _summarize_discovery(discovery).to_csv(ANALYSIS_DIR / "real_paper_discovery_summary.csv", index=False)

    extraction_rows, error_frames, raw_frames, trace_frames = [], [], [], []
    for _, source in manifest.iterrows():
        cached_text = CACHE_DIR / f"{source.source_id}.txt"
        fetch_meta = {}
        if mode == "live" or source.source_id in (refresh_sources or set()):
            text, fetch_meta = fetch_parsed_text(source)
            cached_text.write_text(text, encoding="utf-8")
        elif cached_text.exists():
            text = cached_text.read_text(encoding="utf-8")
            fetch_meta = {"fetch_status": "cached", "parse_status": "ok" if text.strip() else "empty", "parsed_characters": len(text)}
        else:
            continue

        llm_params = extractor.extract_parameters(text, source.title)
        regex_vars, regex_corrs = extractor._extract_with_regex(text)
        regex_params = Parameters(
            variables=regex_vars, correlations=regex_corrs,
            meta=MetaParams(source=source.title, extracted_at=datetime.now(timezone.utc).isoformat(), extraction_method="regex_fallback"),
        )
        conditions = {"gold": gold_to_parameters(source.source_id, gold), "llm": llm_params, "regex": regex_params}
        for condition in ("llm", "regex"):
            metrics, errors = score_extraction(source.source_id, conditions[condition], gold)
            extraction_rows.append({"condition": condition, **fetch_meta, **metrics})
            if not errors.empty:
                errors.insert(1, "condition", condition)
                error_frames.append(errors)
        raw, traces = synthesize_conditions(source, conditions)
        raw_frames.append(raw)
        trace_frames.append(traces)

    extraction = pd.DataFrame(extraction_rows)
    extraction.to_csv(ANALYSIS_DIR / _tagged("real_paper_extraction.csv", tag), index=False)
    _summarize_extraction(extraction).to_csv(ANALYSIS_DIR / _tagged("real_paper_extraction_summary.csv", tag), index=False)
    pd.concat(error_frames, ignore_index=True).to_csv(ANALYSIS_DIR / _tagged("real_paper_errors.csv", tag), index=False) if error_frames else None
    if raw_frames:
        raw = pd.concat(raw_frames, ignore_index=True)
        raw.to_csv(RESULTS_DIR / _tagged("real_paper_pipeline_raw.csv", tag), index=False)
        summary = summarize_long_results(raw, ["source_id", "condition", "synthesizer", "extraction_method"])
        summary.to_csv(RESULTS_DIR / _tagged("real_paper_pipeline_summary.csv", tag), index=False)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(RESULTS_DIR / _tagged("real_paper_optimization_traces.csv", tag), index=False)
    write_json(RESULTS_DIR / _tagged("real_paper_run_manifest.json", tag), {
        **provenance,
        "provider": "openrouter", "model": model, "tag": tag,
        "extraction_route": "llm and forced regex on identical parsed text",
        "llm_calls": extractor.call_history,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("live", "cached"), default="cached")
    parser.add_argument("--refresh-source", action="append", default=[])
    parser.add_argument("--model", default=MODEL,
                        help="OpenRouter model for LLM extraction. Default is the "
                             "committed study model; tagged runs write suffixed outputs.")
    parser.add_argument("--tag", default="",
                        help="Suffix for output filenames (e.g. fivelab_gpt). Empty "
                             "keeps the committed filenames; tagged runs never "
                             "rewrite shared discovery files.")
    args = parser.parse_args()
    run(args.mode, set(args.refresh_source), model=args.model, tag=args.tag)
