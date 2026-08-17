"""Measure complete per-paper AgentDataset cost and latency.

Live mode performs query expansion, web search, source suggestion, target
fetch/parse, extraction, and three local synthesize/validate/refine timing
repetitions. Cached mode is an API-free local-compute check and writes separate
filenames so it cannot overwrite the live evidence.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura

from agentdataset.core.discovery import DiscoveryAgent, PDF_PATH_PREFIX
from agentdataset.core.extractor import Extractor
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.evaluation import collect_provenance, write_json
from agentdataset.models.schemas import DiscoveryResult, Parameters
from analysis.real_paper_eval import (
    ENV_VAR,
    GOLD_PATH,
    MANIFEST_PATH,
    MODEL,
    _is_target,
    gold_to_parameters,
    load_env_safely,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".agentdataset_cache" / "end_to_end_cost"
SPEND_CAP_USD = 0.25
MAX_LLM_ATTEMPTS = 12
MAX_RETRIES_PER_CALL = 2
TIMING_REPETITIONS = (0, 1, 2)
OPTIMIZATION_ITERATIONS = 3

STAGES = (
    "query_expansion",
    "web_search",
    "source_suggestion",
    "fetch",
    "parse",
    "extraction",
    "synthesis",
    "validation",
    "refinement_overhead",
    "artifact_write",
)


class BudgetStopped(RuntimeError):
    """Raised before a paid attempt when a configured guard is exhausted."""


@dataclass
class SpendGuard:
    cap_usd: float = SPEND_CAP_USD
    max_attempts: int = MAX_LLM_ATTEMPTS
    known_cost_usd: float = 0.0
    attempts: int = 0

    def before_attempt(self) -> int:
        if self.known_cost_usd >= self.cap_usd:
            raise BudgetStopped(
                f"recorded cost ${self.known_cost_usd:.4f} reached cap ${self.cap_usd:.2f}"
            )
        if self.attempts >= self.max_attempts:
            raise BudgetStopped(f"LLM attempt cap {self.max_attempts} reached")
        self.attempts += 1
        return self.attempts

    def record_known_cost(self, value: Any) -> None:
        try:
            cost = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(cost):
            self.known_cost_usd += cost


class AccountingExtractor(Extractor):
    """Experiment-only explicit retries with one audit row per attempt."""

    def __init__(self, *args, guard: SpendGuard, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard = guard
        self.attempt_history: list[dict[str, Any]] = []

    def _attempt(
        self,
        purpose: str,
        callback: Callable[[], Any],
        num_retries: int,
    ) -> Any:
        for retry_index in range(num_retries + 1):
            try:
                global_attempt = self.guard.before_attempt()
            except BudgetStopped as exc:
                self.attempt_history.append({
                    "purpose": purpose,
                    "attempt": None,
                    "retry_index": retry_index,
                    "status": "budget_stopped",
                    "error_type": type(exc).__name__,
                    "latency_seconds": 0.0,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "cost_usd": None,
                    "cost_known": False,
                })
                raise

            started = time.perf_counter()
            history_before = len(self.call_history)
            try:
                result = callback()
            except Exception as exc:
                completion_row = (
                    self.call_history[-1]
                    if len(self.call_history) > history_before
                    else None
                )
                row = self._attempt_row(
                    purpose,
                    global_attempt,
                    retry_index,
                    "failed",
                    type(exc).__name__,
                    time.perf_counter() - started,
                    completion_row,
                )
                self.attempt_history.append(row)
                if row["cost_known"]:
                    self.guard.record_known_cost(row["cost_usd"])
                if retry_index >= num_retries:
                    raise
            else:
                completion_row = self.call_history[-1]
                row = self._attempt_row(
                    purpose,
                    global_attempt,
                    retry_index,
                    "ok",
                    None,
                    time.perf_counter() - started,
                    completion_row,
                )
                self.attempt_history.append(row)
                self.guard.record_known_cost(row["cost_usd"])
                return result
        raise AssertionError("retry loop exited without returning or raising")

    @staticmethod
    def _attempt_row(
        purpose: str,
        global_attempt: int,
        retry_index: int,
        status: str,
        error_type: str | None,
        elapsed: float,
        completion_row: dict[str, Any] | None,
    ) -> dict[str, Any]:
        completion_row = completion_row or {}
        raw_cost = completion_row.get("cost_usd")
        try:
            cost = float(raw_cost)
            cost_known = math.isfinite(cost)
        except (TypeError, ValueError):
            cost, cost_known = None, False
        return {
            "purpose": purpose,
            "attempt": global_attempt,
            "retry_index": retry_index,
            "status": status,
            "error_type": error_type,
            "latency_seconds": completion_row.get("latency_seconds", elapsed),
            "prompt_tokens": completion_row.get("prompt_tokens"),
            "completion_tokens": completion_row.get("completion_tokens"),
            "total_tokens": completion_row.get("total_tokens"),
            "cost_usd": cost,
            "cost_known": cost_known,
        }

    def llm_call(
        self,
        prompt: str,
        *,
        purpose: str = "general_llm",
        num_retries: int = MAX_RETRIES_PER_CALL,
    ) -> str:
        return self._attempt(
            purpose,
            lambda: super(AccountingExtractor, self).llm_call(
                prompt, purpose=purpose, num_retries=0
            ),
            num_retries,
        )

    def _extract_with_llm(
        self,
        text: str,
        system_prompt: str | None = None,
        model: str | None = None,
        num_retries: int = MAX_RETRIES_PER_CALL,
        purpose: str = "extraction",
    ) -> dict[str, Any]:
        return self._attempt(
            purpose,
            lambda: super(AccountingExtractor, self)._extract_with_llm(
                text,
                system_prompt=system_prompt,
                model=model,
                num_retries=0,
                purpose=purpose,
            ),
            num_retries,
        )


def _base_row(source_id: str, run_id: str, stage: str, **values) -> dict[str, Any]:
    row = {
        "experiment_id": "end_to_end_cost_v1",
        "run_id": run_id,
        "source_id": source_id,
        "record_type": "stage_measurement",
        "stage": stage,
        "timing_repetition": None,
        "route": None,
        "status": "ok",
        "error_type": None,
        "provider": "local",
        "model": None,
        "llm_calls": 0,
        "retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_cost_usd": 0.0,
        "cost_known": True,
        "api_seconds": 0.0,
        "network_seconds": 0.0,
        "local_seconds": 0.0,
        "wall_seconds": 0.0,
        "documents_processed": 0,
        "optimization_iterations": 0,
        "cache_hit": False,
        "notes": None,
    }
    row.update(values)
    return row


def _llm_stage_row(
    source_id: str,
    run_id: str,
    stage: str,
    attempts: list[dict[str, Any]],
    wall_seconds: float,
    **values,
) -> dict[str, Any]:
    issued = [row for row in attempts if row["attempt"] is not None]
    costs = [row["cost_usd"] for row in issued if row["cost_known"]]
    statuses = {row["status"] for row in attempts}
    status = "ok" if issued and statuses == {"ok"} else (
        "budget_stopped" if "budget_stopped" in statuses else "failed"
    )
    return _base_row(
        source_id,
        run_id,
        stage,
        status=status,
        error_type=next((row["error_type"] for row in reversed(attempts) if row["error_type"]), None),
        provider="openrouter",
        model=MODEL,
        llm_calls=len(issued),
        retry_count=max(0, len(issued) - 1),
        prompt_tokens=sum(int(row["prompt_tokens"] or 0) for row in issued),
        completion_tokens=sum(int(row["completion_tokens"] or 0) for row in issued),
        total_tokens=sum(int(row["total_tokens"] or 0) for row in issued),
        api_cost_usd=sum(costs),
        cost_known=all(row["cost_known"] for row in issued) and bool(issued),
        api_seconds=sum(float(row["latency_seconds"] or 0.0) for row in issued),
        wall_seconds=wall_seconds,
        **values,
    )


def _attempt_audit_rows(
    source_id: str,
    run_id: str,
    stage: str,
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for attempt in attempts:
        rows.append(_base_row(
            source_id,
            run_id,
            stage,
            record_type="llm_attempt",
            status=attempt["status"],
            error_type=attempt["error_type"],
            provider="openrouter",
            model=MODEL,
            llm_calls=1 if attempt["attempt"] is not None else 0,
            retry_count=attempt["retry_index"],
            prompt_tokens=attempt["prompt_tokens"],
            completion_tokens=attempt["completion_tokens"],
            total_tokens=attempt["total_tokens"],
            api_cost_usd=attempt["cost_usd"],
            cost_known=attempt["cost_known"],
            api_seconds=attempt["latency_seconds"],
            wall_seconds=attempt["latency_seconds"],
            notes=f"global_attempt={attempt['attempt']}",
        ))
    return rows


def _deduplicate(results: list[DiscoveryResult]) -> list[DiscoveryResult]:
    seen = set()
    unique = []
    for result in results:
        if result.url not in seen:
            seen.add(result.url)
            unique.append(result)
    return unique


def _select_target(
    results: list[DiscoveryResult], source: pd.Series
) -> tuple[DiscoveryResult, str]:
    source_hosts = {
        urlparse(str(source.source_url)).netloc.lower().removeprefix("www."),
        urlparse(str(source.direct_url)).netloc.lower().removeprefix("www."),
    }
    source_terms = {
        term for term in str(source.title).lower().replace("-", " ").split()
        if len(term.strip(".,:;()")) > 5
    }

    def score(result: DiscoveryResult) -> tuple[int, int]:
        result_host = urlparse(result.url).netloc.lower().removeprefix("www.")
        result_terms = set(result.title.lower().replace("-", " ").split())
        exact_url = int(result.url.rstrip("/") in {
            str(source.source_url).rstrip("/"), str(source.direct_url).rstrip("/")
        })
        same_host = int(result_host in source_hosts)
        title_overlap = len(source_terms & result_terms)
        return exact_url * 100 + same_host * 20 + title_overlap, title_overlap

    candidates = [result for result in results if _is_target(result, source)]
    if candidates:
        target = max(candidates, key=score)
        return target, "discovered_target"
    return DiscoveryResult(
        title=source.title,
        url=source.direct_url,
        source_type=source.source_type,
        relevance_score=1.0,
    ), "prespecified_direct_url_fallback"


def _fetch_and_parse(result: DiscoveryResult) -> tuple[str, float, float, str, str]:
    """Return text, exclusive fetch/parse seconds, and their statuses."""
    fetch_started = time.perf_counter()
    payload: Any = None
    payload_kind = "html"
    if result.url.endswith("fullTextXML"):
        response = requests.get(result.url, timeout=30)
        response.raise_for_status()
        payload = response.content
        payload_kind = "xml"
    elif result.source_type == "pdf" or result.url.lower().endswith(".pdf"):
        payload = DiscoveryAgent().fetch_content(result)
        payload_kind = "pdf"
    else:
        payload = trafilatura.fetch_url(result.url)
    fetch_seconds = time.perf_counter() - fetch_started
    fetch_status = "ok" if payload else "empty"

    parse_started = time.perf_counter()
    text = ""
    try:
        if payload_kind == "pdf" and str(payload).startswith(PDF_PATH_PREFIX):
            path = str(payload)[len(PDF_PATH_PREFIX):]
            try:
                text = Extractor().pdf_to_markdown(path)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        elif payload_kind == "xml" and payload:
            root = ET.fromstring(payload)
            text = "\n".join(value.strip() for value in root.itertext() if value.strip())
        elif payload:
            text = trafilatura.extract(payload) or ""
        parse_status = "ok" if text.strip() else "empty"
    except Exception as exc:
        parse_status = f"failed:{type(exc).__name__}"
    parse_seconds = time.perf_counter() - parse_started
    return text, fetch_seconds, parse_seconds, fetch_status, parse_status


def _time_local_pipeline(
    source: pd.Series,
    params: Parameters,
    run_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for repetition in TIMING_REPETITIONS:
        orchestrator = Orchestrator(
            session_id=f"cost_{source.source_id}_{repetition}",
            base_dir=str(CACHE_DIR / "sessions"),
            model=MODEL,
            env_var=ENV_VAR,
        )
        orchestrator.synthesizer = Synthesizer(
            n_rows=int(source.sample_size), seed=repetition
        )
        orchestrator.run_optimization_loop(
            params, iterations=OPTIMIZATION_ITERATIONS
        )
        trace = orchestrator.optimization_trace
        measurements = {
            "synthesis": sum(row["synthesis_seconds"] for row in trace),
            "validation": sum(row["validation_seconds"] for row in trace),
            "refinement_overhead": sum(
                row["refinement_overhead_seconds"] for row in trace
            ),
            "artifact_write": sum(row["artifact_write_seconds"] for row in trace),
        }
        for stage, seconds in measurements.items():
            rows.append(_base_row(
                source.source_id,
                run_id,
                stage,
                timing_repetition=repetition,
                local_seconds=seconds,
                wall_seconds=seconds,
                documents_processed=1,
                optimization_iterations=OPTIMIZATION_ITERATIONS,
            ))
    return rows


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    stage_raw = raw.loc[raw["record_type"].eq("stage_measurement")].copy()
    numeric = [
        "llm_calls", "retry_count", "prompt_tokens", "completion_tokens",
        "total_tokens", "api_cost_usd", "api_seconds", "network_seconds",
        "local_seconds", "wall_seconds", "documents_processed",
        "optimization_iterations",
    ]
    rows = []
    for (run_id, source_id, stage), group in stage_raw.groupby(
        ["run_id", "source_id", "stage"], sort=False
    ):
        row = _base_row(source_id, run_id, stage)
        row["route"] = next((value for value in group["route"] if pd.notna(value)), None)
        row["status"] = "ok" if group["status"].eq("ok").all() else ";".join(
            sorted(set(group["status"]))
        )
        row["error_type"] = next(
            (value for value in group["error_type"] if pd.notna(value)), None
        )
        row["provider"] = next((value for value in group["provider"] if value != "local"), "local")
        row["model"] = next((value for value in group["model"] if pd.notna(value)), None)
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else None
        row["cost_known"] = bool(group["cost_known"].all())
        row["cache_hit"] = bool(group["cache_hit"].all())
        rows.append(row)

    summary = pd.DataFrame(rows)
    totals = []
    for (run_id, source_id), group in summary.groupby(["run_id", "source_id"], sort=False):
        total = _base_row(source_id, run_id, "total")
        total["route"] = next((value for value in group["route"] if pd.notna(value)), None)
        total["status"] = "ok" if group["status"].eq("ok").all() else "degraded"
        total["provider"] = "mixed"
        total["model"] = MODEL
        for column in (
            "llm_calls", "retry_count", "prompt_tokens", "completion_tokens",
            "total_tokens", "api_cost_usd", "api_seconds", "network_seconds",
            "local_seconds", "wall_seconds",
        ):
            total[column] = float(pd.to_numeric(group[column], errors="coerce").sum())
        total["documents_processed"] = 1
        total["optimization_iterations"] = OPTIMIZATION_ITERATIONS
        total["cost_known"] = bool(group["cost_known"].all())
        totals.append(total)
    summary = pd.concat([summary, pd.DataFrame(totals)], ignore_index=True)

    mean_rows = []
    for stage, group in summary.groupby("stage", sort=False):
        overall = _base_row("mean_one_dataset", "mean_one_dataset", stage)
        overall["route"] = "mean_across_real_papers"
        overall["status"] = "ok" if group["status"].eq("ok").all() else "degraded"
        overall["provider"] = "mixed" if stage == "total" else next(
            (value for value in group["provider"] if value != "local"), "local"
        )
        overall["model"] = MODEL if overall["provider"] != "local" else None
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            overall[column] = float(values.mean()) if values.notna().any() else None
        overall["documents_processed"] = 1
        overall["optimization_iterations"] = (
            OPTIMIZATION_ITERATIONS if stage in {
                "synthesis", "validation", "refinement_overhead",
                "artifact_write", "total",
            } else 0
        )
        overall["cost_known"] = bool(group["cost_known"].all())
        mean_rows.append(overall)
    return pd.concat([summary, pd.DataFrame(mean_rows)], ignore_index=True)


def run(
    mode: str = "cached",
    source_ids: set[str] | None = None,
    initial_known_cost_usd: float = 0.0,
    initial_attempts: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    load_env_safely()
    if mode == "live" and not os.environ.get(ENV_VAR):
        raise RuntimeError(f"{ENV_VAR} is required for live accounting")
    manifest = pd.read_csv(MANIFEST_PATH)
    if source_ids:
        manifest = manifest.loc[manifest["source_id"].isin(source_ids)].copy()
        missing = source_ids - set(manifest["source_id"])
        if missing:
            raise ValueError(f"unknown source IDs: {sorted(missing)}")
    gold = pd.read_csv(GOLD_PATH)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    guard = SpendGuard(
        known_cost_usd=initial_known_cost_usd,
        attempts=initial_attempts,
    )
    raw_rows: list[dict[str, Any]] = []

    for source in manifest.itertuples(index=False):
        source = pd.Series(source._asdict())
        run_id = f"end_to_end_cost_v1:{source.source_id}"
        if mode == "cached":
            cached = ROOT / ".agentdataset_cache" / "real_papers" / f"{source.source_id}.txt"
            if not cached.exists():
                raise FileNotFoundError(f"cached parsed text is missing: {cached}")
            for stage in ("query_expansion", "web_search", "source_suggestion", "fetch", "extraction"):
                raw_rows.append(_base_row(
                    source.source_id, run_id, stage,
                    status="skipped_cached", cache_hit=True,
                ))
            parse_started = time.perf_counter()
            cached.read_text(encoding="utf-8")
            parse_seconds = time.perf_counter() - parse_started
            raw_rows.append(_base_row(
                source.source_id, run_id, "parse", cache_hit=True,
                local_seconds=parse_seconds, wall_seconds=parse_seconds,
                documents_processed=1,
            ))
            params = gold_to_parameters(source.source_id, gold)
            raw_rows.extend(_time_local_pipeline(source, params, run_id))
            continue

        extractor = AccountingExtractor(model=MODEL, env_var=ENV_VAR, guard=guard)
        orchestrator = Orchestrator(
            session_id=f"cost_discovery_{source.source_id}",
            base_dir=str(CACHE_DIR / "sessions"),
            model=MODEL,
            env_var=ENV_VAR,
        )
        orchestrator.extractor = extractor

        attempt_start = len(extractor.attempt_history)
        started = time.perf_counter()
        queries = orchestrator.optimize_query(source.discovery_query)
        elapsed = time.perf_counter() - started
        attempts = extractor.attempt_history[attempt_start:]
        raw_rows.append(_llm_stage_row(source.source_id, run_id, "query_expansion", attempts, elapsed))
        raw_rows.extend(_attempt_audit_rows(source.source_id, run_id, "query_expansion", attempts))

        search_started = time.perf_counter()
        discovered: list[DiscoveryResult] = []
        search_status = "ok"
        search_error = None
        for query_index, query in enumerate(queries):
            request_started = time.perf_counter()
            try:
                found = orchestrator.discovery.search(query)
                discovered.extend(found)
                status, error_type = "ok", None
            except Exception as exc:
                status, error_type = "failed", type(exc).__name__
                search_status, search_error = "failed", error_type
                found = []
            seconds = time.perf_counter() - request_started
            raw_rows.append(_base_row(
                source.source_id, run_id, "web_search",
                record_type="search_request", status=status,
                error_type=error_type, network_seconds=seconds,
                wall_seconds=seconds,
                notes=f"query_index={query_index};results={len(found)}",
            ))
        search_seconds = time.perf_counter() - search_started
        unique = _deduplicate(discovered)
        raw_rows.append(_base_row(
            source.source_id, run_id, "web_search", status=search_status,
            error_type=search_error, network_seconds=search_seconds,
            wall_seconds=search_seconds,
            notes=f"queries={len(queries)};unique_results={len(unique)}",
        ))

        attempt_start = len(extractor.attempt_history)
        started = time.perf_counter()
        suggested = orchestrator.suggest_sources(unique)
        elapsed = time.perf_counter() - started
        attempts = extractor.attempt_history[attempt_start:]
        raw_rows.append(_llm_stage_row(
            source.source_id, run_id, "source_suggestion", attempts, elapsed,
            notes=f"results={len(unique)};suggested={len(suggested)}",
        ))
        raw_rows.extend(_attempt_audit_rows(source.source_id, run_id, "source_suggestion", attempts))

        target, route = _select_target(unique, source)
        try:
            text, fetch_seconds, parse_seconds, fetch_status, parse_status = _fetch_and_parse(target)
            fetch_error = parse_error = None
        except Exception as exc:
            text, fetch_seconds, parse_seconds = "", 0.0, 0.0
            fetch_status = parse_status = "failed"
            fetch_error = parse_error = type(exc).__name__
        raw_rows.append(_base_row(
            source.source_id, run_id, "fetch", route=route,
            status=fetch_status, error_type=fetch_error,
            network_seconds=fetch_seconds, wall_seconds=fetch_seconds,
            documents_processed=1,
        ))
        raw_rows.append(_base_row(
            source.source_id, run_id, "parse", route=route,
            status=parse_status, error_type=parse_error,
            local_seconds=parse_seconds, wall_seconds=parse_seconds,
            documents_processed=1,
            notes=f"parsed_characters={len(text)}",
        ))
        (CACHE_DIR / f"{source.source_id}.txt").write_text(text, encoding="utf-8")

        attempt_start = len(extractor.attempt_history)
        started = time.perf_counter()
        params = extractor.extract_parameters(text, source.title)
        elapsed = time.perf_counter() - started
        attempts = extractor.attempt_history[attempt_start:]
        extraction_status = "ok" if params.meta.extraction_method == "llm" else "degraded_regex"
        extraction_row = _llm_stage_row(
            source.source_id, run_id, "extraction", attempts, elapsed,
            route=route, documents_processed=1,
            notes=f"method={params.meta.extraction_method};variables={len(params.variables)}",
        )
        extraction_row["status"] = extraction_status if attempts else "failed"
        raw_rows.append(extraction_row)
        raw_rows.extend(_attempt_audit_rows(source.source_id, run_id, "extraction", attempts))
        if params.variables:
            raw_rows.extend(_time_local_pipeline(source, params, run_id))

    raw = pd.DataFrame(raw_rows)
    summary = _summarize(raw)
    metadata = {
        "provider": "openrouter",
        "model": MODEL,
        "mode": mode,
        "spend_cap_usd": SPEND_CAP_USD,
        "max_llm_attempts": MAX_LLM_ATTEMPTS,
        "max_retries_per_call": MAX_RETRIES_PER_CALL,
        "known_cost_usd": guard.known_cost_usd,
        "llm_attempts": guard.attempts,
        "sources": manifest["source_id"].tolist(),
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "timing_repetitions": list(TIMING_REPETITIONS),
        "human_source_review_time_included": False,
    }
    return raw, summary, metadata


def main(
    mode: str,
    source_ids: set[str] | None = None,
    merge_existing: bool = False,
    summarize_existing: bool = False,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if mode == "live" else "_cached"
    source_args = "".join(f" --source {source}" for source in sorted(source_ids or set()))
    merge_arg = " --merge-existing" if merge_existing else ""
    command = f"python analysis/end_to_end_cost.py --mode {mode}{source_args}{merge_arg}"
    provenance = collect_provenance(command, TIMING_REPETITIONS)
    raw_path = RESULTS_DIR / f"end_to_end_cost_latency_raw{suffix}.csv"
    if summarize_existing:
        if not raw_path.exists():
            raise FileNotFoundError(f"cannot summarize; raw output is missing: {raw_path}")
        _summarize(pd.read_csv(raw_path)).to_csv(
            RESULTS_DIR / f"end_to_end_cost_latency{suffix}.csv", index=False
        )
        return
    existing = None
    initial_cost = 0.0
    initial_attempts = 0
    if merge_existing:
        if mode != "live" or not source_ids:
            raise ValueError("--merge-existing requires live mode and at least one --source")
        if not raw_path.exists():
            raise FileNotFoundError(f"cannot merge; live raw output is missing: {raw_path}")
        existing = pd.read_csv(raw_path)
        prior_attempts = existing.loc[
            existing["record_type"].eq("llm_attempt")
            & pd.to_numeric(existing["llm_calls"], errors="coerce").eq(1)
        ]
        initial_attempts = len(prior_attempts)
        initial_cost = float(pd.to_numeric(prior_attempts["api_cost_usd"], errors="coerce").sum())

    raw, summary, metadata = run(
        mode,
        source_ids=source_ids,
        initial_known_cost_usd=initial_cost,
        initial_attempts=initial_attempts,
    )
    if existing is not None:
        existing = existing.loc[~existing["source_id"].isin(source_ids)].copy()
        raw = pd.concat([existing, raw], ignore_index=True)
        summary = _summarize(raw)
        final_attempts = raw.loc[
            raw["record_type"].eq("llm_attempt")
            & pd.to_numeric(raw["llm_calls"], errors="coerce").eq(1)
        ]
        metadata.update({
            "known_cost_usd": float(pd.to_numeric(final_attempts["api_cost_usd"], errors="coerce").sum()),
            "llm_attempts": len(final_attempts),
            "physical_known_cost_usd_including_replaced_run": metadata["known_cost_usd"],
            "physical_llm_attempts_including_replaced_run": metadata["llm_attempts"],
            "replaced_sources": sorted(source_ids),
            "sources": sorted(set(raw["source_id"])),
        })
    raw.to_csv(raw_path, index=False)
    summary.to_csv(RESULTS_DIR / f"end_to_end_cost_latency{suffix}.csv", index=False)
    write_json(
        RESULTS_DIR / f"end_to_end_cost_latency_run_manifest{suffix}.json",
        {**provenance, **metadata},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("live", "cached"), default="cached")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    main(
        args.mode,
        source_ids=set(args.source),
        merge_existing=args.merge_existing,
        summarize_existing=args.summarize_existing,
    )
