"""
AgentDataset Orchestrator
The Agentic Workflow Engine
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import pandas as pd
from agentdataset.models.schemas import (
    SessionContext,
    Parameters,
    FidelityReport,
    DiscoveryResult,
    VariableParams,
    CorrelationParams,
    MetaParams,
)
from agentdataset.core.discovery import DiscoveryAgent, PDF_PATH_PREFIX
from agentdataset.core.extractor import Extractor
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.core.validator import Validator

logger = logging.getLogger(__name__)

MAX_NOISE = 2.0
MIN_NOISE = 0.01
PATIENCE = 2  # non-improvement streak length that triggers a pivot
MAX_SESSIONS = 3  # oldest session dirs beyond this limit are deleted on startup
AGENTDATASET_CACHE_DIR = ".agentdataset_cache"
DEFAULT_SESSION_DIR = str(Path(AGENTDATASET_CACHE_DIR) / "sessions")
# §9.7 showed the fidelity landscape falls off above noise 0.05 and fixed 0.01
# beats the adaptive schedule at equal budget — so start quiet.
INITIAL_NOISE = 0.02
EXPLORE_MULT = 1.05
EXPLOIT_FACTOR = 0.5
RESET_MULTIPLE = 2  # reset fires every PATIENCE * RESET_MULTIPLE non-improvements


@dataclass(frozen=True)
class NoisePivotConfig:
    """Knobs for the noise-pivot schedule.

    Defaults reproduce the shipped schedule exactly, so omitting the config
    leaves optimization behaviour bit-identical. §9.7 varies these on a
    predeclared grid; see analysis/noise_pivot_ablation.py.

    `reset_multiple=None` disables the reset branch (a study arm, not a
    supported production setting).
    """

    patience: int = PATIENCE
    initial_noise: float = INITIAL_NOISE
    explore_mult: float = EXPLORE_MULT
    exploit_factor: float = EXPLOIT_FACTOR
    reset_multiple: Optional[int] = RESET_MULTIPLE
    min_noise: float = MIN_NOISE
    max_noise: float = MAX_NOISE

    def __post_init__(self):
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.reset_multiple is not None and self.reset_multiple < 1:
            raise ValueError(
                f"reset_multiple must be >= 1 or None, got {self.reset_multiple}"
            )
        if not 0 < self.min_noise <= self.max_noise:
            raise ValueError(
                f"require 0 < min_noise <= max_noise, got "
                f"({self.min_noise}, {self.max_noise})"
            )


DEFAULT_PIVOT = NoisePivotConfig()


def next_noise(
    streak: int, noise: float, cfg: NoisePivotConfig = DEFAULT_PIVOT
) -> tuple[float, str]:
    """One step of the noise-pivot schedule, as a pure function.

    `streak` is the non-improvement streak length *after* incrementing for the
    current failed iteration. Branch order is reset → exploit → explore, which
    matters because a full-cycle streak also satisfies the exploit condition.

    Returns (new_noise, phase) with phase in {"reset", "exploit", "explore"}.
    """
    full_cycle = None if cfg.reset_multiple is None else cfg.patience * cfg.reset_multiple

    if full_cycle is not None and streak % full_cycle == 0:
        return cfg.initial_noise, "reset"
    if streak % cfg.patience == 0:
        return max(noise * cfg.exploit_factor, cfg.min_noise), "exploit"
    return min(noise * cfg.explore_mult, cfg.max_noise), "explore"


def _prune_old_sessions(base_dir: str, keep: int = MAX_SESSIONS) -> None:
    """Delete oldest session directories beyond the keep limit."""
    if not base_dir:
        return
    base = Path(base_dir).resolve()
    if not base.exists() or not base.is_dir():
        return
    try:
        sessions = sorted(
            p for p in base.iterdir() if p.is_dir() and not p.is_symlink()
        )
    except OSError as e:
        logger.warning("Could not list sessions in %s: %s", base, e)
        return
    if len(sessions) <= keep:
        return
    for old in sessions[:-keep]:
        try:
            import shutil

            shutil.rmtree(old)
            logger.info("Pruned old session: %s", old)
        except OSError as e:
            logger.warning("Could not prune session %s: %s", old, e)


class Orchestrator:
    def __init__(
        self,
        session_id: str,
        base_dir: str = DEFAULT_SESSION_DIR,
        model: str = "gpt-4o",
        api_key: str = "",
        env_var: str = "OPENAI_API_KEY",
    ):
        _prune_old_sessions(base_dir)
        self.context = SessionContext(
            session_id=session_id, path=str(Path(base_dir) / session_id)
        )
        os.makedirs(self.context.path, exist_ok=True)

        if "OPENROUTER" in (env_var or "").upper():
            or_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        else:
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
        or_key = api_key if "OPENROUTER" in (env_var or "").upper() else os.environ.get("OPENROUTER_API_KEY", "")
        if not or_key:
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.discovery = DiscoveryAgent(
            api_key=or_key,
            model=model,
        )
        self.extractor = Extractor(model=model, api_key=api_key, env_var=env_var)
        self.synthesizer = Synthesizer()
        self.validator = Validator()

        # Start below any real score so a legitimate first score of 0.0 still ratchets.
        self.best_score = -1.0
        self.best_params: Optional[Parameters] = None
        self.best_data: Optional[pd.DataFrame] = None
        self.optimization_trace: list[dict] = []

    def merge_parameters(self, params_list: List[Parameters]) -> Parameters:
        """Merge parameters extracted from multiple sources.

        Same variable name across sources → average mean and std.
        Unique variable names → include all.
        Same correlation pair → average correlation value.
        """
        if not params_list:
            raise ValueError("params_list must not be empty")
        if len(params_list) == 1:
            return params_list[0]

        # Accumulate mean/std per variable name across sources
        var_accum: dict = {}  # name -> {"mean": [], "std": [], "distribution": str}
        for params in params_list:
            for name, vp in params.variables.items():
                if name not in var_accum:
                    var_accum[name] = {
                        "mean": [],
                        "std": [],
                        "distribution": vp.distribution,
                    }
                var_accum[name]["mean"].append(vp.mean)
                var_accum[name]["std"].append(vp.std)

        merged_variables: dict = {}
        for name, acc in var_accum.items():
            mean = sum(acc["mean"]) / len(acc["mean"])
            std = sum(acc["std"]) / len(acc["std"])
            merged_variables[name] = VariableParams(
                name=name,
                distribution=acc["distribution"],
                mean=mean,
                std=std,
                min=mean - 3 * std,
                max=mean + 3 * std,
            )

        # Merge correlations: same pair key → average correlation
        corr_accum: dict = {}  # key -> {"values": [], "var1": str, "var2": str, "direction": str}
        for params in params_list:
            for key, cp in params.correlations.items():
                if key not in corr_accum:
                    corr_accum[key] = {
                        "values": [],
                        "var1": cp.var1,
                        "var2": cp.var2,
                        "direction": cp.direction,
                    }
                corr_accum[key]["values"].append(cp.correlation)

        merged_correlations: dict = {}
        for key, acc in corr_accum.items():
            merged_correlations[key] = CorrelationParams(
                var1=acc["var1"],
                var2=acc["var2"],
                correlation=sum(acc["values"]) / len(acc["values"]),
                direction=acc["direction"],
            )

        sources = ", ".join(p.meta.source for p in params_list)
        return Parameters(
            variables=merged_variables,
            correlations=merged_correlations,
            meta=MetaParams(
                source=f"merged({sources})",
                extracted_at=params_list[0].meta.extracted_at,
                extraction_method=params_list[0].meta.extraction_method,
            ),
        )

    def optimize_query(self, query: str) -> List[str]:
        """Use LLM to expand a vague query into multiple search-optimized strings.

        Returns a list of optimized search queries designed to find statistical data.
        """
        prompt = (
            f"User wants to find statistical data/distributions for: {query}. "
            "Generate 3-5 diverse, highly specific search queries. "
            "Focus on finding research papers, PDFs, and reports containing means, "
            "standard deviations, and correlation matrices. "
            "Return ONLY the queries, one per line, no numbering, no quotes."
        )

        try:
            response = self.extractor.llm_call(
                prompt, purpose="query_expansion"
            )  # Reusing extractor's LLM interface
            queries = [q.strip() for q in response.split("\n") if q.strip()]
            # Always include the original query for safety
            return [query] + queries[:4]
        except Exception as e:
            logger.error("Query optimization failed: %s", e)
            return [query]

    def run_discovery(self, query: str) -> List[DiscoveryResult]:
        """Phase 0: Discovery (with query optimization)."""
        optimized_queries = self.optimize_query(query)
        logger.info("Optimized queries: %s", optimized_queries)

        all_results = []
        for q in optimized_queries:
            try:
                results = self.discovery.search(q)
            except Exception as e:
                logger.warning("Discovery search failed for query %r: %s", q, e)
                continue
            all_results.extend(results)

        # De-duplicate by URL to avoid redundant sources from different queries
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                unique_results.append(r)
                seen_urls.add(r.url)

        return unique_results

    def process_source(self, result: DiscoveryResult) -> Parameters:
        """Phase 1: Extraction.

        Never raises: a failure in fetch/parse/extract is logged and converted
        into an empty Parameters so one bad source cannot abort a batch run.
        """
        from datetime import datetime as _dt

        try:
            content = self.discovery.fetch_content(result)

            if content.startswith(PDF_PATH_PREFIX):
                pdf_path = content[len(PDF_PATH_PREFIX):]

                try:
                    text = self.extractor.pdf_to_markdown(pdf_path)
                finally:
                    try:
                        os.remove(pdf_path)
                    except OSError:
                        pass
            else:
                text = content

            if not (text or "").strip():
                logger.warning("No text recovered from source %r; returning empty parameters.", result.title)
                return Parameters(
                    variables={},
                    correlations={},
                    meta=MetaParams(
                        source=result.title,
                        extracted_at=_dt.now().isoformat(timespec="seconds"),
                        extraction_method="none",
                    ),
                )

            return self.extractor.extract_parameters(text, result.title)
        except Exception as e:
            logger.error("Failed to process source %r: %s", result.title, e, exc_info=True)
            return Parameters(
                variables={},
                correlations={},
                meta=MetaParams(
                    source=result.title,
                    extracted_at=_dt.now().isoformat(timespec="seconds"),
                    extraction_method="error",
                ),
            )

    def suggest_sources(self, results: List[DiscoveryResult]) -> List[int]:
        """Analyze discovery results and suggest which ones likely contain statistical data.

        Returns a list of indices of the suggested sources.
        """
        if not results:
            return []

        # Prepare a list of sources for the LLM to review
        sources_text = "\n".join(
            [
                f"ID {i}: {res.title}\nSnippet: {res.snippet}"
                for i, res in enumerate(results)
            ]
        )

        prompt = (
            f"Review these search results for a research project. "
            "Identify which sources are most likely to contain raw statistical parameters "
            "(means, std devs, correlations, distributions) rather than just general summaries. "
            f"\n\nSources:\n{sources_text}\n\n"
            "Return ONLY a comma-separated list of IDs (e.g., '0,2,5'). "
            "If none are promising, return 'None'."
        )

        try:
            response = self.extractor.llm_call(
                prompt, purpose="source_suggestion"
            ).strip()
            if response.lower() == "none" or not response:
                return []
            # Extract digits from the response
            import re

            ids = [int(i) for i in re.findall(r"\d+", response)]
            # Filter to ensure IDs are within valid range
            return [i for i in ids if 0 <= i < len(results)]
        except Exception as e:
            logger.error("Source suggestion failed: %s", e)
            return []

    def run_optimization_loop(
        self,
        parameters: Parameters,
        iterations: int = 5,
        pivot: Optional[NoisePivotConfig] = None,
        source_text: Optional[str] = None,
    ):
        """Phase 2 & 3: The Engine (Synthesis-Validation Loop).

        Noise pivot strategy — patience + reset:
          - Every PATIENCE consecutive non-improvements  → exploit: halve noise
          - Every PATIENCE*2 consecutive non-improvements → reset to initial noise
          - Single non-improvement steps (streak % PATIENCE != 0) → explore: raise noise

        `pivot` overrides the schedule; omitting it uses DEFAULT_PIVOT, which
        reproduces the shipped constants exactly.

        Raises ValueError if `parameters` has no variables — synthesizing an empty
        frame would otherwise produce a spurious score and write an empty data.csv.
        """
        if not parameters.variables:
            raise ValueError(
                "Cannot run optimization loop: no variables were extracted. "
                "Check the source/extraction step before synthesizing."
            )

        cfg = pivot if pivot is not None else DEFAULT_PIVOT
        current_params = parameters
        noise_level = cfg.initial_noise
        no_improve_streak = 0
        self.optimization_trace = []
        last_df = None

        for i in range(iterations):
            iteration_started = time.perf_counter()
            logger.info("Loop %d/%d (noise=%.4f)...", i + 1, iterations, noise_level)

            # Synthesis
            synthesis_started = time.perf_counter()
            try:
                df = self.synthesizer.synthesize(current_params, noise_level=noise_level)
            except Exception as e:
                logger.error("Synthesis failed on iteration %d: %s", i + 1, e, exc_info=True)
                no_improve_streak += 1
                noise_level, phase = next_noise(no_improve_streak, noise_level, cfg)
                trace_row = {
                    "iteration": i + 1,
                    "noise_level": noise_level,
                    "overall_score": None,
                    "error": "synthesis: %s: %s" % (type(e).__name__, e),
                    "improved": False,
                    "phase": phase,
                    "noise_after": noise_level,
                    "synthesis_seconds": time.perf_counter() - synthesis_started,
                    "validation_seconds": 0.0,
                    "artifact_write_seconds": 0.0,
                }
                self.optimization_trace.append(trace_row)
                trace_row["iteration_seconds"] = time.perf_counter() - iteration_started
                continue
            synthesis_seconds = time.perf_counter() - synthesis_started

            # Validation
            validation_started = time.perf_counter()
            try:
                report = self.validator.validate(df, current_params, source_text=source_text)
            except Exception as e:
                logger.error("Validation failed on iteration %d: %s", i + 1, e, exc_info=True)
                no_improve_streak += 1
                noise_level, phase = next_noise(no_improve_streak, noise_level, cfg)
                trace_row = {
                    "iteration": i + 1,
                    "noise_level": noise_level,
                    "overall_score": None,
                    "error": "validation: %s: %s" % (type(e).__name__, e),
                    "improved": False,
                    "phase": phase,
                    "noise_after": noise_level,
                    "synthesis_seconds": synthesis_seconds,
                    "validation_seconds": time.perf_counter() - validation_started,
                    "artifact_write_seconds": 0.0,
                }
                self.optimization_trace.append(trace_row)
                last_df = df
                trace_row["iteration_seconds"] = time.perf_counter() - iteration_started
                continue
            validation_seconds = time.perf_counter() - validation_started
            logger.info("  Fidelity Score: %s", report.overall_score)
            last_df = df
            improved = report.overall_score > self.best_score
            trace_row = {
                "iteration": i + 1,
                "noise_level": noise_level,
                "overall_score": report.overall_score,
                "ks_score": report.ks_score,
                "corr_score": report.corr_score,
                "bias_score": report.bias_score,
                "improved": improved,
                # Filled in below once the pivot branch is known. "keep" means the
                # ratchet accepted this iteration, so no pivot fired.
                "phase": "keep",
                "noise_after": noise_level,
                "synthesis_seconds": synthesis_seconds,
                "validation_seconds": validation_seconds,
                "artifact_write_seconds": 0.0,
            }
            self.optimization_trace.append(trace_row)

            # Ratchet Logic
            if improved:
                logger.info("  [KEEP] New best score!")
                self.best_score = report.overall_score
                self.best_params = current_params
                self.best_data = df
                no_improve_streak = 0

                # Save artifacts — I/O failure must not abort the loop / lose progress.
                artifact_started = time.perf_counter()
                try:
                    df.to_csv(Path(self.context.path) / "data.csv", index=False)
                    with open(Path(self.context.path) / "parameters.json", "w", encoding="utf-8") as f:
                        f.write(current_params.model_dump_json(indent=2))
                    datacard = self.validator.generate_datacard(
                        report, current_params, df
                    )
                    with open(Path(self.context.path) / "DATACARD.md", "w", encoding="utf-8") as f:
                        f.write(datacard)
                except OSError as e:
                    logger.warning(
                        "Failed to write session artifacts to %s: %s",
                        self.context.path,
                        e,
                    )
                finally:
                    trace_row["artifact_write_seconds"] = (
                        time.perf_counter() - artifact_started
                    )
            else:
                no_improve_streak += 1
                noise_level, phase = next_noise(no_improve_streak, noise_level, cfg)
                trace_row["phase"] = phase
                trace_row["noise_after"] = noise_level
                logger.info(
                    "  [DISCARD] Streak=%d — %s: noise → %.4f",
                    no_improve_streak,
                    phase,
                    noise_level,
                )

            trace_row["iteration_seconds"] = time.perf_counter() - iteration_started
            trace_row["refinement_overhead_seconds"] = max(
                0.0,
                trace_row["iteration_seconds"]
                - trace_row["synthesis_seconds"]
                - trace_row["validation_seconds"]
                - trace_row["artifact_write_seconds"],
            )

        if self.best_data is None and last_df is not None:
            self.best_data = last_df
        return self.best_score, self.best_data
