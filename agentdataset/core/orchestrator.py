"""
AgentDataset Orchestrator
The Autonomous Engine (Brain)
"""

import logging
import os
import time
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


def _prune_old_sessions(base_dir: str, keep: int = MAX_SESSIONS) -> None:
    """Delete oldest session directories beyond the keep limit."""
    base = Path(base_dir)
    if not base.exists():
        return
    sessions = sorted(p for p in base.iterdir() if p.is_dir())
    for old in sessions[:-keep] if len(sessions) > keep else []:
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
        base_dir: str = "sessions",
        model: str = "gpt-4o",
        api_key: str = "",
        env_var: str = "OPENAI_API_KEY",
    ):
        _prune_old_sessions(base_dir)
        self.context = SessionContext(
            session_id=session_id, path=str(Path(base_dir) / session_id)
        )
        os.makedirs(self.context.path, exist_ok=True)

        self.discovery = DiscoveryAgent()
        self.extractor = Extractor(model=model, api_key=api_key, env_var=env_var)
        self.synthesizer = Synthesizer()
        self.validator = Validator()

        # Start below any real score so a legitimate first score of 0.0 still ratchets.
        self.best_score = -1.0
        self.best_params: Optional[Parameters] = None
        self.best_data: Optional[pd.DataFrame] = None

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
                prompt
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
            results = self.discovery.search(q)
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
        """Phase 1: Extraction."""
        content = self.discovery.fetch_content(result)

        if content.startswith(PDF_PATH_PREFIX):
            pdf_path = content[len(PDF_PATH_PREFIX) :]
            try:
                text = self.extractor.pdf_to_markdown(pdf_path)
            finally:
                # Clean up temp file regardless of extraction outcome
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
        else:
            text = content

        return self.extractor.extract_parameters(text, result.title)

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
            response = self.extractor.llm_call(prompt).strip()
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

    def run_optimization_loop(self, parameters: Parameters, iterations: int = 5):
        """Phase 2 & 3: The Engine (Synthesis-Validation Loop).

        Noise pivot strategy — patience + reset:
          - Every PATIENCE consecutive non-improvements  → exploit: halve noise
          - Every PATIENCE*2 consecutive non-improvements → reset to initial noise
          - Single non-improvement steps (streak % PATIENCE != 0) → explore: raise noise

        Raises ValueError if `parameters` has no variables — synthesizing an empty
        frame would otherwise produce a spurious score and write an empty data.csv.
        """
        if not parameters.variables:
            raise ValueError(
                "Cannot run optimization loop: no variables were extracted. "
                "Check the source/extraction step before synthesizing."
            )

        current_params = parameters
        initial_noise = 0.1
        noise_level = initial_noise
        no_improve_streak = 0

        for i in range(iterations):
            logger.info("Loop %d/%d (noise=%.4f)...", i + 1, iterations, noise_level)

            # Synthesis
            df = self.synthesizer.synthesize(current_params, noise_level=noise_level)

            # Validation
            report = self.validator.validate(df, current_params)
            logger.info("  Fidelity Score: %s", report.overall_score)

            # Ratchet Logic
            if report.overall_score > self.best_score:
                logger.info("  [KEEP] New best score!")
                self.best_score = report.overall_score
                self.best_params = current_params
                self.best_data = df
                no_improve_streak = 0

                # Save artifacts — I/O failure must not abort the loop / lose progress.
                try:
                    df.to_csv(Path(self.context.path) / "data.csv", index=False)
                    with open(Path(self.context.path) / "parameters.json", "w") as f:
                        f.write(current_params.model_dump_json(indent=2))
                    datacard = self.validator.generate_datacard(
                        report, current_params, df
                    )
                    with open(Path(self.context.path) / "DATACARD.md", "w") as f:
                        f.write(datacard)
                except OSError as e:
                    logger.warning(
                        "Failed to write session artifacts to %s: %s",
                        self.context.path,
                        e,
                    )
            else:
                no_improve_streak += 1
                full_cycle = PATIENCE * 2

                if no_improve_streak % full_cycle == 0:
                    # Full cycle with no gain — reset to initial noise
                    noise_level = initial_noise
                    logger.info(
                        "  [DISCARD] Streak=%d — reset noise to %.4f",
                        no_improve_streak,
                        noise_level,
                    )
                elif no_improve_streak % PATIENCE == 0:
                    # Exploit phase: tighten noise to improve fit
                    noise_level = max(noise_level * 0.5, MIN_NOISE)
                    logger.info(
                        "  [DISCARD] Streak=%d — exploit: noise → %.4f",
                        no_improve_streak,
                        noise_level,
                    )
                else:
                    # Explore phase: expand noise for more variance
                    noise_level = min(noise_level * 1.1, MAX_NOISE)
                    logger.info(
                        "  [DISCARD] Streak=%d — explore: noise → %.4f",
                        no_improve_streak,
                        noise_level,
                    )

        return self.best_score, self.best_data
