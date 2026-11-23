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
    SourceRecord,
    Claim,
)
from agentdataset.core.manifest import build_manifest, write_manifest
from agentdataset.core.discovery import DiscoveryAgent, PDF_PATH_PREFIX
from agentdataset.core.extractor import Extractor, strip_json_fence
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.core.cache import ContentCache
from agentdataset.core.review import ClaimReview
from agentdataset.core.validator import Validator

logger = logging.getLogger(__name__)

MAX_SESSIONS = 3  # oldest session dirs beyond this limit are deleted on startup
AGENTDATASET_CACHE_DIR = ".agentdataset_cache"
DEFAULT_SESSION_DIR = str(Path(AGENTDATASET_CACHE_DIR) / "sessions")



def _prune_old_sessions(base_dir: str, keep: int = MAX_SESSIONS) -> None:
    """Delete oldest session directories beyond the keep limit."""
    if not base_dir:
        return
    base = Path(base_dir).resolve()
    if not base.exists() or not base.is_dir():
        return
    try:
        sessions = sorted(
            (p for p in base.iterdir() if p.is_dir() and not p.is_symlink()),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError as e:
        logger.warning("Could not list sessions in %s: %s", base, e)
        return
    if len(sessions) <= keep:
        return
    # Never delete a session another (possibly concurrent) process may still
    # be writing to. Only old, idle directories are eligible for pruning.
    now = time.time()
    eligible = [p for p in sessions if now - p.stat().st_mtime > 24 * 3600]
    for old in eligible[: len(sessions) - keep]:
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
        model: str = "openrouter/auto",
        api_key: str = "",
        env_var: str = "OPENROUTER_API_KEY",
        pdf_backend: str = "pymupdf4llm",
        fallback_models: list[str] | None = None,
        cache=None,
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
        self.extractor = Extractor(model=model, api_key=api_key, env_var=env_var, pdf_backend=pdf_backend, fallback_models=fallback_models)
        self.synthesizer = Synthesizer()
        self.validator = Validator()
        self.cache = cache if cache is not None else ContentCache()

        # Start below any real score so a legitimate first score of 0.0 still ratchets.
        self.best_score = -1.0
        self.best_params: Optional[Parameters] = None
        self.best_data: Optional[pd.DataFrame] = None
        self.optimization_trace: list[dict] = []
        # Evidence-first provenance (§10.9): sources/claims captured during the run.
        self.run_sources: list = []
        self.run_claims: list = []

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

    @staticmethod
    def _normalize_url(url: str) -> str:
        from urllib.parse import urlparse, urlunparse

        try:
            p = urlparse(str(url).strip())
        except Exception:
            return str(url).strip().lower()
        # Drop scheme, trailing slash, and fragment for dedup purposes.
        netloc = p.netloc.lower()
        path = p.path.rstrip("/")
        return f"{netloc}{path}".lower()

    @staticmethod
    def _score_candidate(r: DiscoveryResult) -> float:
        """Inexpensive relevance heuristic for ranking candidates (§10.1).

        Favours sources whose snippets mention statistical quantities and PDFs.
        """
        score = 0.0
        text = " ".join(
            filter(None, [r.title, r.snippet])
        ).lower()
        for kw in ("mean", "std", "standard deviation", "correlation", "distribution",
                   "table", "results", "dataset", "sample"):
            if kw in text:
                score += 1.0
        if r.source_type == "pdf":
            score += 1.5
        # Prefer earlier ranks from their originating query.
        if r.rank is not None:
            score += max(0.0, 3.0 - (r.rank or 1))
        # Incorporate any backend relevance score.
        try:
            score += float(r.relevance_score or 0.0)
        except (TypeError, ValueError):
            pass
        return score

    def run_discovery(
        self,
        query: str,
        max_candidates: int = 20,
        per_query: int | None = None,
        concurrency: int = 1,
    ) -> List[DiscoveryResult]:
        """Phase 0: Discovery (broad funnel + ranked candidates, §10.1).

        Expands the query, searches each angle (optionally in parallel), dedups
        by normalized URL while recording the dedup decision, ranks candidates by
        an inexpensive statistical-availability heuristic, and returns the full
        ranked candidate pool up to ``max_candidates``. Fetching/extraction is
        deliberately left to the caller (user selection).
        """
        optimized_queries = self.optimize_query(query)
        logger.info("Optimized queries: %s", optimized_queries)

        # Collect per-query results (optionally concurrent).
        collected: list[tuple[str, List[DiscoveryResult]]] = []
        if concurrency and concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                future_map = {
                    pool.submit(
                        (lambda qq: self.discovery.search(qq, per_query) if per_query is not None else self.discovery.search(qq)),
                        q,
                    ): q
                    for q in optimized_queries
                }
                for fut in as_completed(future_map):
                    q = future_map[fut]
                    try:
                        collected.append((q, fut.result()))
                    except Exception as e:
                        logger.warning("Discovery search failed for query %r: %s", q, e)
        else:
            for q in optimized_queries:
                try:
                    collected.append((q, (self.discovery.search(q, per_query) if per_query is not None else self.discovery.search(q))))
                except Exception as e:
                    logger.warning("Discovery search failed for query %r: %s", q, e)

        # Dedup by normalized URL, recording provenance + dedup decision.
        seen: dict[str, DiscoveryResult] = {}
        all_results: list[DiscoveryResult] = []
        for q, results in collected:
            for r in results:
                norm = self._normalize_url(r.url)
                if norm in seen:
                    r.dedup_decision = "duplicate"
                    # keep the first (higher-ranked) occurrence
                    continue
                r.dedup_decision = "kept"
                if r.query is None:
                    r.query = q
                seen[norm] = r
                all_results.append(r)

        # Rank by inexpensive heuristic.
        ranked = sorted(all_results, key=self._score_candidate, reverse=True)

        # Trim to candidate pool size (keep full list available for inspection).
        if max_candidates is not None and max_candidates > 0:
            ranked = ranked[:max_candidates]

        # Record provenance into run_sources.
        try:
            for r in ranked:
                self.record_source(r)
        except Exception:
            pass

        logger.info(
            "Discovery funnel: %d queries -> %d candidates (pool cap %s)",
            len(optimized_queries), len(ranked), max_candidates,
        )
        return ranked

    def process_source(self, result: DiscoveryResult) -> Parameters:
        """Phase 1: Extraction.

        Never raises: a failure in fetch/parse/extract is logged and converted
        into an empty Parameters so one bad source cannot abort a batch run.
        """
        from datetime import datetime as _dt

        cache_key = self.cache.key(
            "process_source", getattr(result, "url", ""), self.extractor.model,
            self.extractor.env_var, "v1",
        )
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            try:
                from agentdataset.models.schemas import Parameters as _P
                return _P.model_validate(cached)
            except Exception:
                pass

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
                _empty = Parameters(
                    variables={},
                    correlations={},
                    meta=MetaParams(
                        source=result.title,
                        extracted_at=_dt.now().isoformat(timespec="seconds"),
                        extraction_method="none",
                    ),
                )
                try:
                    self.cache.set(cache_key, _empty.model_dump())
                except Exception:
                    pass
                return _empty

            _params = self.extractor.extract_parameters(text, result.title)
            try:
                self.cache.set(cache_key, _params.model_dump())
            except Exception:
                pass
            return _params
        except Exception as e:
            logger.error("Failed to process source %r: %s", result.title, e, exc_info=True)
            _err = Parameters(
                variables={},
                correlations={},
                meta=MetaParams(
                    source=result.title,
                    extracted_at=_dt.now().isoformat(timespec="seconds"),
                    extraction_method="error",
                ),
            )
            try:
                self.cache.set(cache_key, _err.model_dump())
            except Exception:
                pass
            return _err

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

    def record_source(self, result) -> None:
        """Record a discovered source into the run manifest provenance."""
        try:
            rec = SourceRecord(
                source_id=getattr(result, "url", None) or str(len(self.run_sources)),
                url=getattr(result, "url", ""),
                title=getattr(result, "title", ""),
                source_type=getattr(result, "source_type", "html"),
                query=getattr(result, "query", None),
                rank=getattr(result, "rank", None),
                snippet=getattr(result, "snippet", None),
                retrieved_at=getattr(result, "retrieved_at", None),
                dedup_decision=getattr(result, "dedup_decision", None),
            )
        except Exception:
            return
        self.run_sources.append(rec)

    def record_claims(self, claims) -> None:
        """Record extracted claims into the run manifest provenance."""
        for c in (claims or []):
            self.run_claims.append(c)

    def review_claims_with_agent(self, claims) -> ClaimReview:
        """Have the LLM agent accept, reject, or defer each extracted claim.

        The agent checks each statistic against its evidence passage and the
        declared population/arm/timepoint before the claim reaches synthesis.
        On an LLM failure every claim stays proposed for explicit review.
        """
        import json

        review = ClaimReview(claims)
        if not review.claims:
            return review

        rows = []
        for c in review.claims:
            passage = ""
            if c.evidence and c.evidence.passage:
                passage = str(c.evidence.passage).replace("\n", " ")[:300]
            rows.append(
                f"- id={c.claim_id} | variable={c.variable_name} | statistic={c.statistic_type} "
                f"| value={c.value} | population={c.population} | subgroup={c.subgroup} "
                f"| time={c.time_point} | arm={c.arm} | confidence={c.confidence:.2f} "
                f"| conflict={c.conflict_flag} | evidence={passage!r}"
            )

        prompt = (
            "You are a meticulous data curator. Review each proposed statistic for "
            "grounding in its evidence passage and internal consistency. Reject claims "
            "whose number is unsupported, that conflate populations, arms, or time "
            "points, or that are conflicting without a clear resolution. Defer "
            "borderline cases. Decide for every id. Return JSON only with shape: "
            '{"decisions":[{"claim_id":"...","decision":"accept|reject|defer",'
            '"reason":"..."}]}.\n\n'
            + "\n".join(rows)
        )

        try:
            raw = self.extractor.llm_call(prompt, purpose="claim_review")
            payload = json.loads(strip_json_fence(raw))
            known = {c.claim_id for c in review.claims}
            for decision in payload.get("decisions", []):
                claim_id = str(decision.get("claim_id", "")).strip()
                if claim_id not in known:
                    continue
                action = str(decision.get("decision", "defer")).strip().lower()
                reason = f"agent: {str(decision.get('reason', 'review'))[:300]}"
                if action == "accept":
                    review.accept(claim_id, note=reason)
                elif action == "reject":
                    review.reject(claim_id, note=reason)
                else:
                    review.defer(claim_id, note=reason)
        except Exception as e:
            logger.warning("Agent claim review failed: %s", e)
        return review

    def process_sources_parallel(self, results, max_workers: int = 4):
        """Process multiple sources with bounded concurrency (§10.8)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(self.process_source, r): r for r in results}
            for fut in as_completed(future_map):
                r = future_map[fut]
                try:
                    out[r.url] = fut.result()
                    self.record_source(r)
                    self.record_claims(getattr(self.extractor, "last_claims", []) or [])
                except Exception as e:
                    logger.warning("Failed to process source %r: %s", r.title, e)
        return out

    def write_run_manifest(self, extra: dict | None = None) -> None:
        """Write a versioned run manifest to the session directory (§10.9)."""
        try:
            from agentdataset.models.schemas import SynthesisConfig as _SC
        except Exception:
            _SC = None
        try:
            params = self.best_params
            cfg = None
            if _SC is not None:
                cfg = _SC()
            manifest = build_manifest(
                run_id=self.context.session_id,
                query=None,
                sources=list(self.run_sources),
                raw_claims=list(self.run_claims),
                synthesis_config=cfg,
            )
            if params is not None:
                manifest.parameters_path = "parameters.json"
            if self.best_data is not None:
                manifest.data_path = "data.csv"
            manifest.datacard_path = "DATACARD.md"
            manifest.fidelity = {"best_score": self.best_score}
            if extra:
                manifest.notes = str(extra)
            write_manifest(manifest, Path(self.context.path) / "run_manifest.json")
        except Exception as e:
            logger.warning("Failed to write run manifest: %s", e)


    def run_agent_loop(
        self,
        parameters: Parameters,
        iterations: int = 5,
        source_text: Optional[str] = None,
        max_noise: float = 2.0,
    ):
        if not parameters.variables:
            raise ValueError(
                "Cannot run agent loop: no variables were extracted. "
                "Check the source/extraction step before synthesizing."
            )

        current_params = parameters
        noise_level = 0.02
        self.best_score = -1.0
        self.best_params = None
        self.best_data = None
        self.optimization_trace = []
        last_df = None

        for i in range(iterations):
            iteration_started = time.perf_counter()
            logger.info("Agent loop %d/%d (noise=%.4f)...", i + 1, iterations, noise_level)

            synth_started = time.perf_counter()
            try:
                df = self.synthesizer.synthesize(current_params, noise_level=noise_level)
            except Exception as e:
                logger.error("Synthesis failed on iteration %d: %s", i + 1, e, exc_info=True)
                self.optimization_trace.append({
                    "iteration": i + 1, "noise_level": noise_level,
                    "overall_score": None, "error": "synthesis: %s" % e,
                    "action": "stop", "improved": False,
                })
                continue
            synth_seconds = time.perf_counter() - synth_started
            last_df = df

            val_started = time.perf_counter()
            try:
                report = self.validator.validate(df, current_params, source_text=source_text)
            except Exception as e:
                logger.error("Validation failed on iteration %d: %s", i + 1, e, exc_info=True)
                self.optimization_trace.append({
                    "iteration": i + 1, "noise_level": noise_level,
                    "overall_score": None, "error": "validation: %s" % e,
                    "action": "stop", "improved": False,
                })
                continue
            val_seconds = time.perf_counter() - val_started
            logger.info("  Fidelity Score: %s (trust %s)", report.overall_score, report.trust_score)

            improved = report.overall_score > self.best_score
            if improved:
                self.best_score = report.overall_score
                self.best_params = current_params
                self.best_data = df
                try:
                    df.to_csv(Path(self.context.path) / "data.csv", index=False)
                    with open(Path(self.context.path) / "parameters.json", "w", encoding="utf-8") as f:
                        f.write(current_params.model_dump_json(indent=2))
                    datacard = self.validator.generate_datacard(report, current_params, df)
                    with open(Path(self.context.path) / "DATACARD.md", "w", encoding="utf-8") as f:
                        f.write(datacard)
                except OSError as e:
                    logger.warning("Failed to write artifacts: %s", e)
                self.write_run_manifest()

            critique_text = None
            try:
                critique_text = self.generate_critique(current_params, df, report, source_text)
                report.critique = critique_text
            except Exception as e:
                logger.warning("Critique failed: %s", e)

            decision = self._agent_decide(i + 1, iterations, noise_level, report, critique_text)
            action = decision.get("action", "continue")
            new_noise = decision.get("noise_level", noise_level)
            try:
                new_noise = float(new_noise)
            except (TypeError, ValueError):
                new_noise = noise_level
            new_noise = max(0.001, min(2.0, new_noise))

            trace_row = {
                "iteration": i + 1,
                "noise_level": noise_level,
                "overall_score": report.overall_score,
                "ks_score": report.ks_score,
                "corr_score": report.corr_score,
                "bias_score": report.bias_score,
                "trust_score": report.trust_score,
                "improved": improved,
                "action": action,
                "reason": decision.get("reason", ""),
                "noise_after": new_noise,
                "synthesis_seconds": synth_seconds,
                "validation_seconds": val_seconds,
                "iteration_seconds": time.perf_counter() - iteration_started,
            }
            self.optimization_trace.append(trace_row)

            if action in ("accept", "stop"):
                logger.info("  Agent decided to %s: %s", action, decision.get("reason", ""))
                break

            repair = decision.get("corr_repair")
            if repair in ("drop", "psd", "none"):
                self.synthesizer.corr_repair = repair

            noise_level = new_noise

        if self.best_data is None and last_df is not None:
            self.best_data = last_df
            self.best_params = current_params
        return self.best_score, self.best_data

    def run_optimization_loop(self, *args, **kwargs):
        parameters = kwargs.pop("parameters", args[0] if args else None)
        iterations = kwargs.pop("iterations", 5)
        source_text = kwargs.pop("source_text", None)
        return self.run_agent_loop(parameters, iterations=iterations, source_text=source_text)

    def generate_critique(self, parameters, df, report, source_text=None) -> str:
        prompt = (
            "You are a critical reviewer of a synthetic dataset generated from "
            "literature statistics.\n"
            "Parameters: %s\n" % parameters.model_dump_json()[:4000] +
            "Fidelity report: overall=%s ks=%s corr=%s bias=%s trust=%s\n"
            % (report.overall_score, report.ks_score, report.corr_score, report.bias_score, report.trust_score) +
            "Issues: %s\n" % (report.issues or []) +
            (("Source text excerpt: %s\n" % (source_text or "")[:3000]) if source_text else "") +
            "Write a concise critique (<=150 words): is this dataset faithful to "
            "the source? Any suspicious variables/correlations? Suggested next action."
        )
        return self.extractor.llm_call(prompt, purpose="validation_critique")

    def _agent_decide(self, iteration, total, noise_level, report, critique) -> dict:
        import json as _json
        import re as _re
        prompt = (
            "You are controlling a synthetic-data generation loop.\n"
            "Iteration %d of %d. Current noise_level=%.4f.\n"
            "Fidelity: overall=%s ks=%s corr=%s bias=%s trust=%s. Issues=%s.\n"
            "Critique: %s\n"
            "Decide the next action. Return ONLY JSON with keys: "
            "action (one of: continue, accept, stop, adjust_noise, repair_corr), "
            "noise_level (number), corr_repair (one of: drop, psd, none or null), "
            "reason (short string).\n"
            "Accept if fidelity is good (>=90) and trust high. Stop if no progress. "
            "Otherwise continue, possibly lowering noise or switching corr_repair to psd."
        ) % (
            iteration, total, noise_level,
            report.overall_score, report.ks_score, report.corr_score, report.bias_score, report.trust_score,
            report.issues or [],
            (critique or "")[:500],
        )
        try:
            raw = self.extractor.llm_call(prompt, purpose="agent_decision")
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if m:
                return _json.loads(m.group(0))
            return _json.loads(raw)
        except Exception as e:
            logger.warning("Agent decision failed (%s); using fallback rule.", e)
            if report.overall_score >= 90:
                return {"action": "accept", "noise_level": noise_level, "corr_repair": None, "reason": "fallback accept"}
            return {"action": "continue", "noise_level": max(0.001, noise_level * 0.5), "corr_repair": None, "reason": "fallback lower noise"}
