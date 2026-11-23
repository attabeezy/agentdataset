"""
AgentDataset Extractor
PDF/Markdown -> Parameters (Pydantic)
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any
import re
import time
import pymupdf as fitz  # PyMuPDF (2.x)
import litellm
from litellm import completion
from agentdataset.models.schemas import (
    Parameters,
    VariableParams,
    CorrelationParams,
    MetaParams,
    Claim,
    EvidenceLocator,
)

logger = logging.getLogger(__name__)

# The Caveman Protocol: maximally compressed instructions. SUPERSEDED as the
# production prompt — retained because the paper documents it and the §9.6
# ablation still measures it (benchmark.run_extraction_challenge).
#
# The ablation found the compression is not free: over 930 extractions across
# three models it scored F1 .710 against .778 for NATURAL_LANGUAGE_PROMPT below
# (paired Wilcoxon p=.0001), losing precision to a hallucination rate of .154
# against .098 — while saving only 3.5% of per-call cost, because completion
# tokens dominate this workload. See results/extraction_challenge_summary.csv.
CAVEMAN_PROMPT = """
You are an expert statistician.
Strip prose.
Extract variables, distributions, correlations.
Output strict JSON matching this schema exactly:
{
  "variables": {
    "<name>": {"distribution": "normal|uniform|gamma|categorical", "mean": 0.0, "std": 1.0, "min": null, "max": null, "categories": null}
  },
  "correlations": {
    "<key>": {"var1": "<name>", "var2": "<name>", "correlation": 0.5, "direction": "positive|negative"}
  }
}
"categories" is only used when distribution is "categorical": an object mapping
each category label to its probability, e.g. {"yes": 0.3, "no": 0.7}.
No fluff. No greeting. Output JSON only.
"""

# Requests exactly the same schema as CAVEMAN_PROMPT — same distributions
# (including "categorical"), same "categories" field — differing only in prose
# style. That parity is load-bearing for the §9.6 prompt ablation and is enforced
# by tests/test_extraction_challenge.py::test_prompt_schema_parity. Do not edit
# one prompt's field set without the other.
NATURAL_LANGUAGE_PROMPT = """
You are an expert statistician with many years of experience analyzing research
papers and extracting statistical information from them. Please carefully read
through the following text and identify all of the statistical variables that
are explicitly reported with numbers, along with their probability distributions
(such as normal, uniform, gamma, or categorical), their means, and their
standard deviations. Also, please identify any correlations that are described
between pairs of variables, including the strength and direction (positive or
negative) of each correlation.

Grounding rules — follow these strictly:
- Extract a statistic ONLY if a numeric value is explicitly stated in the text.
  Never infer, estimate, or invent a number.
- Every correlation's "var1" and "var2" MUST be keys already present in your
  "variables" object. If a marginal was never reported, omit that correlation
  rather than referencing an undeclared variable.
- Do NOT treat standard errors (SE), confidence intervals (CI), or p-values as
  standard deviations. If only SE/CI is given, omit the variable.
- Do NOT extract from negated findings without numbers (e.g. "no difference",
  "not significant") — return nothing for those.
- If the text reports conflicting values, prefer the results Table over the
  abstract; note nothing, just choose the Table value.
- If no reportable statistics are present, return empty objects:
  {"variables": {}, "correlations": {}} — abstention is correct.

Please format your response as a JSON object that matches the following schema
exactly:
{
  "variables": {
    "<name>": {"distribution": "normal|uniform|gamma|categorical", "mean": 0.0, "std": 1.0, "min": null, "max": null, "categories": null}
  },
  "correlations": {
    "<key>": {"var1": "<name>", "var2": "<name>", "correlation": 0.5, "direction": "positive|negative"}
  }
}

The "categories" field should only be used when the distribution is
"categorical". In that case it should be an object that maps each category
label to its probability, for example {"yes": 0.3, "no": 0.7}.

Please make sure to only output the JSON object itself and nothing else - no
introductory text, no explanations, and no markdown code fences.
"""

# Chunking for long documents. The LLM call truncates at ~10k chars, which drops
# tables at the end of real papers. Longer inputs are split into overlapping
# windows and merged, so recall no longer depends on where the table sits.
LLM_CHUNK_SIZE = 8000
LLM_CHUNK_OVERLAP = 500

# The production system prompt. Switched from CAVEMAN_PROMPT to
# NATURAL_LANGUAGE_PROMPT on the strength of the §9.6 ablation (see above).
EXTRACTION_PROMPT = NATURAL_LANGUAGE_PROMPT

# Separator between a keyword and its numeric value: "= 3.5", ": 3.5", "is 3.5", "of 3.5"
def _resolve_correlation_name(
    name: str, variables: Dict[str, Any], by_normalized: Dict[str, list]
) -> str:
    """Map a correlation's variable name onto a declared variable, if it can be.

    Strictly a fallback. An exact hit is returned untouched, so a response whose
    correlations already reference declared variables is passed through
    unchanged and no previously reported number can move. Only when the literal
    name matches nothing do we retry on the normalized form, and only when that
    normalized form identifies exactly one declared variable -- two variables
    that differ solely by case or punctuation are genuinely ambiguous, and
    guessing between them would silently merge distinct columns.

    An unresolvable name is returned as given rather than dropped: the
    synthesizer then records it in `n_pairs_skipped_unknown_var`, which is how a
    correlation whose variable was never extracted stays visible instead of
    vanishing. That case is legitimate -- papers often report a correlation
    without reporting both marginals.
    """
    if name in variables:
        return name
    candidates = by_normalized.get(_normalize_variable_name(name), [])
    if len(candidates) == 1:
        return candidates[0]
    return name


def _normalize_variable_name(name: str) -> str:
    """Canonical form used to match correlation pairs against variables.

    Prose and correlation clauses disagree about spelling: source text says
    "heart rate" while the correlation clause says "heart_rate". Both sides go
    through this, so they meet in the middle.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    return cleaned.strip("_").lower()


# Some providers ignore response_format={"type": "json_object"} and wrap the
# object in a markdown fence. Anthropic models via OpenRouter do this on every
# call; OpenAI and Google honour the flag. Without this strip, json.loads fails
# and extract_parameters silently falls back to regex for those providers — a
# failure that looks like "the LLM could not extract anything" in the results.
# Discovered by the P1 9.6 extraction ablation.
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def strip_json_fence(content: str) -> str:
    """Return `content` with a wrapping markdown code fence removed, if present."""
    match = _JSON_FENCE.search(content or "")
    return match.group(1) if match else content


def _chunk_text(text: str, size: int = 8000, overlap: int = 500) -> list[str]:
    """Split long documents into overlapping windows for LLM extraction."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def _merge_chunk_results(
    results: list[tuple[Dict, Dict]],
) -> tuple[Dict, Dict]:
    """Merge per-chunk (variables, correlations), deduplicating across chunks.

    Variables are grouped by their normalized name, so ``age`` and ``Age`` (or
    ``heart rate`` / ``heart_rate``) collapse into a single entry instead of
    becoming two columns. Their mean/std are averaged across occurrences.

    Correlations are grouped by their normalized, order-insensitive
    (var1, var2) pair, so the same pair reported in two overlapping windows
    no longer survives twice under two different chunk-local keys (``c1``,
    ``c2``, ...) while two distinct pairs that both happened to be ``c1`` in
    different chunks no longer collapse into one.
    """
    from agentdataset.models.schemas import VariableParams as _VP
    from agentdataset.models.schemas import CorrelationParams as _CP
    from collections import defaultdict

    if not results:
        return {}, {}

    # --- Variables: group by normalized name ---
    var_groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for variables, _ in results:
        for name, vp in (variables or {}).items():
            var_groups[_normalize_variable_name(name)].append((name, vp))

    merged_vars: dict = {}
    for items in var_groups.values():
        # Canonical name: first occurrence (deterministic).
        canonical = items[0][0]
        means = [vp.mean for _, vp in items]
        stds = [vp.std for _, vp in items]
        mean = sum(means) / len(means)
        std = sum(stds) / len(stds)
        if not (std is not None and std > 0):
            continue
        # Prefer optional fields carried by some chunk over a None from another.
        distribution = items[0][1].distribution
        categories = next(
            (vp.categories for _, vp in items if getattr(vp, "categories", None)),
            items[0][1].categories,
        )
        min_val = next(
            (vp.min for _, vp in items if getattr(vp, "min", None) is not None),
            None,
        )
        max_val = next(
            (vp.max for _, vp in items if getattr(vp, "max", None) is not None),
            None,
        )
        try:
            merged_vars[canonical] = _VP(
                name=canonical,
                distribution=distribution,
                mean=mean,
                std=std,
                min=min_val if min_val is not None else mean - 3 * std,
                max=max_val if max_val is not None else mean + 3 * std,
                categories=categories,
            )
        except Exception:
            continue

    # --- Correlations: group by normalized, order-insensitive pair ---
    corr_groups: dict[tuple[str, str], list] = defaultdict(list)
    for _, correlations in results:
        for cp in (correlations or {}).values():
            v1n = _normalize_variable_name(getattr(cp, "var1", "") or "")
            v2n = _normalize_variable_name(getattr(cp, "var2", "") or "")
            pair_key = tuple(sorted((v1n, v2n)))
            corr_groups[pair_key].append(cp)

    # Normalized -> canonical variable name map for resolution.
    by_normalized: dict[str, list] = defaultdict(list)
    for name in merged_vars:
        by_normalized[_normalize_variable_name(name)].append(name)

    merged_corrs: dict = {}
    for pair_key, cps in corr_groups.items():
        # Self-correlations are always 1.0 and never a useful dependence statement.
        if pair_key[0] and pair_key[0] == pair_key[1]:
            continue
        vals = [cp.correlation for cp in cps]
        corr_val = sum(vals) / len(vals)
        cp0 = cps[0]
        var1 = _resolve_correlation_name(cp0.var1, merged_vars, by_normalized)
        var2 = _resolve_correlation_name(cp0.var2, merged_vars, by_normalized)
        direction = cp0.direction
        key = f"{var1}__{var2}"
        if key in merged_corrs:
            i = 2
            while f"{key}__{i}" in merged_corrs:
                i += 1
            key = f"{key}__{i}"
        try:
            merged_corrs[key] = _CP(
                var1=var1,
                var2=var2,
                correlation=corr_val,
                direction=direction,
            )
        except Exception:
            continue

    return merged_vars, merged_corrs


class Extractor:
    def __init__(
        self, model: str = "gpt-4o", api_key: str = "", env_var: str = "OPENAI_API_KEY"
    ):
        self.model = model
        self.api_key = api_key
        self.env_var = env_var
        self.call_history: list[dict] = []
        # Last per-chunk (variables, correlations) pairs from an LLM extraction,
        # kept so chunk-merge behaviour can be replayed/tested offline.
        self.last_chunk_results: list[tuple[Dict, Dict]] = []
        # Evidence captured during the most recent extraction: maps a variable
        # name or correlation key to the supporting passage (best-effort).
        self.last_evidence: dict[str, str] = {}
        # Claims produced during the most recent extraction (§10.2).
        self.last_claims: list = []

    def _record_completion(
        self, response, started_at: float, purpose: str, model: str | None = None
    ) -> None:
        """Capture non-secret per-call usage for end-to-end accounting."""
        usage = getattr(response, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:
            # Unmapped models (new OpenRouter IDs in old litellm) raise here;
            # OpenRouter itself reports upstream cost on usage, so prefer that
            # over recording NaN and losing the spend signal.
            try:
                cost = float(usage.cost)
            except (TypeError, ValueError, AttributeError):
                cost = float("nan")
        self.call_history.append({
            "purpose": purpose,
            "model": model or self.model,
            "status": "ok",
            "error_type": None,
            "latency_seconds": time.perf_counter() - started_at,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "cost_usd": cost,
        })

    def __init__(
        self, model: str = "openrouter/auto", api_key: str = "", env_var: str = "OPENROUTER_API_KEY",
        pdf_backend: str = "pymupdf4llm",
        fallback_models: list[str] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.env_var = env_var
        self.pdf_backend = pdf_backend
        # Fallback chain used when the primary model fails. Defaults to the
        # primary only; callers can pass free-model slugs for resilience.
        self.fallback_models = fallback_models if fallback_models is not None else [model]
        self.call_history: list[dict] = []
        self.last_chunk_results: list[tuple[Dict, Dict]] = []
        self.last_evidence: dict[str, str] = {}
        self.last_claims: list = []

    @property
    def _model_chain(self) -> list[str]:
        chain = [self.model] + [m for m in self.fallback_models if m != self.model]
        seen = set()
        out = []
        for m in chain:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out or [self.model]

    @property
    def _model_chain(self) -> list[str]:
        chain = [self.model] + [m for m in self.fallback_models if m != self.model]
        # de-dupe preserving order
        seen = set()
        out = []
        for m in chain:
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out or [self.model]
        self.call_history: list[dict] = []
        self.last_chunk_results: list[tuple[Dict, Dict]] = []
        self.last_evidence: dict[str, str] = {}
        self.last_claims: list = []

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to clean Markdown/text for LLM consumption.

        Default backend is pymupdf4llm (layout-aware Markdown). Falls back to
        PyMuPDF raw text if the chosen backend is unavailable or yields nothing.
        """
        backend = (self.pdf_backend or "pymupdf4llm").lower()
        text = ""
        if backend == "pymupdf4llm":
            try:
                import pymupdf4llm
                text = pymupdf4llm.to_markdown(pdf_path) or ""
            except Exception as e:
                logger.warning("pymupdf4llm failed on %s: %s; falling back to PyMuPDF", pdf_path, e)
                text = ""
        if not (text or "").strip():
            # Fallback: raw PyMuPDF text.
            doc = None
            try:
                doc = fitz.open(pdf_path)
                sections = []
                for page in doc:
                    try:
                        sections.append(page.get_text("text") or "")
                    except Exception as e:
                        logger.warning("Failed to extract text from a page in %s: %s", pdf_path, e)
                text = "\n\n".join(sections)
            except Exception as e:
                logger.error("Failed to open/read PDF %s: %s", pdf_path, e, exc_info=True)
                text = ""
            finally:
                if doc is not None:
                    try:
                        doc.close()
                    except Exception:
                        pass
        return text or ""

    def _extract_with_llm(
        self,
        text: str,
        system_prompt: str | None = None,
        model: str | None = None,
        num_retries: int = 2,
        purpose: str = "structured_extraction",
    ) -> Dict[str, Any]:
        """Call litellm and return parsed JSON dict. Raises on any failure.

        `system_prompt` / `model` / `num_retries` default to production behaviour
        (EXTRACTION_PROMPT, this extractor's model, two retries). The prompt
        ablation in benchmark.py overrides them to compare prompt styles and models
        without retries masking nondeterminism; `purpose` labels the resulting
        call_history row so token/cost accounting is attributable to an arm.
        """
        prompt = f"Extract stats from this text:\n\n{text[:10000]}"
        # Pass api_key directly rather than mutating os.environ (avoids cross-call
        # leakage); fall back to None so litellm reads the env var itself.
        api_key = self.api_key.strip() if self.api_key else None
        models_to_try = [model] if model else list(self._model_chain)
        last_exc = None
        for m in models_to_try:
            try:
                started_at = time.perf_counter()
                response = completion(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_prompt or EXTRACTION_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    api_key=api_key,
                    temperature=0,
                    num_retries=num_retries,
                )
                self._record_completion(response, started_at, purpose, model=m)
                if not response.choices or response.choices[0].message.content is None:
                    raise ValueError("LLM returned an empty response (no choices/content).")
                content = response.choices[0].message.content
                payload = strip_json_fence(content)
                if self.call_history:
                    self.call_history[-1]["fence_stripped"] = payload != content
                    self.call_history[-1]["model_used"] = m
                return json.loads(payload)
            except Exception as e:
                last_exc = e
                logger.warning("LLM extraction failed on model %s: %s; trying next.", m, e)
                continue
        raise last_exc if last_exc is not None else RuntimeError("No models available")

    def _parse_llm_result(self, data: Dict[str, Any]) -> tuple[Dict, Dict]:
        """Convert raw LLM JSON dict into (variables, correlations) dicts of Pydantic models."""
        variables: Dict[str, VariableParams] = {}
        raw_vars = data.get("variables", {}) if isinstance(data, dict) else {}
        if not isinstance(raw_vars, dict):
            raw_vars = {}
        for name, v in raw_vars.items():
            if not isinstance(v, dict):
                logger.warning("Skipping malformed variable entry %r: not an object", name)
                continue
            # LLM output may explicitly set "mean"/"std" to null for categorical
            # variables (they don't apply), not just omit the keys.
            try:
                mean = float(v.get("mean") if v.get("mean") is not None else 0.0)
                std = float(v.get("std") if v.get("std") is not None else 1.0)
            except (TypeError, ValueError):
                logger.warning("Skipping variable %r: non-numeric mean/std", name)
                continue
            categories = v.get("categories") or None
            if isinstance(categories, dict):
                try:
                    categories = {
                        str(label): float(prob) for label, prob in categories.items()
                        if prob is not None
                    } or None
                except (TypeError, ValueError):
                    categories = None
            else:
                categories = None
            try:
                variables[name] = VariableParams(
                    name=name,
                    distribution=v.get("distribution", "normal"),
                    mean=mean,
                    std=std,
                    min=float(v["min"]) if v.get("min") is not None else mean - 3 * std,
                    max=float(v["max"]) if v.get("max") is not None else mean + 3 * std,
                    categories=categories,
                )
            except Exception as e:
                logger.warning("Skipping variable %r: %s", name, e)
                continue

        # Names are resolved against the declared variables only when the literal
        # name does not already match, so a self-consistent response is passed
        # through byte-for-byte. See _resolve_correlation_name.
        by_normalized: Dict[str, list] = {}
        for declared in variables:
            by_normalized.setdefault(_normalize_variable_name(declared), []).append(declared)

        correlations: Dict[str, CorrelationParams] = {}
        raw_corrs = data.get("correlations", {}) if isinstance(data, dict) else {}
        if not isinstance(raw_corrs, dict):
            raw_corrs = {}
        for key, c in raw_corrs.items():
            if not isinstance(c, dict):
                logger.warning("Skipping malformed correlation entry %r", key)
                continue
            try:
                correlations[key] = CorrelationParams(
                    var1=_resolve_correlation_name(c.get("var1", ""), variables, by_normalized),
                    var2=_resolve_correlation_name(c.get("var2", ""), variables, by_normalized),
                    correlation=float(c.get("correlation") if c.get("correlation") is not None else 0.0),
                    direction=c.get("direction", "positive"),
                )
            except Exception as e:
                logger.warning("Skipping correlation %r: %s", key, e)
                continue

        return variables, correlations

    def llm_call(
        self,
        prompt: str,
        *,
        purpose: str = "general_llm",
        num_retries: int = 2,
    ) -> str:
        """General purpose LLM call for non-structured extraction tasks.

        Returns the raw text response from the LLM.
        """
        api_key = self.api_key.strip() if self.api_key else None
        last_exc = None
        for m in self._model_chain:
            try:
                started_at = time.perf_counter()
                response = completion(
                    model=m,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    temperature=0,
                    num_retries=num_retries,
                )
                self._record_completion(response, started_at, purpose, model=m)
                if not response.choices or response.choices[0].message.content is None:
                    raise ValueError("LLM returned an empty response.")
                return response.choices[0].message.content
            except Exception as e:
                last_exc = e
                logger.warning("llm_call failed on model %s: %s; trying next.", m, e)
                continue
        raise last_exc if last_exc is not None else RuntimeError("No models available")

    def extract_parameters(self, text: str, source_name: str) -> Parameters:
        """Extract statistical parameters — LLM first, regex fallback."""
        from datetime import datetime as _dt
        self.last_chunk_results = []
        self.last_evidence = {}
        self.last_claims = []
        if not (text or "").strip():
            return Parameters(
                variables={},
                correlations={},
                meta=MetaParams(
                    source=source_name,
                    extracted_at=_dt.now().isoformat(timespec="seconds"),
                    extraction_method="none",
                ),
            )

        if not (self.api_key or os.environ.get(self.env_var)):
            raise ValueError(
                "LLM extraction requires an API key (set api_key or env var %s). "
                "Regex fallback has been removed." % self.env_var
            )

        method = "llm"
        variables: Dict[str, VariableParams] = {}
        correlations: Dict[str, CorrelationParams] = {}

        chunks = _chunk_text(text, LLM_CHUNK_SIZE, LLM_CHUNK_OVERLAP)
        per_chunk: list[tuple[Dict, Dict]] = []
        for i, chunk in enumerate(chunks):
            try:
                data = self._extract_with_llm(
                    chunk, purpose=f"structured_extraction_chunk_{i}"
                )
                per_chunk.append(self._parse_llm_result(data))
            except Exception as e:
                logger.warning(
                    "LLM extraction failed on chunk %d/%d: %s; skipping chunk.",
                    i + 1, len(chunks), e,
                )
        if not per_chunk:
            raise RuntimeError("All LLM extraction chunks failed.")
        if len(per_chunk) == 1:
            variables, correlations = per_chunk[0]
        else:
            variables, correlations = _merge_chunk_results(per_chunk)
        self.last_chunk_results = list(per_chunk)
        variables = {
            n: v for n, v in variables.items() if v.std is not None and v.std > 0
        }
        logger.info(
            "LLM extraction succeeded: %d variables, %d correlations",
            len(variables),
            len(correlations),
        )

        params = Parameters(
            variables=variables,
            correlations=correlations,
            meta=MetaParams(
                source=source_name,
                extracted_at=datetime.now().isoformat(timespec="seconds"),
                extraction_method=method,
            ),
        )
        self.last_claims = self._build_claims(params, source_name, method)
        return params

    def _build_claims(self, params: Parameters, source_name: str, method: str) -> list:
        """Convert extracted Parameters into evidence-tagged Claims (§10.2).

        Evidence is best-effort: regex extraction captures a surrounding passage;
        LLM extraction does not yet return per-variable locators, so those claims
        carry only the source id and method. claim_type defaults to ``reported``
        because the extractor only emits directly stated statistics.
        """
        from datetime import datetime as _dt

        confidence = 0.85 if method == "llm" else 0.55
        claims = []
        for name, vp in (params.variables or {}).items():
            claims.append(Claim(
                claim_id=f"{source_name}::{name}",
                source_id=source_name,
                variable_name=name,
                statistic_type="mean" if vp.distribution != "categorical" else "category_prob",
                value={"mean": vp.mean, "std": vp.std, "distribution": vp.distribution},
                units=None,
                claim_type="reported",
                confidence=confidence,
                method=method,
                evidence=EvidenceLocator(
                    source_id=source_name,
                    passage=self.last_evidence.get(name),
                ) if self.last_evidence.get(name) else None,
                created_at=_dt.now().isoformat(timespec="seconds"),
            ))
        for key, cp in (params.correlations or {}).items():
            claims.append(Claim(
                claim_id=f"{source_name}::{key}",
                source_id=source_name,
                variable_name=key,
                statistic_type="correlation",
                value=cp.correlation,
                claim_type="reported",
                confidence=confidence,
                method=method,
                evidence=EvidenceLocator(
                    source_id=source_name,
                    passage=self.last_evidence.get(key),
                ) if self.last_evidence.get(key) else None,
                created_at=_dt.now().isoformat(timespec="seconds"),
            ))
        return claims

    def extract_with_claims(self, text: str, source_name: str):
        """Like extract_parameters, but also return the produced Claims."""
        params = self.extract_parameters(text, source_name)
        return params, list(self.last_claims)
