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
import fitz  # PyMuPDF
import litellm
from litellm import completion
from agentdataset.models.schemas import (
    Parameters,
    VariableParams,
    CorrelationParams,
    MetaParams,
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
are mentioned, along with their probability distributions (such as normal,
uniform, gamma, or categorical), their means, and their standard deviations.
Also, please identify any correlations that are described between pairs of
variables, including the strength and direction (positive or negative) of each
correlation.

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

# The production system prompt. Switched from CAVEMAN_PROMPT to
# NATURAL_LANGUAGE_PROMPT on the strength of the §9.6 ablation (see above).
EXTRACTION_PROMPT = NATURAL_LANGUAGE_PROMPT

# Separator between a keyword and its numeric value: "= 3.5", ": 3.5", "is 3.5", "of 3.5"
_SEP = r"\s*(?:is|=|:|of)?\s*"

# A number: optional sign, integer/decimal, optional scientific notation (e.g. -5, 3.5, 1.2e-3)
_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"

# Pattern A: mean then std (e.g. "mean = 3.5 ... std = 1.2", "mean is 3.5 ... standard deviation is 1.2")
_PATTERN_MEAN_STD = re.compile(
    r"(?:mean|average|μ)"
    + _SEP
    + _NUM
    + r".{0,80}?"
    + r"(?:std|s\.?d\.?|standard\s+deviation|σ)"
    + _SEP
    + _NUM,
    re.IGNORECASE,
)

# Pattern B: std then mean (e.g. "SD = 1.2, mean = 3.5")
_PATTERN_STD_MEAN = re.compile(
    r"(?:std|s\.?d\.?|standard\s+deviation|σ)"
    + _SEP
    + _NUM
    + r".{0,80}?"
    + r"(?:mean|average|μ)"
    + _SEP
    + _NUM,
    re.IGNORECASE,
)

# Correlation between two named variables, e.g. "correlation between X and Y is 0.65",
# "corr(X, Y) = -0.4", "r = 0.8".  Captures (var1, var2, value) where names are optional.
_PATTERN_CORR_NAMED = re.compile(
    r"correlation\s+between\s+(\w+)\s+and\s+(\w+)"
    + _SEP
    + r"(-?(?:0?\.\d+|1(?:\.0+)?))",
    re.IGNORECASE,
)
_PATTERN_CORR_FUNC = re.compile(
    r"corr(?:elation)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)"
    + _SEP
    + r"(-?(?:0?\.\d+|1(?:\.0+)?))",
    re.IGNORECASE,
)

# Categorical variable with any number of categories, e.g. "The categorical
# variable sex takes value 'Male' with probability 0.67 and 'Female' with
# probability 0.33." or "... takes value 'a' with probability 0.2, 'b' with
# probability 0.3, and 'c' with probability 0.5." Captures (name, blob) where
# blob is the whole run of label/probability pairs; _PATTERN_CATEGORY_PAIR
# then pulls the individual (label, prob) pairs out of the blob.
_PROB = r"(?:0?\.\d+|1(?:\.0+)?|0)"
_PATTERN_CATEGORICAL = re.compile(
    r"categorical\s+variable\s+(\w+)\s+takes\s+value\s+"
    r"((?:'[^']+'\s+with\s+probability\s+" + _PROB + r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)?)+)",
    re.IGNORECASE,
)
_PATTERN_CATEGORY_PAIR = re.compile(
    r"'([^']+)'\s+with\s+probability\s+(" + _PROB + r")",
    re.IGNORECASE,
)

# Function words that are never part of a variable name. Without this guard the
# name capture below swallows the article in "a heart rate distribution".
#
# Deliberately limited to grammatical filler. Content nouns are *not* listed:
# excluding a word like "study" does not yield no-name, it yields the wrong name
# ("study hours" captures as "hours"), and a wrong name mismatches silently,
# which is the failure this whole change exists to remove.
_STOPWORDS = (
    r"(?:an?|the|we|this|that|these|those|there|its?|and|or|but|of|for|with|"
    r"in|on|at|by|from|was|were|is|are|be|been|has|have|had|which|who|whose|"
    r"when|where)"
)

# One to three words, none of them a stopword. Hyphens and underscores are kept
# because real column names use them (education_num, follow-up).
_NAME = (
    r"(?!" + _STOPWORDS + r"\b)[A-Za-z][A-Za-z0-9_-]*"
    r"(?:\s+(?!" + _STOPWORDS + r"\b)[A-Za-z][A-Za-z0-9_-]*){0,2}"
)

# Recover the variable name that a mean/std match belongs to.
#
# The mean/std patterns capture only the two numbers, so on their own they
# cannot say which variable the numbers describe -- which is why unnamed
# variables fall back to "var_N". That fallback is not harmless: a correlation
# naming a variable that no extracted variable matches is dropped silently by
# the synthesizer, so an entire correlation structure can vanish while every
# validity check still passes (see analysis/corr_validity_audit.py).
#
# This is matched against the text *preceding* a mean/std hit and anchored to
# the end of it, so only the immediately adjacent phrasing can supply a name.
# Deliberately conservative: a wrong name is worse than no name, because it
# mismatches silently instead of falling back to var_N.
_PATTERN_NAME_BEFORE_STAT = re.compile(
    r"(?:"
    # "The variable age has ", "for variable bmi with "
    r"variables?\s+(?P<v_var>" + _NAME + r")\s+(?:has|had|is|was|with|,)\s*"
    # "a heart rate distribution with ", "Blood pressure was normally distributed with "
    r"|(?P<v_dist>" + _NAME + r")\s+(?:"
    r"distribution\s+with|(?:was|is|were|are)\s+normally\s+distributed\s+with"
    r")\s*"
    r")$",
    re.IGNORECASE,
)


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


class Extractor:
    def __init__(
        self, model: str = "gpt-4o", api_key: str = "", env_var: str = "OPENAI_API_KEY"
    ):
        self.model = model
        self.api_key = api_key
        self.env_var = env_var
        self.call_history: list[dict] = []

    def _record_completion(
        self, response, started_at: float, purpose: str, model: str | None = None
    ) -> None:
        """Capture non-secret per-call usage for end-to-end accounting."""
        usage = getattr(response, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=response))
        except Exception:
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

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to clean text."""
        doc = fitz.open(pdf_path)
        sections = []
        for page in doc:
            text = page.get_text("text")
            sections.append(text)
        doc.close()
        return "\n\n".join(sections)

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
        started_at = time.perf_counter()
        response = completion(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system_prompt or EXTRACTION_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            api_key=api_key,
            temperature=0,
            num_retries=num_retries,
        )
        self._record_completion(response, started_at, purpose, model=model)
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("LLM returned an empty response (no choices/content).")
        content = response.choices[0].message.content
        payload = strip_json_fence(content)
        # Recorded so the ablation can report how often the production parser
        # would have failed without the fence strip, per provider.
        if self.call_history:
            self.call_history[-1]["fence_stripped"] = payload != content
        return json.loads(payload)

    def _parse_llm_result(self, data: Dict[str, Any]) -> tuple[Dict, Dict]:
        """Convert raw LLM JSON dict into (variables, correlations) dicts of Pydantic models."""
        variables: Dict[str, VariableParams] = {}
        for name, v in data.get("variables", {}).items():
            # LLM output may explicitly set "mean"/"std" to null for categorical
            # variables (they don't apply), not just omit the keys.
            mean = float(v.get("mean") if v.get("mean") is not None else 0.0)
            std = float(v.get("std") if v.get("std") is not None else 1.0)
            categories = v.get("categories") or None
            if categories is not None:
                categories = {
                    label: float(prob) for label, prob in categories.items()
                    if prob is not None
                } or None
            variables[name] = VariableParams(
                name=name,
                distribution=v.get("distribution", "normal"),
                mean=mean,
                std=std,
                min=float(v["min"]) if v.get("min") is not None else mean - 3 * std,
                max=float(v["max"]) if v.get("max") is not None else mean + 3 * std,
                categories=categories,
            )

        # Names are resolved against the declared variables only when the literal
        # name does not already match, so a self-consistent response is passed
        # through byte-for-byte. See _resolve_correlation_name.
        by_normalized: Dict[str, list] = {}
        for declared in variables:
            by_normalized.setdefault(_normalize_variable_name(declared), []).append(declared)

        correlations: Dict[str, CorrelationParams] = {}
        for key, c in data.get("correlations", {}).items():
            correlations[key] = CorrelationParams(
                var1=_resolve_correlation_name(c["var1"], variables, by_normalized),
                var2=_resolve_correlation_name(c["var2"], variables, by_normalized),
                correlation=float(c.get("correlation") if c.get("correlation") is not None else 0.0),
                direction=c.get("direction", "positive"),
            )

        return variables, correlations

    def _extract_with_regex(self, text: str) -> tuple[Dict, Dict]:
        """Fallback regex extraction returning (variables, correlations).

        Continuous variables are named from the surrounding prose when it says
        clearly enough which variable a mean/std pair belongs to, and fall back
        to a positional "var_N" when it does not.
        """
        variables: Dict[str, VariableParams] = {}
        seen = set()

        def add_variable(mean: float, std: float, start: int) -> None:
            """Record one continuous variable, named from the text where possible."""
            key = (mean, std)
            if key in seen:
                return
            seen.add(key)
            name = self._name_for_stat(text, start, variables)
            variables[name] = VariableParams(
                name=name,
                distribution="normal",
                mean=mean,
                std=std,
                min=mean - 3 * std,
                max=mean + 3 * std,
            )

        for match in _PATTERN_MEAN_STD.finditer(text):
            add_variable(float(match.group(1)), float(match.group(2)), match.start())

        for match in _PATTERN_STD_MEAN.finditer(text):
            add_variable(float(match.group(2)), float(match.group(1)), match.start())

        # Categorical variables (any category count) — always captured with their
        # real name. Category *labels* are left exactly as written; only the
        # variable name is normalized, so it can match a correlation clause.
        for match in _PATTERN_CATEGORICAL.finditer(text):
            name, blob = match.groups()
            name = _normalize_variable_name(name) or name
            pairs = _PATTERN_CATEGORY_PAIR.findall(blob)
            # A single pair would normalize to probability 1.0 — almost certainly
            # a parse fragment rather than a real categorical description.
            if name not in variables and len(pairs) >= 2:
                variables[name] = VariableParams(
                    name=name,
                    distribution="categorical",
                    categories={label: float(prob) for label, prob in pairs},
                )

        # Correlation extraction. Names go through the same normalization as the
        # variables above so that a pair can actually resolve; a pair naming a
        # variable that was never extracted is still emitted, and the synthesizer
        # records it as skipped rather than silently pretending it applied.
        correlations: Dict[str, CorrelationParams] = {}
        for pattern in (_PATTERN_CORR_NAMED, _PATTERN_CORR_FUNC):
            for match in pattern.finditer(text):
                v1 = _normalize_variable_name(match.group(1)) or match.group(1)
                v2 = _normalize_variable_name(match.group(2)) or match.group(2)
                value = float(match.group(3))
                key = f"{v1}__{v2}"
                if key not in correlations:
                    correlations[key] = CorrelationParams(
                        var1=v1,
                        var2=v2,
                        correlation=value,
                        direction="positive" if value >= 0 else "negative",
                    )

        return variables, correlations

    @staticmethod
    def _name_for_stat(text: str, start: int, existing: Dict[str, VariableParams]) -> str:
        """Name the variable a mean/std hit at `start` describes.

        Looks only at the 80 characters immediately before the hit, so a name
        has to be adjacent to the statistic to be claimed. Returns "var_N" when
        no confident name is available, or when the name is already taken --
        reusing a name would silently merge two distinct variables.
        """
        window = text[max(0, start - 80):start]
        match = _PATTERN_NAME_BEFORE_STAT.search(window)
        if match:
            raw = match.group("v_var") or match.group("v_dist")
            name = _normalize_variable_name(raw or "")
            if name and name not in existing:
                return name
        return f"var_{len(existing) + 1}"

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
        started_at = time.perf_counter()
        response = completion(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            api_key=api_key,
            temperature=0,
            num_retries=num_retries,
        )
        self._record_completion(response, started_at, purpose)
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("LLM returned an empty response.")
        return response.choices[0].message.content

    def extract_parameters(self, text: str, source_name: str) -> Parameters:
        """Extract statistical parameters — LLM first, regex fallback."""
        method = "regex_fallback"
        variables: Dict[str, VariableParams] = {}
        correlations: Dict[str, CorrelationParams] = {}

        if self.api_key or os.environ.get(self.env_var):
            try:
                data = self._extract_with_llm(text)
                variables, correlations = self._parse_llm_result(data)
                method = "llm"
                logger.info(
                    "LLM extraction succeeded: %d variables, %d correlations",
                    len(variables),
                    len(correlations),
                )
            except Exception as e:
                # Name the cause (RateLimitError / AuthenticationError / JSONDecodeError …)
                # so failures are diagnosable rather than a generic message.
                logger.warning(
                    "LLM extraction failed [%s]: %s; falling back to regex.",
                    type(e).__name__,
                    e,
                )
                variables, correlations = self._extract_with_regex(text)
        else:
            variables, correlations = self._extract_with_regex(text)

        return Parameters(
            variables=variables,
            correlations=correlations,
            meta=MetaParams(
                source=source_name,
                extracted_at=datetime.now().isoformat(timespec="seconds"),
                extraction_method=method,
            ),
        )
