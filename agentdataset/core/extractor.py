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
import fitz  # PyMuPDF
from litellm import completion
from agentdataset.models.schemas import (
    Parameters,
    VariableParams,
    CorrelationParams,
    MetaParams,
)

logger = logging.getLogger(__name__)

CAVEMAN_PROMPT = """
You are an expert statistician.
Strip prose.
Extract variables, distributions, correlations.
Output strict JSON matching this schema exactly:
{
  "variables": {
    "<name>": {"distribution": "normal|uniform|gamma", "mean": 0.0, "std": 1.0, "min": null, "max": null}
  },
  "correlations": {
    "<key>": {"var1": "<name>", "var2": "<name>", "correlation": 0.5, "direction": "positive|negative"}
  }
}
No fluff. No greeting. Output JSON only.
"""

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


class Extractor:
    def __init__(
        self, model: str = "gpt-4o", api_key: str = "", env_var: str = "OPENAI_API_KEY"
    ):
        self.model = model
        self.api_key = api_key
        self.env_var = env_var

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to clean text."""
        doc = fitz.open(pdf_path)
        sections = []
        for page in doc:
            text = page.get_text("text")
            sections.append(text)
        doc.close()
        return "\n\n".join(sections)

    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Call litellm and return parsed JSON dict. Raises on any failure."""
        prompt = f"Extract stats from this text:\n\n{text[:10000]}"
        # Pass api_key directly rather than mutating os.environ (avoids cross-call
        # leakage); fall back to None so litellm reads the env var itself.
        api_key = self.api_key.strip() if self.api_key else None
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": CAVEMAN_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            api_key=api_key,
            num_retries=2,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise ValueError("LLM returned an empty response (no choices/content).")
        return json.loads(response.choices[0].message.content)

    def _parse_llm_result(self, data: Dict[str, Any]) -> tuple[Dict, Dict]:
        """Convert raw LLM JSON dict into (variables, correlations) dicts of Pydantic models."""
        variables: Dict[str, VariableParams] = {}
        for name, v in data.get("variables", {}).items():
            mean = float(v.get("mean", 0.0))
            std = float(v.get("std", 1.0))
            variables[name] = VariableParams(
                name=name,
                distribution=v.get("distribution", "normal"),
                mean=mean,
                std=std,
                min=float(v["min"]) if v.get("min") is not None else mean - 3 * std,
                max=float(v["max"]) if v.get("max") is not None else mean + 3 * std,
            )

        correlations: Dict[str, CorrelationParams] = {}
        for key, c in data.get("correlations", {}).items():
            correlations[key] = CorrelationParams(
                var1=c["var1"],
                var2=c["var2"],
                correlation=float(c.get("correlation", 0.0)),
                direction=c.get("direction", "positive"),
            )

        return variables, correlations

    def _extract_with_regex(self, text: str) -> tuple[Dict, Dict]:
        """Fallback regex extraction returning (variables, correlations)."""
        variables: Dict[str, VariableParams] = {}
        seen = set()

        for match in _PATTERN_MEAN_STD.finditer(text):
            mean, std = float(match.group(1)), float(match.group(2))
            key = (mean, std)
            if key not in seen:
                seen.add(key)
                name = f"var_{len(variables) + 1}"
                variables[name] = VariableParams(
                    name=name,
                    distribution="normal",
                    mean=mean,
                    std=std,
                    min=mean - 3 * std,
                    max=mean + 3 * std,
                )

        for match in _PATTERN_STD_MEAN.finditer(text):
            std, mean = float(match.group(1)), float(match.group(2))
            key = (mean, std)
            if key not in seen:
                seen.add(key)
                name = f"var_{len(variables) + 1}"
                variables[name] = VariableParams(
                    name=name,
                    distribution="normal",
                    mean=mean,
                    std=std,
                    min=mean - 3 * std,
                    max=mean + 3 * std,
                )

        # Best-effort correlation extraction (synthesizer ignores pairs whose
        # variable names are not present, so mismatched names are harmless).
        correlations: Dict[str, CorrelationParams] = {}
        for pattern in (_PATTERN_CORR_NAMED, _PATTERN_CORR_FUNC):
            for match in pattern.finditer(text):
                v1, v2, value = match.group(1), match.group(2), float(match.group(3))
                key = f"{v1}__{v2}"
                if key not in correlations:
                    correlations[key] = CorrelationParams(
                        var1=v1,
                        var2=v2,
                        correlation=value,
                        direction="positive" if value >= 0 else "negative",
                    )

        return variables, correlations

    def llm_call(self, prompt: str) -> str:
        """General purpose LLM call for non-structured extraction tasks.

        Returns the raw text response from the LLM.
        """
        api_key = self.api_key.strip() if self.api_key else None
        response = completion(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            api_key=api_key,
            num_retries=2,
        )
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
