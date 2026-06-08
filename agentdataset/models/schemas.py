# AgentDataset Models

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime

_ALLOWED_DISTRIBUTIONS = {"normal", "uniform", "gamma"}
_ALLOWED_SOURCE_TYPES = {"pdf", "html"}

class VariableParams(BaseModel):
    name: str
    distribution: str = "normal"
    mean: float = 0.0
    std: float = 1.0
    min: Optional[float] = None
    max: Optional[float] = None

    @field_validator("distribution", mode="before")
    @classmethod
    def _coerce_distribution(cls, v):
        # Unknown distributions (e.g. an LLM hallucination) degrade to normal
        # rather than crashing extraction; the synthesizer treats them the same.
        return v if v in _ALLOWED_DISTRIBUTIONS else "normal"

class CorrelationParams(BaseModel):
    var1: str
    var2: str
    correlation: float
    direction: str = "positive"

    @field_validator("correlation", mode="before")
    @classmethod
    def _clamp_correlation(cls, v):
        # Clamp to a valid Pearson range instead of rejecting, so a bad LLM value
        # doesn't abort the whole extraction.
        try:
            return max(-1.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

class MetaParams(BaseModel):
    source: str
    extracted_at: str
    extraction_method: str = "regex_fallback"

class Parameters(BaseModel):
    variables: Dict[str, VariableParams]
    correlations: Dict[str, CorrelationParams]
    meta: MetaParams

class FidelityReport(BaseModel):
    overall_score: float
    ks_score: float
    corr_score: float
    bias_score: float
    ks_pvalues: Dict[str, float]
    bias_details: Dict[str, Any]
    privacy_details: Dict[str, float]
    approved: bool

class SessionContext(BaseModel):
    session_id: str
    path: str
    created_at: datetime = Field(default_factory=datetime.now)
    run_tag: Optional[str] = None

class DiscoveryResult(BaseModel):
    title: str
    url: str
    source_type: str  # "pdf" or "html"
    relevance_score: float
    snippet: Optional[str] = None

    @field_validator("source_type", mode="before")
    @classmethod
    def _coerce_source_type(cls, v):
        return v if v in _ALLOWED_SOURCE_TYPES else "html"
