# AgentDataset Models

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class VariableParams(BaseModel):
    name: str
    distribution: str = "normal"
    mean: float = 0.0
    std: float = 1.0
    min: Optional[float] = None
    max: Optional[float] = None

class CorrelationParams(BaseModel):
    var1: str
    var2: str
    correlation: float
    direction: str = "positive"

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
