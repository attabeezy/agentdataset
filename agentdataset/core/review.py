"""Human review of extracted claims before synthesis (§10.4).

Manages accept/reject/edit/defer decisions over a list of Claims and can build
the reviewed synthesis specification from the accepted subset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentdataset.models.schemas import Claim, Parameters, VariableParams, CorrelationParams, MetaParams


class ClaimReview:
    def __init__(self, claims: Optional[List[Claim]] = None):
        self.claims: List[Claim] = list(claims or [])
        self.audit_log: List[Dict[str, Any]] = []

    def _find(self, claim_id: str) -> Claim:
        for c in self.claims:
            if c.claim_id == claim_id:
                return c
        raise KeyError(f"Unknown claim_id: {claim_id}")

    def accept(self, claim_id: str, note: str | None = None) -> Claim:
        c = self._find(claim_id)
        c.status = "accepted"
        if note:
            c.review_note = note
        self.audit_log.append({"action": "accept", "claim_id": claim_id, "note": note})
        return c

    def reject(self, claim_id: str, note: str | None = None) -> Claim:
        c = self._find(claim_id)
        c.status = "rejected"
        if note:
            c.review_note = note
        self.audit_log.append({"action": "reject", "claim_id": claim_id, "note": note})
        return c

    def defer(self, claim_id: str, note: str | None = None) -> Claim:
        c = self._find(claim_id)
        c.status = "deferred"
        if note:
            c.review_note = note
        self.audit_log.append({"action": "defer", "claim_id": claim_id, "note": note})
        return c

    def edit(self, claim_id: str, value: Any, note: str | None = None) -> Claim:
        c = self._find(claim_id)
        c.status = "edited"
        c.edited_value = value
        if note:
            c.review_note = note
        self.audit_log.append({"action": "edit", "claim_id": claim_id, "value": value, "note": note})
        return c

    def accepted_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.status in ("accepted", "edited")]

    def rejected_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.status == "rejected"]

    def pending_claims(self) -> List[Claim]:
        return [c for c in self.claims if c.status in ("proposed", "deferred")]

    def value_of(self, claim: Claim) -> Any:
        return claim.edited_value if claim.status == "edited" and claim.edited_value is not None else claim.value

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "claims": [c.model_dump() for c in self.claims],
            "audit_log": self.audit_log,
        }
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ClaimReview":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        claims = [Claim.model_validate(c) for c in data.get("claims", [])]
        r = cls(claims)
        r.audit_log = data.get("audit_log", [])
        return r


def parameters_from_claims(claims: List[Claim], source: str = "agent_review") -> Parameters:
    """Build synthesis parameters from the claims the agent accepted.

    Only reported/edited values are used. Continuous statistics are grouped by
    variable name; correlation claims use ``var1__var2`` (or a comma/pipe) as the
    variable-name separator. Claims the agent rejected or deferred are ignored.
    """
    from collections import defaultdict
    import re
    from datetime import datetime

    stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
    correlations: Dict[str, CorrelationParams] = {}

    for claim in claims:
        if claim.status not in {"accepted", "edited"}:
            continue
        value = claim.edited_value if claim.status == "edited" and claim.edited_value is not None else claim.value
        if value is None:
            continue
        if claim.statistic_type in {"mean", "std", "min", "max"}:
            stats[claim.variable_name][claim.statistic_type] = value
        elif claim.statistic_type == "correlation":
            parts = [p.strip() for p in re.split(r"__|\||,", claim.variable_name) if p.strip()]
            if len(parts) < 2:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            key = f"{parts[0]}__{parts[1]}"
            correlations[key] = CorrelationParams(
                var1=parts[0],
                var2=parts[1],
                correlation=value,
                direction="positive" if value >= 0 else "negative",
            )

    variables: Dict[str, VariableParams] = {}
    for name, values in stats.items():
        if "mean" not in values:
            continue
        try:
            mean = float(values["mean"])
            std = float(values.get("std", 1.0))
        except (TypeError, ValueError):
            continue
        if std <= 0:
            std = 1.0
        variables[name] = VariableParams(
            name=name,
            distribution="normal",
            mean=mean,
            std=std,
            min=float(values["min"]) if "min" in values else mean - 3 * std,
            max=float(values["max"]) if "max" in values else mean + 3 * std,
        )

    return Parameters(
        variables=variables,
        correlations=correlations,
        meta=MetaParams(
            source=source,
            extracted_at=datetime.now().isoformat(timespec="seconds"),
            extraction_method="llm",
        ),
    )