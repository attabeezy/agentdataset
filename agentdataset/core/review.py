"""Human review of extracted claims before synthesis (§10.4).

Manages accept/reject/edit/defer decisions over a list of Claims and can build
the reviewed synthesis specification from the accepted subset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from agentdataset.models.schemas import Claim


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