import tempfile
from pathlib import Path

from agentdataset.core.review import ClaimReview
from agentdataset.models.schemas import Claim


def _claims():
    return [Claim(claim_id="c1", source_id="s", variable_name="age", value=1.0, status="proposed"),
            Claim(claim_id="c2", source_id="s", variable_name="bmi", value=2.0, status="proposed")]


def test_review_accept_reject_defer_edit():
    r = ClaimReview(_claims())
    r.accept("c1")
    r.reject("c2", "bad")
    assert len(r.accepted_claims()) == 1
    assert len(r.rejected_claims()) == 1
    r2 = ClaimReview(_claims())
    r2.defer("c1")
    assert len(r2.pending_claims()) == 2  # deferred counts as pending
    r3 = ClaimReview(_claims())
    r3.edit("c1", 9.0)
    assert r3.accepted_claims()[0].edited_value == 9.0


def test_review_save_load_roundtrip():
    r = ClaimReview(_claims())
    r.accept("c1")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "claims.json"
        r.save(p)
        r2 = ClaimReview.load(p)
        assert len(r2.claims) == 2
        assert r2.accepted_claims()[0].claim_id == "c1"
        assert len(r2.audit_log) == 1