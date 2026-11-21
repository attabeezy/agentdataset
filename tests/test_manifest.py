import json
import tempfile
from pathlib import Path

from agentdataset.core.manifest import build_manifest, write_manifest, read_manifest
from agentdataset.models.schemas import SourceRecord, Claim, RunManifest


def test_build_manifest_defaults():
    m = build_manifest("run_1", query="test query")
    assert m.run_id == "run_1"
    assert m.query == "test query"
    assert m.manifest_version == "1.0"
    assert m.created_at
    assert m.code_revision is None or isinstance(m.code_revision, str)


def test_write_read_roundtrip():
    m = build_manifest("run_2", query="q")
    m.sources.append(SourceRecord(source_id="s1", url="http://x", title="t", source_type="html"))
    m.raw_claims.append(Claim(claim_id="c1", source_id="s1", variable_name="age", value=1.0))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        write_manifest(m, p)
        assert p.exists()
        m2 = read_manifest(p)
        assert m2.run_id == "run_2"
        assert len(m2.sources) == 1
        assert m2.raw_claims[0].variable_name == "age"


def test_claim_validation_coercion():
    c = Claim(claim_id="c", source_id="s", variable_name="v", claim_type="bogus", status="bogus",
              statistic_type="bogus", confidence=2.0)
    assert c.claim_type == "reported"
    assert c.status == "proposed"
    assert c.statistic_type == "other"
    assert c.confidence == 1.0