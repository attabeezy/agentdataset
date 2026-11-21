"""Run-manifest helpers for evidence-first provenance (§10.9).

A manifest links a run's query, retrieved sources, extracted claims, review
decisions, reconciliation, synthesis configuration, validation, and outputs so
that the final CSV can be traced back to each accepted claim.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentdataset.models.schemas import RunManifest, SourceRecord, Claim


def _git_revision() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def build_manifest(
    run_id: str,
    query: Optional[str] = None,
    sources: Optional[List[SourceRecord]] = None,
    raw_claims: Optional[List[Claim]] = None,
    **kwargs: Any,
) -> RunManifest:
    """Construct a RunManifest with sensible defaults filled in."""
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        code_revision=_git_revision(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        query=query,
        sources=sources or [],
        raw_claims=raw_claims or [],
        **kwargs,
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    """Write a manifest as JSON. Returns the written path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    return p


def read_manifest(path: str | Path) -> RunManifest:
    with open(path, "r", encoding="utf-8") as f:
        return RunManifest.model_validate_json(f.read())


def manifest_filename(run_id: str) -> str:
    return f"run_manifest_{run_id}.json"