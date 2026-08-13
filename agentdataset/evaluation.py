"""Shared provenance and uncertainty helpers for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RAW_RESULT_COLUMNS = [
    "experiment_id", "run_id", "experiment_type", "source_id", "dataset",
    "task_type", "condition", "model_family", "synthesizer", "seed",
    "split_seed", "synthesis_seed", "metric", "value", "status",
    "failure_stage", "extraction_method", "provider", "model",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(command: str, seeds: Sequence[int] = ()) -> dict:
    """Collect safe, non-secret metadata for an experiment run."""
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    lock = Path("uv.lock")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_lock": str(lock),
        "dependency_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "command": command,
        "seeds": list(seeds),
    }


def validate_raw_results(frame: pd.DataFrame) -> None:
    missing = [column for column in RAW_RESULT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Raw result schema missing columns: {missing}")
    if frame["run_id"].isna().any() or frame["metric"].isna().any():
        raise ValueError("run_id and metric are required")
    valid_statuses = {"ok", "failed", "skipped"}
    invalid = set(frame["status"].dropna()) - valid_statuses
    if invalid:
        raise ValueError(f"Invalid result statuses: {sorted(invalid)}")


def bootstrap_ci(
    values: Iterable[float], n_resamples: int = 10_000, seed: int = 20260802
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_resamples, len(array)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_long_results(
    raw: pd.DataFrame,
    group_columns: Sequence[str],
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    validate_raw_results(raw)
    successful = raw.loc[raw["status"].eq("ok")].copy()
    successful["value"] = pd.to_numeric(successful["value"], errors="coerce")
    rows = []
    for keys, group in successful.groupby([*group_columns, "metric"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["value"].dropna().to_numpy(dtype=float)
        low, high = bootstrap_ci(values, n_resamples=n_resamples)
        row = dict(zip([*group_columns, "metric"], keys))
        row.update({
            "n": len(values),
            "mean": float(np.mean(values)) if len(values) else float("nan"),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

