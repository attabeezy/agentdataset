"""Guard the frozen pre-revision baselines in results/baseline_manifest.json.

The manifest states: "Hashes freeze the exact historical result bytes without
duplicating them. New experiments write seed-level raw files and must not
overwrite these files."

Nothing enforced that until an experiment in this revision regenerated
cost_latency.csv and caveman_token_savings.csv in place, silently destroying two
of the six historical baselines. This test makes that failure loud.

If a baseline legitimately needs to change, the manifest hash and its
`recorded_at` / `git_commit` provenance must be updated in the same commit, and
the reason stated -- that is a deliberate act, not a side effect of a rerun.
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "baseline_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_exists_and_lists_files():
    payload = _manifest()
    assert payload["files"], "baseline manifest lists no files"
    assert payload.get("snapshot")
    assert payload.get("git_commit")


@pytest.mark.parametrize("filename", sorted(_manifest()["files"]))
def test_frozen_baseline_is_unmodified(filename):
    expected = _manifest()["files"][filename]
    path = ROOT / "results" / filename
    assert path.exists(), f"frozen baseline {filename} is missing"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"{filename} no longer matches its frozen hash. A current run overwrote a "
        f"historical baseline. Restore it (git checkout master -- results/{filename}) "
        f"and write the new measurement to a new filename instead."
    )


@pytest.mark.parametrize("script", [
    "analysis/extraction_ablation.py",
    "analysis/real_paper_eval.py",
    "analysis/p0_utility_benchmark.py",
    "analysis/adult_calibration.py",
])
def test_provenance_is_captured_before_outputs_are_written(script):
    """collect_provenance() must run before the first write, not after.

    Called afterwards, git_dirty is structurally always True: the run's own CSVs
    are sitting in the working tree by the time provenance is recorded, so the
    manifest can never report a clean tree however clean the checkout was.
    """
    source = (ROOT / script).read_text(encoding="utf-8")
    call = source.index("collect_provenance(")
    writes = [source.index(m) for m in ("to_csv(", "write_json(") if m in source]
    assert writes, f"{script} writes no outputs; test assumption is wrong"
    assert call < min(writes), (
        f"{script} calls collect_provenance() after its first output write, "
        f"which forces git_dirty=True in the manifest"
    )


def test_current_runs_do_not_target_frozen_filenames():
    """The producers must write alongside the baselines, never over them."""
    import benchmark

    source = Path(benchmark.__file__).read_text(encoding="utf-8")
    frozen = set(_manifest()["files"])
    for filename in frozen:
        assert f'_save_results(df, "{filename}")' not in source, (
            f"benchmark.py still writes {filename}, a frozen baseline"
        )
        assert f'"{filename}")' not in source.split("def run_cost_latency_analysis")[-1], (
            f"run_cost_latency_analysis still writes the frozen baseline {filename}"
        )
