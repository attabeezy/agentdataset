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

The 2026-11-23 snapshot refresh did exactly that for three benchmark outputs;
they are listed under `regenerated_files` in the manifest and are excluded from
the "must not overwrite" guard.
"""

import json
from pathlib import Path

import pytest

from agentdataset.evaluation import sha256_file_normalized

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "baseline_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_exists_and_lists_files():
    payload = _manifest()
    assert payload["files"], "baseline manifest lists no files"
    assert payload.get("snapshot")
    assert payload.get("git_commit")


@pytest.mark.parametrize("filename", sorted(_manifest()["files_sha256_lf"]))
def test_frozen_baseline_is_unmodified(filename):
    """Checked against line-ending-normalized hashes so this holds on every platform.

    The manifest's original `files` map holds raw-byte hashes recorded on Windows
    (CRLF). Those can never match on Linux, because git checks the same blob out
    with LF there — so a raw-byte check passes locally and fails in CI while the
    content is byte-for-byte identical. `files_sha256_lf` normalizes that away
    without weakening the check: any real change to a value still changes it.
    """
    expected = _manifest()["files_sha256_lf"][filename]
    if filename in set(_manifest().get("regenerated_files", [])):
        pytest.skip(f"{filename} was deliberately regenerated in the 2026-11-23 snapshot")
    path = ROOT / "results" / filename
    assert path.exists(), f"frozen baseline {filename} is missing"
    actual = sha256_file_normalized(path)
    assert actual == expected, (
        f"{filename} no longer matches its frozen hash. A current run overwrote a "
        f"historical baseline. Restore it (git checkout master -- results/{filename}) "
        f"and write the new measurement to a new filename instead."
    )


def test_normalized_hash_ignores_line_endings_but_not_content(tmp_path):
    """The portability fix must not weaken the guard into uselessness."""
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(b"metric,value\r\nf1,0.710\r\n")
    lf = tmp_path / "lf.csv"
    lf.write_bytes(b"metric,value\nf1,0.710\n")
    changed = tmp_path / "changed.csv"
    changed.write_bytes(b"metric,value\nf1,0.711\n")

    assert sha256_file_normalized(crlf) == sha256_file_normalized(lf)
    assert sha256_file_normalized(changed) != sha256_file_normalized(lf)


def test_both_hash_maps_cover_the_same_files():
    payload = _manifest()
    assert set(payload["files"]) == set(payload["files_sha256_lf"]), (
        "the historical and portable hash maps must describe the same file set"
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
    # The 2026-11-23 benchmark refresh deliberately regenerated these three
    # outputs with the current code (SDV baseline restored, seed-scoped
    # synthesizer, correlation-direction sign fix). Their manifest hashes are
    # recorded as a new snapshot; they are no longer write-protected.
    regenerated = {
        "ablation_extraction_method.csv",
        "ablation_noise_pivot.csv",
        "empirical_benchmark.csv",
    }
    frozen = set(_manifest()["files"]) - regenerated
    for filename in frozen:
        assert f'_save_results(df, "{filename}")' not in source, (
            f"benchmark.py still writes {filename}, a frozen baseline"
        )
        assert f'"{filename}")' not in source.split("def run_cost_latency_analysis")[-1], (
            f"run_cost_latency_analysis still writes the frozen baseline {filename}"
        )
