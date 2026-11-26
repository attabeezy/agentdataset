"""Submission guards for the AAAI 2027 AI4Research paper and evidence contracts."""

from __future__ import annotations

import re
from pathlib import Path

from analysis import paper_artifacts


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "submission" / "paper.tex"


def _paper_text() -> str:
    return PAPER.read_text(encoding="utf-8")


def test_generated_tables_match_committed_results():
    paper_artifacts.validate_manifests()
    for name, expected in paper_artifacts.render_tables().items():
        path = paper_artifacts.GENERATED / name
        assert path.exists(), name
        assert path.read_text(encoding="utf-8") == expected, name


def test_paper_consumes_every_generated_result_table():
    text = _paper_text()
    for name, expected in paper_artifacts.render_tables().items():
        assert expected.strip() in text, f"paper is missing generated table {name}"


def test_paper_has_no_stale_placeholders_or_external_inputs():
    text = _paper_text()
    for name in paper_artifacts.render_tables():
        assert "\n" + name.removesuffix(".tex") + "\n" not in text, name
    assert "\\input{" not in text


def test_abstract_is_compact_and_claim_scoped():
    text = _paper_text()
    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.DOTALL
    ).group(1)
    plain = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", abstract)
    plain = re.sub(r"[{}$~\\]", " ", plain)
    words = re.findall(r"\b[\w.-]+\b", plain)
    assert len(words) <= 200
    assert ".708" in abstract
    assert "privacy mechanism" in abstract
    assert "unattended" in abstract


def test_q1_uses_final_deployed_prompt_replication():
    text = _paper_text()
    q1 = text.split(r"\subsection{Q1: Real papers}", 1)[1].split(
        r"\subsection{Q2: Downstream utility}", 1
    )[0]
    for token in [".575", ".920", ".708", ".304", r"\$0.122", "44.4"]:
        assert token in q1, token
    assert r"\$0.0343" not in q1


def test_figure_walkthrough_is_source_backed_and_candid():
    text = _paper_text()
    for token in ["rank 1", "N{=}447", "35.7", "r{=}.64", "omitted", "96.81"]:
        assert token in text, token


def test_algorithm_uses_statistical_specification_not_theta():
    text = _paper_text()
    algorithm = text.split(r"\begin{algorithm}[t]", 1)[1].split(
        r"\end{algorithm}", 1
    )[0]
    assert r"\theta" not in algorithm
    for token in [
        r"\eta_0", r"\eta_{\min}", r"\eta_{\max}", r"\alpha", r"\beta",
        r"\rho", r"p\rho",
    ]:
        assert token in algorithm


def test_live_web_test_remains_separate_from_default_suite():
    live_test = (ROOT / "tests" / "test_e2e_live.py").read_text(encoding="utf-8")
    config = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "@pytest.mark.live_e2e" in live_test
    assert "live_e2e:" in config
    assert 'not live_e2e' in config


def test_paper_uses_anonymous_aaai_submission_format():
    text = _paper_text()
    assert r"\usepackage[submission]{aaai2027}" in text
    assert r"\bibliography{aaai2027}" in text
    assert "\\author{Anonymous Submission}" in text
    assert "neurips_2026" not in text
    assert "Author Name" not in text


def test_evidence_map_targets_current_submission():
    text = (ROOT / "analysis" / "EVIDENCE_TO_CLAIM_MAP.md").read_text(
        encoding="utf-8"
    )
    assert "docs/submission/paper.tex" in text
    assert "neurips_2026" not in text
