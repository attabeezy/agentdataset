"""Submission guards for P2 paper artifacts and evidence contracts."""

from __future__ import annotations

import re
from pathlib import Path

from analysis import paper_artifacts


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "neurips" / "neurips_2026.tex"
FIGURE = ROOT / "docs" / "neurips" / "Figures" / "architecture.tex"


def test_generated_tables_match_committed_results():
    paper_artifacts.validate_manifests()
    for name, expected in paper_artifacts.render_tables().items():
        path = paper_artifacts.GENERATED / name
        assert path.exists(), name
        assert path.read_text(encoding="utf-8") == expected, name


def test_main_consumes_every_generated_result_table():
    text = PAPER.read_text(encoding="utf-8")
    for name in paper_artifacts.render_tables():
        assert f"\\input{{generated_tables/{name}}}" in text


def test_abstract_is_compact_and_claim_scoped():
    text = PAPER.read_text(encoding="utf-8")
    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.DOTALL
    ).group(1)
    plain = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", abstract)
    plain = re.sub(r"[{}$~\\]", " ", plain)
    words = re.findall(r"\b[\w.-]+\b", plain)
    assert len(words) <= 200
    assert ".304" in abstract
    assert "privacy mechanism" in abstract
    assert "unattended" in abstract


def test_q1_uses_final_deployed_prompt_replication():
    text = PAPER.read_text(encoding="utf-8")
    q1 = text.split(r"\subsection{Q1:", 1)[1].split(r"\subsection{Q2:", 1)[0]
    assert ".222, .480, and .304" in q1
    assert r"\$0.0343" in q1
    assert "13.9" in q1
    assert ".247" not in q1


def test_figure_walkthrough_is_source_backed_and_candid():
    text = FIGURE.read_text(encoding="utf-8")
    for token in ["rank 1", "N{=}447", "35.7", "r{=}.64", "omitted", "96.81"]:
        assert token in text


def test_algorithm_uses_statistical_specification_not_theta():
    text = PAPER.read_text(encoding="utf-8")
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


def test_paper_uses_neurips_template_and_completed_checklist():
    text = PAPER.read_text(encoding="utf-8")
    checklist = (PAPER.parent / "checklist.tex").read_text(encoding="utf-8")
    assert r"\usepackage{neurips_2026}" in text
    assert r"\input{checklist.tex}" in text
    assert "aaai2027" not in text
    assert r"\answerTODO" not in checklist
    assert r"\justificationTODO" not in checklist
