"""Apply the extraction-ablation paper blocks (PROJECT.md 9.6) to AuthorKit27/main.tex.

AuthorKit27/ is git-ignored, so the paper cannot travel through a branch or pull
request. This script applies the version-controlled blocks in
`analysis/extraction_ablation_paper_snippets.tex` to a working copy of the paper.

Sibling of `apply_paper_snippets.py` (the Adult calibration blocks), which only
needed to insert. This one also has to *replace* superseded text -- the 21.9%
compression claim, the cost-table rows, and two passages that describe the Caveman
Protocol as the deployed prompt -- so each block declares its own operation.

It is idempotent: if the new table label is already present it changes nothing.
Every anchor must match exactly once, or the script refuses to touch the file
rather than guess. Because main.tex is not under version control, a .bak is
written before any change.

    python analysis/apply_extraction_snippets.py --dry-run   # preview
    python analysis/apply_extraction_snippets.py             # apply
    python analysis/apply_extraction_snippets.py --paper PATH
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNIPPETS = HERE / "extraction_ablation_paper_snippets.tex"

# The worktree copy of this script sits under .claude/worktrees/<name>/analysis/,
# so fall back to the main checkout's paper when the sibling one is absent.
_CANDIDATE_PAPERS = [
    HERE.parents[0] / "AuthorKit27" / "main.tex",
    HERE.parents[3] / "AuthorKit27" / "main.tex",
]

BLOCK_HEADING = re.compile(r"^%\s*=+\s*BLOCK:\s*([a-z0-9-]+)\s*$", re.MULTILINE)

ALREADY_APPLIED = r"\label{tab:extraction}"

# Anchors are verbatim slices of main.tex. Each is asserted to appear exactly
# once; "replace" swaps the anchor for the block, "after" keeps the anchor and
# appends the block below it.
OPERATIONS: list[tuple[str, str, str]] = [
    (
        "contribution-bullet",
        "replace",
        """    \\item \\textbf{Cost-aware agentic prompting.} Autonomous loops make many LLM
    calls. We measure a compressed prompting scheme (the ``Caveman Protocol'')
    and report its real token, cost, and latency effects across three
    commercial providers.""",
    ),
    (
        "cost-aware-prompting",
        "replace",
        """The extraction system prompt uses a compressed, filler-free style (the
``Caveman Protocol'': no articles, pleasantries, or hedging), applied only to
internal prompts while user-facing output remains in natural language. We
measure its effect rather than assert it (see Evaluation, Q4).""",
    ),
    (
        "q3-pointer",
        "after",
        """and regex as a guaranteed-available fallback rather than the reverse.""",
    ),
    (
        "q5-paragraph",
        "replace",
        """Google's Gemini~2.5~Flash is both the
cheapest ($\\$0.0006$/call) and fastest (1.23\\,s); Anthropic's Claude
Sonnet~4.5 is the most expensive ($\\$0.0038$/call). We separately measure the
Caveman Protocol's prompt compression at a \\textbf{21.9\\%} reduction in prompt
tokens (242~$\\rightarrow$~189) on our extraction prompt.""",
    ),
    (
        "cost-table-rows",
        "replace",
        """Google    & 1.23 & 0.0006 & 250 & 200 \\\\
OpenAI    & 3.24 & 0.0020 & 241 & 137 \\\\
Anthropic & 3.67 & 0.0038 & 257 & 203 \\\\""",
    ),
    (
        "q6-subsection",
        "after",
        """\\label{tab:cost}
\\end{table}""",
    ),
    (
        "appendix-caveman-intro",
        "replace",
        """The compressed extraction system prompt is reproduced below; the token savings
in Q5 compare it against a naturally-worded equivalent of identical semantics
(both included in the released code).""",
    ),
    (
        "appendix-extraction",
        "after",
        """No fluff. No greeting. Output JSON only.
\\end{verbatim}\\end{scriptsize}\\end{quote}""",
    ),
]


def default_paper() -> Path:
    for candidate in _CANDIDATE_PAPERS:
        if candidate.exists():
            return candidate
    return _CANDIDATE_PAPERS[0]


def parse_blocks(text: str) -> dict[str, str]:
    """Split the snippets file into its named blocks."""
    matches = list(BLOCK_HEADING.finditer(text))
    if not matches:
        raise SystemExit(f"No BLOCK headings found in {SNIPPETS}")
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end]
        # Drop the explanatory comment lines that follow each heading.
        body = "\n".join(
            line for line in body.splitlines() if not line.startswith("%")
        )
        blocks[match.group(1)] = body.strip("\n")
    missing = [name for name, _, _ in OPERATIONS if name not in blocks]
    if missing:
        raise SystemExit(f"Snippets file is missing blocks: {missing}")
    return blocks


def apply_blocks(paper: str, blocks: dict[str, str]) -> str:
    for name, op, anchor in OPERATIONS:
        count = paper.count(anchor)
        if count == 0:
            raise SystemExit(
                f"Anchor for block {name!r} not found in the paper:\n---\n{anchor}\n---\n"
                "The surrounding text has changed; update OPERATIONS or apply by hand."
            )
        if count > 1:
            raise SystemExit(
                f"Anchor for block {name!r} is ambiguous (appears {count} times)."
            )
        block = blocks[name].strip("\n")
        replacement = block if op == "replace" else f"{anchor}\n\n{block}"
        paper = paper.replace(anchor, replacement, 1)
    return paper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paper_path = args.paper or default_paper()
    if not paper_path.exists():
        raise SystemExit(f"Paper not found: {paper_path}\nPass --paper with the path to main.tex.")

    original = paper_path.read_text(encoding="utf-8")
    if ALREADY_APPLIED in original:
        print(f"Already applied: {paper_path} contains {ALREADY_APPLIED}. Nothing to do.")
        return

    updated = apply_blocks(original, parse_blocks(SNIPPETS.read_text(encoding="utf-8")))
    added = len(updated.splitlines()) - len(original.splitlines())

    if args.dry_run:
        print(f"Dry run: {len(OPERATIONS)} blocks would apply cleanly to {paper_path} "
              f"({added:+d} lines).")
        for name, op, _ in OPERATIONS:
            print(f"  {op:7s} {name}")
        return

    backup = paper_path.with_suffix(".tex.bak")
    shutil.copy2(paper_path, backup)
    paper_path.write_text(updated, encoding="utf-8")
    print(f"Applied {len(OPERATIONS)} blocks to {paper_path} ({added:+d} lines)")
    print(f"Previous version saved to {backup}")
    print("Rebuild with pdflatex + bibtex and confirm the 6-page main-content budget.")


if __name__ == "__main__":
    main()
