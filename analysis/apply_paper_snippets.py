"""Insert the Adult calibration paper blocks into AuthorKit27/main.tex.

AuthorKit27/ is git-ignored, so the paper cannot travel through a branch or pull
request. This script applies the version-controlled blocks in
`analysis/adult_calibration_paper_snippets.tex` to a working copy of the paper.

It is idempotent: if the calibration table label is already present it reports
that and changes nothing. Run from the repository root:

    python analysis/apply_paper_snippets.py                 # default paper path
    python analysis/apply_paper_snippets.py --paper PATH    # explicit path
    python analysis/apply_paper_snippets.py --dry-run       # preview only
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNIPPETS = HERE / "adult_calibration_paper_snippets.tex"

# The worktree copy of this script sits under .claude/worktrees/<name>/analysis/,
# so fall back to the main checkout's paper when the sibling one is absent.
_CANDIDATE_PAPERS = [
    HERE.parents[0] / "AuthorKit27" / "main.tex",
    HERE.parents[3] / "AuthorKit27" / "main.tex",
]

BLOCK_HEADING = re.compile(r"^%\s*-+\s*(Block 1|Block 2|Optional)\b.*$", re.MULTILINE)

# Each block is inserted directly after the paragraph ending with its anchor.
ANCHORS = {
    "Block 1": "rather than an unsupported robustness claim.",
    "Block 2": "share each seed's split, encoders, and test set.",
}

ALREADY_APPLIED = r"\label{tab:calibration}"


def default_paper() -> Path:
    for candidate in _CANDIDATE_PAPERS:
        if candidate.exists():
            return candidate
    return _CANDIDATE_PAPERS[0]


def parse_blocks(text: str) -> dict[str, str]:
    """Split the snippets file into its named blocks."""
    matches = list(BLOCK_HEADING.finditer(text))
    if not matches:
        raise SystemExit(f"No block headings found in {SNIPPETS}")
    blocks = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.end():end].strip("\n")
    missing = set(ANCHORS) - set(blocks)
    if missing:
        raise SystemExit(f"Snippets file is missing blocks: {sorted(missing)}")
    return blocks


def apply_blocks(paper_text: str, blocks: dict[str, str]) -> str:
    for name, anchor in ANCHORS.items():
        if anchor not in paper_text:
            raise SystemExit(
                f"Anchor for {name} not found in the paper: {anchor!r}\n"
                "The surrounding text has changed; insert the block by hand instead."
            )
        if paper_text.count(anchor) > 1:
            raise SystemExit(f"Anchor for {name} is ambiguous ({anchor!r} appears more than once)")
        insertion = f"{anchor}\n\n{blocks[name].strip()}"
        paper_text = paper_text.replace(anchor, insertion, 1)
    return paper_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paper = args.paper or default_paper()
    if not paper.exists():
        raise SystemExit(f"Paper not found: {paper}\nPass --paper with the path to main.tex.")

    paper_text = paper.read_text(encoding="utf-8")
    if ALREADY_APPLIED in paper_text:
        print(f"Already applied: {paper} contains {ALREADY_APPLIED}. Nothing to do.")
        return

    updated = apply_blocks(paper_text, parse_blocks(SNIPPETS.read_text(encoding="utf-8")))

    if args.dry_run:
        added = len(updated.splitlines()) - len(paper_text.splitlines())
        print(f"Dry run: would insert 2 blocks ({added} lines) into {paper}")
        return

    backup = paper.with_suffix(".tex.bak")
    shutil.copy2(paper, backup)
    paper.write_text(updated, encoding="utf-8")
    print(f"Inserted the Q2 paragraph and appendix table into {paper}")
    print(f"Previous version saved to {backup}")
    print("Rebuild with pdflatex + bibtex and confirm the page budget.")


if __name__ == "__main__":
    main()
