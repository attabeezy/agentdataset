"""
AutoSource Extraction Pipeline
PDF → Markdown → parameters.json

Usage: uv run extract.py --pdf source_library/your_report.pdf
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Any
import re

import fitz  # PyMuPDF
import requests


# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autosource"
OUTPUT_DIR = Path("results")
PARAMS_FILENAME = "parameters.json"


# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------


def pdf_to_markdown(pdf_path: str) -> str:
    """Convert PDF to clean Markdown using PyMuPDF."""
    doc = fitz.open(pdf_path)
    sections = []

    for page_num, page in enumerate(doc):
        # Extract text blocks
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text, block_type, block_id = block

            if block_type == 0:  # Text block
                text = text.strip()
                if not text:
                    continue

                # Detect headers (large font, first line, short text)
                is_header = len(text.split()) <= 5 and page_num == 0

                if is_header:
                    sections.append(f"# {text}\n")
                else:
                    # Basic formatting: paragraphs, lists
                    if text.startswith(("•", "-", "*")):
                        sections.append(f"- {text.lstrip('•-*) ')}\n")
                    elif re.match(r"\d+\.", text):
                        sections.append(f"{text}\n")
                    else:
                        sections.append(f"{text}\n\n")

    doc.close()
    return "".join(sections)


def extract_statistics(markdown_text: str) -> dict[str, Any]:
    """
    Extract statistical parameters from Markdown using LLM.

    This is a simplified placeholder. In production, you'd call an LLM API
    with a prompt like:

    > "You are an expert statistician. Strip the report of all prose.
    > Extract only variables (name, distribution type, parameters).
    > Output strict JSON."

    For now, we use regex-based extraction as a demo.
    """
    stats = {
        "variables": {},
        "correlations": {},
        "meta": {
            "source": "YARA Credit Report",
            "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "extraction_method": "regex_placeholder",
        },
    }

    # Example: Extract simple Mean/STD patterns (replace with LLM in production)
    # Pattern: "Mean: X, Std: Y" or "Average: X (Std: Y)"
    pattern = r"(?:mean|average|expected|expected\s+value)\s*[:\s]+([\d.]+)\s*,?\s*(?:std|standard\s+deviation|variance|std\s+dev)\s*[:\s]+([\d.]+)"

    for match in re.finditer(pattern, markdown_text, re.IGNORECASE):
        mean = float(match.group(1))
        std = float(match.group(2))

        # Generate variable name from context (simplified)
        var_name = f"variable_{len(stats['variables']) + 1}"
        stats["variables"][var_name] = {
            "distribution": "normal",
            "mean": mean,
            "std": std,
            "min": mean - 2 * std,
            "max": mean + 2 * std,
        }

    # Example: Extract correlation patterns
    corr_pattern = r"(?:correlation|correl|relationship)\s+between\s+(\w+)\s+and\s+(\w+)\s+is\s+(?:positive|negative)?\s*(?:strong|moderate|weak)?\s*(?:correlation)?\s*[:\s]*r\s*=\s*([\d.]+)"

    for match in re.finditer(corr_pattern, markdown_text, re.IGNORECASE):
        var1, var2, r_value = match.groups()
        stats["correlations"][f"{var1} ↔ {var2}"] = {
            "correlation": float(r_value),
            "direction": "positive" if float(r_value) > 0 else "negative",
        }

    return stats


def save_parameters(stats: dict, output_path: Path) -> None:
    """Save extracted parameters to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Parameters saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Extract statistics from PDF reports")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR / PARAMS_FILENAME,
        help="Output parameters.json path",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Extracting from: {pdf_path}")
    output_path = Path(args.output)

    t0 = time.time()

    # Step 1: PDF → Markdown
    print("Stage 1: Converting PDF to Markdown...")
    markdown = pdf_to_markdown(str(pdf_path))
    print(f"  Extracted {len(markdown):,} characters")

    # Step 2: Markdown → parameters.json (placeholder for LLM)
    print("Stage 2: Extracting statistics...")
    stats = extract_statistics(markdown)

    # Step 3: Save results
    save_parameters(stats, output_path)

    t1 = time.time()
    print(f"Total extraction time: {t1 - t0:.1f}s")
    print(
        f"Found {len(stats['variables'])} variables and {len(stats['correlations'])} correlations"
    )

    # Print summary
    print("\nExtracted Statistics Summary:")
    for var_name, params in stats["variables"].items():
        print(
            f"  {var_name}: {params['distribution']}(μ={params['mean']:.2f}, σ={params['std']:.2f})"
        )


if __name__ == "__main__":
    main()
