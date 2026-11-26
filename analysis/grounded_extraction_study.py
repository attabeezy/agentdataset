"""Before/after study for evidence-grounded LLM extraction.

The production extractor asks only for values, so unsupported numbers survive as
claims. This study reruns the same three real papers with a prompt that also
requires a verbatim evidence span, then drops any statistic whose numeric value
does not appear in that span. Both arms are scored against the same dual-pass
gold standard, so the precision/recall delta is attributable to grounding.

Outputs (results/):
  grounded_extraction_study.csv       per-source, per-arm extraction scores
  grounded_extraction_study_raw.csv   per-source, per-arm extracted variables
  grounded_extraction_study_manifest.json provenance and call accounting
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from agentdataset.core.extractor import (
    LLM_CHUNK_OVERLAP,
    LLM_CHUNK_SIZE,
    Extractor,
    _chunk_text,
    _merge_chunk_results,
)
from agentdataset.evaluation import collect_provenance, write_json

from analysis.real_paper_eval import (
    ANALYSIS_DIR,
    ENV_VAR,
    MODEL,
    RESULTS_DIR,
    fetch_parsed_text,
    score_extraction,
)


GROUND_PROMPT = """
You are an expert statistician extracting statistics from a research paper.
For every statistic you report, you MUST quote the exact words from the text that
support it. A statistic without a quotation is invalid and must be omitted.

Grounding rules:
- Extract a statistic ONLY if a numeric value is explicitly stated in the text.
- The "evidence" field MUST be a verbatim substring of the input text and MUST
  contain the number you report. Never paraphrase evidence.
- Do NOT treat standard errors, confidence intervals, or p-values as standard deviations.
- Do NOT extract from negated findings without numbers.
- If a paper reports a correlation whose marginals are not reported, omit it.
- If no reportable statistics are present, return empty objects. This is correct.

Return a JSON object with this exact schema:
{
  "variables": {
    "<name>": {
      "distribution": "normal|uniform|gamma|categorical",
      "mean": 0.0,
      "std": 1.0,
      "min": null,
      "max": null,
      "categories": null,
      "population": "<population or empty string>",
      "arm": "<arm or empty string>",
      "time": "<time point or empty string>",
      "evidence": "<verbatim excerpt containing the number>"
    }
  },
  "correlations": {
    "<key>": {
      "var1": "<name>", "var2": "<name>",
      "correlation": 0.0, "direction": "positive|negative",
      "evidence": "<verbatim excerpt containing the number>"
    }
  }
}
Output only the JSON object.
"""

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_TOL = 0.011


def _grounded(value, evidence: str, tolerance: float = _TOL) -> bool:
    """True when the reported number appears in the verbatim evidence span."""
    if value is None or not evidence:
        return False
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    return any(
        abs(float(token) - target) <= tolerance
        for token in _NUMBER.findall(str(evidence))
    )


def _name_tokens(name: str) -> list[str]:
    return [
        token for token in re.split(r"[^a-z0-9]+", str(name).lower())
        if len(token) >= 4 and not token.isdigit()
    ]


def _name_supported(name: str, evidence: str, chunk: str, *values) -> bool:
    """Require the variable label to be near a reported number in source text."""
    tokens = _name_tokens(name)
    if not tokens:
        return False
    lowered_evidence = evidence.lower()
    if any(token in lowered_evidence for token in tokens):
        return True
    for value in values:
        if value is None:
            continue
        try:
            target = float(value)
        except (TypeError, ValueError):
            continue
        for match in _NUMBER.finditer(chunk):
            if abs(float(match.group()) - target) > _TOL:
                continue
            window = chunk[
                max(0, match.start() - 150): min(len(chunk), match.end() + 150)
            ].lower()
            if any(token in window for token in tokens):
                return True
    return False


def _filter_grounded(data: dict, chunk: str) -> dict:
    """Keep only entries whose numbers are grounded near their variable label."""
    raw_vars = data.get("variables", {}) if isinstance(data, dict) else {}
    variables = {}
    if isinstance(raw_vars, dict):
        for name, value in raw_vars.items():
            if not isinstance(value, dict):
                continue
            evidence = str(value.get("evidence") or "")
            if value.get("distribution") == "categorical" and value.get("categories"):
                categories = value.get("categories")
                if isinstance(categories, dict):
                    probabilities = [p for p in categories.values() if p is not None]
                elif isinstance(categories, list):
                    probabilities = [
                        p.get("probability") for p in categories
                        if isinstance(p, dict) and p.get("probability") is not None
                    ]
                else:
                    probabilities = []
                if not probabilities or not all(_grounded(p, evidence) for p in probabilities):
                    continue
                if not _name_supported(name, evidence, chunk, *probabilities):
                    continue
                variables[name] = value
                continue
            if not _grounded(value.get("mean"), evidence):
                continue
            if not _grounded(value.get("std"), evidence):
                continue
            if not _name_supported(name, evidence, chunk, value.get("mean"), value.get("std")):
                continue
            variables[name] = value

    raw_corrs = data.get("correlations", {}) if isinstance(data, dict) else {}
    correlations = {}
    if isinstance(raw_corrs, dict):
        for key, value in raw_corrs.items():
            if not isinstance(value, dict):
                continue
            if _grounded(value.get("correlation"), str(value.get("evidence") or "")):
                correlations[key] = value
    return {"variables": variables, "correlations": correlations}


def _extract(extractor: Extractor, text: str, grounded: bool):
    chunks = _chunk_text(text, LLM_CHUNK_SIZE, LLM_CHUNK_OVERLAP)
    per_chunk = []
    for index, chunk in enumerate(chunks):
        kwargs = {"purpose": f"grounded_chunk_{index}"} if grounded else {
            "purpose": f"baseline_chunk_{index}"
        }
        if grounded:
            kwargs["system_prompt"] = GROUND_PROMPT
        try:
            data = extractor._extract_with_llm(chunk, **kwargs)
        except Exception as exc:
            print(f"chunk {index} failed: {exc}")
            continue
        if grounded:
            data = _filter_grounded(data, chunk)
            grounded_variables = len(data["variables"])
        else:
            grounded_variables = 0
        variables, correlations = extractor._parse_llm_result(data)
        per_chunk.append((variables, correlations))
        print(
            f"  chunk {index + 1}/{len(chunks)}: kept "
            f"{grounded_variables if grounded else len(variables)} variables (grounded)"
            if grounded else f"  chunk {index + 1}/{len(chunks)}: {len(variables)} variables"
        )
    if not per_chunk:
        raise RuntimeError("all chunks failed")
    return _merge_chunk_results(per_chunk) if len(per_chunk) > 1 else per_chunk[0]


def run() -> None:
    gold = pd.read_csv(ANALYSIS_DIR / "real_paper_ground_truth.csv")
    manifest = pd.read_csv(ANALYSIS_DIR / "real_paper_manifest.csv")
    provenance = collect_provenance(
        f"python analysis/grounded_extraction_study.py --model {MODEL}", (0,)
    )
    extractor = Extractor(model=MODEL, env_var=ENV_VAR)

    rows, raw_rows = [], []
    for _, source in manifest.iterrows():
        source_id = str(source.source_id)
        text, fetch_meta = fetch_parsed_text(source)
        print(f"{source_id}: {len(text)} chars")
        for arm, grounded in [("baseline", False), ("grounded", True)]:
            variables, correlations = _extract(extractor, text, grounded)
            from agentdataset.models.schemas import MetaParams, Parameters

            params = Parameters(
                variables=variables,
                correlations=correlations,
                meta=MetaParams(
                    source=str(source.title),
                    extracted_at=datetime.now(timezone.utc).isoformat(),
                    extraction_method="grounded_llm" if grounded else "llm",
                ),
            )
            metrics, _ = score_extraction(source_id, params, gold)
            rows.append({"source_id": source_id, "arm": arm, **metrics})
            for name, variable in variables.items():
                raw_rows.append({
                    "source_id": source_id, "arm": arm, "name": name,
                    "mean": variable.mean, "std": variable.std,
                    "distribution": variable.distribution,
                })
            print(
                f"  {arm}: TP={metrics['true_positives']} FP={metrics['false_positives']} "
                f"FN={metrics['false_negatives']} F1={metrics['f1']:.3f}"
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "grounded_extraction_study.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(
        RESULTS_DIR / "grounded_extraction_study_raw.csv", index=False
    )
    totals = frame.groupby("arm")[["true_positives", "false_positives", "false_negatives"]].sum()
    totals["precision"] = totals["true_positives"] / (totals["true_positives"] + totals["false_positives"])
    totals["recall"] = totals["true_positives"] / (totals["true_positives"] + totals["false_negatives"])
    totals["f1"] = 2 * totals["precision"] * totals["recall"] / (totals["precision"] + totals["recall"])
    print("\nMICRO TOTALS\n", totals)
    write_json(RESULTS_DIR / "grounded_extraction_study_manifest.json", {
        **provenance,
        "model": MODEL,
        "prompt": "GROUND_PROMPT with verbatim evidence requirement",
        "tolerance": _TOL,
        "calls": len(extractor.call_history),
        "call_history": extractor.call_history,
    })
    totals.round(6).to_csv(RESULTS_DIR / "grounded_extraction_study_totals.csv")


if __name__ == "__main__":
    run()
