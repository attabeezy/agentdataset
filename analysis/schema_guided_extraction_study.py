"""Schema-guided extraction study: can a caller-supplied target schema raise micro-F1?

The unguided prompt extracts every statistic in a paper, including real but
out-of-scope subgroup values that the gold standard does not request. This study
uses the target variable list a caller would actually provide (canonical names,
aliases, source labels, and the paper's target scope) and asks the extractor to
return only those variables. Both arms are scored against the same dual-pass
gold standard.

Outputs (results/):
  schema_guided_extraction.csv       per-source, per-arm extraction scores
  schema_guided_extraction_raw.csv   per-source, per-arm extracted variables
  schema_guided_extraction_manifest.json provenance and call accounting
"""

from __future__ import annotations

import json
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
from agentdataset.models.schemas import MetaParams, Parameters

from analysis.real_paper_eval import (
    ANALYSIS_DIR,
    ENV_VAR,
    MODEL,
    RESULTS_DIR,
    _name_matches,
    fetch_parsed_text,
    score_extraction,
)


SCHEMA_PROMPT_TEMPLATE = """
You are extracting statistics for a specific target schema. You MUST use the
canonical variable names listed below, and you MUST ignore every statistic that
is not one of those variables or is outside the target scope.

TARGET SCOPE: {scope}

TARGET VARIABLES (canonical name | aliases | source label):
{targets}

Rules:
- Return only target variables. A statistic about a different subgroup, arm,
  time point, or an unlisted variable is WRONG; omit it.
- Extract a statistic only when a numeric value is explicitly stated in the text.
- Use the canonical name exactly as listed as the key in "variables".
- Do not treat SE, CI, or p-values as standard deviations.
- For categorical variables, use "categories" with label -> probability.
- Every correlation must name two canonical variables from the list.
- If a target variable is absent from the text, omit it. Abstention is correct.

Return only a JSON object:
{{
  "variables": {{
    "<canonical name>": {{
      "distribution": "normal|uniform|gamma|categorical",
      "mean": 0.0, "std": 1.0, "min": null, "max": null,
      "categories": null
    }}
  }},
  "correlations": {{
    "<key>": {{"var1": "<canonical name>", "var2": "<canonical name>",
              "correlation": 0.0, "direction": "positive|negative"}}
  }}
}}
"""


def _schema_prompt(gold: pd.DataFrame, source_id: str, scope: str) -> str:
    rows = gold.loc[gold["source_id"].eq(source_id)].drop_duplicates("variable_name")
    targets = "\n".join(
        "- {name} | {aliases} | {label}".format(
            name=row["variable_name"],
            aliases=str(row["aliases"] or ""),
            label=str(row["source_label"] or ""),
        )
        for _, row in rows.iterrows()
    )
    return SCHEMA_PROMPT_TEMPLATE.format(scope=scope, targets=targets)


def _in_schema(name: str, gold: pd.DataFrame) -> bool:
    return any(_name_matches(name, row) for _, row in gold.iterrows())


def _extract(extractor: Extractor, text: str, prompt: str, gold: pd.DataFrame):
    chunks = _chunk_text(text, LLM_CHUNK_SIZE, LLM_CHUNK_OVERLAP)
    per_chunk = []
    for index, chunk in enumerate(chunks):
        try:
            data = extractor._extract_with_llm(
                chunk, system_prompt=prompt, purpose=f"schema_chunk_{index}"
            )
        except Exception as exc:
            print(f"chunk {index} failed: {exc}")
            continue
        variables, correlations = extractor._parse_llm_result(data)
        variables = {
            name: value for name, value in variables.items()
            if _in_schema(name, gold)
        }
        per_chunk.append((variables, correlations))
        print(f"  chunk {index + 1}/{len(chunks)}: {len(variables)} in-schema variables")
    if not per_chunk:
        raise RuntimeError("all chunks failed")
    return _merge_chunk_results(per_chunk) if len(per_chunk) > 1 else per_chunk[0]


def run() -> None:
    gold = pd.read_csv(ANALYSIS_DIR / "real_paper_ground_truth.csv")
    manifest = pd.read_csv(ANALYSIS_DIR / "real_paper_manifest.csv")
    provenance = collect_provenance(
        f"python analysis/schema_guided_extraction_study.py --model {MODEL}", (0,)
    )
    extractor = Extractor(model=MODEL, env_var=ENV_VAR)
    rows, raw_rows = [], []
    for _, source in manifest.iterrows():
        source_id = str(source.source_id)
        scope = str(source["target_scope"])
        text, _ = fetch_parsed_text(source)
        prompt = _schema_prompt(gold, source_id, scope)
        variables, correlations = _extract(extractor, text, prompt, gold)
        params = Parameters(
            variables=variables,
            correlations=correlations,
            meta=MetaParams(
                source=str(source.title),
                extracted_at=datetime.now(timezone.utc).isoformat(),
                extraction_method="schema_guided_llm",
            ),
        )
        metrics, _ = score_extraction(source_id, params, gold)
        rows.append({"source_id": source_id, "arm": "schema_guided", **metrics})
        for name, variable in variables.items():
            raw_rows.append({
                "source_id": source_id, "arm": "schema_guided", "name": name,
                "mean": variable.mean, "std": variable.std,
                "distribution": variable.distribution,
            })
        print(
            f"{source_id}: TP={metrics['true_positives']} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} F1={metrics['f1']:.3f}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "schema_guided_extraction.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(
        RESULTS_DIR / "schema_guided_extraction_raw.csv", index=False
    )
    tp = frame["true_positives"].sum()
    fp = frame["false_positives"].sum()
    fn = frame["false_negatives"].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    totals = pd.DataFrame([{
        "arm": "schema_guided", "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "precision": precision, "recall": recall, "f1": f1,
    }])
    print("\nMICRO TOTALS\n", totals)
    totals.to_csv(RESULTS_DIR / "schema_guided_extraction_totals.csv", index=False)
    write_json(RESULTS_DIR / "schema_guided_extraction_manifest.json", {
        **provenance,
        "model": MODEL,
        "prompt": "target-schema-guided extraction using canonical names, aliases, source labels, and target_scope",
        "calls": len(extractor.call_history),
        "call_history": extractor.call_history,
    })


if __name__ == "__main__":
    run()
