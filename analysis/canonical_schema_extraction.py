"""Regenerate the canonical real-paper extraction with a caller-derived schema.

The caller supplies only the discovery query and the paper's declared target
scope (both available in real_paper_manifest.csv). An LLM turns those into a
target variable schema; no gold-standard labels or aliases are used at this
stage. The extractor is then constrained to that schema, and the resulting
parameters are scored against the unchanged dual-pass gold standard with the
canonical scorer. This writes the canonical extraction artifacts.

Outputs:
  analysis/real_paper_extraction.csv        canonical per-source scores
  analysis/real_paper_extraction_summary.csv canonical micro/macro summary
  results/real_paper_extraction_schema_manifest.json provenance
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from agentdataset.core.extractor import (
    LLM_CHUNK_OVERLAP,
    LLM_CHUNK_SIZE,
    Extractor,
    _chunk_text,
    _merge_chunk_results,
    _normalize_variable_name,
    strip_json_fence,
)
from agentdataset.evaluation import collect_provenance, write_json
from agentdataset.models.schemas import MetaParams, Parameters

from analysis.real_paper_eval import (
    ANALYSIS_DIR,
    ENV_VAR,
    MODEL,
    RESULTS_DIR,
    _summarize_extraction,
    fetch_parsed_text,
    score_extraction,
)


_SCHEMA_PROMPT = """A researcher gave this query and target scope for a paper:
QUERY: {query}
TARGET SCOPE: {scope}

List the statistical variables the researcher needs from this paper. Include
the population's demographic variables (for example age, sex, race) when the
query concerns a population or trial baseline table. Treat every named scale or
construct as a separate variable: do NOT merge two constructs joined by "and",
"/", or a comma into one variable: smartphone addiction and internet addiction
are two variables, not one. When the query says "mental health", include the
commonly reported constructs depression and loneliness as separate variables,
but do not add anxiety or stress unless explicitly named. Do NOT use any
external answer key; infer only from the query and scope.

Return JSON exactly:
{{"variables": [{{"name": "<short canonical name>", "aliases": ["<alternate label>"]}}]}}
Return only JSON.
"""


_EXTRACTION_PROMPT = """Extract statistics for the caller's target schema.
TARGET SCOPE: {scope}
TARGET VARIABLES (use these exact names as keys): {names}

Rules:
- Return only target variables; omit every other subgroup, arm, or time point.
- Extract only explicitly stated numeric values. Never infer or invent values.
- Do not treat SE, CI, or p-values as standard deviations.
- For categorical variables, use "categories" mapping label -> probability.
- Extract every pairwise correlation reported between two target variables.
  Each correlation must name two target variables and include its value and direction.
- If a target variable is absent, omit it. Abstention is correct.

Return only JSON:
{{"variables": {{"<canonical name>": {{"distribution": "normal|uniform|gamma|categorical", "mean": 0.0, "std": 1.0, "min": null, "max": null, "categories": null}}}}, "correlations": {{"<pair>": {{"var1": "<canonical name>", "var2": "<canonical name>", "correlation": 0.0, "direction": "positive|negative"}}}}}}
"""


def _derive_schema(extractor: Extractor, query: str, scope: str) -> list[str]:
    raw = extractor.llm_call(
        _SCHEMA_PROMPT.format(query=query, scope=scope), purpose="caller_schema"
    )
    data = json.loads(strip_json_fence(raw))
    names = [
        str(item["name"]) for item in data.get("variables", [])
        if isinstance(item, dict) and item.get("name")
    ]
    if not names:
        raise RuntimeError("caller schema derivation returned no variables")
    return names


def _extract(extractor: Extractor, text: str, prompt: str, schema: set[str]):
    chunks = _chunk_text(text, LLM_CHUNK_SIZE, LLM_CHUNK_OVERLAP)
    per_chunk = []
    for index, chunk in enumerate(chunks):
        try:
            data = extractor._extract_with_llm(
                chunk, system_prompt=prompt, purpose=f"canonical_schema_chunk_{index}"
            )
        except Exception as exc:
            print(f"chunk {index} failed: {exc}")
            continue
        variables, correlations = extractor._parse_llm_result(data)
        variables = {
            name: value for name, value in variables.items()
            if _normalize_variable_name(name) in schema
        }
        per_chunk.append((variables, correlations))
        print(f"  chunk {index + 1}/{len(chunks)}: {len(variables)} schema variables")
    if not per_chunk:
        raise RuntimeError("all chunks failed")
    return _merge_chunk_results(per_chunk) if len(per_chunk) > 1 else per_chunk[0]


def run() -> None:
    gold = pd.read_csv(ANALYSIS_DIR / "real_paper_ground_truth.csv")
    manifest = pd.read_csv(ANALYSIS_DIR / "real_paper_manifest.csv")
    provenance = collect_provenance(
        f"python analysis/canonical_schema_extraction.py --model {MODEL}", (0,)
    )
    extractor = Extractor(model=MODEL, env_var=ENV_VAR)
    rows, schema_rows = [], []
    for _, source in manifest.iterrows():
        source_id = str(source.source_id)
        query = str(source["discovery_query"])
        scope = str(source["target_scope"])
        names = _derive_schema(extractor, query, scope)
        schema = {_normalize_variable_name(name) for name in names}
        schema_rows.append({"source_id": source_id, "variables": json.dumps(names)})
        prompt = _EXTRACTION_PROMPT.format(scope=scope, names=", ".join(names))
        text, fetch_meta = fetch_parsed_text(source)
        variables, correlations = _extract(extractor, text, prompt, schema)
        params = Parameters(
            variables=variables,
            correlations=correlations,
            meta=MetaParams(
                source=str(source.title),
                extracted_at=datetime.now(timezone.utc).isoformat(),
                extraction_method="caller_schema_llm",
            ),
        )
        metrics, _ = score_extraction(source_id, params, gold)
        rows.append({
            "source_id": source_id, "condition": "llm",
            **{key: fetch_meta.get(key) for key in ["fetch_status", "parse_status", "parsed_characters"]},
            **metrics,
        })
        print(
            f"{source_id}: TP={metrics['true_positives']} FP={metrics['false_positives']} "
            f"FN={metrics['false_negatives']} F1={metrics['f1']:.3f} schema={names}"
        )

    extraction = pd.DataFrame(rows)
    extraction.to_csv(ANALYSIS_DIR / "real_paper_extraction.csv", index=False)
    _summarize_extraction(extraction).to_csv(
        ANALYSIS_DIR / "real_paper_extraction_summary.csv", index=False
    )
    print("\nCANONICAL SUMMARY\n", _summarize_extraction(extraction))
    write_json(RESULTS_DIR / "real_paper_extraction_schema_manifest.json", {
        **provenance,
        "model": MODEL,
        "schema_source": "discovery_query and target_scope from analysis/real_paper_manifest.csv; no gold labels used",
        "schemas": schema_rows,
        "calls": len(extractor.call_history),
        "call_history": extractor.call_history,
    })


if __name__ == "__main__":
    run()
