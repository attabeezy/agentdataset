"""Measure the agentic review gate and cross-source reconciliation.

The frozen real-paper artifacts call ``Extractor.extract_parameters`` directly,
so they do not exercise the review stage added in §10. This study closes that
gap: it fetches three source documents, builds claims with real evidence passages
from the dual-pass gold standard and with no passage from recorded unsupported
extractions, runs the shipped LLM reviewer over them, and measures the effect of
accepting only reviewed claims on deterministic synthesis. The reviewer is the
production agent; the synthesis loop is replaced by a single fixed-noise sample
so the comparison is cheap and reproducible.

Outputs (all under results/):
  review_gate_study_raw.csv       per-source, per-condition, per-seed metrics
  review_gate_study_summary.csv   one row per source and condition
  review_gate_study_manifest.json provenance, decisions, and call accounting
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.reconciliation import reconcile_parameters
from agentdataset.core.review import parameters_from_claims
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.core.validator import Validator
from agentdataset.evaluation import collect_provenance, write_json
from agentdataset.models.schemas import Claim, EvidenceLocator

from analysis.real_paper_eval import (
    ANALYSIS_DIR,
    CACHE_DIR,
    ENV_VAR,
    MODEL,
    RESULTS_DIR,
    SEEDS,
    fetch_parsed_text,
    gold_to_parameters,
)


NOISE = 0.02
STATS = {"mean", "std", "min", "max"}
_UNSUPPORTED = re.compile(r"^(.+):(mean|std|min|max)=(.+)$")


def _passage(text: str, needle: str, width: int = 600) -> str | None:
    """Return a source-text window around a gold label, or None if absent."""
    if not needle:
        return None
    position = text.lower().find(needle.lower())
    if position < 0:
        return None
    return text[max(0, position - width // 2): position + width // 2]


def _claims_for_source(
    source_id: str, gold: pd.DataFrame, errors: pd.DataFrame, source_text: str
):
    """Return (claims, claim_group) from verified and recorded unsupported items."""
    claims, groups = [], {}
    for _, row in gold.loc[gold["source_id"].eq(source_id)].iterrows():
        stat = str(row["statistic_type"])
        if stat not in STATS and stat != "correlation":
            continue
        claim = Claim(
            claim_id=str(row["gold_id"]),
            source_id=source_id,
            variable_name=str(row["variable_name"]),
            statistic_type=stat,
            value=float(row["value"]),
            claim_type="reported",
            method="manual",
            confidence=0.9,
            evidence=EvidenceLocator(
                source_id=source_id,
                locator=str(row["page_or_table"] or ""),
                passage=_passage(source_text, str(row["source_label"] or ""))
                or _passage(source_text, str(row["value"])),
            ),
            population=(None if pd.isna(row["population"]) else str(row["population"])),
            subgroup=(None if pd.isna(row["subgroup"]) else str(row["subgroup"])),
            time_point=(None if pd.isna(row["time_period"]) else str(row["time_period"])),
            arm=(None if pd.isna(row["arm"]) else str(row["arm"])),
        )
        claims.append(claim)
        groups[claim.claim_id] = "verified"

    unsupported = errors.loc[
        errors["source_id"].eq(source_id)
        & errors["error_category"].eq("unsupported_or_out_of_scope")
    ]
    for idx, detail in enumerate(unsupported["detail"]):
        match = _UNSUPPORTED.match(str(detail))
        if not match:
            continue
        name, stat, value = match.groups()
        claim = Claim(
            claim_id=f"{source_id}_unsupported_{idx}",
            source_id=source_id,
            variable_name=name.strip(),
            statistic_type=stat,
            value=float(value),
            claim_type="inferred",
            method="llm",
            confidence=0.3,
        )
        claims.append(claim)
        groups[claim.claim_id] = "unsupported"
    return claims, groups


def _metrics(params, source, seed: int) -> dict:
    if not params.variables:
        return {
            "n_variables": 0, "n_pairs_declared": len(params.correlations),
            "n_pairs_applied": 0, "fidelity_score": 0.0,
            "correlation_score": 0.0, "ks_score": 0.0, "bias_score": 0.0,
        }
    synth = Synthesizer(n_rows=int(source.sample_size), seed=seed)
    frame = synth.synthesize(params, noise_level=NOISE)
    report = Validator().validate(frame, params)
    return {
        "n_variables": len(params.variables),
        "n_pairs_declared": len(params.correlations),
        "n_pairs_applied": sum(d.n_pairs_applied for d in synth.diagnostics),
        "fidelity_score": report.overall_score,
        "correlation_score": report.corr_score,
        "ks_score": report.ks_score,
        "bias_score": report.bias_score,
    }


def run() -> None:
    gold = pd.read_csv(ANALYSIS_DIR / "real_paper_ground_truth.csv")
    errors = pd.read_csv(ANALYSIS_DIR / "real_paper_errors.csv")
    manifest = pd.read_csv(ANALYSIS_DIR / "real_paper_manifest.csv")
    source_ids = [str(s) for s in manifest["source_id"]]

    provenance = collect_provenance(
        f"python analysis/pipeline_review_study.py --model {MODEL}", SEEDS
    )
    # Keep sessions out of the real-paper cache directory: its mere existence
    # un-skips offline tests that expect cached source text to be present.
    session_dir = CACHE_DIR.parent / "review_study" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, decision_rows, fetch_meta_rows, gold_params = [], [], [], {}
    for _, source in manifest.iterrows():
        source_id = str(source.source_id)
        source_text, fetch_meta = fetch_parsed_text(source)
        fetch_meta_rows.append({"source_id": source_id, **fetch_meta})
        claims, groups = _claims_for_source(source_id, gold, errors, source_text)
        if not claims:
            raise AssertionError(f"no claims built for {source_id}")

        orchestrator = Orchestrator(
            session_id=f"review_study_{source_id}",
            base_dir=str(session_dir), model=MODEL, env_var=ENV_VAR,
        )
        review = orchestrator.review_claims_with_agent(claims)
        for claim in review.claims:
            decision_rows.append({
                "source_id": source_id,
                "claim_id": claim.claim_id,
                "group": groups[claim.claim_id],
                "statistic_type": claim.statistic_type,
                "decision": claim.status,
                "review_note": claim.review_note,
            })

        accepted = [c for c in review.claims if c.status in {"accepted", "edited"}]
        reviewed = parameters_from_claims(accepted, source=source_id)
        reviewed.meta.extraction_method = "agent_review"
        unreviewed = parameters_from_claims(
            [c.model_copy(update={"status": "accepted"}) for c in claims],
            source=source_id,
        )
        unreviewed.meta.extraction_method = "llm_unreviewed"

        gold_params[source_id] = gold_to_parameters(source_id, gold)
        for condition, params in {
            "gold": gold_params[source_id],
            "unreviewed": unreviewed,
            "reviewed": reviewed,
        }.items():
            for seed in SEEDS:
                raw_rows.append({
                    "source_id": source_id,
                    "condition": condition,
                    "seed": seed,
                    "noise": NOISE,
                    **_metrics(params, source, seed),
                })

    _, decisions = reconcile_parameters(list(gold_params.values()))
    reconciliation = [
        {
            "target_variable": d.target_variable,
            "method": d.method,
            "accepted_claim_ids": len(d.accepted_claim_ids),
        }
        for d in decisions
    ]

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(RESULTS_DIR / "review_gate_study_raw.csv", index=False)
    summary = (
        raw.groupby(["source_id", "condition"])[
            ["n_variables", "n_pairs_declared", "n_pairs_applied",
             "fidelity_score", "correlation_score", "ks_score", "bias_score"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(
        RESULTS_DIR / "review_gate_study_summary.csv", index=False
    )
    pd.DataFrame(decision_rows).to_csv(
        RESULTS_DIR / "review_gate_study_decisions.csv", index=False
    )
    write_json(RESULTS_DIR / "review_gate_study_manifest.json", {
        **provenance,
        "model": MODEL,
        "fetch": fetch_meta_rows,
        "noise": NOISE,
        "seeds": list(SEEDS),
        "claim_groups": {"verified": "gold-standard statistics", "unsupported": "recorded unsupported extractions"},
        "reconciliation_decisions": reconciliation,
        "reconciliation_methods": pd.Series(
            [d["method"] for d in reconciliation]
        ).value_counts().to_dict(),
        "review_calls": len(decision_rows),
    })
    print(f"wrote review-gate study for {len(source_ids)} sources")


if __name__ == "__main__":
    run()
