"""Correlation-matrix validity and fallback audit (§9.7, P1).

Answers the question the paper currently cannot: how often is the advertised
copula path actually taken? For every correlation matrix the pipeline builds we
record the minimum eigenvalue, positive-definite status, the action taken, and
the source class, then compare the shipped drop-all fallback against a nearest
positive-semidefinite repair.

Input classes swept:
  templated       - embedded ground-truth specs and literature-style prose
                    rendered from real UCI dataset statistics (dense pairwise)
  gold            - human-annotated real-paper statistics and challenge-set gold
  auto_extracted  - the regex extractor run over the same texts (no API calls)

Usage (PowerShell):
  uv run python analysis/corr_validity_audit.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark  # noqa: E402
from agentdataset.core.extractor import Extractor  # noqa: E402
from agentdataset.core.synthesizer import Synthesizer  # noqa: E402
from agentdataset.core.validator import Validator  # noqa: E402
from agentdataset.evaluation import (  # noqa: E402
    RAW_RESULT_COLUMNS,
    collect_provenance,
    summarize_long_results,
    validate_raw_results,
    write_json,
)

RESULTS_DIR = ROOT / "results"
EXPERIMENT_ID = "corr_validity"
N_ROWS = 2000
SEED = 42


def _spec(source_id, source_class, params, extraction_method):
    return {
        "source_id": source_id,
        "source_class": source_class,
        "extraction_method": extraction_method,
        "params": params,
    }


def collect_specs() -> tuple[list[dict], list[str]]:
    """Gather every correlation-bearing spec the pipeline can produce.

    Returns (specs, notes) where notes records any source that could not be
    loaded, so a partial sweep is visible in the manifest rather than silent.
    """
    specs: list[dict] = []
    notes: list[str] = []

    texts = benchmark._build_ablation_texts()

    # --- templated: embedded ground truth (sparse, 2 variables, 1 pair) -------
    for text in texts:
        specs.append(_spec(
            text["id"], "templated",
            benchmark._params_from_ground_truth(text, "corr_validity"),
            "ground_truth",
        ))

    # --- templated: dense prose rendered from real UCI dataset statistics -----
    # _dataset_to_source_text emits every pairwise correlation, which is where
    # near-singular matrices actually come from in practice.
    for loader_name in ("_load_adult", "_load_pima", "_load_german_credit"):
        loader = getattr(benchmark, loader_name, None)
        if loader is None:
            notes.append(f"{loader_name}: not found")
            continue
        try:
            dataset = loader()
            text = benchmark._dataset_to_source_text(dataset)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{loader_name}: {type(exc).__name__}: {exc}")
            continue

        name = loader_name.removeprefix("_load_")
        extractor = Extractor(api_key="", env_var="_CORR_AUDIT_NO_KEY")
        specs.append(_spec(
            f"uci_{name}", "templated",
            extractor.extract_parameters(text, f"uci_{name}"),
            "regex_on_dataset_prose",
        ))

    # --- gold: annotated real-paper statistics --------------------------------
    gold_path = ROOT / "analysis" / "real_paper_ground_truth.csv"
    if gold_path.exists():
        from analysis.real_paper_eval import gold_to_parameters

        gold = pd.read_csv(gold_path)
        for source_id in sorted(gold.source_id.unique()):
            params = gold_to_parameters(source_id, gold)
            if params.correlations:
                specs.append(_spec(source_id, "gold", params, "gold"))
    else:
        notes.append("real_paper_ground_truth.csv: missing")

    # --- gold: extraction challenge set ---------------------------------------
    try:
        cases = benchmark.load_challenge_set()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"challenge_set: {type(exc).__name__}: {exc}")
        cases = []

    for case in cases:
        if not case.get("gold_correlations"):
            continue
        params = _challenge_case_to_parameters(case)
        if params is not None and params.correlations:
            specs.append(_spec(
                f"challenge_{case['case_id']}", "gold", params, "gold_challenge"
            ))

    # --- auto_extracted: regex over the templated prose ------------------------
    for text in texts:
        extractor = Extractor(api_key="", env_var="_CORR_AUDIT_NO_KEY")
        params = extractor.extract_parameters(text["text"], text["id"])
        if params.correlations and len(params.variables) >= 2:
            specs.append(_spec(
                f"regex_{text['id']}", "auto_extracted", params, "regex"
            ))

    for case in cases:
        extractor = Extractor(api_key="", env_var="_CORR_AUDIT_NO_KEY")
        params = extractor.extract_parameters(case["text"], case["case_id"])
        if params.correlations and len(params.variables) >= 2:
            specs.append(_spec(
                f"regex_challenge_{case['case_id']}", "auto_extracted", params, "regex"
            ))

    # --- synthetic stress cases -----------------------------------------------
    # merge_parameters() averages duplicate pairs independently across sources,
    # which does not preserve positive-definiteness. These document the failure
    # mode explicitly rather than hoping a natural source happens to hit it.
    specs.extend(_stress_specs())

    return specs, notes


def _challenge_case_to_parameters(case):
    from agentdataset.models.schemas import (
        CorrelationParams, MetaParams, Parameters, VariableParams,
    )

    variables = {}
    for stats in case.get("gold_variables") or []:
        name, mean, std = stats.get("name"), stats.get("mean"), stats.get("std")
        if not name or mean is None or std is None:
            continue
        variables[name] = VariableParams(
            name=name, mean=float(mean), std=float(std),
            min=float(mean) - 3 * float(std), max=float(mean) + 3 * float(std),
        )

    correlations = {}
    for entry in case.get("gold_correlations") or []:
        v1, v2 = entry.get("var1"), entry.get("var2")
        value = entry.get("correlation")
        if not v1 or not v2 or value is None:
            continue
        # Deliberately kept even when a variable is absent from gold_variables:
        # a correlation naming an unextracted variable is exactly the
        # incomplete-spec case the audit is meant to count.
        correlations[f"{v1}__{v2}"] = CorrelationParams(
            var1=v1, var2=v2, correlation=float(value),
            direction="positive" if float(value) >= 0 else "negative",
        )

    if not variables or not correlations:
        return None
    return Parameters(
        variables=variables, correlations=correlations,
        meta=MetaParams(source=case["case_id"], extracted_at="challenge",
                        extraction_method="gold_challenge"),
    )


def _stress_specs() -> list[dict]:
    from agentdataset.models.schemas import (
        CorrelationParams, MetaParams, Parameters, VariableParams,
    )

    def build(label, corrs, n_vars):
        names = [chr(ord("a") + i) for i in range(n_vars)]
        return _spec(
            label, "stress",
            Parameters(
                variables={
                    n: VariableParams(name=n, mean=0.0, std=1.0) for n in names
                },
                correlations={
                    f"{a}{b}": CorrelationParams(
                        var1=a, var2=b, correlation=r,
                        direction="positive" if r >= 0 else "negative",
                    )
                    for a, b, r in corrs
                },
                meta=MetaParams(source=label, extracted_at="stress",
                                extraction_method="synthetic"),
            ),
            "synthetic",
        )

    return [
        build("stress_valid", [("a", "b", 0.5), ("a", "c", 0.3), ("b", "c", 0.2)], 3),
        build("stress_slightly_invalid",
              [("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", 0.61)], 3),
        build("stress_severely_invalid",
              [("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", -0.9)], 3),
        build("stress_singular", [("a", "b", 1.0)], 2),
        build("stress_dense_invalid",
              [("a", "b", 0.8), ("a", "c", 0.8), ("a", "d", 0.8),
               ("b", "c", -0.5), ("b", "d", -0.5), ("c", "d", -0.5)], 4),
    ]


def audit_spec(spec: dict) -> tuple[dict, dict]:
    """Run the spec under both policies. Returns (diagnostics, scores)."""
    validator = Validator()
    scores: dict[str, float] = {}
    diag_record: dict = {}

    for mode in ("drop", "psd"):
        synthesizer = Synthesizer(n_rows=N_ROWS, seed=SEED, corr_repair=mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df = synthesizer.synthesize(spec["params"], source_label=spec["source_id"])
        diag = synthesizer.last_diagnostics
        report = validator.validate(df, spec["params"])
        scores[f"corr_score_{mode}"] = report.corr_score
        scores[f"overall_score_{mode}"] = report.overall_score
        if mode == "drop":
            diag_record = diag.as_row()
        else:
            diag_record["repair_action"] = diag.action
            diag_record["frobenius_delta"] = diag.frobenius_delta
            diag_record["min_eigenvalue_after"] = diag.min_eigenvalue_after

    scores["corr_score_delta_psd_minus_drop"] = (
        scores["corr_score_psd"] - scores["corr_score_drop"]
    )
    scores["overall_score_delta_psd_minus_drop"] = (
        scores["overall_score_psd"] - scores["overall_score_drop"]
    )
    return diag_record, scores


def main() -> None:
    specs, notes = collect_specs()
    print(f"{len(specs)} correlation-bearing specs collected")
    if notes:
        for note in notes:
            print(f"  note: {note}")

    provenance = collect_provenance(
        command="uv run python analysis/corr_validity_audit.py", seeds=[SEED]
    )

    raw_rows: list[dict] = []
    diag_rows: list[dict] = []

    for spec in specs:
        base = {
            "experiment_id": EXPERIMENT_ID,
            "experiment_type": "correlation_validity",
            "source_id": spec["source_id"],
            "dataset": spec["source_class"],
            "task_type": "synthesis",
            "condition": spec["source_class"],
            "model_family": "none",
            "synthesizer": "agentdataset",
            "seed": SEED,
            "split_seed": SEED,
            "synthesis_seed": SEED,
            "extraction_method": spec["extraction_method"],
            "provider": "none",
            "model": "none",
        }
        try:
            diag, scores = audit_spec(spec)
        except Exception as exc:  # noqa: BLE001 - recorded, never dropped
            raw_rows.append({
                **base, "run_id": f"{EXPERIMENT_ID}:{spec['source_id']}",
                "metric": "is_pd", "value": float("nan"), "status": "failed",
                "failure_stage": f"{type(exc).__name__}: {exc}"[:200],
            })
            continue

        diag_rows.append({
            "source_id": spec["source_id"],
            "source_class": spec["source_class"],
            "extraction_method": spec["extraction_method"],
            **diag,
            **scores,
        })

        declared = diag["n_pairs_declared"]
        metrics = {
            "is_pd": float(diag["is_pd"]),
            "dropped_all": float(diag["action"] == "dropped_all"),
            # A matrix can pass the PD check and still impose nothing, if every
            # declared pair named a variable that was never extracted. This is
            # the dominant way dependence is lost — see the regex arm.
            "vacuous_matrix": float(diag["n_pairs_applied"] == 0),
            "pairs_applied_rate": (
                float(diag["n_pairs_applied"]) / declared if declared else float("nan")
            ),
            "effective_copula": float(
                diag["n_pairs_applied"] > 0 and diag["action"] == "cholesky_ok"
            ),
            "min_eigenvalue": diag["min_eigenvalue"],
            "n_vars": float(diag["n_vars"]),
            "n_pairs_declared": float(diag["n_pairs_declared"]),
            "n_pairs_applied": float(diag["n_pairs_applied"]),
            "n_pairs_skipped_unknown_var": float(diag["n_pairs_skipped_unknown_var"]),
            **scores,
        }
        for metric, value in metrics.items():
            raw_rows.append({
                **base,
                "run_id": f"{EXPERIMENT_ID}:{spec['source_id']}:{metric}",
                "metric": metric,
                "value": float(value),
                "status": "ok",
                "failure_stage": "",
            })

    raw = pd.DataFrame(raw_rows)
    for column in RAW_RESULT_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""
    validate_raw_results(raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RESULTS_DIR / f"{EXPERIMENT_ID}_raw.csv", index=False)

    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(RESULTS_DIR / f"{EXPERIMENT_ID}_per_matrix.csv", index=False)

    summary = summarize_long_results(raw, ["dataset", "condition"])
    summary.to_csv(RESULTS_DIR / f"{EXPERIMENT_ID}_summary.csv", index=False)

    # Headline the paper needs: the share of matrices that take the copula path.
    by_class = (
        diagnostics.groupby("source_class")
        .agg(
            n=("source_id", "size"),
            pd_rate=("is_pd", "mean"),
            dropped_rate=("action", lambda s: float((s == "dropped_all").mean())),
            vacuous_rate=("n_pairs_applied", lambda s: float((s == 0).mean())),
            pairs_applied_rate=("n_pairs_applied", "sum"),
            pairs_declared=("n_pairs_declared", "sum"),
            mean_min_eigenvalue=("min_eigenvalue", "mean"),
        )
        .reset_index()
    )
    by_class["effective_copula_rate"] = (
        by_class["pairs_applied_rate"] / by_class["pairs_declared"]
    )
    print(by_class.to_string(index=False))

    write_json(
        RESULTS_DIR / f"{EXPERIMENT_ID}_run_manifest.json",
        {
            **provenance,
            "n_specs": len(specs),
            "n_rows_per_dataset": N_ROWS,
            "source_classes": by_class.to_dict(orient="records"),
            "overall_pd_rate": float(diagnostics["is_pd"].mean()),
            "collection_notes": notes,
        },
    )
    print(f"Wrote {len(raw)} raw rows and {len(diagnostics)} matrices to {RESULTS_DIR}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
