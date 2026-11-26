"""Generate and verify the result-backed LaTeX table rows used by the paper.

The committed CSV/JSON artifacts are the source of truth. The write mode updates
the TeX fragments that are inlined into docs/submission/paper.tex (the AAAI 2027
AI4Research submission); check mode fails if a fragment is stale or a result
contract no longer holds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = ROOT / "analysis"
GENERATED = ROOT / "docs" / "submission" / "generated_tables"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required paper artifact is missing: {path}")
    return pd.read_csv(path)


def _one(frame: pd.DataFrame, **filters) -> pd.Series:
    rows = frame
    for column, value in filters.items():
        rows = rows.loc[rows[column].eq(value)]
    if len(rows) != 1:
        raise AssertionError(f"expected one row for {filters}, found {len(rows)}")
    return rows.iloc[0]


def _omit_zero(text: str) -> str:
    if text.startswith("-0."):
        return "-." + text[3:]
    if text.startswith("0."):
        return "." + text[2:]
    return text


def _f(value: float, digits: int = 3, *, omit_zero: bool = True) -> str:
    text = f"{float(value):.{digits}f}"
    return _omit_zero(text) if omit_zero else text


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise AssertionError(f"no numeric values for {column}")
    return float(values.mean())


def _real_paper_rows() -> str:
    discovery = _read_csv(ANALYSIS / "real_paper_discovery_summary.csv")
    extraction = _read_csv(ANALYSIS / "real_paper_extraction.csv")
    pipeline = _read_csv(RESULTS / "real_paper_pipeline_summary.csv")
    labels = {
        "cdc_nhanes_rural_urban": "CDC/NHANES",
        "calerie_rct": "CALERIE RCT",
        "student_mental_health": "Student health",
    }
    lines = []
    for source_id, label in labels.items():
        rank = int(_one(discovery, source_id=source_id)["first_relevant_rank"])
        llm = _one(extraction, source_id=source_id, condition="llm")
        regex_rows = extraction.loc[
            extraction["source_id"].eq(source_id) & extraction["condition"].eq("regex")
        ]
        if llm["parse_status"] != "ok":
            raise AssertionError(f"paper assumes successful parsing for {source_id}")
        regex_f1 = "---" if regex_rows.empty else _f(regex_rows.iloc[0]["f1"])
        gold_fidelity = _one(
            pipeline, source_id=source_id, condition="gold", metric="fidelity_score"
        )["mean"]
        llm_fidelity = _one(
            pipeline, source_id=source_id, condition="llm", metric="fidelity_score"
        )["mean"]
        lines.append(
            f"{label} & {rank} & OK & {_f(llm['f1'])} & {regex_f1} & "
            f"{float(gold_fidelity):.2f} / {float(llm_fidelity):.2f} \\\\"
        )
    return "\n".join(lines) + "\n"


def _utility_rows() -> str:
    frame = _read_csv(RESULTS / "p0_utility_summary.csv")
    if set(frame["n"].dropna().astype(int)) != {30}:
        raise AssertionError("utility table requires 30 seeds in every summary cell")
    datasets = [
        ("adult_income", "Adult", "classification"),
        ("pima_diabetes", "Pima", "classification"),
        ("german_credit", "German", "classification"),
        ("concrete_compressive_strength", "Concrete", "regression"),
    ]
    models = [
        ("logistic_regression", "linear_regression", "LR"),
        ("hist_gradient_boosting", "hist_gradient_boosting", "HGB"),
    ]
    synths = [
        "real", "agentdataset", "copula_no_loop", "independent",
        "sdv_gaussian_copula",
    ]
    lines = []
    for dataset, label, task in datasets:
        for index, (class_model, reg_model, short_model) in enumerate(models):
            model = class_model if task == "classification" else reg_model
            cells = []
            for synth in synths:
                condition = "trtr" if synth == "real" else "tstr"
                if task == "classification":
                    a = _one(
                        frame, dataset=dataset, condition=condition,
                        model_family=model, synthesizer=synth,
                        metric="minority_f1",
                    )["mean"]
                    b = _one(
                        frame, dataset=dataset, condition=condition,
                        model_family=model, synthesizer=synth, metric="roc_auc",
                    )["mean"]
                    cells.append(f"{_f(a)}/{_f(b)}")
                else:
                    a = _one(
                        frame, dataset=dataset, condition=condition,
                        model_family=model, synthesizer=synth, metric="r2",
                    )["mean"]
                    b = _one(
                        frame, dataset=dataset, condition=condition,
                        model_family=model, synthesizer=synth, metric="rmse",
                    )["mean"]
                    cells.append(f"{_f(a)}/{float(b):.2f}")
            row_label = label if index == 0 else ""
            lines.append(f"{row_label} & {short_model} & " + " & ".join(cells) + " \\\\")
        if dataset == "german_credit":
            lines.append("\\midrule")
    return "\n".join(lines) + "\n"


def _noise_rows() -> str:
    frame = _read_csv(RESULTS / "ablation_noise_pivot.csv")
    means, stds = [], []
    for value in [0, 1, 2, 3, 5, 10]:
        scores = frame.loc[frame["iterations"].eq(value), "score"]
        if len(scores) != 40:
            raise AssertionError(f"expected 40 rows for iterations={value}")
        means.append(f"{scores.mean():.2f}")
        stds.append(f"{scores.std(ddof=1):.2f}")
    return (
        "Mean fidelity & " + " & ".join(means) + " \\\\\n"
        + "Std & " + " & ".join(stds) + " \\\\\n"
    )


def _provider_cost_rows() -> str:
    frame = _read_csv(RESULTS / "extraction_cost_latency.csv")
    lines = []
    for provider in ["Google", "OpenAI", "Anthropic"]:
        rows = frame.loc[frame["provider"].eq(provider)]
        if len(rows) != 5:
            raise AssertionError(f"expected five cost repetitions for {provider}")
        lines.append(
            f"{provider} & {_mean(rows, 'latency_s'):.2f} & "
            f"{_mean(rows, 'cost_usd'):.4f} & "
            f"{_mean(rows, 'prompt_tokens'):.0f} & "
            f"{_mean(rows, 'completion_tokens'):.0f} \\\\"
        )
    return "\n".join(lines) + "\n"


def _challenge_rows() -> str:
    frame = _read_csv(RESULTS / "extraction_challenge_raw_run1.csv")
    if len(frame) != 930 or frame["api_error"].astype(bool).any():
        raise AssertionError("primary extraction table requires complete run 1")
    arms = ["caveman", "natural_language", "regex"]
    specs = [
        ("$F_1$", "f1"), ("Precision", "precision"), ("Recall", "recall"),
        ("Hallucination rate", "hallucination_rate"),
        ("Context-error rate", "context_error_rate"),
    ]
    lines = []
    for label, metric in specs:
        values = [_mean(frame.loc[frame["arm"].eq(arm)], metric) for arm in arms]
        lines.append(label + " & " + " & ".join(_f(v) for v in values) + " \\\\")
    abstention = []
    for arm in arms:
        rows = frame.loc[frame["arm"].eq(arm) & frame["n_gold_groups"].eq(0)]
        abstention.append(float(rows["correct_abstention"].astype(bool).mean()))
    lines.append(
        "Correct abstention & " + " & ".join(_f(v) for v in abstention) + " \\\\"
    )
    lines.append("\\midrule")
    prompt = [_mean(frame.loc[frame["arm"].eq(arm)], "prompt_tokens") for arm in arms]
    cost = [_mean(frame.loc[frame["arm"].eq(arm)], "cost_usd") for arm in arms]
    lines.append("Prompt tokens & " + " & ".join(f"{v:.0f}" for v in prompt) + " \\\\")
    lines.append(
        "Cost (\\$/call) & "
        + " & ".join(_f(v, 4) if v else "0" for v in cost)
        + " \\\\"
    )
    return "\n".join(lines) + "\n"


def _validity_rows() -> str:
    frame = _read_csv(RESULTS / "fidelity_validity_probe_summary.csv")
    specs = [
        ("None", "shift_mean", 0.0),
        ("Mean $+1$\\,SD", "shift_mean", 1.0),
        ("Mean $+2$\\,SD", "shift_mean", 2.0),
        ("Statistics swapped", "swap_stats", 1.0),
    ]
    lines = []
    for label, corruption, epsilon in specs:
        rows = frame.loc[
            frame["corruption"].eq(corruption) & frame["epsilon"].eq(epsilon)
        ]
        values = [
            _mean(rows.loc[rows["metric"].eq(metric)], "mean")
            for metric in [
                "reported_fidelity", "external_ks_vs_real",
                "external_mean_error_sd",
            ]
        ]
        lines.append(
            f"{label} & {values[0]:.1f} & {values[1]:.2f} & {values[2]:.2f} \\\\"
        )
    return "\n".join(lines) + "\n"


def _pivot_rows() -> str:
    frame = _read_csv(RESULTS / "noise_pivot_budget.csv")
    specs = [
        ("Noise pivot (shipped)", "iterations", "iterations=5"),
        ("Fixed noise $0.01$", "baseline_fixed", "fixed_noise=0.01,iterations=5"),
        ("Fixed noise $0.05$", "baseline_fixed", "fixed_noise=0.05,iterations=5"),
        ("Fixed noise $0.10$", "baseline_fixed", "fixed_noise=0.1,iterations=5"),
        ("Random search", "baseline_random", "random_search,iterations=5"),
    ]
    lines = []
    for label, family, condition in specs:
        row = _one(frame, family=family, condition=condition, metric="best_score")
        if int(row["count"]) != 220:
            raise AssertionError(f"pivot table requires n=220 for {condition}")
        lines.append(f"{label} & {float(row['mean']):.2f} & {float(row['std']):.2f} \\\\")
    return "\n".join(lines) + "\n"


def _corr_rows() -> str:
    frame = _read_csv(RESULTS / "corr_validity_per_matrix.csv")
    initial = {
        "templated": (11, 20), "gold": (3, 4),
        "auto_extracted": (0, 8), "stress": (16, 16),
    }
    labels = {
        "templated": "Templated", "gold": "Gold",
        "auto_extracted": "Auto-extracted", "stress": "Constructed invalid",
    }
    lines = []
    for source_class in ["templated", "gold", "auto_extracted", "stress"]:
        rows = frame.loc[frame["source_class"].eq(source_class)]
        before_applied, before_declared = initial[source_class]
        lines.append(
            f"{labels[source_class]} & {len(rows)} & "
            f"{rows['is_pd'].astype(bool).mean():.2f} & "
            f"{before_applied}/{before_declared} & "
            f"{int(rows['n_pairs_applied'].sum())}/"
            f"{int(rows['n_pairs_declared'].sum())} \\\\"
        )
    return "\n".join(lines) + "\n"


def _calibration_rows() -> str:
    frame = _read_csv(RESULTS / "adult_calibration.csv")
    if set(frame["n"].dropna().astype(int)) != {30}:
        raise AssertionError("calibration table requires 30 seeds")
    specs = [
        ("Uncalibrated ($0.5$)", "uncalibrated_full_fit"),
        ("Platt ($0.5$)", "platt"),
        ("Isotonic ($0.5$)", "isotonic"),
        ("Threshold on synth.\\ val.", "threshold_synth_val"),
        ("Oracle threshold$^\\ddagger$", "oracle_threshold_real_test"),
    ]
    synths = [
        ("real", "trtr"), ("agentdataset", "tstr"), ("independent", "tstr"),
        ("sdv_gaussian_copula", "tstr"),
    ]
    lines = []
    for index, (label, strategy) in enumerate(specs):
        if index == 4:
            lines.append("\\midrule")
        cells = []
        for synth, condition in synths:
            values = []
            for model in ["logistic_regression", "hist_gradient_boosting"]:
                value = _one(
                    frame, condition=condition, synthesizer=synth,
                    model_family=model, calibration_strategy=strategy,
                    metric="minority_f1",
                )["mean"]
                values.append(_f(value))
            cells.append("/".join(values))
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def _e2e_rows() -> str:
    frame = _read_csv(RESULTS / "end_to_end_cost_latency.csv")
    rows = frame.loc[frame["source_id"].eq("mean_one_dataset")]
    specs = [
        ("Query expansion", "query_expansion"), ("Web search", "web_search"),
        ("Source suggestion", "source_suggestion"), ("Fetch", "fetch"),
        ("Parse", "parse"), ("Extraction", "extraction"),
        ("Synthesis", "synthesis"), ("Validation", "validation"),
        ("Refinement overhead", "refinement_overhead"),
        ("Artifact writing", "artifact_write"),
    ]
    lines = []
    for label, stage in specs:
        row = _one(rows, stage=stage)
        values = []
        for column, digits in [
            ("api_cost_usd", 5), ("api_seconds", 2),
            ("network_seconds", 2), ("local_seconds", 2),
        ]:
            value = float(row[column])
            if value == 0:
                values.append("---")
            elif column == "local_seconds" and value < 0.005:
                values.append("$<$.01")
            else:
                values.append(_f(value, digits))
        lines.append(label + " & " + " & ".join(values) + " \\\\")
    total = _one(rows, stage="total")
    lines.append("\\midrule")
    values = [
        f"\\textbf{{{_f(float(total[column]), digits)}}}"
        for column, digits in [
            ("api_cost_usd", 5), ("api_seconds", 2),
            ("network_seconds", 2), ("local_seconds", 2),
        ]
    ]
    lines.append("\\textbf{One dataset} & " + " & ".join(values) + " \\\\")
    return "\n".join(lines) + "\n"


def _review_gate_rows() -> str:
    frame = _read_csv(RESULTS / "review_gate_study_summary.csv")
    labels = {
        "cdc_nhanes_rural_urban": "CDC/NHANES",
        "calerie_rct": "CALERIE RCT",
        "student_mental_health": "Student health",
    }
    conditions = {
        "gold": "Gold",
        "unreviewed": "No review gate",
        "reviewed": "Agent reviewed",
    }
    lines = []
    for source_id, label in labels.items():
        for condition, condition_label in conditions.items():
            row = _one(frame, source_id=source_id, condition=condition)
            lines.append(
                f"{label if condition == 'gold' else ''} & {condition_label} & "
                f"{_mean(frame.loc[frame['source_id'].eq(source_id) & frame['condition'].eq(condition)], 'n_variables_mean'):.1f} & "
                f"{float(row['n_pairs_applied_mean']):.1f} & "
                f"{float(row['fidelity_score_mean']):.1f} & "
                f"{float(row['correlation_score_mean']):.1f} \\\\"
            )
        lines.append("\\midrule")
    return "\n".join(lines[:-1]) + "\n"


def render_tables() -> dict[str, str]:
    bodies = {
        "realpapers.tex": _real_paper_rows(),
        "utility.tex": _utility_rows(),
        "noise.tex": _noise_rows(),
        "provider_cost.tex": _provider_cost_rows(),
        "extraction.tex": _challenge_rows(),
        "validity.tex": _validity_rows(),
        "pivot.tex": _pivot_rows(),
        "correlation.tex": _corr_rows(),
        "calibration.tex": _calibration_rows(),
        "end_to_end_cost.tex": _e2e_rows(),
        "review_gate.tex": _review_gate_rows(),
    }
    # Booktabs rules must be expanded immediately after the final row; placing
    # \bottomrule after an \input boundary makes TeX treat it as misplaced.
    return {name: body + "\\bottomrule\n" for name, body in bodies.items()}


def validate_manifests() -> None:
    manifests = [
        RESULTS / "real_paper_run_manifest.json",
        RESULTS / "p0_utility_run_manifest.json",
        RESULTS / "adult_calibration_run_manifest.json",
        RESULTS / "extraction_challenge_run_manifest_run1.json",
        RESULTS / "noise_pivot_ablation_run_manifest.json",
        RESULTS / "corr_validity_run_manifest.json",
        RESULTS / "fidelity_validity_probe_run_manifest.json",
        RESULTS / "end_to_end_cost_latency_run_manifest.json",
    ]
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance", payload)
        for key in ["command", "git_commit"]:
            if not provenance.get(key):
                raise AssertionError(f"{path.name} lacks provenance field {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    validate_manifests()
    rendered = render_tables()
    if args.write:
        GENERATED.mkdir(parents=True, exist_ok=True)
        for name, content in rendered.items():
            (GENERATED / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {len(rendered)} paper table fragments")
        return 0

    stale = []
    for name, expected in rendered.items():
        path = GENERATED / name
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            stale.append(name)
    if stale:
        raise SystemExit(
            "stale or missing paper table fragments: " + ", ".join(stale)
        )
    print(f"verified {len(rendered)} paper table fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
