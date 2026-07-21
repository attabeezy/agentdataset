import os
import time
import json
import logging
import numpy as np
import pandas as pd
import litellm
from litellm import completion
from scipy import stats as scipy_stats
from pathlib import Path
from typing import Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.extractor import Extractor, CAVEMAN_PROMPT
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.core.validator import Validator
from agentdataset.models.schemas import Parameters, MetaParams, VariableParams, CorrelationParams
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark")

def load_env():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("'\"")

# All LLM-backed benchmarking (empirical benchmark, ablation, cost/latency)
# runs through OpenRouter (one key covers all vendors) rather than per-provider
# keys, per the user's instruction.
_OPENROUTER_MODEL = "openrouter/openai/gpt-4o"
_OPENROUTER_ENV_VAR = "OPENROUTER_API_KEY"

# Where benchmark outputs are persisted so runs produce reusable artifacts
# instead of only console output.
RESULTS_DIR = Path("results")


def _save_results(df: pd.DataFrame, filename: str) -> None:
    """Best-effort CSV write; a save failure shouldn't abort a benchmark run."""
    if df is None or df.empty:
        return
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / filename
        df.to_csv(path, index=False)
        print(f"Saved results to {path}")
    except OSError as e:
        logger.warning("Failed to save results to %s: %s", filename, e)


def _load_adult() -> dict:
    """UCI Adult Income (id=2): demographic/income classification."""
    ds = fetch_ucirepo(id=2)
    df = ds.data.features.join(ds.data.targets)
    df = df.rename(columns={"education-num": "education_num", "marital-status": "marital_status"})
    df["income"] = df["income"].str.rstrip(".")  # train/test splits differ ("<=50K" vs "<=50K.")
    # Collapse the 7 raw marital-status labels into 3 categories so the
    # benchmark exercises a real multi-category (3+) feature.
    df["marital_status"] = df["marital_status"].map({
        "Married-civ-spouse": "married",
        "Married-spouse-absent": "married",
        "Married-AF-spouse": "married",
        "Never-married": "never_married",
        "Divorced": "prev_married",
        "Separated": "prev_married",
        "Widowed": "prev_married",
    })
    df = df[["age", "education_num", "sex", "marital_status", "income"]].dropna()
    return {
        "name": "adult_income",
        "domain": "demographic",
        "df": df,
        "continuous": ["age", "education_num"],
        "categorical_features": ["sex", "marital_status"],
        "target": "income",
    }


def _load_pima() -> dict:
    """Pima Indians Diabetes (not on ucimlrepo; classic CSV mirror): medical classification."""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
    ]
    df = pd.read_csv(url, names=cols)
    df["outcome"] = df["outcome"].map({0: "no_diabetes", 1: "diabetes"})
    df = df[["glucose", "bmi", "outcome"]].dropna()
    return {
        "name": "pima_diabetes",
        "domain": "medical",
        "df": df,
        "continuous": ["glucose", "bmi"],
        "categorical_features": [],
        "target": "outcome",
    }


def _load_german_credit() -> dict:
    """Statlog German Credit (id=144): financial credit-risk classification."""
    ds = fetch_ucirepo(id=144)
    df = ds.data.features.join(ds.data.targets)
    df = df.rename(columns={
        "Attribute2": "duration",
        "Attribute5": "credit_amount",
        "Attribute20": "foreign_worker",
        "class": "credit_risk",
    })
    df["foreign_worker"] = df["foreign_worker"].map({"A201": "yes", "A202": "no"})
    df["credit_risk"] = df["credit_risk"].map({1: "good", 2: "bad"})
    df = df[["duration", "credit_amount", "foreign_worker", "credit_risk"]].dropna()
    return {
        "name": "german_credit",
        "domain": "financial",
        "df": df,
        "continuous": ["duration", "credit_amount"],
        "categorical_features": ["foreign_worker"],
        "target": "credit_risk",
    }


def _dataset_to_source_text(dataset: dict) -> str:
    """Build a "literature style" description from the REAL dataset's own
    statistics (not fabricated), phrased so both the LLM and the regex
    fallback can extract it. Categorical variables may have any number of
    categories — see extractor._PATTERN_CATEGORICAL."""
    df = dataset["df"]
    continuous = dataset["continuous"]
    categorical = dataset["categorical_features"] + [dataset["target"]]
    lines = []

    for col in continuous:
        mean, std = float(df[col].mean()), float(df[col].std())
        lines.append(f"The variable {col} has mean {mean:.4f} and std {std:.4f}.")

    # Canonical encoding: labels sorted alphabetically -> codes 0..N-1. Used
    # consistently for both the stated probabilities and the correlation
    # numbers below, so extraction round-trips correctly regardless of which
    # order the LLM/regex happens to preserve.
    label_order = {}
    for col in categorical:
        labels = sorted(df[col].dropna().unique().tolist())
        label_order[col] = labels
        probs = df[col].value_counts(normalize=True)
        parts = [f"'{label}' with probability {float(probs[label]):.4f}" for label in labels]
        # "'a' with probability p and 'b' with probability q" for 2 labels,
        # "'a' with probability p, 'b' with probability q, and 'c' with
        # probability r" for 3+ — both forms match the extractor regex.
        listed = " and ".join(parts) if len(parts) == 2 else ", ".join(parts[:-1]) + f", and {parts[-1]}"
        lines.append(f"The categorical variable {col} takes value {listed}.")

    numeric_df = df.copy()
    for col, labels in label_order.items():
        numeric_df[col] = numeric_df[col].map({label: i for i, label in enumerate(labels)})

    pairs = [(c, dataset["target"]) for c in continuous if dataset["target"] != c]
    pairs += [(c, dataset["target"]) for c in dataset["categorical_features"]]
    pairs += [(continuous[i], continuous[j]) for i in range(len(continuous)) for j in range(i + 1, len(continuous))]
    for v1, v2 in pairs:
        corr_val = float(numeric_df[v1].corr(numeric_df[v2]))
        lines.append(f"The correlation between {v1} and {v2} is {corr_val:.4f}.")

    return "\n".join(lines)


# SDV is an optional, heavy dependency that may fail to install/import on this
# Python 3.13 environment. Guard the import so the whole benchmark can still run
# with the other two synthesizers when SDV is unavailable.
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    _SDV_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on environment
    logger.warning("SDV is unavailable (%s); its baseline will be skipped.", e)
    _SDV_AVAILABLE = False


def synthesize_independent_marginals(params: Parameters, n_rows: int, seed: int) -> pd.DataFrame:
    """Independent-marginals baseline: draw EACH variable independently from its
    own extracted marginal, completely ignoring correlations.

    Delegates to the same Synthesizer machinery so per-variable marginal handling
    (normal/uniform/gamma/categorical) is byte-for-byte identical to AgentDataset;
    the ONLY difference is that the copula/correlation structure is dropped. We
    deep-copy params (pydantic model) and clear .correlations so the shared params
    object is never mutated. With no correlations, Synthesizer skips the Cholesky
    step and each column keeps its rank-transformed marginal but is uncorrelated.
    """
    params_no_corr = params.model_copy(deep=True)
    params_no_corr.correlations = {}
    return Synthesizer(n_rows=n_rows, seed=seed).synthesize(params_no_corr)


def synthesize_sdv_gaussian_copula(
    train_df: pd.DataFrame, feature_cols: List[str], target: str, n_rows: int, seed: int
) -> Optional[pd.DataFrame]:
    """SDV GaussianCopula baseline: fit SDV's GaussianCopulaSynthesizer on the REAL
    training data (SDV learns from data, not from extracted params) restricted to
    feature_cols + [target], and sample n_rows rows.

    Returns None (with a logged warning) if SDV is unavailable or if fitting/
    sampling raises, so the benchmark still completes with the other synthesizers.
    """
    if not _SDV_AVAILABLE:
        return None
    try:
        cols = feature_cols + [target]
        real = train_df[cols].copy()
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real)
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(real)
        # SDV has no direct seed argument on sample(); seed numpy for parity.
        np.random.seed(seed)
        return synthesizer.sample(num_rows=n_rows)
    except Exception as e:
        logger.warning("SDV GaussianCopula synthesis failed (seed=%d): %s", seed, e)
        return None


def run_empirical_benchmark(dataset_loaders: Optional[List] = None, n_seeds: int = 5) -> pd.DataFrame:
    print("\n" + "="*50)
    print("1. EMPIRICAL BENCHMARK (Downstream ML Performance on real datasets)")
    print("="*50)

    if dataset_loaders is None:
        dataset_loaders = [_load_adult, _load_pima, _load_german_credit]

    rows = []
    for loader in dataset_loaders:
        try:
            # Loading (network fetch) happens inside the try block too, so a
            # transient failure for one dataset doesn't abort the others.
            dataset = loader()
            name = dataset["name"]
            target = dataset["target"]
            feature_cols = dataset["continuous"] + dataset["categorical_features"]
            df = dataset["df"]

            print(f"\n--- {name} ({dataset['domain']}) ---")
            text = _dataset_to_source_text(dataset)

            # Extraction is an expensive LLM call and is NOT seed-dependent, so
            # it runs ONCE per dataset, hoisted above the seed loop. The
            # orchestrator built here is only used for extraction; a fresh one
            # is created per seed below so the optimization ratchet starts clean.
            extract_orchestrator = Orchestrator(
                session_id=f"real_benchmark_{name}",
                model=_OPENROUTER_MODEL,
                env_var=_OPENROUTER_ENV_VAR,
            )
            params = extract_orchestrator.extractor.extract_parameters(text, name)
            if target not in params.variables:
                logger.warning("Extraction did not recover target '%s' for %s; skipping.", target, name)
                continue

            # TRTR is the real-data baseline and is identical across all three
            # synthesizers, so it is collected once per dataset (one value/seed).
            trtr_metrics = {"trtr_accuracy": [], "trtr_f1": [], "trtr_roc_auc": []}

            # Per-synthesizer TSTR metric collectors; each list gets (at most) one
            # value per seed. fidelity_score only applies to AgentDataset (it
            # validates against the extracted params, which the baselines don't use).
            _SYNTHESIZERS = ["agentdataset", "independent", "sdv_gaussian_copula"]
            seed_metrics = {
                synth: {
                    "tstr_accuracy": [], "tstr_f1": [], "tstr_roc_auc": [],
                    "fidelity_score": [],
                }
                for synth in _SYNTHESIZERS
            }

            for seed in range(n_seeds):
                # Seed threads into the split so each seed sees a different
                # train/test partition ...
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=seed, stratify=df[target]
                )

                # ... and into synthesis. The Synthesizer draws all randomness
                # from self.rng (np.random.default_rng(seed), set in its
                # __init__ — see agentdataset/core/synthesizer.py:13-16); it does
                # NOT re-seed per synthesize() call. So constructing a fresh
                # Synthesizer(seed=seed) per iteration is what makes each seed
                # produce a different synthetic dataset. np.random.seed(seed) is
                # also set for parity with run_ablation_noise_pivot.
                np.random.seed(seed)
                orchestrator = Orchestrator(
                    session_id=f"real_benchmark_{name}_{seed}",
                    model=_OPENROUTER_MODEL,
                    env_var=_OPENROUTER_ENV_VAR,
                )
                orchestrator.synthesizer = Synthesizer(n_rows=len(train_df), seed=seed)

                best_score, df_synth = orchestrator.run_optimization_loop(params, iterations=3)
                if df_synth is None or any(c not in df_synth.columns for c in feature_cols + [target]):
                    logger.warning("Synthesis is missing required columns for %s (seed=%d); skipping seed.", name, seed)
                    continue

                fidelity_report = orchestrator.validator.validate(df_synth, params)

                # Encoders are fit on this seed's train split so TRTR and every
                # synthesizer's TSTR share the same encoding and test set — a fair
                # comparison across all three synthesizers within the seed.
                target_encoder = LabelEncoder().fit(train_df[target])
                feature_encoders = {
                    col: LabelEncoder().fit(train_df[col]) for col in dataset["categorical_features"]
                }

                def _prepare_X_y(frame: pd.DataFrame):
                    X = frame[feature_cols].copy()
                    for col, enc in feature_encoders.items():
                        X[col] = enc.transform(X[col])
                    y = target_encoder.transform(frame[target])
                    return X, y

                X_test_real, y_test_real = _prepare_X_y(test_df)
                X_train_real, y_train_real = _prepare_X_y(train_df)

                def _fit_and_score(X_train, y_train) -> dict:
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test_real)
                    probs = model.predict_proba(X_test_real)[:, 1]
                    return {
                        "accuracy": accuracy_score(y_test_real, preds),
                        "f1": f1_score(y_test_real, preds),
                        "roc_auc": roc_auc_score(y_test_real, probs),
                    }

                # TRTR: real-data baseline, shared across all synthesizers.
                trtr = _fit_and_score(X_train_real, y_train_real)
                trtr_metrics["trtr_accuracy"].append(trtr["accuracy"])
                trtr_metrics["trtr_f1"].append(trtr["f1"])
                trtr_metrics["trtr_roc_auc"].append(trtr["roc_auc"])

                # Build each synthesizer's synthetic training frame. Baselines
                # produce the same feature_cols + [target] columns so _prepare_X_y
                # works. AgentDataset's df_synth was validated above; fidelity only
                # applies to it (baselines aren't measured against extracted params).
                synth_frames = {
                    "agentdataset": (df_synth, fidelity_report.overall_score),
                    "independent": (
                        synthesize_independent_marginals(params, len(train_df), seed),
                        None,
                    ),
                    "sdv_gaussian_copula": (
                        synthesize_sdv_gaussian_copula(
                            train_df, feature_cols, target, len(train_df), seed
                        ),
                        None,
                    ),
                }

                for synth_name, (frame, fidelity) in synth_frames.items():
                    if frame is None:
                        # SDV unavailable or failed; skip only this synthesizer.
                        continue
                    if any(c not in frame.columns for c in feature_cols + [target]):
                        logger.warning(
                            "%s synthetic frame missing required columns for %s (seed=%d); skipping.",
                            synth_name, name, seed,
                        )
                        continue
                    try:
                        # Defensive: baseline categoricals may contain labels the
                        # per-seed encoder never saw; skip that synthesizer/seed.
                        X_train_synth, y_train_synth = _prepare_X_y(frame)
                    except ValueError as ve:
                        logger.warning(
                            "%s produced unseen labels for %s (seed=%d): %s; skipping.",
                            synth_name, name, seed, ve,
                        )
                        continue
                    tstr = _fit_and_score(X_train_synth, y_train_synth)
                    seed_metrics[synth_name]["tstr_accuracy"].append(tstr["accuracy"])
                    seed_metrics[synth_name]["tstr_f1"].append(tstr["f1"])
                    seed_metrics[synth_name]["tstr_roc_auc"].append(tstr["roc_auc"])
                    if fidelity is not None:
                        seed_metrics[synth_name]["fidelity_score"].append(fidelity)

            if not trtr_metrics["trtr_accuracy"]:
                logger.warning("No seed produced usable results for %s; skipping.", name)
                continue

            def _agg(metrics: dict) -> dict:
                # Aggregate mean/std across seeds (ddof=1 to match prior behavior).
                out = {}
                for metric, vals in metrics.items():
                    out[f"{metric}_mean"] = np.mean(vals) if vals else float("nan")
                    out[f"{metric}_std"] = (
                        np.std(vals, ddof=1) if len(vals) >= 2 else 0.0
                    )
                return out

            # TRTR aggregation is shared and included on every synthesizer row.
            trtr_agg = _agg(trtr_metrics)

            print(
                f"TRTR: acc={trtr_agg['trtr_accuracy_mean']:.3f}±{trtr_agg['trtr_accuracy_std']:.3f} "
                f"f1={trtr_agg['trtr_f1_mean']:.3f}±{trtr_agg['trtr_f1_std']:.3f} "
                f"auc={trtr_agg['trtr_roc_auc_mean']:.3f}±{trtr_agg['trtr_roc_auc_std']:.3f}"
            )

            # One output row per (dataset, synthesizer).
            for synth_name in _SYNTHESIZERS:
                tstr_vals = seed_metrics[synth_name]
                if not tstr_vals["tstr_accuracy"]:
                    logger.warning(
                        "Synthesizer %s produced no usable results for %s; skipping its row.",
                        synth_name, name,
                    )
                    continue
                synth_agg = _agg(tstr_vals)
                print(
                    f"  [{synth_name}] TSTR: "
                    f"acc={synth_agg['tstr_accuracy_mean']:.3f}±{synth_agg['tstr_accuracy_std']:.3f} "
                    f"f1={synth_agg['tstr_f1_mean']:.3f}±{synth_agg['tstr_f1_std']:.3f} "
                    f"auc={synth_agg['tstr_roc_auc_mean']:.3f}±{synth_agg['tstr_roc_auc_std']:.3f} | "
                    f"fidelity={synth_agg['fidelity_score_mean']:.1f}±{synth_agg['fidelity_score_std']:.1f} | "
                    f"n_seeds={len(tstr_vals['tstr_accuracy'])} | extraction={params.meta.extraction_method}"
                )
                rows.append({
                    "dataset": name,
                    "domain": dataset["domain"],
                    "synthesizer": synth_name,
                    "extraction_method": params.meta.extraction_method,
                    "n_seeds": len(tstr_vals["tstr_accuracy"]),
                    **trtr_agg,
                    **synth_agg,
                })
        except Exception:
            logger.exception("Real-dataset benchmark failed for loader %s", loader.__name__)

    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        print("\n" + df_results.to_string(index=False))
    else:
        print("No dataset produced usable results.")
    _save_results(df_results, "empirical_benchmark.csv")
    return df_results


# Ground-truth-embedded "literature" texts spanning several domains, used for both
# ablation studies. Regex/LLM extraction is compared against the known mean/std/
# correlation values that were used to write each text, instead of against a
# single hand-picked snippet.
_ABLATION_DOMAINS = [
    ("heart_rate", "blood_pressure", 72.0, 8.0, 120.0, 15.0, 0.35),
    ("cholesterol", "bmi", 190.0, 35.0, 27.0, 5.0, 0.42),
    ("temperature", "humidity", 22.0, 4.0, 55.0, 12.0, -0.30),
    ("stock_price", "trading_volume", 150.0, 40.0, 2000.0, 500.0, 0.50),
    ("credit_score", "loan_amount", 680.0, 60.0, 15000.0, 5000.0, 0.60),
    ("study_hours", "exam_score", 5.0, 2.0, 75.0, 10.0, 0.55),
    ("commute_time", "job_satisfaction", 35.0, 12.0, 6.5, 1.5, -0.40),
    ("rainfall", "crop_yield", 800.0, 150.0, 3.2, 0.8, 0.48),
]


def _build_ablation_texts():
    """Build source texts with known ground-truth statistics for ablation scoring."""
    texts = []
    for i, (v1, v2, m1, s1, m2, s2, corr) in enumerate(_ABLATION_DOMAINS):
        label1, label2 = v1.replace("_", " "), v2.replace("_", " ")
        text = f"""
        We observed a {label1} distribution with mean {m1} and std {s1}.
        {label2.capitalize()} was normally distributed with mean {m2} and std {s2}.
        The correlation between {v1} and {v2} is {corr}.
        """
        texts.append(
            {
                "id": f"domain_{i}_{v1}",
                "text": text,
                "ground_truth": {
                    v1: {"mean": m1, "std": s1},
                    v2: {"mean": m2, "std": s2},
                },
                "corr_pair": (v1, v2),
                "corr_val": corr,
            }
        )
    return texts


def _score_extraction(extracted_vars: dict, ground_truth: dict) -> dict:
    """Match extracted variables to ground truth by nearest mean and score relative error.

    LLM output uses real variable names; regex output uses generic "var_N" names,
    so matching by name is not possible in general — nearest-mean matching works
    for both without requiring the extractor to name variables correctly.
    """
    if not extracted_vars:
        return {"success_rate": 0.0, "mean_rel_err": None, "std_rel_err": None}

    gt_items = list(ground_truth.items())
    ext_items = list(extracted_vars.items())
    used = set()
    matched = 0
    mean_errs, std_errs = [], []

    for _, gt in gt_items:
        best_name, best_vp, best_diff = None, None, None
        for name, vp in ext_items:
            if name in used:
                continue
            diff = abs(vp.mean - gt["mean"])
            if best_diff is None or diff < best_diff:
                best_name, best_vp, best_diff = name, vp, diff
        if best_name is None:
            continue
        used.add(best_name)
        matched += 1
        mean_errs.append(abs(best_vp.mean - gt["mean"]) / max(abs(gt["mean"]), 1e-9))
        std_errs.append(abs(best_vp.std - gt["std"]) / max(abs(gt["std"]), 1e-9))

    return {
        "success_rate": matched / len(gt_items),
        "mean_rel_err": (sum(mean_errs) / len(mean_errs)) if mean_errs else None,
        "std_rel_err": (sum(std_errs) / len(std_errs)) if std_errs else None,
    }


def run_ablation_extraction_method(extractor: Extractor, n_llm_repeats: int = 3) -> pd.DataFrame:
    """Ablation A: LLM vs regex extraction, scored against known ground truth
    across multiple varied source texts (not a single snippet)."""
    print("\n--- Ablation A: Extraction Method (LLM vs Regex) ---")
    texts = _build_ablation_texts()
    rows = []

    for t in texts:
        t0 = time.perf_counter()
        v, c = extractor._extract_with_regex(t["text"])
        latency = time.perf_counter() - t0
        score = _score_extraction(v, t["ground_truth"])
        rows.append({"method": "regex", "text_id": t["id"], "rep": 0, "latency_s": latency, **score})

        for rep in range(n_llm_repeats):
            t0 = time.perf_counter()
            try:
                data = extractor._extract_with_llm(t["text"])
                latency = time.perf_counter() - t0
                v_llm, c_llm = extractor._parse_llm_result(data)
                score = _score_extraction(v_llm, t["ground_truth"])
                rows.append({"method": "llm", "text_id": t["id"], "rep": rep, "latency_s": latency, **score})
            except Exception as e:
                logger.warning("LLM extraction failed on %s (rep %d): %s", t["id"], rep, e)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No extraction results collected.")
        return df

    summary = df.groupby("method").agg(
        n=("latency_s", "count"),
        mean_latency_s=("latency_s", "mean"),
        mean_success_rate=("success_rate", "mean"),
        mean_rel_err=("mean_rel_err", "mean"),
        std_rel_err=("std_rel_err", "mean"),
    )
    print(summary.to_string())

    # Paired comparison per text_id (average LLM reps per text first so both
    # methods contribute exactly one observation per text_id).
    llm_by_text = df[df.method == "llm"].groupby("text_id")["mean_rel_err"].mean()
    regex_by_text = df[df.method == "regex"].groupby("text_id")["mean_rel_err"].mean()
    paired = pd.concat([llm_by_text, regex_by_text], axis=1, keys=["llm", "regex"]).dropna()
    if len(paired) >= 2:
        t_stat, p_val = scipy_stats.ttest_rel(paired["llm"], paired["regex"])
        print(f"\nPaired t-test (LLM vs regex mean_rel_err, n={len(paired)}): t={t_stat:.3f}, p={p_val:.4f}")
    else:
        print("\nNot enough paired texts with both methods succeeding for a significance test.")

    return df


def _params_from_ground_truth(t: dict, source_tag: str) -> Parameters:
    """Build Parameters directly from a domain's known ground truth (bypasses
    extraction, since Ablation B isolates the optimization loop, not extraction)."""
    variables = {
        name: VariableParams(
            name=name,
            distribution="normal",
            mean=gt["mean"],
            std=gt["std"],
            min=gt["mean"] - 3 * gt["std"],
            max=gt["mean"] + 3 * gt["std"],
        )
        for name, gt in t["ground_truth"].items()
    }
    v1, v2 = t["corr_pair"]
    correlations = {
        f"{v1}__{v2}": CorrelationParams(
            var1=v1,
            var2=v2,
            correlation=t["corr_val"],
            direction="positive" if t["corr_val"] >= 0 else "negative",
        )
    }
    return Parameters(
        variables=variables,
        correlations=correlations,
        meta=MetaParams(source=source_tag, extracted_at=datetime.now().isoformat(), extraction_method="ground_truth"),
    )


def run_ablation_noise_pivot(
    iterations_grid=(0, 1, 2, 3, 5, 10), n_seeds: int = 5
) -> pd.DataFrame:
    """Ablation B: overall_score vs Noise Pivot iteration count, across multiple
    (source text, seed) combinations, with paired significance testing."""
    print("\n--- Ablation B: Optimization Loop (Iterations 0..10, multiple seeds/texts) ---")
    texts = _build_ablation_texts()
    synthesizer = Synthesizer()
    validator = Validator()
    rows = []

    for t in texts:
        params = _params_from_ground_truth(t, source_tag=t["id"])
        for seed in range(n_seeds):
            for iterations in iterations_grid:
                np.random.seed(seed)
                if iterations == 0:
                    df = synthesizer.synthesize(params, noise_level=0.1)
                    score = validator.validate(df, params).overall_score
                else:
                    # Fresh Orchestrator per cell so the ratchet's best_score
                    # starts clean instead of carrying over across cells.
                    orch = Orchestrator(session_id=f"ablation_{t['id']}_{seed}_{iterations}")
                    orch.synthesizer = synthesizer
                    orch.validator = validator
                    score, _ = orch.run_optimization_loop(params, iterations=iterations)
                rows.append({"text_id": t["id"], "seed": seed, "iterations": iterations, "score": score})

    df = pd.DataFrame(rows)
    summary = df.groupby("iterations")["score"].agg(["mean", "std", "count"])
    print(summary.to_string())

    # Paired test: iterations=0 vs iterations=5, matched by (text_id, seed).
    zero = df[df.iterations == 0].set_index(["text_id", "seed"])["score"]
    five = df[df.iterations == 5].set_index(["text_id", "seed"])["score"]
    paired = pd.concat([zero, five], axis=1, keys=["zero", "five"]).dropna()
    if len(paired) >= 2:
        t_stat, p_val = scipy_stats.ttest_rel(paired["five"], paired["zero"])
        w_stat, w_p = scipy_stats.wilcoxon(paired["five"], paired["zero"])
        print(
            f"\n0 vs 5 iterations (n={len(paired)}): "
            f"paired t-test t={t_stat:.3f} p={p_val:.4f}; Wilcoxon p={w_p:.4f}"
        )
        print(f"Mean improvement (5 - 0 iterations): {(paired['five'] - paired['zero']).mean():+.4f}")

    return df


def run_ablation_study():
    print("\n" + "="*50)
    print("2. ABLATION STUDIES")
    print("="*50)

    extractor = Extractor(model=_OPENROUTER_MODEL, env_var=_OPENROUTER_ENV_VAR)
    df_extraction = run_ablation_extraction_method(extractor)
    _save_results(df_extraction, "ablation_extraction_method.csv")
    df_noise_pivot = run_ablation_noise_pivot()
    _save_results(df_noise_pivot, "ablation_noise_pivot.csv")


# Providers checked for cost/latency measurement. All three are routed through
# OpenRouter (litellm's "openrouter/<vendor>/<model>" model strings) so a single
# OPENROUTER_API_KEY covers OpenAI/Anthropic/Google instead of needing three
# separate provider keys.
_COST_LATENCY_PROVIDERS = [
    {"name": "OpenAI", "model": "openrouter/openai/gpt-4o", "key_env": "OPENROUTER_API_KEY"},
    {"name": "Anthropic", "model": "openrouter/anthropic/claude-sonnet-4.5", "key_env": "OPENROUTER_API_KEY"},
    {"name": "Google", "model": "openrouter/google/gemini-2.5-flash", "key_env": "OPENROUTER_API_KEY"},
]

# A deliberately verbose, "naturally written" version of extractor.CAVEMAN_PROMPT,
# used only to measure the real token savings of the compressed prompt.
_VERBOSE_PROMPT = """
You are an expert statistician with many years of experience analyzing research
papers and extracting statistical information from them. Please carefully read
through the following text and identify all of the statistical variables that
are mentioned, along with their probability distributions (such as normal,
uniform, or gamma), their means, and their standard deviations. Also, please
identify any correlations that are described between pairs of variables,
including the strength and direction (positive or negative) of each correlation.

Please format your response as a JSON object that matches the following schema
exactly:
{
  "variables": {
    "<name>": {"distribution": "normal|uniform|gamma", "mean": 0.0, "std": 1.0, "min": null, "max": null}
  },
  "correlations": {
    "<key>": {"var1": "<name>", "var2": "<name>", "correlation": 0.5, "direction": "positive|negative"}
  }
}

Please make sure to only output the JSON object itself and nothing else - no
introductory text, no explanations, and no markdown code fences.
"""

_SAMPLE_SOURCE_TEXT = _ABLATION_DOMAINS and """
We observed a heart rate distribution with mean 72 and std 8.
Blood pressure was normally distributed with mean 120 and std 15.
The correlation between heart rate and blood pressure is 0.35.
"""


def run_cost_latency_analysis(n_repeats: int = 5):
    print("\n" + "="*50)
    print("3. COST / LATENCY ANALYSIS (real litellm calls where keys are available)")
    print("="*50)

    available = [p for p in _COST_LATENCY_PROVIDERS if os.environ.get(p["key_env"])]
    if not available:
        print("No OPENROUTER_API_KEY found — skipping live cost/latency calls for all providers.")

    rows = []
    for p in available:
        for rep in range(n_repeats):
            t0 = time.perf_counter()
            try:
                response = completion(
                    model=p["model"],
                    messages=[
                        {"role": "system", "content": CAVEMAN_PROMPT},
                        {"role": "user", "content": f"Extract stats from this text:\n\n{_SAMPLE_SOURCE_TEXT}"},
                    ],
                    response_format={"type": "json_object"},
                    num_retries=0,
                )
                latency = time.perf_counter() - t0
                cost = litellm.completion_cost(completion_response=response)
                usage = response.usage
                rows.append(
                    {
                        "provider": p["name"],
                        "model": p["model"],
                        "rep": rep,
                        "latency_s": latency,
                        "cost_usd": cost,
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                    }
                )
            except Exception as e:
                logger.warning("Call to %s failed (rep %d): %s", p["name"], rep, e)

    df = pd.DataFrame(rows)
    if not df.empty:
        summary = df.groupby(["provider", "model"]).agg(
            n=("latency_s", "count"),
            mean_latency_s=("latency_s", "mean"),
            std_latency_s=("latency_s", "std"),
            mean_cost_usd=("cost_usd", "mean"),
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
        )
        print(summary.to_string())
    elif available:
        print("No successful calls were recorded.")

    # Real Caveman-vs-verbose token count comparison, per model. token_counter
    # works offline (no API key needed), so this always runs.
    print("\nCaveman Protocol token savings (measured via litellm.token_counter):")
    caveman_rows = []
    for p in _COST_LATENCY_PROVIDERS:
        try:
            caveman_tokens = litellm.token_counter(
                model=p["model"], messages=[{"role": "system", "content": CAVEMAN_PROMPT}]
            )
            verbose_tokens = litellm.token_counter(
                model=p["model"], messages=[{"role": "system", "content": _VERBOSE_PROMPT}]
            )
            reduction = (verbose_tokens - caveman_tokens) / verbose_tokens * 100
            print(
                f"  {p['name']:<10} ({p['model']}): verbose={verbose_tokens} tok, "
                f"caveman={caveman_tokens} tok, reduction={reduction:.1f}%"
            )
            caveman_rows.append({
                "provider": p["name"], "model": p["model"],
                "verbose_tokens": verbose_tokens, "caveman_tokens": caveman_tokens,
                "reduction_pct": reduction,
            })
        except Exception as e:
            logger.warning("Token counting failed for %s: %s", p["name"], e)

    _save_results(df, "cost_latency.csv")
    _save_results(pd.DataFrame(caveman_rows), "caveman_token_savings.csv")
    return df


if __name__ == "__main__":
    load_env()
    print("Starting AgentDataset IAAI-27 Benchmarks...\n")
    try:
        run_empirical_benchmark()
        run_ablation_study()
        run_cost_latency_analysis()
        print("\nBenchmarks complete. Ready for IAAI-27 submission.")
    except Exception as e:
        logger.exception("Benchmark failed")
