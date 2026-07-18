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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

def run_empirical_benchmark():
    print("\n" + "="*50)
    print("1. EMPIRICAL BENCHMARK (Downstream ML Performance)")
    print("="*50)

    # 1. Simulate a "Real" Dataset (Oracle)
    np.random.seed(42)
    n_samples = 1000
    income = np.random.normal(60000, 15000, n_samples)
    age = np.random.normal(45, 10, n_samples)

    # Target variable depends on income and age
    loan_amount = 0.5 * income + 1000 * age + np.random.normal(0, 5000, n_samples)

    df_real = pd.DataFrame({
        "income": income,
        "age": age,
        "loan_amount": loan_amount
    })

    # Calculate Oracle Statistics
    means = df_real.mean()
    stds = df_real.std()
    corr = df_real.corr()

    # Text representing the literature about this dataset
    source_text = f"""
    A recent study on lending practices analyzed a dataset of {n_samples} borrowers.
    Income distribution: mean = {means['income']:.0f}, std = {stds['income']:.0f}.
    Age distribution: mean = {means['age']:.0f}, std = {stds['age']:.0f}.
    Loan amount distribution: mean = {means['loan_amount']:.0f}, std = {stds['loan_amount']:.0f}.

    Correlation analysis revealed:
    - correlation between income and loan_amount is {corr.loc['income', 'loan_amount']:.2f}.
    - correlation between age and loan_amount is {corr.loc['age', 'loan_amount']:.2f}.
    - correlation between income and age is {corr.loc['income', 'age']:.2f}.
    """

    print("Generating synthetic data based on extracted parameters...")

    orchestrator = Orchestrator(session_id="benchmark_empirical")
    params = orchestrator.extractor.extract_parameters(source_text, "Oracle_Study")

    if not params.variables:
        print("Extraction failed. Using regex fallback mock...")
        v, c = orchestrator.extractor._extract_with_regex(source_text)
        params = Parameters(variables=v, correlations=c, meta=MetaParams(source="Oracle_Study", extracted_at=datetime.now().isoformat(), extraction_method="regex_fallback"))

    best_score, df_synth = orchestrator.run_optimization_loop(params, iterations=3)

    # Train test split for real data
    X_real = df_real[['income', 'age']]
    y_real = df_real['loan_amount']
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

    # Train test split for synthetic data
    X_train_synth = df_synth.iloc[:, :2]
    y_train_synth = df_synth.iloc[:, 2]

    # Model 1: Trained on Real, Tested on Real
    model_real = LinearRegression()
    model_real.fit(X_train_real, y_train_real)
    preds_real = model_real.predict(X_test_real)
    mse_real = mean_squared_error(y_test_real, preds_real)

    # Model 2: Trained on Synthetic, Tested on Real (TRTS)
    model_synth = LinearRegression()
    model_synth.fit(X_train_synth, y_train_synth)
    preds_synth = model_synth.predict(X_test_real)
    mse_synth = mean_squared_error(y_test_real, preds_synth)

    print(f"\nResults:")
    print(f"Model trained on REAL data -> MSE on Real Test Set: {mse_real:.2f}")
    print(f"Model trained on SYNTH data -> MSE on Real Test Set: {mse_synth:.2f}")
    print(f"Relative Performance (TRTS / TRTR): {mse_synth / mse_real:.2f}x (closer to 1.0 is better)")


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

    extractor = Extractor()
    run_ablation_extraction_method(extractor)
    run_ablation_noise_pivot()


# Providers checked for cost/latency measurement. All three are routed through
# OpenRouter (litellm's "openrouter/<vendor>/<model>" model strings) so a single
# OPENROUTER_API_KEY covers OpenAI/Anthropic/Google instead of needing three
# separate provider keys.
_COST_LATENCY_PROVIDERS = [
    {"name": "OpenAI", "model": "openrouter/openai/gpt-4o", "key_env": "OPENROUTER_API_KEY"},
    {"name": "Anthropic", "model": "openrouter/anthropic/claude-3.5-sonnet", "key_env": "OPENROUTER_API_KEY"},
    {"name": "Google", "model": "openrouter/google/gemini-2.0-flash-001", "key_env": "OPENROUTER_API_KEY"},
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
        except Exception as e:
            logger.warning("Token counting failed for %s: %s", p["name"], e)

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
