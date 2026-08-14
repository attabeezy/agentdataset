# Benchmark Results

CSV outputs written by `benchmark.py`. Each file is produced (and overwritten)
by the corresponding function on every run:

- `empirical_benchmark.csv` — `run_empirical_benchmark()`: TRTR vs TSTR
  metrics per dataset (UCI Adult, Pima Diabetes, German Credit).
- `ablation_extraction_method.csv` — `run_ablation_extraction_method()`:
  per-domain LLM vs regex extraction scores.
- `ablation_noise_pivot.csv` — `run_ablation_noise_pivot()`: fidelity score
  per iteration count / seed / source text.
- `cost_latency.csv` — `run_cost_latency_analysis()`: per-call latency,
  cost, and token usage per provider. Measures the *shipped* pipeline, so it
  tracks `EXTRACTION_PROMPT`; regenerated when production switched from the
  Caveman prompt to the natural-language prompt (see §9.6 below).
- `caveman_token_savings.csv` — measured Caveman vs verbose prompt token
  counts per model. The verbose baseline was corrected in the P1 §9.6 work: it
  previously omitted the `categorical` distribution and the `categories` field,
  so it requested strictly less than `CAVEMAN_PROMPT` and *understated* the
  saving. Against a schema-matched baseline the reduction is **38.6%**
  (308 → 189 tokens), not the previously reported 21.9% (242 → 189).
  `tests/test_extraction_challenge.py::test_prompt_schema_parity` prevents the
  two prompts from drifting apart again.

Files are only written for functions that ran and produced at least one row.

## P0 reviewer-revision artifacts

- `baseline_manifest.json` freezes the pre-revision CSV hashes and repository
  provenance.
- `real_paper_pipeline_raw.csv`, `real_paper_pipeline_summary.csv`, and
  `real_paper_optimization_traces.csv` contain the 30-seed synthesis study for
  the three sources in `analysis/real_paper_manifest.csv`. Discovery and
  extraction scores live in `analysis/real_paper_*.csv`; the run manifest
  records the model, token use, latency, cost, lockfile hash, and command.
- `p0_utility_raw.csv` contains one long-format observation per metric and seed
  for 30 paired splits. `p0_utility_summary.csv` reports mean, sample standard
  deviation, and deterministic 10,000-resample bootstrap intervals. It covers
  three classification datasets with logistic regression and histogram
  gradient boosting, plus Concrete Compressive Strength regression with linear
  regression and histogram gradient boosting.

## P1 reviewer-revision artifacts

- `adult_calibration_raw.csv` and `adult_calibration.csv` contain the Adult
  post-hoc calibration study over 30 paired splits: eight threshold and
  calibration strategies crossed with the real training data, AgentDataset,
  independent marginals, and the SDV Gaussian copula, for logistic regression
  and histogram gradient boosting. Raw rows follow the shared schema plus two
  extra columns: `calibration_strategy`, and `uses_real_test_labels`, which is
  set only on the `oracle_threshold_real_test` diagnostic.
- Calibrators and thresholds are fitted on a validation split carved out of the
  *synthetic* training frame; the real test fold is scored against but never
  tuned on. The single exception is `oracle_threshold_real_test`, which is an
  upper bound on what thresholding alone could achieve and is not a usable
  method — it is flagged in the schema so it cannot be read as a result.
- `uncalibrated_full_fit` reproduces the P0 protocol (model fitted on the whole
  training frame, decision threshold 0.5) and exists to reconcile this study
  against `p0_utility_summary.csv`.
- Isotonic regression is skipped, with `status="skipped"` and an explicit
  `failure_stage`, when the validation split has fewer than 500 rows or fewer
  than 50 members of either class.

### Extraction robustness and prompt ablation (§9.6)

- `extraction_challenge_raw.csv` — one row per (case, arm, model, repetition)
  over the 30-case challenge set in `analysis/extraction_challenge_set.json`.
  Arms are `caveman` (the production prompt), `natural_language` (a verbose
  prompt requesting the identical schema), and `regex` (the fallback, run once
  and never folded into LLM results). Columns split false positives by cause:
  `fp_distractor` (context error), `fp_fabricated` (hallucination),
  `fp_conflated` (mean and SD taken from different populations), and `fp_other`.
- `extraction_challenge_summary.csv` — per-arm × per-model means with
  deterministic 10,000-resample bootstrap intervals, via
  `summarize_long_results` in `agentdataset/evaluation.py`.
- `extraction_challenge_by_category.csv` — the same metrics per taxonomy
  category (negated findings, multiple populations, footnotes, …), so a reader
  can see which kind of messiness each method fails on.
- `extraction_challenge_run_manifest.json` — provenance, model list, challenge
  set sha256, real/constructed split, per-call token and cost history, and the
  API error count.

Headline from the committed 930-extraction sweep: the Caveman prompt buys a
30.0% prompt-token reduction (404.6 → 283.3 tokens per request) but only a 3.5%
cost reduction, because completion tokens dominate — and it costs quality. F1 is
.710 [.657,.760] against .778 [.729,.825] for the schema-matched
natural-language prompt (paired Wilcoxon p=.0001, same direction on all three
models). Recall is near-ceiling for both; the loss is precision, driven by
hallucination rising from .098 to .154. Regex reaches F1 .156.

**Acted on:** production now uses the natural-language prompt
(`EXTRACTION_PROMPT` in `agentdataset/core/extractor.py`). `CAVEMAN_PROMPT` is
retained and still measured by the ablation. Note that the extraction-backed
artifacts here — `real_paper_*.csv`, `real_paper_pipeline_*.csv` — predate the
switch and were produced with the Caveman prompt; see the follow-up item in
PROJECT.md §9.6.

A second finding is recorded in `fence_stripped`: Anthropic models via OpenRouter
ignore `response_format=json_object` and wrap the object in a Markdown fence on
100% of calls, so a strict `json.loads` fails on every one. `strip_json_fence` in
`agentdataset/core/extractor.py` handles this; without it the pipeline silently
falls back to regex for that provider while appearing to succeed.

API failures are recorded as rows with `api_error=True` rather than dropped, so
they cannot bias the reported rates downward. All arms keep the production
`response_format={"type": "json_object"}` setting, so the malformed rate is
schema-level; retries are disabled so nondeterminism stays visible. Every model
is routed through OpenRouter, so one `OPENROUTER_API_KEY` covers the sweep. See
`analysis/EXTRACTION_CHALLENGE_PROTOCOL.md` for the annotation rules and
limitations.

Reproduce the new studies with:

```powershell
.venv\Scripts\python.exe analysis\real_paper_eval.py --mode live
.venv\Scripts\python.exe analysis\real_paper_eval.py --mode cached
.venv\Scripts\python.exe analysis\p0_utility_benchmark.py --seeds 30
.venv\Scripts\python.exe analysis\adult_calibration.py --seeds 30

# Extraction challenge set and prompt ablation (§9.6)
.venv\Scripts\python.exe analysis\build_extraction_challenge_set.py --check
.venv\Scripts\python.exe analysis\extraction_ablation.py --dry-run          # regex only, no API key
.venv\Scripts\python.exe analysis\extraction_ablation.py --models openai --reps 1 --cases negated_finding
.venv\Scripts\python.exe analysis\extraction_ablation.py --reps 5           # full 3-model sweep
```

The utility benchmark computes parameters from each training fold only. Raw
rows use the shared schema defined in `agentdataset/evaluation.py`; empty and
failed runs remain explicit rather than disappearing during aggregation.
