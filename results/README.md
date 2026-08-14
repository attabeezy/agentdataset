# Benchmark Results

CSV outputs written by `benchmark.py`. Each file is produced (and overwritten)
by the corresponding function on every run:

- `empirical_benchmark.csv` — `run_empirical_benchmark()`: TRTR vs TSTR
  metrics per dataset (UCI Adult, Pima Diabetes, German Credit).
- `ablation_extraction_method.csv` — `run_ablation_extraction_method()`:
  per-domain LLM vs regex extraction scores.
- `ablation_noise_pivot.csv` — `run_ablation_noise_pivot()`: fidelity score
  per iteration count / seed / source text.
- `cost_latency.csv` — historical per-call latency, cost, and token usage per
  provider, measured with the Caveman prompt. **Frozen** (see
  `baseline_manifest.json`); `run_cost_latency_analysis()` now writes current
  measurements to `extraction_cost_latency.csv` instead.
- `caveman_token_savings.csv` — historical Caveman vs verbose prompt token
  counts (21.9%, 242 → 189). **Frozen**; superseded but not overwritten.
  That verbose baseline omitted the `categorical` distribution and the
  `categories` field, so it requested strictly less than `CAVEMAN_PROMPT` and
  *understated* the saving.
- `extraction_prompt_token_savings.csv` — the corrected, schema-matched
  measurement: **38.6%** (308 → 189 tokens).
  `tests/test_extraction_challenge.py::test_prompt_schema_parity` prevents the
  two prompts from drifting apart again.
- `extraction_cost_latency.csv` — current per-call latency, cost, and token
  usage per provider, measured on whichever prompt is in production
  (`EXTRACTION_PROMPT`). Regenerated after the switch to the natural-language
  prompt: Google $0.0006/2.15 s, OpenAI $0.0023/2.00 s, Anthropic $0.0042/3.95 s.

**The six files listed in `baseline_manifest.json` are frozen historical bytes
and must never be regenerated in place.** `tests/test_baseline_manifest.py`
enforces this by hash and also asserts that `run_cost_latency_analysis()` does
not target their filenames.

The manifest carries two hash maps. `files` holds the original raw-byte hashes
as recorded on Windows and is kept as the historical record; it is **not**
portable, because git checks the same blob out with CRLF on Windows and LF on
Linux, so those hashes can never match in CI. `files_sha256_lf` holds
line-ending-normalized hashes of the identical content and is the map the test
enforces. Normalization does not weaken the check — any change to a value still
changes the hash, which `test_normalized_hash_ignores_line_endings_but_not_content`
verifies directly.

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
retained and still measured by the ablation.

The extraction-backed artifacts — `analysis/real_paper_*.csv` and
`real_paper_pipeline_*.csv` — were regenerated under the new prompt with
`real_paper_eval.py --mode cached` (cached text, so only the prompt changed).
The `gold` condition is bit-identical across all 18 summary metrics, confirming
only extraction moved. That evaluation independently replicates the §9.6 result:
extraction F1 .247 → .308 and precision .167 → .226 with **recall unchanged** at
.480, because false positives roughly halved (cdc 27 → 12, calerie 21 → 17)
while true positives were identical. The natural-language prompt wins by
inventing less, not by finding more.

Downstream synthesis shifted accordingly, because a smaller parameter set is
being synthesized. A `calerie_rct` fallback warning seen in the first
post-switch run did **not** reproduce on an independent re-run (0 across all
sources), so it was a property of that run's particular extraction rather than
a structural issue.

### Replication (two independent runs)

`extraction_challenge_raw_run1.csv` preserves the first 930-extraction sweep;
`extraction_challenge_raw.csv` is the second, run from a clean tree. Both reach
the same conclusion:

| pooled LLM arms | F1 | precision | recall | hallucination |
|---|---|---|---|---|
| caveman run1 → run2 | .710 → .719 | .639 → .648 | .960 → .963 | .154 → .154 |
| natural run1 → run2 | .778 → .782 | .720 → .723 | .989 → .988 | .098 → .094 |

Caveman-minus-natural F1 gap: **−.0686** (run 1) and **−.0629** (run 2), both
significant (Wilcoxon p=.0001 and p=.0004). The regex arm is bit-identical
across runs (max |ΔF1| = 0.0), confirming the spread is LLM-side.

Run 2 hit 45 `APIError` rows on Anthropic (15% of its calls) where run 1 hit
none. Retries are disabled by design so provider instability stays visible;
failures are recorded as rows and shrink the paired sample rather than biasing
the rates. Recompute the comparison from the two committed raw files with:

```powershell
.venv\Scripts\python.exe -c "import pandas as pd; a=pd.read_csv('results/extraction_challenge_raw_run1.csv'); b=pd.read_csv('results/extraction_challenge_raw.csv'); m=['f1','precision','recall','hallucination_rate']; print(pd.concat({'run1':a[a.arm!='regex'].groupby('arm')[m].mean(),'run2':b[b.arm!='regex'].groupby('arm')[m].mean()},axis=1).round(4))"
```

Both run manifests record `git_dirty: false` at a commit present in history.
`collect_provenance()` is now called before any output is written — previously
it ran afterwards, which made a dirty flag structurally unavoidable.

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
