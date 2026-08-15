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

### Synthesis and optimization diagnostics (§9.7)

**Headline: the advertised copula path is almost never bypassed for the reason
we assumed, and very often bypassed for a reason we had not measured.**
Positive-definiteness is not the binding constraint on real inputs — every
templated, gold, and auto-extracted matrix in the sweep is positive-definite.
What actually destroys dependence is *name resolution*: a declared correlation
whose variables are absent from the extracted variable dictionary is discarded
silently, leaving an identity matrix that passes every validity check and
imposes nothing.

- `corr_validity_per_matrix.csv` — one row per correlation matrix the pipeline
  builds, with `min_eigenvalue`, `condition_number`, `is_pd`, `action`
  (`cholesky_ok` / `dropped_all` / `repaired_psd` / `no_correlations`),
  `n_pairs_declared`, `n_pairs_applied`, `n_pairs_skipped_unknown_var`,
  `n_pairs_overwritten`, and the `corr_score` obtained under both the shipped
  drop-all fallback and the nearest-PSD repair.
- `corr_validity_raw.csv` / `_summary.csv` — the same observations in the shared
  long schema, grouped by input class. Key metrics: `is_pd`, `dropped_all`,
  `vacuous_matrix` (no pair resolved), `pairs_applied_rate`, and
  `effective_copula` (positive-definite *and* at least one pair applied).
- Effective copula rate **as first measured**: auto-extracted 0/8 pairs applied
  (0%), templated 11/20 (55%), gold 3/4 (75%). The regex extractor named
  continuous variables positionally (`var_1`, `var_2`, …) while emitting
  correlations under the names found in the text, so no regex-extracted pair
  could ever resolve.
- **This was a real defect and has been fixed.** `_extract_with_regex` now
  recovers the variable name from the prose immediately preceding a mean/std
  hit and normalizes names on both sides, so `heart rate` in the prose matches
  `heart_rate` in the correlation clause. After the fix: auto-extracted **8/8
  (100%)**, templated **20/20 (100%)**, gold unchanged at 3/4 — that remaining
  pair names a variable the source genuinely never reports, which is correct.
  On the eight templated domains the correlation score rises **77.3 → 99.3**
  and overall fidelity **50.9 → 59.7**, consistent across all eight.
- Where the prose supplies no confident name the positional `var_N` fallback is
  kept on purpose: a *wrong* name mismatches silently, which is worse than a
  visibly unresolved one. Guarded by
  `tests/test_corr_validity_audit.py::test_regex_correlations_resolve_against_extracted_variables`
  and eight naming tests in `tests/test_extractor.py`.
- The fix does **not** move other reported numbers. `_score_extraction` matches
  by nearest mean rather than by name, so §9.6/Q3 extraction F1 is unchanged,
  and `real_paper_pipeline_raw.csv` holds only the `gold` and `llm` conditions
  (regex extracted no target items from the real papers), so Q1 is unchanged.
  The LLM parse path is deliberately untouched.
- Non-positive-definite matrices appear only in the constructed `stress` class
  (4/5), which exists precisely so the 100% PD rate on natural inputs is a
  measured result rather than an untested assumption.
- The nearest-PSD repair (eigenvalue clipping with diagonal renormalisation,
  Higham's first iterate) is **opt-in**: `Synthesizer(corr_repair="psd")`. The
  default remains `"drop"`, so every frozen baseline in `baseline_manifest.json`
  stays valid. On the slightly-invalid stress case the repair recovers
  correlation fidelity that dropping destroys.
- `noise_pivot_ablation_raw.csv.gz` / `_summary.csv` — the sensitivity grid over
  `patience`, `initial_noise`, `explore_mult`, `exploit_factor`,
  `reset_multiple`, and iteration budget, with fixed-noise, random-search, and
  no-loop baselines matched on synthesize+validate call count. Extra columns:
  `family`, `factor`, `level`, `iterations`, `split`.
- `noise_pivot_traces.csv.gz` — per-iteration noise level, score, and pivot
  `phase`, which is what makes the reset frequency and the budget-dependent
  variance measurable rather than asserted.
- `noise_pivot_tests.csv` / `_budget.csv` / `_variance.csv` — produced by
  `analysis/noise_pivot_report.py` from the raw grid: paired Wilcoxon tests
  against the shipped configuration with Holm–Bonferroni correction inside each
  factor family, score and runtime by budget, and the variance decomposition.
- Sources are split into a **tune** set and a held-out **report** set, declared
  in code before any result was inspected and recorded in the run manifest, so
  the question "were these hyperparameters tuned on the data you report on?" has
  a checkable answer. Every finding below replicates on the held-out split with
  near-identical effect sizes.

**Headline: the shipped noise-pivot defaults are not empirically justified.**
The fidelity landscape is monotone decreasing in noise with a cliff between
0.05 and 0.1 (mean per-iteration score 97.8 at noise 0.01, 93.2 at 0.05, 70.8
at 0.1, 62.0 at ≥0.2), and the shipped `initial_noise=0.1` sits on the wrong
side of it. At the shipped 5-iteration budget the adaptive schedule scores
88.50 (across-seed SD 10.0) against **99.88 (SD 0.07) for a fixed noise of
0.01** — Δ +11.47, paired Cohen's *d* 1.09, Holm-adjusted Wilcoxon
*p* = 3.9e-16. Budget-matched random search also wins (99.49). Zero iterations
at noise 0.01 beats the incumbent's five-iteration run by +10.09. The loop is
largely compensating for a bad default starting noise. The defaults are left
unchanged here: that is a behaviour change and its own decision.

**The 10-iteration variance is explained.** Outcomes are a two-point mixture —
a run either accumulates enough halving steps to cross the noise cliff
(scoring ~100) or does not (~60), with nothing in between; the frozen
`ablation_noise_pivot.csv` spans [59.94, 100.00] at 10 iterations with 21
distinct values in 40 cells. Across-run SD is therefore the spread of a
Bernoulli mixture, maximal at the budget where about half the runs have
crossed. That midpoint is 10 iterations in the paper's configuration (SD
17.31) and ~5 here (43.6% reach ≥95 at 5 iterations, 95.5% at 10), so in this
study SD *falls* monotonically with budget: 12.0 → 10.7 → 3.7 → 0.69 for
budgets 0 → 5 → 10 → 20. It is a deterministic descent crossing a threshold,
not the schedule getting lucky.

**Caveat on the frozen `ablation_noise_pivot.csv`.** `run_ablation_noise_pivot`
calls `np.random.seed(seed)`, which seeds the legacy global RNG, while
`Synthesizer` draws from an instance-scoped `default_rng` — verified: changing
the global seed leaves synthesis byte-identical. Since that function shares one
`Synthesizer` across all cells, its `seed` column is a label rather than a
control and the variation is the shared stream's position. The file stays
frozen; the new study seeds a fresh `Synthesizer` per trial.

### What the fidelity score certifies (§9.7 follow-up)

**The reported fidelity score is exactly invariant to errors in the extracted
parameters.** `fidelity_validity_probe_raw.csv` builds parameters from three
real UCI datasets, corrupts them, synthesizes from the corrupted parameters,
and records both what the system reports (`reported_*`, scored against the
parameters it believes) and how far the output actually is from the real data
(`external_*`). Within every (dataset, seed) cell the reported score takes a
**single value across all 14 corruption conditions**, while the external
two-sample KS distance rises 0.126 → 0.967 and the mean error reaches 141.9
standard deviations. Attaching each variable's statistics to a different
variable changes the reported score by zero.

This is invariance by construction. The KS term (weight .40) compares each
column against the same parameterized CDF, with the same mean and standard
deviation, that the synthesizer drew it from; the bias term (.20) compares the
sample mean against that same declared mean. Any location–scale error in the
parameters moves the data and the reference together and cancels. The
correlation term (.40) is the exception and does carry signal — it detects a
dropped or unresolved matrix (52–67 against 80–100 repaired) — but it too
reproduces a wrong *declared* correlation faithfully.

Net: the score measures whether the synthesizer did what it was told, never
whether it was told the right thing. That is the common cause behind high
intrinsic fidelity coexisting with weak TSTR/HGB utility, and behind extraction
quality never surfacing in fidelity.

**Acted on:** the paper was reframed rather than renumbered — no reported value
changed. The self-refinement contribution bullet became an audit of the
internal objective, and System Overview, Q1, Discussion, and the scoring
appendix now state the score's scope.

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

# Synthesis and optimization diagnostics (§9.7) — no API key required
.venv\Scripts\python.exe analysis\corr_validity_audit.py
.venv\Scripts\python.exe analysis\noise_pivot_ablation.py --quick          # smoke grid
.venv\Scripts\python.exe analysis\noise_pivot_ablation.py --seeds 20       # full grid
.venv\Scripts\python.exe analysis\noise_pivot_report.py                    # tests and tables
.venv\Scripts\python.exe analysis\fidelity_validity_probe.py --seeds 5     # what the score certifies
```

The utility benchmark computes parameters from each training fold only. Raw
rows use the shared schema defined in `agentdataset/evaluation.py`; empty and
failed runs remain explicit rather than disappearing during aggregation.
