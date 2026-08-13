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
  cost, and token usage per provider.
- `caveman_token_savings.csv` — measured Caveman vs verbose prompt token
  counts per model.

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

Reproduce the new studies with:

```powershell
.venv\Scripts\python.exe analysis\real_paper_eval.py --mode live
.venv\Scripts\python.exe analysis\real_paper_eval.py --mode cached
.venv\Scripts\python.exe analysis\p0_utility_benchmark.py --seeds 30
.venv\Scripts\python.exe analysis\adult_calibration.py --seeds 30
```

The utility benchmark computes parameters from each training fold only. Raw
rows use the shared schema defined in `agentdataset/evaluation.py`; empty and
failed runs remain explicit rather than disappearing during aggregation.
