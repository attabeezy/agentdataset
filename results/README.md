# Benchmark Results

## Introduction

This directory contains the quantitative evidence for AgentDataset. The CSV files are the source of truth for the numbers in the [manuscript](../docs/MANUSCRIPT.md): every reported value is derived from one of these artifacts, and the table captions in the manuscript name their source file. Results produced by `benchmark.py` are current; artifacts from earlier reviewer-revision studies are retained as historical records and are identified explicitly below. The 2026-11-23 refresh regenerated three previously frozen benchmark outputs after fixing correlation-sign handling, making ablation seeds real, and restoring the SDV baseline.

## Body

**<u>Current benchmark outputs</u>**

The active benchmark suite is defined in `benchmark.py` and writes four families of results. All of them were regenerated after the correlation-direction fix, so the values below reflect the current code. These files back Tables 1, 3, and 4 of the manuscript.

- **Empirical utility.** `empirical_benchmark.csv` holds 9 rows: three datasets × three synthesizers, five seeds each. AgentDataset reaches TSTR accuracy 0.809 on Adult, 0.769 on Pima, and 0.689 on German Credit, against TRTR 0.816, 0.771, and 0.693; ROC-AUC differs from TRTR by at most 0.006. SDV achieves 0.772, 0.758, and 0.698, and independent marginals 0.761, 0.649, and 0.700. Before the sign fix, Pima TSTR ROC-AUC was 0.18; after it, 0.823.
- **Optimization budget.** `ablation_noise_pivot.csv` contains all 240 cells of the eight-domain × five-seed × six-budget sweep. Mean fidelity is 59.82 ± 0.09 at zero iterations (noise 0.1), 95.82 ± 8.14 at one, 99.34 ± 3.16 at two, and 99.84 ± 0.08 at three, five, and ten. The paired 0-versus-5 contrast is +40.02 fidelity (n = 40, paired *t* p ≈ 2.3e-123, Wilcoxon p ≈ 2.1e-8).
- **Noise-matched control.** `one_shot_noise_matched.csv` holds 40 one-shot cells at the agent loop's starting noise of 0.02. It scores 95.82 ± 8.14, identical cell-by-cell to the one-iteration arm, showing that the large zero-to-one jump is a noise-level effect. Three iterations improve on it by +4.02 fidelity (paired *t* p = 0.0034, Wilcoxon p = 0.011).
- **Extraction and cost.** `ablation_extraction_method.csv` records 24/24 successful synthetic-text extractions with zero relative error at 1.91 s per call. `extraction_cost_latency.csv` reports five calls per model: $0.0023 / 2.00 s for `gpt-4o`, $0.0042 / 3.95 s for Claude Sonnet 4.5, and $0.0006 / 2.15 s for Gemini 2.5 Flash. `extraction_prompt_token_savings.csv` measures the Caveman prompt at 515 → 189 tokens (63.3% saving); it was not adopted because it loses 2.8–8.9 F1 points on the challenge set.

**<u>Downstream utility with gold summaries</u>**

The P0 utility study separates synthesis quality from LLM extraction by computing gold summaries from each training fold. It uses 30 paired splits, four datasets, logistic regression and histogram gradient boosting for classification, and linear regression and gradient boosting for Concrete Compressive Strength. These artifacts back Table 2 of the manuscript.

- **Classification utility.** `p0_utility_summary.csv` reports logistic-regression TSTR accuracy of 0.799 on Adult, 0.757 on Pima, and 0.702 on German Credit, against SDV at 0.770, 0.757, and 0.701. AgentDataset is effectively tied with the one-shot copula on all three, while independent marginals reach 0.761, 0.649, and 0.700.
- **Regression utility.** On Concrete, gradient-boosted TSTR R² is 0.528 for AgentDataset, 0.497 for one-shot, −0.256 for independent marginals, and 0.622 for SDV. This is the clearest case where record-level access matters: SDV can model nonlinear structure beyond the reported pairwise correlations.
- **Minority-F1 calibration.** `adult_calibration.csv` and `adult_f1_gap.csv` show that the Adult minority-F1 gap is largely a threshold artifact. At the default 0.5 threshold, isotonic and Platt calibration give minority F1 0.342 and 0.366; selecting the threshold on synthetic validation raises it to 0.613, close to the oracle value 0.624, with ROC-AUC unchanged at about 0.85.

**<u>Extraction quality</u>**

Extraction is evaluated on both a controlled adversarial set and three heterogeneous real papers. The controlled set is the more diagnostic test; the real-paper study shows how close extracted parameters come to a hand-curated specification.

- **Adversarial challenge.** `extraction_challenge_summary.csv` covers 30 cases across ten failure categories, five repetitions per case and condition. With natural-language prompting, F1 is 0.791 for `gpt-4o`, 0.789 for Gemini 2.5 Flash, and 0.764 for Claude Sonnet 4.5; hallucination rates are 0.093, 0.106, and 0.099, and variable-name accuracy remains 0.44–0.56. The Caveman prompt is consistently worse (F1 0.675–0.763), which is why production uses natural language.
- **Real-paper extraction.** `real_paper_pipeline_summary.csv` evaluates 30 synthesis seeds per paper. LLM extraction fidelity is 97.41 versus 98.61 gold on the randomized trial, 61.63 versus 63.35 on the NHANES report, and 96.81 versus 98.45 on the student survey; correlation scores are 96.7–99.5 in both conditions. The NHANES specification is intrinsically hard for this sampler because its reported structure is coarse and categorical.
- **Historical extraction sweeps.** `extraction_challenge_raw.csv`, `_summary.csv`, and `_by_category.csv` preserve the earlier 930-call prompt ablation and its replication. They predate the current production prompt and the correlation-sign fix and are kept for provenance, not as current headline numbers.

**<u>Synthesis diagnostics and score validity</u>**

Two studies explain what the internal fidelity score does and does not certify. Together they are the reason the manuscript describes fidelity as a sampler-consistency diagnostic rather than evidence of source fidelity.

- **Correlation matrices.** `corr_validity_summary.csv` reports correlation scores of 97.6–99.7 for gold, templated, and automatically extracted specifications, with every declared pair applied. In five stress specifications, only one matrix is positive definite; without repair the correlation score is 66.6 ± 19.1, and opt-in nearest-PSD repair raises it to 92.5 ± 9.8. Repair improves fit but cannot recover information that was never declared.
- **Fidelity invariance.** `fidelity_validity_probe_summary.csv` perturbs declared parameters before synthesis. For Adult under a mean shift, reported fidelity stays 95.64, reported correlation 99.10, and reported KS 90.0 at every corruption level from ε = 0 to 2, while external KS against the real data rises from 0.143 to 0.709. The score checks whether the synthesizer did what it was told, not whether the instruction was correct.
- **Consequence for evaluation.** High intrinsic fidelity can coexist with weak downstream utility, and extraction quality does not surface in the fidelity score. The manuscript reports an independent trust score and external utility tests for this reason; no privacy guarantee is claimed.

**<u>End-to-end cost and latency</u>**

Cost was measured both per extraction call and over the complete discovery-to-validation pipeline. The dominant expense is document discovery, not the statistical model.

- **Per-dataset spend.** `end_to_end_cost_latency.csv` averages three real papers to $0.0169 and 41.84 s per final dataset, with a range of $0.0133–$0.0209 and 39.07–45.64 s. Each run uses three LLM calls, about 4,977 prompt tokens and 442 completion tokens.
- **Stage breakdown.** Mean wall time is 7.80 s in LLM APIs, 33.06 s in search and fetch network activity, and under 1 s in local synthesis and validation. Human source-review time is excluded.
- **Current versus frozen cost files.** `extraction_cost_latency.csv` is the current per-call measurement; `cost_latency.csv` and `caveman_token_savings.csv` are frozen historical baselines and are not overwritten. Older end-to-end artifacts also predate the 2026-11-23 refresh.

**<u>Agentic review and reconciliation (new pipeline)</u>**

The frozen benchmark artifacts call `Extractor.extract_parameters()` directly and do not
exercise the review stage added after those runs. A separate study closes that gap:

- `review_gate_study_raw.csv`, `review_gate_study_summary.csv`,
  `review_gate_study_decisions.csv`, and `review_gate_study_manifest.json` come from
  `analysis/pipeline_review_study.py`. It fetches the three real-paper sources, builds 55
  claims (20 gold-standard statistics with real evidence passages, 35 recorded unsupported
  extractions without one), and runs the production LLM reviewer over them with 30
  deterministic synthesis seeds at noise 0.02.
- The reviewer rejected 35/35 unsupported claims but accepted only 3/20 verified claims
  (12 deferred, 5 rejected), leaving two sources with no synthesizable variables. The current
  gate is conservative; these are mechanism and calibration results, not evidence that review
  improves downstream utility. Reconciliation of the three gold specifications kept the one
  shared variable (`age`) separate rather than averaging across incompatible studies.

**<u>Provenance and reproduction</u>**

`baseline_manifest.json` records hashes and repository provenance for six result files. Three of them were deliberately regenerated in the 2026-11-23 refresh; the other three remain frozen.

- **Manifest contract.** The manifest lists `files` (raw-byte hashes recorded on Windows) and `files_sha256_lf` (line-ending-normalized hashes enforced by `tests/test_baseline_manifest.py`). It records `regenerated_files` for `empirical_benchmark.csv`, `ablation_extraction_method.csv`, and `ablation_noise_pivot.csv`, with the reason for each refresh. Frozen files remain `cost_latency.csv`, `caveman_token_savings.csv`, and `adult_f1_gap.csv`.
- **Refresh rationale.** The regenerated files incorporate the restored SDV baseline, a fresh `Synthesizer(seed=seed)` per ablation cell, and the fix that folds an explicit negative correlation direction into the stored sign. The old label-only ablation checkpoint was discarded rather than mixed with the corrected run.
- **Reproduce the main tables.** Run `uv run python benchmark.py --only empirical`, `--only ablation-b`, and `--only cost`. Ablation B resumes from its checkpoint by default; `--fresh` restarts it. The extended studies live in `analysis/`, with `p0_utility_benchmark.py`, `adult_calibration.py`, `corr_validity_audit.py`, and `fidelity_validity_probe.py` running locally without an API key.
- **Source of truth.** Generated CSVs are authoritative; the manuscript and this index are written against them. When a result changes, update the CSV, the manifest, and the corresponding manuscript table in the same change.

## Conclusion

These artifacts support a narrow but reproducible claim: AgentDataset reconstructs per-variable marginals and pairwise monotone dependence from published statistics, and its synthetic tables retain much of the downstream utility of real data at low cost. The same results define its limits — no higher-order dependence, no privacy guarantee, and a fidelity score that is blind to upstream parameter error. Current benchmark outputs are identified above; older files are retained as historical evidence and should not be read as the final measurements.
