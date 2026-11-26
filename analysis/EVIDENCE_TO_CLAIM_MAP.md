# Evidence-to-Claim Map

This map implements docs/PROJECT.md §9.1 for the canonical submission paper at
`docs/submission/paper.tex` (AAAI 2027 Workshop on AI for Scientific Research,
double-blind). Locations use stable section and table labels rather than line
numbers. A claim without a reproducible artifact or an inspectable code path is
not treated as empirical evidence.

## Headline and real-paper claims

| Claim | Paper location | Evidence | Re-derive or bound |
|---|---|---|---|
| Discovery finds all three prespecified target papers. | Abstract; §4.1 (Q1) | `analysis/real_paper_discovery_summary.csv` | Verify `target_found=True` for all three sources. CALERIE first appears at rank 13, so this is target recovery, not recall@5. |
| Caller-schema-guided extraction reaches .708 micro-F1 on prespecified fields. | Abstract; §4.1 (Q1); `tab:realpapers` | `analysis/real_paper_extraction_summary.csv`; `results/real_paper_extraction_schema_manifest.json` | Read the `llm` row: precision .575, recall .920, F1 .7077. The caller schema comes from `discovery_query` and `target_scope`, not gold labels; the earlier unguided F1 was .304. |
| Regex extracts none of the prespecified real-paper fields. | §4.1 (Q1); `tab:realpapers` | Same file | Read the `regex` row: precision, recall, and F1 are zero. |
| Source scope and extracted claims require agentic/human review. | Abstract; §2.2; §4.1; §5 | `analysis/real_paper_errors.csv`; `analysis/real_paper_extraction_summary.csv` | This is an operating bound supported by incomplete extraction and recorded error categories, not a performance claim about human review. |
| The three-paper caller-schema extraction replicate cost $0.122 and took 44.4 seconds over 21 calls. | §4.1 (Q1) | `results/real_paper_extraction_schema_manifest.json` | Sum `cost_usd` and `latency_seconds` in `call_history`. This replaces the earlier extraction-only replicate cost of $0.0343 / 13.9 s. |

## Agentic review and reconciliation (new-pipeline evidence)

The frozen benchmark artifacts call `Extractor.extract_parameters` directly and
do not exercise the review stage. The rows below come from a separate study of
the shipped gate (`analysis/pipeline_review_study.py`) and must not be presented
as part of the frozen real-paper benchmark.

| Claim | Paper location | Evidence | Re-derive or bound |
|---|---|---|---|
| The review gate rejects unsupported claims but is conservative on verified ones. | §4.7 (Q7); `tab:review_gate` | `results/review_gate_study_decisions.csv`; `results/review_gate_study_summary.csv` | 35/35 unsupported claims rejected; 3/20 verified accepted, 12 deferred, 5 rejected. Two sources end with no synthesizable variables. This measures the current prompt, not review quality in general. |
| Reconciliation flags incompatible cross-source measures instead of averaging them. | §4.7 (Q7) | `results/review_gate_study_manifest.json` (`reconciliation_decisions`) | The only shared variable (`age`) is `kept_separate`; no `pooled_average` decisions. |

## Downstream utility and baseline-input claims

| Claim | Paper location | Evidence | Re-derive or bound |
|---|---|---|---|
| With verified summaries, synthetic data preserve much linear ranking signal but lose minority-class and nonlinear utility. | Abstract; §4.2 (Q2); `tab:utility` | `results/p0_utility_raw.csv`; `results/p0_utility_summary.csv` | Compare TRTR and TSTR metrics by dataset, model, and seed. The claim is restricted to the evaluated tasks. |
| AgentDataset is stronger than independent marginals on the evaluated utility metrics where dependence matters. | §4.2 (Q2); `tab:utility` | `results/p0_utility_baseline_contrasts_raw.csv`; `results/p0_utility_baseline_contrasts.csv` | Use paired seed-level contrasts with the declared metric direction. |
| AgentDataset and independent marginals receive only training-fold summaries. | §3 (Q2); `tab:utility` caption | `analysis/p0_utility_benchmark.py` (`_gold_params_from_train`); `results/p0_utility_run_manifest.json` | Inspect the code path and manifest scope. This is an input-protocol claim, not a measured result. |
| SDV is fitted on real training records and is a data-access upper bound. | §4.2 (Q2); `tab:utility` caption | `analysis/p0_utility_benchmark.py` (`_sdv_baseline`) | `_sdv_baseline` receives the raw training DataFrame. It is not a like-for-like literature-summary baseline. |
| Calibration can recover threshold-sensitive minority F1 but cannot restore lost ranking signal. | §4.2 (Q2); §4.6 (Q6); `tab:calibration` | `results/adult_calibration_raw.csv`; `results/adult_calibration.csv` | Compare honest synthetic-validation thresholding with the fixed threshold and the explicitly labeled oracle. |

## Extraction, optimization, and fidelity claims

| Claim | Paper location | Evidence | Re-derive or bound |
|---|---|---|---|
| The Caveman prompt reduces billed prompt tokens by 30.0% but saves only 3.5% cost and lowers F1 from .778 to .710. | §4.3 (Q3); `tab:extraction` | `results/extraction_challenge_raw.csv`; `results/extraction_challenge_summary.csv` | Compare schema-matched `natural` and `caveman` arms; paired tests are recorded in the summary artifacts. |
| Regex reaches F1 .156 on the challenge set. | §4.3 (Q3); `tab:extraction` | Same files | Read the forced-regex arm; it is reported separately from paid-model repetitions. |
| The shipped noise-pivot defaults are not empirically optimal, and `reset_multiple` is inert at the shipped budget. | §4.4 (Q4); `tab:noise`; `tab:pivot` | `results/noise_pivot_ablation_summary.csv`; `results/noise_pivot_knob_ranking.csv` | Compare the shipped schedule with fixed-noise arms and inspect the ranked knob effect ranges. |
| The internal fidelity score cannot detect incorrect extracted parameters. | Abstract; §4.6 (Q6); §5; `tab:validity` | `results/fidelity_validity_probe_raw.csv`; `results/fidelity_validity_probe_summary.csv` | Compare reported fidelity across controlled parameter corruptions. The correlation term remains a synthesis diagnostic, so the paper does not call the score task validity. |
| The repaired extraction path applies all declared resolvable correlation pairs in the audit. | §4.6 (Q6); `tab:correlation` | `results/corr_validity_per_matrix.csv`; `results/corr_validity_summary.csv` | Compare `n_pairs_applied` with `n_pairs_declared`; this does not establish higher-order dependence recovery. |

## End-to-end cost accounting (§9.8)

Source artifacts are `results/end_to_end_cost_latency_raw.csv` (attempt and
timing rows), `results/end_to_end_cost_latency.csv` (retained per-stage rows and
totals), and `results/end_to_end_cost_latency_run_manifest.json` (run
provenance). The producer is `analysis/end_to_end_cost.py`.

| Claim | Paper location | Evidence | Re-derive or bound |
|---|---|---|---|
| One finished dataset averages **$0.01686 and 41.84 seconds**. | §4.5 (Q5); `tab:e2e` | `end_to_end_cost_latency.csv`, `source_id=mean_one_dataset`, `stage=total` | Read `api_cost_usd=.01686` and `wall_seconds=41.8383`. This is one retained workflow run per source, not one LLM call. |
| Observed per-paper ranges are **$0.01327–$0.02091** and **39.07–45.64 s**. | §4.5 (Q5) | Same file, the three source-level `stage=total` rows | Take minima and maxima of `api_cost_usd` and `wall_seconds`. |
| The mean split is **7.80 s API, 33.06 s network, and .98 s local**. | §4.5 (Q5); `tab:e2e` | Same mean-total row | Read `api_seconds`, `network_seconds`, and `local_seconds`. These timing classes can overlap only as defined by the harness; the reported wall total comes from retained stage measurements. |
| Search and fetch dominate wall time at **27.97 s and 5.09 s**. | §4.5 (Q5); `tab:e2e` | Same file, `source_id=mean_one_dataset`, stages `web_search` and `fetch` | Read `network_seconds` for each stage. |
| Local synthesis, validation, refinement, and artifact writing total about **.34 s**. | §4.5 (Q5); `tab:e2e` | Same file, corresponding mean stage rows | Sum `local_seconds`; parse is excluded because it is a separate preprocessing stage. |
| A finished dataset uses three LLM calls: query expansion, source suggestion, and extraction. | §4.5 (Q5) | Same file and manifest | Count the nonzero-`llm_calls` stages per source. This corrects the narrower extraction-only accounting. |
| Retained runs cost **$0.05058 over nine successful calls**, with 14,932 prompt and 1,325 completion tokens. | §4.5 (Q5) | Manifest plus retained stage rows | Sum retained per-call records; do not include the superseded corrective run. |
| Physical experiment spend including the corrective rerun was **$0.057425 over 12 calls**. | §4.5 (Q5) | `end_to_end_cost_latency_run_manifest.json` | Read `physical_known_cost_usd_including_replaced_run` and `physical_llm_attempts_including_replaced_run`. |
| Human source-review time is excluded. | §4.5 (Q5) | Same manifest | Read `human_source_review_time_included=false`. This bounds the latency claim. |

## Standing claim bounds

The paper's headline is intentionally limited to an **assisted, reviewable**
workflow. The evidence above does not establish unattended scientific curation,
formal privacy, generalization beyond the evaluated datasets and models, or
SDV-equivalent performance from literature summaries. The following
distinctions must remain explicit in future revisions:

1. real-paper extraction is incomplete and requires verification;
2. intrinsic fidelity measures reproduction of supplied summaries, not whether
   those summaries are correct or sufficient;
3. SDV sees real training records, unlike the summary-only arms; and
4. §9.8 cost is per retained finished dataset and excludes human review time.
