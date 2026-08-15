# Evidence-to-Claim Map

Required by PROJECT.md §9.1: *"For each abstract, introduction, and conclusion
claim, identify the experiment or analysis that supports it. Downgrade claims
that are supported only by templated inputs, data-access baselines, or
descriptive evidence."*

Every row names the claim, where it appears in `AuthorKit27/main.tex`, the
artifact that supports it, and how to re-derive the number. A claim with no
re-derivation path is not evidence-backed and is marked as such.

**Scope note.** This file was created when §9.8 was implemented, so the §9.8
rows are exhaustive and were verified directly against the released artifacts.
Rows for §9.2–§9.7 point at the records those sections produced; where a value
was transcribed from `results/README.md` rather than recomputed here, the
"Re-derive" column says so.

---

## §9.8 — Baselines and cost accounting

Source artifacts: `results/end_to_end_cost_latency.csv` (long schema),
`results/end_to_end_cost_latency_summary.csv` (per source × stage + `TOTAL`),
`results/end_to_end_cost_latency_run_manifest.json` (provenance + per-call
history). Producer: `analysis/end_to_end_cost.py`. Run: GPT-4o via OpenRouter,
3 sources, 3 synthesis seeds, `iterations=3`, 6 LLM calls, 0 failures.

| Claim | Where | Evidence | Re-derive |
|---|---|---|---|
| One finished dataset costs **$0.0122** | Q5 ¶2; Table 10 total | `..._summary.csv`, `source_id=ALL_SOURCES`, `stage=TOTAL`, `metric=api_cost_usd` → 0.0367 | Grand total ÷ 3 sources = 0.01225 |
| One finished dataset takes **44 s** | Q5 ¶2 | Same file: `wall_seconds` grand total 132.16 | 132.16 ÷ 3 = 44.05 |
| Split is **6.05 s API / 38.01 s local** | Table 10 total row | `api_latency_seconds` 18.14 and `local_compute_seconds` 114.02, grand totals | Each ÷ 3 |
| Extraction on a real paper costs **$0.0113** at **2,856 prompt tokens** | Q5 ¶2; Table 10 | Run manifest `llm_calls[]`, the three `stage="extraction"` entries: $.00787/.01368/.01245, tokens 2332/3629/2607 | Mean of the three |
| That is **~5×** the per-call cost of Table 4 | Q5 ¶2 | Table 4 OpenAI row: $0.0023 at 363 prompt tokens | 0.01133 ÷ 0.0023 = 4.93 |
| …because Table 4 used short snippets that never fill the input budget | Q5 ¶2 | `_extract_with_llm` truncates at `text[:10000]`; challenge cases are far shorter | Compare prompt tokens: 2,856 vs 363 |
| Web search dominates wall time, **≈31 s** | Q5 ¶2; Table 10 | `..._summary.csv`, `stage=search`, `metric=local_compute_seconds` per source: 31.92/30.57/32.31 | Mean = 31.60 |
| Whole synthesize–validate–refine loop is **under 0.25 s** | Q5 ¶2 | Same file, stages `synthesis` + `validation` + `refinement_overhead` | 0.029 + 0.115 + 0.058 = 0.21 s |
| Three papers together: **$0.0367**, 6 calls, 8,791 prompt / 1,477 completion tokens | Table 10 caption | Run manifest `totals` | Read directly |
| **All six LLM calls succeeded**; failed calls would be recorded with elapsed latency | App. §End-to-End Cost | Manifest `totals.llm_call_failures` = 0; `Extractor._record_failure` writes a `status="failed"` row | Read manifest; see `extractor.py` |
| Extraction here ($0.034 / 11.2 s) is consistent with Q1's **$0.0319 / 15.5 s** | `results/README.md` §9.8 | Sum the three manifest `extraction` calls | Cross-check against Q1, main.tex:392 |
| **Agent and Indep. receive only training-fold summary statistics** | Table 2 & Table 9 captions | `analysis/p0_utility_benchmark.py:82` `_gold_params_from_train` — *"Create gold summary parameters from a training split only"*; manifest key `summary_parameter_scope: "training split only"`; `extraction_method="gold_train_summary"` | Code path, not a measurement |
| **SDV is fitted on the real training records** and is a data-access upper bound | Table 2 & Table 9 captions | `analysis/p0_utility_benchmark.py:75` `_sdv_baseline(train, …)` receives the raw `train` DataFrame; called at line 202 | Code path, not a measurement |
| TRTR trains on the real training fold | Table 2 caption | Same benchmark, real-data reference arm | Code path |

**Downgrades and bounds.** SDV is labeled a data-access upper bound in both
captions and is never presented as a like-for-like competitor — PROJECT.md §9.1
requires exactly this downgrade. The per-stage numbers describe **one** dataset
per source; they are not a throughput or amortized-cost claim, and the search
stage reflects one search-engine backend on one day, so it is a measurement of
this pipeline, not a general claim about web search.

**Not claimed.** No non-iterative copula baseline was added. §9.8 floats one as
optional; it would differ from the shipped synthesizer only in the optimization
loop, which §9.7 already measures directly (and finds inert at the shipped
budget), so it would duplicate an existing result rather than isolate a new one.

---

## §9.2 — Real-paper study (Q1, Table 1)

| Claim | Where | Evidence | Re-derive |
|---|---|---|---|
| Discovery finds all three target papers (recall@5: 1, 0, 1) | Abstract; Q1 | `analysis/real_paper_discovery_summary.csv` | Read `recall_at_5` |
| Extraction reaches **.247** micro-F1 (precision .167, recall .480) | Abstract:74; Q1:390 | Frozen pre-prompt-switch run behind Table 1 | See divergence note below |
| 82.5% of LLM outputs unsupported or out of scope | Q1:390 | `analysis/real_paper_errors.csv` | Count by `error_category` |
| Three LLM calls cost $0.0319 / 15.5 s | Q1:392 | `results/real_paper_run_manifest.json` → `llm_calls` | Sum cost and latency |
| Regex extracted no target items | Q1:391 | `analysis/real_paper_extraction_summary.csv`, `regex` row: all metrics 0 | Read directly |

**Disclosed divergence — not an error.** The abstract and Q1 quote **.247**
micro-F1, while the released `analysis/real_paper_extraction_summary.csv` holds
**.3038** and the appendix ("Effect on Q1", main.tex:1162–1169) states the rise
from .247 to .308 explicitly, attributing it to the §9.6 prompt switch and noting
the re-run spread (.308 then .304). The headline retains the pre-switch value so
it matches the frozen Table 1; the appendix carries the current one. Anyone
updating the abstract must move Table 1 with it.

---

## §9.3–§9.5 — Utility, privacy, calibration (Q2)

| Claim | Where | Evidence | Re-derive |
|---|---|---|---|
| Synthetic data preserve much linear ranking signal | Abstract; Q2 | `results/p0_utility_summary.csv`; Table 2 | AUC .850/.850 Adult, .818/.819 Pima |
| …but lose minority-class and nonlinear utility | Abstract; Q2 | Same | HGB AUC deltas; Concrete $R^2$ .926 → .528 |
| Consistently stronger than independent marginals | Q2 | Same, `Indep.` column | Paired per-seed deltas in raw rows |
| Calibration recovers most minority-F1 | App. Table 9 | `results/adult_calibration.csv` | Threshold-on-synthetic-validation row |
| Oracle threshold is an upper bound, not a method | Table 9 caption | Raw rows flagged `uses_real_test_labels` | Excluded from every utility claim |
| No matched literature-grounded distribution-shift pair | Q2:447–449 | — | **Explicitly not claimed**; stated as future work |

---

## §9.6 — Extraction and prompt ablation (Q3, Q6)

Transcribed from `results/README.md` §9.6, which records the committed
930-extraction sweep; not recomputed here.

| Claim | Where | Evidence | Re-derive |
|---|---|---|---|
| Caveman buys 30.0% prompt-token reduction | Q5; Q6 | `results/extraction_challenge_summary.csv` | 404.6 → 283.3 tokens |
| …but only 3.5% cost reduction | Q5; Q6 | Same | Completion tokens dominate |
| …and costs .069 F1 (.710 vs .778) | Q6 | Same; paired Wilcoxon p=.0001 | Replicated twice (see §Replication) |
| Loss is precision, from hallucination .098 → .154 | Q6 | Same | Per-arm FP breakdown columns |
| Regex reaches F1 .156 | Q6 | Same, `regex` arm | Run once, reported separately |
| Anthropic wraps JSON in a fence on 100% of calls | App. | `fence_stripped` column | Read directly |

---

## §9.7 — Optimization and fidelity diagnostics (Q4)

Transcribed from `results/README.md` §9.7 and its follow-up.

| Claim | Where | Evidence | Re-derive |
|---|---|---|---|
| Shipped noise-pivot defaults are not empirically justified | Q4; App. | `results/noise_pivot_ablation_summary.csv` | 88.50 vs 99.88 for fixed noise 0.01 |
| `reset_multiple` is inert at the shipped budget | App. | `results/noise_pivot_knob_ranking.csv` | Effect range exactly 0.00 |
| Fidelity score is invariant to errors in extracted parameters | Q1; Discussion; App. | `results/fidelity_validity_probe_raw.csv` | Reported score constant across 14 corruptions |
| Correlation term is the exception and carries signal | App. | Same | 52–67 dropped vs 80–100 repaired |
| Effective copula rate after the naming fix: 100% | App. | `results/corr_validity_per_matrix.csv` | `n_pairs_applied` vs `n_pairs_declared` |

**Downgrade recorded.** Because the fidelity score is invariant to parameter
errors, it is described throughout as a diagnostic of *reproduced summaries*, not
as evidence of task suitability or correctness of extraction.

---

## Standing bounds on the headline claim

The abstract's framing — *"a reproducible assisted workflow with explicitly
bounded claims, not an unattended substitute for scientific curation"* — is the
claim §9.1 exists to protect. It is supported by, and must remain consistent
with:

1. extraction micro-F1 well below 1.0 on real papers (§9.2);
2. the fidelity score's measured invariance to wrong parameters (§9.7);
3. SDV never being presented as a like-for-like competitor (§9.8); and
4. cost being reported per finished dataset, not per call (§9.8).

Nothing in the repository currently claims demonstrated *autonomous*
web-to-dataset performance.
