# SPEC.md: AgentDataset Architecture

## 1. Overview

AgentDataset is an assisted pipeline: discover research documents → extract statistical parameters → synthesize a dataset → validate the result. Each phase is a standalone module; the `Orchestrator` wires them together. The CLI surfaces candidate sources and extracted parameters for review before synthesis. The proposed system direction in §10 addresses remaining gaps and limitations identified in the project audits.

> **Revision status:** §2–§9 primarily document the existing implementation and completed research. §10 is the forward-looking change plan. Its items are proposals until implemented and evaluated; do not read them as current capabilities or as claims supported by the frozen results.
>
> **2026-11-23 update:** this document is a project log, not the current paper. The
> paper draft and its result tables live in `docs/MANUSCRIPT.md`. Since this log
> was written, the Streamlit UI, the regex fallback, the frozen
> `analysis/noise_pivot_ablation.py` grid, and the `docs/neurips/` build were
> removed. Descriptions of those items below are historical.

---

## 2. Modules

### 2.1 Discovery (`core/discovery.py` and `Orchestrator.run_discovery`)

- **LLM-guided query expansion**: `Orchestrator.optimize_query()` asks the configured LLM for search-optimized queries focused on research papers, PDFs, reports, means, standard deviations, and correlations. The original query is always included; on LLM failure, only the original query is used.
- **Search**: OpenRouter's Exa-powered web search (`OPENROUTER_API_KEY`); results are parsed as JSON, deduplicated by URL, and capped. Falls back to the original query on failure.
- **Deduplication**: discovery results are deduplicated by URL.
- **Source suggestion**: `Orchestrator.suggest_sources()` asks the LLM which results are likely to contain statistical parameters; on failure, no suggestions are returned.
- **PDF fetch**: `requests.get()` streams the file to a `NamedTemporaryFile`; returns `pdf://<path>` to the caller.
- **HTML fetch**: `trafilatura` extracts clean text from the page.
- **Error handling**: Network failures are caught and logged; snippet fallback is used for PDFs that fail to download.

### 2.2 Extraction (`core/extractor.py`)

- **Extraction is LLM-only.** Calls `litellm.completion()` with a structured JSON prompt enforcing the schema below. `extraction_method = "llm"`. The regex fallback was removed; a failed call is reported, not silently replaced.
- **PDF parsing**: `pymupdf4llm` converts each page to layout-aware Markdown; PyMuPDF (`fitz`) is the fallback backend. Temp files are deleted after parsing.
- **Statistical density check**: ratio of numeric tokens to word tokens — used to assess whether a document is worth extracting from.

**LLM output schema:**
```json
{
  "variables": {
    "<name>": {"distribution": "normal|uniform|gamma", "mean": 0.0, "std": 1.0, "min": null, "max": null}
  },
  "correlations": {
    "<key>": {"var1": "<name>", "var2": "<name>", "correlation": 0.5, "direction": "positive|negative"}
  }
}
```

### 2.3 Orchestrator (`core/orchestrator.py`)

Central controller. Key responsibilities:

- **Session management**: creates `.agentdataset_cache/sessions/<run_id>/`; prunes oldest dirs beyond `MAX_SESSIONS = 3`.
- **Multi-source merging** (`merge_parameters`): when multiple sources are selected, averages same-named variables and unions unique ones; averages duplicate correlation pairs.
- **PDF dispatch**: detects `pdf://` prefix from Discovery, routes to `extractor.pdf_to_markdown()`, then deletes the temp file.
- **Optimization loop**: iterates Synthesis → Validation with a ratchet + pivot strategy (see §3).
- **Artifact saving**: best `data.csv`, `parameters.json`, and `DATACARD.md` are written to the session directory on each improvement.

### 2.4 Synthesizer (`core/synthesizer.py`)

- Generates per-variable data arrays from `VariableParams` (normal, uniform, gamma).
- Applies `noise_level` to all three distributions (uniform expands bounds symmetrically).
- Builds correlation structure via Cholesky decomposition on the correlation matrix; applies it via rank transform.
- Uses `np.random.default_rng(seed)` — instance-scoped, does not mutate global NumPy state.
- Emits a `RuntimeWarning` if the correlation matrix is not positive-definite (falls back to independent synthesis).
- Records a `CorrDiagnostics` per call on `last_diagnostics` / `diagnostics`: minimum eigenvalue, condition number, PD status, action taken, and pairs declared vs applied vs skipped. Positive-definiteness is measured with `eigvalsh`, not inferred from a Cholesky exception. **A pair naming a variable that was not extracted is skipped silently** — the resulting identity matrix is valid but imposes nothing, which §9.7 found to be the dominant way dependence is lost.
- `corr_repair` selects the invalid-matrix policy: `"drop"` (default, discard all correlations), `"psd"` (project onto the nearest PSD correlation matrix), or `"none"`. The valid path keeps using the Cholesky factor; the repaired path uses an eigendecomposition factor, which also succeeds on semidefinite matrices.

### 2.5 Validator (`core/validator.py`)

Produces a `FidelityReport` with four components:

| Component | Weight | Method |
|-----------|--------|--------|
| KS score | 40% | Fraction of variables passing KS-test (`p ≥ 0.05`), × 100 |
| Correlation score | 40% | Mean absolute error of off-diagonal correlations: `1 − mean(|synth − target|)/2`, clamped to [0, 1] (diagonal excluded so it can't inflate the score) |
| Bias score | 20% | Fraction of variables within 20% mean deviation |
| Row-dispersion diagnostic | — (reported separately) | Avg nearest-neighbour distance on a 500-row synthetic-only subsample, normalised to [0, 1]; not a privacy audit or guarantee |

Distribution CDFs used in KS-test: `stats.norm` (normal), `stats.uniform` (uniform), `stats.gamma` (gamma).

---

## 3. Optimization Loop & Noise Pivot

The loop runs for `iterations` steps. On each step:

1. Synthesize a dataset with current `noise_level`.
2. Validate → get `overall_score`.
3. **If score improves**: save artifacts, reset `no_improve_streak = 0`.
4. **If score does not improve**: increment `no_improve_streak` and pivot:

```
streak % (PATIENCE*2) == 0  → reset:    noise = initial (0.1)
streak % PATIENCE == 0      → exploit:  noise *= 0.5  (floor MIN_NOISE = 0.01)
otherwise                   → explore:  noise *= 1.1  (cap MAX_NOISE = 2.0)
```

(Checked in that order; "otherwise" = any streak not divisible by `PATIENCE`.)

`PATIENCE = 2` — so the cycle is: explore → exploit → explore → reset.

All seven schedule and bound values are exposed as `NoisePivotConfig` (with `next_noise()` as a pure transition function); the defaults reproduce the constants above exactly. **§9.7 measured the five active schedule choices and they are not optimal**: the fidelity landscape falls off sharply above noise 0.05, so the shipped `initial_noise = 0.1` starts on the wrong side of the cliff and a fixed noise of 0.01 beats the schedule at equal budget. The defaults are retained so published results stay reproducible — see §9.7 before treating them as tuned.

---

## 4. API Provider Support

Managed via `litellm`. The provider is selected in the UI; `Extractor` receives the matching `env_var` name and sets it before each LLM call.

| Provider | `env_var` | litellm model prefix |
|----------|-----------|----------------------|
| OpenAI | `OPENAI_API_KEY` | none (e.g. `gpt-4o`) |
| Anthropic | `ANTHROPIC_API_KEY` | none (e.g. `claude-sonnet-4-6`) |
| Google | `GEMINI_API_KEY` | `gemini/` (e.g. `gemini/gemini-2.0-flash`) |

---

## 5. Data Models (`models/schemas.py`)

| Model | Purpose |
|-------|---------|
| `VariableParams` | Distribution type, mean, std, min, max |
| `CorrelationParams` | var1, var2, correlation coefficient, direction |
| `MetaParams` | Source name, extraction timestamp, method |
| `Parameters` | Full parameter set (variables + correlations + meta) |
| `FidelityReport` | All scores, KS p-values, bias/privacy details, approved flag |
| `SessionContext` | Session ID, filesystem path, creation time |
| `DiscoveryResult` | Title, URL, source type, relevance score, snippet |

---

## 6. Generated Filesystem

```
.agentdataset_cache/
└── sessions/
    └── run_<timestamp>/
    ├── data.csv          # Best synthetic dataset
    ├── parameters.json   # Parameters used for best run
    └── DATACARD.md       # Fidelity + synthetic-row dispersion report
```

`.agentdataset_cache/` also holds runtime artifacts, migrated results, memory files, and live e2e reports. It is ignored by git. Only the 3 most recent session directories are retained. Older ones are deleted at `Orchestrator.__init__`.

---

## 7. Research Viability & Publishability

AgentDataset sits at the intersection of agentic AI, synthetic data generation, and statistical disclosure concerns. It offers a structured, assisted approach to generating synthetic datasets from statistics reported in academic literature.

### 7.1 Novel Contributions

- **End-to-End Assisted Workflow**: Combines discovery, LLM extraction, and synthetic generation in one workflow while requiring human verification of source scope and extracted parameters.
- **The "Noise Pivot" Optimization Loop**: An explore/exploit ratcheting system for tuning the `noise_level` against a multi-component fidelity score (KS-test, Correlation MAE, Bias).
- **Fidelity and Diagnostic Reporting**: Automatically generates a `DATACARD.md` with statistical-fidelity measures and a synthetic-row dispersion diagnostic that is explicitly not a privacy guarantee.
- **Prompt-Compression Audit**: Measures the compressed "Caveman Protocol" and shows that its modest cost saving does not justify its extraction-quality loss; the deployed extractor uses the natural-language prompt.
- **Mixed Continuous/Categorical Synthesis**: The synthesizer's Cholesky/rank-transform copula extends to categorical variables with any number of categories (e.g. classification targets, demographic features), so the pipeline can model real datasets whose columns aren't purely continuous. The copula treats category codes as ordinal (alphabetical label order in the benchmark, matching the validator's encoding), a documented modeling choice for nominal variables.

### 7.2 Potential Publication Venues

- **Machine Learning / AI Venues**:
  - *NeurIPS Datasets and Benchmarks Track*: Highly relevant for frameworks generating datasets.
  - *Innovative Applications of Artificial Intelligence (IAAI-27)*: Excellent fit for the "Emerging Applications" track, given this is a highly practical, deployable system that solves a real-world data scarcity problem using applied AI.
  - *ICLR or ICML (AI for Science / Applied AI workshops)*: For the autonomous agent architecture.
- **Data Engineering & Databases**:
  - *VLDB or SIGMOD*: Positioned as an automated data curation and synthesis pipeline.
- **Privacy & Security**:
  - *PETS (Privacy Enhancing Technologies Symposium)*: Emphasizing a disclosure-risk threat model, the limits of aggregate-only inputs, and statistically accurate proxies for sensitive domains without claiming anonymity or differential privacy.

### 7.3 Evaluation Status

All three strengthening items below are implemented as real, non-mocked measurements in `benchmark.py` (run via `python benchmark.py`, or each function individually).

1. **Empirical Benchmarks** (`run_empirical_benchmark`) — **Done.** Runs the pipeline on 3 real public datasets: UCI Adult Income (demographic), Pima Indians Diabetes (medical), and Statlog German Credit (financial). Real `describe()`/`corr()` statistics are templated into extractable "literature style" text, extracted through OpenRouter (single key covering OpenAI/Anthropic/Google, falling back to regex automatically if unavailable), synthesized, and compared via TRTR vs. TSTR logistic regression (accuracy, F1, ROC-AUC) against real held-out data, alongside the existing fidelity report. This required adding categorical-variable support to the core pipeline (see 7.1) so classification targets like income or credit risk could be modeled, not just continuous regression targets. The Adult benchmark also includes a 3-category marital-status feature, exercising multi-category (beyond binary) extraction, synthesis, and validation end-to-end.
2. **Ablation Studies** (`run_ablation_study`) — **Done.** LLM vs. regex extraction is scored against known ground truth across 8 varied domain texts (not a single snippet), with paired t-tests. The Noise Pivot optimization loop is measured across iteration counts 0/1/2/3/5/10, over multiple seeds and source texts, with paired t-test and Wilcoxon significance tests comparing 0 vs. 5 iterations.
3. **Cost/Latency Analysis** (`run_cost_latency_analysis`) — **Done.** Real litellm cost, token, and latency tracking (`completion_cost`, `usage`, `token_counter`) across OpenAI/Anthropic/Google, routed through OpenRouter with a single key, gracefully skipping providers when no key is configured. The Caveman Protocol's token savings are measured directly per model rather than asserted: the compressed system prompt is 189 tokens vs. 242 for a naturally-worded equivalent, a **21.9%** reduction (`results/caveman_token_savings.csv`), not the earlier unverified claims of 43%/73%.

This original benchmark is retained for provenance. The P0 revision in Section 9
supersedes it with 30-seed linear/nonlinear classification and regression.

---

## 8. Paper (NeurIPS 2026) — Status & Pickup Notes

**Target venue:** NeurIPS 2026, Main Track.
**Format:** official NeurIPS 2026 template, **9-page** main-content limit; acknowledgments, references, checklist, and optional technical appendices do not count toward that limit.
**Review model:** **double-blind** — the default `neurips_2026` style suppresses the author block and adds review line numbers.

### 8.1 Draft location & build
The paper lives in `docs/neurips/`; canonical sources are tracked while build artifacts are ignored:

- `docs/neurips/neurips_2026.tex` — canonical paper source using the repository's official NeurIPS 2026 template.
- `docs/neurips/references.bib` — references, including the three real-paper sources and disclosure-control literature.
- `docs/neurips/Figures/architecture.tex` → `architecture.pdf` — TikZ pipeline diagram (standalone-compiled to vector PDF, included via `\includegraphics`).
- `docs/neurips/checklist.tex` — completed official NeurIPS paper checklist, `\input` at the end of the paper.
- `docs/neurips/generated_tables/` — deterministic result-backed TeX fragments.

**Build:** from `docs/neurips/`, run `pdflatex neurips_2026` → `bibtex neurips_2026` → `pdflatex neurips_2026` ×2.
Compliance verified after migration: main text ends on page 9 and references start on page 10; the 25-page submission PDF has 0 overfull boxes, 0 undefined refs, no Type-3 fonts, all fonts embedded, and PDF 1.5.

### 8.2 Historical reviewer-readiness assessment (pre-P0)
These five-seed results are retained as provenance and are superseded by the
30-seed P0 evidence in Section 9 and `results/p0_utility_*.csv`:

1. **Error bars on empirical benchmark — DONE.** `run_empirical_benchmark` now extracts once per dataset then runs a 5-seed loop varying both the split (`random_state=seed`) and synthesis (`Synthesizer(seed=seed)`), reporting mean ± sample-std (ddof=1) for every TRTR/TSTR metric in `results/empirical_benchmark.csv`.
2. **Categorical narrow / pairwise-only — HALF DONE.** The *binary-only* limitation is **resolved** (commit `0004dd8`: arbitrary-cardinality support + 3-category Adult `marital_status`). Still true and *stated in the paper* (Discussion + appendix): only pairwise correlations (no joint >2-var structure), and a compact 3-feature-per-dataset schema.
3. **Adult F1 gap — DONE (characterized).** With 5-seed error bars: TRTR F1 0.547±0.007 vs TSTR 0.119±0.013 — stable, not noise. `analysis/adult_f1_gap.py` (→ `results/adult_f1_gap.csv`) shows the cause is **not** a distorted target marginal (synthetic `>50K` prevalence ≈ real ≈ 0.24) but **attenuated feature↔target correlation** (AgentDataset retains ~73% of the real point-biserial signal; independent-marginals flattens it to ~0 → F1 0.0), driving minority-class under-prediction at the 0.5 threshold (recall 0.46→0.07). Partly a calibration artifact: even SDV (AUC 0.852) collapses to F1 0.11 at that threshold. Written up in the paper's Evaluation (`\paragraph{Characterizing the Adult F1 gap}`) + Discussion.
4. **Caveman "measured vs claimed" honesty — DONE.** Reconciled to measured **21.9%** (189 vs 242 tokens) in both this doc and the paper; framed as measured, not asserted.
5. **No baseline synthesizer — DONE.** Added two baselines in `benchmark.py`, both run inside the same 5-seed TSTR protocol (results carry a `synthesizer` column: `agentdataset` | `independent` | `sdv_gaussian_copula`): `synthesize_independent_marginals` (deep-copies params, clears correlations, reuses `Synthesizer` — no new deps) and `synthesize_sdv_gaussian_copula` (SDV `GaussianCopulaSynthesizer` fit on real train data; **guarded import** so a missing/broken `sdv` skips gracefully; `sdv>=1.17.0` added to deps). Finding: AgentDataset beats independent-marginals where correlations matter (Adult F1 .119 vs .000; Pima AUC .822 vs .397) and trails SDV (Adult AUC .601 vs .852) — but SDV is fit on **real** records, so it's framed in the paper as a data-access upper bound, not a like-for-like competitor.

### 8.3 Prioritized next steps (pick up here)
1. **(crit #1) — DONE.** 5-seed loop + mean ± std in `run_empirical_benchmark`.
2. **(crit #5) — DONE.** Independent-marginals + SDV GaussianCopula baselines on the same TSTR protocol.
3. **(crit #3) — DONE.** Adult F1 gap analyzed (`analysis/adult_f1_gap.py`) and written up.
4. **DONE.** Folded into the paper's Evaluation: the utility table includes error bars and baseline columns in `docs/neurips/neurips_2026.tex`.
5. **Housekeeping before submission — OPEN.** Confirm the co-author list/affiliations, replace the `% TODO` camera-ready author block, and perform the final proofread.

**Historical story:** these five-seed numbers overturned the earlier single-seed narrative. The current paper no longer uses them; its conclusions come from the 30-seed benchmark and real-paper study.

**Note:** re-running steps 1–3 requires `python benchmark.py` / `python analysis/adult_f1_gap.py` (LLM API calls / compute), which regenerates `results/*.csv`; the paper's numbers/tables must then be updated to match.

---

## 9. July 2026 Reviewer Revision Plan

**Source:** `docs/review-agent-dataset.pdf`, reviewed by Dr. Justice Owusu Agyemang in July 2026.

**Reviewer position:** The system is novel, well-engineered, candid about failures, and reproducible, but the paper needs stronger evidence that the full autonomous pipeline works on real literature. The three central revision requirements are:

1. evaluate discovery and extraction on real published documents;
2. narrow or formally support the privacy claims; and
3. broaden downstream utility evaluation beyond logistic-regression classification on three datasets.

The work below is ordered by submission impact. A checkbox should be marked complete only when the implementation, saved results, paper update, and stated acceptance criteria are all complete.

### 9.1 Revision-wide rules

- [x] **Freeze the current benchmark results before new experiments.**
  - Preserve the current `results/*.csv` files or record their commit hash before regenerating them.
  - Record the Python version, dependency lockfile, date, model/provider identifiers, extraction route (`llm` or `regex_fallback`), seeds, and command used for every new experiment.
  - Never silently mix results produced by different code revisions or extraction methods.
  - **Done when:** every new table row can be traced to a command, code revision, seed set, and raw result file.

- [x] **Use one results schema and one reporting convention.**
  - Report the number of documents, datasets, runs, and seeds for every experiment.
  - Report mean, sample standard deviation, and preferably a 95% bootstrap confidence interval for downstream metrics.
  - Keep per-run observations in machine-readable CSV; do not save only aggregated values.
  - Separate pre-specified primary metrics from exploratory diagnostics.
  - **Done when:** the paper, appendix, and result files use the same sample counts and metric definitions.

- [x] **Maintain an evidence-to-claim map.**
  - For each abstract, introduction, and conclusion claim, identify the experiment or analysis that supports it.
  - Downgrade claims that are supported only by templated inputs, data-access baselines, or descriptive evidence.
  - **Done when:** there are no claims of demonstrated autonomous web-to-dataset performance unless the real-paper study succeeds.
  - **Deliverable:** `analysis/EVIDENCE_TO_CLAIM_MAP.md`, updated for the canonical NeurIPS manuscript and the final §9.8 artifacts.

### 9.2 P0 — Real-paper end-to-end study

This is the reviewer's most important request and should be completed before optional method extensions.

- [x] **Select 2–4 real published sources with manually verifiable statistics.**
  - Include heterogeneous source types, ideally:
    - a public-health surveillance or government report;
    - a clinical-trial paper or results table;
    - a social-science paper with descriptive statistics;
    - optionally, a finance/economics report.
  - Prefer openly accessible, stable URLs and documents that report enough marginals and dependence information to synthesize a useful dataset.
  - Avoid selecting only unusually clean documents; the study should include realistic tables, footnotes, multi-column layouts, or contextual ambiguity.
  - Record document title, publisher, URL/DOI, publication year, access date, license/access status, page/table locations, and why it was selected.
  - **Deliverable:** `analysis/real_paper_manifest.csv`.
  - **Done when:** the selected set covers at least three document/layout styles and can be redistributed or precisely cited.

- [x] **Create a manually verified extraction gold standard.**
  - Two passes should transcribe each usable statistic directly from the source.
  - For each statistic, record:
    - normalized variable name and the exact source label;
    - statistic type (mean, standard deviation, bound, category probability, correlation, sample size, etc.);
    - value and units;
    - population, subgroup, time period, and experimental arm;
    - page, table/figure, and surrounding textual evidence;
    - whether the statistic is directly reported or derived.
  - Resolve ambiguous or conflicting values explicitly rather than choosing one silently.
  - If only one annotator is available, perform a delayed second pass and document that limitation.
  - **Deliverables:** `analysis/real_paper_ground_truth.csv` and a short annotation protocol.
  - **Done when:** every scored gold item is traceable to a source location and contextual scope.

- [x] **Evaluate discovery separately from extraction.**
  - Start from realistic user queries that do not contain the exact paper title.
  - Run `Orchestrator.run_discovery()` and save all returned ranks, URLs, snippets, source types, deduplication outcomes, and LLM suggestions.
  - Measure:
    - target document found/not found;
    - rank of first relevant document;
    - Recall@k and Mean Reciprocal Rank where the sample permits;
    - fetch success;
    - PDF/HTML parse success;
    - time and cost.
  - Record search date because web results are time-dependent.
  - Run a direct-URL condition separately so discovery failures do not prevent extraction evaluation.
  - **Deliverable:** `analysis/real_paper_discovery.csv`.
  - **Done when:** the paper distinguishes retrieval, download, parsing, and extraction failures.

- [x] **Evaluate extraction against the gold standard.**
  - Run both the LLM extractor and regex fallback on exactly the same parsed document text.
  - Score more than “success/failure”:
    - precision, recall, and F1 for identifying statistics;
    - exact or tolerance-based numeric accuracy;
    - variable-name matching;
    - statistic-type accuracy;
    - context/scope accuracy;
    - correlation-pair accuracy;
    - unsupported-value or hallucination rate.
  - Define numeric tolerances before running the comparison.
  - Count parsing failures and empty outputs rather than excluding them.
  - Add error categories for wrong population, wrong arm, wrong time point, negation, unit mismatch, table association, and conflicting values.
  - **Deliverables:** `analysis/real_paper_extraction.csv` and `analysis/real_paper_errors.csv`.
  - **Done when:** every false positive and false negative can be assigned an error category.

- [x] **Run synthesis and validation from the real-paper extractions.**
  - Synthesize from:
    - manually transcribed gold statistics, to isolate synthesis quality;
    - automatically extracted statistics, to measure full-pipeline performance.
  - Report fidelity scores, correlation errors, warnings, invalid/non-positive-definite correlation matrices, fallback-to-independence events, and optimization traces.
  - Where a compatible individual-level evaluation dataset exists, run TSTR against it.
  - Where no compatible microdata exists, report fidelity to published statistics and clearly state that downstream utility cannot be validated.
  - Do not imply the source paper's participants were reconstructed.
  - **Deliverables:** per-source result CSVs, optimization traces, generated data cards, and a consolidated `results/real_paper_pipeline.csv`.
  - **Done when:** the effect of discovery/extraction errors on the final dataset is visible rather than hidden by using gold parameters only.

- [x] **Add the real-paper study to the paper.**
  - Add a concise main-paper table with discovery, parsing, extraction, and synthesis outcomes.
  - Put source-level details, annotation rules, error examples, and full metrics in the appendix.
  - Update the abstract and conclusion only if the results support stronger claims.
  - Discuss failures as operating-boundary evidence, not merely implementation bugs.
  - **Done when:** a reader can assess whether “autonomous” applies to the demonstrated system and under which document conditions.

### 9.3 P0 — Privacy claims and disclosure-risk analysis

- [x] **Audit and narrow all privacy language.**
  - Search the paper, README, UI, data card, and project documentation for claims such as “privacy-preserving,” “no member to infer,” or equivalent guarantees.
  - Distinguish:
    - no direct access to individual-level records;
    - reduced conventional record-memorization risk;
    - possible disclosure of aggregate or group attributes;
    - absence of a formal differential-privacy guarantee.
  - Reframe the nearest-neighbor value as a diagnostic, not a formal privacy score or proof.
  - **Likely files:** `docs/neurips/neurips_2026.tex`, `README.md`, `agentdataset/core/validator.py`, generated `DATACARD.md` text, and UI copy.
  - **Done when:** no text equates aggregate-only input with guaranteed privacy.

- [x] **Add a threat model and limitations subsection.**
  - Define the adversary, available published statistics, auxiliary knowledge, protected information, and the intended release setting.
  - Discuss at least:
    - small-cell and rare-category disclosure;
    - group-attribute inference;
    - differencing/composition across multiple publications;
    - reconstruction from rich or sufficient statistics;
    - the fact that public inputs may already encode sensitive group facts.
  - State what membership inference means when the generator has no record-level training set, while avoiding the claim that all privacy risk disappears.
  - **Done when:** the paper says precisely which risks the architecture avoids and which it does not.

- [x] **Add statistical disclosure-control related work.**
  - Include the reviewer-requested foundational references:
    - Hundepool et al., *Statistical Disclosure Control* (2012);
    - Rubin (1993) on synthetic data;
    - Taub et al. (2018) on attribute disclosure from synthetic data.
  - Add recent primary literature only after checking the original source and relevance.
  - Explain how aggregate-statistics synthesis relates to reconstruction and disclosure limitation rather than adding citations as an isolated list.
  - **Likely files:** `docs/neurips/references.bib`, related work and limitations in `docs/neurips/neurips_2026.tex`.
  - **Done when:** privacy framing is connected to the established disclosure-control literature.

- [x] **Decide whether formal differential privacy is in scope for this revision.**
  - **Minimum acceptable revision:** carefully scoped claims plus a threat model and limitations.
  - **Optional stronger extension:** define neighboring inputs, bound the sensitivity of each supported statistic, add calibrated noise, compose privacy loss across statistics, and measure the privacy–utility trade-off.
  - Do not claim differential privacy by adding arbitrary noise to extracted statistics; the adjacency definition, sensitivity, mechanism, and composition accounting must be explicit.
  - **Done when:** the paper either provides a defensible formal mechanism or explicitly lists it as future work.

- [x] **Repair the data-card terminology.**
  - Rename the current nearest-neighbor “privacy score” to a neutral label such as “synthetic-row dispersion diagnostic.”
  - Explain that it measures spacing among generated rows, not distance from private training records, because no such records are available to the validator.
  - Add warnings for small sample sizes, mixed-scale features, categorical encodings, and sensitivity to normalization.
  - Add/update validator tests for the new labels and explanation.
  - **Likely files:** `agentdataset/core/validator.py`, `tests/test_validator.py`, paper method/appendix.
  - **Done when:** a generated data card cannot be mistaken for a privacy audit or guarantee.

### 9.4 P0 — Broaden downstream utility evaluation

- [x] **Add at least one nonlinear classifier.**
  - Preferred: gradient-boosted trees (XGBoost if dependency/licensing constraints are acceptable; otherwise HistGradientBoosting or an equivalent documented choice).
  - Use the same train/test splits, seeds, preprocessing, class labels, and TRTR/TSTR protocol as the logistic-regression benchmark.
  - Report accuracy, minority-class F1, ROC-AUC, and calibration-sensitive metrics where probabilities are available.
  - Explain whether nonlinear models recover utility lost by attenuated pairwise relationships or expose additional higher-order deficiencies.
  - **Likely file:** `benchmark.py`, with tests in `tests/test_benchmark.py`.
  - **Done when:** logistic and nonlinear results are directly comparable for all classification datasets.

- [x] **Add at least one regression task.**
  - Select a public dataset whose published-style summary statistics match the supported marginal/correlation schema.
  - Predefine the target, features, train/test split, transformations, and evaluation metrics.
  - Report at least MAE, RMSE, and R² for TRTR and TSTR, including the independent-marginals and real-data SDV upper-bound baselines where meaningful.
  - Ensure the source-text template or real-paper statistics do not leak test-set information.
  - **Deliverable:** regression rows in `results/empirical_benchmark.csv` or a clearly linked companion file.
  - **Done when:** the paper's utility conclusions are no longer classification-only.

- [x] **Increase the number of runs or use bootstrap intervals.**
  - Five seeds are insufficient for strong inferential claims.
  - Prefer at least 20–30 seeds if compute permits; otherwise retain five seeds but use bootstrap confidence intervals and explicitly describe the low power.
  - Do not pool heterogeneous texts or datasets as if they were interchangeable independent replicates.
  - Report effect sizes and intervals, not only p-values.
  - **Done when:** the statistical-testing section accurately reflects the experimental unit and uncertainty.

- [x] **Add a distribution-shift condition if feasible.**
  - Use literature statistics from one population/time period and evaluate against a distinct target population/time period.
  - Label this as a stress test, not a fair in-distribution utility comparison.
  - Quantify shifts in marginals, target prevalence, and feature–target relationships before interpreting TSTR results.
  - **Done when:** the paper explicitly shows how literature-to-target mismatch affects utility, or clearly lists this experiment as future work if no defensible dataset pair is available.

### 9.5 P1 — Complete the Adult calibration story

- [x] **Implement post-hoc calibration and threshold evaluation.**
  - Keep the default 0.5-threshold result as the primary unadjusted comparison.
  - Compare:
    - threshold selected on synthetic validation data;
    - Platt scaling;
    - isotonic regression where sample size supports it.
  - Never tune on the real held-out test set.
  - Report minority-class precision, recall, F1, predicted-positive rate, Brier score, and ROC-AUC.
  - Include confidence intervals across seeds and compare AgentDataset, independent marginals, and SDV.
  - **Likely files:** extend `analysis/adult_f1_gap.py` or add `analysis/adult_calibration.py`; save `results/adult_calibration.csv`.
  - **Done when:** the paper shows how much of the F1 loss is recoverable without real validation labels and how much remains structural.
  - **Result.** `analysis/adult_calibration.py` (→ `results/adult_calibration_raw.csv`,
    `results/adult_calibration.csv`, `results/adult_calibration_run_manifest.json`) runs eight
    strategies × four sources × two models over the same 30 paired splits as the P0 utility
    benchmark. The `uncalibrated_full_fit` strategy reproduces the P0 protocol and reconciles
    exactly (LR/AgentDataset minority-F1 .3811, TRTR .5417, SDV .1118), confirming the split and
    encoding are shared.
    - **Most of the Adult minority-F1 gap is a decision-threshold artifact.** Selecting the
      threshold on a validation split of the *synthetic* training frame cuts AgentDataset's LR
      deficit from −.161 [−.165, −.156] to −.008 [−.011, −.005] (95% of the gap) and HGB's from
      −.227 to −.032 [−.037, −.028] (86%). No real labels are used.
    - **Probability reshaping alone does nothing.** Platt (−.162) and isotonic (−.182) at the
      default 0.5 threshold leave the LR gap intact; the recovery comes from the operating point,
      not the calibration map. Uncalibrated synthetic-trained models predict positive at .086
      against a true prevalence of .239.
    - **The residual is structural and equals the ranking deficit.** An oracle threshold fitted on
      the real test labels (reported only as an upper bound, flagged `uses_real_test_labels=1`)
      still leaves HGB −.025 below TRTR — exactly its ROC-AUC loss — and independent marginals
      −.192 below, since thresholding cannot restore dependence that was never synthesized
      (AUC .509). This is the concrete link to the pairwise-dependence limitation in §9.7.
    - Leakage is prevented structurally: `_select_f1_threshold` and `_fit_calibrator` take only
      validation arrays. `tests/test_adult_calibration.py` asserts that flipping every real test
      label leaves all non-oracle thresholds and predicted-positive rates unchanged while the
      oracle threshold moves, so the guard cannot pass vacuously.
    - Paper text and the appendix table are applied to the draft; the blocks and their
      revisions are integrated in the canonical NeurIPS paper source (see "Paper material" below).

### 9.6 P1 — Extraction robustness and prompt ablations

- [x] **Build a messy/ambiguous extraction challenge set.**
  - Include negated findings, multiple populations, conflicting values across sections, repeated variables at different time points, units, confidence intervals, table footnotes, scientific notation, and irrelevant nearby numbers.
  - Prefer real excerpts that can be legally quoted in short form; supplement with clearly labeled constructed examples.
  - Annotate expected output and “should not extract” values.
  - Keep challenge cases separate from development examples to reduce overfitting.
  - **Deliverables:** challenge-set fixture plus annotation metadata under `analysis/` or `tests/fixtures/`.
  - **Done when:** the benchmark distinguishes correct extraction, abstention, and context errors.
  - **Done:** `analysis/extraction_challenge_set.json` — 30 cases, three per taxonomy
    category, 11 real and 19 constructed. Real excerpts are *generated*, not typed:
    `analysis/build_extraction_challenge_set.py` declares anchor pairs and slices the
    cached sources verbatim, recording character offsets and the source sha256, so the
    fixture cannot drift from the papers. Constructed cases all open with "In this
    constructed …" and are never presented as real literature (enforced by test).
  - Gold values reuse the dual-pass adjudicated `analysis/real_paper_ground_truth.csv`
    rather than being re-annotated. Annotation rules, the scoring contract, and the
    known limitations are in `analysis/EXTRACTION_CHALLENGE_PROTOCOL.md`.
  - Abstention is a first-class outcome: `benchmark._score_challenge_case` gives
    precision 1.0 for emitting nothing and recall 1.0 on a gold-empty case, so a case
    whose correct answer is silence scores F1 = 1.0 and is penalised only for speaking.
    Three cases (`neg_001`, `time_001`, `sdse_003`) have silence as the correct answer.

- [x] **Compare LLM, regex, and prompt styles on the same challenge set.**
  - Compare the Caveman Protocol with the natural-language prompt using identical model, temperature, documents, and repetitions.
  - Measure accuracy, precision/recall/F1, malformed-output rate, hallucination rate, input/output tokens, latency, and cost.
  - Use multiple LLM repetitions to expose nondeterminism.
  - Test the regex fallback separately rather than treating fallback outputs as LLM results.
  - **Likely file:** extend `run_ablation_extraction_method()` in `benchmark.py`.
  - **Done when:** the paper can state whether the 21.9% prompt-token reduction changes extraction quality.
  - **Done:** `benchmark.run_extraction_challenge()` (dispatched from
    `run_ablation_extraction_method()` when a challenge set is supplied, so the legacy
    synthetic-text ablation and `results/ablation_extraction_method.csv` are unchanged),
    driven by `analysis/extraction_ablation.py`. Three arms × 3 OpenRouter-routed models
    × 5 repetitions; regex runs once and is reported separately.
  - **Prompt-parity correction.** `_VERBOSE_PROMPT` omitted the `categorical`
    distribution and the `categories` field, so the 21.9% figure was measured against a
    baseline that requested strictly less than `CAVEMAN_PROMPT` — a strawman that
    *understated* the saving. Against a schema-matched baseline the reduction is
    **38.6%** (308 → 189 tokens). `test_prompt_schema_parity` now blocks the two prompts
    from diverging again.
  - Failures are recorded (`api_error=True`), not dropped, so they cannot bias the rates
    downward. `response_format=json_object` is kept at its production setting, so the
    malformed rate is schema-level only — stated as a limitation. Retries are disabled so
    nondeterminism is visible.
  - **Answer: the compression is not free.** Over 930 extractions (30 cases × 2 prompts ×
    3 models × 5 reps, $2.17, zero API errors), the Caveman prompt reaches F1 .710
    [.657,.760] against .778 [.729,.825] for the schema-matched natural-language prompt —
    a paired deficit of −.069 (Wilcoxon p=.0001, n=90), in the same direction on all three
    models. Recall is near-ceiling for both (.960 vs .989); the loss is precision
    (.639 vs .720), driven by hallucination rising from .098 to .154 (paired p=.0088).
    Meanwhile the 30.0% prompt-token saving yields only a 3.5% cost saving, because
    completion tokens dominate. Recommendation: use the natural-language prompt for
    extraction; the Caveman Protocol's value is schema discipline, not token count.
  - Regex reaches F1 .156 — its patterns need literal "mean … std" phrasing and almost
    never fire on real tables. Its precision of 1.000 is an artifact of near-total silence,
    which also gives it the best abstention (3/3 silence cases correct, vs 11% for
    natural-language and 0% for Caveman). Neither LLM arm reliably declines to answer.
  - Real excerpts are much harder than constructed ones (F1 .550 vs .803, hallucination
    .331 vs .052 for the Caveman arm) — direct evidence that synthetic evaluation text
    overstates extraction quality. By category, table footnotes (.911) and scientific
    notation (.852) are solved; negated findings (.463) and repeated timepoints (.575) are
    not, and the failure is judgement, not parsing.
  - **Acted on: production switched to the natural-language prompt.**
    `agentdataset/core/extractor.py` now defines `NATURAL_LANGUAGE_PROMPT` and
    `EXTRACTION_PROMPT = NATURAL_LANGUAGE_PROMPT`; `_extract_with_llm` defaults to it.
    `CAVEMAN_PROMPT` is retained (the paper documents it, the ablation still measures
    it) and marked superseded. `benchmark._VERBOSE_PROMPT` is now an alias of the
    shared constant so the token-savings comparison cannot silently drift, and
    `_prompt_for_arm` names both texts explicitly so the ablation stays reproducible
    regardless of which is default. Current cost/latency under the shipped prompt is in
    `results/extraction_cost_latency.csv`: Google $0.0006/2.15 s, OpenAI $0.0023/2.00 s,
    Anthropic $0.0042/3.95 s.
  - **Frozen-baseline invariant restored and now enforced.** `baseline_manifest.json`
    states that new experiments "must not overwrite" the six pre-revision files, but
    nothing checked it, and this revision regenerated `cost_latency.csv` and
    `caveman_token_savings.csv` in place, destroying two historical baselines. Both are
    restored to their manifest hashes; `run_cost_latency_analysis()` now writes
    `extraction_cost_latency.csv` and `extraction_prompt_token_savings.csv` instead; and
    `tests/test_baseline_manifest.py` verifies all six hashes and asserts the producer
    no longer targets those filenames.
  - Paper blocks applied to the draft; their final text now lives in `docs/neurips/neurips_2026.tex` (see below).

### Paper material is tracked with the repository

The canonical NeurIPS manuscript, official style/checklist, bibliography, figure
source, and generated result-table fragments live in `docs/neurips/`. Auxiliary
LaTeX files and rendered PDFs remain ignored; quantitative fragments are rebuilt
and checked by `analysis/paper_artifacts.py`.

These four files previously lived under `analysis/`. They were moved because they
carry verbatim draft prose, which does not belong beside the public code. Both
snippet sets are already applied to the current draft; rerunning either applier
reports "Already applied" and changes nothing.

Numbers quoted in the paper are transcribed from `results/`, which stays here.

- [x] **Follow-up from the prompt switch — regenerate prompt-dependent artifacts.**
  - Regenerated with `analysis/real_paper_eval.py --mode cached`, which reuses the
    cached source text so the papers cannot shift underneath the comparison and only
    the prompt changes. `p0_utility_*` and `adult_calibration_*` are unaffected — they
    compute parameters from data frames, not from LLM extraction. Discovery is
    untouched (it uses `llm_call`, which has no system prompt).
  - **Integrity check passed:** the `gold` condition is bit-identical across all 18
    summary metrics, confirming the seeded synthesis is deterministic and that only
    extraction changed.
  - **The P0 evaluation independently replicates the §9.6 finding.** Real-paper
    extraction F1 rose .247 → .308 and precision .167 → .226, with recall unchanged at
    .480 — the same precision-only signature the challenge set showed
    (.639 → .720 precision, .960 → .989 recall). The mechanism is visible in the item
    counts: false positives roughly halved (cdc 27 → 12, calerie 21 → 17) while true
    positives were identical (2, 0, 10). The natural-language prompt wins by inventing
    less, not by finding more. Two independent evaluations, same conclusion.
  - Absolute numbers stay poor (F1 .308, unsupported/out-of-scope .801), consistent
    with the challenge set's real-vs-constructed gap (F1 .550 vs .803). Real papers are
    genuinely hard and the paper should not imply otherwise.
  - **Downstream synthesis shifted, and not uniformly.** With roughly half as many
    extracted variables, the `llm` condition synthesizes a smaller parameter set:
    fidelity 97.6 → 95.8 (calerie) and 64.3 → 61.5 (cdc), cdc ks 11.3 → 4.4, calerie
    correlation 96.7 → 91.1. These are not quality regressions in the extraction —
    they measure how the synthesizer behaves on a different (smaller, more accurate)
    parameter set.

- [x] **Investigate new synthesizer fallback warnings on `calerie_rct` — did not reproduce.**
  - `fallback_warning_count` was 0 → 3 for the `llm` condition on `calerie_rct` in the
    first post-switch run. On an independent re-run it is **0.0 for all three sources**.
  - So the warnings track the specific parameter set a given extraction happened to
    produce — which varies run to run under LLM nondeterminism — not a structural
    problem with parameter shape, as first suspected. Closed as transient rather than
    left open on the strength of a single observation.
  - Worth noting as a general lesson: a single run of an LLM-backed pipeline is not
    enough to distinguish a systematic defect from sampling noise.

- [x] **Regenerate §9.6 and P0 with clean provenance, and replicate.**
  - **The regeneration was not needed for correctness.** An audit confirmed the fixture
    sha256 still matched, both prompt texts were byte-identical across the move into
    `extractor.py` (1286 and 624 chars), and `_score_challenge_case`, `_numbers_in_text`
    and `run_extraction_challenge` had zero diff hunks since the sweep. `_prompt_for_arm`
    pins both arms to fixed texts, so the ablation never depended on which prompt was
    production.
  - **It surfaced a real defect instead.** Every analysis entry point called
    `collect_provenance()` *after* writing its CSVs, so `git_dirty: True` was structural —
    the run's own outputs were in the tree by the time provenance was recorded, and no
    run could ever report a clean checkout. Fixed in all four entry points; guarded by
    `tests/test_baseline_manifest.py::test_provenance_is_captured_before_outputs_are_written`.
    Both regenerated manifests now record `git_dirty: False` at a commit in history.
  - **The ablation replicates.** Run 1 preserved as `extraction_challenge_raw_run1.csv`.
    Caveman-minus-natural-language F1 gap: −.0686 (run 1) vs −.0629 (run 2), both
    significant (Wilcoxon p=.0001, p=.0004). Pooled: caveman F1 .710 → .719, natural
    .778 → .782; hallucination .154 → .154 and .098 → .094; recall near-ceiling in both.
    The regex arm is bit-identical across runs (max |ΔF1| = 0.0), confirming the spread
    is LLM-side. The fenced-JSON finding is exact: `fence_stripped` is 1.0 for all 255
    successful Anthropic calls and 0.0 for all 600 OpenAI and Google calls.
  - **P0 replicates.** Extraction F1 .247 (Caveman) → .308 (run 1) → .304 (run 2);
    precision .167 → .226 → .222; recall flat at .480 in all three. Predicted items
    fell 29 → 14 → 12 on cdc with true positives constant at 2. The `gold` condition is
    bit-identical in both regenerations, confirming the seeded synthesis is deterministic.
  - **One difference, reported not smoothed:** run 2 hit 45 `APIError` rows on Anthropic
    (15% of its calls) where run 1 hit zero. Retries are disabled by design so provider
    instability stays visible; the failures are recorded as rows, so they shrink the
    paired sample to 85 rather than biasing the rates. Within-(case, model) F1 std
    roughly doubled (.009 → .018 caveman, .007 → .012 natural), consistent with that.
    Anthropic's usable-call rate was 85%.

### 9.7 P1 — Synthesis and optimization diagnostics

- [x] **Analyze noise-pivot sensitivity.**
  - Vary `PATIENCE`, initial noise, exploration multiplier, exploitation factor, reset interval, and iteration budget using a predeclared grid or efficient design.
  - Report best score, final score, convergence rate, variance, runtime, and frequency of resets.
  - Compare the existing schedule with simple fixed-noise and/or random-search baselines.
  - Explain whether hyperparameters were tuned on the same datasets used for final reporting.
  - Convert the current “use 5+ iterations” observation into a supported operational recommendation.
  - **Likely files:** `benchmark.py`, `agentdataset/core/orchestrator.py` if configuration is exposed, and optimizer tests.
  - **Done when:** the chosen defaults are justified empirically and the high 10-iteration variance is explained.
  - **Result.** Schedule exposed as `NoisePivotConfig` + a pure `next_noise()` in `agentdataset/core/orchestrator.py`; defaults reproduce the shipped constants bit-for-bit. `analysis/noise_pivot_ablation.py` ran 20,900 trials (95 conditions × 11 sources × 20 seeds, 124,960 synthesize+validate steps) → `results/noise_pivot_ablation_raw.csv`, `noise_pivot_traces.csv`; `analysis/noise_pivot_report.py` → `noise_pivot_tests.csv`, `_budget.csv`, `_variance.csv`. Sources carry a predeclared tune/report split; **every conclusion below replicates on the held-out report split with near-identical effect sizes**, so the defaults were not chosen on the reporting data.
  - **The defaults are not justified — the data says the opposite.** The fidelity landscape is monotone decreasing in noise with a cliff between 0.05 and 0.1 (mean per-iteration score: 97.8 at noise 0.01, 93.2 at 0.05, 70.8 at 0.1, 62.0 at ≥0.2). The shipped `initial_noise=0.1` sits on the wrong side of that cliff, and the incumbent spends 890 of 1,100 logged iterations in the (0.09, 0.11] band scoring ~69.
  - **A fixed low noise beats the adaptive schedule outright.** At the shipped 5-iteration budget: incumbent 88.50 (across-seed SD 10.0) vs `fixed_noise=0.01` **99.88 (SD 0.07)**, Δ +11.47, paired Cohen's d 1.09, Wilcoxon Holm-adjusted p 3.9e-16 (tune) / 3.5e-19 (report). Budget-matched random search also beats it (99.49, Δ +11.23). Even **zero** iterations at noise 0.01 beats the incumbent's five-iteration run by +10.09. The loop is largely compensating for a bad default starting noise.
  - Of the single-factor changes, only `initial_noise=0.05` (+11.44, d 1.09), `patience=1` (+7.57), and `exploit_factor=0.25` (+1.67) help; `explore_mult` and `reset_multiple` changes are not significant. All point the same way: get to low noise faster, and stop exploring upward.
  - **Knob ranking** (`results/noise_pivot_knob_ranking.csv`, effect range of the per-level mean): `initial_noise` 37.50 ≫ `patience` 16.18 > `exploit_factor` 9.18 > `explore_mult` 2.91 ≫ `reset_multiple` **0.00**.
  - **The reset branch is dead code at the shipped budget.** `reset_multiple` moves the mean best score by *exactly* zero across all four arms, including the one that disables reset entirely: at five iterations with `patience=2` a run cannot accumulate the streak the reset branch requires. The paper's Method paragraph and `alg:pivot` both describe reset as one of three behaviours; at the shipped settings it never fires.
  - **Independently corroborated.** A separate implementation of this item (different script, grid, and levels; preserved as `results/noise_pivot_knob_ranking_replication.csv`) reproduces the ordering and the magnitudes: `initial_noise` 37.03, `patience` 15.57, `exploitation_factor` 9.33, `exploration_multiplier` 3.69, both selecting the lowest available initial noise. It reports 7.93 for the reset interval rather than 0.00 because its screen used a longer budget where reset can fire — consistent with the dead-branch finding. Caveat: both score against the same fidelity objective, which is invariant to parameter error, so this corroborates the ordering *for that objective* only.
  - **The 10-iteration variance is explained, and it is not instability.** Outcomes are a two-point mixture: a run either accumulates enough exploit (halving) steps within its budget to cross the noise cliff, scoring ~100, or it does not, scoring ~60. Nothing lands in between — the frozen `results/ablation_noise_pivot.csv` at 10 iterations ranges [59.94, 100.00] with 21 distinct values in 40 cells. Across-run SD is therefore the spread of a Bernoulli mixture, **maximal at the budget where about half the runs have crossed** and collapsing on either side. In the paper's configuration that midpoint is 10 iterations (hence SD 17.31); in this study's configuration it is ~5 (share reaching ≥95: 43.6% at 5 iterations, 95.5% at 10), and SD falls monotonically 12.0 → 10.7 → 3.7 → 0.69 across budgets 0 → 5 → 10 → 20. The old draft's reading — "the schedule occasionally discovers a much better noise regime" — attributes to luck what is a deterministic descent crossing a threshold.
  - **Why the original ablation looked stochastic at all.** `benchmark.run_ablation_noise_pivot` calls `np.random.seed(seed)`, which seeds the legacy global RNG, while `Synthesizer` draws from an instance-scoped `np.random.default_rng`. Verified empirically: changing the global seed leaves synthesis byte-identical. Because that function shares one `Synthesizer` across every cell, the `seed` column is a label rather than a control, and the observed variation is the shared generator's stream position — which advances by different amounts depending on the iteration count of the preceding cells. `results/ablation_noise_pivot.csv` stays frozen; the new study seeds a fresh `Synthesizer` per trial.
  - **Operational recommendation, replacing "use 5+ iterations":** lower the initial noise to ≤0.05 (or expose it), where a 1–3 iteration budget already reaches ~99.8. Iteration budget is a weak and expensive substitute for a correct starting noise; keep 5 only as a safety margin if the initial noise stays at 0.1.
  - **Not changed:** the shipped defaults are left untouched in this change so the frozen baselines and every published number stay valid. Changing `initial_noise` is a behaviour change that should be its own decision, taken with the paper's numbers regenerated.

- [x] **Measure correlation-matrix validity and fallback behavior.**
  - Log the minimum eigenvalue, positive-definite status, repair/fallback action, and source of every correlation matrix.
  - Report frequency by templated inputs, real-paper gold statistics, and automatic extraction.
  - Evaluate a documented nearest-positive-semidefinite repair instead of immediately dropping all correlations, but retain the current behavior as a baseline.
  - Add unit tests for slightly invalid, severely invalid, incomplete, and contradictory matrices.
  - **Likely files:** `agentdataset/core/synthesizer.py`, `tests/test_synthesizer.py`, benchmark outputs.
  - **Done when:** the paper quantifies how often the advertised copula path is actually used.
  - **Result.** `Synthesizer` now records `CorrDiagnostics` per matrix (minimum eigenvalue, condition number, PD status, action, pairs declared/applied/skipped/overwritten) and measures positive-definiteness with `eigvalsh` instead of inferring it from a Cholesky exception. `analysis/corr_validity_audit.py` sweeps 26 correlation-bearing specs across templated, gold, auto-extracted, and constructed stress inputs → `results/corr_validity_per_matrix.csv`, `_raw.csv`, `_summary.csv`, `_run_manifest.json`.
  - **The premise of this item turned out to be wrong, in a way worth reporting.** Non-positive-definiteness is *not* how the copula path gets bypassed on real inputs: every templated, gold, and auto-extracted matrix in the sweep is positive-definite (PD rate 100%; non-PD occurs only in the 5 constructed stress cases, 4/5). The actual mechanism is **name resolution**. A declared correlation whose variables are absent from the extracted variable dictionary is dropped silently at `synthesizer.py`, leaving an identity matrix that passes every validity check and imposes nothing.
  - **Effective copula rate as first measured** (pairs applied / declared): **auto-extracted 0/8 = 0%**, templated 11/20 = 55%, gold 3/4 = 75%. The regex extractor named continuous variables positionally (`var_1`, `var_2`, …) while emitting correlations under the names found in the text, so no regex-extracted pair could ever resolve.
  - **Fixed.** `_extract_with_regex` now recovers the variable name from the prose immediately preceding a mean/std hit (`_PATTERN_NAME_BEFORE_STAT`, an 80-character window anchored to the statistic) and normalizes names on both sides (`_normalize_variable_name`, "heart rate" ↔ `heart_rate`). Where the prose supplies no confident name the positional `var_N` fallback is retained, deliberately: a *wrong* name mismatches silently, which is worse than a visibly unresolved one. The stopword guard is restricted to grammatical filler for the same reason — excluding a content noun such as "study" does not produce no-name, it produces the wrong name (`study hours` → `hours`).
  - **After the fix:** auto-extracted **8/8 = 100%**, templated **20/20 = 100%**, gold unchanged at 3/4 (that pair names a variable the source genuinely never reports, which is correct behaviour). On the eight templated domains the correlation score rises **77.3 → 99.3** and overall fidelity **50.9 → 59.7**, consistent across all eight.
  - **Blast radius is contained.** `benchmark._score_extraction` matches extracted variables to ground truth by nearest mean rather than by name, so Q3/§9.6 extraction F1 is unchanged. `results/real_paper_pipeline_raw.csv` contains only the `gold` and `llm` conditions — regex extracted no target items from the real papers, so Q1 is unchanged too. `_parse_llm_result` now carries a **fallback-only** resolution (see below) that cannot move published numbers.
  - Guarded by `tests/test_corr_validity_audit.py::test_regex_correlations_resolve_against_extracted_variables` plus eight naming tests in `tests/test_extractor.py` covering both prose forms, the collision case, the adjacency limit, and the filler-rejection case.
  - **LLM path: fallback-only name resolution.** `_parse_llm_result` had the same structural gap — `var1`/`var2` were taken verbatim with no cross-check — and `NATURAL_LANGUAGE_PROMPT` never states that they must be keys of `variables`. The likelier failure there is not misspelling (models are self-consistent within a response) but the common case of a paper reporting a correlation whose marginals it never reports.
  - `_resolve_correlation_name` therefore resolves **only when the literal name matches nothing**, and **only when the normalized form identifies exactly one** declared variable. An exact hit is returned untouched, so a self-consistent response is passed through byte-for-byte and no previously reported number can move — pinned by `test_llm_exact_names_pass_through_unchanged`. Ambiguous names (`BMI` vs `bmi`) are deliberately left unresolved rather than guessed, since merging distinct columns is worse than a visible skip. Unresolvable names are preserved, not dropped, so `n_pairs_skipped_unknown_var` still records them.
  - **Not done, deliberately:** the prompt is unchanged. Requiring `var1`/`var2` to be declared keys is the correct fix at source, but it changes extraction output and so regenerates Q3 and Q6 — a real cost for an unmeasured benefit this close to submission. Frequency is currently unmeasurable offline: no raw LLM output is cached, so quantifying it needs an API run. Do both together after submission.
  - Nearest-PSD repair (eigenvalue clipping + diagonal renormalisation, Higham's first iterate) added as **opt-in** `Synthesizer(corr_repair="psd")`; drop-all remains the default so the six frozen hashes in `results/baseline_manifest.json` stay valid. On the slightly-invalid stress case the repair recovers correlation fidelity that dropping destroys. The valid path deliberately keeps the Cholesky factor: the eigendecomposition factor is equally correct but produces different samples, which would silently invalidate every frozen baseline.
  - Unit tests added for all four required cases in `tests/test_synthesizer.py` (slightly invalid r=(.9,.9,.61), severely invalid (.9,.9,−.9), incomplete with implicit zeros, contradictory with an unknown variable and a duplicate pair), plus an exactly-singular case and a regression test reconstructing the pre-change code path.
  - `analysis/real_paper_eval.py` no longer string-matches the RuntimeWarning text; it reads the structured diagnostics and additionally reports `vacuous_matrix_count`.

- [x] **Clarify dependence-model limitations.**
  - Explain why Pearson correlation is currently used.
  - Discuss categorical encoding, nonlinear dependence, higher-order interactions, Spearman/Kendall alternatives, and vine-copula feasibility.
  - Avoid promising recovery of relationships not represented in the published statistics.
  - An implementation change to vine copulas is optional; a precise operating envelope is required.
  - **Done when:** the Adult results are connected to the pairwise-dependence limitation in the method and discussion.
  - **Result.** Integrated into Method and Discussion in `docs/neurips/neurips_2026.tex`. The stated envelope: AgentDataset reproduces per-variable marginals and *pairwise monotone* association, and nothing beyond that.
  - **Why Pearson:** it is the only dependence statistic reliably present in published aggregate statistics. Spearman/Kendall would be nearly free given the rank transform already in the pipeline (`synthesizer.py` reorders each marginal by the latent's ranks, so the imposed dependence is closer to a rank correlation than to Pearson's — worth stating precisely rather than claiming Pearson fidelity). Vine copulas need joint data or far richer reported statistics and are therefore outside the literature-only operating envelope; not implemented, and the paper says so rather than promising it.
  - **Categorical encoding is a hard limit, not a footnote.** Categorical variables are drawn as integer category codes and rank-transformed, so a correlation involving a categorical is a correlation against an arbitrary label ordering. This is now stated in Method.
  - **Connected to the Adult results (the acceptance criterion):** §9.5 showed an oracle threshold fitted on real test labels still leaves HGB −.025 below TRTR — exactly its ROC-AUC loss — and independent marginals −.192 at AUC .509. The residual *is* the ranking deficit: thresholding cannot restore dependence that was never synthesized. §9.7's audit supplies the mechanism from the other end — for auto-extracted specs, 0% of declared correlations are ever applied, so those datasets are independent marginals in all but name.

- [x] **Follow-up: establish what the fidelity score actually certifies.** *(Not an original §9.7 item; it came out of the noise-pivot result and affects the paper's headline metric, so it is recorded here.)*
  - `analysis/fidelity_validity_probe.py` builds parameters from three real UCI datasets, corrupts them by a controlled amount, synthesizes from the corrupted parameters, and compares the **reported** score against the **actual** distance from the real data → `results/fidelity_validity_probe_raw.csv`, `_summary.csv`, `_run_manifest.json`.
  - **The reported fidelity score is exactly invariant to parameter error.** Within each (dataset, seed) cell it takes a single value across all 14 corruption conditions, while the synthetic data move from KS 0.126 to 0.967 against the real distribution and from 0.02 to as much as 141.9 SD of mean error. Swapping each variable's statistics onto a different variable changes the reported score by zero.
  - **This is invariance by construction, not a sampling artifact.** The KS term (weight .40) compares each column against the same parameterized CDF *and the same mean/std* used to draw it; the bias term (.20) compares the sample mean to the declared mean. Any location–scale error moves data and reference together and cancels.
  - **The correlation term (.40) is the exception and does carry signal** — it detects synthesis failures such as a dropped or unresolved matrix (52–67 vs 80–100 repaired). It still reproduces a wrong *declared* correlation faithfully.
  - **Net:** the score measures whether the synthesizer did what it was told, never whether it was told the right thing. This is the common cause behind the §9.4/§9.5 pattern of high intrinsic fidelity with weak TSTR/HGB utility, and behind extraction quality (Q3) not showing up in fidelity at all.
  - **Acted on (paper reframing, no reported number changed).** Integrated into the canonical paper: the self-refinement contribution bullet became "An audit of the internal objective"; System Overview, Q1, and Discussion state the score's scope; appendix `app:scoring` carries the corruption table (`tab:validity`).
  - **Not done, deliberately:** the score itself is unchanged. Scoring against the published statistics or against downstream utility instead of the sampled CDF is the real fix, and it is a new contribution rather than a revision — see §9.9/§9.10 sequencing before attempting it.

### 9.8 P1 — Baselines and cost accounting

- [x] **Add a fair summary-statistics-only baseline presentation.**
  - Keep independent marginals as the primary like-for-like baseline.
  - Consider adding a simple non-iterative copula synthesis baseline if it is meaningfully different from the current synthesizer.
  - Continue labeling SDV as a real-data-access upper bound, never as a directly comparable competitor.
  - State exactly which inputs each method receives in table captions.
  - **Done when:** readers can separate the value of correlations, optimization, extraction, and access to real records.
  - **Result.** `analysis/p0_utility_benchmark.py` now adds a paired
    `copula_no_loop` arm: the exact first synthesis call of the shipped loop
    (same gold training-fold parameters, row count, seed, correlation policy,
    and initial noise 0.1), but without validation or refinement. The 30-seed
    rerun produced 12,960 successful metric rows with no failures, plus explicit
    seed-level contrasts in `results/p0_utility_baseline_contrasts_raw.csv` and
    deterministic bootstrap summaries in
    `results/p0_utility_baseline_contrasts.csv`.
  - **Correlations supply the benefit.** One-shot improves over independent on
    every primary utility metric: Adult LR AUC +.341 [.284,.400], Pima LR AUC
    +.405 [.322,.483], German LR AUC +.153 [.116,.192], and Concrete LR $R^2$
    +.612 [.584,.641]. AgentDataset minus one-shot includes zero for every
    classification primary metric. Only Concrete HGB shows a small refinement
    benefit ($R^2$ +.030 [.003,.059]; RMSE -.347 [-.674,-.043]). The loop adds
    no consistent downstream benefit at the shipped three-iteration budget.
  - **Input access is now explicit.** Independent sees gold marginals only;
    one-shot and AgentDataset see the same gold marginals plus pairwise
    correlations; SDV and TRTR see individual real training rows. SDV remains
    labeled a data-access upper bound, never a like-for-like competitor. The
    paper table and appendix input audit state that extraction is held perfect
    here, so Q2 measures a synthesis ceiling while Q1 evaluates extraction
    separately. No paid API calls are involved.

- [x] **Report total end-to-end cost and latency.**
  - Instrument discovery query expansion, search/fetch, document parsing, extraction for every selected document, synthesis, validation, refinement, and retry/failure overhead.
  - Separate paid API cost from local compute time.
  - Report per-stage and total values for each real-paper pipeline run.
  - Include the number of LLM calls, input/output tokens, documents processed, and optimization iterations.
  - **Deliverable:** `results/end_to_end_cost_latency.csv`.
  - **Done when:** a practitioner can estimate the cost of producing one final dataset, not just one extraction call.
  - **Result:** Three live real-paper runs exercised query expansion, web
    search, source suggestion, fetch, parse, extraction, three synthesis /
    validation / refinement iterations, and artifact writing. One finished
    dataset averages **\$0.01686** and **41.84 s** (ranges \$0.01327--\$0.02091
    and 39.07--45.64 s), with 3 LLM calls, 4,977 prompt tokens, and 442
    completion tokens. Mean time separates into 7.80 s API, 33.06 s network,
    and 0.98 s local compute; human source review is excluded.
  - **Accounting guard:** All 9 retained calls succeeded without retries or
    unknown-cost failures. The harness records every failed attempt and caps a
    run at \$0.25 or 12 attempts. A corrected CDC source-selection rerun brought
    physical experiment spend to \$0.057425 over 12 calls; the final retained
    three-paper artifact is \$0.05058 over 9 calls. Cached mode reproduces the
    accounting schema with zero API calls.

### 9.9 P2 — Paper presentation and clarity

- [x] **Tighten the abstract.**
  - Move the list of practical problems into the introduction.
  - Lead with the problem, system, evaluation setting, and one defensible quantitative result.
  - Avoid privacy guarantees and autonomous-performance claims stronger than the new evidence.
  - **Done when:** the abstract fits the venue limit and every claim is supported elsewhere in the paper.
  - **Result:** the 184-word abstract leads with the assisted workflow and reports the final deployed extractor's `.304` micro-F1 while explicitly rejecting unattended-curation and privacy-mechanism interpretations.

- [x] **Improve Figure 1 with a concrete walkthrough.**
  - Add a compact example showing query → discovered source → extracted statistic → generated rows/fidelity result.
  - Use a real-paper study example if space and licensing allow.
  - Keep labels readable at two-column print size.
  - **Likely file:** `docs/neurips/Figures/architecture.tex`.
  - **Done when:** a first-time reader can understand the input and output of each phase without reading the method section.
  - **Result:** Figure 1 now follows the rank-1 student-mental-health paper from query through the returned SAS-SV summary, 447 generated rows, and the internal-fidelity limitation caused by an omitted source correlation.

- [x] **Clarify Algorithm 1 notation.**
  - Replace ambiguous `θ` for extracted statistics with `S` or another unambiguous symbol.
  - Define every schedule constant and connect it to the sensitivity analysis.
  - **Done when:** extracted statistical parameters cannot be mistaken for learned model weights.
  - **Result:** the algorithm now uses statistical specification `S`, defines every bound and schedule constant, and identifies the five choices exercised by the sensitivity analysis.

- [x] **Update the section-by-section narrative.**
  - Related work: add disclosure control and clarify what prior aggregate-reconstruction systems required that AgentDataset automates.
  - Method: state the supported distributions, pairwise-only dependence, categorical encoding choice, matrix-fallback behavior, and privacy-diagnostic limitation.
  - Evaluation: add real-paper, nonlinear-model, regression, calibration, robustness, and end-to-end-cost results.
  - Discussion: state the operating envelope, distribution-shift risk, and remaining formal-privacy gap.
  - Conclusion: describe demonstrated capabilities only.
  - **Done when:** terminology and numbers agree across the abstract, tables, discussion, appendix, README, and data card.
  - **Result:** the paper consistently describes an assisted, pairwise aggregate-reconstruction workflow; all quantitative table rows are generated from committed result artifacts, and limitations cover extraction error, distribution shift, and the absence of formal privacy guarantees.

### 9.10 P2 — Reproducibility and submission closeout

- [x] **Add commands for every new experiment.**
  - Document real-paper study, broader utility benchmark, calibration, prompt ablation, optimizer sensitivity, and cost accounting in `results/README.md`.
  - Note which commands require network access or paid API credentials.
  - Provide a deterministic offline path using stored parsed text or regex where licensing permits.
  - **Done when:** a reviewer can reproduce each non-web-dependent table from the repository.
  - **Result:** `results/README.md` provides a study-to-command/output matrix, separates cached/offline and paid live paths, and documents deterministic table regeneration and checking.

- [x] **Expand automated tests.**
  - Add tests for new benchmark schemas, regression metrics, nonlinear-model preprocessing, threshold selection without test leakage, privacy-label changes, matrix diagnostics/repair, and challenge-set scoring.
  - Keep live web/API tests separate from the default unit-test suite.
  - **Done when:** `pytest` passes and result-schema regressions are caught automatically.
  - **Result:** 230 default tests pass, including paper-artifact/result reconciliation; the credentialed end-to-end test is marked and excluded from the default suite.

- [ ] **Perform final paper consistency and build checks.**
  - Rebuild all tables from saved result files.
  - Verify citations, cross-references, page limit, embedded fonts, undefined references, and overfull boxes.
  - Confirm the co-author list and affiliations and resolve the existing author-block TODO.
  - Proofread specifically for stale claims from the earlier single-seed narrative.
  - **Done when:** the built PDF matches committed results and contains no unresolved submission TODOs.
  - **Status:** technical checks pass after migration to the official NeurIPS 2026 template: ten generated table fragments match committed results; main text ends on page 9 and references start on page 10; the 25-page PDF has embedded Type 1 fonts; and the final log has no undefined references, citation warnings, overfull boxes, or stale result claims. The official NeurIPS checklist is complete. This item remains open only for confirmed camera-ready author names/affiliations and removal of the deliberate author-block TODO.

### 9.11 Recommended execution order

1. Freeze current results and define common result/reporting schemas.
2. Select real papers and create the extraction gold standard.
3. Run discovery, parsing, extraction, and full-pipeline real-paper evaluation.
4. Audit privacy claims, add the threat model, and repair data-card terminology.
5. Add the nonlinear classifier, regression task, and stronger uncertainty reporting.
6. Complete Adult calibration and the messy-text/Caveman ablations.
7. Run optimizer sensitivity, matrix-validity diagnostics, and end-to-end cost accounting.
8. Revise the paper's figures, notation, related work, abstract, discussion, and conclusion.
9. Re-run tests, regenerate tables, rebuild the PDF, and complete the submission proofread.

### 9.12 Minimum revision package

If time is constrained, the minimum defensible response to this review is:

- [x] a real-paper study covering at least three heterogeneous documents, with discovery and extraction scored separately;
- [x] carefully scoped privacy claims, a disclosure-risk threat model, and relevant disclosure-control citations;
- [x] one nonlinear classifier and one regression task;
- [x] Adult threshold/calibration results that avoid real-test-set tuning;
- [x] end-to-end rather than per-call cost reporting;
- [x] an explicit limitations discussion for idealized extraction, pairwise dependence, sample size, and distribution shift;
- [ ] updated paper tables, appendix details, reproducibility commands, and a clean final build.
  - Technical artifacts and build are complete; the submission build remains open pending the author-block confirmation in Section 9.10.

## 10. Product and Research Direction: Evidence-First Literature Synthesis

This section consolidates the next system revision. The goal is to make a wider search useful, preserve defensible inference without presenting it as reported fact, and make every generated dataset traceable to reviewed evidence. This is a proposed direction, not a description of implemented behavior. Existing benchmark outputs remain frozen; results from a revised pipeline must be saved under new names and compared against the current baseline.

### 10.1 Discovery — broaden coverage with a ranked source funnel

**Current state:** `optimize_query()` returns the original query plus up to four generated variants. `DiscoveryAgent` requests up to five results per query, and `run_discovery()` searches those queries sequentially and deduplicates by URL. Source suggestions are advisory in the UI; users can still select any result.

**Change:** Expand search across deliberately different query angles and, where practical, multiple scholarly and public-source indexes. Gather a configurable candidate pool, normalize URLs and metadata, deduplicate records, and rank candidates using topical relevance, source quality, likely availability of statistics, and coverage diversity. Keep the full candidate list available for inspection; suggestions must not silently discard sources.

**Essential details:**
- Make per-query and total-result limits, search concurrency, timeouts, and spend/API limits configurable and visible.
- Record the query that found each source, its rank, URL, title, snippet, retrieval time, and deduplication decision.
- Prefer broad retrieval followed by inexpensive ranking and user selection; reserve full-document fetching and LLM extraction for selected candidates.
- Treat search snippets as discovery evidence only, not as a substitute for the paper when extracting statistics.

**Acceptance checks:** Measure recall of relevant sources against a labeled query set, ranking quality at the user-selection cutoff, duplicate rate, fetch success, latency, and cost. Compare the wider funnel to the existing search behavior before changing defaults.

### 10.2 Extraction — retain evidence and label inference

**Current state:** The production prompt strongly prefers explicitly stated numeric statistics. Long documents are chunked and merged, but chunk outputs can duplicate the same entity or statistic. The output schema mostly stores values and source-level metadata, not claim-level evidence.

**Change:** Let the extractor propose both directly reported statistics and cautious, useful inferences, while representing them as distinct claim types. Every candidate should carry a supporting passage or table locator, source identity, extraction method, assumptions, and calibrated confidence. Preserve abstention when evidence is insufficient.

**Essential details:**
- Distinguish `reported`, `derived`, and `inferred` values. Record derivations (for example, converting SE to SD only when sample size is available) rather than silently rewriting the source.
- Preserve units, population/subgroup, sample size, time point, statistic type, table/figure/page location, and whether a value is adjusted or unadjusted.
- Do not treat a p-value or confidence interval as a standard deviation without a justified conversion. Preserve conflicting abstract/table values as separate claims with locators and a conflict flag.
- Merge overlapping chunk results by source location and normalized variable/measure identity, not by blindly unioning or averaging all outputs.
- Evaluate direct extraction and inference separately, including precision, recall, calibration, unsupported-claim rate, and evidence-locator accuracy.

**Acceptance checks:** On an annotated real-paper set, report separate quality for reported and inferred claims. Review high-impact or low-confidence claims before allowing them into synthesis.

### 10.3 Cross-source reconciliation — do not average unlike studies

**Current state:** `merge_parameters()` averages means and standard deviations for same-named variables, unions unique variables, and averages duplicate correlation keys. It does not account for sample size, population, study design, units, or whether two measures with the same name are actually comparable.

**Change:** Replace automatic numeric averaging with evidence-aware reconciliation. Normalize names and units, group only comparable measures, retain per-study estimates, and surface disagreement. Pooling should be an explicit, documented statistical operation with assumptions and uncertainty, not the default merge behavior.

**Essential details:**
- Match measures using name plus definition, unit, population, time point, and study context; ambiguous matches remain separate for review.
- Show source-level values and confidence intervals/uncertainty side by side. Support weighted pooling only where required inputs and compatibility assumptions are present.
- Preserve correlations as study-specific unless the variable definitions, sample/population, and statistic are compatible.
- Store the accepted or pooled value with links back to all contributing claims and the reconciliation decision.

**Acceptance checks:** Include fixtures with unit changes, subgroup differences, duplicate reports, and conflicting estimates. Verify the system flags incompatibility rather than silently combining values.

### 10.4 Human review — approve the specification before synthesis

**Current state:** The Streamlit app supports source selection. It currently proceeds from extraction to merging and synthesis without a parameter-review step, despite the intended human-in-the-loop workflow described earlier in this document.

**Change:** Add a review stage between extraction and reconciliation/synthesis. Present the source passage beside each proposed statistic, its evidence type (reported/derived/inferred), uncertainty, source context, and any conflict or name-matching warning. Let the user accept, reject, edit, or defer claims.

**Essential details:**
- Make approval state and editor changes explicit and audit logged.
- Prevent unreviewed low-confidence inferences or unresolved source conflicts from entering the default synthesis specification; allow opt-in inclusion with a visible label.
- Keep source selection and claim approval separate: a relevant paper can be useful even when some of its extracted claims are rejected.

**Acceptance checks:** A saved run must reconstruct the exact reviewed specification from its audit record. UI tests should cover accepted, rejected, edited, conflicting, and inferred claims.

### 10.5 Terminal CLI — make the same workflow available from a shell

**Current state:** The primary user interface is the Streamlit app. Existing scripts under `analysis/` and `benchmark.py` have task-specific command-line entry points, but there is no supported general-purpose `agentdataset` command for running and managing ordinary project workflows.

**Change:** Add a small installable CLI as a second front end to the shared discovery, extraction, review, reconciliation, synthesis, and validation services. The CLI should not fork the pipeline logic or produce a different interpretation of the same reviewed specification.

**Essential details:**
- Register an `agentdataset` console command in `pyproject.toml`. Prefer the Python standard library `argparse` initially unless a richer command structure clearly justifies another dependency.
- Provide a guided `run` path for a research query, plus focused commands to discover sources and resume/review a saved run. Keep the initial command surface small and grow it from real use.
- Support interactive terminal review of sources and extracted claims, including evidence snippets, reported/derived/inferred labels, conflicts, and accept/edit/reject choices. Non-interactive mode must consume an explicit reviewed specification or require an explicit policy; it must not silently skip review of uncertain claims.
- Expose practical options such as output directory, row count, seed, model/provider, result limits, and machine-readable output. Load credentials from environment/configuration, never require secrets as command-line arguments.
- Write the same versioned manifest, evidence, CSV, parameters, and data card as the UI. Provide clear progress, per-source failures, stage timing/cost where available, and stable non-zero exit codes for failed or incomplete runs.
- Document installation, `--help`, examples, environment setup, resume behavior, and the difference between interactive and non-interactive operation in `README.md` when implementation begins.

**Acceptance checks:** Exercise the CLI in a clean environment with mocked/offline discovery and extraction, verify argument validation, review decisions, resumability, output paths, and exit codes. For the same saved reviewed specification and seed, compare CLI and Streamlit outputs and manifests for semantic parity. Keep live API tests separate and opt-in.

### 10.6 Synthesis — use an auditable, replaceable engine interface

**Current state:** `Synthesizer` samples parametric marginals and uses a Gaussian latent correlation transform. Its correlation diagnostics are valuable, but the project audit found that unresolved names can leave declared correlations unapplied; categorical rank coding also makes dependence sensitive to arbitrary category ordering. The optimizer changes noise against an internal score that is not independent evidence of source fidelity.

**Change:** Separate the specification from the sampling engine and make synthesis methods selectable behind a stable interface. Start with methods supported by published aggregate inputs: fitted marginal distributions plus an explicit dependence model. Compare the existing Gaussian-copula-style method with defensible alternatives only where the available statistics identify their parameters. Do not introduce a record-trained neural generator as the default for summary-only inputs.

**Essential details:**
- Validate variable references, units, marginal parameters, and correlation-matrix consistency before sampling. Unresolved declared correlations must be prominent errors or review items, not quiet omissions.
- Treat positive-semidefinite repair as a documented transformation: retain original and repaired matrices, report the change, and allow rejection when repair materially changes the evidence.
- Add a categorical dependence method that does not interpret alphabetical or arbitrary labels as ordinal measurements. If available aggregate inputs cannot identify a valid categorical dependence model, report that limitation and omit the unsupported dependence rather than implying it was preserved.
- Keep SDV's real-record-trained synthesizers as benchmark baselines or options for workflows that actually have record-level training data; they are not a drop-in engine for literature-only summary statistics.
- Expose row count, random seed, marginal model, dependence model, repair policy, and all approximation choices in the output manifest.

**Acceptance checks:** Compare engines on held-out known datasets using marginal error, dependence error, downstream utility, interval coverage, runtime, and failure rate. Include independent-marginal and one-shot/no-refinement baselines. Retain the existing implementation as a reproducibility baseline until the new engine demonstrates a measured improvement.

### 10.7 Validation and optimization — score against independent evidence

**Current state:** The KS and bias components compare generated data to the same parameter specification used to generate it. The audit in §9.7 found the resulting score invariant to errors in those input parameters. It therefore measures implementation consistency, not whether extracted statistics are correct or whether generated rows match the source population. Grounding exists in `Validator`, but the app's optimization call does not pass source text; without it, grounding defaults to 100. The noise-pivot loop can optimize the internal score without improving downstream usefulness.

**Change:** Report separate validation dimensions and make independent evidence the basis for claims about source fidelity. Keep internal sampler diagnostics, but label them as such and do not use them alone to select or certify a dataset. Make refinement optional and retain a no-loop result as a baseline.

**Essential details:**
- Validate extracted claims against source passages and validate synthetic marginals/dependence against reviewed source statistics with uncertainty-aware tolerances.
- Where public benchmark records are available, compare against held-out real data and downstream tasks; clearly separate these benchmark-only checks from normal literature-only runs.
- Pass evidence through to validation. Replace substring/any-number grounding with claim-level checks tied to the cited passage, including correct statistic, variable, units, and context.
- Report per-variable and per-pair errors, unresolved or skipped correlations, matrix repairs, confidence intervals across seeds, and all score definitions.
- Keep nearest-neighbor row dispersion labeled only as a synthetic-row diagnostic. It is not a privacy audit or formal guarantee; no differential-privacy claim follows from aggregate-only inputs.

**Acceptance checks:** Corrupt a known specification in controlled ways and verify validation detects the error. Test that score changes when source statistics change even if the sampler faithfully follows its inputs. Evaluate each claimed validation metric against a predeclared external target.

### 10.8 Throughput, caching, and failure handling

**Current state:** The UI processes selected sources serially. Fetching and PDF parsing can dominate run time; failed fetches may fall back to short snippets. Repeating a run can repeat network and extraction work.

**Change:** Use bounded concurrency for independent fetch, parse, and extraction work, with per-provider rate limits. Cache fetched documents, parsed text, and extraction results using stable content hashes plus model/prompt/schema versions. Make partial completion recoverable.

**Essential details:**
- Record per-source stage status, elapsed time, retries, and failure cause. Continue when one source fails and allow retry of only failed stages.
- Keep cache invalidation explicit: changed content, prompt, model, or schema creates a distinct result.
- Bound download sizes, timeouts, concurrency, API attempts, and spend. Preserve current URL safety checks and PDF size limits.
- Treat snippet-only extraction as low-evidence and require a clear label or user approval.

**Acceptance checks:** Demonstrate bounded resource use, correct cache invalidation, resumability, and a measured reduction in end-to-end latency without loss of source coverage or extraction quality.

### 10.9 Provenance, retention, and evaluation discipline

**Current state:** Session artifacts include the best CSV, merged parameters, and a data card. The orchestrator prunes sessions beyond three. Some benchmark artifacts and manifests are already detailed, but the product run does not yet preserve the full claim/review/reconciliation chain needed to reproduce a dataset.

**Change:** Save a versioned run manifest linking query, retrieved sources, source contents or permitted references/hashes, extracted claims, review decisions, reconciliation, synthesis configuration, validation, and outputs. Make retention configurable and explicit rather than silently deleting prior research runs on startup.

**Essential details:**
- Include code revision, dependency/model identifiers, prompt/schema versions, timestamps, random seeds, source URLs and content hashes, and cost/latency by stage.
- Preserve raw per-source claims separately from the final synthesis specification; never overwrite the frozen baseline artifacts.
- Keep privacy language narrow: aggregate-source synthesis may reduce reliance on record-level training data, but it does not itself establish anonymity or differential privacy.
- Evaluate each pipeline stage separately and end to end. Report uncertainty, missingness, failures, and cost; keep live API/web experiments distinct from deterministic offline replays.

**Acceptance checks:** A run can be replayed from its manifest where licensing and cached content permit, and its final CSV can be traced back to each accepted claim. Retention tests confirm older runs are preserved according to the configured policy.

### 10.10 Recommended implementation sequence

1. Define a versioned claim/evidence schema and run manifest; preserve current outputs as frozen baselines.
2. Add extraction evidence locators, `reported`/`derived`/`inferred` labels, and chunk-level deduplication; measure on annotated papers.
3. Add the parameter review flow and replace blind averaging with study-aware reconciliation and conflict reporting.
4. Build the CLI and UI on shared workflow services; validate parity on saved reviewed specifications.
5. Broaden discovery into a configurable ranked funnel; evaluate recall, ranking, duplicates, cost, and latency on labeled queries.
6. Add resumable caching and bounded parallel source processing; compare end-to-end costs and failure rates.
7. Repair validation so it tests claims against independent evidence; audit the optimizer objective and keep a no-loop baseline.
8. Introduce a synthesis-engine interface and benchmark dependence/categorical alternatives on known data before changing the default.
9. Run a staged end-to-end evaluation on real papers, publish new artifacts under new names, and update the paper/data card only after claims are supported by those results.

