# SPEC.md: AgentDataset Architecture

## 1. Overview

AgentDataset is a 4-phase autonomous pipeline: discover research documents → extract statistical parameters → synthesize a dataset → validate fidelity. Each phase is a standalone module; the `Orchestrator` wires them together and manages the iterative refinement loop.

---

## 2. Modules

### 2.1 Discovery (`core/discovery.py` and `Orchestrator.run_discovery`)

- **LLM-guided query expansion**: `Orchestrator.optimize_query()` asks the configured LLM for search-optimized queries focused on research papers, PDFs, reports, means, standard deviations, and correlations. The original query is always included; on LLM failure, only the original query is used.
- **Search**: DuckDuckGo (`DDGS`) queries for both `filetype:pdf` and general HTML results for each optimized query.
- **Deduplication**: discovery results are deduplicated by URL.
- **Source suggestion**: `Orchestrator.suggest_sources()` asks the LLM which results are likely to contain statistical parameters; on failure, no suggestions are returned.
- **PDF fetch**: `requests.get()` streams the file to a `NamedTemporaryFile`; returns `pdf://<path>` to the caller.
- **HTML fetch**: `trafilatura` extracts clean text from the page.
- **Error handling**: Network failures are caught and logged; snippet fallback is used for PDFs that fail to download.

### 2.2 Extraction (`core/extractor.py`)

- **LLM path** (when API key is present): calls `litellm.completion()` with a structured JSON prompt enforcing the schema below. `extraction_method = "llm"`.
- **Regex fallback** (on any LLM failure or no key): regex patterns match mean/std pairs in either order (incl. `SD`, `σ`, `s.d.` variants, negative and scientific-notation numbers), categorical variables with any number of categories (`takes value 'a' with probability 0.2, 'b' with probability 0.3, and 'c' with probability 0.5`), and best-effort named correlations (`correlation between X and Y is 0.6`, `corr(X, Y) = -0.4`). `extraction_method = "regex_fallback"`. On LLM failure the log names the cause (e.g. `RateLimitError`).
- **PDF parsing**: `fitz` (PyMuPDF) converts each page to text; temp file is deleted after parsing.
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

### 2.5 Validator (`core/validator.py`)

Produces a `FidelityReport` with four components:

| Component | Weight | Method |
|-----------|--------|--------|
| KS score | 40% | Fraction of variables passing KS-test (`p ≥ 0.05`), × 100 |
| Correlation score | 40% | Mean absolute error of off-diagonal correlations: `1 − mean(|synth − target|)/2`, clamped to [0, 1] (diagonal excluded so it can't inflate the score) |
| Bias score | 20% | Fraction of variables within 20% mean deviation |
| Privacy score | — (reported separately) | Avg nearest-neighbour distance on 500-row subsample, normalised to [0, 1] |

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
    └── DATACARD.md       # Fidelity + privacy report
```

`.agentdataset_cache/` also holds runtime artifacts, migrated results, memory files, and live e2e reports. It is ignored by git. Only the 3 most recent session directories are retained. Older ones are deleted at `Orchestrator.__init__`.

---

## 7. Research Viability & Publishability

AgentDataset sits at the intersection of Agentic AI, Synthetic Data Generation, and Data Privacy. It offers a structured, autonomous approach to solving the problem of generating high-fidelity synthetic datasets from academic literature.

### 7.1 Novel Contributions

- **End-to-End Autonomous Workflow**: Combines LLM extraction tools and synthetic data generators into a closed-loop autonomous agent that searches the web, reads PDFs, and optimizes its own output.
- **The "Noise Pivot" Optimization Loop**: An explore/exploit ratcheting system for tuning the `noise_level` against a multi-component fidelity score (KS-test, Correlation MAE, Bias).
- **Fidelity vs. Privacy Quantification**: Automatically generates a `DATACARD.md` that quantifies both the statistical fidelity (KS tests, correlation matrices) and privacy (nearest-neighbor distance).
- **The "Caveman Protocol"**: A token-optimization strategy using compressed, filler-free prompting for extraction to reduce cost and latency in agentic loops.
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

## 8. Paper (IAAI-27) — Status & Pickup Notes

**Target venue:** IAAI-27 (co-located with AAAI-27), Emerging Applications track.
**Format:** AAAI two-column, **6-page** main-content limit; references + appendices unlimited.
**Review model:** **single-blind** — author names are required (not anonymized).

### 8.1 Draft location & build
The paper lives in `AuthorKit27/` (git-ignored — build artifacts excluded from the repo):

- `AuthorKit27/main.tex` — canonical paper source. Compiles clean via MiKTeX `pdflatex` + `bibtex` to 9 pages: **main content ends on page 4** (within the limit), references pp. 5–6, technical appendix + reproducibility checklist pp. 7–9. (The `AnonymousSubmission2027.tex`/`CameraReady2027.tex` files in the kit are unused AAAI template boilerplate.)
- `AuthorKit27/references.bib` — 35 references, including the three real-paper sources and disclosure-control literature.
- `AuthorKit27/Figures/architecture.tex` → `architecture.pdf` — TikZ pipeline diagram (standalone-compiled to vector PDF, included via `\includegraphics`).
- `AuthorKit27/ReproducibilityChecklist.tex` — filled in, `\input` at end of paper.
- Uses the `[preprint]` style option (authors shown, copyright slug suppressed during review). **Do NOT** use `[submission]` — that anonymizes the author block (wrong for single-blind).

**Build:** from `AuthorKit27/`, run `pdflatex main` → `bibtex main` → `pdflatex main` ×2.
Compliance verified: 0 overfull boxes, 0 undefined refs, no Type-3 fonts, all fonts embedded, PDF 1.5.

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
4. **DONE.** Folded into the paper's Evaluation: Table 1 rebuilt with error bars + baseline columns (`main.tex`, rebuilt clean — 0 overfull, 0 undefined, main content ≤ 6 pp).
5. **Housekeeping before submission — OPEN.** Confirm co-author list/affiliations (IAAI recommends a deploying-institution co-author; note the `% TODO` in the author block of `main.tex`); final proofread.

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
  - **Likely files:** `AuthorKit27/main.tex`, `README.md`, `agentdataset/core/validator.py`, generated `DATACARD.md` text, and UI copy.
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
  - **Likely files:** `AuthorKit27/references.bib`, related work and limitations in `AuthorKit27/main.tex`.
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

- [ ] **Implement post-hoc calibration and threshold evaluation.**
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

### 9.6 P1 — Extraction robustness and prompt ablations

- [ ] **Build a messy/ambiguous extraction challenge set.**
  - Include negated findings, multiple populations, conflicting values across sections, repeated variables at different time points, units, confidence intervals, table footnotes, scientific notation, and irrelevant nearby numbers.
  - Prefer real excerpts that can be legally quoted in short form; supplement with clearly labeled constructed examples.
  - Annotate expected output and “should not extract” values.
  - Keep challenge cases separate from development examples to reduce overfitting.
  - **Deliverables:** challenge-set fixture plus annotation metadata under `analysis/` or `tests/fixtures/`.
  - **Done when:** the benchmark distinguishes correct extraction, abstention, and context errors.

- [ ] **Compare LLM, regex, and prompt styles on the same challenge set.**
  - Compare the Caveman Protocol with the natural-language prompt using identical model, temperature, documents, and repetitions.
  - Measure accuracy, precision/recall/F1, malformed-output rate, hallucination rate, input/output tokens, latency, and cost.
  - Use multiple LLM repetitions to expose nondeterminism.
  - Test the regex fallback separately rather than treating fallback outputs as LLM results.
  - **Likely file:** extend `run_ablation_extraction_method()` in `benchmark.py`.
  - **Done when:** the paper can state whether the 21.9% prompt-token reduction changes extraction quality.

### 9.7 P1 — Synthesis and optimization diagnostics

- [ ] **Analyze noise-pivot sensitivity.**
  - Vary `PATIENCE`, initial noise, exploration multiplier, exploitation factor, reset interval, and iteration budget using a predeclared grid or efficient design.
  - Report best score, final score, convergence rate, variance, runtime, and frequency of resets.
  - Compare the existing schedule with simple fixed-noise and/or random-search baselines.
  - Explain whether hyperparameters were tuned on the same datasets used for final reporting.
  - Convert the current “use 5+ iterations” observation into a supported operational recommendation.
  - **Likely files:** `benchmark.py`, `agentdataset/core/orchestrator.py` if configuration is exposed, and optimizer tests.
  - **Done when:** the chosen defaults are justified empirically and the high 10-iteration variance is explained.

- [ ] **Measure correlation-matrix validity and fallback behavior.**
  - Log the minimum eigenvalue, positive-definite status, repair/fallback action, and source of every correlation matrix.
  - Report frequency by templated inputs, real-paper gold statistics, and automatic extraction.
  - Evaluate a documented nearest-positive-semidefinite repair instead of immediately dropping all correlations, but retain the current behavior as a baseline.
  - Add unit tests for slightly invalid, severely invalid, incomplete, and contradictory matrices.
  - **Likely files:** `agentdataset/core/synthesizer.py`, `tests/test_synthesizer.py`, benchmark outputs.
  - **Done when:** the paper quantifies how often the advertised copula path is actually used.

- [ ] **Clarify dependence-model limitations.**
  - Explain why Pearson correlation is currently used.
  - Discuss categorical encoding, nonlinear dependence, higher-order interactions, Spearman/Kendall alternatives, and vine-copula feasibility.
  - Avoid promising recovery of relationships not represented in the published statistics.
  - An implementation change to vine copulas is optional; a precise operating envelope is required.
  - **Done when:** the Adult results are connected to the pairwise-dependence limitation in the method and discussion.

### 9.8 P1 — Baselines and cost accounting

- [ ] **Add a fair summary-statistics-only baseline presentation.**
  - Keep independent marginals as the primary like-for-like baseline.
  - Consider adding a simple non-iterative copula synthesis baseline if it is meaningfully different from the current synthesizer.
  - Continue labeling SDV as a real-data-access upper bound, never as a directly comparable competitor.
  - State exactly which inputs each method receives in table captions.
  - **Done when:** readers can separate the value of correlations, optimization, extraction, and access to real records.

- [ ] **Report total end-to-end cost and latency.**
  - Instrument discovery query expansion, search/fetch, document parsing, extraction for every selected document, synthesis, validation, refinement, and retry/failure overhead.
  - Separate paid API cost from local compute time.
  - Report per-stage and total values for each real-paper pipeline run.
  - Include the number of LLM calls, input/output tokens, documents processed, and optimization iterations.
  - **Deliverable:** `results/end_to_end_cost_latency.csv`.
  - **Done when:** a practitioner can estimate the cost of producing one final dataset, not just one extraction call.

### 9.9 P2 — Paper presentation and clarity

- [ ] **Tighten the abstract.**
  - Move the list of practical problems into the introduction.
  - Lead with the problem, system, evaluation setting, and one defensible quantitative result.
  - Avoid privacy guarantees and autonomous-performance claims stronger than the new evidence.
  - **Done when:** the abstract fits the venue limit and every claim is supported elsewhere in the paper.

- [ ] **Improve Figure 1 with a concrete walkthrough.**
  - Add a compact example showing query → discovered source → extracted statistic → generated rows/fidelity result.
  - Use a real-paper study example if space and licensing allow.
  - Keep labels readable at two-column print size.
  - **Likely file:** `AuthorKit27/Figures/architecture.tex`.
  - **Done when:** a first-time reader can understand the input and output of each phase without reading the method section.

- [ ] **Clarify Algorithm 1 notation.**
  - Replace ambiguous `θ` for extracted statistics with `S` or another unambiguous symbol.
  - Define every schedule constant and connect it to the sensitivity analysis.
  - **Done when:** extracted statistical parameters cannot be mistaken for learned model weights.

- [ ] **Update the section-by-section narrative.**
  - Related work: add disclosure control and clarify what prior aggregate-reconstruction systems required that AgentDataset automates.
  - Method: state the supported distributions, pairwise-only dependence, categorical encoding choice, matrix-fallback behavior, and privacy-diagnostic limitation.
  - Evaluation: add real-paper, nonlinear-model, regression, calibration, robustness, and end-to-end-cost results.
  - Discussion: state the operating envelope, distribution-shift risk, and remaining formal-privacy gap.
  - Conclusion: describe demonstrated capabilities only.
  - **Done when:** terminology and numbers agree across the abstract, tables, discussion, appendix, README, and data card.

### 9.10 P2 — Reproducibility and submission closeout

- [ ] **Add commands for every new experiment.**
  - Document real-paper study, broader utility benchmark, calibration, prompt ablation, optimizer sensitivity, and cost accounting in `results/README.md`.
  - Note which commands require network access or paid API credentials.
  - Provide a deterministic offline path using stored parsed text or regex where licensing permits.
  - **Done when:** a reviewer can reproduce each non-web-dependent table from the repository.

- [ ] **Expand automated tests.**
  - Add tests for new benchmark schemas, regression metrics, nonlinear-model preprocessing, threshold selection without test leakage, privacy-label changes, matrix diagnostics/repair, and challenge-set scoring.
  - Keep live web/API tests separate from the default unit-test suite.
  - **Done when:** `pytest` passes and result-schema regressions are caught automatically.

- [ ] **Perform final paper consistency and build checks.**
  - Rebuild all tables from saved result files.
  - Verify citations, cross-references, page limit, embedded fonts, undefined references, and overfull boxes.
  - Confirm the co-author list and affiliations and resolve the existing author-block TODO.
  - Proofread specifically for stale claims from the earlier single-seed narrative.
  - **Done when:** the built PDF matches committed results and contains no unresolved submission TODOs.

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

- [ ] a real-paper study covering at least three heterogeneous documents, with discovery and extraction scored separately;
- [ ] carefully scoped privacy claims, a disclosure-risk threat model, and relevant disclosure-control citations;
- [ ] one nonlinear classifier and one regression task;
- [ ] Adult threshold/calibration results that avoid real-test-set tuning;
- [ ] end-to-end rather than per-call cost reporting;
- [ ] an explicit limitations discussion for idealized extraction, pairwise dependence, sample size, and distribution shift;
- [ ] updated paper tables, appendix details, reproducibility commands, and a clean final build.

