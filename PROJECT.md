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
  - *PETS (Privacy Enhancing Technologies Symposium)*: Emphasizing the privacy score (nearest-neighbor distance) and the ability to generate statistically accurate proxies for highly sensitive data without exposing PII.

### 7.3 Evaluation Status

All three strengthening items below are implemented as real, non-mocked measurements in `benchmark.py` (run via `python benchmark.py`, or each function individually).

1. **Empirical Benchmarks** (`run_empirical_benchmark`) — **Done.** Runs the pipeline on 3 real public datasets: UCI Adult Income (demographic), Pima Indians Diabetes (medical), and Statlog German Credit (financial). Real `describe()`/`corr()` statistics are templated into extractable "literature style" text, extracted through OpenRouter (single key covering OpenAI/Anthropic/Google, falling back to regex automatically if unavailable), synthesized, and compared via TRTR vs. TSTR logistic regression (accuracy, F1, ROC-AUC) against real held-out data, alongside the existing fidelity report. This required adding categorical-variable support to the core pipeline (see 7.1) so classification targets like income or credit risk could be modeled, not just continuous regression targets. The Adult benchmark also includes a 3-category marital-status feature, exercising multi-category (beyond binary) extraction, synthesis, and validation end-to-end.
2. **Ablation Studies** (`run_ablation_study`) — **Done.** LLM vs. regex extraction is scored against known ground truth across 8 varied domain texts (not a single snippet), with paired t-tests. The Noise Pivot optimization loop is measured across iteration counts 0/1/2/3/5/10, over multiple seeds and source texts, with paired t-test and Wilcoxon significance tests comparing 0 vs. 5 iterations.
3. **Cost/Latency Analysis** (`run_cost_latency_analysis`) — **Done.** Real litellm cost, token, and latency tracking (`completion_cost`, `usage`, `token_counter`) across OpenAI/Anthropic/Google, routed through OpenRouter with a single key, gracefully skipping providers when no key is configured. The Caveman Protocol's token savings are measured directly per model rather than asserted: the compressed system prompt is 189 tokens vs. 242 for a naturally-worded equivalent, a **21.9%** reduction (`results/caveman_token_savings.csv`), not the earlier unverified claims of 43%/73%.

Remaining follow-up work: broaden the empirical benchmark beyond 3 datasets / a single downstream model (logistic regression) if reviewers ask for more breadth.

---

## 8. Paper (IAAI-27) — Status & Pickup Notes

**Target venue:** IAAI-27 (co-located with AAAI-27), Emerging Applications track.
**Format:** AAAI two-column, **6-page** main-content limit; references + appendices unlimited.
**Review model:** **single-blind** — author names are required (not anonymized).

### 8.1 Draft location & build
The paper lives in `AuthorKit27/` (git-ignored — build artifacts excluded from the repo):

- `AuthorKit27/AgentDataset.tex` — main paper. Compiles clean via MiKTeX `pdflatex` + `bibtex` to 9 pages: **main content ends within page 5** (within the limit), references p. 6, technical appendix + reproducibility checklist pp. 7–9. (The `AnonymousSubmission2027.tex`/`CameraReady2027.tex` files in the kit are unused AAAI template boilerplate.)
- `AuthorKit27/references.bib` — 29 references (from two literature-review passes; flagged citations verified/corrected).
- `AuthorKit27/Figures/architecture.tex` → `architecture.pdf` — TikZ pipeline diagram (standalone-compiled to vector PDF, included via `\includegraphics`).
- `AuthorKit27/ReproducibilityChecklist.tex` — filled in, `\input` at end of paper.
- Uses the `[preprint]` style option (authors shown, copyright slug suppressed during review). **Do NOT** use `[submission]` — that anonymizes the author block (wrong for single-blind).

**Build:** from `AuthorKit27/`, run `pdflatex AgentDataset` → `bibtex AgentDataset` → `pdflatex AgentDataset` ×2.
Compliance verified: 0 overfull boxes, 0 undefined refs, no Type-3 fonts, all fonts embedded, PDF 1.5.

### 8.2 Reviewer-readiness assessment (5 critique points)
Status of an external critique of the evaluation, as of this writing:

1. **Error bars on empirical benchmark — DONE.** `run_empirical_benchmark` now extracts once per dataset then runs a 5-seed loop varying both the split (`random_state=seed`) and synthesis (`Synthesizer(seed=seed)`), reporting mean ± sample-std (ddof=1) for every TRTR/TSTR metric in `results/empirical_benchmark.csv`.
2. **Categorical narrow / pairwise-only — HALF DONE.** The *binary-only* limitation is **resolved** (commit `0004dd8`: arbitrary-cardinality support + 3-category Adult `marital_status`). Still true and *stated in the paper* (Discussion + appendix): only pairwise correlations (no joint >2-var structure), and a compact 3-feature-per-dataset schema.
3. **Adult F1 gap — DONE (characterized).** With 5-seed error bars: TRTR F1 0.547±0.007 vs TSTR 0.119±0.013 — stable, not noise. `analysis/adult_f1_gap.py` (→ `results/adult_f1_gap.csv`) shows the cause is **not** a distorted target marginal (synthetic `>50K` prevalence ≈ real ≈ 0.24) but **attenuated feature↔target correlation** (AgentDataset retains ~73% of the real point-biserial signal; independent-marginals flattens it to ~0 → F1 0.0), driving minority-class under-prediction at the 0.5 threshold (recall 0.46→0.07). Partly a calibration artifact: even SDV (AUC 0.852) collapses to F1 0.11 at that threshold. Written up in the paper's Evaluation (`\paragraph{Characterizing the Adult F1 gap}`) + Discussion.
4. **Caveman "measured vs claimed" honesty — DONE.** Reconciled to measured **21.9%** (189 vs 242 tokens) in both this doc and the paper; framed as measured, not asserted.
5. **No baseline synthesizer — DONE.** Added two baselines in `benchmark.py`, both run inside the same 5-seed TSTR protocol (results carry a `synthesizer` column: `agentdataset` | `independent` | `sdv_gaussian_copula`): `synthesize_independent_marginals` (deep-copies params, clears correlations, reuses `Synthesizer` — no new deps) and `synthesize_sdv_gaussian_copula` (SDV `GaussianCopulaSynthesizer` fit on real train data; **guarded import** so a missing/broken `sdv` skips gracefully; `sdv>=1.17.0` added to deps). Finding: AgentDataset beats independent-marginals where correlations matter (Adult F1 .119 vs .000; Pima AUC .822 vs .397) and trails SDV (Adult AUC .601 vs .852) — but SDV is fit on **real** records, so it's framed in the paper as a data-access upper bound, not a like-for-like competitor.

### 8.3 Prioritized next steps (pick up here)
1. **(crit #1) — DONE.** 5-seed loop + mean ± std in `run_empirical_benchmark`.
2. **(crit #5) — DONE.** Independent-marginals + SDV GaussianCopula baselines on the same TSTR protocol.
3. **(crit #3) — DONE.** Adult F1 gap analyzed (`analysis/adult_f1_gap.py`) and written up.
4. **DONE.** Folded into the paper's Evaluation: Table 1 rebuilt with error bars + baseline columns (`AgentDataset.tex`, rebuilt clean — 0 overfull, 0 undefined, main content ≤ 6 pp).
5. **Housekeeping before submission — OPEN.** Confirm co-author list/affiliations (IAAI recommends a deploying-institution co-author; note the `% TODO` at `AgentDataset.tex:28`); final proofread.

**Story shift to be aware of:** the 5-seed numbers overturned the old single-seed narrative — **Pima is now the clean success case** (TSTR ≈ TRTR on all metrics), not the failure; Adult's minority-class F1 collapse is the headline limitation. Q1 prose + Discussion were rewritten to match.

**Note:** re-running steps 1–3 requires `python benchmark.py` / `python analysis/adult_f1_gap.py` (LLM API calls / compute), which regenerates `results/*.csv`; the paper's numbers/tables must then be updated to match.

