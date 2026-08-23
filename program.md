# AgentDataset: Workflow Guide

This document describes the end-to-end workflow — what each phase does, what decisions are made, and what the system produces.

---

## Phase 0 — Discovery

**Entry point:** `Orchestrator.run_discovery(query)`

1. Use the configured LLM to expand the user query into several search-optimized queries focused on statistical sources. If that fails, use the original query only.
2. Search DuckDuckGo for each optimized query.
3. PDF results get `relevance_score = 1.0`; HTML results get `0.8`.
4. Deduplicate results by URL and return `DiscoveryResult` objects shown in the UI for the user to select.
5. Use `suggest_sources()` to ask the LLM which results are most likely to contain means, standard deviations, correlations, or distributions.

**Network errors** are caught and logged — partial results are returned rather than crashing.

---

## Phase 1 — Extraction

**Entry point:** `Orchestrator.process_source(result)` (called once per selected source)

1. `DiscoveryAgent.fetch_content()` is called:
   - **HTML**: trafilatura extracts clean text.
   - **PDF**: streamed to a temp file; returns `pdf://<path>`.
2. If `pdf://` prefix is detected, `Extractor.pdf_to_markdown()` parses it with PyMuPDF; temp file is deleted afterwards.
3. `Extractor.extract_parameters()` runs:
   - **With API key**: calls the LLM via litellm, parses structured JSON response → `Parameters`.
   - **Without key or on LLM failure**: falls back to regex (matches mean/std in either order, supports `SD`, `σ`, `s.d.`, negative/scientific numbers, and best-effort named correlations). LLM failures log the cause (e.g. `RateLimitError`).
4. `meta.extraction_method` is set to `"llm"` or `"regex_fallback"` accordingly.

**Multi-source:** if the user selects multiple sources, `process_source` runs for each and results are merged via `merge_parameters()`:
- Same variable name → average mean and std.
- Unique variable names → included in full.
- Same correlation pair → average correlation value.

---

## Phase 2 & 3 — Synthesis-Validation Loop

**Entry point:** `Orchestrator.run_optimization_loop(parameters, iterations)`

Each iteration:
1. `Synthesizer.synthesize(parameters, noise_level)` generates a correlated DataFrame.
2. `Validator.validate(df, parameters)` scores it:
   - **KS score** (40%): fraction of variables passing KS-test.
   - **Correlation score** (40%): off-diagonal match of synthetic vs target correlations (`1 − mean(|Δ|)/2`, diagonal excluded).
   - **Bias score** (20%): fraction of variables within 20% mean deviation.
   - **Privacy score** (separate): avg nearest-neighbour distance, normalised to [0, 1].
3. **Ratchet**: if `overall_score > best_score`, keep the dataset and save artifacts.
4. **Pivot** on no improvement (see noise strategy below).

### Noise Pivot Strategy

| Streak | Action |
|--------|--------|
| Any single miss | Explore: `noise *= 1.1` (max 2.0) |
| Every 2nd consecutive miss | Exploit: `noise *= 0.5` (min 0.01) |
| Every 4th consecutive miss | Reset: `noise = 0.1` |

---

## What Each Phase Costs

`analysis/end_to_end_cost.py` measures the full workflow over three real papers
with GPT-4o via OpenRouter, three local timing repetitions, and three refinement
iterations. Mean cost for one finished dataset is:

| Phase | Stage | API ($) | API (s) | Network (s) | Local (s) |
|---|---|---:|---:|---:|---:|
| 0 | Query expansion | .00094 | 2.22 | — | — |
| 0 | Web search | — | — | 27.97 | — |
| 0 | Source suggestion | .00647 | 1.90 | — | — |
| 1 | Fetch | — | — | 5.09 | — |
| 1 | Parse | — | — | — | .63 |
| 1 | Extraction | .00945 | 3.69 | — | — |
| 2–3 | Synthesis, validation, refinement, artifact writing | — | — | — | .34 |
| | **One dataset** | **.01686** | **7.80** | **33.06** | **.98** |

Search and fetch dominate the 41.8-second wall time. The three LLM stages—not
just extraction—must be included when budgeting a completed dataset. Human
source-review time is outside the measurement. Raw stage rows and retained-run
provenance live in `results/end_to_end_cost_latency.csv` and
`results/end_to_end_cost_latency_run_manifest.json`.

---

## Output Artifacts

Written to `.agentdataset_cache/sessions/<run_id>/` on each improvement:

| File | Contents |
|------|----------|
| `data.csv` | Best synthetic dataset |
| `parameters.json` | Parameters that produced the best score |
| `DATACARD.md` | Full fidelity report including privacy score |

The 3 most recent session directories are kept; older ones are deleted automatically.

---

## Caveman Protocol

Internal LLM prompts use compressed, filler-free language to minimise token usage:

- No articles, no pleasantries, no hedging.
- Pattern: `[thing] [action] [reason].`
- Example: `"Mean 50. Std 10. Normal dist. Variable: income."`

This applies to the extraction system prompt only. User-facing output (DATACARD, UI) uses normal language.

---

## Guardrails

| Rule | Enforcement |
|------|-------------|
| No LLM hallucinations | JSON schema enforced in prompt; malformed responses fall back to regex |
| Iteration budget | `max_iters` slider in UI (1–10); loop exits cleanly |
| Correlation validity | Non-positive-definite matrix → `RuntimeWarning` + independent synthesis |
| Temp file cleanup | PDF temp files deleted in `finally` block regardless of extraction outcome |
| Session disk usage | Max 3 session directories retained |
