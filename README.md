# AgentDataset: Assisted Data Synthesis

**AgentDataset** is an assisted workflow that discovers web pages and PDFs,
extracts statistical parameters using an LLM, and generates synthetic datasets
through an iterative synthesis-validation loop. Real-paper evaluation shows
that source selection and extracted parameters require human review.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run the dashboard
uv run streamlit run app.py
```

Then open the Streamlit UI, select an API provider, enter your key, and type a research query.

---

## Pipeline

```
Query → Discovery → Extraction → Synthesis ⇄ Validation → data.csv + DATACARD.md
```

| Phase | Module | What it does |
|-------|--------|--------------|
| 0 — Discovery | `core/discovery.py` | DuckDuckGo search for PDFs and HTML; downloads PDFs to temp files |
| 1 — Extraction | `core/extractor.py` | LLM extraction (litellm) with regex fallback; parses PDFs via PyMuPDF |
| 2 — Synthesis | `core/synthesizer.py` | Generates correlated DataFrames from parameters (normal, uniform, gamma) |
| 3 — Validation | `core/validator.py` | KS-test, correlation similarity, bias check, synthetic-row dispersion diagnostic |

---

## What It Costs

Producing **one finished dataset** from a real paper costs about **$0.0122 in
API spend and 44 seconds**, measured end to end over three real papers with
GPT-4o via OpenRouter:

| | API ($) | API (s) | Local (s) |
|---|--------:|--------:|----------:|
| Discovery (query expansion + search) | .0009 | 2.31 | 31.60 |
| Extraction (fetch + parse + LLM) | .0113 | 3.74 | 6.20 |
| Synthesis, validation, refinement | — | — | 0.21 |
| **Total** | **.0122** | **6.05** | **38.01** |

Wall time is dominated by web search, not by the LLM or the optimization loop.
Extraction cost scales with document length — full-length papers cost roughly 5×
the per-call figure measured on short benchmark snippets.

Per-stage breakdown: `results/end_to_end_cost_latency.csv`. Reproduce with
`uv run python analysis/end_to_end_cost.py` (add `--dry-run` to run without an
API key).

---

## Supported API Providers

Select in the sidebar on app load. The correct env var is pre-filled automatically.

| Provider | Env Var | Models |
|----------|---------|--------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-3.5-turbo` |
| Claude | `ANTHROPIC_API_KEY` | `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| Gemini | `GEMINI_API_KEY` | `gemini/gemini-2.0-flash`, `gemini/gemini-1.5-pro` |

Without an API key, extraction falls back to regex (mean/std pattern matching).

---

## Output

Generated files live under `.agentdataset_cache/`, which is ignored by git. Each app run writes to `.agentdataset_cache/sessions/<run_id>/`:

| File | Contents |
|------|----------|
| `data.csv` | Best synthetic dataset from the optimization loop |
| `parameters.json` | Extracted parameters used for synthesis |
| `DATACARD.md` | Fidelity report plus a non-privacy synthetic-row dispersion diagnostic |

Sessions are pruned automatically — only the 3 most recent are kept.

---

## LLM-Guided Discovery

Discovery starts by asking the configured LLM to expand the user's research prompt into several search-optimized queries focused on papers, PDFs, reports, means, standard deviations, and correlations. The original query is always included, results are deduplicated by URL, and the source suggestion step asks the LLM which discovered sources are most likely to contain statistical parameters.

If the LLM call fails or no API key is available, discovery falls back to the original query and extraction can still use the regex fallback.

---

## Project Structure

```
agentdataset/
├── app.py                    # Streamlit dashboard
├── agentdataset/
│   ├── core/
│   │   ├── discovery.py      # Search + PDF download
│   │   ├── extractor.py      # LLM/regex parameter extraction
│   │   ├── orchestrator.py   # Pipeline orchestration + optimization loop
│   │   ├── synthesizer.py    # Synthetic data generation
│   │   └── validator.py      # Fidelity + row-dispersion diagnostics
│   └── models/
│       └── schemas.py        # Pydantic data models
├── analysis/                 # Reviewer-revision studies, protocols, evidence map
├── results/                  # Released result CSVs and run manifests
├── tests/                    # 216 unit tests plus opt-in live e2e
└── .agentdataset_cache/      # Ignored runtime output, reports, sessions, artifacts
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

The live API-backed e2e test runs automatically when a matching key is present in a repo-root `.env` file, and skips otherwise:

```env
OPENAI_API_KEY=...
```

Then run:

```powershell
uv run pytest tests/test_e2e_live.py -m live_e2e -v
```

The live test writes an inspection report to `.agentdataset_cache/e2e/live_e2e_report.md`.
