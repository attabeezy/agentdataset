# AgentDataset: Autonomous Data Factory

**AgentDataset** is an autonomous pipeline that discovers knowledge from web pages and PDFs, extracts statistical parameters using an LLM, and generates high-fidelity synthetic datasets through an iterative synthesis-validation loop.

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
| 3 — Validation | `core/validator.py` | KS-test, correlation similarity, bias check, privacy score |

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
| `DATACARD.md` | Fidelity report (KS-test, correlation, bias, privacy score) |

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
│   │   └── validator.py      # Fidelity + privacy scoring
│   └── models/
│       └── schemas.py        # Pydantic data models
├── tests/                    # 58 unit tests plus opt-in live e2e
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
