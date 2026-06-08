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

Each run writes to `sessions/<run_id>/`:

| File | Contents |
|------|----------|
| `data.csv` | Best synthetic dataset from the optimization loop |
| `parameters.json` | Extracted parameters used for synthesis |
| `DATACARD.md` | Fidelity report (KS-test, correlation, bias, privacy score) |

Sessions are pruned automatically — only the 3 most recent are kept.

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
├── tests/                    # 38 unit tests
├── sessions/                 # Runtime session artifacts (auto-pruned)
└── artifacts/                # Downloaded research PDFs
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```
