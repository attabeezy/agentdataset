# AgentDataset: Assisted Data Synthesis

AgentDataset is an assisted pipeline that turns a research query into a synthetic
dataset. It discovers documents, extracts statistical parameters with an LLM
(regex fallback), synthesizes correlated data, and validates fidelity. Human review
of sources and extracted parameters is part of the intended workflow.

## Quick start

```bash
uv sync
uv run streamlit run app.py
```

Set `OPENROUTER_API_KEY` in `.env`. The app runs on Streamlit; open the printed URL.

## Pipeline

```
Query → Discovery → Extraction → Synthesis ⇄ Validation → data.csv + DATACARD.md
```

| Phase | Module | What it does |
|-------|--------|--------------|
| 0 — Discovery | `core/discovery.py` | OpenRouter (Exa) web search; fetches PDFs/HTML |
| 1 — Extraction | `core/extractor.py` | LLM extraction with regex fallback; parses PDFs via PyMuPDF |
| 2 — Synthesis | `core/synthesizer.py` | Generates correlated DataFrames (normal, uniform, gamma, categorical) |
| 3 — Validation | `core/validator.py` | KS-test, correlation error, bias, row-dispersion diagnostic |

## Output

Each run writes to `.agentdataset_cache/sessions/<run_id>/`:

| File | Contents |
|------|----------|
| `data.csv` | Best synthetic dataset from the optimization loop |
| `parameters.json` | Parameters used for the best run |
| `DATACARD.md` | Fidelity report and dispersion diagnostic |

Only the 3 most recent session directories are kept.

## Supported providers

Select in the sidebar. The matching env var is used automatically.

| Provider | Env var | Example model |
|----------|---------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| Google | `GEMINI_API_KEY` | `gemini/gemini-2.0-flash` |

Discovery uses `OPENROUTER_API_KEY`. Extraction falls back to regex when no key is set.

## Project structure

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard |
| `agentdataset/core/` | Discovery, extraction, orchestration, synthesis, validation |
| `agentdataset/models/schemas.py` | Pydantic data models |
| `agentdataset/evaluation.py` | Benchmark helpers |
| `benchmark.py` | Empirical benchmarks and ablations |
| `analysis/` | Study scripts and protocols |
| `docs/` | Project spec, experiment notes, report, NeurIPS paper |
| `results/` | Generated result CSVs |
| `tests/` | Unit tests (live e2e deselected by default) |

## Running tests

```bash
uv run pytest
```

Live, API-backed tests are marked `live_e2e` and skipped by default:

```bash
uv run pytest tests/test_e2e_live.py -m live_e2e -v
```

## Safety notes

- The pipeline does not execute LLM-generated Python; synthesis runs through the
  vetted `Synthesizer`.
- Discovery allowlists `http(s)` URLs, caps PDF downloads at 25 MiB, and cleans
  temp files.
- LLM/search calls use bounded retries with backoff and a small circuit breaker.

## Research artifacts

The NeurIPS 2026 paper, evidence-to-claim map, and result CSVs live under
`docs/` and `results/`. See `docs/PROJECT.md` for the full revision ledger.