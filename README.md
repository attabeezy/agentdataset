# AgentDataset: Assisted Data Synthesis (CLI)

AgentDataset is a CLI tool that turns a research query into a synthetic dataset.
It discovers documents, extracts statistical parameters with an LLM, synthesizes
correlated data, and validates fidelity. Human review of sources and extracted
parameters is part of the intended workflow. No Streamlit UI; no regex fallback.

## Quick start

```bash
uv sync
agentdataset doctor          # verify environment
agentdataset --help
```

Set `OPENROUTER_API_KEY` in `.env` (recommended; a single key covers discovery
and extraction). Other providers (OpenAI/Anthropic/Google) are supported via
their own keys.

## Pipeline

```
Query -> Discovery -> Extraction -> Synthesis <-> Validation -> data.csv + DATACARD.md
```

| Phase | Module | What it does |
|-------|--------|--------------|
| 0 — Discovery | `core/discovery.py` | LLM query expansion + Exa web search; fetches PDFs/HTML |
| 1 — Extraction | `core/extractor.py` | LLM extraction (chunked, merged); PDFs via pymupdf4llm |
| 2 — Synthesis | `core/synthesizer.py` | Correlated DataFrames (normal, uniform, gamma, categorical) |
| 3 — Validation | `core/validator.py` | KS-test, correlation error, bias, trust score, critique |

## CLI commands

```bash
agentdataset discover "query" [--limit N] [--json]
agentdataset run "query" [--sources N] [--all] [--output DIR]
agentdataset extract <file|url>
agentdataset synthesize <params.json> [--output DIR] [--iterations N]
agentdataset validate <data.csv> <params.json> [--source-text TEXT]
agentdataset review <claims.json> [--accept-all|--reject-all|--interactive]
agentdataset doctor [--fix]
```

## Output

Each run writes to `.agentdataset_cache/sessions/<run_id>/`:

| File | Contents |
|------|----------|
| `data.csv` | Best synthetic dataset from the agentic loop |
| `parameters.json` | Parameters used for the best run |
| `DATACARD.md` | Fidelity report, trust score, and dispersion diagnostic |

Only the 3 most recent session directories are kept.

## PDF extraction

Default backend is `pymupdf4llm` (layout-aware Markdown). Select with
`--pdf-backend {pymupdf4llm,pymupdf}`. Scanned PDFs need Tesseract (optional).

## Providers

Managed via `litellm`. Discovery uses `OPENROUTER_API_KEY`.

| Provider | Env var | Example model |
|----------|---------|---------------|
| OpenRouter (recommended) | `OPENROUTER_API_KEY` | `openrouter/auto` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| Google | `GEMINI_API_KEY` | `gemini/gemini-2.0-flash` |

Default model is `openrouter/auto`, which routes to free models when you have no
credits. Run `agentdataset doctor` to verify reachability.

## Caching

Fetched/extracted results are cached by content hash + model/prompt/schema version
under `.agentdataset_cache/cache/`. Re-runs skip redundant network and LLM work.

## Safety notes

- The pipeline does not execute LLM-generated Python; synthesis runs through the
  vetted `Synthesizer`.
- Discovery allowlists `http(s)` URLs, caps PDF downloads at 25 MiB, and cleans
  temp files.
- LLM/search calls use bounded retries with backoff and a small circuit breaker.
- Extracted parameters are reviewed before synthesis (claim review).

## Running tests

```bash
uv run pytest
```

Live, API-backed tests are marked `live_e2e` and skipped by default:

```bash
uv run pytest -m live_e2e tests/test_e2e_live.py -v
```

## Project structure

| Path | Purpose |
|------|---------|
| `agentdataset/cli.py` | Terminal CLI (primary interface) |
| `agentdataset/core/` | Discovery, extraction, orchestration, synthesis, validation |
| `agentdataset/models/schemas.py` | Pydantic data models |
| `agentdataset/core/cache.py` | Content-addressed cache |
| `tests/` | Unit tests (live e2e deselected by default) |
| `results/` | Generated result CSVs (research artifacts) |
| `docs/` | Project spec and notes |
