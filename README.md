# AgentDataset

AgentDataset turns published statistical reports into synthetic tabular data. It discovers candidate documents, extracts marginals and pairwise correlations with an LLM, synthesizes correlated rows through a Gaussian copula, and validates the result. Source selection and extracted claims pass through an agentic review before synthesis.

The tool is CLI-only and extraction is LLM-only — there is no web UI and no regex fallback. It is intended for settings where record-level data cannot be shared but aggregate statistics are public. It does **not** provide a privacy guarantee and models pairwise dependence only; see the [manuscript](docs/MANUSCRIPT.md) for the evaluation and operating envelope.

## Install

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
agentdataset doctor --fix        # creates a .env template and checks dependencies
# set OPENROUTER_API_KEY in .env
agentdataset doctor              # all checks should pass
```

The default model is `openrouter/auto`; an API key is required even when free models are available.

## Quick start

```bash
agentdataset discover "adult income by education" --limit 10
# run extracts claims, the agent reviews them, and parameters.json is written
agentdataset run "adult income by education" --sources 3 --output .agentdataset_cache/demo
agentdataset synthesize .agentdataset_cache/demo/parameters.json \
  --output .agentdataset_cache/demo --iterations 3
agentdataset validate .agentdataset_cache/demo/data.csv \
  .agentdataset_cache/demo/parameters.json
```

Use `agentdataset review .agentdataset_cache/demo/claims.json --interactive` to override an agent decision; `--agent` re-runs the agent review on an existing claims file.

## Commands

| Command | Purpose |
| --- | --- |
| `discover QUERY [--limit N] [--json]` | Search and rank candidate documents. |
| `run QUERY [--sources N] [--all] [--output DIR]` | Discover, extract, and stage claims for review. |
| `review CLAIMS [--agent \| --interactive \| --accept-all \| --reject-all]` | Re-run agent review or apply a human override. |
| `synthesize PARAMS [--output DIR] [--iterations N]` | Run the agent loop and write the best table. |
| `validate DATA PARAMS [--source-text TEXT]` | Report fidelity, trust score, and critique. |
| `doctor [--fix]` | Check keys, imports, PDF backend, cache, and model access. |

All commands that call a model accept `--model` and `--env-var`; `--pdf-backend {pymupdf4llm,pymupdf}` selects the PDF parser.

## Workflow and outputs

```text
query → discover → extract → agent review → synthesize ⇄ validate → data.csv
```

`run` writes `claims.json` and the agent-approved `parameters.json` to the output directory. The agent accepts, rejects, or defers each claim against its evidence; a human can override those decisions with `review`. `synthesize` writes `data.csv` to the output directory and `DATACARD.md` plus `run_manifest.json` to `sessions/<session-id>/`.

Content is cached by hash under `.agentdataset_cache/cache/`; idle sessions beyond the three most recent are pruned after 24 hours.

Fidelity measures consistency with the declared parameters, not correctness of those parameters. The trust score adds grounding and plausibility signals; neither is a privacy guarantee.

## Repository layout

```text
agentdataset/
├── cli.py                   # command-line entry point and doctor
├── core/
│   ├── discovery.py          # search, rank, and fetch PDF/HTML sources
│   ├── extractor.py          # text/PDF → Parameters (LLM only)
│   ├── review.py             # agent claim review + human override
│   ├── reconciliation.py     # study-aware merge of multiple sources
│   ├── orchestrator.py       # agent loop and end-to-end coordination
│   ├── synthesizer.py        # parametric marginals + Gaussian copula
│   ├── synthesis_engine.py   # engine interface and registry
│   ├── validator.py          # KS, correlation, bias, trust, critique
│   ├── cache.py              # content-addressed cache
│   ├── retry.py              # bounded retries and circuit breaker
│   └── manifest.py           # provenance manifests
├── models/schemas.py         # Parameters, Claim, FidelityReport
└── evaluation.py             # provenance and bootstrap helpers

benchmark.py                  # empirical, ablation, and cost suites
analysis/                     # extended studies and report generators
docs/                         # manuscript and project specification
results/                      # generated CSVs and evidence index
tests/                        # offline suite plus opt-in live tests
```

## Configuration

Provider access is handled by `litellm`; store credentials in `.env`.

| Provider | Environment variable | Example model |
| --- | --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/auto` |
| OpenAI | `OPENAI_API_KEY` | `openrouter/openai/gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `openrouter/anthropic/claude-sonnet-4.5` |
| Google | `GEMINI_API_KEY` | `openrouter/google/gemini-2.5-flash` |

## Tests

```bash
uv run pytest
uv run pytest -m live_e2e -v tests/test_e2e_live.py     # requires API credentials
uv run python benchmark.py --only empirical            # utility table
uv run python benchmark.py --only ablation-b           # resumes from checkpoint
uv run python benchmark.py --only cost                 # live cost measurements
```

Add `--fresh` to `ablation-b` to discard its checkpoint. Result files and the baseline manifest are documented in the [results index](results/README.md).

## Documentation

- [Manuscript](docs/MANUSCRIPT.md) — method, experiments, and result tables
- [Results index](results/README.md) — measurements and provenance
- [Project specification](docs/PROJECT.md) — design history and planned work
- [MIT License](LICENSE)
