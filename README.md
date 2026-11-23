# AgentDataset

AgentDataset is a command-line tool that turns published statistical reports into a synthetic tabular dataset. It discovers candidate documents, extracts marginals and pairwise correlations with a large language model, synthesizes correlated rows through a Gaussian copula, and validates the output against the declared statistics. Candidate sources and extracted claims are surfaced for review: the tool assists a researcher rather than replacing them.

The intended use case is a setting where record-level data cannot be shared but aggregate statistics are public. AgentDataset is not a privacy mechanism, and it models pairwise monotone dependence only. The [manuscript](docs/MANUSCRIPT.md) describes the evaluation and its operating envelope; the [results index](results/README.md) records the measurements behind every reported number.

## Highlights

- **LLM-only extraction.** No regex fallback. Chunked extraction, normalized merging, grounding rules, and a model fallback chain through OpenRouter.
- **Assisted workflow.** `discover` → `run` → `review` keeps a human in the loop before synthesis.
- **Agentic refinement.** An LLM proposes synthesis parameters, critiques the validation report, and decides whether to accept or adjust the next sample.
- **Reproducible artifacts.** Each synthesis step writes parameters, a data card, and a provenance manifest; fetched and extracted content is cached by content hash.
- **Honest validation.** Sampler-consistency fidelity is reported separately from a trust score that adds grounding and plausibility signals.

## Requirements

- Python 3.13 (see `pyproject.toml`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- An `OPENROUTER_API_KEY` (recommended) or keys for the individual providers you plan to use
- Optional: Tesseract, needed only for scanned PDFs

## Installation

```bash
uv sync
agentdataset doctor --fix     # creates a .env template and checks dependencies
# edit .env and set OPENROUTER_API_KEY
agentdataset doctor           # should report all checks OK
```

The default model is `openrouter/auto`. It requires no credits when free models are available, but a key is still needed for discovery and extraction.

## Quick start

```bash
# 1. Find candidate sources for a research question.
agentdataset discover "adult income dataset income by education" --limit 10

# 2. Extract statistics from the top sources and stage them for review.
agentdataset run "adult income dataset income by education" \
  --sources 3 --output .agentdataset_cache/demo

# 3. Approve or reject the extracted claims.
agentdataset review .agentdataset_cache/demo/claims.json --interactive

# 4. Synthesize from a reviewed parameters JSON, then validate it.
agentdataset synthesize parameters.json --output .agentdataset_cache/demo --iterations 3
agentdataset validate .agentdataset_cache/demo/data.csv parameters.json \
  --source-text "source passage used for extraction"
```

`run` writes `claims.json` for review. `synthesize` consumes a `Parameters` JSON; for scripted workflows it can be produced from an extraction result with the Python API:

```python
from pathlib import Path
from agentdataset.core.extractor import Extractor

params = Extractor(model="openrouter/auto", env_var="OPENROUTER_API_KEY").extract_parameters(
    text, "source-id"
)
Path("parameters.json").write_text(params.model_dump_json(indent=2), encoding="utf-8")
```

## Command reference

| Command | Purpose |
| --- | --- |
| `agentdataset discover QUERY [--limit N] [--json]` | Search and rank candidate documents. |
| `agentdataset run QUERY [--sources N] [--all] [--output DIR]` | Discover, extract, and write `claims.json` for review. |
| `agentdataset review CLAIMS.json [--interactive \| --accept-all \| --reject-all]` | Accept, reject, or defer extracted claims. |
| `agentdataset synthesize PARAMS.json [--output DIR] [--iterations N]` | Run the agent loop and write the best `data.csv`. |
| `agentdataset validate DATA.csv PARAMS.json [--source-text TEXT]` | Report fidelity, trust score, and an LLM critique. |
| `agentdataset doctor [--fix]` | Check API keys, imports, the PDF backend, cache, and model reachability. |

Every command accepts `--model` and `--env-var` where relevant; `--pdf-backend {pymupdf4llm,pymupdf}` selects the PDF parser.

## Outputs

Synthesis artifacts are written under the chosen output directory and its `sessions/<session-id>/` subdirectory:

| File | Contents |
| --- | --- |
| `data.csv` | Best synthetic table produced by the agent loop |
| `parameters.json` | Parameters used for that table |
| `DATACARD.md` | Fidelity report, trust score, and row-dispersion diagnostic |
| `run_manifest.json` | Model, token use, timing, and artifact provenance |

Idle session directories beyond the three most recent are pruned after 24 hours. Fetched and extracted content is cached under `.agentdataset_cache/cache/` and reused across runs.

## Configuration

Provider access is managed through `litellm`. Put credentials in `.env`; `doctor` reports which keys are present.

| Provider | Environment variable | Example model string |
| --- | --- | --- |
| OpenRouter (recommended) | `OPENROUTER_API_KEY` | `openrouter/auto` |
| OpenAI | `OPENAI_API_KEY` | `openrouter/openai/gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `openrouter/anthropic/claude-sonnet-4.5` |
| Google | `GEMINI_API_KEY` | `openrouter/google/gemini-2.5-flash` |

All provider examples route through OpenRouter so a single key covers discovery, extraction, and critique. PDF text is extracted with `pymupdf4llm` by default; PyMuPDF is available as a fallback backend.

## Project layout

| Path | Purpose |
| --- | --- |
| `agentdataset/cli.py` | Terminal interface and `doctor` |
| `agentdataset/core/` | Discovery, extraction, review, orchestration, synthesis, validation, caching |
| `agentdataset/models/schemas.py` | Pydantic schemas for parameters, claims, and reports |
| `benchmark.py` | Reproducible experiment suite (`--only empirical \| ablation-a \| ablation-b \| cost`) |
| `analysis/` | Additional studies and result generators |
| `results/` | Generated CSVs; see the [results index](results/README.md) |
| `docs/` | [Manuscript](docs/MANUSCRIPT.md) and [project specification](docs/PROJECT.md) |

## Testing and reproducibility

```bash
uv run pytest                                   # offline tests; live tests skipped
uv run pytest -m live_e2e -v tests/test_e2e_live.py   # requires a configured API key
uv run python benchmark.py --only empirical     # main utility table
uv run python benchmark.py --only ablation-b    # optimization-budget sweep (resumes by default)
uv run python benchmark.py --only cost          # live cost and latency measurements
```

Ablation B checkpoints after every `(text, seed)` cell and resumes automatically; `--fresh` restarts it. The baseline manifest and its guard are documented in [results/README.md](results/README.md).

## Safety and limitations

- The pipeline does not execute LLM-generated Python; synthesis runs through the vetted `Synthesizer`.
- Discovery allowlists `http(s)` URLs, caps PDF downloads at 25 MiB, and cleans temporary files.
- LLM and search calls use bounded retries with backoff and a small circuit breaker.
- Extracted claims are reviewed before synthesis. The fidelity score measures consistency with the declared parameters; it cannot detect an error in those parameters. Aggregate-only input is not a privacy guarantee.

## Support and contributing

Questions and bug reports belong in the repository issue tracker. Pull requests are welcome; please run `uv run pytest` and update the result artifacts when behavior changes. A result-backed overview is available in the [manuscript](docs/MANUSCRIPT.md), and the historical design log is in [docs/PROJECT.md](docs/PROJECT.md).

## License

AgentDataset is released under the [MIT License](LICENSE).
