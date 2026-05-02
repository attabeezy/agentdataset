# AutoSource: Synthetic Data Factory for Africa

An autonomous agentic pipeline that transforms static PDF reports into high-fidelity synthetic datasets with verifiable DataCards.

---

## Vision

**SOMA** (Statistical Orchestration for Model Augmentation) — *Soma* also means "to learn/read" in Swahili.

AutoSource solves the "data desert" problem in Africa by autonomously generating synthetic data from verified reports when real datasets are unavailable or private.

---

## Core Architecture

```
autosource/
├── source_library/     # PDF inputs
├── results/            # Experiments, metrics, DATACARDs
├── fetch_reports.py    # World Bank API → PDF downloader
├── extract.py          # PDF → parameters.json (fixed)
├── synthesize.py       # parameters.json → data.csv (agent-modifiable)
├── validate.py         # fidelity_score + DATACARD.md (fixed)
├── program.md          # Agent instructions
├── pyproject.toml       # uv project config
└── README.md
```

---

## The Three-Phase Pipeline

### Phase 1: Extraction (`extract.py` - Fixed)
- Converts messy PDFs into clean Markdown
- Extracts statistical parameters using LLM:
  - **Variables** (Age, Income, Credit Score)
  - **Distributions** (Mean, STD, Min, Max)
  - **Correlations** (e.g., "Education ↔ Loan Repayment: r=0.42, positive")

### Phase 2: Synthesis Loop (`synthesize.py` - Agent-Modifiable)
- Generates synthetic CSV using Lightweight Gaussian Copulas
- **Fixed time budget**: 5-10 minutes per experiment
- **Auto-Ratchet loop**: Keep improvements, discard regressions
- **Metric**: Fidelity Score (KS-test + Correlation Similarity)

### Phase 3: Validation (`validate.py` - Fixed)
- Compares synthetic vs source distributions
- Generates **DATACARD.md**:
  - Provenance (source reports)
  - Bias detection
  - Privacy guardrail checks

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Fetch reports from World Bank API
uv run fetch_reports.py --region Africa --topic credit --limit 5

# 3. Extract parameters from a PDF report
uv run extract.py --pdf source_library/your_report.pdf

# 3. Run baseline synthesis (agent-modifiable)
uv run synthesize.py

# 4. Validate and generate DATACARD
uv run validate.py
```

---

## Autoloop (Agent Instructions)

1. **Setup**: Read `program.md`, verify `parameters.json` exists
2. **Baseline**: Run synthesis **as-is**, log first fidelity score
3. **Loop** (while time budget > 0):
   - Modify `synthesize.py` hyperparameters
   - `git commit` if code changed
   - Run: `uv run synthesize.py > run.log 2>&1`
   - Extract: `grep "fidelity_score" run.log`
   - Log to `results/results.tsv`: commit, score, memory, status, description
   - **Keep** if fidelity improved, **discard/revert** otherwise
4. **Output**: DATACARD.md, `data.csv`, `progress.png`

---

## Design Choices

- **Lightweight**: No external frameworks (SDV/Gretel), just NumPy/scipy
- **CPU-only**: Runs on mid-range laptops (no GPU required)
- **Verifiable**: Statistical distance metrics, not LLM "opinions"
- **Open**: Modular code for African ML community

---

## Expected Output

```
results/
├── run.log                    # Experiment logs
├── data.csv                   # Generated synthetic dataset
├── fidelity_score.txt         # Final score (e.g., 92.4)
├── DATACARD.md                # Trust package
└── progress.png               # fidelity vs experiment # chart
```

---

## Iron Laws

1. **No hallucinations**: Generator can only use parameters from `parameters.json`
2. **Open source**: All code modular, documented for African ML community
3. **Efficiency**: Must run on mid-range laptop
4. **Statistical fidelity**: KS-test p > 0.05, correlation similarity > 0.8

---

## Future: The "Auto-Data" Factory

With this skeleton, you can:

- Scale to 100s of PDF reports (Afrobarometer, World Bank, KNUST, etc.)
- Chain experiments overnight (12/hour, ~100 overnight)
- Compare fidelity across multiple extraction strategies
- Build a repository of verified synthetic datasets for African research

---

*AutoSource borrows the "Ratchet" philosophy from Karpathy's AutoResearch, but swaps neural net training for synthetic data generation — turning PDFs into verifiable, train-ready datasets.*
