# Experiment: Five-Lab Rerun (October 2026)

Simple question: do the three code fixes (grounded reader, trust score, quiet
tuner) produce promising results on current models?

## 1. Code under test (uncommitted, see `git diff`)

| Fix | File | Change |
|---|---|---|
| Grounded reader | `agentdataset/core/extractor.py` | Grounding rules in prompt; 8k/500 chunk+merge; `M/SD` and `±` regex; drop non-positive SD |
| Trust score | `agentdataset/core/validator.py`, `models/schemas.py` | `trust = 0.5·overall + 0.3·grounding + 0.2·plausibility`; `run_optimization_loop(..., source_text=)` threads text through |
| Quiet tuner | `agentdataset/core/orchestrator.py` | `INITIAL_NOISE 0.1 → 0.02`, `EXPLORE_MULT 1.1 → 1.05` (`Synthesizer` default stays 0.1 for frozen baselines) |

## 2. Model panel (one per lab, upper-mid tier, all via `OPENROUTER_API_KEY`)

| Lab | Model | Rationale |
|---|---|---|
| DeepSeek | `deepseek-v4-pro` (fallback `v4.1-flash`) | Best open-weight (MIT) |
| GLM (Z.ai) | `glm-5.3` | Full model, not the Flash cut-down |
| OpenAI | `gpt-6.1-sol` | Value-frontier, near-Astra at 1/5 price |
| Anthropic | `claude-sonnet-5.5` | One step below Opus 5.5, Sep 2026 fresh |
| Google | `gemini-3.8-flash` | Fastest mid-tier; Flash class already tied gpt-4o here |

Verify exact IDs against OpenRouter `/api/v1/models` before spending.
Harness is fixed for all arms: temperature 0, no retries, `json_object`
response format. Caveman arm still measured so the §9.6 comparison replicates.

## 3. Rerun matrix (cheapest first, most expensive last)

| Order | Study | Command | Cost | Paid? |
|---|---|---|---|---|
| 1 | Local suites | `uv run pytest tests/ -q` | $0 | No |
| 2 | Regex dry-run | `extraction_ablation.py --dry-run` | $0 | No |
| 3 | Utility / calibration / audits | `p0_utility_benchmark.py`, `adult_calibration.py`, `corr_validity_audit.py`, `fidelity_validity_probe.py`, `noise_pivot_ablation.py --quick` | $0 | No |
| 4 | Real-paper cached (all 5 models) | `real_paper_eval.py --mode cached` | ~$0.25 | Yes |
| 5 | Cost/latency refresh | `run_cost_latency_analysis()` → `extraction_cost_latency.csv` | ~$0.02 | Yes |
| 6 | Challenge sweep (5 models) | `extraction_ablation.py --reps 5` | ~$2.60 | Yes |
| 7 | End-to-end live | `end_to_end_cost.py --mode live` | ~$0.10 | Yes |

Budget: **~$3 of $10** on the key. Abort after any step that answers the
question against the new code.

## 4. Freeze rules (§9.1 — non-negotiable)

- Record the commit hash before regenerating anything.
- The six files in `results/baseline_manifest.json` are **never overwritten**;
  new outputs go to new filenames (e.g. `extraction_challenge_fivelab_*.csv`).
- `test_baseline_manifest.py` enforces this by hash — if it fails, stop.

## 5. What "promising" means (predeclared)

- Extraction: natural-arm F1 beats the committed .778 on the challenge set,
  or any lab's real-paper F1 clearly beats .304 — with precision (not recall)
  doing the work, per the grounding-rules hypothesis.
- Trust: `trust_score` separates corrupted from clean params where
  `overall_score` is invariant (probe: synth-from-wrong-params → overall ~100,
  trust ≪ overall). Already verified once locally (100.0 vs 70.0).
- Tuner: loop starting at 0.02 reaches ≥99 mean fidelity within 1–3 iterations
  on the noise-pivot grid; fixed-0.01-vs-schedule gap shrinks or vanishes.

## 6. Docs map (post-cleanup; root keeps only `README.md`)

- `../README.md` — user guide. `PROJECT.md` (same dir) — architecture +
  revision spec. `experiment.md` (this file) — current experiment plan.
- `../results/README.md` — results index + reproduce commands.
- `../analysis/EVIDENCE_TO_CLAIM_MAP.md` (+ `.csv` twin) — claim provenance.
- `../analysis/*_PROTOCOL.md` — annotation rules (real-paper, challenge).
- `../program.md` removed 2026-10-10: stale (Caveman-as-current, "privacy
  score"), zero references, superseded by `PROJECT.md` + `README.md`.
- `../benchmark.py` is shared library code (imported by 5 analysis scripts +
  4 test files) — not clutter, do not delete.
