# Five-Lab Experiment Report (2026-10-10)

Question from `docs/experiment.md`: do the three code fixes (grounded reader,
trust score, quiet tuner) produce promising results on current models?

Short answer: **yes for reader and tuner, mechanism-only for trust** — with
two honest caveats (reasoning-model verbosity, one flaky evening). Spend was
**~$3.85 of $10** on `OPENROUTER_API_KEY` for 1,650+ extractions.

## 1. Code under test

Commits `dc35ed8` (fixes) and `d752986` (harness), plus small follow-ups in
`d92fad4` (all on `master`, tree clean):

| Fix | Files | What changed |
|---|---|---|
| Grounded reader | `agentdataset/core/extractor.py` | Grounding rules in `NATURAL_LANGUAGE_PROMPT` (explicit numbers only, `var1/var2` must be declared, SE/CI ≠ SD, negations abstain, Tables beat abstract); 8k/500 chunk+merge for long docs; `M/SD` and `±` regex forms; drop non-positive SD |
| Trust score | `validator.py`, `models/schemas.py`, `orchestrator.py` | `trust = 0.5·overall + 0.3·grounding + 0.2·plausibility`; `run_optimization_loop(..., source_text=None)` threads text through; DATACARD shows the trust layer |
| Quiet tuner | `orchestrator.py` | `INITIAL_NOISE 0.1 → 0.02`, `EXPLORE_MULT 1.1 → 1.05` (`Synthesizer` default stays 0.1 for frozen baselines) |
| Harness | `benchmark.py`, `analysis/extraction_ablation.py`, `analysis/real_paper_eval.py`, `analysis/end_to_end_cost.py` | `_FIVELAB_PROVIDERS`; `--models` over all 8 providers; `--model`/`--tag` on real-paper eval (tagged runs never rewrite shared discovery files); `usage.cost` fallback for unmapped OpenRouter IDs; e2e caps raised 12→40 attempts, $0.25→$1.00 for chunked extraction |

All 234 tests pass, including 4 new harness tests. Frozen baselines untouched
(`test_baseline_manifest` green); new outputs use `fivelab*` names.

## 2. Model panel (one per lab, upper-mid tier, IDs verified live)

`deepseek-v4.1-flash` (V4 Pro measured ~57 s/chunk — infeasible, swapped per
pre-approved fallback), `z-ai/glm-5.3`, `openai/gpt-6.1-sol`,
`anthropic/claude-sonnet-5.5`, `google/gemini-3.8-flash`. Harness fixed for all:
temperature 0, no retries, `json_object`, Caveman arm still measured.

## 3. Results vs the predeclared criteria (`experiment.md` §5)

**Extraction — PROMISING.** Challenge set, natural arm (committed: F1 .778,
hallucination .098):

| Model | F1 | Hallucination | Errors |
|---|---|---|---|
| Gemini 3.8 Flash | .881 | .017 | 0/330 |
| Claude Sonnet 5.5 | .884 | .046 | 173 (score over usable calls) |
| DeepSeek V4.1 Flash | .870 | .017 | 0/330 ($0.23 total) |
| GPT-6.1 Sol | .847 | .017 | 3 |
| GLM 5.3 | .833 | .133 | 250 (50 usable) |
| Regex (new patterns) | .272 (was .156) | — | $0 |

Hallucination → .017 on 3/5 labs confirms the mechanism (win by inventing
less). Caveman-vs-natural gap replicates on all five.

Real papers (new chunked pipeline): Gemini .248, Sonnet .240, DeepSeek .224,
Sol **.099 (below regex .165** — precision .052, over-extracts). All below the
committed gpt-4o .304, but confounded: chunk-union inflates recall (.68–.92 vs
.48) and depresses precision. **Merge-dedup is the next fix, not a verdict.**

**Trust — mechanism verified, integration pending.** Synth-from-wrong-params:
overall 100.0, trust 70.0, grounding 0.0. But `synthesize_conditions` never
passes `source_text`, so pipeline runs graded grounding=100 throughout.

**Tuner — verified.** Traces start at 0.02 and reach 99.6+ fidelity on
iteration 1 (cdc/gold). The 0.1 hole is gone.

## 4. Caveats and incidents

- Reasoning-heavy models are unfit for chunked extraction: GLM 5.3 did 73 s /
  15k tokens per chunk with 0 vars; its real-paper run was skipped ($1.10 and
  22 min saved). Its sweep seat stays on usable-call numbers with wide CIs.
- 2026-10-10 evening was flaky provider-side (Claude 173, GLM 250 API errors;
  e2e live #2 fully failed). e2e live #1 hit the old 12-attempt cap mid-run —
  kept as `end_to_end_cost_latency_*_fivelab_partial.*`; caps then raised.
- Per-dataset cost derived from manifests (no clean live run): ~$0.06
  (Gemini) to ~$0.30 (Sonnet), up from $0.017 — driven by ~6 chunk calls per
  paper, not by model prices.
- Absolute F1 is not comparable to committed tables (new models + new
  harness). The paired Caveman-vs-natural claim is what replicates.

## 5. Spend ledger (from run manifests)

Challenge sweeps: gemini $0.86, deepseek $0.23, gpt $0.91, claude $0.53,
glm $0.37. Real-paper: gemini $0.16, deepseek $0.05, gpt $0.26, claude $0.24.
Probes + partial e2e ≈ $0.25. **Total ≈ $3.85.**

## 6. Recommended next steps (highest value first)

1. Chunk-merge dedup (union now double-counts entities across chunks —
   visible as .92 recall / .05–.15 precision on real papers).
2. Thread `source_text` through `synthesize_conditions` so trust/grounding
   grade real runs, not just probes.
3. GLM-seat decision: Flash, drop, or keep-full-with-CIs.
4. Re-run e2e live on a quiet evening for a clean per-dataset number.
