# AutoSource Agent Instructions

This document guides the AI agent through the autonomous data generation workflow.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g., `may2`). The branch `autosource/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autosource/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — project vision and context.
   - `extract.py` — fixed PDF extraction, parameter extraction. Do not modify.
   - `synthesize.py` — the file you modify. Distribution choices, hyperparameters, generation logic.
   - `validate.py` — fixed fidelity metric computation. Do not modify.
4. **Verify extraction**: Check that `results/parameters.json` exists. If not, run `uv run extract.py --pdf <path/to/report.pdf>`.
5. **Initialize results.tsv**: Create `results/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

---

## Experimentation

Each experiment generates synthetic data and evaluates fidelity. The generation script runs for a **fixed time budget** (default 10 minutes). You launch it simply as: `uv run synthesize.py`.

**What you CAN do:**
- Modify `synthesize.py` — this is the only file you edit. Everything is fair game: distribution types, correlation methods, noise levels, number of rows, etc.

**What you CANNOT do:**
- Modify `extract.py` or `validate.py`. They are read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `compute_overall_fidelity` function is the ground truth metric.

**The goal is simple: get the highest fidelity_score.** Since the time budget is fixed, you don't need to worry about generation time — it's always ~5-10 minutes. Everything is fair game: change the distribution families, correlation methods, noise levels, etc. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is not a constraint (CPU-only generation), but **speed** is. Don't add slow components (e.g., MCMC sampling) unless they give significant fidelity gains.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the generation script as is.

---

## Output Format

Once the script finishes it prints a summary like this:

```
---
generation_seconds: 67.3
n_rows: 10000
n_columns: 5
```

The validation script then outputs:

```
Fidelity score saved to results/fidelity_score.txt: 92.4/100
```

Extract key metrics:
```
grep "^Fidelity score" results/fidelity_score.txt
grep "generation_seconds" results/run.log
```

---

## Logging Results

When an experiment is done, log it to `results/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	fidelity_score	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. fidelity_score achieved (e.g., 92.45) — use 0.0 for crashes
3. memory_gb — not applicable for CPU (use 0.0 for now)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	fidelity_score	memory_gb	status	description
a1b2c3d	92.45	0.0	keep	baseline (normal distributions)
b2c3d4e	93.12	0.0	keep	add gamma distribution for income variable
c3d4e5f	91.87	0.0	discard	switch to t-distribution (no improvement)
d4e5f6g	0.00	0.0	crash	double N_ROWS (memory overflow)
```

---

## The Experiment Loop

The experiment runs on a dedicated branch (e.g., `autosource/may2` or `autosource/may2-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `synthesize.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run synthesize.py > results/run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Run validation: `uv run validate.py` to get fidelity score
6. Read out the results: `grep "^Fidelity score" results/fidelity_score.txt`
7. If the grep output is empty, the run crashed. Run `tail -n 50 results/run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If fidelity_score improved (higher), you "advance" the branch, keeping the git commit
10. If fidelity_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous data generator trying out generation strategies. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5-10 minutes total. If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g., a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical distribution choices. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5-10 minutes then you can run approx 6-12/hour, for a total of about 60-100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

---

## Quick Reference: Distribution Types to Try

### Numerical Distributions (in `generate_variable` function):
- **Normal**: `np.random.normal(mean, std)` — symmetric, bell curve
- **Uniform**: `np.random.uniform(min, max)` — flat distribution
- **Gamma**: `np.random.gamma(shape, scale)` — skewed positive data (income, claims)
- **Beta**: `np.random.beta(a, b)` — bounded [0,1] (rates, proportions)
- **Lognormal**: `np.random.lognormal(mean, std)` — highly skewed positive data

### Correlation Methods (in `generate_correlated_dataset` function):
- **Gaussian Copula** (default): Good for mild correlations
- **t-Copula**: Better for heavy-tailed correlations
- **Simplified independence**: Faster, but ignores correlations

### Noise Levels to Try:
- **0.0**: Exact match to source stats (no noise)
- **0.05-0.1**: Low noise (usually best for fidelity)
- **0.2-0.3**: Medium noise (may help generalization)
- **0.5+**: High noise (usually hurts fidelity)

---

## Fidelity Metric Details

The fidelity score is computed as a weighted average of:
- **KS-test component (40%)**: Distribution fit (p-value threshold: 0.05)
- **Correlation similarity (40%)**: How well correlations are preserved
- **Bias score (20%)**: demographic representation balance

Overall threshold: **> 90/100** required to "keep" an experiment.
