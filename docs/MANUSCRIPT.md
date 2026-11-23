# MANUSCRIPT.md — AgentDataset: Assisted Synthesis of Datasets from Published Statistics

> Draft manuscript. Every reported number is taken from a CSV under `results/`;
> the table captions name the source file. Research artifacts under `analysis/`
> and `results/` remain the source of truth for every claim.

---

## Title

AgentDataset: An Assisted Pipeline for Synthesizing Datasets from Published
Statistical Reports

## Abstract

We present AgentDataset, a command-line tool that turns a research query into a
synthetic tabular dataset by (1) discovering candidate documents, (2) extracting
statistical parameters (marginals and pairwise correlations) with a large
language model, (3) synthesizing correlated rows through a Gaussian copula over
parametric marginals, and (4) validating the result against the declared
statistics. The system is *assisted*, not autonomous: source selection and
extracted claims are reviewed by an LLM agent before synthesis (with a human
override), and every generated artifact carries a provenance manifest. We reframe the internal
fidelity score as a sampler-consistency diagnostic rather than evidence of
source fidelity, and add a trust score that combines grounding, plausibility,
and claim grounding. A corruption probe shows that the sampler-consistency score
alone is blind to upstream parameter error. We evaluate on four public datasets
(Adult, Pima, German Credit, Concrete) and a real-paper study. AgentDataset
reproduces per-variable marginals and pairwise monotone association; it does not
recover higher-order dependence, and aggregate-only input does not constitute a
privacy mechanism.

## 1. Introduction

Data scarcity and access restrictions limit reproducible research in medicine,
finance, and the social sciences. Published papers, however, routinely report
the summary statistics needed to reconstruct a plausible population: means,
standard deviations, category probabilities, and pairwise correlations.
AgentDataset automates the tedious translation from these aggregates to a
usable table while keeping a human in the loop.

Contributions:
1. An end-to-end assisted workflow (discovery, LLM extraction, synthesis,
   validation) exposed as a CLI with an agentic review gate.
2. An LLM-driven agentic loop that proposes, synthesizes, validates, and
   critiques, replacing a fixed hyperparameter schedule.
3. A copula-based synthesizer with correlation diagnostics and an opt-in
   nearest-PSD repair.
4. An honest validation story: sampler-consistency fidelity plus a trust score
   that adds grounding, plausibility, and claim-grounding signals, and an
   explicit operating envelope (pairwise monotone dependence only).

## 2. Related Work

### 2.1 Synthetic data from aggregate statistics

Classical approaches reconstruct populations from published tables using
iterative proportional fitting and copulas. SDV (Patki et al., 2016) and
similar tools fit synthesizers to *record-level* training data and serve as
data-access upper bounds rather than summary-only competitors. Copula-based
methods (Joe, 2014; Trivedi & Zimmer, 2006) model dependence through marginal
CDFs and a latent correlation structure; vine copulas (Aas et al., 2009) extend
this to higher dimensions but require richer inputs than a typical paper
provides. We adopt a Gaussian-copula-style rank transform because Pearson
correlations are the dependence statistic most often reported.

### 2.2 LLM extraction from scientific literature

LLMs have been applied to information extraction from clinical and scientific
text (Nasar et al., 2018; Liu et al., 2025; Song et al., 2025; Sror et al.,
2026). Common
failure modes include hallucinated numbers, conflating standard errors with
standard deviations, and context/scope errors (wrong population, arm, or time
point). We mitigate these with grounding rules in the prompt, abstention on
insufficient evidence, chunked extraction with merge-by-normalized-name, and a
claim-level grounding score in validation.

### 2.3 Privacy and disclosure control

Aggregate-only input reduces reliance on record-level training data but does not
establish anonymity or differential privacy. We follow the disclosure-control
literature (Hundepool et al., 2012; Rubin, 1993; Taub et al., 2018) and
reframe the nearest-neighbour value as a synthetic-row dispersion diagnostic,
not a privacy audit. Threats include small-cell disclosure, group-attribute
inference, differencing/composition across publications, and reconstruction
from sufficient statistics.

### 2.4 Synthetic data evaluation

Utility is commonly measured by train-on-synthetic/test-on-real (TSTR) versus
TRTR. We report logistic regression on the main three-dataset benchmark and, in
the extended utility study, logistic regression, gradient-boosted trees, and a
regression task over multiple seeds, with independent-marginals and SDV
upper-bound baselines.

## 3. Method

### 3.1 Discovery

`DiscoveryAgent` expands the query with an LLM, searches via OpenRouter's
Exa-backed search, deduplicates by normalized URL, ranks candidates by a
statistical-availability heuristic, and fetches content (HTML via trafilatura,
PDFs via a size-capped stream download). Content is cached by URL.

### 3.2 Extraction

`Extractor.extract_parameters` chunks long text (8000 chars, 500 overlap) and
calls an LLM per chunk with a natural-language prompt enforcing grounding
rules. Results are merged by normalized variable name and correlation pair.
Implausible outputs (non-positive std) are dropped. A fallback chain
(`openrouter/auto` then other models) improves robustness. No regex fallback.
The prompt asks for a correlation magnitude and a direction; the schema folds
the direction into the sign of the stored value before synthesis or validation,
so a reported negative association cannot be applied as a positive one.

### 3.3 Agentic review

Each extracted claim carries a value and an evidence passage. Before synthesis,
an LLM agent accepts, rejects, or defers every claim against that evidence and
the declared population, arm, and time point; rejected and deferred claims are
excluded from the synthesis specification. A human can override the agent's
decisions through the `review` command. Accepted continuous statistics are
grouped by variable name, and accepted correlation claims populate the Gaussian
copula.

### 3.4 Synthesis

`Synthesizer` draws parametric marginals (normal, uniform, gamma, categorical)
and imposes pairwise correlations via a rank transform over a latent Gaussian.
Diagnostics record declared/applied/skipped pairs and matrix conditioning. An
opt-in nearest-PSD repair is available. Categorical variables are drawn as
integer codes and rank-transformed (ordinal coding is a documented limitation).

### 3.5 Validation

`Validator` reports KS (or chi-square for categorical) fit, correlation
similarity, bias, and a synthetic-row dispersion diagnostic. `plausibility`,
`grounding`, and `claim_grounding` scores feed a `trust_score`. An LLM critique
summarizes fidelity and suggests next actions.

### 3.6 Agentic loop

`Orchestrator.run_agent_loop` iterates synthesize-validate-critique, asking an
LLM to choose the next action (continue, accept, stop, adjust noise, repair
correlations). Artifacts are written on each improvement.

## 4. Experiments

All live calls were routed through OpenRouter via `litellm`. Unless stated, the
extraction model is `openrouter/openai/gpt-4o`, the agent-loop model is
`openrouter/auto`, and five seeds control data splits and synthesis. Metrics are
reported as mean ± std across seeds.

### 4.1 Downstream utility

Three UCI classification datasets: Adult (demographic), Pima Indians Diabetes
(medical), and German Credit (financial). One LLM extraction is run per dataset
from a literature-style summary of the real data. Three synthesizers are
compared: AgentDataset (three agent-loop iterations), independent marginals, and
an SDV Gaussian copula fitted to the record-level training split (data-access
upper bound). A logistic regression is trained on each synthetic training split
and evaluated on the held-out real test split (TSTR), against
train-real/test-real (TRTR). We report accuracy, F1, and ROC-AUC over five seeds.

### 4.2 Extended utility and calibration

- `analysis/p0_utility_benchmark.py`: Adult, Pima, German Credit, and Concrete
  (regression); 30 seeds; gold summaries computed from each training split
  (no LLM extraction); AgentDataset vs one-shot copula vs independent vs SDV;
  logistic regression, gradient-boosted trees, and regression metrics.
- `analysis/adult_calibration.py`: calibration strategies on Adult (30 seeds).
- `analysis/adult_f1_gap.py`: decomposition of the Adult minority-F1 gap.

### 4.3 Extraction

- Synthetic ground-truth texts: eight domains × three repetitions, scored on
  success rate, relative error, and latency.
- A 30-case challenge set (`analysis/extraction_challenge_set.json`) spanning
  ten failure categories, five repetitions each: Caveman vs natural-language
  prompts across three models, scored with precision/recall/F1,
  hallucination, and abstention.
- Correlation validity audit: declared vs realized correlations, including
  stress specifications.

### 4.4 Optimization loop

Eight domains × five seeds × iteration budget {0, 1, 2, 3, 5, 10} = 240 cells.
Each cell constructs a fresh `Synthesizer(seed=seed)`, so the seed is a real
control and budgets can be paired within a seed. Budget 0 is one-shot synthesis
at the standalone default noise (0.1), while the agent loop starts at 0.02; a
separate noise-matched control synthesizes once at 0.02 for the same 40
(text, seed) cells (`results/one_shot_noise_matched.csv`). Larger budgets run
the LLM-controlled agent loop. We report fidelity (sampler consistency) and
paired tests of 0 vs 5 iterations.

### 4.5 Cost and latency

Per-call latency, USD cost, and token usage for the extraction prompt on three
OpenRouter-routed models (five repetitions), plus measured Caveman-vs-natural
language token counts.

### 4.6 Real-paper study

Three heterogeneous documents are run through discovery, parsing, extraction,
and synthesis; each stage is scored separately, with end-to-end cost/latency
reported.

The main benchmark tables are regenerated with `uv run python benchmark.py
--only empirical`, `--only ablation-b` (add `--fresh` to discard the checkpoint),
and `--only cost`.

## 5. Results

### 5.1 Downstream utility

AgentDataset's TSTR accuracy is within 0.007 of TRTR on all three datasets
(Adult 0.809 vs 0.816, Pima 0.769 vs 0.771, German 0.689 vs 0.693) and ROC-AUC
is within 0.006. SDV, which fits the record-level training split and therefore
has strictly more information, gives comparable AUC (0.852 / 0.825 / 0.605) but
lower accuracy and minority F1 on Adult (0.772 / 0.133) than AgentDataset
(0.809 / 0.504). Independent marginals collapse to the majority class on Adult
(F1 0.000) and lose association structure (AUC 0.40–0.60). F1 values depend on
which encoded class is treated as positive; AUC is the class-orientation-free
comparison. All numbers were produced after the correlation-direction fix
described in Section 3.2. The high fidelity scores are sampler-consistency
scores only, as quantified below.

Table 1. TSTR vs TRTR on the three-dataset benchmark
(`results/empirical_benchmark.csv`; mean ± std, five seeds;
fidelity is defined only for AgentDataset, which validates against the extracted
parameters).

| Dataset | Synthesizer | TRTR acc | TSTR acc | TSTR F1 | TSTR AUC | Fidelity |
|---------|-------------|----------|----------|---------|----------|----------|
| Adult | AgentDataset | 0.816 ± 0.003 | 0.809 ± 0.004 | 0.504 ± 0.012 | 0.850 ± 0.005 | 97.56 ± 3.56 |
| Adult | Independent | 0.816 ± 0.003 | 0.761 ± 0.000 | 0.000 ± 0.000 | 0.599 ± 0.057 | — |
| Adult | SDV copula | 0.816 ± 0.003 | 0.772 ± 0.002 | 0.133 ± 0.015 | 0.852 ± 0.006 | — |
| Pima | AgentDataset | 0.771 ± 0.020 | 0.769 ± 0.021 | 0.838 ± 0.014 | 0.823 ± 0.029 | 98.77 ± 0.31 |
| Pima | Independent | 0.771 ± 0.020 | 0.649 ± 0.000 | 0.787 ± 0.000 | 0.397 ± 0.284 | — |
| Pima | SDV copula | 0.771 ± 0.020 | 0.758 ± 0.019 | 0.835 ± 0.013 | 0.825 ± 0.029 | — |
| German | AgentDataset | 0.693 ± 0.009 | 0.689 ± 0.011 | 0.811 ± 0.009 | 0.615 ± 0.038 | 99.47 ± 0.17 |
| German | Independent | 0.693 ± 0.009 | 0.700 ± 0.000 | 0.824 ± 0.000 | 0.465 ± 0.081 | — |
| German | SDV copula | 0.693 ± 0.009 | 0.698 ± 0.003 | 0.821 ± 0.004 | 0.605 ± 0.037 | — |

### 5.2 Extended utility, calibration, and score validity

The 30-seed extended study uses gold summaries computed from each training
split, so it isolates synthesis quality from LLM extraction
(`results/p0_utility_summary.csv`); it adds a one-shot copula, independent
marginals, and SDV, and repeats the comparison with a gradient-boosted learner.
Table 2 reports the primary metric. AgentDataset
matches SDV within about 0.03 accuracy on the three classification tasks and
beats independent marginals by 0.2–10.8 accuracy points (German Credit, whose
classes are nearly balanced, is essentially tied). On the Concrete regression
task SDV is substantially better (R² 0.62 vs 0.53), which is expected: SDV sees
the record-level training split, and nonlinear regression benefits from
dependence structure beyond the reported pairwise correlations. The agent loop
adds little over one-shot synthesis at 30 seeds (Adult 0.799 vs 0.799, Pima
0.757 vs 0.756, German 0.702 vs 0.699, Concrete R² 0.528 vs 0.497).

Table 2. Extended TSTR utility, 30 seeds, mean ± std
(`results/p0_utility_summary.csv`).

| Dataset | Learner | TRTR | AgentDataset | One-shot | Independent | SDV |
|---------|---------|------|--------------|----------|-------------|-----|
| Adult | logistic reg. (acc) | 0.814 | 0.799 ± 0.003 | 0.799 ± 0.003 | 0.761 ± 0.000 | 0.770 ± 0.003 |
| Pima | logistic reg. (acc) | 0.769 | 0.757 ± 0.029 | 0.756 ± 0.028 | 0.649 ± 0.001 | 0.757 ± 0.029 |
| German | logistic reg. (acc) | 0.704 | 0.702 ± 0.012 | 0.699 ± 0.011 | 0.700 ± 0.000 | 0.701 ± 0.007 |
| Concrete | gradient boosting (R²) | 0.926 | 0.528 ± 0.068 | 0.497 ± 0.086 | −0.256 ± 0.143 | 0.622 ± 0.089 |

The Adult minority-F1 gap is largely a calibration artifact rather than only a
dependence failure (`results/adult_calibration.csv`,
`results/adult_f1_gap.csv`). At the default 0.5 threshold the TSTR classifier
predicts a positive rate of 0.076 against a true rate of 0.239, giving minority
F1 0.34–0.37; choosing the threshold on synthetic validation raises minority F1
to 0.61, close to the oracle threshold on real test labels (0.62), while ROC-AUC
is unchanged at 0.85.

A corruption probe quantifies the score-validity problem
(`results/fidelity_validity_probe_summary.csv`). We perturb the declared
parameters (mean shift, standard-deviation scaling, swapped statistics) before
synthesis. Reported fidelity, correlation, and KS scores are identical at every
corruption level (Adult: 95.6, 99.1, and 90.0 respectively), because the
validator compares generated rows with the *declared* parameters. An external
KS distance against the real data rises from 0.143 at ε = 0 to 0.709 under a
two-standard-deviation mean shift. The internal fidelity score therefore
certifies sampler consistency, not source fidelity, and cannot by itself detect
a parameter error introduced upstream.

### 5.3 Extraction and correlation validity

On eight synthetic literature-style domains with three repetitions
(`results/ablation_extraction_method.csv`), `gpt-4o` extracted all 24 cells with
zero relative error on mean and standard deviation, at 1.91 s per call. This
set is deliberately easy: the numbers are stated explicitly and unambiguously.

The 30-case challenge set is the informative extraction test
(`results/extraction_challenge_summary.csv`, five repetitions per case and
condition). Natural-language prompting beats the compressed Caveman prompt for
all three models on F1 and hallucination rate. With `gpt-4o`,
natural-language prompting reaches precision 0.74, recall 1.00, F1 0.79, and a
0.093 hallucination rate; the Caveman prompt reaches F1 0.76 and 0.136.
Gemini 2.5 Flash is close (F1 0.79 vs 0.71), and Claude Sonnet 4.5 is weakest
(F1 0.76 vs 0.68). Variable-name accuracy remains low (0.44–0.56), confirming
that value extraction is easier than assigning the correct variable identity.

Correlation validity (`results/corr_validity_summary.csv`) is high when the
declared matrix is positive semidefinite: correlation scores are 97.6–99.7 for
gold, templated, and automatically extracted specifications, with every declared
pair applied. The failure mode is non-PSD input. On five stress specifications
only one matrix is positive definite (mean minimum eigenvalue −0.26); without
repair the correlation score drops to 66.6 ± 19.1 and the overall score to
52.7 ± 9.0. Opt-in nearest-PSD repair raises them to 92.5 ± 9.8 and 63.0 ± 10.4.
Repair improves correlation fit but cannot invent the missing dependence
information.

### 5.4 Optimization budget

Fidelity rises steeply with the first few iterations, but most of that jump is
not iteration count. The budget-0 arm uses the standalone default noise of 0.1
and scores 59.82 ± 0.09; the agent loop starts at 0.02. A noise-matched one-shot
control at 0.02 scores 95.82 ± 8.14 — exactly the same cells as the one-iteration
arm, where the LLM accepts the first sample without changing anything. Three or
more iterations reach 99.84 with almost no variance. The paired 0-vs-5 contrast
is +40.02 fidelity (n = 40, paired t p = 2.3e-123, Wilcoxon p = 2.1e-8), but the
informative comparison is the noise-matched control: 3 iterations beat it by
+4.02 (paired t p = 0.0034, Wilcoxon p = 0.011). The remaining gain comes from
extra accept/reject steps after the noise is already well chosen, not from the
initial noise adjustment.

Table 3. Fidelity by agent-loop iteration budget (40 paired cells per row;
`results/ablation_noise_pivot.csv`, `results/one_shot_noise_matched.csv`).

| Iterations | Mean fidelity | Std |
|------------|---------------|-----|
| 0 (noise 0.1) | 59.82 | 0.09 |
| 0 (noise-matched control, 0.02) | 95.82 | 8.14 |
| 1 | 95.82 | 8.14 |
| 2 | 99.34 | 3.16 |
| 3 | 99.84 | 0.08 |
| 5 | 99.84 | 0.08 |
| 10 | 99.84 | 0.08 |

### 5.5 Cost and latency

Extraction cost is small per call but varies by provider
(`results/extraction_cost_latency.csv`, five calls per model). Gemini 2.5 Flash
is roughly 4× cheaper than `gpt-4o` and 7× cheaper than Claude Sonnet 4.5 at
comparable or lower latency.

Table 4. Extraction-prompt cost and latency, mean ± std over five calls.

| Model | Latency (s) | Cost / call (USD) | Prompt tokens | Completion tokens |
|-------|-------------|-------------------|---------------|-------------------|
| `openai/gpt-4o` | 2.00 ± 0.53 | 0.0023 | 363 | 140 |
| `anthropic/claude-sonnet-4.5` | 3.95 ± 0.34 | 0.0042 | 379 | 201 |
| `google/gemini-2.5-flash` | 2.15 ± 0.56 | 0.0006 | 367 | 194 |

The Caveman compression saves 63.3% of prompt tokens (515 → 189;
`results/extraction_prompt_token_savings.csv`), but the Section 5.3 challenge
shows it loses 2.8–8.9 F1 points, so the natural-language prompt remains the
default. In the full pipeline these calls are a small part of the end-to-end
wall time; discovery dominates (Section 5.6).

### 5.6 Real-paper study

Three heterogeneous documents (a randomized trial, an NHANES rural/urban
comparison, and a student mental-health survey) were run through discovery,
parsing, extraction, synthesis, and validation; each specification is evaluated
over 30 synthesis seeds (`results/real_paper_pipeline_summary.csv`). LLM
extraction is close to the hand-curated gold specification: fidelity 97.4 vs
98.6 on the trial, 61.6 vs 63.3 on the NHANES report, and 96.8 vs 98.5 on the
survey; correlation scores are 96.7–99.4 in both conditions. The NHANES
specification is intrinsically hard for this sampler because much of its
reported structure is categorical and coarse, and the fidelity score reflects
that even with gold inputs.

End-to-end, one dataset costs about $0.017 and 42 s of wall time on average
($0.0169, 41.8 s; `results/end_to_end_cost_latency.csv`). Of the 42 s, 33 s is
network time dominated by web search and document fetch, 7.8 s is LLM API time
across three calls (query expansion, source suggestion, extraction), and under
1 s is local synthesis and validation. Cost and latency are therefore dominated
by discovery, not by the statistical pipeline.

## 6. Discussion and limitations

- Fidelity score is sampler-consistency, not source fidelity; the corruption probe
  shows it is unchanged under parameter error, so trust/grounding must carry that
  weight.
- Pairwise monotone dependence only; no higher-order structure. Record-level SDV
  is better on nonlinear regression (Concrete R² 0.62 vs 0.53).
- Categorical dependence uses ordinal codes.
- Extraction is accurate on explicit numbers but weak on variable identity and
  scope: name accuracy is 0.44–0.56 and hallucination 9–20% on the adversarial
  challenge set. Abstention is unreliable.
- Threshold choice, not only dependence loss, drives the Adult minority-F1 gap.
- No formal privacy guarantee; aggregate inputs may still disclose.

## 7. Conclusion

AgentDataset provides an assisted, auditable path from published statistics to
synthetic data, with explicit operating limits and honest validation. Future
work: claim-level evidence, study-aware reconciliation, validation against
independent evidence, and stronger categorical dependence models.

## References (selected)

- Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). Pair-copula
  constructions of multiple dependence. *Insurance: Mathematics and Economics*.
- Hundepool, A., et al. (2012). *Statistical Disclosure Control*. Wiley.
- Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
- Liu, L., et al. (2025). Human level information extraction from clinical
  reports with finetuned language models. *Scientific Reports*.
  doi:10.1038/s41598-025-28767-z.
- Nasar, Z., Jaffry, S. W., & Malik, M. K. (2018). Information extraction from
  scientific articles: a survey. *Scientometrics*. doi:10.1007/s11192-018-2921-5.
- Patki, N., Wedge, R., & Veeramachaneni, K. (2016). The Synthetic Data Vault.
  *DSAA*.
- Rubin, D. B. (1993). Statistical disclosure limitation. *J. Official
  Statistics*.
- Song, Z., et al. (2025). A scientific-article key-insight extraction system
  based on multi-actor of fine-tuned open-source large language models.
  *Scientific Reports*. doi:10.1038/s41598-025-85715-7.
- Sror, G., et al. (2026). AI-driven methodology for mining scientific
  literature and extracting databases: A case study on a chemical process.
  *Energy and AI*. doi:10.1016/j.egyai.2026.100741.
- Taub, J., Elliot, M., & Pampaka, M. (2018). The impact of synthetic data
  generation on data utility. *UNECE*.
- Trivedi, P. K., & Zimmer, D. M. (2006). Copula modeling. *Foundations and
  Trends in Econometrics*.
