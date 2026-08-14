# Extraction Challenge Set — Annotation Protocol

Companion to `analysis/extraction_challenge_set.json` and the P1 §9.6 ablation in
`analysis/extraction_ablation.py`. Sister document to
`REAL_PAPER_ANNOTATION_PROTOCOL.md`, which governs the source-level ground truth
this challenge set reuses.

## Purpose

The pre-existing extraction ablation scored LLM and regex extraction on eight
formulaic synthetic sentences of the form *"We observed a heart rate distribution
with mean 72 and std 8."* Those texts are trivially parseable and contain no
distractors, so they measure transcription, not extraction. They also could not
expose a false positive: the legacy scorer (`benchmark._score_extraction`) matches
each gold variable to its nearest extracted mean, so an extractor emitting fifty
junk variables scores identically to one emitting two correct ones.

This challenge set is built to be hard in the specific ways real papers are hard,
and is annotated so that three outcomes are distinguishable:

1. **correct extraction** — the reported value, reproduced,
2. **correct abstention** — emitting nothing when the document supports nothing,
3. **context error / hallucination** — emitting a number that is in the document
   but is not a distribution parameter, or emitting one that is not there at all.

## Composition

30 cases across 10 categories, three per category: 11 real excerpts, 19
constructed.

| Category | What it probes |
|---|---|
| `negated_finding` | A numeric result whose claim the source declines to assert ("a trend", "no significant association") |
| `multiple_populations` | The same variable reported for several arms, sites or rural/urban strata |
| `conflicting_sections` | Abstract, table and caption disagreeing about the same quantity |
| `repeated_timepoints` | One variable at baseline / 6 mo / 12 mo, with numeric timepoint labels |
| `units_mismatch` | One quantity restated in a second unit system (mmol/L and mg/dL) |
| `confidence_intervals` | CI bounds that must not become min/max or be read as the point estimate |
| `table_footnotes` | Significance markers fused to values (`201.6 (26.3)b`) and footnote glosses |
| `scientific_notation` | `1.2e-3`, `2.5E+04`, `6.02 x 10^5` |
| `irrelevant_numbers` | p-values, sample sizes, instrument ranges, Cronbach's α, DOIs, years, page numbers |
| `sd_vs_se` | A standard error presented where the schema wants a standard deviation |

## Provenance rules

**Real cases** (`"provenance": "real"`) are verbatim excerpts from the three
open-access sources already cached under `.agentdataset_cache/real_papers/` and
catalogued in `real_paper_manifest.csv`. They are **never typed by hand**.
`build_extraction_challenge_set.py` declares each excerpt as a pair of anchor
strings, slices the cached text at those anchors, and records the resulting
character offsets plus the source file's sha256. Re-running the builder against a
changed source fails loudly rather than silently drifting.

Where an excerpt spans non-contiguous regions (a table header plus one data row),
the pieces are joined with the literal marker `\n[...]\n` so the elision is
visible to both the model and the reader. Each piece is individually verbatim.

Excerpts are kept short — prose cases run to roughly 40 words; table cases are
longer only because a table row is meaningless without its column headers.

**Constructed cases** (`"provenance": "constructed"`) carry their text inline and
every one opens with the words "In this constructed …". They are synthetic
examples written to isolate a failure mode, and must never be presented or cited
as real literature. Results are reported with the real/constructed split visible
(`extraction_challenge_by_category.csv`, and the counts in the run manifest) so a
reader can weigh them separately.

## Gold annotation

Gold values for real cases are **taken from**, not re-derived for, the dual-pass
adjudicated `real_paper_ground_truth.csv` (columns `pass_1`, `pass_2`,
`adjudication`). Where that file records a value the annotators flagged as odd —
the CDC age SD of 0.4 years, the DUFSSQ-8 total lying outside both subgroup means —
the gold follows the **adjudicated, verbatim** reading. Extraction is scored
against what the source says, not against a reinterpretation of what it ought to
have said; the oddity is what the case is testing.

Each gold variable carries a `variant_group`. Several entries share a group when
the source legitimately reports one variable for several populations, arms or
timepoints. Reporting **any one** member satisfies the group; reporting several is
also correct. Recall is therefore computed over groups, not over entries — an
extractor is not penalised for choosing the urban column over the rural one when
the excerpt gives no basis to prefer either.

`must_not_extract` lists values that appear in the text but are not distribution
parameters, each with a stated reason. This is the list that makes context errors
countable.

## Scoring (`benchmark._score_challenge_case`)

Matching is by **value, not by name**. The regex fallback emits generic `var_N`
names, so name-based matching would zero it out for a reason unrelated to whether
it read the numbers correctly. Name correctness is reported separately as
`name_accuracy`. A variable matches a gold entry when both its mean and its SD
fall within a relative tolerance of 1e-3 — tight on purpose, because extraction
should reproduce reported values, not approximate them.

False positives are split by cause:

| Outcome | Meaning |
|---|---|
| `fp_distractor` | Matches a `must_not_extract` value → **context error** |
| `fp_fabricated` | Appears nowhere in the source text → **hallucination** |
| `fp_conflated` | Mean from one population paired with the SD of another |
| `fp_other` | Present in the text, but neither gold nor annotated |

Hallucination is grounded in `_numbers_in_text`, which enumerates every numeric
literal a reader could lift from the excerpt (handling thousands separators,
exponential forms, and both readings of a percentage — 77.6 and 0.776). A value
outside that set was not read out of the document.

**Abstention is rewarded, not punished.** Precision is 1.0 when nothing is
extracted, because emitting nothing produces no false positives. Recall is 1.0
when a case has no gold. A case whose correct answer is silence
(`neg_001`, `time_001`, `sdse_003`) therefore scores F1 = 1.0 for an extractor
that stays silent, and is penalised only for speaking.

## Known limitations

- **Categorical variables are not value-scored.** The schema supports categorical
  distributions with per-label probabilities, but comparing category maps needs a
  label-alignment step this version does not implement. Categorical content in the
  CALERIE excerpt acts as a distractor for the numeric arms instead. Prompt parity
  on the `categories` field is still enforced (see below).
- **Malformed output is measured at the schema level only.** All LLM arms keep the
  production `response_format={"type": "json_object"}` setting, which makes
  syntactically invalid JSON nearly impossible. The reported malformed rate counts
  missing or unparseable schema fields, not brace errors. A free-form arm would
  measure the latter but would no longer measure the shipped pipeline.
- **Real excerpts cluster in three sources.** Only three papers are cached, so the
  real cases are not independent across categories.

## Prompt parity

The ablation compares the Caveman Protocol prompt against a natural-language
prompt. That comparison is only meaningful if both request the *same* schema.
An earlier version of `benchmark._VERBOSE_PROMPT` omitted the `categorical`
distribution and the `categories` field entirely, so it asked for strictly less
than `CAVEMAN_PROMPT` — which both understated the compressed prompt's token
saving and made any quality comparison invalid.

`tests/test_extraction_challenge.py::test_prompt_schema_parity` now enforces that
both prompts mention the same distribution names and schema fields. Do not edit
one prompt without the other.

## Rebuilding

```powershell
.venv\Scripts\python.exe analysis\build_extraction_challenge_set.py           # rebuild
.venv\Scripts\python.exe analysis\build_extraction_challenge_set.py --check   # verify only
```

`--check` fails if the committed JSON no longer matches the case definitions, so
CI catches an edited fixture that was not regenerated from its annotations.
