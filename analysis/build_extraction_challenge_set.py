"""Build the extraction challenge set (P1 9.6) from annotated case definitions.

Real cases are not typed by hand: each declares *anchors* into a cached source
paper under `.agentdataset_cache/real_papers/`, and this script slices the text
verbatim at those anchors. That keeps the fixture provably faithful to the source
(no transcription drift) and records the character offsets plus the source file
sha256 so a reviewer can re-derive every excerpt.

Constructed cases carry their text inline and are labelled `provenance:
"constructed"` so they are never mistaken for real literature.

Gold values for real cases are taken from the already dual-pass-adjudicated
`analysis/real_paper_ground_truth.csv` (see REAL_PAPER_ANNOTATION_PROTOCOL.md);
this script does not re-annotate them.

Usage:
    python analysis/build_extraction_challenge_set.py
    python analysis/build_extraction_challenge_set.py --check   # verify only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".agentdataset_cache" / "real_papers"
OUT_PATH = ROOT / "analysis" / "extraction_challenge_set.json"

sys.path.insert(0, str(ROOT))
from agentdataset.evaluation import sha256_file  # noqa: E402

SPAN_JOIN = "\n[...]\n"

# Categories the taxonomy requires (see EXTRACTION_CHALLENGE_PROTOCOL.md).
CATEGORIES = [
    "negated_finding",
    "multiple_populations",
    "conflicting_sections",
    "repeated_timepoints",
    "units_mismatch",
    "confidence_intervals",
    "table_footnotes",
    "scientific_notation",
    "irrelevant_numbers",
    "sd_vs_se",
]


def _v(name, mean, std, *, aliases=(), units=None, population=None, group=None,
       distribution="normal"):
    """Shorthand for a gold variable annotation."""
    return {
        "name": name,
        "aliases": list(aliases),
        "distribution": distribution,
        "mean": mean,
        "std": std,
        "units": units,
        "population": population,
        "variant_group": group or name,
    }


def _c(var1, var2, value, *, group=None):
    return {"var1": var1, "var2": var2, "correlation": value,
            "variant_group": group or f"{var1}__{var2}"}


def _x(value, reason):
    return {"value": value, "reason": reason}


# --------------------------------------------------------------------------
# Case definitions.
#
# Real cases:        "spans": [(start_anchor, end_anchor), ...]
# Constructed cases: "text": "..."
# --------------------------------------------------------------------------
CASES = [
    # ---------------------------------------------------------------- negated
    {
        "case_id": "neg_001",
        "category": "negated_finding",
        "provenance": "real",
        "source_id": "calerie_rct",
        "spans": [("There was a trend for a correlation", "groups were combined.")],
        "gold_variables": [],
        "gold_correlations": [],
        "must_not_extract": [
            _x(0.14, "correlation explicitly described as only a 'trend', not an established finding"),
            _x(0.07, "p-value, above the conventional 0.05 threshold"),
        ],
        "notes": "Correct behaviour is abstention: a numeric r is present but the source "
                 "declines to assert the correlation. Extracting r=0.14 is a context error.",
    },
    {
        "case_id": "neg_002",
        "category": "negated_finding",
        "provenance": "real",
        "source_id": "cdc_nhanes_rural_urban",
        "spans": [("Rural residents were less active", "(760 counts per minute).")],
        "gold_variables": [
            _v("activity_2020_bouts", 55.9, 2.8, aliases=("physical activity 2020 CPM",
               "high-intensity long bout physical activity"), units="min/week",
               population="urban", group="activity_2020_bouts"),
            _v("activity_2020_bouts_rural", 42.5, 6.2, aliases=("physical activity 2020 CPM",),
               units="min/week", population="rural", group="activity_2020_bouts"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(760, "an accelerometer intensity threshold (counts per minute), not a measured variable"),
            _x(2020, "an accelerometer intensity threshold, not a measured variable"),
            _x(10, "a bout-length criterion in minutes, not a measured variable"),
        ],
        "notes": "The clause 'but the difference disappeared' negates a comparison, and the "
                 "sentence is dense with threshold constants that are not variables.",
    },
    {
        "case_id": "neg_003",
        "category": "negated_finding",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed example, serum ferritin had a mean of 88.4 ng/mL "
            "(SD 21.7). There was no significant association between serum ferritin "
            "and fatigue score (r = 0.03, p = 0.62), and no difference between sites "
            "was detected."
        ),
        "gold_variables": [
            _v("serum_ferritin", 88.4, 21.7, aliases=("ferritin",), units="ng/mL"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(0.03, "correlation explicitly reported as non-significant"),
            _x(0.62, "p-value"),
        ],
        "notes": "Constructed. The mean/SD is genuine gold; the negated correlation is not.",
    },

    # ----------------------------------------------------- multiple populations
    {
        "case_id": "pop_001",
        "category": "multiple_populations",
        "provenance": "real",
        "source_id": "cdc_nhanes_rural_urban",
        "spans": [
            ("| Variables | Urban (n = 3,828)", "| Mean (SD) | Mean (SD) | Mean (SD) | Mean (SD) | |"),
            ("| Physical activity in", "| 145.7 (12.0)a |"),
        ],
        "gold_variables": [
            _v("activity_2020_bouts", 55.9, 2.8, units="min/week", population="urban",
               group="activity_2020_bouts"),
            _v("activity_2020_bouts", 42.5, 6.2, units="min/week", population="all rural",
               group="activity_2020_bouts"),
            _v("activity_any_2020", 188.3, 4.3, units="min/week", population="urban",
               group="activity_any_2020"),
            _v("activity_any_2020", 162.4, 11.5, units="min/week", population="all rural",
               group="activity_any_2020"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(3828, "column-header sample size"),
            _x(1228, "column-header sample size"),
            _x(624, "column-header sample size"),
            _x(604, "column-header sample size"),
        ],
        "notes": "Four population columns for the same variable. Reporting any one column "
                 "is correct; mixing a mean from one column with an SD from another is a "
                 "conflation error.",
    },
    {
        "case_id": "pop_002",
        "category": "multiple_populations",
        "provenance": "real",
        "source_id": "calerie_rct",
        "spans": [("Calorie Restriction (n=143)", "37.9 (6.9)")],
        "gold_variables": [
            _v("age", 38.0, 7.2, units="years", population="calorie restriction arm", group="age"),
            _v("age", 37.9, 6.9, units="years", population="ad libitum arm", group="age"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(143, "arm sample size"),
            _x(75, "arm sample size"),
            _x(99, "female count, not a distribution parameter"),
            _x(44, "male count, not a distribution parameter"),
        ],
        "notes": "Two randomised arms side by side, plus categorical race/sex rows given as "
                 "counts with percentages. Categorical variables are not value-scored in "
                 "this version of the challenge set (see EXTRACTION_CHALLENGE_PROTOCOL.md); "
                 "the race/sex counts here act as distractors for the numeric arms.",
    },
    {
        "case_id": "pop_003",
        "category": "multiple_populations",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed three-arm trial, resting heart rate at baseline was "
            "68.2 (SD 9.1) beats/min in the placebo arm, 67.5 (SD 8.7) in the low-dose "
            "arm, and 69.1 (SD 9.4) in the high-dose arm. Randomisation allocated 120 "
            "participants per arm."
        ),
        "gold_variables": [
            _v("resting_heart_rate", 68.2, 9.1, units="beats/min", population="placebo",
               group="resting_heart_rate"),
            _v("resting_heart_rate", 67.5, 8.7, units="beats/min", population="low dose",
               group="resting_heart_rate"),
            _v("resting_heart_rate", 69.1, 9.4, units="beats/min", population="high dose",
               group="resting_heart_rate"),
        ],
        "gold_correlations": [],
        "must_not_extract": [_x(120, "per-arm sample size")],
        "notes": "Constructed. Three near-identical means invite conflation.",
    },

    # ----------------------------------------------------- conflicting sections
    {
        "case_id": "conf_001",
        "category": "conflicting_sections",
        "provenance": "real",
        "source_id": "student_mental_health",
        "spans": [
            ("Table 2.\nDescriptive Statistics", "Mean ± SD\nMean ± SD\nMean ± SD"),
            ("Duke/UNC Functional Social Support Questionnaire (DUFSSQ-8)\n42.27", "0.909"),
        ],
        "gold_variables": [
            _v("social_support", 30.83, 7.03, aliases=("DUFSSQ-8", "functional social support"),
               units="score", population="total sample", group="social_support"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(42.27, "male subgroup mean, not the total-sample value"),
            _x(40.48, "female subgroup mean, not the total-sample value"),
            _x(0.053, "p-value for the sex difference"),
            _x(0.909, "Cronbach's alpha reliability coefficient, not a distribution parameter"),
            _x(164, "column-header sample size"),
            _x(283, "column-header sample size"),
            _x(447, "column-header sample size"),
        ],
        "notes": "Flagged in real_paper_manifest.csv as a source conflict: the reported total "
                 "(30.83) lies outside both subgroup means (42.27, 40.48). Gold follows the "
                 "adjudicated annotation, which retains the reported total.",
    },
    {
        "case_id": "conf_002",
        "category": "conflicting_sections",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed example the abstract states that mean systolic blood "
            "pressure was 128.4 mmHg. Table 2, which reports the same cohort, gives a "
            "mean systolic blood pressure of 126.9 mmHg (SD 14.2). The table is the "
            "authoritative source."
        ),
        "gold_variables": [
            _v("systolic_blood_pressure", 126.9, 14.2, aliases=("SBP",), units="mmHg"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(128.4, "abstract value contradicted by the authoritative table"),
        ],
        "notes": "Constructed. Tests whether the extractor honours a stated precedence rule.",
    },
    {
        "case_id": "conf_003",
        "category": "conflicting_sections",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed example the results text reports a mean HbA1c of 7.4% "
            "(SD 1.1). The figure caption for the same panel reads: 'Mean HbA1c 7.8%'. "
            "The results text supersedes the caption, which was not updated after "
            "reanalysis."
        ),
        "gold_variables": [_v("hba1c", 7.4, 1.1, units="%")],
        "gold_correlations": [],
        "must_not_extract": [_x(7.8, "stale figure-caption value superseded by the results text")],
        "notes": "Constructed.",
    },

    # ------------------------------------------------------ repeated timepoints
    {
        "case_id": "time_001",
        "category": "repeated_timepoints",
        "provenance": "real",
        "source_id": "calerie_rct",
        "spans": [("compared to AL, CR significantly reduced", "(Figure 2A).")],
        "gold_variables": [],
        "gold_correlations": [],
        "must_not_extract": [
            _x(12, "a timepoint in months, not a measured value"),
            _x(24, "a timepoint in months, not a measured value"),
            _x(0.03, "p-value"),
            _x(0.002, "p-value"),
        ],
        "notes": "Abstention case: a significant effect is described but no mean or SD is "
                 "given. Any variable emitted here is fabricated or a context error.",
    },
    {
        "case_id": "time_002",
        "category": "repeated_timepoints",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed cohort, fasting glucose was 5.4 mmol/L (SD 0.6) at "
            "baseline, 5.1 mmol/L (SD 0.5) at 6 months, and 5.2 mmol/L (SD 0.7) at 12 "
            "months. Follow-up was complete for 340 participants."
        ),
        "gold_variables": [
            _v("fasting_glucose", 5.4, 0.6, units="mmol/L", population="baseline",
               group="fasting_glucose"),
            _v("fasting_glucose", 5.1, 0.5, units="mmol/L", population="6 months",
               group="fasting_glucose"),
            _v("fasting_glucose", 5.2, 0.7, units="mmol/L", population="12 months",
               group="fasting_glucose"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(6, "a timepoint in months"),
            _x(12, "a timepoint in months"),
            _x(340, "sample size"),
        ],
        "notes": "Constructed. The same variable three times; timepoint labels are numeric.",
    },
    {
        "case_id": "time_003",
        "category": "repeated_timepoints",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed study, mean step count was 6,820 (SD 2,140) in week 1 "
            "and 7,050 (SD 2,265) in week 8. Change from week 1 to week 8 was 230 steps "
            "(SD 410)."
        ),
        "gold_variables": [
            _v("step_count", 6820.0, 2140.0, units="steps/day", population="week 1",
               group="step_count"),
            _v("step_count", 7050.0, 2265.0, units="steps/day", population="week 8",
               group="step_count"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(230, "a change score, not a level; extracting it as the variable's mean is wrong"),
            _x(1, "a week label"),
            _x(8, "a week label"),
        ],
        "notes": "Constructed. Tests level-vs-change confusion.",
    },

    # ---------------------------------------------------------- units mismatch
    {
        "case_id": "unit_001",
        "category": "units_mismatch",
        "provenance": "real",
        "source_id": "cdc_nhanes_rural_urban",
        "spans": [("Rural residents reported more total physical activity", "12.5 min/wk)")],
        "gold_variables": [
            _v("reported_total_activity", 438.3, 35.3, units="min/week", population="rural",
               group="reported_total_activity"),
            _v("reported_total_activity", 371.2, 12.5, units="min/week", population="urban",
               group="reported_total_activity"),
        ],
        "gold_correlations": [],
        "must_not_extract": [],
        "notes": "Note the missing space in '35.3min/wk' in the source — a tokenisation trap.",
    },
    {
        "case_id": "unit_002",
        "category": "units_mismatch",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed example, total cholesterol averaged 5.2 mmol/L "
            "(SD 0.9), which corresponds to 201 mg/dL (SD 34.8). Report the value in "
            "mmol/L, the unit used throughout the paper."
        ),
        "gold_variables": [
            _v("total_cholesterol", 5.2, 0.9, aliases=("cholesterol",), units="mmol/L"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(201, "the same quantity restated in mg/dL; extracting both double-counts"),
            _x(34.8, "SD in the secondary unit"),
        ],
        "notes": "Constructed. One variable, two unit systems.",
    },
    {
        "case_id": "unit_003",
        "category": "units_mismatch",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed example, participants slept a mean of 7.1 hours per "
            "night (SD 1.2), equivalent to 426 minutes (SD 72). Weekly exercise averaged "
            "3.5 hours (SD 2.1)."
        ),
        "gold_variables": [
            _v("sleep_duration", 7.1, 1.2, units="hours/night", group="sleep_duration"),
            _v("weekly_exercise", 3.5, 2.1, units="hours/week", group="weekly_exercise"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(426, "sleep duration restated in minutes"),
            _x(72, "sleep SD restated in minutes"),
        ],
        "notes": "Constructed.",
    },

    # ------------------------------------------------------ confidence intervals
    {
        "case_id": "ci_001",
        "category": "confidence_intervals",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed analysis, mean birth weight was 3,410 g (SD 480; "
            "95% CI, 3,368 to 3,452). The confidence interval describes the precision "
            "of the mean, not the spread of the population."
        ),
        "gold_variables": [_v("birth_weight", 3410.0, 480.0, units="g")],
        "gold_correlations": [],
        "must_not_extract": [
            _x(3368, "lower confidence bound for the mean, not a population minimum"),
            _x(3452, "upper confidence bound for the mean, not a population maximum"),
            _x(95, "the confidence level"),
        ],
        "notes": "Constructed. CI bounds must not become min/max.",
    },
    {
        "case_id": "ci_002",
        "category": "confidence_intervals",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed study the adjusted odds ratio was 1.42 (95% CI "
            "1.08-1.87). Mean waist circumference was 94.6 cm (SD 12.3)."
        ),
        "gold_variables": [_v("waist_circumference", 94.6, 12.3, units="cm")],
        "gold_correlations": [],
        "must_not_extract": [
            _x(1.42, "an odds ratio, not a distribution mean"),
            _x(1.08, "confidence bound on an odds ratio"),
            _x(1.87, "confidence bound on an odds ratio"),
        ],
        "notes": "Constructed. An effect estimate sits next to a genuine mean/SD.",
    },
    {
        "case_id": "ci_003",
        "category": "confidence_intervals",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed sample, the correlation between grip strength and lean "
            "mass was r = 0.58 (95% CI 0.49 to 0.66, p < 0.001). Grip strength averaged "
            "32.7 kg (SD 8.4)."
        ),
        "gold_variables": [_v("grip_strength", 32.7, 8.4, units="kg")],
        "gold_correlations": [_c("grip_strength", "lean_mass", 0.58)],
        "must_not_extract": [
            _x(0.49, "confidence bound on the correlation, not the point estimate"),
            _x(0.66, "confidence bound on the correlation, not the point estimate"),
            _x(0.001, "p-value"),
        ],
        "notes": "Constructed. A genuine correlation flanked by bounds that resemble it.",
    },

    # ---------------------------------------------------------- table footnotes
    {
        "case_id": "foot_001",
        "category": "table_footnotes",
        "provenance": "real",
        "source_id": "cdc_nhanes_rural_urban",
        "spans": [
            ("| Variables | Urban (n = 3,828)", "| Mean (SD) | Mean (SD) | Mean (SD) | Mean (SD) | |"),
            ("| Household physical activity", "| 24.7 (4.7)b |"),
        ],
        "gold_variables": [
            _v("household_activity", 124.4, 8.8, units="min/week", population="urban",
               group="household_activity"),
            _v("household_activity", 201.6, 26.3, units="min/week", population="all rural",
               group="household_activity"),
            _v("transportation_activity", 40.5, 4.6, units="min/week", population="urban",
               group="transportation_activity"),
            _v("transportation_activity", 29.8, 4.9, units="min/week", population="all rural",
               group="transportation_activity"),
        ],
        "gold_correlations": [],
        "must_not_extract": [],
        "notes": "Values carry trailing significance markers (a/b/c) fused to the number, e.g. "
                 "'201.6 (26.3)b'. The marker must not be absorbed into the value.",
    },
    {
        "case_id": "foot_002",
        "category": "table_footnotes",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed table, mean serum sodium was 139.2 (3.1)*, where * "
            "denotes p < 0.05 versus control, and mean serum potassium was 4.1 (0.4)†, "
            "where † denotes measurements taken on 2 analysers."
        ),
        "gold_variables": [
            _v("serum_sodium", 139.2, 3.1, units="mmol/L", group="serum_sodium"),
            _v("serum_potassium", 4.1, 0.4, units="mmol/L", group="serum_potassium"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(0.05, "significance threshold in a footnote"),
            _x(2, "a count of analysers in a footnote"),
        ],
        "notes": "Constructed. Footnote glosses introduce numbers that are not measurements.",
    },
    {
        "case_id": "foot_003",
        "category": "table_footnotes",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed table, mean BMI was 27.3 (4.9)a and mean waist-to-hip "
            "ratio was 0.88 (0.07)b. Footnote a: adjusted for age. Footnote b: n = 812 "
            "because of 34 missing values."
        ),
        "gold_variables": [
            _v("bmi", 27.3, 4.9, units="kg/m2", group="bmi"),
            _v("waist_to_hip_ratio", 0.88, 0.07, group="waist_to_hip_ratio"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(812, "footnote sample size"),
            _x(34, "footnote count of missing values"),
        ],
        "notes": "Constructed.",
    },

    # ------------------------------------------------------- scientific notation
    {
        "case_id": "sci_001",
        "category": "scientific_notation",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed assay, mean analyte concentration was 1.2e-3 mol/L "
            "(SD 4.0e-4). The association reached p = 3.4e-8."
        ),
        "gold_variables": [
            _v("analyte_concentration", 0.0012, 0.0004, units="mol/L"),
        ],
        "gold_correlations": [],
        "must_not_extract": [_x(3.4e-8, "p-value in scientific notation")],
        "notes": "Constructed. Values must be parsed from exponential notation, not truncated "
                 "to the mantissa.",
    },
    {
        "case_id": "sci_002",
        "category": "scientific_notation",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed experiment, mean particle count was 6.02 x 10^5 per mL "
            "(SD 1.15 x 10^5), and the detection limit was 1 x 10^2 per mL."
        ),
        "gold_variables": [
            _v("particle_count", 602000.0, 115000.0, units="per mL"),
        ],
        "gold_correlations": [],
        "must_not_extract": [_x(100, "instrument detection limit, not a measured variable")],
        "notes": "Constructed. Uses 'x 10^n' rather than 'e' notation.",
    },
    {
        "case_id": "sci_003",
        "category": "scientific_notation",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed dataset, mean gene expression was 2.5E+04 counts "
            "(SD 8.0E+03) and the false discovery rate was held at 5.0E-02."
        ),
        "gold_variables": [
            _v("gene_expression", 25000.0, 8000.0, units="counts"),
        ],
        "gold_correlations": [],
        "must_not_extract": [_x(0.05, "false discovery rate threshold")],
        "notes": "Constructed. Capital-E notation with explicit positive exponent.",
    },

    # ------------------------------------------------------- irrelevant numbers
    {
        "case_id": "irrel_001",
        "category": "irrelevant_numbers",
        "provenance": "real",
        "source_id": "student_mental_health",
        "spans": [
            ("Table 2.\nDescriptive Statistics", "Mean ± SD\nMean ± SD\nMean ± SD"),
            ("UCLA 3-Item Loneliness Scale (0–6)\n4.92", "0.815"),
        ],
        "gold_variables": [
            _v("loneliness", 4.89, 1.66, aliases=("UCLA-3", "UCLA 3-Item Loneliness Scale"),
               units="score", population="total sample", group="loneliness"),
            _v("loneliness", 4.92, 1.78, aliases=("UCLA-3",), units="score",
               population="male", group="loneliness"),
            _v("loneliness", 4.84, 1.58, aliases=("UCLA-3",), units="score",
               population="female", group="loneliness"),
            _v("depression", 6.39, 5.97, aliases=("PHQ-9",), units="score",
               population="total sample", group="depression"),
            _v("depression", 6.56, 6.33, aliases=("PHQ-9",), units="score",
               population="male", group="depression"),
            _v("depression", 6.30, 5.69, aliases=("PHQ-9",), units="score",
               population="female", group="depression"),
            _v("sleep_quality", 2.81, 2.83, aliases=("B-PSQI",), units="score",
               population="total sample", group="sleep_quality"),
            _v("sleep_quality", 2.94, 2.96, aliases=("B-PSQI",), units="score",
               population="male", group="sleep_quality"),
            _v("sleep_quality", 2.90, 2.85, aliases=("B-PSQI",), units="score",
               population="female", group="sleep_quality"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(0.643, "p-value for the sex difference"),
            _x(0.658, "p-value for the sex difference"),
            _x(0.885, "p-value for the sex difference"),
            _x(0.836, "Cronbach's alpha"),
            _x(0.900, "Cronbach's alpha"),
            _x(0.815, "Cronbach's alpha"),
            _x(27, "upper bound of the PHQ-9 instrument range"),
            _x(15, "upper bound of the B-PSQI instrument range"),
            _x(164, "column-header sample size"),
            _x(283, "column-header sample size"),
            _x(447, "column-header sample size"),
        ],
        "notes": "Instrument ranges, p-values and reliability coefficients sit in the same "
                 "rows as the gold means and SDs.",
    },
    {
        "case_id": "irrel_002",
        "category": "irrelevant_numbers",
        "provenance": "real",
        "source_id": "student_mental_health",
        "spans": [("Smartphone addiction showed a strong positive correlation",
                   "poor sleep quality (r = 0.31)")],
        "gold_variables": [],
        "gold_correlations": [
            _c("smartphone_addiction", "internet_addiction", 0.64),
            _c("smartphone_addiction", "loneliness", 0.28),
            _c("smartphone_addiction", "depression", 0.23),
            _c("smartphone_addiction", "sleep_quality", 0.17),
            _c("internet_addiction", "loneliness", 0.31),
            _c("internet_addiction", "depression", 0.40),
        ],
        "must_not_extract": [
            _x(0.05, "significance threshold from 'p < 0.05', not a correlation"),
        ],
        "notes": "A correlation run where a p-threshold is numerically indistinguishable from "
                 "a weak correlation.",
    },
    {
        "case_id": "irrel_003",
        "category": "irrelevant_numbers",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed report (doi:10.1000/182, published 2019, pages 45-58, "
            "grant R01-HL120987), the 1,204 enrolled participants had a mean age of 52.6 "
            "years (SD 11.4). Response rate was 78%."
        ),
        "gold_variables": [_v("age", 52.6, 11.4, units="years")],
        "gold_correlations": [],
        "must_not_extract": [
            _x(2019, "publication year"),
            _x(45, "page number"),
            _x(58, "page number"),
            _x(1204, "sample size"),
            _x(78, "response rate percentage, not a distribution parameter"),
            _x(182, "DOI fragment"),
        ],
        "notes": "Constructed. Bibliographic and administrative numbers crowd a single gold "
                 "mean/SD.",
    },

    # -------------------------------------------------------------- SD vs SE
    {
        "case_id": "sdse_001",
        "category": "sd_vs_se",
        "provenance": "real",
        "source_id": "cdc_nhanes_rural_urban",
        "spans": [
            ("| Variables | Urban (n = 3,828)", "| Mean (SD) | Mean (SD) | Mean (SD) | Mean (SD) | |"),
            ("| Age, y |", "| 46.2 (1.1)b |"),
        ],
        "gold_variables": [
            _v("age", 43.4, 0.4, units="years", population="urban", group="age"),
            _v("age", 46.4, 0.8, units="years", population="all rural", group="age"),
        ],
        "gold_correlations": [],
        "must_not_extract": [],
        "notes": "The source labels the parenthesised value as SD, but 0.4 years for a "
                 "population aged 20-75 is implausibly small and is almost certainly a "
                 "standard error. Gold follows the adjudicated annotation (verbatim SD) so "
                 "extraction is scored against what the source says, not against a "
                 "reinterpretation. Flagged in real_paper_ground_truth.csv (cdc_006).",
    },
    {
        "case_id": "sdse_002",
        "category": "sd_vs_se",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed sample of 400 adults, mean systolic blood pressure was "
            "131.5 mmHg (SEM 0.9). The corresponding standard deviation is 18.0 mmHg."
        ),
        "gold_variables": [
            _v("systolic_blood_pressure", 131.5, 18.0, aliases=("SBP",), units="mmHg"),
        ],
        "gold_correlations": [],
        "must_not_extract": [
            _x(0.9, "standard error of the mean, not the standard deviation"),
            _x(400, "sample size"),
        ],
        "notes": "Constructed. Both SEM and SD are present; only the SD is the distribution "
                 "parameter.",
    },
    {
        "case_id": "sdse_003",
        "category": "sd_vs_se",
        "provenance": "constructed",
        "source_id": None,
        "text": (
            "In this constructed cohort, mean forced vital capacity was 3.86 L "
            "(standard error 0.04). No standard deviation is reported."
        ),
        "gold_variables": [],
        "gold_correlations": [],
        "must_not_extract": [
            _x(0.04, "standard error; the standard deviation is not recoverable without n"),
        ],
        "notes": "Constructed abstention case: a mean is present but the spread parameter the "
                 "schema needs is not, and the SE must not be substituted for it.",
    },
]


def _slice(text: str, start_anchor: str, end_anchor: str, source_id: str,
           case_id: str) -> tuple[int, int]:
    start = text.find(start_anchor)
    if start < 0:
        raise ValueError(f"{case_id}: start anchor not found in {source_id}: {start_anchor!r}")
    end_rel = text.find(end_anchor, start + len(start_anchor))
    if end_rel < 0:
        raise ValueError(f"{case_id}: end anchor not found after start in {source_id}: {end_anchor!r}")
    return start, end_rel + len(end_anchor)


def build() -> list[dict]:
    cache: dict[str, str] = {}
    digests: dict[str, str] = {}
    cases = []

    for case in CASES:
        out = {
            "case_id": case["case_id"],
            "category": case["category"],
            "provenance": case["provenance"],
            "source_id": case["source_id"],
        }
        if case["provenance"] == "real":
            source_id = case["source_id"]
            path = CACHE_DIR / f"{source_id}.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"{case['case_id']}: cached source missing: {path}. Run "
                    f"`python analysis/real_paper_eval.py --mode live` to repopulate the cache."
                )
            if source_id not in cache:
                cache[source_id] = path.read_text(encoding="utf-8", errors="replace")
                digests[source_id] = sha256_file(path)
            text = cache[source_id]
            spans, pieces = [], []
            for start_anchor, end_anchor in case["spans"]:
                start, end = _slice(text, start_anchor, end_anchor, source_id, case["case_id"])
                spans.append({"start": start, "end": end})
                pieces.append(text[start:end])
            out["source_sha256"] = digests[source_id]
            out["spans"] = spans
            out["text"] = SPAN_JOIN.join(pieces)
        else:
            out["source_sha256"] = None
            out["spans"] = []
            out["text"] = case["text"]

        out["gold_variables"] = case["gold_variables"]
        out["gold_correlations"] = case["gold_correlations"]
        out["must_not_extract"] = case["must_not_extract"]
        out["notes"] = case["notes"]
        cases.append(out)

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Rebuild and compare against the committed fixture without writing.")
    args = parser.parse_args()

    cases = build()

    missing = set(CATEGORIES) - {c["category"] for c in cases}
    if missing:
        raise SystemExit(f"Taxonomy categories with no cases: {sorted(missing)}")
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate case_id values in CASES")

    payload = {
        "schema_version": 1,
        "description": (
            "Extraction challenge set for the P1 9.6 robustness and prompt ablation. "
            "Real cases are verbatim excerpts sliced from cached open-access sources; "
            "constructed cases are synthetic and labelled as such."
        ),
        "span_join": SPAN_JOIN,
        "categories": CATEGORIES,
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not OUT_PATH.exists():
            raise SystemExit(f"{OUT_PATH} does not exist; run without --check first.")
        if OUT_PATH.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{OUT_PATH} is out of date; rerun without --check.")
        print(f"OK: {OUT_PATH.name} matches the case definitions ({len(cases)} cases).")
        return

    OUT_PATH.write_text(rendered, encoding="utf-8")
    by_cat: dict[str, int] = {}
    real = 0
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
        real += c["provenance"] == "real"
    print(f"Wrote {OUT_PATH} — {len(cases)} cases ({real} real, {len(cases) - real} constructed)")
    for category in CATEGORIES:
        print(f"  {category:<24} {by_cat.get(category, 0)}")


if __name__ == "__main__":
    main()
