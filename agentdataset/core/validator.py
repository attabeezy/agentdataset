"""
AgentDataset Validator
Data + Parameters -> FidelityReport
"""

import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Optional
from agentdataset.models.schemas import Parameters, FidelityReport, Claim

MIN_KS_PVALUE = 0.05
MIN_FIDELITY_SCORE = 90.0


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

class Validator:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "ks_pvalue": 0.05,
            "corr_similarity": 0.8,
            "fidelity_score": 90.0
        }

    def compute_ks_test(self, df: pd.DataFrame, parameters: Parameters) -> Dict[str, float]:
        """Compute distribution-fit p-values: KS-test for continuous variables,
        chi-square goodness-of-fit for categorical ones. Stored in the same
        dict/field regardless of which test produced them."""
        results = {}
        for name, var_params in parameters.variables.items():
            if name not in df.columns: continue

            if var_params.distribution == "categorical" and var_params.categories:
                labels = list(var_params.categories.keys())
                observed_counts = np.array([(df[name] == label).sum() for label in labels], dtype=float)
                expected_counts = np.array([var_params.categories[label] for label in labels]) * len(df)
                if (expected_counts > 0).all():
                    _, p_val = stats.chisquare(observed_counts, expected_counts)
                else:
                    p_val = 0.0
                results[name] = float(p_val)
                continue

            data = df[name].values

            # Theoretical CDF mapping — use default-arg capture to avoid late-binding closure bug
            if var_params.distribution == "normal":
                theoretical_cdf = lambda x, m=var_params.mean, s=var_params.std: stats.norm.cdf(x, loc=m, scale=s)
            elif var_params.distribution == "uniform":
                low = var_params.min if var_params.min is not None else var_params.mean - 2*var_params.std
                high = var_params.max if var_params.max is not None else var_params.mean + 2*var_params.std
                theoretical_cdf = lambda x, l=low, h=high: stats.uniform.cdf(x, loc=l, scale=h-l)
            elif var_params.distribution == "gamma" and var_params.mean > 0 and var_params.std > 0:
                shape = (var_params.mean / var_params.std) ** 2
                scale = var_params.std ** 2 / var_params.mean
                theoretical_cdf = lambda x, a=shape, sc=scale: stats.gamma.cdf(x, a=a, scale=sc)
            else:
                # normal, unknown, or gamma with non-positive mean/std → normal CDF
                # (matches synthesizer fallback for the same invalid-gamma case)
                theoretical_cdf = lambda x, m=var_params.mean, s=var_params.std: stats.norm.cdf(x, loc=m, scale=s)
            
            _, p_val = stats.kstest(data, theoretical_cdf)
            results[name] = float(p_val)
        return results

    def compute_correlation_similarity(self, df: pd.DataFrame, parameters: Parameters) -> float:
        """Score how well synthetic correlations match the target ones.

        Compares only the upper-triangle (off-diagonal) entries — the diagonal is
        always 1 and would otherwise dominate, inflating the score regardless of
        how well the actual correlations were reproduced. Returns
        ``1 - mean(|synthetic - target|) / 2`` over those pairs, clamped to [0, 1]
        (the divisor 2 maps the worst case |Δ|=2 to a score of 0). With no declared
        correlations the target is the identity, so a well-decorrelated synthetic
        set still scores near 1.
        """
        var_names = list(parameters.variables.keys())
        if len(var_names) < 2:
            return 1.0

        # Categorical columns are strings; encode as 0/1/2... (order given by
        # `categories`) so .corr() can include them — exactly point-biserial
        # correlation for the binary case.
        numeric_df = df[var_names].copy()
        for name in var_names:
            var_params = parameters.variables[name]
            if var_params.distribution == "categorical" and var_params.categories:
                labels = list(var_params.categories.keys())
                numeric_df[name] = numeric_df[name].map({label: i for i, label in enumerate(labels)})

        synthetic_corr = numeric_df.corr().fillna(0).values
        target_corr = np.eye(len(var_names))

        for key, corr_params in parameters.correlations.items():
            v1, v2 = corr_params.var1, corr_params.var2
            if v1 in var_names and v2 in var_names:
                idx1, idx2 = var_names.index(v1), var_names.index(v2)
                target_corr[idx1, idx2] = corr_params.correlation
                target_corr[idx2, idx1] = corr_params.correlation

        # Off-diagonal (upper-triangle) entries only.
        triu = np.triu_indices(len(var_names), k=1)
        synth_off = synthetic_corr[triu]
        target_off = target_corr[triu]
        if synth_off.size == 0:
            return 1.0

        mean_abs_err = float(np.mean(np.abs(synth_off - target_off)))
        return float(max(0.0, min(1.0, 1.0 - mean_abs_err / 2.0)))

    def compute_plausibility(self, parameters: Parameters) -> Dict[str, Any]:
        """Score whether the declared parameters look trustworthy on their own.

        The KS/bias terms compare synthetic data against the same declared
        params used to draw them, so they are invariant to parameter error by
        construction. This check inspects the params themselves and penalizes:
        non-positive std, min>=max, placeholder (0.0, 1.0) defaults the parser
        emits when the LLM omits numbers, correlations naming missing variables,
        duplicate names after normalization, and non-PD correlation structure.
        Returns {"score": 0-100, "issues": [...]}, score starts at 100 and
        loses 15-25 per issue class present.
        """
        issues: list[str] = []
        variables = parameters.variables or {}
        correlations = parameters.correlations or {}

        for name, vp in variables.items():
            if vp.distribution != "categorical" and (vp.std is None or vp.std <= 0):
                issues.append(f"{name}: non-positive std {vp.std}")
            if vp.min is not None and vp.max is not None and vp.min >= vp.max:
                issues.append(f"{name}: min {vp.min} >= max {vp.max}")
            if (
                vp.distribution != "categorical"
                and vp.mean == 0.0 and vp.std == 1.0
                and not vp.categories
            ):
                issues.append(f"{name}: placeholder mean=0/std=1 (number likely omitted)")
            if vp.distribution == "categorical" and not vp.categories:
                issues.append(f"{name}: categorical without categories")

        seen_norm: dict[str, str] = {}
        for name in variables:
            norm = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
            if norm in seen_norm:
                issues.append(f"duplicate normalized name: {name} collides with {seen_norm[norm]}")
            else:
                seen_norm[norm] = name

        missing_corr = 0
        for key, cp in correlations.items():
            if cp.var1 not in variables or cp.var2 not in variables:
                missing_corr += 1
        if missing_corr:
            issues.append(f"{missing_corr} correlation(s) name undeclared variables")

        score = 100.0
        # One penalty per issue class, capped so a single bad param cannot zero out.
        if any("non-positive std" in i for i in issues):
            score -= 25
        if any("placeholder" in i for i in issues):
            score -= 25
        if any("undeclared variables" in i for i in issues):
            score -= 20
        if any("min " in i and ">=" in i for i in issues):
            score -= 15
        if any("duplicate normalized" in i for i in issues):
            score -= 15
        if any("without categories" in i for i in issues):
            score -= 15
        return {"score": float(max(0.0, score)), "issues": issues}

    def compute_grounding(
        self, parameters: Parameters, source_text: str | None
    ) -> Dict[str, Any]:
        """Score whether declared numbers actually appear in the source text.

        For each continuous variable, its mean must appear numerically (within
        1% relative or 1e-6 absolute tolerance of any number in the text) AND
        its name (normalized, underscores<->spaces) must appear as a substring.
        Categorical variables only need the name. Fraction grounded * 100 is the
        score. With no source_text, returns 100 so old callers are unaffected.
        """
        if not source_text:
            return {"score": 100.0, "details": {"skipped": True}}
        import re as _re

        text_low = source_text.lower().replace("_", " ")
        numbers = [_f for _f in (_to_float(m) for m in _re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", source_text)) if _f is not None]
        grounded = 0
        total = len(parameters.variables) or 1
        per_var: dict[str, bool] = {}
        for name, vp in parameters.variables.items():
            name_hit = name.lower().replace("_", " ") in text_low
            if vp.distribution == "categorical":
                ok = name_hit
            else:
                num_hit = any(
                    abs(n - vp.mean) <= max(1e-6, 0.01 * abs(vp.mean)) for n in numbers
                )
                ok = name_hit and num_hit
            per_var[name] = ok
            grounded += 1 if ok else 0
        return {"score": 100.0 * grounded / total, "details": per_var}

    def compute_row_dispersion(self, df: pd.DataFrame, sample_size: int = 500) -> Dict[str, Any]:
        """Describe nearest-neighbour spacing among generated numeric rows.

        This is a geometry diagnostic, not a privacy or disclosure-risk score:
        it never compares generated rows with confidential training records.
        """
        numeric = df.select_dtypes(include="number")
        diagnostic_warnings = [
            "Synthetic rows only: this diagnostic is not a privacy audit or guarantee."
        ]
        if numeric.empty or len(numeric) < 2:
            diagnostic_warnings.append("At least two rows and one numeric column are required.")
            return {
                "avg_nearest_neighbor_distance": 0.0,
                "normalized_row_dispersion": 0.0,
                "sampled_rows": min(len(numeric), sample_size),
                "numeric_columns": len(numeric.columns),
                "warnings": diagnostic_warnings,
            }

        if len(df) < 50:
            diagnostic_warnings.append("Small samples can make nearest-neighbour distances unstable.")
        if len(numeric.columns) != len(df.columns):
            diagnostic_warnings.append("Categorical/non-numeric columns are excluded from the distance calculation.")
        diagnostic_warnings.append("The value depends on standardization and mixed-scale feature choices.")

        # Standardise to zero mean / unit variance
        std = numeric.std().replace(0, 1)
        standardised = ((numeric - numeric.mean()) / std).values

        # Subsample to keep O(n²) tractable
        n = min(sample_size, len(standardised))
        idx = np.random.default_rng(0).choice(len(standardised), size=n, replace=False)
        sample = standardised[idx]

        dists = cdist(sample, sample, metric="euclidean")
        np.fill_diagonal(dists, np.inf)          # exclude self-distance
        avg_min_dist = float(np.mean(np.min(dists, axis=1)))

        # Normalise: theoretical max distance across n_cols dimensions
        n_cols = standardised.shape[1]
        max_dist = float(np.sqrt(n_cols)) if n_cols > 0 else 1.0
        normalized = float(min(avg_min_dist / max_dist, 1.0))

        return {
            "avg_nearest_neighbor_distance": round(avg_min_dist, 4),
            "normalized_row_dispersion": round(normalized, 4),
            "sampled_rows": n,
            "numeric_columns": n_cols,
            "warnings": diagnostic_warnings,
        }

    def compute_privacy_score(self, df: pd.DataFrame, sample_size: int = 500) -> Dict[str, Any]:
        """Deprecated alias for :meth:`compute_row_dispersion`."""
        return self.compute_row_dispersion(df, sample_size=sample_size)

    def compute_claim_grounding(
        self, claims: Optional[List[Claim]] = None, source_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Claim-level grounding: each claim's value must appear in its evidence
        passage (or the whole source text as fallback). Returns score + details.
        """
        if not claims:
            return {"score": 100.0, "details": {"skipped": True}}
        import re as _re
        grounded = 0
        per_claim = {}
        for c in claims:
            passage = None
            if c.evidence and c.evidence.passage:
                passage = c.evidence.passage
            elif source_text:
                passage = source_text
            if passage is None:
                per_claim[c.claim_id] = False
                continue
            text_low = passage.lower().replace("_", " ")
            # value presence: try numeric mean if dict
            val = c.value
            num = None
            if isinstance(val, dict):
                num = val.get("mean")
            elif isinstance(val, (int, float)):
                num = val
            ok = True
            if num is not None:
                nums = [_f for _f in (_to_float(m) for m in _re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", passage)) if _f is not None]
                num_hit = any(abs(n - float(num)) <= max(1e-6, 0.01 * abs(float(num))) for n in nums)
                ok = ok and num_hit
            # name presence
            if c.variable_name and "_" in c.variable_name:
                name_hit = c.variable_name.lower().replace("_", " ") in text_low
            else:
                name_hit = c.variable_name.lower() in text_low
            if c.variable_name:
                ok = ok and name_hit
            per_claim[c.claim_id] = ok
            grounded += 1 if ok else 0
        total = len(claims) or 1
        return {"score": 100.0 * grounded / total, "details": per_claim}

    def validate(
        self,
        df: pd.DataFrame,
        parameters: Parameters,
        source_text: Optional[str] = None,
        claims: Optional[List[Any]] = None,
    ) -> FidelityReport:
        """Run full validation suite.

        `overall_score` is unchanged (synthesis diagnostic). `trust_score` is
        designed to catch parameter error: 0.4*overall + 0.25*grounding +
        0.2*plausibility + 0.15*claim_grounding. Without source_text, grounding
        defaults to 100 so behavior is unchanged.
        """
        ks_pvalues = self.compute_ks_test(df, parameters)
        corr_sim = self.compute_correlation_similarity(df, parameters)
        
        # Simple bias score: relative mean deviation for continuous variables,
        # absolute category-frequency deviation for categorical ones.
        bias_count = 0
        for name, var_params in parameters.variables.items():
            if name not in df.columns:
                continue
            if var_params.distribution == "categorical" and var_params.categories:
                observed_freqs = df[name].value_counts(normalize=True)
                if any(
                    abs(observed_freqs.get(label, 0.0) - target_prob) > 0.1
                    for label, target_prob in var_params.categories.items()
                ):
                    bias_count += 1
            else:
                denom = abs(var_params.mean) if var_params.mean != 0 else (var_params.std or 1.0)
                if abs(df[name].mean() - var_params.mean) / denom > 0.2:
                    bias_count += 1
        bias_score = 1.0 - (bias_count / len(parameters.variables)) if parameters.variables else 1.0
        
        # Overall Score: fraction of variables whose distribution fits (p >= threshold), scaled 0-100
        passing = sum(1 for p in ks_pvalues.values() if p >= self.thresholds["ks_pvalue"])
        ks_score = (passing / len(ks_pvalues)) * 100 if ks_pvalues else 100.0
        overall_score = 0.4 * ks_score + 0.4 * (corr_sim * 100) + 0.2 * (bias_score * 100)

        plaus = self.compute_plausibility(parameters)
        ground = self.compute_grounding(parameters, source_text)
        claim_ground = self.compute_claim_grounding(claims, source_text)
        trust_score = round(
            0.4 * overall_score + 0.25 * ground["score"] + 0.2 * plaus["score"] + 0.15 * claim_ground["score"], 2
        )

        return FidelityReport(
            overall_score=round(overall_score, 2),
            ks_score=round(ks_score, 2),
            corr_score=round(corr_sim * 100, 2),
            bias_score=round(bias_score * 100, 2),
            ks_pvalues=ks_pvalues,
            bias_details={},
            dispersion_details=self.compute_row_dispersion(df),
            approved=overall_score >= self.thresholds["fidelity_score"],
            plausibility_score=round(plaus["score"], 2),
            grounding_score=round(ground["score"], 2),
            trust_score=trust_score,
            issues=plaus["issues"],
        )

    def generate_datacard(self, report: FidelityReport, parameters: Parameters, df: pd.DataFrame) -> str:
        """Generate a Markdown DATACARD report."""
        source = parameters.meta.source
        extracted_at = parameters.meta.extracted_at
        
        var_details = []
        for name, p_val in report.ks_pvalues.items():
            status = "[PASS]" if p_val >= self.thresholds["ks_pvalue"] else "[FAIL]"
            var_params = parameters.variables.get(name)
            is_categorical = var_params is not None and var_params.distribution == "categorical" and var_params.categories
            label = "Chi2 p-value" if is_categorical else "KS p-value"
            var_details.append(f"- {status} **{name}**: {label}={p_val:.4f}")

        card = f"""# DATACARD: Synthetic Dataset

## Overview
- **Source**: {source}
- **Generated**: {extracted_at}
- **Rows**: {len(df)}
- **Columns**: {len(df.columns)}
- **Fidelity Score**: **{report.overall_score}**/100

## Statistical Fidelity

### Distribution Fit (KS-test)
{"\n".join(var_details)}

### Correlation Preservation
**Correlation Similarity**: {report.corr_score/100:.4f}

## Bias Detection
**Bias Score**: {report.bias_score:.2f}%

## Trust Score (catches parameter error)
**Trust Score**: **{report.trust_score}**/100 (fidelity {report.overall_score}, grounding {report.grounding_score}, plausibility {report.plausibility_score})
{"Parameter issues: " + "; ".join(report.issues) if report.issues else "No parameter issues flagged."}
Grounding checks each declared mean against numbers in the source text; a low
score means the extraction likely hallucinated or misread values even when
fidelity is high.

## Synthetic-row dispersion diagnostic (not a privacy audit)
**Average nearest-neighbour distance**: {report.dispersion_details.get("avg_nearest_neighbor_distance", "n/a")}
**Normalized row dispersion**: {report.dispersion_details.get("normalized_row_dispersion", "n/a")}

This diagnostic measures spacing among generated numeric rows only. It does not
compare against private records and provides no disclosure-risk or differential-
privacy guarantee. It can be unstable for small samples, excludes categorical
columns, and is sensitive to standardization and mixed feature scales.

---
*DATACARD generated by AgentDataset*
"""
        return card
