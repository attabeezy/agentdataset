"""
AgentDataset Validator
Data + Parameters -> FidelityReport
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from typing import Dict, Any, List
from agentdataset.models.schemas import Parameters, FidelityReport

MIN_KS_PVALUE = 0.05
MIN_FIDELITY_SCORE = 90.0

class Validator:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "ks_pvalue": 0.05,
            "corr_similarity": 0.8,
            "fidelity_score": 90.0
        }

    def compute_ks_test(self, df: pd.DataFrame, parameters: Parameters) -> Dict[str, float]:
        """Compute KS-test p-values."""
        results = {}
        for name, var_params in parameters.variables.items():
            if name not in df.columns: continue
            data = df[name].values
            
            # Theoretical CDF mapping — use default-arg capture to avoid late-binding closure bug
            if var_params.distribution == "normal":
                theoretical_cdf = lambda x, m=var_params.mean, s=var_params.std: stats.norm.cdf(x, loc=m, scale=s)
            elif var_params.distribution == "uniform":
                low = var_params.min if var_params.min is not None else var_params.mean - 2*var_params.std
                high = var_params.max if var_params.max is not None else var_params.mean + 2*var_params.std
                theoretical_cdf = lambda x, l=low, h=high: stats.uniform.cdf(x, loc=l, scale=h-l)
            elif var_params.distribution == "gamma":
                shape = (var_params.mean / var_params.std) ** 2
                scale = var_params.std ** 2 / var_params.mean
                theoretical_cdf = lambda x, a=shape, sc=scale: stats.gamma.cdf(x, a=a, scale=sc)
            else:
                theoretical_cdf = lambda x, m=var_params.mean, s=var_params.std: stats.norm.cdf(x, loc=m, scale=s)
            
            _, p_val = stats.kstest(data, theoretical_cdf)
            results[name] = float(p_val)
        return results

    def compute_correlation_similarity(self, df: pd.DataFrame, parameters: Parameters) -> float:
        """Compute cosine similarity of correlation matrices."""
        var_names = list(parameters.variables.keys())
        if len(var_names) < 2: return 1.0
        
        synthetic_corr = df[var_names].corr().fillna(0).values
        target_corr = np.eye(len(var_names))
        
        for key, corr_params in parameters.correlations.items():
            v1, v2 = corr_params.var1, corr_params.var2
            if v1 in var_names and v2 in var_names:
                idx1, idx2 = var_names.index(v1), var_names.index(v2)
                target_corr[idx1, idx2] = corr_params.correlation
                target_corr[idx2, idx1] = corr_params.correlation
        
        cos_sim = np.sum(synthetic_corr * target_corr) / (
            np.sqrt(np.sum(synthetic_corr**2)) * np.sqrt(np.sum(target_corr**2))
        )
        return float(max(0.0, min(1.0, (cos_sim + 1) / 2)))

    def compute_privacy_score(self, df: pd.DataFrame, sample_size: int = 500) -> Dict[str, float]:
        """Estimate privacy via average nearest-neighbour distance.

        Standardises each column then computes, for a random subsample,
        the distance from each row to its closest other row. Higher
        avg_min_dist means more spread-out data and lower re-identification risk.
        Score is normalised to [0, 1] by dividing by the theoretical max
        (distance between corner points of the unit hypercube in n dims).
        """
        numeric = df.select_dtypes(include="number")
        if numeric.empty or len(numeric) < 2:
            return {"avg_min_dist": 0.0, "privacy_score": 0.0}

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
        privacy_score = float(min(avg_min_dist / max_dist, 1.0))

        return {"avg_min_dist": round(avg_min_dist, 4), "privacy_score": round(privacy_score, 4)}

    def validate(self, df: pd.DataFrame, parameters: Parameters) -> FidelityReport:
        """Run full validation suite."""
        ks_pvalues = self.compute_ks_test(df, parameters)
        corr_sim = self.compute_correlation_similarity(df, parameters)
        
        # Simple bias score based on mean deviation
        bias_count = 0
        for name, var_params in parameters.variables.items():
            if name in df.columns:
                denom = abs(var_params.mean) if var_params.mean != 0 else (var_params.std or 1.0)
                if abs(df[name].mean() - var_params.mean) / denom > 0.2:
                    bias_count += 1
        bias_score = 1.0 - (bias_count / len(parameters.variables)) if parameters.variables else 1.0
        
        # Overall Score: fraction of variables whose distribution fits (p >= threshold), scaled 0-100
        passing = sum(1 for p in ks_pvalues.values() if p >= self.thresholds["ks_pvalue"])
        ks_score = (passing / len(ks_pvalues)) * 100 if ks_pvalues else 100.0
        overall_score = 0.4 * ks_score + 0.4 * (corr_sim * 100) + 0.2 * (bias_score * 100)
        
        return FidelityReport(
            overall_score=round(overall_score, 2),
            ks_score=round(ks_score, 2),
            corr_score=round(corr_sim * 100, 2),
            bias_score=round(bias_score * 100, 2),
            ks_pvalues=ks_pvalues,
            bias_details={},
            privacy_details=self.compute_privacy_score(df),
            approved=overall_score >= self.thresholds["fidelity_score"]
        )

    def generate_datacard(self, report: FidelityReport, parameters: Parameters, df: pd.DataFrame) -> str:
        """Generate a Markdown DATACARD report."""
        source = parameters.meta.source
        extracted_at = parameters.meta.extracted_at
        
        var_details = []
        for name, p_val in report.ks_pvalues.items():
            status = "[PASS]" if p_val >= self.thresholds["ks_pvalue"] else "[FAIL]"
            var_details.append(f"- {status} **{name}**: KS p-value={p_val:.4f}")

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

## Privacy
**Avg Nearest-Neighbour Distance**: {report.privacy_details.get("avg_min_dist", "n/a")}
**Privacy Score**: {report.privacy_details.get("privacy_score", "n/a")}

---
*DATACARD generated by AgentDataset*
"""
        return card
