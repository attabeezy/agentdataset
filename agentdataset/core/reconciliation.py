"""Evidence-aware reconciliation of parameters from multiple sources (§10.3).

Replaces the blind numeric averaging in ``Orchestrator.merge_parameters`` with
study-aware grouping: measures are matched by normalized name, compatibility is
checked before pooling, per-study estimates are retained, and disagreements are
surfaced as conflicts rather than silently combined.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
from collections import defaultdict

from agentdataset.models.schemas import (
    Parameters, VariableParams, CorrelationParams, MetaParams,
    ReconciliationDecision,
)


def _norm(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", (name or "").strip()).strip("_").lower()


def _compatible_means(a: float, b: float, tol: float = 0.2) -> bool:
    """Relative-difference compatibility check (default 20%)."""
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


class Reconciler:
    def __init__(self, mean_tol: float = 0.2, std_tol: float = 0.3):
        self.mean_tol = mean_tol
        self.std_tol = std_tol

    def reconcile(self, params_list: List[Parameters]) -> Tuple[Parameters, List[ReconciliationDecision]]:
        if not params_list:
            raise ValueError("params_list must not be empty")
        if len(params_list) == 1:
            return params_list[0], []

        decisions: List[ReconciliationDecision] = []

        # --- Variables ---
        groups: Dict[str, List[Tuple[int, str, VariableParams]]] = defaultdict(list)
        for idx, params in enumerate(params_list):
            for name, vp in (params.variables or {}).items():
                groups[_norm(name)].append((idx, name, vp))

        merged_variables: Dict[str, VariableParams] = {}
        for norm_name, items in groups.items():
            # Determine compatibility across all pairs.
            compatible = True
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if not (_compatible_means(items[i][2].mean, items[j][2].mean, self.mean_tol)
                            and _compatible_means(items[i][2].std, items[j][2].std, self.std_tol)):
                        compatible = False
                        break
                if not compatible:
                    break

            accepted_ids = [f"src{idx}:{orig}" for idx, orig, _ in items]
            if compatible and len(items) > 1:
                mean = sum(vp.mean for _, _, vp in items) / len(items)
                std = sum(vp.std for _, _, vp in items) / len(items)
                dist = items[0][2].distribution
                canonical = items[0][1]
                merged_variables[canonical] = VariableParams(
                    name=canonical,
                    distribution=dist,
                    mean=mean,
                    std=std,
                    min=mean - 3 * std,
                    max=mean + 3 * std,
                    categories=items[0][2].categories,
                )
                decisions.append(ReconciliationDecision(
                    target_variable=canonical,
                    accepted_claim_ids=accepted_ids,
                    pooled_value={"mean": mean, "std": std},
                    method="pooled_average",
                    assumption=f"means within {self.mean_tol:.0%}, stds within {self.std_tol:.0%}",
                ))
            else:
                # Keep per-source entries separate; surface conflict if >1 source.
                for idx, orig, vp in items:
                    key = orig if len(items) == 1 else f"{orig}__src{idx}"
                    merged_variables[key] = VariableParams(
                        name=key,
                        distribution=vp.distribution,
                        mean=vp.mean,
                        std=vp.std,
                        min=vp.min,
                        max=vp.max,
                        categories=vp.categories,
                    )
                if len(items) > 1:
                    decisions.append(ReconciliationDecision(
                        target_variable=norm_name,
                        accepted_claim_ids=accepted_ids,
                        method="kept_separate",
                        assumption="incompatible means/stds across sources",
                        note="conflict: measures not pooled",
                    ))

        # --- Correlations ---
        corr_groups: Dict[tuple, List[Tuple[int, CorrelationParams]]] = defaultdict(list)
        for idx, params in enumerate(params_list):
            for key, cp in (params.correlations or {}).items():
                pair = tuple(sorted((_norm(cp.var1), _norm(cp.var2))))
                corr_groups[pair].append((idx, cp))

        merged_correlations: Dict[str, CorrelationParams] = {}
        for pair, items in corr_groups.items():
            # Compatibility: correlation values within tolerance.
            vals = [cp.correlation for _, cp in items]
            spread = max(vals) - min(vals)
            compatible = spread <= 0.25
            if compatible and len(items) > 1:
                corr_val = sum(vals) / len(vals)
                cp0 = items[0][1]
                key = f"{cp0.var1}__{cp0.var2}"
                merged_correlations[key] = CorrelationParams(
                    var1=cp0.var1, var2=cp0.var2,
                    correlation=corr_val, direction=cp0.direction,
                )
                decisions.append(ReconciliationDecision(
                    target_variable=key,
                    accepted_claim_ids=[f"src{idx}" for idx, _ in items],
                    pooled_value=corr_val,
                    method="pooled_average",
                    assumption="correlation values within 0.25",
                ))
            else:
                for idx, cp in items:
                    key = f"{cp.var1}__{cp.var2}" if len(items) == 1 else f"{cp.var1}__{cp.var2}__src{idx}"
                    merged_correlations[key] = CorrelationParams(
                        var1=cp.var1, var2=cp.var2,
                        correlation=cp.correlation, direction=cp.direction,
                    )
                if len(items) > 1:
                    decisions.append(ReconciliationDecision(
                        target_variable=f"{pair[0]}__{pair[1]}",
                        accepted_claim_ids=[f"src{idx}" for idx, _ in items],
                        method="kept_separate",
                        note="conflict: correlation values disagree",
                    ))

        sources = ", ".join(p.meta.source for p in params_list)
        return Parameters(
            variables=merged_variables,
            correlations=merged_correlations,
            meta=MetaParams(
                source=f"reconciled({sources})",
                extracted_at=params_list[0].meta.extracted_at,
                extraction_method=params_list[0].meta.extraction_method,
            ),
        ), decisions


def reconcile_parameters(params_list: List[Parameters]) -> Tuple[Parameters, List[ReconciliationDecision]]:
    return Reconciler().reconcile(params_list)