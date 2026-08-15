"""
AgentDataset Synthesizer
Parameters -> DataFrame (Persona-Driven)
"""

import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from agentdataset.models.schemas import Parameters, VariableParams

logger = logging.getLogger(__name__)

# Eigenvalues below this are treated as non-positive-definite. Chosen well above
# float64 round-off on a unit-diagonal matrix so exactly-singular inputs (a
# perfectly collinear pair, r=1.0) are classified as invalid rather than as
# borderline-valid, which Cholesky would reject anyway.
PD_EIGENVALUE_TOL = 1e-8

# Repair modes for an invalid correlation matrix.
CORR_REPAIR_MODES = ("drop", "psd", "none")


@dataclass
class CorrDiagnostics:
    """One record per synthesize() call that declared correlations.

    Written to `Synthesizer.last_diagnostics` and appended to
    `Synthesizer.diagnostics`. §9.7 uses these to quantify how often the
    advertised copula path actually runs; see analysis/corr_validity_audit.py.
    """

    source_label: str = ""
    n_vars: int = 0
    n_pairs_declared: int = 0
    n_pairs_applied: int = 0
    n_pairs_skipped_unknown_var: int = 0
    n_pairs_overwritten: int = 0
    min_eigenvalue: float = float("nan")
    condition_number: float = float("nan")
    is_pd: bool = False
    repair_mode: str = "drop"
    # "no_correlations" | "cholesky_ok" | "dropped_all" | "repaired_psd" | "used_invalid"
    action: str = "no_correlations"
    frobenius_delta: float = 0.0
    min_eigenvalue_after: float = float("nan")

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


def nearest_psd_correlation(
    matrix: np.ndarray, eps: float = PD_EIGENVALUE_TOL
) -> np.ndarray:
    """Project a symmetric matrix onto the nearest PSD correlation matrix.

    Eigenvalue clipping followed by renormalisation of the diagonal to 1. This
    is the first iterate of Higham's (2002) alternating-projection algorithm;
    the full alternating projection is closer in Frobenius norm but needs an
    iteration loop and a convergence tolerance, which is not worth the extra
    moving parts for the matrix sizes seen here (n_vars is single digits).

    Renormalising the diagonal can reintroduce a slightly negative eigenvalue,
    so the clip is repeated once; callers should still read
    `min_eigenvalue_after` rather than assume strict positive-definiteness.
    """
    sym = (matrix + matrix.T) / 2.0

    for _ in range(2):
        eigenvalues, eigenvectors = np.linalg.eigh(sym)
        if eigenvalues.min() >= eps:
            break
        clipped = np.clip(eigenvalues, eps, None)
        sym = (eigenvectors * clipped) @ eigenvectors.T
        # Rescale to a correlation matrix: unit diagonal.
        scale = np.sqrt(np.diag(sym))
        scale[scale <= 0] = 1.0
        sym = sym / np.outer(scale, scale)
        np.fill_diagonal(sym, 1.0)
        sym = (sym + sym.T) / 2.0

    return sym


def _correlated_normals(
    base_data: np.ndarray, corr_matrix: np.ndarray
) -> np.ndarray:
    """Impose `corr_matrix` on iid standard normals.

    Uses the eigendecomposition factor V diag(sqrt(max(lambda,0))) rather than a
    Cholesky factor. The two are equivalent for positive-definite input, but the
    eigendecomposition also succeeds on positive-*semi*definite matrices, where
    Cholesky raises. A repaired matrix sits exactly on that boundary, so this
    avoids having to add jitter to make the factorisation go through.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return base_data @ factor.T


class Synthesizer:
    def __init__(
        self,
        n_rows: int = 10000,
        seed: int = 42,
        corr_repair: str = "drop",
    ):
        """corr_repair selects the invalid-correlation-matrix policy:

        - "drop" (default): discard *all* correlations and synthesize the
          variables independently. This is the shipped behaviour and is kept as
          the default so existing results stay reproducible.
        - "psd":  project onto the nearest PSD correlation matrix and sample
          from that, retaining as much of the declared structure as possible.
        - "none": sample from the matrix as declared. Only meaningful as a
          study arm; an invalid matrix yields an ill-defined target.
        """
        if corr_repair not in CORR_REPAIR_MODES:
            raise ValueError(
                f"corr_repair must be one of {CORR_REPAIR_MODES}, got {corr_repair!r}"
            )
        self.n_rows = n_rows
        self.seed = seed
        self.corr_repair = corr_repair
        self.rng = np.random.default_rng(seed)
        self.last_diagnostics: Optional[CorrDiagnostics] = None
        self.diagnostics: List[CorrDiagnostics] = []

    def generate_variable(self, params: VariableParams, noise_level: float) -> np.ndarray:
        """Generate single variable data."""
        mean = params.mean
        std = params.std

        if params.distribution == "categorical" and params.categories:
            # Rank-transform (in synthesize()) needs sortable values, so draw
            # integer category codes here; synthesize() maps them back to labels.
            labels = list(params.categories.keys())
            probs = np.array([params.categories[label] for label in labels], dtype=float)
            probs = probs / probs.sum()
            data = self.rng.choice(len(labels), size=self.n_rows, p=probs)
        elif params.distribution == "uniform":
            low = params.min if params.min is not None else mean - 2*std
            high = params.max if params.max is not None else mean + 2*std
            spread = (high - low) * noise_level * 0.5
            data = self.rng.uniform(low - spread, high + spread, self.n_rows)
        elif params.distribution == "gamma" and mean > 0 and std > 0:
            # Gamma requires shape>0 and scale>0; only valid for positive mean/std.
            shape = (mean / std) ** 2
            scale = std**2 / mean
            data = self.rng.gamma(shape, scale, self.n_rows)
        else:
            # normal, unknown distribution, or gamma with non-positive mean/std
            data = self.rng.normal(mean, std * (1 + noise_level), self.n_rows)

        return data

    def _build_correlation_matrix(
        self, parameters: Parameters, var_names: List[str], diag: CorrDiagnostics
    ) -> np.ndarray:
        """Assemble the target matrix, recording what was dropped on the way."""
        n_vars = len(var_names)
        corr_matrix = np.eye(n_vars)
        seen: set[tuple[int, int]] = set()

        for key, corr_params in parameters.correlations.items():
            v1, v2 = corr_params.var1, corr_params.var2
            if v1 not in var_names or v2 not in var_names:
                diag.n_pairs_skipped_unknown_var += 1
                continue
            idx1, idx2 = var_names.index(v1), var_names.index(v2)
            if idx1 == idx2:
                # A self-correlation cannot be represented; the diagonal is 1.
                diag.n_pairs_skipped_unknown_var += 1
                continue
            pair = (min(idx1, idx2), max(idx1, idx2))
            if pair in seen:
                # Later declarations overwrite earlier ones (last write wins).
                diag.n_pairs_overwritten += 1
            else:
                seen.add(pair)
            corr_matrix[idx1, idx2] = corr_params.correlation
            corr_matrix[idx2, idx1] = corr_params.correlation

        diag.n_pairs_applied = len(seen)
        return corr_matrix

    def _handle_invalid_matrix(
        self, base_data: np.ndarray, corr_matrix: np.ndarray, diag: CorrDiagnostics
    ) -> np.ndarray:
        """Apply the configured policy for a non-positive-definite matrix."""
        if self.corr_repair == "psd":
            repaired = nearest_psd_correlation(corr_matrix)
            diag.frobenius_delta = float(
                np.linalg.norm(repaired - corr_matrix, ord="fro")
            )
            diag.min_eigenvalue_after = float(np.linalg.eigvalsh(repaired).min())
            diag.action = "repaired_psd"
            logger.info(
                "Correlation matrix for %r is not positive-definite (min "
                "eigenvalue %.3e); repaired to nearest PSD (Frobenius delta %.3e).",
                diag.source_label,
                diag.min_eigenvalue,
                diag.frobenius_delta,
            )
            return _correlated_normals(base_data, repaired)

        if self.corr_repair == "none":
            diag.action = "used_invalid"
            return _correlated_normals(base_data, corr_matrix)

        # "drop": every correlation is discarded, including the well-behaved
        # pairs, leaving base_data as the iid draw the caller made.
        diag.action = "dropped_all"
        logger.warning(
            "Correlation matrix for %r is not positive-definite (min eigenvalue "
            "%.3e); dropping all %d correlations.",
            diag.source_label,
            diag.min_eigenvalue,
            diag.n_pairs_applied,
        )
        warnings.warn(
            "Correlation matrix is not positive-definite; synthesizing variables independently.",
            RuntimeWarning,
            stacklevel=3,
        )
        return base_data

    def synthesize(
        self,
        parameters: Parameters,
        noise_level: float = 0.1,
        source_label: str = "",
    ) -> pd.DataFrame:
        """Generate correlated dataset."""
        var_names = list(parameters.variables.keys())
        n_vars = len(var_names)

        if n_vars == 0:
            return pd.DataFrame()

        # Generate base correlation structure
        base_data = self.rng.standard_normal((self.n_rows, n_vars))

        diag = CorrDiagnostics(
            source_label=source_label or parameters.meta.source,
            n_vars=n_vars,
            n_pairs_declared=len(parameters.correlations or {}),
            repair_mode=self.corr_repair,
        )

        if parameters.correlations:
            corr_matrix = self._build_correlation_matrix(parameters, var_names, diag)

            # Measure PD status directly instead of inferring it from a Cholesky
            # exception — the eigenvalue is the number the paper needs to report.
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            diag.min_eigenvalue = float(eigenvalues.min())
            diag.min_eigenvalue_after = diag.min_eigenvalue
            largest = float(np.abs(eigenvalues).max())
            smallest = float(np.abs(eigenvalues).min())
            diag.condition_number = (
                float("inf") if smallest == 0 else largest / smallest
            )
            diag.is_pd = diag.min_eigenvalue > PD_EIGENVALUE_TOL

            if diag.is_pd:
                # Deliberately still Cholesky, not the eigendecomposition used
                # for repaired matrices: the two factors are both valid but
                # produce different samples, and this path's output is
                # hash-frozen in results/baseline_manifest.json.
                try:
                    L = np.linalg.cholesky(corr_matrix)
                    base_data = base_data @ L.T
                    diag.action = "cholesky_ok"
                except np.linalg.LinAlgError:
                    # Eigenvalues said PD but the factorisation disagreed —
                    # borderline conditioning. Fall through to the configured
                    # invalid-matrix policy rather than silently continuing.
                    diag.is_pd = False

            if diag.action != "cholesky_ok":
                base_data = self._handle_invalid_matrix(base_data, corr_matrix, diag)

        self.last_diagnostics = diag
        self.diagnostics.append(diag)

        data_dict = {}
        for i, name in enumerate(var_names):
            params = parameters.variables[name]
            raw_data = self.generate_variable(params, noise_level)

            # Apply rank transform for correlation
            ranks = pd.Series(base_data[:, i]).rank()
            reordered = np.take(np.sort(raw_data), np.argsort(ranks).argsort())

            if params.distribution == "categorical" and params.categories:
                labels = np.array(list(params.categories.keys()))
                data_dict[name] = labels[reordered.astype(int)]
            else:
                data_dict[name] = reordered

        return pd.DataFrame(data_dict)
