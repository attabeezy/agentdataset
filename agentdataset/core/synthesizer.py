"""
AgentDataset Synthesizer
Parameters -> DataFrame (Persona-Driven)
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from agentdataset.models.schemas import Parameters, VariableParams

class Synthesizer:
    def __init__(self, n_rows: int = 10000, seed: int = 42):
        self.n_rows = n_rows
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_variable(self, params: VariableParams, noise_level: float) -> np.ndarray:
        """Generate single variable data."""
        mean = params.mean
        std = params.std
        noise_std = std * noise_level

        if params.distribution == "normal":
            data = self.rng.normal(mean, std * (1 + noise_level), self.n_rows)
        elif params.distribution == "uniform":
            low = params.min if params.min is not None else mean - 2*std
            high = params.max if params.max is not None else mean + 2*std
            spread = (high - low) * noise_level * 0.5
            data = self.rng.uniform(low - spread, high + spread, self.n_rows)
        elif params.distribution == "gamma":
            shape = (mean / std) ** 2
            scale = std**2 / mean
            data = self.rng.gamma(shape, scale, self.n_rows)
        else:
            data = self.rng.normal(mean, std, self.n_rows)
        
        return data

    def synthesize(self, parameters: Parameters, noise_level: float = 0.1) -> pd.DataFrame:
        """Generate correlated dataset."""
        var_names = list(parameters.variables.keys())
        n_vars = len(var_names)
        
        if n_vars == 0:
            return pd.DataFrame()

        # Generate base correlation structure
        base_data = self.rng.standard_normal((self.n_rows, n_vars))
        
        if parameters.correlations:
            corr_matrix = np.eye(n_vars)
            for key, corr_params in parameters.correlations.items():
                v1, v2 = corr_params.var1, corr_params.var2
                if v1 in var_names and v2 in var_names:
                    idx1, idx2 = var_names.index(v1), var_names.index(v2)
                    corr_matrix[idx1, idx2] = corr_params.correlation
                    corr_matrix[idx2, idx1] = corr_params.correlation
            
            try:
                L = np.linalg.cholesky(corr_matrix)
                base_data = base_data @ L.T
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Correlation matrix is not positive-definite; synthesizing variables independently.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        data_dict = {}
        for i, name in enumerate(var_names):
            params = parameters.variables[name]
            raw_data = self.generate_variable(params, noise_level)
            
            # Apply rank transform for correlation
            ranks = pd.Series(base_data[:, i]).rank()
            data_dict[name] = np.take(np.sort(raw_data), np.argsort(ranks).argsort())

        return pd.DataFrame(data_dict)
