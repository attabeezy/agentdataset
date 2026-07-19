"""Offline tests for the benchmark's source-text template.

These prove the template <-> regex-extraction contract (including 3+ category
variables) without any network access or API keys.
"""

import numpy as np
import pandas as pd

from agentdataset.core.extractor import Extractor
from benchmark import _dataset_to_source_text


def _toy_dataset() -> dict:
    """Small in-memory dataset with a 3-category feature and a binary target."""
    rng = np.random.default_rng(0)
    n = 1000
    status = rng.choice(["married", "never_married", "prev_married"], size=n, p=[0.5, 0.3, 0.2])
    age = rng.normal(40, 10, n)
    outcome = np.where(rng.random(n) < 0.4, "yes", "no")
    df = pd.DataFrame({"age": age, "status": status, "outcome": outcome})
    return {
        "name": "toy",
        "domain": "test",
        "df": df,
        "continuous": ["age"],
        "categorical_features": ["status"],
        "target": "outcome",
    }


def test_source_text_round_trips_through_regex():
    dataset = _toy_dataset()
    df = dataset["df"]
    text = _dataset_to_source_text(dataset)

    variables, correlations = Extractor()._extract_with_regex(text)

    # 3-category feature round-trips with its real frequencies.
    assert variables["status"].distribution == "categorical"
    status_freqs = df["status"].value_counts(normalize=True)
    for label in ("married", "never_married", "prev_married"):
        assert abs(variables["status"].categories[label] - status_freqs[label]) < 0.001

    # Binary target still round-trips.
    assert variables["outcome"].distribution == "categorical"
    outcome_freqs = df["outcome"].value_counts(normalize=True)
    for label in ("yes", "no"):
        assert abs(variables["outcome"].categories[label] - outcome_freqs[label]) < 0.001

    # Continuous variable and the declared correlations parse too.
    mean_vars = [v for v in variables.values() if v.distribution == "normal"]
    assert any(abs(v.mean - df["age"].mean()) < 0.001 for v in mean_vars)
    assert len(correlations) == 2  # age-outcome and status-outcome
