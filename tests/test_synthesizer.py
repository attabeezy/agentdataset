import pytest
import pandas as pd
import numpy as np
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.models.schemas import Parameters, VariableParams, CorrelationParams, MetaParams

def test_synthesizer_init():
    syn = Synthesizer(n_rows=500, seed=123)
    assert syn.n_rows == 500
    assert syn.seed == 123

def test_generate_variable():
    syn = Synthesizer(n_rows=100)
    params = VariableParams(name="v1", distribution="normal", mean=10.0, std=1.0)
    data = syn.generate_variable(params, noise_level=0.0)
    assert len(data) == 100
    assert abs(np.mean(data) - 10.0) < 0.5

def test_synthesize_no_vars():
    syn = Synthesizer()
    params = Parameters(variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N"))
    df = syn.synthesize(params)
    assert df.empty

def test_synthesize_with_vars():
    syn = Synthesizer(n_rows=100)
    v1 = VariableParams(name="v1", mean=0, std=1)
    v2 = VariableParams(name="v2", mean=10, std=2)
    params = Parameters(
        variables={"v1": v1, "v2": v2},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    df = syn.synthesize(params)
    assert df.shape == (100, 2)
    assert "v1" in df.columns
    assert "v2" in df.columns


def test_generate_variable_categorical():
    syn = Synthesizer(n_rows=1000)
    params = VariableParams(name="v1", distribution="categorical", categories={"yes": 0.7, "no": 0.3})
    data = syn.generate_variable(params, noise_level=0.0)
    assert len(data) == 1000
    # Integer codes (0/1), mapped to labels later in synthesize().
    assert set(np.unique(data)).issubset({0, 1})


def test_synthesize_categorical_variable_produces_labels():
    syn = Synthesizer(n_rows=1000, seed=0)
    params = Parameters(
        variables={
            "sex": VariableParams(name="sex", distribution="categorical", categories={"Female": 0.4, "Male": 0.6}),
        },
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    assert set(df["sex"].unique()) <= {"Female", "Male"}
    freqs = df["sex"].value_counts(normalize=True)
    assert abs(freqs["Male"] - 0.6) < 0.1


def test_synthesize_categorical_correlated_with_continuous():
    """A continuous variable strongly correlated with a binary categorical
    variable should show a real difference in means across categories."""
    syn = Synthesizer(n_rows=2000, seed=0)
    params = Parameters(
        variables={
            "score": VariableParams(name="score", distribution="normal", mean=0.0, std=1.0),
            "outcome": VariableParams(name="outcome", distribution="categorical", categories={"no": 0.5, "yes": 0.5}),
        },
        correlations={
            "c": CorrelationParams(var1="score", var2="outcome", correlation=0.8),
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    mean_yes = df.loc[df["outcome"] == "yes", "score"].mean()
    mean_no = df.loc[df["outcome"] == "no", "score"].mean()
    assert mean_yes > mean_no


def test_synthesize_negative_direction_reverses_dependence():
    syn = Synthesizer(n_rows=2000, seed=0)
    params = Parameters(
        variables={
            "score": VariableParams(name="score", distribution="normal", mean=0.0, std=1.0),
            "outcome": VariableParams(name="outcome", distribution="categorical", categories={"no": 0.5, "yes": 0.5}),
        },
        correlations={
            "c": CorrelationParams(var1="score", var2="outcome", correlation=0.8, direction="negative"),
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    mean_yes = df.loc[df["outcome"] == "yes", "score"].mean()
    mean_no = df.loc[df["outcome"] == "no", "score"].mean()
    assert mean_yes < mean_no


def test_generate_variable_categorical_three_categories():
    syn = Synthesizer(n_rows=3000)
    params = VariableParams(
        name="v1", distribution="categorical", categories={"a": 0.5, "b": 0.3, "c": 0.2}
    )
    data = syn.generate_variable(params, noise_level=0.0)
    assert set(np.unique(data)).issubset({0, 1, 2})
    freqs = pd.Series(data).value_counts(normalize=True)
    assert abs(freqs[0] - 0.5) < 0.05
    assert abs(freqs[1] - 0.3) < 0.05
    assert abs(freqs[2] - 0.2) < 0.05


def test_synthesize_three_category_variable_produces_labels():
    syn = Synthesizer(n_rows=3000, seed=0)
    params = Parameters(
        variables={
            "status": VariableParams(
                name="status",
                distribution="categorical",
                categories={"married": 0.5, "never_married": 0.3, "prev_married": 0.2},
            ),
        },
        correlations={},
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    assert set(df["status"].unique()) == {"married", "never_married", "prev_married"}
    freqs = df["status"].value_counts(normalize=True)
    assert abs(freqs["married"] - 0.5) < 0.05
    assert abs(freqs["never_married"] - 0.3) < 0.05
    assert abs(freqs["prev_married"] - 0.2) < 0.05


def test_synthesize_three_category_correlated_with_continuous():
    """The copula treats category codes as ordinal (insertion order of the
    categories dict), so a strong positive correlation should produce
    monotonically increasing group means across the three categories."""
    syn = Synthesizer(n_rows=3000, seed=0)
    params = Parameters(
        variables={
            "score": VariableParams(name="score", distribution="normal", mean=0.0, std=1.0),
            "level": VariableParams(
                name="level",
                distribution="categorical",
                categories={"low": 0.4, "mid": 0.3, "high": 0.3},
            ),
        },
        correlations={
            "c": CorrelationParams(var1="score", var2="level", correlation=0.8),
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )
    df = syn.synthesize(params)
    mean_low = df.loc[df["level"] == "low", "score"].mean()
    mean_mid = df.loc[df["level"] == "mid", "score"].mean()
    mean_high = df.loc[df["level"] == "high", "score"].mean()
    assert mean_low < mean_mid < mean_high


def test_gamma_with_non_positive_mean_does_not_crash():
    """Gamma needs mean>0; non-positive mean must fall back to normal, not crash."""
    syn = Synthesizer(n_rows=100)
    params = VariableParams(name="g", distribution="gamma", mean=-3.0, std=2.0)
    data = syn.generate_variable(params, noise_level=0.1)
    assert len(data) == 100
    assert np.all(np.isfinite(data))


def test_non_positive_definite_corr_warns_and_falls_back():
    """An impossible correlation matrix triggers a RuntimeWarning, not a crash."""
    syn = Synthesizer(n_rows=100)
    params = Parameters(
        variables={
            "a": VariableParams(name="a", mean=0, std=1),
            "b": VariableParams(name="b", mean=0, std=1),
            "c": VariableParams(name="c", mean=0, std=1),
        },
        # Mutually contradictory correlations → not positive-definite
        correlations={
            "ab": CorrelationParams(var1="a", var2="b", correlation=0.9),
            "ac": CorrelationParams(var1="a", var2="c", correlation=0.9),
            "bc": CorrelationParams(var1="b", var2="c", correlation=-0.9),
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )
    with pytest.warns(RuntimeWarning):
        df = syn.synthesize(params)
    assert df.shape == (100, 3)


# --- §9.7: correlation-matrix validity diagnostics and repair -----------------


def _params(corrs, n_vars=3):
    """Standard-normal variables named a, b, c, ... plus the given pairs."""
    names = [chr(ord("a") + i) for i in range(n_vars)]
    return Parameters(
        variables={n: VariableParams(name=n, mean=0.0, std=1.0) for n in names},
        correlations={
            f"{v1}{v2}": CorrelationParams(var1=v1, var2=v2, correlation=r)
            for v1, v2, r in corrs
        },
        meta=MetaParams(source="S", extracted_at="N"),
    )


def test_valid_matrix_records_cholesky_path():
    """A well-formed matrix reports is_pd and takes the advertised copula path."""
    syn = Synthesizer(n_rows=500, seed=0)
    syn.synthesize(_params([("a", "b", 0.5), ("a", "c", 0.3), ("b", "c", 0.2)]))

    diag = syn.last_diagnostics
    assert diag.is_pd is True
    assert diag.action == "cholesky_ok"
    assert diag.min_eigenvalue > 0
    assert diag.n_pairs_applied == 3
    assert diag.frobenius_delta == 0.0


def test_slightly_invalid_matrix_is_detected_and_repairable():
    """r=(.9,.9,.61) is just outside the PD cone: repair recovers it closely."""
    corrs = [("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", 0.61)]

    dropper = Synthesizer(n_rows=4000, seed=0, corr_repair="drop")
    with pytest.warns(RuntimeWarning):
        dropper.synthesize(_params(corrs))
    diag = dropper.last_diagnostics
    assert diag.is_pd is False
    # "Slightly" invalid: negative, but only just.
    assert -0.05 < diag.min_eigenvalue < 0
    assert diag.action == "dropped_all"

    repairer = Synthesizer(n_rows=4000, seed=0, corr_repair="psd")
    df = repairer.synthesize(_params(corrs))
    rdiag = repairer.last_diagnostics
    assert rdiag.action == "repaired_psd"
    assert rdiag.min_eigenvalue_after >= -1e-10
    # A small perturbation should stay small.
    assert rdiag.frobenius_delta < 0.2

    # Repair keeps the declared structure; dropping destroys it.
    assert df["a"].corr(df["b"]) > 0.75


def test_severely_invalid_matrix_repairs_to_a_valid_compromise():
    """(.9,.9,-.9) is unsatisfiable; repair must yield a usable matrix anyway."""
    corrs = [("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", -0.9)]

    repairer = Synthesizer(n_rows=2000, seed=0, corr_repair="psd")
    df = repairer.synthesize(_params(corrs))
    diag = repairer.last_diagnostics

    assert diag.action == "repaired_psd"
    assert diag.min_eigenvalue < -0.3, "fixture should be severely invalid"
    assert diag.min_eigenvalue_after >= -1e-10
    assert diag.frobenius_delta > 0.5, "severe input should move a long way"
    assert df.shape == (2000, 3)
    # Deliberately NOT asserting the target correlations are met — they are
    # jointly impossible. The contract is only that the output is well-defined.
    assert df.notna().all().all()


def test_incomplete_matrix_is_valid_with_implicit_zeros():
    """Undeclared pairs default to zero, which stays positive-definite."""
    syn = Synthesizer(n_rows=500, seed=0)
    syn.synthesize(_params([("a", "b", 0.5), ("c", "d", -0.4)], n_vars=4))

    diag = syn.last_diagnostics
    assert diag.is_pd is True
    assert diag.action == "cholesky_ok"
    assert diag.n_vars == 4
    assert diag.n_pairs_declared == 2
    assert diag.n_pairs_applied == 2
    assert diag.n_pairs_skipped_unknown_var == 0


def test_contradictory_declarations_are_counted():
    """Unknown variables are skipped; duplicate pairs are last-write-wins."""
    params = _params([("a", "b", 0.5)])
    params.correlations["ghost"] = CorrelationParams(
        var1="a", var2="nonexistent", correlation=0.8
    )
    params.correlations["dup"] = CorrelationParams(var1="a", var2="b", correlation=-0.2)

    syn = Synthesizer(n_rows=500, seed=0)
    syn.synthesize(params)
    diag = syn.last_diagnostics

    assert diag.n_pairs_declared == 3
    assert diag.n_pairs_skipped_unknown_var == 1
    assert diag.n_pairs_overwritten == 1
    assert diag.n_pairs_applied == 1
    assert diag.is_pd is True


def test_diagnostics_accumulate_across_calls():
    syn = Synthesizer(n_rows=200, seed=0)
    syn.synthesize(_params([("a", "b", 0.5)]))
    syn.synthesize(_params([("a", "b", 0.1)]))
    assert len(syn.diagnostics) == 2
    assert syn.diagnostics[-1] is syn.last_diagnostics


def test_no_correlations_records_no_correlations_action():
    syn = Synthesizer(n_rows=200, seed=0)
    syn.synthesize(_params([]))
    # No correlations declared → the block is skipped entirely.
    assert syn.last_diagnostics.action == "no_correlations"


def test_invalid_repair_mode_rejected():
    with pytest.raises(ValueError, match="corr_repair"):
        Synthesizer(corr_repair="nonsense")


def test_default_output_unchanged_by_diagnostics():
    """Regression guard: results/baseline_manifest.json hashes depend on this.

    The valid path must keep using the Cholesky factor. The eigendecomposition
    factor used for repaired matrices is equally correct but produces different
    samples, which would silently invalidate every frozen baseline.
    """
    params = _params([("a", "b", 0.6), ("a", "c", 0.2), ("b", "c", 0.1)])

    df = Synthesizer(n_rows=1000, seed=42).synthesize(params)

    # Reproduce the pre-change code path explicitly.
    rng = np.random.default_rng(42)
    base = rng.standard_normal((1000, 3))
    corr = np.eye(3)
    for i, j, r in [(0, 1, 0.6), (0, 2, 0.2), (1, 2, 0.1)]:
        corr[i, j] = corr[j, i] = r
    base = base @ np.linalg.cholesky(corr).T
    expected = {}
    for i, name in enumerate(["a", "b", "c"]):
        raw = rng.normal(0.0, 1.0 * (1 + 0.1), 1000)
        ranks = pd.Series(base[:, i]).rank()
        expected[name] = np.take(np.sort(raw), np.argsort(ranks).argsort())

    pd.testing.assert_frame_equal(df, pd.DataFrame(expected))
