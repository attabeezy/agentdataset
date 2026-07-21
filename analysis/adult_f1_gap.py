"""Characterize the Adult-Income TSTR F1 gap (paper critique #3).

On UCI Adult, TSTR F1 for AgentDataset (~0.12) collapses relative to TRTR
(~0.55) even though TSTR accuracy stays decent (~0.74). The hypothesis is
minority-class (`>50K`) miscalibration: a classifier trained on AgentDataset
synthetic data under-predicts the positive class because the pairwise-correlation
copula fails to reproduce the joint feature structure that separates the minority
positive class.

This script reproduces the *exact* Adult pipeline used by
`benchmark.run_empirical_benchmark` (same loader, source text, extraction,
split, encoding, and LogisticRegression), then computes evidence for/against the
hypothesis for four training sources: real (TRTR), agentdataset-synth,
independent-marginals-synth, and sdv-gaussian-copula-synth.

It does NOT reimplement extraction/synthesis/splitting; it imports and reuses the
`benchmark` module so the numbers line up with Table 1. Run (a reviewer runs it,
not the author) with a live OPENROUTER_API_KEY in .env.

Metrics produced (per training source):
  1. Per-class precision/recall/F1 on the real test set (classification_report).
  2. Confusion matrices (printed readably) for a representative seed.
  3. Predicted-positive rate (fraction of test rows predicted `>50K`) vs the true
     test positive rate, averaged across seeds.
  4. Positive-class prevalence in the synthetic training data vs real train.
  5. Feature<->target correlation preservation for each continuous feature
     (real train vs each synthesizer), point-biserial via np.corrcoef.

Items 3-5 are aggregated across seeds (mean +/- std). The classification report
and confusion matrices are shown for a single representative seed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import benchmark
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.synthesizer import Synthesizer

# Keep it cheap: extraction is one LLM call, reused across all seeds (as the
# benchmark does). Fewer seeds than the 5-seed benchmark since this is a
# diagnostic, not a headline number.
SEEDS = list(range(3))

# The four training sources characterized. "real" is TRTR; the rest are TSTR.
_SOURCES = ["real", "agentdataset", "independent", "sdv_gaussian_copula"]


def _positive_label(target_encoder: LabelEncoder) -> tuple:
    """Return (positive_label_str, positive_code) for the minority `>50K` class.

    LabelEncoder sorts alphabetically, so `<=50K` -> 0 and `>50K` -> 1; the
    positive class of interest is `>50K`. Resolve it by name rather than
    assuming the code so the analysis is robust to label formatting.
    """
    classes = list(target_encoder.classes_)
    pos_label = next((c for c in classes if c.strip().startswith(">")), classes[-1])
    pos_code = int(target_encoder.transform([pos_label])[0])
    return pos_label, pos_code


def _feature_target_corrs(frame, continuous, target, target_encoder):
    """Point-biserial corr(feature, encoded target) for each continuous feature.

    np.corrcoef on the label-encoded (0/1) target reduces to point-biserial
    correlation, which is the signal a logistic-regression classifier leans on to
    separate the minority class. If AgentDataset preserves the target marginal but
    flattens these correlations, that points to broken feature<->target joint
    structure rather than a skewed marginal.
    """
    y = target_encoder.transform(frame[target])
    out = {}
    for col in continuous:
        x = frame[col].to_numpy(dtype=float)
        if np.std(x) == 0 or np.std(y) == 0:
            out[col] = float("nan")
        else:
            out[col] = float(np.corrcoef(x, y)[0, 1])
    return out


def main():
    benchmark.load_env()

    dataset = benchmark._load_adult()
    name = dataset["name"]
    target = dataset["target"]
    continuous = dataset["continuous"]
    categorical = dataset["categorical_features"]
    feature_cols = continuous + categorical
    df = dataset["df"]

    print("=" * 60)
    print(f"Adult F1-gap analysis: {name} ({dataset['domain']})")
    print("=" * 60)

    # Extraction: one LLM call, hoisted above the seed loop (matches the
    # benchmark). The orchestrator built here is only used for extraction; a
    # fresh one is created per seed below so the optimization ratchet is clean.
    text = benchmark._dataset_to_source_text(dataset)
    extract_orchestrator = Orchestrator(
        session_id=f"adult_f1_gap_{name}",
        model=benchmark._OPENROUTER_MODEL,
        env_var=benchmark._OPENROUTER_ENV_VAR,
    )
    params = extract_orchestrator.extractor.extract_parameters(text, name)
    if target not in params.variables:
        raise RuntimeError(
            f"Extraction did not recover target '{target}'; cannot run analysis."
        )
    print(f"Extraction method: {params.meta.extraction_method}\n")

    # Per-seed collectors for the aggregated metrics (items 3-5).
    pred_pos_rate = {s: [] for s in _SOURCES}      # predicted-positive rate on test
    true_pos_rate = []                              # true test positive rate
    train_pos_prev = {s: [] for s in _SOURCES}      # positive prevalence in training frame
    ft_corrs = {s: {c: [] for c in continuous} for s in _SOURCES}  # feature<->target corr

    # Representative seed (first) reports: classification report + confusion matrix.
    rep_reports = {}
    rep_confusions = {}
    rep_seed = SEEDS[0]
    class_labels = None

    for seed in SEEDS:
        print(f"--- seed {seed} ---")
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=seed, stratify=df[target]
        )

        # Synthesis mirrors the benchmark exactly: np.random.seed for parity,
        # fresh Orchestrator + Synthesizer(n_rows=len(train), seed=seed) per seed,
        # 3 optimization iterations.
        np.random.seed(seed)
        orchestrator = Orchestrator(
            session_id=f"adult_f1_gap_{name}_{seed}",
            model=benchmark._OPENROUTER_MODEL,
            env_var=benchmark._OPENROUTER_ENV_VAR,
        )
        orchestrator.synthesizer = Synthesizer(n_rows=len(train_df), seed=seed)
        _, df_synth = orchestrator.run_optimization_loop(params, iterations=3)

        # Encoders fit on the real train split, shared by every source and the
        # test set (same fair comparison as the benchmark).
        target_encoder = LabelEncoder().fit(train_df[target])
        feature_encoders = {
            col: LabelEncoder().fit(train_df[col]) for col in categorical
        }
        pos_label, pos_code = _positive_label(target_encoder)
        if class_labels is None:
            class_labels = list(target_encoder.classes_)

        def _prepare_X_y(frame):
            X = frame[feature_cols].copy()
            for col, enc in feature_encoders.items():
                X[col] = enc.transform(X[col])
            y = target_encoder.transform(frame[target])
            return X, y

        X_test, y_test = _prepare_X_y(test_df)
        true_pos_rate.append(float(np.mean(y_test == pos_code)))

        # Build the four candidate training frames. "real" is TRTR; the three
        # synthesizers reuse the exact benchmark helpers so nothing diverges.
        frames = {
            "real": train_df,
            "agentdataset": df_synth,
            "independent": benchmark.synthesize_independent_marginals(
                params, len(train_df), seed
            ),
            "sdv_gaussian_copula": benchmark.synthesize_sdv_gaussian_copula(
                train_df, feature_cols, target, len(train_df), seed
            ),
        }

        for source, frame in frames.items():
            if frame is None:
                # SDV unavailable/failed; skip only this source for this seed.
                print(f"  [{source}] unavailable this seed; skipping.")
                continue
            if any(c not in frame.columns for c in feature_cols + [target]):
                print(f"  [{source}] missing required columns; skipping.")
                continue

            try:
                X_train, y_train = _prepare_X_y(frame)
            except ValueError as ve:
                # Baseline categoricals may contain unseen labels; skip source/seed.
                # Collect ALL per-source metrics behind this same guard (the
                # benchmark tolerates unseen labels by skipping only that
                # source/seed). _feature_target_corrs and the prevalence check
                # also call target_encoder.transform, so they must sit after
                # _prepare_X_y succeeds to avoid aborting the whole run.
                print(f"  [{source}] unseen labels ({ve}); skipping.")
                continue

            # Positive-class prevalence in the (real or synthetic) training frame.
            train_pos_prev[source].append(
                float(np.mean(frame[target].astype(str).str.strip().str.startswith(">")))
            )
            # Feature<->target correlation preservation.
            corrs = _feature_target_corrs(frame, continuous, target, target_encoder)
            for col in continuous:
                ft_corrs[source][col].append(corrs[col])

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            pred_pos_rate[source].append(float(np.mean(preds == pos_code)))

            if seed == rep_seed:
                rep_reports[source] = classification_report(
                    y_test, preds,
                    labels=list(range(len(target_encoder.classes_))),
                    target_names=[str(c) for c in target_encoder.classes_],
                    zero_division=0,
                )
                rep_confusions[source] = confusion_matrix(
                    y_test, preds, labels=list(range(len(target_encoder.classes_)))
                )

    # ---- Item 1 & 2: per-class report + confusion matrix (representative seed) ----
    print("\n" + "=" * 60)
    print(f"PER-CLASS METRICS + CONFUSION MATRICES (representative seed={rep_seed})")
    print("classes:", class_labels, "(rows=true, cols=predicted)")
    print("=" * 60)
    for source in _SOURCES:
        if source not in rep_reports:
            continue
        print(f"\n[{source}] classification_report:")
        print(rep_reports[source])
        print(f"[{source}] confusion_matrix:")
        print(rep_confusions[source])

    # ---- Item 3: predicted-positive rate vs true positive rate (across seeds) ----
    print("\n" + "=" * 60)
    print("PREDICTED-POSITIVE RATE vs TRUE POSITIVE RATE (mean +/- std across seeds)")
    print("=" * 60)
    true_mean, true_std = np.mean(true_pos_rate), _std(true_pos_rate)
    print(f"  true test positive rate: {true_mean:.4f} +/- {true_std:.4f}")
    for source in _SOURCES:
        vals = pred_pos_rate[source]
        if not vals:
            continue
        print(
            f"  [{source:<20}] predicted-positive rate: "
            f"{np.mean(vals):.4f} +/- {_std(vals):.4f}"
        )

    # ---- Item 4: positive-class prevalence in training data (across seeds) ----
    print("\n" + "=" * 60)
    print("POSITIVE-CLASS PREVALENCE IN TRAINING DATA (mean +/- std across seeds)")
    print("(real = train prevalence; distortion here => skewed marginal, not just joint)")
    print("=" * 60)
    for source in _SOURCES:
        vals = train_pos_prev[source]
        if not vals:
            continue
        print(
            f"  [{source:<20}] train positive prevalence: "
            f"{np.mean(vals):.4f} +/- {_std(vals):.4f}"
        )

    # ---- Item 5: feature<->target correlation preservation (across seeds) ----
    print("\n" + "=" * 60)
    print("FEATURE<->TARGET CORRELATION (point-biserial, mean +/- std across seeds)")
    print("(real corr is the signal; flattened synth corr => broken joint structure)")
    print("=" * 60)
    for col in continuous:
        print(f"\n  feature: {col}")
        for source in _SOURCES:
            vals = [v for v in ft_corrs[source][col] if not np.isnan(v)]
            if not vals:
                continue
            print(
                f"    [{source:<20}] corr({col}, target): "
                f"{np.mean(vals):+.4f} +/- {_std(vals):.4f}"
            )

    # ---- Tidy summary CSV ----
    summary_rows = []
    for source in _SOURCES:
        row = {
            "dataset": name,
            "source": source,
            "true_pos_rate_mean": np.mean(true_pos_rate) if true_pos_rate else float("nan"),
            "true_pos_rate_std": _std(true_pos_rate),
            "pred_pos_rate_mean": np.mean(pred_pos_rate[source]) if pred_pos_rate[source] else float("nan"),
            "pred_pos_rate_std": _std(pred_pos_rate[source]),
            "train_pos_prev_mean": np.mean(train_pos_prev[source]) if train_pos_prev[source] else float("nan"),
            "train_pos_prev_std": _std(train_pos_prev[source]),
            "n_seeds": len(pred_pos_rate[source]),
        }
        for col in continuous:
            vals = [v for v in ft_corrs[source][col] if not np.isnan(v)]
            row[f"corr_{col}_target_mean"] = np.mean(vals) if vals else float("nan")
            row[f"corr_{col}_target_std"] = _std(vals)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    benchmark._save_results(df_summary, "adult_f1_gap.csv")
    print("\n" + df_summary.to_string(index=False))
    return df_summary


def _std(vals) -> float:
    """Sample std (ddof=1) matching the benchmark's aggregation; 0.0 for <2 values."""
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0


if __name__ == "__main__":
    main()
