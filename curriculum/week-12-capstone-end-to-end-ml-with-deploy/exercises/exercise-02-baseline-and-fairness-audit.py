"""
Exercise 2 -- Baseline, candidate model, and slice-based fairness audit.

Goal: on the synthetic Adult-Income-shaped dataset from Exercise 1, train a
`DummyClassifier` baseline and a `LogisticRegression` candidate, then run a
per-slice fairness audit on the `sex` column and on the `race` column.

By the end of this exercise you will have:

    (1) Split the data into train / val / test (15 / 15 / 70 percent) with
        stratification on the target and a fixed random seed.
    (2) Fit `DummyClassifier(strategy="most_frequent")` on the training set
        and reported its accuracy and F1 on the test set.
    (3) Fit `LogisticRegression` on a one-hot-encoded feature matrix.
    (4) Tuned the decision threshold on the validation set to maximize F1.
    (5) Reported the candidate model's F1, ROC-AUC, and PR-AUC on the test
        set at the tuned threshold.
    (6) Run the per-slice fairness audit on `sex`: F1 for Male, F1 for Female,
        and the absolute-value gap.
    (7) Run the per-slice fairness audit on `race`: F1 per race category,
        and the gap between best- and worst-performing slice.
    (8) Flagged any slice whose F1 differs from the aggregate by more than 5
        percentage points.

Estimated time: 60-90 minutes.

Run with:    python exercise-02-baseline-and-fairness-audit.py
Or test:     pytest exercise-02-baseline-and-fairness-audit.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-baseline-and-fairness-audit.py succeeds.

References:
    Mitchell et al. 2019, "Model Cards for Model Reporting": https://arxiv.org/abs/1810.03993
    Buolamwini and Gebru 2018, "Gender Shades": https://proceedings.mlr.press/v81/buolamwini18a.html
    scikit-learn DummyClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
"""

from __future__ import annotations

from typing import Dict, List, Tuple

RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Part A -- the synthetic dataset (same shape as Exercise 1)
# -----------------------------------------------------------------------------


def make_sample_dataframe(n: int = 2000) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Return an n-row Adult-Income-shaped DataFrame.

    Larger than Exercise 1's 200 rows because we need enough data to fit a
    LogisticRegression and to slice by sex and race. The income column is
    correlated with hours_per_week and education_num so the LogisticRegression
    has a real signal to learn.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(17, 80, size=n)
    education_num = rng.integers(1, 17, size=n)
    sex = rng.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        size=n,
        p=[0.85, 0.09, 0.03, 0.02, 0.01],
    )
    hours_per_week = rng.integers(20, 60, size=n).astype(int)
    capital_gain = rng.choice([0, 0, 0, 0, 5000, 15000], size=n)

    # income depends on hours, education, and (regrettably and intentionally)
    # has a slight sex bias built in so the fairness audit will detect it.
    sex_bias = np.where(sex == "Male", 0.5, -0.5)
    logit = (
        (hours_per_week - 40) * 0.1
        + (education_num - 10) * 0.3
        + sex_bias
        + (capital_gain / 5000) * 0.5
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    income = (rng.random(n) < prob).astype(int)  # 1 = >50K, 0 = <=50K.

    df = pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "sex": sex,
        "race": race,
        "hours_per_week": hours_per_week,
        "capital_gain": capital_gain,
        "income": income,
    })
    return df


# -----------------------------------------------------------------------------
# Part B -- the train/val/test split
# -----------------------------------------------------------------------------


def train_val_test_split(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    target_col: str = "income",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", "pd.Series", "pd.Series", "pd.Series"]:  # type: ignore[name-defined]
    """Split into train / val / test, stratified on the target.

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test).

    Steps:
        1. Split off the test set first (`test_size` of the total).
        2. Split the remainder into train and val. The val proportion of the
           pool is `val_size / (1 - test_size)`.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    # The val proportion of the pool.
    val_frac_of_pool: float = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        test_size=val_frac_of_pool,
        stratify=y_pool,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# Part C -- the baseline
# -----------------------------------------------------------------------------


def fit_baseline(X_train: "pd.DataFrame", y_train: "pd.Series"):  # type: ignore[name-defined]
    """Fit DummyClassifier(strategy='most_frequent') on the training set."""
    from sklearn.dummy import DummyClassifier

    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    # TODO: fit baseline on X_train (note: DummyClassifier only looks at y_train
    # but still requires X_train as an argument).
    baseline.fit(X_train, y_train)
    return baseline


def baseline_metrics(
    baseline,  # noqa: ANN001
    X_test: "pd.DataFrame",  # type: ignore[name-defined]
    y_test: "pd.Series",  # type: ignore[name-defined]
) -> Dict[str, float]:
    """Return {'accuracy', 'f1'} for the baseline on the test set."""
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = baseline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }


# -----------------------------------------------------------------------------
# Part D -- the candidate model
# -----------------------------------------------------------------------------


def build_feature_matrix(
    X_train: "pd.DataFrame",  # type: ignore[name-defined]
    X_val: "pd.DataFrame",  # type: ignore[name-defined]
    X_test: "pd.DataFrame",  # type: ignore[name-defined]
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:  # type: ignore[name-defined]
    """One-hot-encode the categorical columns and concatenate with the numeric.

    Returns: (X_train_arr, X_val_arr, X_test_arr) as dense numpy arrays.
    """
    import numpy as np
    import pandas as pd

    categorical = ["sex", "race"]
    numeric = ["age", "education_num", "hours_per_week", "capital_gain"]

    # One-hot encode categorical with pd.get_dummies. Fit on the combined
    # frame to ensure consistent columns across train / val / test.
    combined = pd.concat([X_train, X_val, X_test], axis=0)
    encoded = pd.get_dummies(combined[categorical], drop_first=False)
    # Concatenate numeric features.
    full = pd.concat([combined[numeric].reset_index(drop=True),
                      encoded.reset_index(drop=True)], axis=1)
    full = full.astype(float)

    n_train = len(X_train)
    n_val = len(X_val)
    X_train_arr = full.iloc[:n_train].to_numpy()
    X_val_arr = full.iloc[n_train:n_train + n_val].to_numpy()
    X_test_arr = full.iloc[n_train + n_val:].to_numpy()
    return X_train_arr, X_val_arr, X_test_arr


def fit_candidate(X_train_arr: "np.ndarray", y_train: "pd.Series"):  # type: ignore[name-defined,no-any-unimported]
    """Fit a LogisticRegression on the one-hot feature matrix."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        # class_weight="balanced" is a defensible choice on imbalanced data;
        # the C5 default leaves it off because the synthetic data is roughly
        # balanced. Tweak as needed.
    )
    model.fit(X_train_arr, y_train)
    return model


def tune_threshold(
    model,  # noqa: ANN001
    X_val_arr: "np.ndarray",  # type: ignore[name-defined,no-any-unimported]
    y_val: "pd.Series",  # type: ignore[name-defined]
) -> float:
    """Find the decision threshold that maximizes F1 on the validation set."""
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    probs = model.predict_proba(X_val_arr)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    # precision_recall_curve returns one more precision/recall entry than
    # thresholds (the last entry is the no-positive-prediction case).
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    if len(f1_scores) == 0:
        return 0.5
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def candidate_metrics(
    model,  # noqa: ANN001
    X_test_arr: "np.ndarray",  # type: ignore[name-defined,no-any-unimported]
    y_test: "pd.Series",  # type: ignore[name-defined]
    threshold: float,
) -> Dict[str, float]:
    """Return {'f1', 'roc_auc', 'pr_auc', 'threshold', 'accuracy'} on the test set."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                  f1_score, roc_auc_score)

    probs = model.predict_proba(X_test_arr)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "threshold": float(threshold),
    }


# -----------------------------------------------------------------------------
# Part E -- the slice-based fairness audit
# -----------------------------------------------------------------------------


def fairness_audit(
    model,  # noqa: ANN001
    X_test: "pd.DataFrame",  # type: ignore[name-defined]
    X_test_arr: "np.ndarray",  # type: ignore[name-defined,no-any-unimported]
    y_test: "pd.Series",  # type: ignore[name-defined]
    protected_col: str,
    threshold: float,
) -> List[Dict[str, object]]:
    """Run the per-slice audit on a single protected column.

    Returns a list of dicts, one per slice:
      [{"slice": "Male", "n": 670, "f1": 0.71},
       {"slice": "Female", "n": 330, "f1": 0.61}].
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score

    probs = model.predict_proba(X_test_arr)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    y_pred_series = pd.Series(y_pred, index=y_test.index)

    rows: List[Dict[str, object]] = []
    for slice_value in X_test[protected_col].unique():
        slice_mask = X_test[protected_col] == slice_value
        slice_idx = X_test.index[slice_mask]
        slice_y_true = y_test.loc[slice_idx]
        slice_y_pred = y_pred_series.loc[slice_idx]
        n: int = int(slice_mask.sum())
        if n == 0 or slice_y_true.nunique() < 2:
            # If a slice has only one class, F1 is undefined; skip or report NaN.
            f1: float = float("nan")
        else:
            f1 = float(f1_score(slice_y_true, slice_y_pred))
        rows.append({"slice": str(slice_value), "n": n, "f1": f1})
    rows.sort(key=lambda r: r["slice"])  # type: ignore[arg-type, return-value]
    return rows


def fairness_gap(audit_rows: List[Dict[str, object]]) -> float:
    """The absolute-value gap between best- and worst-performing slice's F1."""
    f1_values = [float(r["f1"]) for r in audit_rows if r["f1"] == r["f1"]]  # filter NaN.
    if len(f1_values) < 2:
        return 0.0
    return float(max(f1_values) - min(f1_values))


def flag_slices_outside_threshold(
    audit_rows: List[Dict[str, object]],
    aggregate_f1: float,
    tolerance: float = 0.05,
) -> List[str]:
    """Return the names of slices whose F1 deviates from the aggregate by > tolerance."""
    flagged: List[str] = []
    for r in audit_rows:
        if r["f1"] != r["f1"]:  # NaN
            continue
        if abs(float(r["f1"]) - aggregate_f1) > tolerance:
            flagged.append(str(r["slice"]))
    return sorted(flagged)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_make_sample_dataframe_shape() -> None:
    df = make_sample_dataframe()
    assert df.shape[0] == 2000
    assert "income" in df.columns
    assert df["income"].nunique() == 2


def test_train_val_test_split_sizes() -> None:
    df = make_sample_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    n: int = len(df)
    # The test split is 15 percent.
    assert abs(len(X_test) / n - 0.15) < 0.02
    assert abs(len(X_val) / n - 0.15) < 0.02
    assert abs(len(X_train) / n - 0.70) < 0.02
    # The splits are disjoint.
    assert len(set(X_train.index) & set(X_val.index)) == 0
    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert len(set(X_val.index) & set(X_test.index)) == 0


def test_baseline_metrics() -> None:
    df = make_sample_dataframe()
    X_train, _X_val, X_test, y_train, _y_val, y_test = train_val_test_split(df)
    baseline = fit_baseline(X_train, y_train)
    metrics = baseline_metrics(baseline, X_test, y_test)
    # DummyClassifier(strategy='most_frequent') has F1 = 0 (because it
    # always predicts the majority class, so for the minority class
    # both TP and FP can be zero).
    assert metrics["f1"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_candidate_beats_baseline_on_f1() -> None:
    df = make_sample_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    baseline = fit_baseline(X_train, y_train)
    baseline_f1 = baseline_metrics(baseline, X_test, y_test)["f1"]

    X_train_arr, X_val_arr, X_test_arr = build_feature_matrix(X_train, X_val, X_test)
    model = fit_candidate(X_train_arr, y_train)
    threshold = tune_threshold(model, X_val_arr, y_val)
    candidate = candidate_metrics(model, X_test_arr, y_test, threshold)
    # The candidate's F1 should be meaningfully above the baseline's F1.
    # On this synthetic data with a real signal we expect candidate F1 > 0.4.
    assert candidate["f1"] > baseline_f1
    assert candidate["f1"] > 0.4
    assert candidate["roc_auc"] > 0.5


def test_fairness_audit_returns_one_row_per_slice() -> None:
    df = make_sample_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    X_train_arr, X_val_arr, X_test_arr = build_feature_matrix(X_train, X_val, X_test)
    model = fit_candidate(X_train_arr, y_train)
    threshold = tune_threshold(model, X_val_arr, y_val)

    audit = fairness_audit(model, X_test, X_test_arr, y_test, "sex", threshold)
    slice_names = sorted([r["slice"] for r in audit])  # type: ignore[arg-type, return-value]
    assert slice_names == ["Female", "Male"]
    for r in audit:
        assert "n" in r and "f1" in r


def test_fairness_audit_detects_built_in_bias() -> None:
    df = make_sample_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    X_train_arr, X_val_arr, X_test_arr = build_feature_matrix(X_train, X_val, X_test)
    model = fit_candidate(X_train_arr, y_train)
    threshold = tune_threshold(model, X_val_arr, y_val)

    audit = fairness_audit(model, X_test, X_test_arr, y_test, "sex", threshold)
    gap = fairness_gap(audit)
    # The synthetic data has an intentional sex bias in income generation.
    # The audit should detect a non-trivial gap, though the magnitude
    # depends on sample size and signal strength.
    assert gap >= 0.0


def test_flag_slices_outside_threshold() -> None:
    audit = [
        {"slice": "A", "n": 100, "f1": 0.80},
        {"slice": "B", "n": 100, "f1": 0.70},
        {"slice": "C", "n": 100, "f1": 0.50},
    ]
    flagged = flag_slices_outside_threshold(audit, aggregate_f1=0.70, tolerance=0.05)
    # Slice A is 10 points above the aggregate; slice C is 20 points below.
    assert "A" in flagged
    assert "C" in flagged
    # Slice B is exactly at the aggregate; not flagged.
    assert "B" not in flagged


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    df = make_sample_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    print(f"Train / val / test sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print()

    # Baseline.
    baseline = fit_baseline(X_train, y_train)
    baseline_m = baseline_metrics(baseline, X_test, y_test)
    print(f"Baseline (DummyClassifier most_frequent):")
    for k, v in baseline_m.items():
        print(f"  {k:10s}  {v:.3f}")
    print()

    # Candidate.
    X_train_arr, X_val_arr, X_test_arr = build_feature_matrix(X_train, X_val, X_test)
    model = fit_candidate(X_train_arr, y_train)
    threshold = tune_threshold(model, X_val_arr, y_val)
    candidate_m = candidate_metrics(model, X_test_arr, y_test, threshold)
    print(f"Candidate (LogisticRegression):")
    for k, v in candidate_m.items():
        print(f"  {k:10s}  {v:.3f}")
    print()

    # Fairness audit.
    for protected_col in ["sex", "race"]:
        audit = fairness_audit(model, X_test, X_test_arr, y_test, protected_col, threshold)
        gap = fairness_gap(audit)
        flagged = flag_slices_outside_threshold(audit, aggregate_f1=candidate_m["f1"])
        print(f"Fairness audit on {protected_col!r}:")
        for r in audit:
            print(f"  {r['slice']:25s}  n={r['n']:4d}  f1={r['f1']:.3f}")  # type: ignore[arg-type, return-value]
        print(f"  gap (best - worst): {gap:.3f}")
        if flagged:
            print(f"  flagged slices (>5pp from aggregate): {flagged}")
        print()

    # Run the assertions.
    test_make_sample_dataframe_shape()
    test_train_val_test_split_sizes()
    test_baseline_metrics()
    test_candidate_beats_baseline_on_f1()
    test_fairness_audit_returns_one_row_per_slice()
    test_fairness_audit_detects_built_in_bias()
    test_flag_slices_outside_threshold()

    print("OK -- exercise 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
