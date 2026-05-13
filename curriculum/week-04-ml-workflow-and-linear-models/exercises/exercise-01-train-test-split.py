"""
Exercise 1 -- Train / Validation / Test Split.

Goal: split a dataset three ways with train_test_split, run k-fold
cross-validation on the train set, then deliberately introduce a
fit-then-split leak and watch the score lie. The takeaway is that the
score is only as honest as the split.

Estimated time: 45 minutes.

Run with:    python exercise-01-train-test-split.py
Or test:     pytest exercise-01-train-test-split.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-train-test-split.py succeeds.

The exercise builds deterministic synthetic data so the tests never
depend on the network. Do not look at the HINT block at the bottom until
you have tried for fifteen minutes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Deterministic data
# -----------------------------------------------------------------------------

def make_data(n: int = 1000, p: int = 5, noise: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset.

    y is a linear function of X plus Gaussian noise, with one feature on a
    100x scale relative to the others (so scaling matters).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(n, p))
    X[:, 0] *= 100.0  # one feature on a much larger scale
    beta = np.array([0.05, 1.0, -2.0, 0.5, 0.0])  # true coefficients
    y = X @ beta + rng.normal(scale=noise, size=n)
    return X, y


# -----------------------------------------------------------------------------
# Part A -- A three-way split
# -----------------------------------------------------------------------------

def split_three_ways(
    X: np.ndarray, y: np.ndarray, val_size: float = 0.20, test_size: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produce a 60/20/20 train/val/test split using train_test_split twice.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test). All splits
    use random_state=RANDOM_STATE so the split is reproducible.

    Hint: the standard trick is to peel off the test set first, then
    split the remainder into train and val. val_size and test_size are
    fractions of the *original* dataset; you will need to convert.
    """
    # TODO: implement using two calls to train_test_split.
    raise NotImplementedError("Part A -- split_three_ways")


# -----------------------------------------------------------------------------
# Part B -- Baseline, fit, evaluate
# -----------------------------------------------------------------------------

def baseline_rmse(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit a DummyRegressor(strategy='mean') on the train data, predict on
    val, return val RMSE.
    """
    # TODO: implement using DummyRegressor and root_mean_squared_error.
    raise NotImplementedError("Part B -- baseline_rmse")


def linreg_rmse(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Fit a Pipeline of (StandardScaler, LinearRegression) on the train
    data, predict on val, return val RMSE.

    The pipeline is the leak-free preprocessing pattern from Lecture 2.
    """
    # TODO: implement using Pipeline + StandardScaler + LinearRegression.
    raise NotImplementedError("Part B -- linreg_rmse")


# -----------------------------------------------------------------------------
# Part C -- Cross-validation on the training set
# -----------------------------------------------------------------------------

def cv_rmse(X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5) -> Tuple[float, float]:
    """Run K-fold cross-validation on the training set.

    Use KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    and the same StandardScaler -> LinearRegression pipeline as in
    linreg_rmse. Return (mean RMSE, std of RMSE) across folds.

    Hint: cross_val_score returns negative RMSE by sklearn convention --
    flip the sign before computing mean/std.
    """
    # TODO: implement using cross_val_score with
    #       scoring="neg_root_mean_squared_error".
    raise NotImplementedError("Part C -- cv_rmse")


# -----------------------------------------------------------------------------
# Part D -- The leak demonstration
# -----------------------------------------------------------------------------

def leaky_score(X: np.ndarray, y: np.ndarray) -> float:
    """Deliberately leak: fit StandardScaler on the ENTIRE X, then split,
    then fit LinearRegression and score on val. Return val RMSE.

    This is the canonical fit-then-split bug. The score will look fine on
    this dataset (the leak is small because the data is i.i.d. and
    plentiful), but on more structured data it lies dramatically.
    """
    # TODO: implement the leaky pattern. Mirror split_three_ways' fractions
    # but pre-scale X first.
    raise NotImplementedError("Part D -- leaky_score")


def honest_score(X: np.ndarray, y: np.ndarray) -> float:
    """The honest pattern: split first, fit the scaler on train only
    (via a Pipeline), score on val. Return val RMSE.
    """
    X_train, X_val, _, y_train, y_val, _ = split_three_ways(X, y)
    return linreg_rmse(X_train, y_train, X_val, y_val)


# -----------------------------------------------------------------------------
# Glue: one function that runs the whole pipeline.
# -----------------------------------------------------------------------------

def run_pipeline() -> dict[str, float]:
    """End-to-end. Returns a dict of all reported numbers."""
    X, y = make_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_three_ways(X, y)
    base = baseline_rmse(X_train, y_train, X_val, y_val)
    model = linreg_rmse(X_train, y_train, X_val, y_val)
    cv_mean, cv_std = cv_rmse(X_train, y_train)
    leak = leaky_score(X, y)
    honest = honest_score(X, y)
    test = linreg_rmse(X_train, y_train, X_test, y_test)
    return {
        "baseline_val_rmse": base,
        "model_val_rmse":    model,
        "cv_mean_rmse":      cv_mean,
        "cv_std_rmse":       cv_std,
        "leaky_val_rmse":    leak,
        "honest_val_rmse":   honest,
        "test_rmse":         test,
    }


# -----------------------------------------------------------------------------
# Pytest-style checks (also run when the file is executed directly).
# -----------------------------------------------------------------------------

def test_split_shapes() -> None:
    X, y = make_data(n=1000)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_three_ways(X, y)
    n = len(X)
    # Allow for rounding wobble; the total must equal n.
    assert len(X_tr) + len(X_va) + len(X_te) == n
    assert abs(len(X_tr) - 600) <= 2
    assert abs(len(X_va) - 200) <= 2
    assert abs(len(X_te) - 200) <= 2
    assert len(X_tr) == len(y_tr)
    assert len(X_va) == len(y_va)
    assert len(X_te) == len(y_te)


def test_baseline_is_above_model() -> None:
    X, y = make_data()
    X_tr, X_va, _, y_tr, y_va, _ = split_three_ways(X, y)
    base = baseline_rmse(X_tr, y_tr, X_va, y_va)
    model = linreg_rmse(X_tr, y_tr, X_va, y_va)
    assert base > model, (
        f"baseline RMSE ({base:.3f}) should be larger than model RMSE ({model:.3f}); "
        f"the linear model should beat the mean predictor on this data."
    )


def test_cv_returns_two_numbers() -> None:
    X, y = make_data()
    X_tr, _, _, y_tr, _, _ = split_three_ways(X, y)
    mean, std = cv_rmse(X_tr, y_tr, n_splits=5)
    assert mean > 0
    assert std >= 0
    # On this clean synthetic data the model's CV RMSE should be close
    # to the noise level (0.5 by construction).
    assert mean < 2.0, f"CV RMSE seems too high ({mean:.3f})"


def test_pipeline_runs_end_to_end() -> None:
    out = run_pipeline()
    for key in (
        "baseline_val_rmse", "model_val_rmse", "cv_mean_rmse",
        "cv_std_rmse", "leaky_val_rmse", "honest_val_rmse", "test_rmse",
    ):
        assert key in out
        assert np.isfinite(out[key])


def test_honest_and_leaky_both_finite() -> None:
    """On this synthetic i.i.d. dataset the leak is small, but the two
    procedures must both run and return finite numbers. The point of
    the demonstration is the *pattern*, not a dramatic gap on toy data.
    """
    X, y = make_data()
    a = leaky_score(X, y)
    b = honest_score(X, y)
    assert np.isfinite(a) and np.isfinite(b)


def _run_all_tests() -> None:
    test_split_shapes()
    test_baseline_is_above_model()
    test_cv_returns_two_numbers()
    test_pipeline_runs_end_to_end()
    test_honest_and_leaky_both_finite()
    print("OK -- exercise 1")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# split_three_ways:
#     # Peel off the test set first.
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=RANDOM_STATE,
#     )
#     # Then split the remainder into train and val.
#     # val_size is a fraction of the original n; convert relative to len(X_temp).
#     val_fraction_of_remainder = val_size / (1.0 - test_size)
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=val_fraction_of_remainder,
#         random_state=RANDOM_STATE,
#     )
#     return X_train, X_val, X_test, y_train, y_val, y_test
#
# baseline_rmse:
#     base = DummyRegressor(strategy="mean").fit(X_train, y_train)
#     return root_mean_squared_error(y_val, base.predict(X_val))
#
# linreg_rmse:
#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("linreg", LinearRegression()),
#     ]).fit(X_train, y_train)
#     return root_mean_squared_error(y_val, pipe.predict(X_val))
#
# cv_rmse:
#     pipe = Pipeline([
#         ("scaler", StandardScaler()),
#         ("linreg", LinearRegression()),
#     ])
#     cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
#     scores = cross_val_score(
#         pipe, X_train, y_train, cv=cv,
#         scoring="neg_root_mean_squared_error",
#     )
#     return float(-scores.mean()), float(scores.std())
#
# leaky_score:
#     # The bug: fit the scaler on ALL of X before splitting.
#     scaler = StandardScaler().fit(X)
#     X_scaled = scaler.transform(X)
#     X_tr, X_va, _, y_tr, y_va, _ = split_three_ways(X_scaled, y)
#     mdl = LinearRegression().fit(X_tr, y_tr)
#     return root_mean_squared_error(y_va, mdl.predict(X_va))
#
# -----------------------------------------------------------------------------
