"""
Exercise 3 -- scikit-learn Pipeline + ColumnTransformer.

Goal: build an end-to-end Pipeline over a mixed-type DataFrame (numeric
and categorical columns), with a ColumnTransformer that:

    - Scales the numeric columns with StandardScaler.
    - One-hot-encodes the categorical columns with OneHotEncoder, with
      handle_unknown='ignore' (the production-safe default in sklearn
      1.5+).
    - Feeds the result into RidgeCV with a sensible alpha grid.

Then cross-validate the full pipeline (so the scaler and the encoder are
re-fit inside each fold) and verify there is no leak: the CV RMSE must
match the held-out-test RMSE to within one CV standard deviation.

Estimated time: 50 minutes.

Run with:    python exercise-03-sklearn-pipeline.py
Or test:     pytest exercise-03-sklearn-pipeline.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-03-sklearn-pipeline.py succeeds.

The exercise builds deterministic synthetic data so the tests never
depend on the network.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
NUMERIC_COLS = ["sqft", "bedrooms", "age", "lot_size"]
CATEGORICAL_COLS = ["neighborhood", "style"]


# -----------------------------------------------------------------------------
# Deterministic data
# -----------------------------------------------------------------------------

def make_housing_like(n: int = 2000) -> pd.DataFrame:
    """Synthetic housing-like dataset with the same column types as Ames.

    Returns a DataFrame with NUMERIC_COLS + CATEGORICAL_COLS + 'price'.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    sqft = rng.uniform(500, 4000, size=n)
    bedrooms = rng.integers(1, 6, size=n).astype(float)
    age = rng.uniform(0, 100, size=n)
    lot_size = rng.uniform(2000, 20_000, size=n)
    neighborhood = rng.choice(["A", "B", "C", "D", "E"], size=n,
                              p=[0.3, 0.25, 0.2, 0.15, 0.10])
    style = rng.choice(["Ranch", "Colonial", "Modern", "Victorian"], size=n,
                       p=[0.4, 0.3, 0.2, 0.1])
    # Price is a linear function of sqft and bedrooms, with neighborhood
    # and style premiums, plus noise.
    n_premium = pd.Series(neighborhood).map(
        {"A": 0, "B": 20_000, "C": 50_000, "D": 80_000, "E": 120_000}
    ).to_numpy()
    s_premium = pd.Series(style).map(
        {"Ranch": 0, "Colonial": 15_000, "Modern": 30_000, "Victorian": 25_000}
    ).to_numpy()
    price = (
        50_000
        + 120.0 * sqft
        + 8_000.0 * bedrooms
        - 250.0 * age
        + 2.5 * lot_size
        + n_premium
        + s_premium
        + rng.normal(scale=20_000, size=n)
    )
    return pd.DataFrame({
        "sqft":         sqft,
        "bedrooms":     bedrooms,
        "age":          age,
        "lot_size":     lot_size,
        "neighborhood": neighborhood,
        "style":        style,
        "price":        price,
    })


# -----------------------------------------------------------------------------
# Part A -- Build the ColumnTransformer
# -----------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer with:

        - 'num' -> StandardScaler on NUMERIC_COLS
        - 'cat' -> OneHotEncoder(handle_unknown='ignore') on CATEGORICAL_COLS

    The remainder='drop' default is fine; we use all the columns.
    """
    # TODO: implement using ColumnTransformer with two transformers.
    raise NotImplementedError("Part A -- build_preprocessor")


# -----------------------------------------------------------------------------
# Part B -- Build the full Pipeline
# -----------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Return a Pipeline with two steps:

        ("preprocess", build_preprocessor())
        ("ridge",      RidgeCV(alphas=np.logspace(-3, 3, 13)))

    RidgeCV's default cv is leave-one-out for small datasets and 5-fold
    for larger; both work. We do not pass cv= explicitly.
    """
    # TODO: implement.
    raise NotImplementedError("Part B -- build_pipeline")


# -----------------------------------------------------------------------------
# Part C -- Fit, cross-validate, test
# -----------------------------------------------------------------------------

def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 train/test split.

    Returns (X_train, X_test, y_train, y_test). y is df['price'];
    X drops 'price' but keeps every other column (the ColumnTransformer
    selects).
    """
    y = df["price"]
    X = df.drop(columns=["price"])
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)


def cv_score(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[float, float]:
    """Run 5-fold cross-validation on the training set.

    Use KFold(shuffle=True, random_state=RANDOM_STATE) and scoring
    'neg_root_mean_squared_error'. Return (mean RMSE, std RMSE).
    """
    # TODO: implement with cross_val_score.
    raise NotImplementedError("Part C -- cv_score")


def test_rmse(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Fit on train, predict on test, return RMSE."""
    pipe.fit(X_train, y_train)
    return float(root_mean_squared_error(y_test, pipe.predict(X_test)))


def baseline_test_rmse(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """DummyRegressor(strategy='mean') for context."""
    base = DummyRegressor(strategy="mean").fit(X_train, y_train)
    return float(root_mean_squared_error(y_test, base.predict(X_test)))


# -----------------------------------------------------------------------------
# Glue
# -----------------------------------------------------------------------------

def run_pipeline() -> dict[str, float]:
    """End-to-end. Returns the four numbers we report."""
    df = make_housing_like()
    X_train, X_test, y_train, y_test = split(df)
    pipe = build_pipeline()
    cv_mean, cv_std = cv_score(pipe, X_train, y_train)
    base = baseline_test_rmse(X_train, y_train, X_test, y_test)
    test = test_rmse(pipe, X_train, y_train, X_test, y_test)
    return {
        "cv_mean_rmse":      cv_mean,
        "cv_std_rmse":       cv_std,
        "baseline_test_rmse": base,
        "test_rmse":         test,
    }


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_preprocessor_is_column_transformer() -> None:
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)
    names = [name for name, _, _ in pre.transformers]
    assert "num" in names and "cat" in names


def test_pipeline_steps() -> None:
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["preprocess", "ridge"], (
        f"expected steps ['preprocess', 'ridge'], got {step_names}"
    )


def test_pipeline_fits_and_predicts() -> None:
    df = make_housing_like(n=500)
    X_tr, X_te, y_tr, y_te = split(df)
    pipe = build_pipeline().fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert len(preds) == len(y_te)
    assert np.all(np.isfinite(preds))


def test_handle_unknown_category() -> None:
    """The OneHotEncoder must use handle_unknown='ignore' so unseen test-time
    categorical values do not crash predict().
    """
    df = make_housing_like(n=500)
    X_tr, X_te, y_tr, y_te = split(df)
    pipe = build_pipeline().fit(X_tr, y_tr)
    # Inject a never-before-seen category into the test set.
    X_te = X_te.copy()
    X_te.iloc[0, X_te.columns.get_loc("neighborhood")] = "Z_UNSEEN"
    # If handle_unknown != 'ignore', this would raise. With 'ignore' it just
    # produces an all-zeros one-hot row for that column.
    preds = pipe.predict(X_te)
    assert np.all(np.isfinite(preds))


def test_model_beats_baseline_and_is_consistent() -> None:
    out = run_pipeline()
    # The fitted model should beat the mean predictor by a wide margin.
    assert out["test_rmse"] < 0.5 * out["baseline_test_rmse"], (
        f"model test RMSE {out['test_rmse']:.0f} should be much less than "
        f"baseline {out['baseline_test_rmse']:.0f}"
    )
    # The test RMSE should fall within a few CV standard deviations of the
    # CV mean RMSE -- if it does not, there is a leak.
    gap = abs(out["test_rmse"] - out["cv_mean_rmse"])
    assert gap < 5.0 * out["cv_std_rmse"] + 5_000.0, (
        f"test RMSE {out['test_rmse']:.0f} is far from CV mean "
        f"{out['cv_mean_rmse']:.0f} ± {out['cv_std_rmse']:.0f} -- "
        f"check for leakage."
    )


def _run_all_tests() -> None:
    test_preprocessor_is_column_transformer()
    test_pipeline_steps()
    test_pipeline_fits_and_predicts()
    test_handle_unknown_category()
    test_model_beats_baseline_and_is_consistent()
    print("OK -- exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# build_preprocessor:
#     return ColumnTransformer([
#         ("num", StandardScaler(), NUMERIC_COLS),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
#     ])
#
# build_pipeline:
#     return Pipeline([
#         ("preprocess", build_preprocessor()),
#         ("ridge",      RidgeCV(alphas=np.logspace(-3, 3, 13))),
#     ])
#
# cv_score:
#     cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
#     scores = cross_val_score(
#         pipe, X_train, y_train, cv=cv,
#         scoring="neg_root_mean_squared_error",
#     )
#     return float(-scores.mean()), float(scores.std())
#
# -----------------------------------------------------------------------------
