"""
Exercise 2 -- Random Forest with sklearn.

Goal: build a defended RandomForestRegressor pipeline on a synthetic
housing-like dataset and exercise the four behaviors that matter:

    (1) The out-of-bag (OOB) estimate as free cross-validation.
    (2) The 'max_features' knob as the de-correlation hyperparameter.
    (3) Feature importance, split-gain version vs. permutation version.
    (4) The Breiman recipe: sweep max_features for the regression default,
        rather than accepting sklearn's max_features=1.0.

The exercise compares to a single DecisionTreeRegressor and to a
DummyRegressor baseline; the forest must beat both.

Estimated time: 60 minutes.

Run with:    python exercise-02-random-forest-sklearn.py
Or test:     pytest exercise-02-random-forest-sklearn.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-random-forest-sklearn.py succeeds.

The exercise builds deterministic synthetic data so the tests never
depend on the network.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Deterministic synthetic data
# -----------------------------------------------------------------------------

def make_housing_like(n: int = 2000) -> pd.DataFrame:
    """Synthetic housing-like dataset.

    Six features: 4 numeric, 2 nominal categorical. The target is a
    deterministic function with interactions plus Gaussian noise.

    The categorical features are passed as one-hot via a small helper
    so we can keep this exercise focused on the forest, not on
    ColumnTransformer plumbing (that was Week 4 Exercise 3).
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
    n_premium = pd.Series(neighborhood).map(
        {"A": 0, "B": 20_000, "C": 50_000, "D": 80_000, "E": 120_000}
    ).to_numpy()
    s_premium = pd.Series(style).map(
        {"Ranch": 0, "Colonial": 15_000, "Modern": 30_000, "Victorian": 25_000}
    ).to_numpy()
    # An interaction: large old houses lose value faster than small old houses.
    interaction = -3.0 * age * (sqft - 2000) / 1000.0
    price = (
        50_000
        + 120.0 * sqft
        + 8_000.0 * bedrooms
        - 250.0 * age
        + 2.5 * lot_size
        + n_premium
        + s_premium
        + interaction
        + rng.normal(scale=15_000, size=n)
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


def to_numeric_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    """One-hot encode the two categorical columns; return (X, column_names)."""
    X = pd.get_dummies(df.drop(columns=["price"]),
                       columns=["neighborhood", "style"], drop_first=False)
    return X.to_numpy(dtype=float), list(X.columns)


# -----------------------------------------------------------------------------
# Part A -- The Breiman forest
# -----------------------------------------------------------------------------

def build_forest(n_estimators: int = 500,
                 max_features: float | str = "sqrt",
                 min_samples_leaf: int = 5) -> RandomForestRegressor:
    """Build a RandomForestRegressor with Breiman's regression default.

    Set:
        - n_estimators        = n_estimators
        - max_features        = max_features  (default 'sqrt'; sklearn's regression
                                               default is 1.0 which is *not* Breiman)
        - min_samples_leaf    = min_samples_leaf
        - oob_score           = True          (free generalization estimate)
        - n_jobs              = -1            (trees are independent; use the cores)
        - random_state        = RANDOM_STATE  (reproducibility)
    """
    # TODO: implement.
    raise NotImplementedError("Part A -- build_forest")


# -----------------------------------------------------------------------------
# Part B -- Sweep max_features
# -----------------------------------------------------------------------------

def sweep_max_features(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Fit a forest at each max_features in a small grid and record OOB R^2.

    Use n_estimators=200 (smaller, for speed) and min_samples_leaf=5.

    The grid (with p = X.shape[1]):
        - "sqrt"  -> floor(sqrt(p))
        - "log2"  -> floor(log2(p))
        - 0.33    -> floor(p / 3)
        - 0.5     -> floor(p / 2)
        - 1.0     -> p     (the sklearn default; not Breiman)

    Return a DataFrame with columns ['max_features', 'oob_r2'].
    """
    # TODO: implement.
    raise NotImplementedError("Part B -- sweep_max_features")


# -----------------------------------------------------------------------------
# Part C -- Compare to a single tree
# -----------------------------------------------------------------------------

def compare_forest_to_tree(X_tr: np.ndarray, y_tr: np.ndarray,
                           X_te: np.ndarray, y_te: np.ndarray) -> dict[str, float]:
    """Fit a single DecisionTreeRegressor and a RandomForestRegressor on the
    same data; return both test RMSEs and the baseline mean-predictor RMSE.

    Use max_depth=None and min_samples_leaf=5 for both, with random_state.
    """
    tree = DecisionTreeRegressor(max_depth=None, min_samples_leaf=5,
                                 random_state=RANDOM_STATE).fit(X_tr, y_tr)
    forest = build_forest(n_estimators=200, max_features="sqrt", min_samples_leaf=5).fit(X_tr, y_tr)
    dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    return {
        "tree_rmse":   float(root_mean_squared_error(y_te, tree.predict(X_te))),
        "forest_rmse": float(root_mean_squared_error(y_te, forest.predict(X_te))),
        "dummy_rmse":  float(root_mean_squared_error(y_te, dummy.predict(X_te))),
    }


# -----------------------------------------------------------------------------
# Part D -- Feature importance (two ways)
# -----------------------------------------------------------------------------

def importances_two_ways(forest: RandomForestRegressor,
                         X_te: np.ndarray, y_te: np.ndarray,
                         column_names: list[str]) -> pd.DataFrame:
    """Compute Gini (split-gain) importance and permutation importance.

    Permutation importance uses n_repeats=10 and random_state=RANDOM_STATE.

    Return a DataFrame indexed by feature name with columns
    ['gini_importance', 'permutation_importance'], sorted by
    permutation_importance descending.
    """
    # TODO: implement using forest.feature_importances_ and
    # sklearn.inspection.permutation_importance.
    raise NotImplementedError("Part D -- importances_two_ways")


# -----------------------------------------------------------------------------
# Glue
# -----------------------------------------------------------------------------

def run_pipeline() -> dict[str, object]:
    df = make_housing_like()
    X, column_names = to_numeric_matrix(df)
    y = df["price"].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    # Sweep max_features.
    sweep = sweep_max_features(X_tr, y_tr)

    # Forest at the recommended setting.
    forest = build_forest(n_estimators=200, max_features="sqrt", min_samples_leaf=5).fit(X_tr, y_tr)
    forest_test_rmse = float(root_mean_squared_error(y_te, forest.predict(X_te)))

    # Tree-vs-forest comparison.
    compare = compare_forest_to_tree(X_tr, y_tr, X_te, y_te)

    # Two-way importances.
    imp = importances_two_ways(forest, X_te, y_te, column_names)

    return {
        "sweep":            sweep,
        "forest_test_rmse": forest_test_rmse,
        "forest_oob_r2":    float(forest.oob_score_),
        "compare":          compare,
        "importances":      imp,
    }


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_forest_has_breiman_defaults() -> None:
    rf = build_forest()
    assert rf.oob_score is True, "oob_score must be True for free CV"
    assert rf.n_jobs == -1, "n_jobs should be -1 -- trees are independent"
    assert rf.random_state == RANDOM_STATE
    assert rf.max_features == "sqrt"


def test_sweep_records_all_settings() -> None:
    df = make_housing_like(n=500)
    X, _ = to_numeric_matrix(df)
    y = df["price"].to_numpy(dtype=float)
    out = sweep_max_features(X, y)
    assert set(out.columns) == {"max_features", "oob_r2"}
    assert len(out) == 5
    assert (out["oob_r2"] > 0.0).all(), "every forest should beat the mean predictor"


def test_forest_beats_single_tree() -> None:
    df = make_housing_like()
    X, _ = to_numeric_matrix(df)
    y = df["price"].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    out = compare_forest_to_tree(X_tr, y_tr, X_te, y_te)
    assert out["forest_rmse"] < out["tree_rmse"], (
        f"forest RMSE {out['forest_rmse']:.0f} did not beat single tree {out['tree_rmse']:.0f}"
    )


def test_forest_beats_dummy() -> None:
    df = make_housing_like()
    X, _ = to_numeric_matrix(df)
    y = df["price"].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    out = compare_forest_to_tree(X_tr, y_tr, X_te, y_te)
    assert out["forest_rmse"] < 0.5 * out["dummy_rmse"], (
        f"forest {out['forest_rmse']:.0f} should be much less than mean predictor "
        f"{out['dummy_rmse']:.0f}"
    )


def test_importances_both_computed() -> None:
    df = make_housing_like(n=800)
    X, column_names = to_numeric_matrix(df)
    y = df["price"].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    forest = build_forest(n_estimators=150, max_features="sqrt", min_samples_leaf=5).fit(X_tr, y_tr)
    imp = importances_two_ways(forest, X_te, y_te, column_names)
    assert set(imp.columns) == {"gini_importance", "permutation_importance"}
    assert len(imp) == X.shape[1]
    # Sqft should be in the top 3 by permutation importance -- it is the
    # largest coefficient in the data-generating process.
    top3 = imp.head(3).index.tolist()
    assert "sqft" in top3, f"sqft should be in the top 3 permutation-importance features, got {top3}"


def test_oob_r2_is_close_to_held_out_r2() -> None:
    df = make_housing_like()
    X, _ = to_numeric_matrix(df)
    y = df["price"].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    forest = build_forest(n_estimators=200, max_features="sqrt", min_samples_leaf=5).fit(X_tr, y_tr)
    held_out_r2 = forest.score(X_te, y_te)
    assert abs(forest.oob_score_ - held_out_r2) < 0.1, (
        f"OOB R^2 {forest.oob_score_:.3f} should track held-out R^2 {held_out_r2:.3f} "
        f"within ~0.1 on this dataset"
    )


def _run_all_tests() -> None:
    test_forest_has_breiman_defaults()
    test_sweep_records_all_settings()
    test_forest_beats_single_tree()
    test_forest_beats_dummy()
    test_importances_both_computed()
    test_oob_r2_is_close_to_held_out_r2()
    print("OK -- exercise 2")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# build_forest:
#     return RandomForestRegressor(
#         n_estimators=n_estimators,
#         max_features=max_features,
#         min_samples_leaf=min_samples_leaf,
#         oob_score=True,
#         n_jobs=-1,
#         random_state=RANDOM_STATE,
#     )
#
# sweep_max_features:
#     p = X.shape[1]
#     settings = ["sqrt", "log2", 0.33, 0.5, 1.0]
#     rows = []
#     for mf in settings:
#         rf = RandomForestRegressor(
#             n_estimators=200,
#             max_features=mf,
#             min_samples_leaf=5,
#             oob_score=True,
#             n_jobs=-1,
#             random_state=RANDOM_STATE,
#         ).fit(X, y)
#         rows.append({"max_features": str(mf), "oob_r2": float(rf.oob_score_)})
#     return pd.DataFrame(rows)
#
# importances_two_ways:
#     gini = pd.Series(forest.feature_importances_, index=column_names, name="gini_importance")
#     perm = permutation_importance(forest, X_te, y_te, n_repeats=10,
#                                   random_state=RANDOM_STATE, n_jobs=-1)
#     perm_s = pd.Series(perm.importances_mean, index=column_names, name="permutation_importance")
#     out = pd.concat([gini, perm_s], axis=1)
#     return out.sort_values("permutation_importance", ascending=False)
#
# -----------------------------------------------------------------------------
