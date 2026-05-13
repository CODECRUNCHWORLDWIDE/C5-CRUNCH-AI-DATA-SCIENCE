"""
Exercise 3 -- XGBoost vs LightGBM (matched hyperparameters).

Goal: fit a gradient-boosted-trees model with three implementations on the
same data and compare them on three axes:

    (1) Test RMSE (they should agree to within ~3% on this dataset).
    (2) Wall-clock training time (they will differ by 2-5x).
    (3) Number of boosting rounds chosen by early stopping (will differ
        modestly because the libraries use slightly different criteria).

The three implementations:
    - sklearn.ensemble.HistGradientBoostingRegressor
    - xgboost.XGBRegressor
    - lightgbm.LGBMRegressor

If XGBoost or LightGBM is not installed, the corresponding tests are
marked as skipped (importorskip) so the file still compiles and runs
the sklearn-only path.

Estimated time: 60 minutes.

Run with:    python exercise-03-xgboost-vs-lightgbm.py
Or test:     pytest exercise-03-xgboost-vs-lightgbm.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass (or skip if XGBoost / LightGBM not installed).
- python -m py_compile exercise-03-xgboost-vs-lightgbm.py succeeds.

The exercise uses sklearn's fetch_california_housing dataset (network on
the first run, cached after that).
"""

from __future__ import annotations

import importlib.util
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
LEARNING_RATE = 0.05
MAX_DEPTH = 6
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS_BUDGET = 2000


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load California Housing as (X, y).

    California Housing is 20,640 rows of 8 numeric features. The target
    is median house value in units of $100,000. No categoricals; no
    missing values. The clean tabular regression benchmark.
    """
    bunch = fetch_california_housing(as_frame=True)
    return bunch.data, bunch.target


def split(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split 70 / 15 / 15 train / val / test.

    The validation set is used for early stopping (XGBoost / LightGBM)
    and HistGradientBoostingRegressor's internal early stopping; the
    test set is scored once at the end.
    """
    X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trv, y_trv, test_size=0.18, random_state=RANDOM_STATE,  # 0.18 of 0.85 ~ 0.153
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# -----------------------------------------------------------------------------
# Part A -- HistGradientBoostingRegressor (sklearn, always available)
# -----------------------------------------------------------------------------

def fit_histgbr(X_tr: pd.DataFrame, X_val: pd.DataFrame,
                y_tr: pd.Series, y_val: pd.Series) -> Tuple[HistGradientBoostingRegressor, float, int]:
    """Fit HistGradientBoostingRegressor with early stopping.

    Set:
        learning_rate            = LEARNING_RATE
        max_iter                 = N_ESTIMATORS_BUDGET   (an overestimate)
        max_depth                = MAX_DEPTH
        early_stopping           = True
        n_iter_no_change         = EARLY_STOPPING_ROUNDS
        validation_fraction      = None                   (we pass the val set explicitly
                                                            -- but HistGBR does not accept an
                                                            external val set; in practice the
                                                            sklearn API does the split itself.
                                                            We pass the entire training set
                                                            and let sklearn carve off a slice.)
        random_state             = RANDOM_STATE

    Return (model, wall_clock_seconds, n_iter_).
    n_iter_ is the number of boosting rounds actually used (after early stop).
    """
    # TODO: implement; fit on the union of X_tr and X_val (HistGBR makes
    # its own validation split via validation_fraction=0.1 by default).
    # The test will check that .n_iter_ < N_ESTIMATORS_BUDGET (early stop
    # actually fired).
    raise NotImplementedError("Part A -- fit_histgbr")


# -----------------------------------------------------------------------------
# Part B -- XGBoost (optional)
# -----------------------------------------------------------------------------

def fit_xgboost(X_tr: pd.DataFrame, X_val: pd.DataFrame,
                y_tr: pd.Series, y_val: pd.Series) -> Optional[Tuple[object, float, int]]:
    """Fit xgboost.XGBRegressor with matched hyperparameters and early stopping.

    Set:
        learning_rate          = LEARNING_RATE
        n_estimators           = N_ESTIMATORS_BUDGET
        max_depth              = MAX_DEPTH
        early_stopping_rounds  = EARLY_STOPPING_ROUNDS
        random_state           = RANDOM_STATE
        eval_set               = [(X_val, y_val)]
        verbosity              = 0

    Return (model, wall_clock_seconds, best_iteration) or None if xgboost
    is not installed.
    """
    if importlib.util.find_spec("xgboost") is None:
        return None
    from xgboost import XGBRegressor                         # noqa: F401 -- local import
    # TODO: implement.
    raise NotImplementedError("Part B -- fit_xgboost")


# -----------------------------------------------------------------------------
# Part C -- LightGBM (optional)
# -----------------------------------------------------------------------------

def fit_lightgbm(X_tr: pd.DataFrame, X_val: pd.DataFrame,
                 y_tr: pd.Series, y_val: pd.Series) -> Optional[Tuple[object, float, int]]:
    """Fit lightgbm.LGBMRegressor with matched hyperparameters and early stopping.

    Set:
        learning_rate          = LEARNING_RATE
        n_estimators           = N_ESTIMATORS_BUDGET
        num_leaves             = 2 ** MAX_DEPTH - 1  (the leaf count for a depth-6 tree
                                                       is 63; LightGBM uses num_leaves
                                                       as the natural depth knob)
        random_state           = RANDOM_STATE
        eval_set               = [(X_val, y_val)]
        callbacks              = [lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS)]

    Return (model, wall_clock_seconds, best_iteration_) or None if lightgbm
    is not installed.
    """
    if importlib.util.find_spec("lightgbm") is None:
        return None
    import lightgbm as lgb                                   # noqa: F401 -- local import
    # TODO: implement; the LGBM early-stopping API is a callback in
    # lightgbm 4.0+.
    raise NotImplementedError("Part C -- fit_lightgbm")


# -----------------------------------------------------------------------------
# Glue -- evaluate all available implementations
# -----------------------------------------------------------------------------

def evaluate(model: object, X_te: pd.DataFrame, y_te: pd.Series) -> float:
    """Return test RMSE for a fitted model."""
    return float(root_mean_squared_error(y_te, model.predict(X_te)))


def run_comparison() -> pd.DataFrame:
    """Fit all available implementations and return a comparison DataFrame."""
    X, y = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)

    rows = []

    # sklearn HistGBR -- always available.
    mdl_hgbr, wc_hgbr, n_iter_hgbr = fit_histgbr(X_tr, X_val, y_tr, y_val)
    rows.append({
        "impl":         "HistGradientBoostingRegressor",
        "test_rmse":    evaluate(mdl_hgbr, X_te, y_te),
        "wall_clock_s": wc_hgbr,
        "n_iter":       int(n_iter_hgbr),
    })

    # XGBoost -- skip if not installed.
    out_xgb = fit_xgboost(X_tr, X_val, y_tr, y_val)
    if out_xgb is not None:
        mdl_xgb, wc_xgb, n_iter_xgb = out_xgb
        rows.append({
            "impl":         "XGBRegressor",
            "test_rmse":    evaluate(mdl_xgb, X_te, y_te),
            "wall_clock_s": wc_xgb,
            "n_iter":       int(n_iter_xgb),
        })

    # LightGBM -- skip if not installed.
    out_lgb = fit_lightgbm(X_tr, X_val, y_tr, y_val)
    if out_lgb is not None:
        mdl_lgb, wc_lgb, n_iter_lgb = out_lgb
        rows.append({
            "impl":         "LGBMRegressor",
            "test_rmse":    evaluate(mdl_lgb, X_te, y_te),
            "wall_clock_s": wc_lgb,
            "n_iter":       int(n_iter_lgb),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def test_histgbr_fits_and_early_stops() -> None:
    X, y = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    mdl, wc, n_iter = fit_histgbr(X_tr, X_val, y_tr, y_val)
    rmse = evaluate(mdl, X_te, y_te)
    assert rmse < 0.65, f"HistGBR RMSE {rmse:.3f} should be well under 0.65 on Cal-Housing"
    assert wc > 0.0
    assert n_iter < N_ESTIMATORS_BUDGET, (
        f"early stopping should have fired before {N_ESTIMATORS_BUDGET} rounds; "
        f"actual n_iter = {n_iter}"
    )


def test_xgboost_fits_if_installed() -> None:
    if not _have("xgboost"):
        return  # skip
    X, y = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    out = fit_xgboost(X_tr, X_val, y_tr, y_val)
    assert out is not None
    mdl, wc, best = out
    rmse = evaluate(mdl, X_te, y_te)
    assert rmse < 0.65, f"XGBoost RMSE {rmse:.3f} should be well under 0.65 on Cal-Housing"
    assert wc > 0.0
    assert best < N_ESTIMATORS_BUDGET


def test_lightgbm_fits_if_installed() -> None:
    if not _have("lightgbm"):
        return  # skip
    X, y = load_data()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    out = fit_lightgbm(X_tr, X_val, y_tr, y_val)
    assert out is not None
    mdl, wc, best = out
    rmse = evaluate(mdl, X_te, y_te)
    assert rmse < 0.65, f"LightGBM RMSE {rmse:.3f} should be well under 0.65 on Cal-Housing"
    assert wc > 0.0
    assert best < N_ESTIMATORS_BUDGET


def test_implementations_agree_within_5_percent() -> None:
    """Where multiple implementations are installed, their RMSEs should agree
    to within ~5%. If they don't, hyperparameters are not matched.
    """
    df = run_comparison()
    if len(df) < 2:
        return  # only one impl available; nothing to compare
    rmses = df["test_rmse"].to_numpy()
    ratio = rmses.max() / rmses.min()
    assert ratio < 1.05, (
        f"implementations disagree by {(ratio - 1) * 100:.1f}%; expected <5% with matched hp.\n"
        f"{df}"
    )


def _run_all_tests() -> None:
    test_histgbr_fits_and_early_stops()
    test_xgboost_fits_if_installed()
    test_lightgbm_fits_if_installed()
    test_implementations_agree_within_5_percent()
    print("OK -- exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# fit_histgbr:
#     X_full = pd.concat([X_tr, X_val], axis=0)
#     y_full = pd.concat([y_tr, y_val], axis=0)
#     mdl = HistGradientBoostingRegressor(
#         learning_rate=LEARNING_RATE,
#         max_iter=N_ESTIMATORS_BUDGET,
#         max_depth=MAX_DEPTH,
#         early_stopping=True,
#         n_iter_no_change=EARLY_STOPPING_ROUNDS,
#         random_state=RANDOM_STATE,
#     )
#     t0 = time.perf_counter()
#     mdl.fit(X_full, y_full)
#     wc = time.perf_counter() - t0
#     return mdl, float(wc), int(mdl.n_iter_)
#
# fit_xgboost:
#     from xgboost import XGBRegressor
#     mdl = XGBRegressor(
#         learning_rate=LEARNING_RATE,
#         n_estimators=N_ESTIMATORS_BUDGET,
#         max_depth=MAX_DEPTH,
#         early_stopping_rounds=EARLY_STOPPING_ROUNDS,
#         random_state=RANDOM_STATE,
#         verbosity=0,
#         tree_method="hist",
#     )
#     t0 = time.perf_counter()
#     mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
#     wc = time.perf_counter() - t0
#     return mdl, float(wc), int(mdl.best_iteration)
#
# fit_lightgbm:
#     import lightgbm as lgb
#     mdl = lgb.LGBMRegressor(
#         learning_rate=LEARNING_RATE,
#         n_estimators=N_ESTIMATORS_BUDGET,
#         num_leaves=2 ** MAX_DEPTH - 1,
#         random_state=RANDOM_STATE,
#         verbose=-1,
#     )
#     t0 = time.perf_counter()
#     mdl.fit(
#         X_tr, y_tr,
#         eval_set=[(X_val, y_val)],
#         callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
#     )
#     wc = time.perf_counter() - t0
#     return mdl, float(wc), int(mdl.best_iteration_)
#
# -----------------------------------------------------------------------------
