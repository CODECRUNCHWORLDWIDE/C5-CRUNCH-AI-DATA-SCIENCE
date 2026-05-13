"""
Exercise 2 -- Linear Regression by Hand.

Goal: fit a linear regression three ways on the same synthetic data and
verify that all three agree to 1e-6:

    (1) The closed-form normal equation via numpy.linalg.lstsq.
    (2) Gradient descent in a hand-written loop.
    (3) sklearn.linear_model.LinearRegression.

The three answers must agree on the coefficients and on the predictions.
The drill is in Lecture 2, Sections 3-4.

Estimated time: 50 minutes.

Run with:    python exercise-02-linear-regression-by-hand.py
Or test:     pytest exercise-02-linear-regression-by-hand.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-linear-regression-by-hand.py succeeds.

The exercise uses sklearn's bundled diabetes dataset (no network).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def load_and_scale() -> Tuple[np.ndarray, np.ndarray]:
    """Load the diabetes dataset and standardize the features.

    Returns (X, y) where X has been mean-centered and variance-scaled
    feature by feature. Scaling makes gradient descent's learning rate
    behave the same for every feature; the closed-form and sklearn would
    work without scaling, but the by-hand gradient descent would not
    converge in a reasonable number of iterations.
    """
    bunch = load_diabetes()
    X_raw = bunch.data
    y = bunch.target.astype(float)
    X = StandardScaler().fit_transform(X_raw)
    return X, y


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X.

    After this transform the intercept is the first element of beta.
    """
    # TODO: return np.c_[np.ones(len(X)), X].
    raise NotImplementedError("add_bias_column")


# -----------------------------------------------------------------------------
# Part A -- The closed-form solution (normal equation)
# -----------------------------------------------------------------------------

def fit_closed_form(X_with_bias: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return beta_hat using numpy.linalg.lstsq.

    np.linalg.lstsq is numerically more stable than
        np.linalg.inv(X.T @ X) @ X.T @ y
    on near-singular designs; both formulas are mathematically equivalent.
    Pass rcond=None to silence the FutureWarning in numpy 2.x.

    Returns a 1-D array of length p+1 (intercept first).
    """
    # TODO: implement using np.linalg.lstsq(X_with_bias, y, rcond=None)[0].
    raise NotImplementedError("Part A -- fit_closed_form")


# -----------------------------------------------------------------------------
# Part B -- Gradient descent by hand
# -----------------------------------------------------------------------------

def fit_gradient_descent(
    X_with_bias: np.ndarray, y: np.ndarray,
    lr: float = 0.1, n_iter: int = 5000,
) -> Tuple[np.ndarray, list[float]]:
    """Minimize MSE by gradient descent.

    Initialize beta to zeros. At each step:
        y_hat = X @ beta
        grad  = (2/n) * X.T @ (y_hat - y)
        beta  = beta - lr * grad

    Append the MSE at each iteration to a list and return both.

    Returns (beta, mse_history). The final MSE should match the
    closed-form MSE to 1e-6 if the loop converged.
    """
    # TODO: implement the gradient descent loop. The MSE convention here
    # is (1/n) sum (y - y_hat)^2 (no 1/2 factor).
    raise NotImplementedError("Part B -- fit_gradient_descent")


# -----------------------------------------------------------------------------
# Part C -- scikit-learn
# -----------------------------------------------------------------------------

def fit_sklearn(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Fit sklearn.linear_model.LinearRegression(fit_intercept=True) on
    the *unaugmented* X (sklearn adds its own intercept).

    Returns (intercept, coef) so that the augmented beta = [intercept, *coef]
    matches the by-hand approach.
    """
    # TODO: implement using LinearRegression(fit_intercept=True).
    raise NotImplementedError("Part C -- fit_sklearn")


def beta_from_sklearn(intercept: float, coef: np.ndarray) -> np.ndarray:
    """Pack sklearn's (intercept, coef) into the same (p+1,) vector
    used by the by-hand solutions.
    """
    return np.concatenate([[intercept], coef])


# -----------------------------------------------------------------------------
# Part D -- Verify all three agree
# -----------------------------------------------------------------------------

def verify_all_three_agree(
    beta_closed: np.ndarray, beta_gd: np.ndarray, beta_sk: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """Return True iff all three coefficient vectors agree elementwise
    within tol. tol is 1e-4 (not 1e-6) because gradient descent is the
    slowest to converge; the closed-form and sklearn agree to 1e-12.
    """
    a = np.allclose(beta_closed, beta_gd, atol=tol)
    b = np.allclose(beta_closed, beta_sk, atol=1e-6)
    c = np.allclose(beta_gd, beta_sk, atol=tol)
    return bool(a and b and c)


# -----------------------------------------------------------------------------
# Glue
# -----------------------------------------------------------------------------

def run_pipeline() -> dict[str, object]:
    """Fit three ways, return the betas and a final 'agree' boolean."""
    X, y = load_and_scale()
    X_aug = add_bias_column(X)

    beta_closed = fit_closed_form(X_aug, y)
    beta_gd, mse_history = fit_gradient_descent(X_aug, y, lr=0.1, n_iter=5000)
    intercept, coef = fit_sklearn(X, y)
    beta_sk = beta_from_sklearn(intercept, coef)

    return {
        "beta_closed": beta_closed,
        "beta_gd":     beta_gd,
        "beta_sk":     beta_sk,
        "mse_history": mse_history,
        "agree":       verify_all_three_agree(beta_closed, beta_gd, beta_sk),
    }


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_bias_column_added() -> None:
    X, _ = load_and_scale()
    X_aug = add_bias_column(X)
    assert X_aug.shape == (X.shape[0], X.shape[1] + 1)
    assert np.allclose(X_aug[:, 0], 1.0)


def test_closed_form_runs() -> None:
    X, y = load_and_scale()
    X_aug = add_bias_column(X)
    beta = fit_closed_form(X_aug, y)
    assert beta.shape == (X_aug.shape[1],)
    assert np.all(np.isfinite(beta))


def test_gradient_descent_converges() -> None:
    X, y = load_and_scale()
    X_aug = add_bias_column(X)
    beta_gd, history = fit_gradient_descent(X_aug, y, lr=0.1, n_iter=5000)
    assert beta_gd.shape == (X_aug.shape[1],)
    assert len(history) == 5000
    # Loss must decrease monotonically (after the first few steps).
    assert history[-1] < history[0], "gradient descent did not reduce MSE"


def test_three_agree() -> None:
    out = run_pipeline()
    assert out["agree"] is True, (
        "the three fits disagree -- check the gradient-descent learning "
        "rate, the iteration count, or the bias-column convention."
    )


def test_against_sklearn_explicitly() -> None:
    """The closed-form fit should match sklearn to floating-point tolerance."""
    X, y = load_and_scale()
    X_aug = add_bias_column(X)
    beta_closed = fit_closed_form(X_aug, y)
    intercept, coef = fit_sklearn(X, y)
    beta_sk = beta_from_sklearn(intercept, coef)
    assert np.allclose(beta_closed, beta_sk, atol=1e-6), (
        f"closed-form intercept={beta_closed[0]:.6f} vs sklearn={beta_sk[0]:.6f}"
    )


def _run_all_tests() -> None:
    test_bias_column_added()
    test_closed_form_runs()
    test_gradient_descent_converges()
    test_three_agree()
    test_against_sklearn_explicitly()
    print("OK -- exercise 2")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# add_bias_column:
#     return np.c_[np.ones(len(X)), X]
#
# fit_closed_form:
#     beta, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)
#     return beta
#
# fit_gradient_descent:
#     n = len(y)
#     beta = np.zeros(X_with_bias.shape[1])
#     history = []
#     for _ in range(n_iter):
#         y_hat = X_with_bias @ beta
#         grad  = (2.0 / n) * X_with_bias.T @ (y_hat - y)
#         beta  = beta - lr * grad
#         history.append(float(np.mean((y - y_hat) ** 2)))
#     return beta, history
#
# fit_sklearn:
#     mdl = LinearRegression(fit_intercept=True).fit(X, y)
#     return float(mdl.intercept_), mdl.coef_
#
# -----------------------------------------------------------------------------
