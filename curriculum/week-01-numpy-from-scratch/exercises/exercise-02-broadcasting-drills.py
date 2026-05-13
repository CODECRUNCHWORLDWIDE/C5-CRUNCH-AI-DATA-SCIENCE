"""
Exercise 2 — Broadcasting drills.

Goal: rewrite six Python loops as one or two vectorized lines. Each
problem is small. The point is not the answer, the point is the *habit*:
when you see a loop over array elements, your first move is to ask "what
shape do I want, what shape do I have, and how do I broadcast between
them?"

Estimated time: 40 minutes.

Run with:    python exercise-02-broadcasting-drills.py
Or test:     pytest exercise-02-broadcasting-drills.py

Acceptance criteria:
- Every TODO is filled in with a vectorized expression (no explicit
  Python `for` loop over array elements).
- All asserts pass; the script prints "OK — exercise 2".

Do not look at the HINT block at the bottom until you have tried for
fifteen minutes per problem.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Problem 1 — Add a scalar to every element
# -----------------------------------------------------------------------------

def add_scalar(a: np.ndarray, s: float) -> np.ndarray:
    """Return a + s, vectorized. No for-loop.

    Loop equivalent:
        out = np.empty_like(a)
        for i in range(len(a)):
            out[i] = a[i] + s
    """
    # TODO: one expression.
    raise NotImplementedError("Problem 1")


# -----------------------------------------------------------------------------
# Problem 2 — Pairwise squared error (an ML staple)
# -----------------------------------------------------------------------------

def squared_error_sum(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return the sum of (y - yhat)**2 across all elements.

    Loop equivalent:
        total = 0.0
        for i in range(len(y)):
            total += (y[i] - yhat[i]) ** 2
        return total
    """
    # TODO: one expression. Cast the result to a Python float for the test.
    raise NotImplementedError("Problem 2")


# -----------------------------------------------------------------------------
# Problem 3 — Distance from a point to each row of a matrix
# -----------------------------------------------------------------------------

def distances_to_point(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Given X of shape (N, D) and p of shape (D,), return a 1-D array of
    Euclidean distances of length N.

    Loop equivalent:
        d = np.empty(len(X))
        for i in range(len(X)):
            d[i] = np.sqrt(((X[i] - p) ** 2).sum())
        return d
    """
    # TODO: use broadcasting + a reduction along axis=1.
    raise NotImplementedError("Problem 3")


# -----------------------------------------------------------------------------
# Problem 4 — Per-column z-score (the StandardScaler one-liner)
# -----------------------------------------------------------------------------

def zscore_columns(X: np.ndarray) -> np.ndarray:
    """Return an array of the same shape as X where each column has been
    standardized to mean=0, std=1. Use the POPULATION std (ddof=0).

    Loop equivalent (sketch):
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            col = X[:, j]
            out[:, j] = (col - col.mean()) / col.std()
        return out
    """
    # TODO: one expression. Watch out for which axis to reduce along.
    raise NotImplementedError("Problem 4")


# -----------------------------------------------------------------------------
# Problem 5 — A 2-D grid of sin(x) * cos(y) (outer product, vectorized)
# -----------------------------------------------------------------------------

def sin_cos_grid(n: int) -> np.ndarray:
    """Return a (n, n) array out[i, j] = sin(i / 10) * cos(j / 10).

    Loop equivalent:
        out = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                out[i, j] = np.sin(i / 10) * np.cos(j / 10)
        return out
    """
    # TODO: build a 1-D array of i / 10 and one of j / 10, then broadcast
    #       using [:, None] and [None, :].
    raise NotImplementedError("Problem 5")


# -----------------------------------------------------------------------------
# Problem 6 — Threshold an image-like 2-D array
# -----------------------------------------------------------------------------

def threshold(a: np.ndarray, t: float) -> np.ndarray:
    """Return an array the same shape as `a` with:
       - 255 where a > t,
       -   0 elsewhere.

    The output dtype must be uint8.

    Loop equivalent:
        out = np.zeros_like(a, dtype=np.uint8)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] > t:
                    out[i, j] = 255
        return out
    """
    # TODO: use np.where, plus an .astype to uint8.
    raise NotImplementedError("Problem 6")


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_add_scalar() -> None:
    a = np.array([1.0, 2.0, 3.0])
    out = add_scalar(a, 10.0)
    assert np.allclose(out, [11.0, 12.0, 13.0])


def test_squared_error_sum() -> None:
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([0.0, 2.0, 5.0])
    # (1-0)^2 + (2-2)^2 + (3-5)^2 = 1 + 0 + 4 = 5
    assert abs(squared_error_sum(y, yhat) - 5.0) < 1e-12


def test_distances_to_point() -> None:
    X = np.array([[0.0, 0.0],
                  [3.0, 4.0],
                  [1.0, 0.0]])
    p = np.array([0.0, 0.0])
    d = distances_to_point(X, p)
    assert d.shape == (3,)
    assert np.allclose(d, [0.0, 5.0, 1.0])


def test_zscore_columns() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=2.0, size=(100, 3))
    Z = zscore_columns(X)
    assert Z.shape == X.shape
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-12)


def test_sin_cos_grid() -> None:
    G = sin_cos_grid(5)
    assert G.shape == (5, 5)
    # Spot check a single entry.
    assert abs(G[2, 3] - (np.sin(0.2) * np.cos(0.3))) < 1e-12


def test_threshold() -> None:
    a = np.array([[10.0, 200.0], [30.0, 250.0]])
    out = threshold(a, 100.0)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 255], [0, 255]]


def _run_all_tests() -> None:
    test_add_scalar()
    test_squared_error_sum()
    test_distances_to_point()
    test_zscore_columns()
    test_sin_cos_grid()
    test_threshold()
    print("OK — exercise 2")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min on a given problem)
# -----------------------------------------------------------------------------
#
# Problem 1:
#     return a + s
#
# Problem 2:
#     return float(((y - yhat) ** 2).sum())
#
# Problem 3:
#     # X - p broadcasts (D,) across N rows.
#     return np.sqrt(((X - p) ** 2).sum(axis=1))
#
# Problem 4:
#     mu    = X.mean(axis=0)
#     sigma = X.std(axis=0)
#     return (X - mu) / sigma
#
# Problem 5:
#     x = np.arange(n) / 10.0
#     y = np.arange(n) / 10.0
#     return np.sin(x[:, None]) * np.cos(y[None, :])
#
# Problem 6:
#     return np.where(a > t, 255, 0).astype(np.uint8)
#
# -----------------------------------------------------------------------------
