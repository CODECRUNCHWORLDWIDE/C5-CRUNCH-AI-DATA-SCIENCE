"""
Exercise 1 -- K-Means from Scratch.

Goal: implement Lloyd's algorithm in pure NumPy that agrees with
sklearn.cluster.KMeans on Gaussian-blob data:

    (1) The assignment step: each row to its nearest centroid by squared
        Euclidean distance (Lecture 1, Section 3).
    (2) The update step: each centroid to the mean of its assigned rows
        (Lecture 1, Sections 2 and 3).
    (3) Convergence detection: stop when labels do not change between
        iterations.
    (4) The k-means++ initialization (Lecture 1, Section 5).

The two clusterings will not necessarily agree on labels (cluster IDs can
swap between runs) but they should agree on the partition up to a label
permutation, which we measure with adjusted_rand_score.

Estimated time: 70 minutes.

Run with:    python exercise-01-kmeans-from-scratch.py
Or test:     pytest exercise-01-kmeans-from-scratch.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-kmeans-from-scratch.py succeeds.

The exercise builds deterministic synthetic data so the tests never
depend on the network.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Part A -- the assignment step
# -----------------------------------------------------------------------------

def assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the label vector: for each row in X, the index of the
    nearest centroid by squared Euclidean distance.

    X         -- (n, p) float array
    centroids -- (k, p) float array
    Returns   -- (n,) int array with values in [0, k)

    Use broadcasting: ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    gives an (n, k) matrix of squared distances. The argmin along axis=1 is
    the label vector.

    The squared distance equals ||x||^2 - 2 x.c + ||c||^2; since ||x||^2 is
    constant per row, you can drop it and get the same argmin. But the full
    formula is faster to debug and we have the memory budget here.
    """
    # TODO: implement the assignment step.
    # See HINT block at the bottom of the file.
    raise NotImplementedError("Part A -- assign_labels")


# -----------------------------------------------------------------------------
# Part B -- the centroid update step
# -----------------------------------------------------------------------------

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int,
                     old_centroids: np.ndarray) -> np.ndarray:
    """Return the new centroids: for each cluster c, the mean of the
    rows assigned to c. If a cluster has no assigned rows, keep its old
    centroid (sklearn re-initializes; we keep it simple).

    X             -- (n, p) float array
    labels        -- (n,) int array with values in [0, k)
    k             -- the number of clusters
    old_centroids -- (k, p) float array (used as fallback for empty clusters)
    Returns       -- (k, p) float array, the new centroids.
    """
    # TODO: implement the centroid update.
    raise NotImplementedError("Part B -- update_centroids")


# -----------------------------------------------------------------------------
# Part C -- k-means++ initialization
# -----------------------------------------------------------------------------

def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """The k-means++ initialization (Lecture 1, Section 5).

    1. Pick the first centroid uniformly at random from X.
    2. For each remaining centroid:
       a. Compute D_i = min squared distance from x_i to any already-chosen centroid.
       b. Sample the next centroid with probability proportional to D_i.
    3. Return the (k, p) centroid matrix.

    Returns -- (k, p) float array.
    """
    n, p = X.shape
    centroids = np.empty((k, p), dtype=float)
    # First centroid: uniform random.
    centroids[0] = X[rng.integers(0, n)]

    # TODO: pick the remaining k - 1 centroids by the k-means++ rule.
    # See HINT block at the bottom.
    raise NotImplementedError("Part C -- kmeans_plus_plus_init")


# -----------------------------------------------------------------------------
# Part D -- the full Lloyd's algorithm
# -----------------------------------------------------------------------------

def kmeans(
    X: np.ndarray, k: int, *,
    max_iter: int = 300,
    init: str = "k-means++",
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Lloyd's algorithm for k-means.

    Parameters
    ----------
    X            -- (n, p) float array
    k            -- the number of clusters
    max_iter     -- the maximum number of iterations (default 300)
    init         -- either "k-means++" or "random" (default "k-means++")
    random_state -- the seed for reproducibility

    Returns
    -------
    labels    -- (n,) int array of cluster IDs
    centroids -- (k, p) array of cluster means
    inertia   -- float, the final WCSS
    n_iter    -- the number of iterations until convergence
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    if k > n:
        raise ValueError(f"k={k} exceeds the number of rows n={n}")

    if init == "k-means++":
        centroids = kmeans_plus_plus_init(X, k, rng)
    elif init == "random":
        idx = rng.choice(n, size=k, replace=False)
        centroids = X[idx].copy()
    else:
        raise ValueError(f"unknown init: {init!r}")

    labels = np.full(n, -1, dtype=int)
    # TODO: implement the Lloyd's-algorithm main loop.
    # See HINT block at the bottom.
    n_iter = 0
    raise NotImplementedError("Part D -- kmeans main loop")


# -----------------------------------------------------------------------------
# Glue -- compare to sklearn
# -----------------------------------------------------------------------------

def run_comparison(k: int = 4, n_samples: int = 600) -> dict[str, object]:
    """Fit both implementations on a Gaussian-blob dataset and return both
    inertias and the adjusted Rand index between the two label vectors.
    """
    X, _ = make_blobs(
        n_samples=n_samples, centers=k, cluster_std=0.7,
        center_box=(-8.0, 8.0), random_state=RANDOM_STATE,
    )

    ours_labels, _, ours_inertia, ours_n_iter = kmeans(
        X, k=k, init="k-means++", random_state=RANDOM_STATE,
    )
    sk = KMeans(n_clusters=k, init="k-means++", n_init=10,
                random_state=RANDOM_STATE).fit(X)

    return {
        "ours_inertia": float(ours_inertia),
        "sk_inertia": float(sk.inertia_),
        "ours_n_iter": int(ours_n_iter),
        "sk_n_iter": int(sk.n_iter_),
        "ari": float(adjusted_rand_score(sk.labels_, ours_labels)),
    }


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_assign_labels_is_correct_on_a_trivial_problem() -> None:
    """Centroids at (0, 0) and (10, 0); rows split between them by x-coordinate."""
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    X = np.array([[1.0, 0.0], [9.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    labels = assign_labels(X, centroids)
    assert np.array_equal(labels, np.array([0, 1, 0, 1]))


def test_update_centroids_returns_the_mean() -> None:
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
    labels = np.array([0, 0, 1, 1])
    old = np.zeros((2, 2))
    new = update_centroids(X, labels, k=2, old_centroids=old)
    assert np.allclose(new[0], [1.0, 0.0])
    assert np.allclose(new[1], [11.0, 0.0])


def test_update_centroids_handles_empty_cluster() -> None:
    """An empty cluster should fall back to its old centroid."""
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0])
    old = np.array([[1.0, 1.0], [99.0, 99.0]])
    new = update_centroids(X, labels, k=2, old_centroids=old)
    assert np.allclose(new[0], [1.0, 0.0])
    assert np.allclose(new[1], [99.0, 99.0])


def test_kmeans_plus_plus_returns_k_unique_centroids() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    centroids = kmeans_plus_plus_init(X, k=3, rng=rng)
    assert centroids.shape == (3, 2)
    # Each centroid should be one of the rows of X.
    for c in centroids:
        assert any(np.allclose(c, x) for x in X)
    # The three centroids should be distinct.
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        assert not np.allclose(centroids[i], centroids[j])


def test_kmeans_converges_on_blobs() -> None:
    """The algorithm should converge in well under max_iter on easy data."""
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.6,
                      random_state=RANDOM_STATE)
    labels, centroids, inertia, n_iter = kmeans(X, k=3, random_state=RANDOM_STATE)
    assert labels.shape == (200,)
    assert centroids.shape == (3, 2)
    assert inertia > 0
    assert n_iter < 50, f"took {n_iter} iterations on easy data"


def test_kmeans_agrees_with_sklearn_on_partition() -> None:
    """Our partition should agree with sklearn's up to a label permutation
    (adjusted_rand_score > 0.9 on a 4-blob dataset).
    """
    out = run_comparison(k=4)
    assert out["ari"] > 0.9, f"ARI = {out['ari']:.3f} is too low"


def test_kmeans_inertia_is_close_to_sklearn() -> None:
    """Our inertia should be within ~5% of sklearn's.

    sklearn uses n_init=10 (10 random restarts), we use n_init=1, so we
    allow some slack.
    """
    out = run_comparison(k=4)
    ratio = out["ours_inertia"] / out["sk_inertia"]
    assert 0.95 <= ratio <= 1.20, (
        f"ours inertia = {out['ours_inertia']:.2f}, "
        f"sklearn inertia = {out['sk_inertia']:.2f}, ratio = {ratio:.3f}"
    )


def _run_all_tests() -> None:
    test_assign_labels_is_correct_on_a_trivial_problem()
    test_update_centroids_returns_the_mean()
    test_update_centroids_handles_empty_cluster()
    test_kmeans_plus_plus_returns_k_unique_centroids()
    test_kmeans_converges_on_blobs()
    test_kmeans_agrees_with_sklearn_on_partition()
    test_kmeans_inertia_is_close_to_sklearn()
    print("OK -- exercise 1")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# assign_labels:
#     # (n, k) matrix of squared distances:
#     dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
#     return dists.argmin(axis=1)
#
# update_centroids:
#     n, p = X.shape
#     new_centroids = old_centroids.copy()
#     for c in range(k):
#         mask = labels == c
#         if mask.any():
#             new_centroids[c] = X[mask].mean(axis=0)
#     return new_centroids
#
# kmeans_plus_plus_init:
#     n, p = X.shape
#     centroids = np.empty((k, p), dtype=float)
#     centroids[0] = X[rng.integers(0, n)]
#     for i in range(1, k):
#         # Squared distance from each row to its nearest existing centroid:
#         d2 = ((X[:, None, :] - centroids[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
#         probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
#         idx = rng.choice(n, p=probs)
#         centroids[i] = X[idx]
#     return centroids
#
# kmeans main loop:
#     for it in range(max_iter):
#         new_labels = assign_labels(X, centroids)
#         if np.array_equal(new_labels, labels):
#             n_iter = it
#             break
#         labels = new_labels
#         centroids = update_centroids(X, labels, k, centroids)
#     else:
#         n_iter = max_iter
#     inertia = float(((X - centroids[labels]) ** 2).sum())
#     return labels, centroids, inertia, n_iter
#
# -----------------------------------------------------------------------------
