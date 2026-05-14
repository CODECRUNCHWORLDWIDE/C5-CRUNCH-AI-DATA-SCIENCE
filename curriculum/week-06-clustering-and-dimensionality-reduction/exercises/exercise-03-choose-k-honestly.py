"""
Exercise 3 -- Choose k Honestly.

Goal: implement the three principal "what is the right k?" diagnostics
and run them on (a) data with real cluster structure and (b) data
without it. Recognize when the diagnostics agree, when they disagree,
and when they tell you not to cluster at all:

    (1) The elbow plot of WCSS vs k (Lecture 3, Section 2).
    (2) The silhouette score, averaged over a range of k
        (Lecture 1, Section 7 and Lecture 3, Section 2).
    (3) The gap statistic (Lecture 3, Section 4 and Tibshirani et al., 2001).
    (4) The column-permutation baseline (Lecture 3, Section 3).

The headline test fits the diagnostics on two datasets and asserts
that they agree the first is clustered and the second is not.

Estimated time: 60 minutes.

Run with:    python exercise-03-choose-k-honestly.py
Or test:     pytest exercise-03-choose-k-honestly.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-03-choose-k-honestly.py succeeds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Part A -- the elbow / WCSS curve
# -----------------------------------------------------------------------------

def wcss_curve(X: np.ndarray, k_range: Sequence[int],
               random_state: int = RANDOM_STATE) -> np.ndarray:
    """For each k in k_range, fit KMeans and return the inertia
    (within-cluster sum of squares).

    Returns -- (len(k_range),) float array.
    """
    # TODO: implement; use KMeans(n_clusters=k, n_init=10, random_state=...).
    raise NotImplementedError("Part A -- wcss_curve")


# -----------------------------------------------------------------------------
# Part B -- the silhouette curve
# -----------------------------------------------------------------------------

def silhouette_curve(X: np.ndarray, k_range: Sequence[int],
                     random_state: int = RANDOM_STATE) -> np.ndarray:
    """For each k in k_range (k >= 2), fit KMeans and return the mean
    silhouette score on X.

    The silhouette is undefined for k = 1, so the caller is expected to
    pass a range starting at 2.

    Returns -- (len(k_range),) float array.
    """
    if min(k_range) < 2:
        raise ValueError("silhouette score is undefined for k < 2")
    # TODO: implement.
    raise NotImplementedError("Part B -- silhouette_curve")


# -----------------------------------------------------------------------------
# Part C -- the gap statistic
# -----------------------------------------------------------------------------

def gap_statistic(X: np.ndarray, k_range: Sequence[int],
                  B: int = 10,
                  random_state: int = RANDOM_STATE
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gap statistic of Tibshirani, Walther, and Hastie (2001).

    Algorithm (Lecture 3, Section 4):
        For each k in k_range:
            1. Compute log(WCSS_k) on the real data X.
            2. Generate B reference datasets uniformly over the
               bounding box of X.
            3. Compute log(WCSS_k) on each reference; take the mean
               and the standard error (= std / sqrt(B)).
            4. Gap(k) = mean(log WCSS_ref) - log WCSS_real.

    Returns
    -------
    gap          -- (len(k_range),) float array
    standard_err -- (len(k_range),) float array
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    lo = X.min(axis=0)
    hi = X.max(axis=0)

    gap = np.zeros(len(k_range))
    standard_err = np.zeros(len(k_range))

    # TODO: implement the gap-statistic loop.
    raise NotImplementedError("Part C -- gap_statistic")


# -----------------------------------------------------------------------------
# Part D -- the column-permutation baseline
# -----------------------------------------------------------------------------

def reshuffle_baseline_silhouette(X: np.ndarray, k: int,
                                  random_state: int = RANDOM_STATE) -> dict[str, float]:
    """Fit k-means on X and on a column-permuted copy of X; return both
    silhouette scores.

    Permuting each column independently destroys joint structure
    (correlations between features) while preserving the marginal
    distribution of each feature. The ratio between the two silhouettes
    is the diagnostic from Lecture 3 Section 3:
        > 2: real clusters
        < 1.5: artefact

    Returns -- dict with keys
        real_silhouette   -- the silhouette on the original X
        perm_silhouette   -- the silhouette on the column-permuted X
        ratio             -- real_silhouette / perm_silhouette
    """
    rng = np.random.default_rng(random_state)

    # TODO: column-permute X and fit KMeans on both.
    raise NotImplementedError("Part D -- reshuffle_baseline_silhouette")


# -----------------------------------------------------------------------------
# Helpers -- synthetic datasets
# -----------------------------------------------------------------------------

def make_clustered_dataset(n: int = 500) -> np.ndarray:
    """Four well-separated Gaussian blobs in 2D."""
    X, _ = make_blobs(n_samples=n, centers=4, cluster_std=0.6,
                      center_box=(-8.0, 8.0), random_state=RANDOM_STATE)
    return X


def make_unclustered_dataset(n: int = 500) -> np.ndarray:
    """A 5-dimensional standard normal sample. No cluster structure."""
    rng = np.random.default_rng(RANDOM_STATE)
    return rng.standard_normal((n, 5))


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_wcss_curve_is_monotone_decreasing() -> None:
    """WCSS strictly decreases as k increases."""
    X = make_clustered_dataset()
    wcss = wcss_curve(X, k_range=[2, 3, 4, 5, 6])
    for i in range(len(wcss) - 1):
        assert wcss[i] > wcss[i + 1], (
            f"WCSS at k={i+2} ({wcss[i]:.2f}) is not greater than "
            f"WCSS at k={i+3} ({wcss[i+1]:.2f})"
        )


def test_silhouette_curve_peaks_at_known_k() -> None:
    """On a 4-blob dataset, the silhouette score peaks at k = 4."""
    X = make_clustered_dataset()
    k_range = list(range(2, 8))
    sils = silhouette_curve(X, k_range=k_range)
    best_k = k_range[int(np.argmax(sils))]
    assert best_k == 4, f"silhouette peaks at k = {best_k}, expected 4"


def test_silhouette_is_low_on_unclustered_data() -> None:
    """On standard-normal noise, the silhouette should be below 0.25 for
    all k.
    """
    X = make_unclustered_dataset()
    sils = silhouette_curve(X, k_range=[2, 3, 4, 5])
    assert sils.max() < 0.25, f"max silhouette = {sils.max():.3f} on noise"


def test_gap_statistic_is_positive_on_clustered_data() -> None:
    """The gap should be positive at the true k = 4 for the blob dataset."""
    X = make_clustered_dataset()
    gap, _ = gap_statistic(X, k_range=[2, 3, 4, 5], B=5)
    # At k = 4 (index 2), the gap should be large and positive.
    assert gap[2] > 0, f"gap at k=4 is {gap[2]:.3f}, expected positive"


def test_gap_statistic_shape() -> None:
    """gap and standard_err should have the same length as k_range."""
    X = make_clustered_dataset(n=200)
    gap, se = gap_statistic(X, k_range=[2, 3, 4], B=3)
    assert gap.shape == (3,)
    assert se.shape == (3,)
    assert np.all(se >= 0)


def test_reshuffle_baseline_separates_clustered_from_noise() -> None:
    """On clustered data the ratio should be > 2; on unclustered data
    the ratio should be near 1.
    """
    X_clustered = make_clustered_dataset()
    X_noise = make_unclustered_dataset()

    out_clustered = reshuffle_baseline_silhouette(X_clustered, k=4)
    out_noise = reshuffle_baseline_silhouette(X_noise, k=4)

    assert out_clustered["ratio"] > 1.5, (
        f"clustered ratio = {out_clustered['ratio']:.3f}, expected > 1.5"
    )
    # On noise the ratio should be near 1 (loose tolerance because of
    # finite-sample noise in the silhouette estimator).
    assert out_noise["ratio"] < 1.6, (
        f"noise ratio = {out_noise['ratio']:.3f}, expected near 1"
    )


def _run_all_tests() -> None:
    test_wcss_curve_is_monotone_decreasing()
    test_silhouette_curve_peaks_at_known_k()
    test_silhouette_is_low_on_unclustered_data()
    test_gap_statistic_is_positive_on_clustered_data()
    test_gap_statistic_shape()
    test_reshuffle_baseline_separates_clustered_from_noise()
    print("OK -- exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# wcss_curve:
#     wcss = np.zeros(len(k_range))
#     for i, k in enumerate(k_range):
#         wcss[i] = KMeans(n_clusters=k, n_init=10,
#                          random_state=random_state).fit(X).inertia_
#     return wcss
#
# silhouette_curve:
#     sils = np.zeros(len(k_range))
#     for i, k in enumerate(k_range):
#         labels = KMeans(n_clusters=k, n_init=10,
#                         random_state=random_state).fit(X).labels_
#         sils[i] = silhouette_score(X, labels)
#     return sils
#
# gap_statistic:
#     for i, k in enumerate(k_range):
#         real_inertia = KMeans(n_clusters=k, n_init=10,
#                               random_state=random_state).fit(X).inertia_
#         log_real = np.log(real_inertia)
#         log_refs = np.zeros(B)
#         for b in range(B):
#             X_ref = rng.uniform(lo, hi, size=(n, p))
#             ref_inertia = KMeans(n_clusters=k, n_init=10,
#                                  random_state=b).fit(X_ref).inertia_
#             log_refs[b] = np.log(ref_inertia)
#         gap[i] = log_refs.mean() - log_real
#         standard_err[i] = log_refs.std() / np.sqrt(B)
#     return gap, standard_err
#
# reshuffle_baseline_silhouette:
#     X_perm = X.copy()
#     for j in range(X.shape[1]):
#         X_perm[:, j] = rng.permutation(X_perm[:, j])
#     real_labels = KMeans(n_clusters=k, n_init=10,
#                          random_state=random_state).fit(X).labels_
#     perm_labels = KMeans(n_clusters=k, n_init=10,
#                          random_state=random_state).fit(X_perm).labels_
#     real_s = silhouette_score(X, real_labels)
#     perm_s = silhouette_score(X_perm, perm_labels)
#     return {
#         "real_silhouette": float(real_s),
#         "perm_silhouette": float(perm_s),
#         "ratio": float(real_s / perm_s) if perm_s > 0 else float("inf"),
#     }
#
# -----------------------------------------------------------------------------
