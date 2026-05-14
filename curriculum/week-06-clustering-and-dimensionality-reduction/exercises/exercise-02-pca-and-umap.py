"""
Exercise 2 -- PCA and UMAP on the digits dataset.

Goal: implement PCA via the SVD, verify it matches sklearn, and (if
umap-learn is installed) compare PCA, t-SNE, and UMAP as 2D embeddings
of the same data:

    (1) PCA from the SVD of centered X (Lecture 2, Section 2).
    (2) Explained-variance ratio (Lecture 2, Section 3).
    (3) The scree plot for picking k.
    (4) A side-by-side 2D embedding of digits via PCA, t-SNE, UMAP
        (Lecture 2, Section 10).

Estimated time: 60 minutes.

Run with:    python exercise-02-pca-and-umap.py
Or test:     pytest exercise-02-pca-and-umap.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-pca-and-umap.py succeeds.

UMAP is optional. If umap-learn is not installed, the UMAP-specific
checks skip with a printed warning.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Part A -- PCA via the SVD
# -----------------------------------------------------------------------------

def pca_from_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA implemented from the SVD of the centered data matrix.

    Steps (Lecture 2, Section 2):
        1. Center X by subtracting the column means.
        2. Compute the (thin) SVD: X_centered = U @ diag(sigma) @ Vt.
        3. The top-k principal directions are the first k rows of Vt.
        4. The projected coordinates are X_centered @ V[:, :k] = U[:, :k] * sigma[:k].
        5. The eigenvalues of the covariance are sigma^2 / (n - 1).
        6. The explained-variance ratio is eigenvalue_k / sum(eigenvalues).

    Parameters
    ----------
    X -- (n, p) data matrix
    k -- the number of principal components to return

    Returns
    -------
    Z                        -- (n, k) projected coordinates
    components               -- (k, p) the principal directions (rows of Vt[:k])
    explained_variance_ratio -- (k,) explained variance ratios summing to <= 1
    """
    # TODO: implement PCA from the SVD.
    raise NotImplementedError("Part A -- pca_from_svd")


# -----------------------------------------------------------------------------
# Part B -- pick k for a variance threshold
# -----------------------------------------------------------------------------

def pick_k_for_variance(explained_variance_ratio: np.ndarray, threshold: float) -> int:
    """Return the smallest k such that the cumulative explained variance
    is at least `threshold` (e.g., 0.9).

    Parameters
    ----------
    explained_variance_ratio -- (n_components,) ratios in decreasing order
    threshold                -- e.g., 0.90 for "explain 90% of the variance"

    Returns -- the smallest k for which the cumulative ratio is >= threshold.
    """
    # TODO: implement.
    raise NotImplementedError("Part B -- pick_k_for_variance")


# -----------------------------------------------------------------------------
# Part C -- the comparison driver
# -----------------------------------------------------------------------------

def run_pca_comparison() -> dict[str, object]:
    """Fit our PCA and sklearn's PCA on the digits dataset; return both
    explained-variance ratios.
    """
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X.astype(float))

    Z_ours, _, evr_ours = pca_from_svd(X, k=10)
    sk = SklearnPCA(n_components=10, random_state=RANDOM_STATE).fit(X)

    return {
        "ours_evr": evr_ours,
        "sk_evr": sk.explained_variance_ratio_,
        "ours_shape": Z_ours.shape,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }


def run_embedding_demo() -> dict[str, object]:
    """Fit PCA(2), t-SNE(2), and (if available) UMAP(2) on the digits dataset
    and report the dimensions of each.

    Returns a dict with keys:
        pca_shape  -- (n, 2)
        tsne_shape -- (n, 2)
        umap_shape -- (n, 2) or None if umap-learn is not installed
    """
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X.astype(float))

    pca_emb = SklearnPCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

    # t-SNE: use init="pca" for reproducibility; perplexity 30 is the default.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne_emb = TSNE(
            n_components=2, perplexity=30, init="pca",
            random_state=RANDOM_STATE, learning_rate="auto",
        ).fit_transform(X)

    umap_emb = None
    try:
        import umap  # noqa: F401 -- optional dependency

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            umap_emb = umap.UMAP(
                n_components=2, random_state=RANDOM_STATE,
            ).fit_transform(X)
    except ImportError:
        pass

    return {
        "pca_shape": pca_emb.shape,
        "tsne_shape": tsne_emb.shape,
        "umap_shape": umap_emb.shape if umap_emb is not None else None,
        "labels_shape": y.shape,
    }


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_pca_shape_is_right() -> None:
    X = np.random.default_rng(RANDOM_STATE).standard_normal((100, 8))
    Z, comps, evr = pca_from_svd(X, k=3)
    assert Z.shape == (100, 3)
    assert comps.shape == (3, 8)
    assert evr.shape == (3,)


def test_pca_explained_variance_sums_to_at_most_one() -> None:
    X = np.random.default_rng(RANDOM_STATE).standard_normal((100, 8))
    _, _, evr = pca_from_svd(X, k=8)
    # With k = p, the cumulative explained variance is ~1.
    assert 0.99 <= evr.sum() <= 1.01


def test_pca_explained_variance_is_decreasing() -> None:
    X = np.random.default_rng(RANDOM_STATE).standard_normal((100, 8))
    _, _, evr = pca_from_svd(X, k=5)
    # PCA components are returned sorted by variance (largest first).
    for i in range(len(evr) - 1):
        assert evr[i] >= evr[i + 1] - 1e-10


def test_pca_matches_sklearn_on_digits() -> None:
    """Our explained-variance ratio should match sklearn's to ~3 decimals."""
    out = run_pca_comparison()
    diff = np.abs(np.array(out["ours_evr"]) - np.array(out["sk_evr"]))
    assert diff.max() < 1e-3, (
        f"ours EVR = {out['ours_evr']}, sk EVR = {out['sk_evr']}, "
        f"max abs diff = {diff.max():.6f}"
    )


def test_pick_k_for_variance_returns_one_when_first_pc_dominates() -> None:
    evr = np.array([0.95, 0.03, 0.01, 0.01])
    assert pick_k_for_variance(evr, 0.90) == 1


def test_pick_k_for_variance_handles_a_typical_curve() -> None:
    # A gentle scree: each PC adds about 0.18 of the variance.
    evr = np.array([0.30, 0.20, 0.18, 0.14, 0.10, 0.05, 0.02, 0.01])
    assert pick_k_for_variance(evr, 0.50) == 2
    assert pick_k_for_variance(evr, 0.80) == 4
    assert pick_k_for_variance(evr, 0.95) == 6


def test_embedding_shapes_are_2d() -> None:
    out = run_embedding_demo()
    assert out["pca_shape"][1] == 2
    assert out["tsne_shape"][1] == 2
    # UMAP is optional; if present, it must also be 2D.
    if out["umap_shape"] is not None:
        assert out["umap_shape"][1] == 2


def _run_all_tests() -> None:
    test_pca_shape_is_right()
    test_pca_explained_variance_sums_to_at_most_one()
    test_pca_explained_variance_is_decreasing()
    test_pca_matches_sklearn_on_digits()
    test_pick_k_for_variance_returns_one_when_first_pc_dominates()
    test_pick_k_for_variance_handles_a_typical_curve()
    test_embedding_shapes_are_2d()
    print("OK -- exercise 2")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# pca_from_svd:
#     mu = X.mean(axis=0, keepdims=True)
#     Xc = X - mu
#     U, sigma, Vt = np.linalg.svd(Xc, full_matrices=False)
#     # Sign convention: numpy.linalg.svd's signs differ from sklearn's.
#     # We do not normalize signs here; the explained-variance ratio
#     # is sign-independent.
#     n = X.shape[0]
#     eigenvalues = sigma ** 2 / (n - 1)
#     total_variance = eigenvalues.sum()
#     components = Vt[:k]
#     Z = U[:, :k] * sigma[:k]
#     explained_variance_ratio = eigenvalues[:k] / total_variance
#     return Z, components, explained_variance_ratio
#
# pick_k_for_variance:
#     cum = np.cumsum(explained_variance_ratio)
#     above = np.where(cum >= threshold)[0]
#     if len(above) == 0:
#         return len(explained_variance_ratio)
#     return int(above[0] + 1)   # +1 because k is 1-indexed
#
# -----------------------------------------------------------------------------
