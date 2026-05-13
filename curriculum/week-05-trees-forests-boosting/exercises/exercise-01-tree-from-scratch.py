"""
Exercise 1 -- A Decision Tree Regressor from Scratch.

Goal: implement a regression tree in pure NumPy that agrees with
sklearn.tree.DecisionTreeRegressor on the same training data:

    (1) The recursive-split algorithm (Lecture 1, Section 5).
    (2) The best-split search using incremental running sums
        (Lecture 1, Section 4).
    (3) The MSE-impurity criterion (Lecture 1, Section 2).
    (4) Stopping rules: max_depth and min_samples_leaf
        (Lecture 1, Section 6).

The two trees will not always agree exactly because their tie-breaking
rules differ; we test against sklearn's predictions, with a generous
tolerance, on a small enough problem that the trees agree on the
structurally significant splits.

Estimated time: 70 minutes.

Run with:    python exercise-01-tree-from-scratch.py
Or test:     pytest exercise-01-tree-from-scratch.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-tree-from-scratch.py succeeds.

The exercise uses sklearn's bundled diabetes dataset (no network).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Tree node
# -----------------------------------------------------------------------------

@dataclass
class Node:
    """A node of the binary regression tree.

    A leaf has is_leaf=True and prediction set; left/right/feature/threshold
    are unused. An internal node has is_leaf=False and feature/threshold
    set, with left/right pointing at the two children.
    """
    is_leaf: bool = False
    prediction: float = 0.0
    feature: int = -1
    threshold: float = 0.0
    left: Optional["Node"] = field(default=None, repr=False)
    right: Optional["Node"] = field(default=None, repr=False)


# -----------------------------------------------------------------------------
# Part A -- The best-split search
# -----------------------------------------------------------------------------

def best_split(X: np.ndarray, y: np.ndarray, min_samples_leaf: int) -> Optional[tuple[int, float, float]]:
    """Find the best (feature_index, threshold, gain) split for this node.

    Use the MSE-impurity criterion: the parent variance minus the weighted
    average of the children variances. The gain is non-negative; only return
    a split with strictly positive gain.

    Use the incremental running-sums trick from Lecture 1 Section 4:
    sort the rows by one feature, then sweep the threshold from left to
    right while maintaining running sums of y and y**2 on each side. This
    is what makes the search O(n log n) per feature rather than O(n^2).

    Respect min_samples_leaf: do not consider a split that leaves either
    side with fewer than min_samples_leaf rows.

    Skip splits between identical feature values (the threshold would be
    on top of a real row, which is ambiguous).

    Return (best_feature, best_threshold, best_gain) or None if no
    positive-gain split exists.
    """
    # TODO: implement the incremental-sums best-split search.
    # The reference implementation is in the HINT block at the bottom.
    raise NotImplementedError("Part A -- best_split")


# -----------------------------------------------------------------------------
# Part B -- The recursive builder
# -----------------------------------------------------------------------------

def build_tree(
    X: np.ndarray, y: np.ndarray,
    depth: int, max_depth: int, min_samples_leaf: int,
) -> Node:
    """Build a tree recursively.

    Stopping rules:
        - If depth >= max_depth, return a leaf.
        - If len(y) < 2 * min_samples_leaf, return a leaf (cannot split
          either side without violating min_samples_leaf).
        - If best_split returns None (no positive-gain split), return a
          leaf.

    Otherwise, take the best split and recurse on the two children.
    """
    node = Node()
    # TODO: implement the stopping rules and the recursive split.
    raise NotImplementedError("Part B -- build_tree")


# -----------------------------------------------------------------------------
# Part C -- The tree class
# -----------------------------------------------------------------------------

class TinyTreeRegressor:
    """A 90-line regression tree.

    Hyperparameters:
        max_depth         -- the longest root-to-leaf path.
        min_samples_leaf  -- the minimum rows per leaf.

    Methods:
        fit(X, y)         -- build the tree.
        predict(X)        -- predict a 1-D array of values.
        get_depth()       -- the depth of the fitted tree.
        get_n_leaves()    -- the leaf count.
    """

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 5) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root_: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TinyTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.root_ = build_tree(X, y, 0, self.max_depth, self.min_samples_leaf)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(self.root_, X[i]) for i in range(len(X))])

    def _predict_one(self, node: Node, x: np.ndarray) -> float:
        # TODO: walk the tree from root to leaf and return the leaf prediction.
        # At each internal node, go left if x[node.feature] <= node.threshold,
        # otherwise go right.
        raise NotImplementedError("Part C -- _predict_one")

    def get_depth(self) -> int:
        return _depth(self.root_)

    def get_n_leaves(self) -> int:
        return _n_leaves(self.root_)


def _depth(node: Optional[Node]) -> int:
    if node is None or node.is_leaf:
        return 0
    return 1 + max(_depth(node.left), _depth(node.right))


def _n_leaves(node: Optional[Node]) -> int:
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return _n_leaves(node.left) + _n_leaves(node.right)


# -----------------------------------------------------------------------------
# Glue -- compare to sklearn
# -----------------------------------------------------------------------------

def run_comparison() -> dict[str, object]:
    """Fit both trees on the diabetes dataset and return both RMSEs."""
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    ours = TinyTreeRegressor(max_depth=4, min_samples_leaf=10).fit(X_tr, y_tr)
    sk   = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10,
                                 random_state=RANDOM_STATE).fit(X_tr, y_tr)

    return {
        "ours_test_rmse": float(root_mean_squared_error(y_te, ours.predict(X_te))),
        "sk_test_rmse":   float(root_mean_squared_error(y_te, sk.predict(X_te))),
        "ours_depth":     ours.get_depth(),
        "sk_depth":       sk.get_depth(),
        "ours_n_leaves":  ours.get_n_leaves(),
        "sk_n_leaves":    sk.get_n_leaves(),
    }


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_best_split_returns_a_real_split() -> None:
    """On a trivially-splittable dataset, best_split must find the obvious split."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0,    1.0,    1.0,    9.0,    9.0,    9.0])
    out = best_split(X, y, min_samples_leaf=1)
    assert out is not None
    feat, thr, gain = out
    assert feat == 0
    assert 3.0 < thr < 4.0, f"threshold should split between x=3 and x=4, got {thr}"
    assert gain > 0


def test_min_samples_leaf_is_respected() -> None:
    """If min_samples_leaf is large enough, no split is possible."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 1.0, 9.0, 9.0])
    out = best_split(X, y, min_samples_leaf=3)
    # With 4 rows and min_samples_leaf=3, no split leaves >=3 rows on both sides.
    assert out is None


def test_tree_fits_and_predicts() -> None:
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    mdl = TinyTreeRegressor(max_depth=4, min_samples_leaf=10).fit(X_tr, y_tr)
    preds = mdl.predict(X_te)
    assert len(preds) == len(y_te)
    assert np.all(np.isfinite(preds))


def test_tree_depth_respects_max_depth() -> None:
    X, y = load_diabetes(return_X_y=True)
    mdl = TinyTreeRegressor(max_depth=3, min_samples_leaf=5).fit(X, y)
    assert mdl.get_depth() <= 3


def test_tree_beats_mean_predictor() -> None:
    X, y = load_diabetes(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    mdl = TinyTreeRegressor(max_depth=4, min_samples_leaf=10).fit(X_tr, y_tr)
    mean_rmse = float(root_mean_squared_error(y_te, np.full_like(y_te, y_tr.mean(), dtype=float)))
    our_rmse  = float(root_mean_squared_error(y_te, mdl.predict(X_te)))
    assert our_rmse < mean_rmse, f"tree {our_rmse:.2f} did not beat mean predictor {mean_rmse:.2f}"


def test_tree_within_a_few_percent_of_sklearn() -> None:
    """Our tree should be within ~15% of sklearn's RMSE on the diabetes dataset.

    The two trees use different tie-breakers and small implementation details
    diverge, so we allow ~15% slack rather than insisting on exact agreement.
    """
    out = run_comparison()
    ratio = out["ours_test_rmse"] / out["sk_test_rmse"]
    assert 0.85 <= ratio <= 1.15, (
        f"ours RMSE = {out['ours_test_rmse']:.2f}, sklearn RMSE = {out['sk_test_rmse']:.2f}, "
        f"ratio = {ratio:.2f} -- the two trees should agree to ~15%"
    )


def _run_all_tests() -> None:
    test_best_split_returns_a_real_split()
    test_min_samples_leaf_is_respected()
    test_tree_fits_and_predicts()
    test_tree_depth_respects_max_depth()
    test_tree_beats_mean_predictor()
    test_tree_within_a_few_percent_of_sklearn()
    print("OK -- exercise 1")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# best_split:
#     n, p = X.shape
#     if n < 2 * min_samples_leaf:
#         return None
#     parent_mean = float(y.mean())
#     parent_var = float(((y - parent_mean) ** 2).mean())
#     best = None
#     best_gain = 0.0
#     for j in range(p):
#         order = np.argsort(X[:, j], kind="mergesort")
#         xs = X[order, j]
#         ys = y[order]
#         sum_l, sum2_l, n_l = 0.0, 0.0, 0
#         sum_r = float(ys.sum())
#         sum2_r = float((ys ** 2).sum())
#         n_r = n
#         for i in range(n - 1):
#             v = float(ys[i])
#             sum_l  += v
#             sum2_l += v * v
#             n_l    += 1
#             sum_r  -= v
#             sum2_r -= v * v
#             n_r    -= 1
#             if n_l < min_samples_leaf or n_r < min_samples_leaf:
#                 continue
#             if xs[i] == xs[i + 1]:
#                 continue
#             var_l = sum2_l / n_l - (sum_l / n_l) ** 2
#             var_r = sum2_r / n_r - (sum_r / n_r) ** 2
#             weighted = (n_l / n) * var_l + (n_r / n) * var_r
#             gain = parent_var - weighted
#             if gain > best_gain:
#                 best_gain = gain
#                 best = (j, 0.5 * (xs[i] + xs[i + 1]), gain)
#     return best
#
# build_tree:
#     node = Node()
#     if depth >= max_depth or len(y) < 2 * min_samples_leaf:
#         node.is_leaf = True
#         node.prediction = float(np.mean(y))
#         return node
#     out = best_split(X, y, min_samples_leaf)
#     if out is None:
#         node.is_leaf = True
#         node.prediction = float(np.mean(y))
#         return node
#     j, t, _gain = out
#     left_mask = X[:, j] <= t
#     node.feature = j
#     node.threshold = t
#     node.left  = build_tree(X[left_mask],  y[left_mask],  depth + 1, max_depth, min_samples_leaf)
#     node.right = build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples_leaf)
#     return node
#
# _predict_one:
#     while not node.is_leaf:
#         node = node.left if x[node.feature] <= node.threshold else node.right
#     return float(node.prediction)
#
# -----------------------------------------------------------------------------
