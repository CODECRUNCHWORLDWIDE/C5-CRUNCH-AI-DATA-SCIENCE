# Lecture 1 — Decision Trees from Scratch

> **Outcome:** You can write down the impurity measures (Gini, entropy, MSE), explain the recursive-split algorithm in five lines of pseudocode, implement a decision-tree regressor in under 200 lines of NumPy, and verify it agrees with `sklearn.tree.DecisionTreeRegressor` to floating-point tolerance. You can read a fitted tree as a sequence of axis-aligned thresholds and recognize overfitting from depth and leaf count alone. After this lecture, "decision tree" stops being a black box and becomes a small piece of code you wrote.

Linear regression draws a hyperplane through the feature space. A decision tree draws axis-aligned rectangles. The two models have nothing in common geometrically, and very different things in common statistically — but the practical implication is striking: most tabular datasets, especially ones with thresholds, interactions, and missing values, are full of rectangles. That is why a single tuned `RandomForestRegressor` will often outperform a tuned `Ridge` on tabular data by a wide margin, and a tuned gradient-boosted-trees model will outperform the forest by a smaller margin.

This lecture builds the **single decision tree**. The next two lectures stack many of them.

We target **scikit-learn 1.5+**, **numpy 2.x**. The `DecisionTreeRegressor`, `DecisionTreeClassifier`, and `tree.export_text` / `plot_tree` APIs have been stable since 2017.

---

## 1. The model in one paragraph

A decision tree predicts a scalar `y` from a row of features `x` by routing `x` down a sequence of yes/no questions. Each internal node asks "is `x[feature_j] ≤ threshold_t`?" The two branches lead to either another internal node or to a **leaf**, which holds a single prediction: the mean of the training `y` values that landed in that leaf (regression), or the majority class (classification).

```text
                       is sqft ≤ 1800?
                       /            \
                     yes            no
                      |              |
            is age ≤ 30?       is OverallQual ≤ 7?
              /     \              /         \
           yes      no           yes         no
            |        |            |           |
          $185k    $145k        $260k        $385k
```

The model has three moving parts and no others:

1. **The questions** (which feature, which threshold).
2. **The tree structure** (which question follows which).
3. **The leaf predictions** (one number per leaf).

A tree fit is an algorithm for picking all three. The standard fit (the CART algorithm, due to Breiman et al., 1984) is **greedy**: at the root, pick the single question that most reduces impurity; recurse on the two children; stop when a node satisfies a stopping rule. The greedy algorithm is not globally optimal — finding the globally optimal binary tree is NP-hard — but it is fast, predictable, and the implementations in `sklearn`, XGBoost, and LightGBM all use variations of it.

---

## 2. Impurity: how mixed is this node?

The single piece of math you need is the **impurity** function. Impurity is a scalar attached to a node that summarizes how mixed the labels in that node are. Zero impurity means the labels are constant; maximal impurity means they are evenly mixed.

For **classification**, the two standard impurity functions are:

- **Gini impurity** of a node with class proportions `p_1, ..., p_K`:

  ```text
  G  =  1 − sum_k p_k²
     =  sum_{k ≠ l} p_k · p_l
  ```

  Zero when one class has probability 1; maximal at `1 − 1/K` when all classes are equally probable (`p_k = 1/K`).

- **Entropy** (Shannon, in nats or bits):

  ```text
  H  =  − sum_k p_k · log(p_k)
  ```

  Same zero, maximal at `log(K)` when uniform. By convention `0 · log(0) = 0`.

For **regression**, the standard impurity is the **mean squared error** of the node's `y` values around their mean:

```text
MSE_node  =  (1/|node|) · sum_{i ∈ node} (y_i − ȳ_node)²
```

Zero when every `y` in the node is identical; grows with the variance of `y` in the node.

> **EXPERIMENT — Gini vs entropy on the same split.** Generate 100 binary labels with `p = 0.3` of class 1, then a candidate split into a left node with `p_left = 0.1` (40 rows) and a right node with `p_right = 0.5` (60 rows). Compute Gini and entropy for the parent and the weighted Gini / entropy after the split. The two impurity measures will rank this split essentially the same way (the weighted-impurity drop is positive for both). The two disagree on fewer than 2% of candidate splits in practice; the choice between them rarely changes the tree.

---

## 3. Split gain: the recursive question

Given an impurity function, the **split gain** of a candidate split `(feature j, threshold t)` is the drop in weighted impurity from parent to children:

```text
gain  =  I(parent)  −  (n_L / n) · I(left)  −  (n_R / n) · I(right)
```

where `n_L` and `n_R` are the row counts on the two sides and `n = n_L + n_R`. The CART algorithm at each node picks the split that maximizes `gain`. (Equivalently, it minimizes the weighted-impurity of the children.) Splits with zero or negative gain are not made; we will see in Section 6 how that connects to the `min_impurity_decrease` stopping rule.

For **regression**, the impurity is MSE and the equivalent form of the gain is more interpretable:

```text
gain_regression  =  Var(parent)  −  (n_L / n) · Var(left)  −  (n_R / n) · Var(right)
```

where `Var(node) = mean((y − ȳ)²)`. The intuition: the split is good if the two children have less within-node variance than the parent did. (The total variance is conserved if and only if the split is uninformative; any positive gain is exactly the "between-children" variance the split has exposed.)

---

## 4. The best-split search

Given a node and `p` features, how do you find the single best `(j, t)` pair?

The naive answer is to iterate over every feature `j` and every possible threshold `t`. For a continuous feature with `n` rows in the node, there are `n − 1` candidate thresholds (the midpoints between consecutive sorted values). The total cost per node is `O(p · n · log(n))` from the sort, plus `O(p · n)` from the linear scan over thresholds.

The standard implementation uses a smarter incremental update:

```text
sort the rows in the node by feature j
maintain running sums for the left half:
    sum_y_left   = 0
    sum_y2_left  = 0
    n_left       = 0
maintain running sums for the right half (start with the whole node):
    sum_y_right  = sum_y_total
    sum_y2_right = sum_y2_total
    n_right      = n

for each candidate threshold (sweep left to right):
    move one row from right to left:
        sum_y_left   += y_i
        sum_y2_left  += y_i²
        n_left       += 1
        sum_y_right  -= y_i
        sum_y2_right -= y_i²
        n_right      -= 1

    compute MSE_left and MSE_right from the running sums:
        MSE_left  = sum_y2_left / n_left  - (sum_y_left / n_left)²
        MSE_right = sum_y2_right / n_right - (sum_y_right / n_right)²

    compute the weighted-children MSE; track the threshold with the best gain.
```

The running sums let you evaluate every candidate threshold in `O(1)`, so the cost per feature is `O(n log n)` from the sort plus `O(n)` from the sweep. That is the algorithm in `sklearn`'s tree builder, with minor variations for handling missing values and `min_samples_leaf`.

> **EXPERIMENT — best split by hand.** Take ten rows with one feature `x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` and `y = [1, 1, 1, 2, 2, 2, 2, 9, 9, 9]`. By eye, the best split is at `x ≤ 7` (the y jumps from 2 to 9 there). Verify: compute MSE_left and MSE_right for the splits at `x ≤ 3.5`, `x ≤ 7.5`, and `x ≤ 9.5`. The minimum weighted-children MSE is at `x ≤ 7.5`. That is the threshold the tree builder will pick.

For **categorical** features with no natural ordering, the search is more delicate. With `K` categories there are `2^(K-1) − 1` non-trivial subsets to consider. For regression with squared error, the optimal subset can be found in `O(K log K)` by sorting the categories by their mean `y` and sweeping in order (a Fisher 1958 result). LightGBM and the newer scikit-learn / XGBoost paths use exactly this trick when `categorical_features` is set. For classification with more than two classes, the trick does not apply and most libraries one-hot-encode or use a fixed-bin approximation.

---

## 5. The recursive algorithm

The full tree-fit, in five lines of pseudocode:

```text
fit(node, X, y):
    if stop(node, X, y):
        node.is_leaf = True
        node.prediction = mean(y)            # regression; majority(y) for classification
        return
    (j, t) = best_split(X, y)
    if gain(j, t) <= 0:
        node.is_leaf = True
        node.prediction = mean(y)
        return
    node.feature, node.threshold = j, t
    left_mask = X[:, j] <= t
    fit(node.left,  X[left_mask],  y[left_mask])
    fit(node.right, X[~left_mask], y[~left_mask])
```

That is the whole algorithm. Everything else is engineering: the `stop` rule, the incremental running sums in `best_split`, the categorical-feature handling, the missing-value handling, the histogram binning (a different lecture).

A working NumPy implementation is about 150 lines. Exercise 1 walks you through writing it.

---

## 6. Stopping rules: when to make a leaf

Without stopping rules, the recursive algorithm fits the training set perfectly: it keeps splitting until every leaf has one row. That is the most-overfit tree imaginable. Five knobs control when to stop:

| Hyperparameter | What it does | Sensible default |
|----------------|--------------|------------------|
| `max_depth` | Maximum root-to-leaf depth. | 4–8 for tabular data; `None` is "no limit" and is rarely right. |
| `min_samples_split` | A node must have at least this many rows before it is even considered for splitting. | 2 (sklearn default); raise to 20+ for small data. |
| `min_samples_leaf` | Each leaf must contain at least this many training rows. | 1 (sklearn default); raise to 5–20 for tabular data. |
| `min_impurity_decrease` | A split is only made if its gain exceeds this value. | 0.0 (sklearn default); rarely tuned. |
| `max_leaf_nodes` | Cap the total number of leaves. The tree is grown best-first instead of depth-first when this is set. | `None`; the sklearn / LightGBM `num_leaves` analogue. |

The single most important knob is `max_depth`. The single most under-used is `min_samples_leaf`. A `max_depth=None` tree on the diabetes dataset will memorize the training set and achieve train R² = 1.0 with test R² well below 0.3; the same tree with `max_depth=4` will have train R² around 0.55 and test R² around 0.40 — much better generalization.

> **EXPERIMENT — overfitting a tree on purpose.** Load `sklearn.datasets.load_diabetes()`. Split 80/20. Fit `DecisionTreeRegressor()` with default hyperparameters (no `max_depth`). Print `mdl.get_depth()` (probably 20+), `mdl.get_n_leaves()` (probably 200+), and the train and test R² (probably 1.00 and 0.05). Re-fit with `max_depth=4`. Print the same numbers. The train R² drops to ~0.55; the test R² rises to ~0.40. That is the bias-variance tradeoff in five lines of code.

---

## 7. The minimal Python implementation

The following 90 lines fit a regression tree by the algorithm in Sections 3–6. It is what you build in Exercise 1, with two small simplifications: only continuous features, and the stopping rule is just `max_depth` and `min_samples_leaf`.

```python
"""A 90-line decision-tree regressor in pure NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Node:
    is_leaf: bool = False
    prediction: float = 0.0
    feature: int = -1
    threshold: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None


class TinyTreeRegressor:
    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 5) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root_: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TinyTreeRegressor":
        self.root_ = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node()
        # Stopping rules:
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            node.is_leaf = True
            node.prediction = float(np.mean(y))
            return node
        # Find the best split (incremental sums).
        best = self._best_split(X, y)
        if best is None:
            node.is_leaf = True
            node.prediction = float(np.mean(y))
            return node
        j, t, _gain = best
        left_mask = X[:, j] <= t
        node.feature = j
        node.threshold = t
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        parent_var = float(np.var(y))
        best = None
        best_gain = 0.0
        for j in range(p):
            order = np.argsort(X[:, j])
            xs = X[order, j]
            ys = y[order]
            # Incremental sums on the left side (start empty).
            sum_l, sum2_l, n_l = 0.0, 0.0, 0
            sum_r = float(ys.sum())
            sum2_r = float((ys ** 2).sum())
            n_r = n
            for i in range(n - 1):
                v = float(ys[i])
                sum_l  += v
                sum2_l += v * v
                n_l    += 1
                sum_r  -= v
                sum2_r -= v * v
                n_r    -= 1
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                if xs[i] == xs[i + 1]:           # cannot split between equal values
                    continue
                # MSE = E[y^2] - E[y]^2 by node.
                var_l = sum2_l / n_l - (sum_l / n_l) ** 2
                var_r = sum2_r / n_r - (sum_r / n_r) ** 2
                weighted = (n_l / n) * var_l + (n_r / n) * var_r
                gain = parent_var - weighted
                if gain > best_gain:
                    best_gain = gain
                    best = (j, 0.5 * (xs[i] + xs[i + 1]), gain)
        return best

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(self.root_, X[i]) for i in range(len(X))])

    def _predict_one(self, node: Node, x: np.ndarray) -> float:
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction
```

That is the entire model. Test it against `DecisionTreeRegressor` and the two agree on most splits (occasional ties go differently because sklearn's tie-breaker is different — both answers are correct). Exercise 1 makes this rigorous.

---

## 8. Reading a fitted tree

Once a tree is fit, you can read it. The single most useful tool is `sklearn.tree.export_text`:

```python
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(as_frame=True, return_X_y=True)
mdl = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
print(export_text(mdl, feature_names=list(X.columns)))
```

The output is:

```text
|--- MedInc <= 5.04
|   |--- MedInc <= 3.07
|   |   |--- Latitude <= 38.48
|   |   |   |--- value: [1.70]
|   |   |--- Latitude >  38.48
|   |   |   |--- value: [1.27]
|   |--- MedInc >  3.07
|   |   |--- HouseAge <= 38.5
|   |   |   |--- value: [2.16]
|   |   |--- HouseAge >  38.5
|   |   |   |--- value: [1.95]
|--- MedInc >  5.04
|   |--- ...
```

You can read the tree top to bottom as "first split on income, then on a second feature, then assign the leaf mean as the prediction." The depth-3 tree is interpretable in a way the deeper trees are not. For *prediction*, deeper is usually better; for *interpretation*, the first three to five levels are the part you can defend in prose.

`sklearn.tree.plot_tree` draws the same tree as a graphical figure. For trees with `max_depth ≤ 4` the figure is readable; deeper than that, prefer `export_text`.

---

## 9. What a tree does well, and where it fails

A single decision tree gets three things right:

1. **Axis-aligned thresholds.** "Is `sqft > 1800`?" is exactly the question a real-estate agent would ask. Every threshold the tree finds is interpretable as a piecewise-constant rule.
2. **Interactions for free.** A tree that splits on `sqft` and then on `OverallQual` inside the `sqft > 1800` branch has discovered the *interaction* between the two features without being told to look for one. Linear models need explicit interaction terms; trees find them in the recursive structure.
3. **Categorical features.** With native categorical support (Section 4), a tree splits on subsets of categories directly. No one-hot encoding required, no cardinality explosion.

A single tree gets three things wrong:

1. **High variance.** Re-fit on a slightly different bootstrap of the training set and the top split may change entirely. The whole rest of the tree changes downstream. This is the variance-reduction problem random forests (Lecture 2) and gradient boosting (Lecture 3) are designed to solve.
2. **Step-function predictions.** A tree's output is piecewise constant. The prediction surface jumps at every threshold. For smooth target functions (real housing prices, say) the steps are an obvious model-misspecification.
3. **Extrapolation.** A tree can only predict the means of leaves it built. It will never predict a value larger than the largest leaf mean in the training set, no matter how extreme the input. Linear models extrapolate freely (sometimes badly); trees do not extrapolate at all.

The first weakness — high variance — is the one ensembles fix. The next two lectures are about ensembling.

> **EXPERIMENT — single tree, two seeds.** Fit `DecisionTreeRegressor(max_depth=8, random_state=42)` and `DecisionTreeRegressor(max_depth=8, random_state=7)` on the same California Housing training set. Print `mdl.get_n_leaves()` (probably identical or close), and `mdl.predict(X_test[:10])` for both. The predictions will differ by a noticeable margin on some rows; on rows that landed in different leaves between the two trees, the difference is the cost of high variance in a single tree.

---

## 10. Classification vs regression — what changes

The lecture has been written for regression because the rest of the week (and the mini-project) is a regression problem on Ames. For **classification**, only two things change:

1. **The impurity function** is Gini or entropy instead of MSE.
2. **The leaf prediction** is the majority class (for "hard" predictions) or the empirical class proportions (for `predict_proba`).

The rest — the recursive algorithm, the stopping rules, the best-split search — is identical. `sklearn.tree.DecisionTreeClassifier` and `DecisionTreeRegressor` share most of their implementation for exactly this reason.

For classification, `predict_proba` returns the class proportions in the leaf the row landed in. With small leaves (`min_samples_leaf=1`), those proportions are 0 or 1 — calibrated probabilities are unattainable from a single small-leaf tree. This is one reason `CalibratedClassifierCV` is sometimes wrapped around tree-based classifiers when probability quality matters.

---

## 11. Tree depth, leaf count, and the cost of memorization

Two diagnostic numbers to record after every tree fit:

- **`mdl.get_depth()`** — the longest root-to-leaf path. With `max_depth=None`, this can grow to `log2(n)` at minimum (perfectly balanced tree) and `n − 1` at maximum (one-row-per-leaf, fully unbalanced).
- **`mdl.get_n_leaves()`** — the leaf count. Equal to the number of training rows the tree memorizes, when `min_samples_leaf=1`.

A tree with `n_leaves > n_train / 3` has memorized a substantial fraction of the training set; its test R² is almost certainly far below its train R². A tree with `n_leaves` in the low double digits is generalizing — and is also the part of the tree you can read in prose.

```python
print(f"depth     {mdl.get_depth()}")
print(f"n_leaves  {mdl.get_n_leaves()}")
print(f"train R²  {mdl.score(X_tr, y_tr):.3f}")
print(f"test  R²  {mdl.score(X_te, y_te):.3f}")
```

A test R² much smaller than the train R² is the classic high-variance signature. The remedies are the stopping rules (Section 6) or — much more effectively — an ensemble (Lectures 2 and 3).

---

## 12. The checklist for a single tree that ships

Before you ship a single decision tree (rare in 2026; usually you ship an ensemble), walk this list:

- [ ] **`max_depth` is set.** `None` is rarely the right answer.
- [ ] **`min_samples_leaf` is set.** Default 1 is too aggressive on small data.
- [ ] **`random_state` is set.** Trees are deterministic given the data and the random state, but `random_state=None` makes runs non-reproducible.
- [ ] **`depth` and `n_leaves` reported.** Both go in the experiment card.
- [ ] **Train R² and test R² reported, side by side.** A large gap is a high-variance signal.
- [ ] **`export_text` saved.** Even if you do not interpret it, having a printable version of the model in your repo is part of the C5 discipline.
- [ ] **Compared to a `DummyRegressor` baseline.** A tree that does not beat the mean predictor is a tree that should be debugged.
- [ ] **Compared to a `LinearRegression` baseline.** On most tabular data a tuned linear model is competitive with a single tree; the tree's case for itself shows up when you ensemble.

---

## 13. Where this leaves you

You can now derive Gini, entropy, and MSE-impurity from their definitions; explain the recursive-split algorithm in five lines; implement a regression tree in 90 lines of NumPy that agrees with `sklearn.tree.DecisionTreeRegressor` to a few floating-point ULPs on most splits; read a fitted tree as a sequence of thresholds; recognize overfitting from depth and leaf count alone.

A single tree is rarely the model you ship. It is the unit of work that the next two lectures stack: bagging (Lecture 2) averages many trees to reduce variance; gradient boosting (Lecture 3) fits successive trees to the residuals of the previous ensemble. Both algorithms reduce to "fit a tree" inside a loop. The tree you built in this lecture is the inside of that loop.

The mini-project at the end of the week takes the same Ames Housing dataset Week 4 used and asks: **can a gradient-boosted-trees model beat the Week 4 `RidgeCV` baseline by ≥10% relative RMSE, and can you explain why in two paragraphs**? That is the test the rest of the week prepares you for.
