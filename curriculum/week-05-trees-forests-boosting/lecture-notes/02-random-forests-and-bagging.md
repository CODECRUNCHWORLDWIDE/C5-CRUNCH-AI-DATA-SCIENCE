# Lecture 2 — Random Forests and Bagging

> **Outcome:** You can explain bagging in one paragraph, write the variance-of-a-mean argument that motivates it, derive the random-forest extra wrinkle (column subsampling at each split), use the out-of-bag estimate as free cross-validation, tune `n_estimators`, `max_features`, and `min_samples_leaf` deliberately, and recognize the two situations where a random forest beats a single tree and the one situation where it cannot. After this lecture, "random forest" stops being a sklearn one-liner and becomes a small set of statistical claims you can defend.

A single decision tree is high variance. Re-bootstrap the training set and the top split may change entirely, and everything downstream changes with it. The cure is to fit many trees and average them — that is **bagging**, due to Breiman in 1996 — with one extra de-correlating trick on top, which makes it a **random forest** (Breiman, 2001).

The two algorithms are short. The intuitions matter more than the code.

We target **scikit-learn 1.5+**, **numpy 2.x**. `RandomForestRegressor`, `RandomForestClassifier`, and `BaggingRegressor` / `BaggingClassifier` are stable APIs and will be the right answer for the rest of the decade.

---

## 1. The model in one paragraph

A random forest predicts a scalar `y` (or a class) by averaging the predictions of `B` decision trees, each fit on a **bootstrap sample** of the training data, each restricted to a random **subset of features** at each split. The averaging reduces variance; the bootstrap and the feature subsampling de-correlate the trees so that the averaging *can* reduce variance.

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=500,        # the number of trees
    max_features="sqrt",     # column-subsampling at each split (sklearn default is 1.0 for regression; we override)
    min_samples_leaf=5,      # smooth each tree slightly
    max_depth=None,          # let each tree grow deep
    n_jobs=-1,               # parallelize across all cores
    random_state=42,
).fit(X_train, y_train)
```

`fit` is `O(B · n · log n · p)` with `B` trees, `n` rows, `p` features; `predict` is `O(B · log n)` (each tree is a tree-walk). For Ames-sized data both finish in a few seconds.

---

## 2. Bagging: the variance argument

Suppose you have a stochastic estimator `f̂(x; D)` — a model whose predictions depend on the training set `D` — and you draw `B` independent training sets `D_1, ..., D_B` from the data-generating distribution. Each fit gives you a prediction `f̂(x; D_b)`, and the **bagged prediction** is the mean:

```text
f̂_bag(x)  =  (1 / B) · sum_{b=1..B} f̂(x; D_b)
```

The variance of `f̂_bag(x)` over draws of the `B` training sets is

```text
Var(f̂_bag(x))  =  (1 / B²) · sum_{b=1..B} Var(f̂(x; D_b))  +  cross-terms
              =  Var(f̂(x; D)) / B          # if the B fits are independent
```

The "if the B fits are independent" caveat is doing a lot of work. In practice you only have one training set, so you cannot draw `B` independent training sets — you draw `B` **bootstrap** samples (sample with replacement from the one training set you have). Bootstrap samples are not independent of each other; they share a lot of rows. So the actual variance reduction is

```text
Var(f̂_bag(x))  =  ρ · Var(f̂(x; D))  +  (1 − ρ) · Var(f̂(x; D)) / B
```

where `ρ` is the pairwise correlation between predictions from two bootstrap-trained models. The first term does not vanish as `B → ∞`; the second does. The strategy for reducing the bagged variance is therefore to **decrease `ρ`** — make the trees as different from each other as possible. The random-forest column-subsampling trick (Section 4) does exactly that.

> **EXPERIMENT — bagging reduces variance.** Generate a small regression dataset (n=300). Fit 50 separate `DecisionTreeRegressor(max_depth=None)` instances, each on a different `random_state` (which produces different bootstrap-like splits). Predict each on a fixed validation set. The predictions for a single test row will scatter widely. Take the average prediction across the 50 fits; the variance of *that* prediction across re-runs of the experiment is much smaller. The reduction factor is roughly `1 / (50 · effective_independence)` — close to 1/50 if the trees are well-decorrelated, much worse if they all agree.

---

## 3. The bootstrap sample and the out-of-bag fraction

A **bootstrap sample** of size `n` is drawn from a dataset of size `n` by sampling with replacement. Two facts about it:

1. The expected number of **distinct** rows in a bootstrap sample is `n · (1 − (1 − 1/n)^n) ≈ n · (1 − 1/e) ≈ 0.632 · n`. About 63% of the rows appear at least once.
2. The remaining ~37% of rows are **out-of-bag** for that bootstrap sample — they never appear in the training of that tree.

The OOB rows are the bagging algorithm's free validation set. For each row `i`, you can collect the predictions of all the trees that did *not* see row `i` in their bootstrap, average them, and compare to `y_i`. That average is an **unbiased estimate of the generalization error** of the bagged predictor — no separate validation set required. Scikit-learn computes this automatically when you set `oob_score=True`:

```python
rf = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    oob_score=True,
    n_jobs=-1,
    random_state=42,
).fit(X_train, y_train)

print(f"OOB R²: {rf.oob_score_:.3f}")          # the OOB estimate of R²
```

The OOB estimate is roughly equivalent to a 3-fold cross-validation estimate, with no extra fit time. For small datasets (Ames-sized) it is a near-free diagnostic.

> **EXPERIMENT — OOB tracks held-out R² closely.** Fit a `RandomForestRegressor(oob_score=True)` on 80% of the diabetes dataset; compute `rf.oob_score_` and `rf.score(X_test, y_test)`. The two numbers will agree to within 0.01–0.03 on this dataset. On larger datasets the agreement is tighter. The OOB estimate is what you watch during early experimentation; the held-out test score is what you commit to at the end.

---

## 4. The random-forest wrinkle: column subsampling

Bagging alone produces trees that are heavily correlated, because the top split in each tree is almost always on the same feature (whichever has the largest split gain at the root). Breiman's 2001 paper introduced the small twist that makes random forests work: at each split inside each tree, consider only a **random subset of `m` features** out of the `p` total.

```text
for each tree b in 1..B:
    bootstrap sample D_b
    for each internal node:
        sample m features uniformly without replacement
        find the best (feature, threshold) split using only those m features
        recurse
```

The hyperparameter is `max_features`. Standard recommendations:

- **Classification:** `max_features = sqrt(p)`.
- **Regression:** `max_features = p / 3`.

`scikit-learn`'s defaults follow the convention (sort of). As of sklearn 1.0+ the regression default is `max_features=1.0` (use all features), which is *not* the Breiman recommendation. For regression problems you almost always want to override to `max_features="sqrt"` or `max_features=0.33`. The classification default is `max_features="sqrt"`, which matches Breiman.

The effect of `max_features < p`:

- The top split in each tree is no longer always on the most-informative feature. The trees diversify at the root.
- The pairwise correlation `ρ` between tree predictions drops.
- The bagged variance drops further. Out-of-bag R² typically improves by 0.02–0.10.

The cost: each individual tree is slightly *worse* than a bagged-only tree, because it sometimes makes a suboptimal split. The ensemble is better because the trees are more diverse. This is the "wisdom of the crowd" argument: a forest of mostly-mediocre, mostly-uncorrelated voters outperforms a forest of identical experts.

> **EXPERIMENT — `max_features` sweep on diabetes.** Fit a `RandomForestRegressor(n_estimators=500, oob_score=True)` on the diabetes dataset for `max_features ∈ {1, 2, 4, "sqrt", "log2", 1.0}`. Plot OOB R² as a function of `max_features`. The curve has an interior maximum around `sqrt(10) ≈ 3` features. At `max_features=1` the trees are too diverse and individually weak; at `max_features=p` (the default) the trees are too correlated. The sweet spot is in between, which is the entire reason `max_features` is a hyperparameter.

---

## 5. The four hyperparameters that matter

A random forest has about a dozen hyperparameters. On Ames-sized data, four matter:

| Hyperparameter | What it does | Sensible default |
|----------------|--------------|------------------|
| `n_estimators` | The number of trees in the forest. More is monotonically (almost) better up to a plateau; beyond ~500 the curve is flat. | 500. Bump to 1000 only if `oob_score` is still climbing. |
| `max_features` | The fraction of features considered at each split. The de-correlation knob. | `"sqrt"` for classification; `0.33` or `"sqrt"` for regression. |
| `min_samples_leaf` | The minimum rows per leaf. Bigger values smooth each tree. | 1 for classification, 5 for regression on tabular data. |
| `max_depth` | The depth limit per tree. `None` is often fine because the bagging already controls variance — but set it if individual trees are blowing memory. | `None`, with a backstop of 30 on huge datasets. |

The other knobs (`min_samples_split`, `min_impurity_decrease`, `max_leaf_nodes`, `bootstrap=False` for "subagging") rarely move the needle on small tabular data. Set the four above; sweep one at a time; do not touch the rest until they are necessary.

One non-obvious fact: increasing `n_estimators` cannot overfit a random forest. More trees gives the average more samples; it does not give the ensemble more flexibility. (This is in sharp contrast to gradient boosting, where more trees absolutely can overfit. Lecture 3.) The practical implication: pick `n_estimators` as the largest value your wall-clock budget allows, and stop tuning it.

---

## 6. When a forest beats a tree

A random forest beats a single decision tree on essentially every tabular dataset with `n ≥ 100`. The cases where the gap is most dramatic:

1. **High-variance datasets** — small `n`, noisy target, many features. The single-tree variance is the bottleneck; the bagged variance is much lower.
2. **Datasets with many weak features.** A single tree's top split is whichever feature happens to have the largest split gain on the bootstrap. With column subsampling, weak features get a chance to contribute lower in the tree.
3. **Datasets with interactions.** Trees find interactions for free (Lecture 1). Forests do too, on average across the ensemble.

And one case where it does *not* beat a single tree by much, or at all:

- **Very low-dimensional, smooth target functions.** If `p = 2` and `y` is a smooth function of `x_1` and `x_2`, the single tree's piecewise-constant fit and the forest's averaged-piecewise-constant fit both lose to a polynomial regression or a smoothing spline. Trees are the wrong tool for smooth low-dimensional regression; they are the right tool for tabular data with thresholds, interactions, and mixed feature types.

---

## 7. Feature importance from a forest

Every fitted tree gives you a per-feature importance score: the total impurity reduction (Gini decrease, MSE decrease) attributable to each feature, summed over splits in the tree. Averaged across the trees of a forest, this gives a per-feature importance from a forest fit.

```python
import pandas as pd

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).head(10)
```

Two warnings about Gini / split-gain importance from a tree-based model:

1. **It is biased toward high-cardinality features.** A continuous feature has many possible thresholds; a binary feature has one. The continuous feature gets more chances to split and so accumulates more impurity reduction even when it carries no real signal.
2. **It does not reflect generalization.** The impurity reduction is measured on the training set inside each tree. It can be inflated by overfitting.

The honest alternative is **permutation importance** (Lecture 3, Section 6, and Challenge 1):

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
```

Permutation importance measures the drop in held-out score when one feature's values are randomly permuted; it is bias-free with respect to feature cardinality, and it measures generalization, not training fit. The cost is `n_features × n_repeats` predict passes — on a forest this is the dominant runtime, but still seconds on Ames-sized data.

In practice, the Gini importance ranking and the permutation-importance ranking agree on the top 5 features ~80% of the time and disagree on the bottom features ~50% of the time. The top is the part you care about; the bottom is noise from cardinality bias. We pull this thread further in Challenge 1.

> **EXPERIMENT — bias of Gini importance.** Create a synthetic dataset with one strong continuous feature (`x1 = N(0,1)`, `y = x1 + ε`), one strong binary feature (`x2 = Bernoulli(0.5)`, with `y` shifted by `+0.5 · x2`), and one *random noise continuous* feature (`x3 = N(0,1)`, no relationship to `y`). Fit `RandomForestRegressor(n_estimators=500)`. Print `feature_importances_`. The noise feature `x3` gets a non-trivial Gini importance (often higher than the binary `x2`) because it has more candidate thresholds. Now compute permutation importance on a held-out set. `x3`'s permutation importance is zero (within noise) and the binary `x2`'s permutation importance reflects its true effect on `y`. The two rankings disagree on `x2` and `x3` precisely because Gini importance is cardinality-biased.

---

## 8. Random forests vs. gradient boosting (the short version)

Both are tree ensembles. Both are competitive on tabular data. The differences:

| | Random Forest | Gradient Boosting |
|-|--------------|-------------------|
| **Trees are fit** | independently, in parallel | sequentially, each on the residuals of the previous |
| **Tree depth** | usually deep (`max_depth=None`) | usually shallow (`max_depth=3-8`) |
| **`learning_rate`** | does not exist | the key knob; usually 0.01–0.1 |
| **`n_estimators`** | bigger is monotonically (almost) better | a hyperparameter that can overfit; use early stopping |
| **Variance** | low | depends on hyperparameters; with low LR, low |
| **Bias** | moderate | can be very low with enough rounds |
| **Wall-clock to train** | fast, parallelizable | slower, mostly sequential |
| **Out-of-the-box on Ames-sized data** | within 1–3% of a tuned GBM | within 0–2% of a tuned RF, often slightly better |

The practical rule: **start with a random forest** for quick model-baseline work because the defaults are forgiving and the OOB estimate gives you a generalization number for free. **Switch to gradient boosting** when you are tuning hyperparameters carefully and have a validation set. On most tabular regressions a tuned gradient-boosted-trees model beats a tuned forest by 1–5% relative RMSE; on a few problems the forest wins.

We cover gradient boosting in detail in Lecture 3.

---

## 8b. The bootstrap, in more detail

The bagging variance argument in Section 2 glossed over what a bootstrap sample actually is. Spelling it out:

A bootstrap sample of size `n` from a dataset `D = {(x_1, y_1), ..., (x_n, y_n)}` is constructed by drawing `n` integer indices, each uniformly from `{1, ..., n}`, **with replacement**. The resulting sample contains roughly `0.632 · n` distinct rows of `D` and roughly `n − 0.632 · n ≈ 0.368 · n` duplicates.

Why `0.632`? The probability a specific row `i` is *not* drawn on a single pull is `1 − 1/n`. The probability it is not drawn in any of `n` pulls is `(1 − 1/n)^n`. As `n → ∞`, that tends to `e^{-1} ≈ 0.368`. So each row has probability `1 − 1/e ≈ 0.632` of appearing at least once in a bootstrap sample, and the expected number of distinct rows is `0.632 · n`.

The remaining `0.368 · n` rows are the **out-of-bag** rows for that bootstrap. They are the rows the tree never saw. They are also, statistically, an iid sample from the same distribution as the training data — which is what makes the OOB prediction (Section 3) an unbiased estimate of generalization error.

Two small implementation details:

1. **Same-sized bootstrap.** The bootstrap sample has size `n` (not `0.632 · n`); the duplicates fill out the count. Scikit-learn's `bootstrap=True` (default) follows this convention. Setting `bootstrap=False` turns "bagging" into "subagging" — sampling without replacement — and gives up the OOB estimate.
2. **`max_samples`.** Some implementations let you draw a bootstrap of size `m < n` to speed things up. Sklearn's `max_samples` parameter does this for `RandomForestRegressor`. On Ames-sized data, leave it at the default of `None` (use the full bootstrap); for `n > 10^6` rows, `max_samples=0.1` is a 10× speed-up at minimal cost in accuracy.

> **EXPERIMENT — confirm the 0.632 number.** Sample `n = 1000` integers from `range(1000)` with replacement. Count the distinct integers. The answer is consistently between 620 and 640. Repeat for `n = 10` — the answer scatters more (300–700%) but averages to ~6.3 distinct rows. The `0.632` constant is `1 − 1/e`; small `n` shows the deviation from the asymptotic value.

---

## 9. The trees-do-not-extrapolate trap

A property of every tree-based model — including random forests, gradient-boosted trees, and the histogram-boosted variants — is that **they cannot extrapolate**. The prediction surface is the average of leaf means; the maximum prediction is the maximum leaf mean in the training set; the minimum prediction is the minimum leaf mean. No matter how extreme the input, the prediction is bounded by the training-set leaf means.

For most prediction problems this is fine. For some — extrapolating a price into a future regime, predicting a quantity that grows monotonically with a feature, modeling a multiplicative phenomenon — it is a problem.

Concrete example: fit a forest on housing data from 2010–2020 and predict 2030 prices. Real inflation alone says 2030 prices are likely higher than any 2020 price. The forest's maximum prediction is bounded by the 2020 leaf means. The forest will systematically under-predict 2030. A linear model with `year` as a feature extrapolates the trend; the tree-based model does not.

The honest fix: when you need extrapolation, either model `log(y)` (multiplicative effects become additive, and the model's bounds in log-space cover a wider range in original-space), or use a model that extrapolates (linear regression, splines, neural networks).

> **EXPERIMENT — the extrapolation cap.** Generate `n = 200` rows of `y = 2 * x + noise` for `x ∈ [0, 10]`. Fit `RandomForestRegressor(n_estimators=200)`. Predict on a test set with `x ∈ [10, 20]` (out of the training range). The forest's predictions saturate at the maximum leaf mean — roughly `2 * 10 + ε`, around 20. The true values extend to 40. A `LinearRegression()` on the same data predicts ~40 at `x = 20`. This is the extrapolation cap of tree-based methods, in one plot.

---

## 9b. Random-forest classification, briefly

The lecture has been written for regression because the rest of the week is a regression problem on Ames. For classification, only two things change:

1. **Each tree votes for a class** (or returns class probabilities). The forest's hard prediction is the majority vote; its `predict_proba` is the average of the per-tree class proportions.
2. **Class imbalance** matters more than it does for regression. `RandomForestClassifier(class_weight="balanced")` re-weights each class so they contribute equally to the loss; `class_weight="balanced_subsample"` re-weights each bootstrap sample independently. On a 5%-positive dataset, the unweighted forest will produce mostly-negative-class predictions and a low PR-AUC; the balanced forest produces a more honest precision-recall tradeoff.

The OOB estimate, the column-subsampling story, and the four-hyperparameters list all transfer unchanged. The metric you report changes (see Week 4, Lecture 3 — precision, recall, F1, ROC-AUC, PR-AUC); the model does not.

One non-obvious point: `RandomForestClassifier(predict_proba)` outputs averaged-leaf-proportions, *not* calibrated probabilities. A forest with `min_samples_leaf=1` and deep trees will return mostly 0.0 or 1.0 probabilities from each tree, and the average across trees ends up smoothed but not calibrated. If you need calibrated probabilities, wrap with `CalibratedClassifierCV(method="isotonic")` on a held-out set.

---

## 9c. Parallelization, memory, and wall-clock cost

A random forest is **embarrassingly parallel**: each tree is fit on a different bootstrap sample with no shared state. Scikit-learn parallelizes this through `joblib`; setting `n_jobs=-1` uses every available CPU core. On a laptop with 8 cores, fitting a 500-tree forest on Ames-sized data takes 2–5 seconds; on the same data with `n_jobs=1`, 15–40 seconds.

Memory cost is the dominant scaling concern. Each tree is stored as a set of NumPy arrays: one row per node, with the feature index, threshold, child pointers, and leaf prediction. For a tree of depth ~20 on 10k rows, that is on the order of `2^20 ≈ 10^6` nodes per tree, which at 64 bytes per node is ~64 MB per tree, ~32 GB for 500 trees. The trees are usually much shallower than `max_depth=None` would allow in principle, so actual memory is much less, but `n_estimators=10000, max_depth=None` on a large dataset can blow your RAM. The fix: set `max_depth` to 20–30 as a backstop, or `max_leaf_nodes` to 1000.

For predict time, the forest is `O(B · log n)` per row — fast but proportional to the number of trees. If predict latency matters at deploy time (a recommender, an ad-bid system), gradient boosting with shallow trees is usually faster than a forest with deep trees, even for the same RMSE.

---

## 10. The forest checklist

Before you ship a random forest, walk this list:

- [ ] **`n_estimators ≥ 500`.** Bigger if `oob_score` is still climbing at 500.
- [ ] **`max_features` overridden** to `"sqrt"` or `0.33` for regression. The sklearn default of 1.0 is *not* a Breiman forest.
- [ ] **`oob_score=True`.** The free generalization estimate. Print it; commit it.
- [ ] **`min_samples_leaf` tuned.** Default 1 can be too aggressive on small data; sweep `{1, 5, 10, 20}`.
- [ ] **`random_state` set.** Reproducibility.
- [ ] **`n_jobs=-1`.** Free speedup; trees are independent.
- [ ] **Feature importances printed *and* permutation importances computed.** Compare them; they should agree on the top features. If they disagree there, write down why.
- [ ] **Compared to a linear-model baseline (Week 4) and to a `DummyRegressor`.** Numbers in dollars or in the metric of choice; no relative claims without a baseline.

---

## 11. Where this leaves you

You can now derive the bagging variance-reduction argument from `Var(mean) = Var / B`; explain why random forests add column-subsampling to bagging; read an out-of-bag R² as a near-free generalization estimate; tune the four hyperparameters that matter (`n_estimators`, `max_features`, `min_samples_leaf`, `max_depth`); recognize Gini-importance cardinality bias; and identify the extrapolation cap as the one structural weakness of every tree ensemble.

Lecture 3 carries the ensemble idea in a different direction: instead of averaging many independently-fit trees, we fit each tree to the **residuals** of the previous ensemble. The algorithm is `gradient boosting`, the implementations are `HistGradientBoostingRegressor`, XGBoost, and LightGBM, and on the Ames Housing mini-project it is what beats the Week 4 linear baseline by ≥10% relative RMSE.
