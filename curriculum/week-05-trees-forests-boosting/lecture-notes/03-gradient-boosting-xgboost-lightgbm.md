# Lecture 3 — Gradient Boosting: HistGBR, XGBoost, LightGBM

> **Outcome:** You can write down the gradient-boosting algorithm in five lines, explain why each new tree is fit to the residuals of the previous ensemble, choose between `HistGradientBoostingRegressor`, `XGBoost`, and `LightGBM` with a one-paragraph justification, tune the four hyperparameters that matter (`learning_rate`, `n_estimators`, `max_depth` / `num_leaves`, `min_samples_leaf` / `min_child_weight`), use early stopping to pick `n_estimators` for free, and compute three flavors of feature importance with their respective failure modes. After this lecture, "XGBoost won" stops being a leaderboard reflex and becomes a defended choice.

A random forest fits many trees independently and averages them; the variance reduction is the point. Gradient boosting takes a different route: fit a tree, look at where the ensemble is wrong, fit another tree to those residuals, add it (shrunk), repeat. Each tree is a small correction. The ensemble's prediction is a sum, not an average. After several hundred rounds of small corrections, the ensemble can fit complicated targets at low bias *and* low variance — if you stop early enough to avoid overfitting.

For ten years (roughly 2014 to 2024) this algorithm — gradient-boosted trees — was the best classical-ML model for tabular data, full stop. In 2026 it is still the best classical-ML model for tabular data; the only models that beat it are not classical (deep tabular nets like TabPFN, sometimes; deep transformers trained on tabular foundation models, rarely; LLMs, almost never).

We target **scikit-learn 1.5+**, **XGBoost 2.0+**, **LightGBM 4.0+**, **numpy 2.x**.

---

## 1. The model in five lines

Let `f_0(x) = ȳ` be the initial prediction (the mean of `y` in the training set). At round `m`:

```text
1. compute residuals     r_i = y_i − f_{m-1}(x_i)
2. fit a tree h_m        h_m = argmin_h sum_i (r_i − h(x_i))²       # one regression tree
3. shrink the predictions ν · h_m(x_i)                              # learning rate ν, usually 0.01–0.1
4. add to the ensemble    f_m(x) = f_{m-1}(x) + ν · h_m(x)
5. repeat M times         m = 1, 2, ..., M
```

After `M` rounds, the prediction is

```text
F(x)  =  ȳ  +  ν · (h_1(x) + h_2(x) + ... + h_M(x))
```

a sum of (shrunk) tree predictions. That is the entire algorithm. Three pieces of intuition:

1. **The first tree fits `y − ȳ`**: the residuals around the mean. Subsequent trees fit the residuals around the *current ensemble*. Each tree corrects what the previous trees got wrong.
2. **The learning rate `ν` is the most important hyperparameter.** It controls the size of each correction. With `ν = 1`, each new tree tries to fix all the residual at once; the ensemble overfits in 20 rounds. With `ν = 0.01`, each tree barely moves the ensemble; you need 5,000 rounds, but the ensemble fits smoothly.
3. **The trees are shallow on purpose.** `max_depth = 3` to `8` is the usual range. Each tree is a *weak learner* — it captures a small piece of the residual structure. The boosting loop turns many weak learners into one strong predictor; deep trees in a boosting loop overfit fast.

This is "gradient boosting for squared error." For an arbitrary differentiable loss `L(y, f)`, replace residuals with the **negative gradient** of the loss with respect to `f`, evaluated at the current prediction:

```text
r_i  =  − ∂L(y_i, f) / ∂f  |_{f = f_{m-1}(x_i)}
```

For squared error `L = (1/2)(y − f)²` the gradient is exactly `f − y`, so the negative gradient is `y − f` — the residual. That equivalence is why "gradient boosting" and "residual boosting" mean the same thing in the regression case. For logistic loss (classification), the negative gradient is `y − sigmoid(f)`, which is the standard "predict probabilities, fit trees to the gap between true labels and predicted probabilities" recipe.

---

## 2. Why "gradient" boosting

The Friedman 2001 framing is **functional gradient descent**: the ensemble `F` is a function in some function space, the loss `L` is a functional on that space, and at each round we take a step in the direction of the negative gradient — but constrained to live in the space of trees of bounded depth.

In ordinary gradient descent (Week 4, Lecture 2) you update a vector `β`:

```text
β  ←  β  −  η · ∇_β L
```

In gradient boosting you update a function `F`:

```text
F  ←  F  −  ν · (tree-approximation of ∇_F L)
```

The "tree-approximation" step fits a tree `h_m` to the gradient values `r_i` at each training point. The tree generalizes the gradient to inputs you have not seen. The learning rate `ν` plays the same role it played in ordinary gradient descent: bigger steps mean faster convergence but a higher risk of overshooting.

XGBoost (Section 4) uses a **second-order** Taylor expansion of the loss: it tracks the Hessian as well as the gradient, which lets it pick the optimal leaf value at each split analytically. This is the trick that made XGBoost faster (and slightly more accurate) than scikit-learn's `GradientBoostingRegressor` in 2016.

> **EXPERIMENT — boosting on a sine wave.** Generate `n = 200` rows of `y = sin(x) + 0.1 · noise` for `x ∈ [0, 4π]`. Fit `GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1)`. Plot the prediction surface every 10 rounds (use `staged_predict`). You can watch the ensemble step from a flat line at `ȳ` (round 0), through a coarse step function (round 20), to a smooth approximation of the sine wave (round 200). Decrease `learning_rate` to 0.01 and the same fit takes ~2000 rounds but is smoother. That picture is the entire point of gradient boosting.

---

## 3. The four hyperparameters that matter

A gradient-boosted-trees model has dozens of hyperparameters. On tabular data, four matter. The same four show up under different names in `HistGradientBoosting*`, XGBoost, and LightGBM — Section 4 gives the translation table.

| Hyperparameter | What it does | Sensible default |
|----------------|--------------|------------------|
| `learning_rate` | The shrinkage applied to each new tree's predictions. Smaller is better, up to a wall-clock cost. | 0.05. Tune in `{0.01, 0.03, 0.05, 0.1}`. |
| `n_estimators` | The number of boosting rounds. Can overfit. **Use early stopping** to set this. | 5000 with `early_stopping_rounds=50`. The algorithm picks the right value. |
| `max_depth` (or `num_leaves` for LightGBM) | The depth of each individual tree. Deeper trees fit more complex interactions but overfit faster. | `max_depth=6` or `num_leaves=31`. Tune in `max_depth ∈ {3, 4, 5, 6, 8}`. |
| `min_samples_leaf` (or `min_child_weight` for XGBoost / LightGBM) | The minimum sample count (or weight sum) in a leaf. Bigger values regularize. | 20 on tabular regression. Default 1 is too small for noisy data. |

That is it. The other knobs — `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`, `max_bin`, `min_split_gain` — can each be tuned for a small additional gain (≤1% relative RMSE in most cases). On Ames-sized data, sweep the four above first and ignore the rest until you cannot improve any further.

> **EXPERIMENT — learning rate vs n_estimators trade.** Fit `HistGradientBoostingRegressor` on the California Housing dataset with `(learning_rate, max_iter)` pairs `(0.5, 50)`, `(0.1, 250)`, `(0.05, 500)`, `(0.01, 2500)`. All four use early stopping on a 20% validation split. Plot the validation RMSE at each step (via `validation_score_` / `staged_predict`). The four curves converge to similar minima; the lowest learning rate has the smallest minimum (by ~1–3%) but takes 10–50× more wall-clock time. The picture is "more rounds at a smaller learning rate" → "smoother optimization, slightly better fit, much slower."

---

## 4. HistGradientBoostingRegressor vs XGBoost vs LightGBM

Three implementations, same algorithm, different defaults and different speed characteristics. The translation table:

| Concept | `HistGradientBoostingRegressor` (sklearn) | XGBoost | LightGBM |
|---------|-------------------------------------------|---------|----------|
| Rounds | `max_iter` | `n_estimators` | `n_estimators` |
| Learning rate | `learning_rate` | `learning_rate` | `learning_rate` |
| Tree depth | `max_depth`, `max_leaf_nodes` | `max_depth` | `num_leaves`, `max_depth` |
| Leaf minimum | `min_samples_leaf` | `min_child_weight` | `min_data_in_leaf` |
| Histogram bins | `max_bins=255` | `max_bin=256` | `max_bin=255` |
| Native categorical | `categorical_features=` | `enable_categorical=True` | `categorical_feature=` |
| Early stopping | `early_stopping=True` (default for `max_iter ≥ 10`) | `early_stopping_rounds=` | `early_stopping_rounds=` |
| Missing values | handled natively | handled natively | handled natively |

All three are histogram-binned implementations: continuous features are pre-bucketed into ~256 bins, and the split search runs over bins instead of raw values. This is the single trick that makes all three orders of magnitude faster than the original `GradientBoostingRegressor` for `n ≥ 10^4`.

**When to pick which:**

- **`HistGradientBoostingRegressor`** — when you want zero extra dependencies, full scikit-learn `Pipeline` integration, and you don't need the categorical-feature path that XGBoost lacked until 2.0. The 1.0+ implementation matches XGBoost on accuracy and is within 1.5× on speed. The C5 default for the Week 5 mini-project.
- **XGBoost** — when you have a Kaggle-shaped workflow, you want SHAP integration (XGBoost ships TreeSHAP natively in the `.predict(..., pred_contribs=True)` path), or you are training on > 1M rows and want to use the GPU.
- **LightGBM** — when you have many categorical features with high cardinality (LightGBM's native categorical-feature handling is the cleanest of the three) or when you have > 10M rows and CPU time matters. LightGBM is consistently 1.5–3× faster than XGBoost on the same hardware.

Practical takeaway: on ≤100k tabular rows, the three implementations give RMSE numbers that agree to within a few percent. Pick on the basis of dependencies and convenience, not RMSE.

> **EXPERIMENT — three implementations, one dataset.** Fit `HistGradientBoostingRegressor`, `XGBRegressor`, and `LGBMRegressor` on the California Housing dataset with matched hyperparameters (`learning_rate=0.05, max_depth=6, n_estimators=500`). Measure test RMSE and wall-clock. The three RMSE numbers will agree to within ~1% on this dataset; the wall-clock numbers will differ by 2–5× (LightGBM fastest, sklearn slowest). Exercise 3 walks you through this comparison rigorously.

---

## 5. Early stopping: the free hyperparameter

Picking `n_estimators` by grid search is wasteful. The honest way is **early stopping**: train one model with many rounds and a validation set; stop when the validation score has not improved for `early_stopping_rounds` consecutive rounds.

```python
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

mdl = LGBMRegressor(
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=10_000,            # an overestimate
    random_state=42,
)
mdl.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50)],
)
print(f"best round: {mdl.best_iteration_}")
print(f"val RMSE at best round: {mdl.best_score_['valid_0']['l2'] ** 0.5:.4f}")
```

XGBoost and `HistGradientBoostingRegressor` have analogous APIs. Two things to know:

1. **Early stopping needs a validation set, not just cross-validation.** You hand the model the validation set explicitly; it watches the score round-by-round and stops on its own. The validation set is one extra holdout off the training set; the test set is still untouched.
2. **The "best round" can be much smaller than the upper bound.** A typical Ames-sized fit will stop at 300–1500 rounds even with `n_estimators=10000`. The model knows when more rounds stop helping; let it tell you.

The combination of early stopping and a small learning rate is the production recipe: `learning_rate=0.05`, `n_estimators=10000`, `early_stopping_rounds=50`. The model takes 30 seconds to train, picks its own `n_estimators`, and lands at a near-optimal score.

---

## 6. Feature importance, three ways

A fitted gradient-boosted-trees model offers three different ways to measure feature importance. Each has a failure mode.

**Split-gain importance** (the default from `xgb.feature_importances_`, `lgbm.feature_importances_`, `sklearn.feature_importances_`). Sum the impurity reduction attributable to each feature, across all trees in the ensemble.

```python
importances = pd.Series(mdl.feature_importances_, index=X.columns)
```

- **Failure mode:** biased toward continuous and high-cardinality features (Lecture 2, Section 7). Free from the fit, so use it as a quick sanity check, not as the headline.

**Permutation importance** (model-agnostic, via `sklearn.inspection.permutation_importance`). For each feature, shuffle its values in the validation set, measure the score drop.

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(mdl, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test.columns)
```

- **Failure mode:** when features are correlated, permuting one of them produces unrealistic rows (the shuffled feature no longer matches the others), which underestimates importance. Use with `n_repeats=10+` for noise reduction.

**SHAP values** (Shapley values from cooperative game theory; `shap.TreeExplainer`). For each prediction, attribute the difference between the prediction and the base value to the individual features.

```python
import shap

explainer = shap.TreeExplainer(mdl)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

- **Failure mode:** SHAP values are exact under a model-conditional expectation that assumes feature *independence*; with correlated features they reflect a slightly different counterfactual than you might intuit. The `TreeExplainer` is fast and exact for tree-based models; the model-agnostic `KernelExplainer` is slow and approximate. Stick with `TreeExplainer` for trees.

In practice, **use all three**. The top features in the three rankings should agree on the top 3–5 features. Where they disagree at the top, write down why. Challenge 1 walks through a worked example on the Ames dataset.

---

## 7. The hyperparameter card (the one to tape to your monitor)

```text
┌─────────────────────────────────────────────────────────┐
│  THE FOUR HYPERPARAMETERS THAT MATTER                   │
│                                                         │
│  learning_rate          0.05         {.01, .03, .05, .1}│
│  n_estimators           early stop   set high, stop early│
│  max_depth / num_leaves 6 / 31       {3, 4, 6, 8} / {15,│
│                                       31, 63, 127}       │
│  min_samples_leaf       20           {5, 10, 20, 50, 100}│
│                                                         │
│  Tune in that order. Stop after these four.             │
│  Set max_bin=255 (default).                             │
│  Set objective='regression' / 'binary' as appropriate.  │
│  Use early stopping. Use n_jobs=-1.                     │
└─────────────────────────────────────────────────────────┘
```

The rest of the parameter list is for special cases — heavy class imbalance, extreme `n`, GPU training, monotonic constraints, custom loss functions. On Ames-sized tabular regression none of them matter to the first decimal of RMSE.

---

## 8. Categorical features: the native path

`HistGradientBoostingRegressor`, XGBoost ≥ 2.0, and LightGBM all support **native categorical splits** — splitting on subsets of categories rather than on one-hot-encoded columns. The native path is usually 1–3% relative RMSE better than one-hot encoding for high-cardinality features, and significantly faster to train.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

# Identify categorical columns by integer index or pandas dtype.
cat_features = [X.columns.get_loc(c) for c in X.select_dtypes("object").columns]

mdl = HistGradientBoostingRegressor(
    categorical_features=cat_features,
    learning_rate=0.05,
    max_iter=2000,
    early_stopping=True,
)
mdl.fit(X_train, y_train)
```

For LightGBM:

```python
from lightgbm import LGBMRegressor

# Make sure categorical columns are pandas 'category' dtype.
for c in cat_cols:
    X_train[c] = X_train[c].astype("category")
    X_test[c]  = X_test[c].astype("category")

mdl = LGBMRegressor(learning_rate=0.05, num_leaves=31, n_estimators=2000).fit(X_train, y_train)
```

LightGBM picks up categorical columns automatically from the dtype; you do not have to pass them explicitly. XGBoost 2.0+ wants `enable_categorical=True` and the same `category` dtype.

One caveat: native categorical splits are not portable across libraries. If you train with `HistGradientBoosting` and deploy with XGBoost (or vice versa), the categorical encoding will not match. Pick one library and stick with it. Or use one-hot encoding for portability — you give up a small amount of RMSE for the ability to swap libraries.

---

## 9. The gradient-boosting failure modes

Three things can go wrong with a gradient-boosted-trees model:

1. **Overfitting from no early stopping.** Train R² is 0.999 and test R² is 0.78. The model fit the training noise. The fix: early stopping, smaller `max_depth`, larger `min_samples_leaf`, smaller `learning_rate` (and more rounds).
2. **Underfitting from too-small `learning_rate` and not enough rounds.** Train and test R² are both mediocre. The fix: more rounds (raise the `n_estimators` upper bound) or a bigger learning rate.
3. **Extrapolation cap** (Lecture 2, Section 9). The model cannot predict outside the training-set leaf-mean range. The fix: model `log(y)` if the target is multiplicative (housing prices), or use a model that extrapolates (linear regression) for the extrapolation piece.

Two diagnostics to check after every fit:

```python
print(f"best iteration: {mdl.best_iteration_}")        # if early stopping, this is M
print(f"train R²: {mdl.score(X_train, y_train):.3f}")
print(f"test  R²: {mdl.score(X_test, y_test):.3f}")
```

A `train R² − test R²` gap larger than ~0.1 is the high-variance signal. The remedies are in the previous paragraph.

> **EXPERIMENT — overfitting on purpose, then early-stopping.** Fit `XGBRegressor(n_estimators=2000, max_depth=10, learning_rate=0.3)` on the diabetes dataset with no early stopping. Print train and test R². The train number is near 1.0 and the test number is far below the early-stopped equivalent. Re-fit with `max_depth=4, learning_rate=0.05, n_estimators=10000, early_stopping_rounds=50` on an explicit validation set. The train R² is lower; the test R² is higher; `best_iteration_` is in the low hundreds, not 2000. That is the gradient-boosted-trees discipline in two lines of code.

---

## 10. The gradient-boosting checklist

Before you ship a gradient-boosted-trees model, walk this list:

- [ ] **`learning_rate ≤ 0.1`.** Default 0.1 is the upper end; smaller is usually better with more rounds.
- [ ] **Early stopping is on.** With an explicit validation set carved off the training set.
- [ ] **`max_depth` (or `num_leaves`) set explicitly.** 3–8 (or 15–127) is the range. Default 6 / 31 is sensible.
- [ ] **`min_samples_leaf` (or `min_child_weight`) tuned.** Defaults are too aggressive on small data.
- [ ] **`random_state` set.** Reproducibility.
- [ ] **`n_jobs=-1`** (XGBoost / LightGBM) or `n_jobs=None` then the thread pool from sklearn (HistGBR). Use your cores.
- [ ] **Best iteration printed.** With early stopping, this is the `n_estimators` the data chose.
- [ ] **Three importances computed.** Split-gain, permutation, SHAP (at least for the top features).
- [ ] **Train and test R² printed, side by side.** Big gap → high variance → tune toward smaller `max_depth` or larger `min_samples_leaf`.
- [ ] **Compared to the Week 4 linear baseline *and* to the Week 5 random forest.** Numbers in the metric of choice; no relative claims without baselines.

---

## 11. Where this leaves you

You can now write down the gradient-boosting algorithm in five lines; explain the connection to functional gradient descent; choose between `HistGradientBoostingRegressor`, XGBoost, and LightGBM with a one-paragraph justification; tune the four hyperparameters that matter; use early stopping to pick `n_estimators` for free; compute three flavors of feature importance and name the failure mode of each; and recognize the three ways a gradient-boosted-trees model can go wrong.

The mini-project at the end of the week is one task: take the Week 4 Ames Housing pipeline, swap `RidgeCV` for `HistGradientBoostingRegressor` (or XGBoost, or LightGBM), tune it deliberately, and beat the Week 4 test RMSE by **≥10% relative**. The deliverable is a notebook plus a SHAP plot plus a two-page report. If you can do that — and if you can explain *why* the trees won in two paragraphs of prose — you have everything you need to ship a tabular-data model into production in 2026.
