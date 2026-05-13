# Week 5 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** A decision-tree regressor splits a node on the (feature, threshold) pair that maximizes:

- A) The number of rows in the larger child.
- B) The reduction in the weighted impurity (MSE for regression, Gini or entropy for classification) from parent to children.
- C) The correlation coefficient between feature and target on the left side.
- D) The depth of the resulting subtree.

---

**Q2.** A decision tree built without any stopping rule on a regression dataset has train R² = 1.000 and test R² = 0.18. The honest diagnosis is:

- A) The model is well-fit; the test set is unrepresentative.
- B) The tree has memorized the training set: `max_depth=None` lets each leaf hold one row. Set `max_depth` and `min_samples_leaf`, or move to an ensemble.
- C) The MSE-impurity criterion is broken for this dataset; switch to Gini.
- D) Decision trees do not work on regression problems.

---

**Q3.** Bagging reduces the variance of an ensemble by:

- A) Increasing the depth of each tree.
- B) Averaging predictions from `B` models, each fit on a different bootstrap sample. The variance of a mean of `B` iid estimators is `Var / B`; bootstrap samples are not iid, so the actual reduction is smaller but still real.
- C) Using a different impurity criterion in each tree.
- D) Voting between trees by accuracy on a held-out set.

---

**Q4.** A bootstrap sample of size `n` from a dataset of size `n` contains roughly:

- A) All `n` distinct rows.
- B) About `n/2` distinct rows.
- C) About `0.632 · n` distinct rows on average, with the other `~37%` of rows "out-of-bag" for that sample.
- D) Exactly `n - 1` distinct rows because one row is reserved for validation.

---

**Q5.** The random-forest hyperparameter `max_features` controls:

- A) The maximum number of leaves per tree.
- B) The maximum number of features used in the entire forest.
- C) The number of features considered as candidate splits at each internal node — the column-subsampling that *decorrelates* the trees and is what distinguishes a random forest from a bagging ensemble of trees.
- D) The number of features in the OOB sample.

---

**Q6.** Scikit-learn's `RandomForestRegressor` default for `max_features` is **1.0** (use all features). Why is this often the wrong setting for regression?

- A) It is too slow.
- B) Breiman's argument for random forests requires column subsampling at each split (`max_features < p`) to decorrelate the trees. With `max_features=1.0`, you have a bagging ensemble of trees, not a random forest. The OOB R² is usually 0.02–0.10 better with `max_features="sqrt"` or `0.33`.
- C) Sklearn deprecated `max_features=1.0` in 1.5.
- D) It causes the OOB estimate to underestimate generalization error.

---

**Q7.** Gradient boosting fits each new tree to:

- A) A random bootstrap sample of the data (like bagging).
- B) The residuals of the current ensemble (or, in general, the negative gradient of the loss with respect to the current prediction). The new tree's predictions are shrunk by a learning rate and added to the ensemble.
- C) The full original target `y`, like in a random forest.
- D) The validation set, to avoid overfitting.

---

**Q8.** Which combination of hyperparameters is the production recipe for a gradient-boosted-trees model?

- A) `learning_rate=1.0`, `n_estimators=50`, `max_depth=None`.
- B) `learning_rate=0.05`, `n_estimators=10000`, `early_stopping_rounds=50`, with an explicit validation set. The algorithm picks `n_estimators` automatically.
- C) `learning_rate=0.5`, `n_estimators=1000`, `max_depth=20`. The deeper trees fit more interactions.
- D) `learning_rate=0.001`, `n_estimators=100`. A very small learning rate without enough rounds is the most common underfitting failure.

---

**Q9.** A model's `feature_importances_` attribute (Gini / split-gain importance) is biased toward:

- A) Categorical features over numeric features.
- B) Features near the leaves of each tree.
- C) High-cardinality continuous features. A feature with many possible split thresholds has more chances to accumulate impurity reduction; this bias is unrelated to whether the feature actually helps generalization. Use **permutation importance** on a held-out set, or **SHAP**, to correct for it.
- D) Features that appear first in the input matrix.

---

**Q10.** You fit `HistGradientBoostingRegressor`, `XGBRegressor`, and `LGBMRegressor` on the same dataset with matched hyperparameters. The three test RMSEs are 0.487, 0.483, 0.491. The honest interpretation is:

- A) LightGBM wins. Use LightGBM going forward.
- B) The differences are within noise; the three implementations are interchangeable on this problem. Pick on the basis of dependencies, categorical-feature handling, and wall-clock — not RMSE.
- C) Something is wrong with the HistGBR fit because it has the highest RMSE.
- D) The implementations disagree only because the hyperparameters were not actually matched.

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — A regression-tree split maximizes the *reduction in impurity* (parent impurity minus weighted children impurity). For regression, the impurity is MSE; the gain is also equivalent to the "between-children variance" exposed by the split. Lecture 1, Section 3.

2. **B** — Train R² = 1.000 and test R² = 0.18 is the textbook signature of a tree with no stopping rule: `max_depth=None` lets each leaf hold one training row, so the tree memorizes everything. Set `max_depth` (4–8 is the usual range for tabular regression), raise `min_samples_leaf` (5–20), or — much more effectively — move to a random forest or a gradient-boosted-trees model. Lecture 1, Sections 6 and 11.

3. **B** — Bagging averages predictions from `B` bootstrap-trained models. The variance argument is `Var(mean of B iid estimators) = Var / B`. Bootstrap samples are not iid (they share rows), so the actual variance reduction depends on the pairwise correlation `ρ` between models — which is the entire reason random forests add column subsampling on top of bagging (to decrease `ρ`). Lecture 2, Section 2.

4. **C** — The probability a specific row is *not* drawn on any of `n` pulls with replacement is `(1 − 1/n)^n → 1/e ≈ 0.368` as `n → ∞`. So each row appears in a bootstrap with probability `1 − 1/e ≈ 0.632`, and the out-of-bag fraction is `~0.368`. The OOB rows are the bagging algorithm's free validation set. Lecture 2, Sections 3 and 8b.

5. **C** — `max_features` is the column-subsampling knob at each split. Breiman's 2001 paper introduced this on top of bagging precisely because trees fit on bootstrapped data are still strongly correlated (the top split is almost always on the same feature). Subsampling the candidate-feature pool at each split breaks that correlation. Lecture 2, Section 4.

6. **B** — Sklearn's default for `RandomForestRegressor` (`max_features=1.0`) is *not* a Breiman random forest; it is a bagging-of-trees ensemble. Override to `"sqrt"` or `0.33` for regression. The OOB R² typically improves by 0.02–0.10. Lecture 2, Section 4.

7. **B** — Each new tree in gradient boosting is fit to the **residuals** of the current ensemble (for squared-error loss; the negative gradient of the loss in general). Its predictions are shrunk by the learning rate and added to the ensemble. This is fundamentally different from bagging, where each tree is fit to a bootstrap of the *original* target. Lecture 3, Section 1.

8. **B** — Set the learning rate small (~0.05), set the number of rounds high (10,000) as an upper bound, and use **early stopping** with `early_stopping_rounds=50` on an explicit validation set. The library tells you when more rounds stop helping. This combination — small LR plus early stopping — is the recipe. A is the overfitting recipe (LR=1.0 makes each tree fix all the residual at once). C overfits via deep trees. D underfits because LR=0.001 with only 100 rounds means the ensemble barely moves off the initial mean. Lecture 3, Sections 3 and 5.

9. **C** — Gini / split-gain importance is biased toward high-cardinality continuous features because they have more candidate split thresholds. The bias is well-documented and well-understood. The remedy is **permutation importance** (model-agnostic, measured on a held-out set, bias-free with respect to feature cardinality) or **SHAP values** (theoretically principled, exact for trees via `TreeExplainer`). Lecture 2, Section 7 and Lecture 3, Section 6.

10. **B** — On Ames-sized and California-Housing-sized data, `HistGradientBoostingRegressor`, XGBoost, and LightGBM with matched hyperparameters agree on RMSE to within a few percent. The differences in the question (0.483, 0.487, 0.491) are well within fold-to-fold noise. Pick on the basis of dependencies (HistGBR adds zero), categorical-feature handling (LightGBM is cleanest), and wall-clock (LightGBM usually wins) — not on a 1% RMSE difference. Lecture 3, Section 4.

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
