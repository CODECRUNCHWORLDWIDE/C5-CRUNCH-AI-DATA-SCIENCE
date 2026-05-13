# Week 4 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** Why do we use a **validation** set in addition to a train and a test set?

- A) Because scikit-learn's `train_test_split` always returns three sets.
- B) To pick hyperparameters and compare candidate models without "spending" the test set.
- C) Because two-way splits are deprecated in scikit-learn 1.5+.
- D) Because the model fits faster when there is a validation set.

---

**Q2.** You have a dataset of 50,000 transactions, with each transaction tagged by a `customer_id`. Each customer has ~10 transactions. You want to predict whether a transaction is fraud. Which cross-validation strategy is honest?

- A) `KFold(n_splits=5, shuffle=True)`.
- B) `StratifiedKFold(n_splits=5)` because fraud is rare.
- C) `GroupKFold(n_splits=5)` with `groups=customer_id`, so no customer appears in both train and validation.
- D) `TimeSeriesSplit(n_splits=5)`.

---

**Q3.** The bias-variance decomposition says, roughly:

- A) `total error = bias × variance + noise`.
- B) `total error = bias² + variance + irreducible noise`.
- C) `total error = bias + variance - noise`.
- D) `total error = (bias + variance) / 2`.

---

**Q4.** A learning curve shows **train RMSE** at ~$10k and **validation RMSE** at ~$45k with a large gap between them, both flat as training-set size grows. The honest diagnosis is:

- A) High bias — the model is too simple; add features or pick a more flexible family.
- B) High variance — the model has memorized the training set; add regularization or more data.
- C) The noise floor of the problem is $45k.
- D) Something is wrong with the random seed.

---

**Q5.** The no-free-lunch theorem (Wolpert, 1996) implies, in practice:

- A) All ML algorithms are equally good in real-world problems.
- B) XGBoost is universally the best classical-ML algorithm.
- C) Match the **inductive bias** of the model to the structure of the problem; the "best" algorithm depends on the data.
- D) Neural networks are always best on large datasets.

---

**Q6.** You fit `LinearRegression` and the residual plot (residuals vs predicted) shows a clear funnel — wider on the right. Which is the right next move?

- A) Add more features.
- B) Switch to `RandomForestRegressor`.
- C) Address heteroscedasticity: log-transform `y`, use `HuberRegressor`, or use weighted regression.
- D) Increase the `max_iter` parameter.

---

**Q7.** You fit `Lasso(alpha=0.5)` on a 10-feature dataset. Four coefficients come back as exactly 0.0. The other six are non-zero. What does this mean?

- A) The fit failed; restart with `alpha=0`.
- B) Lasso has performed implicit feature selection: at this regularization strength, those four features did not pay for themselves.
- C) Lasso is broken in scikit-learn 1.5+.
- D) The four zero coefficients are placeholders for missing data.

---

**Q8.** On a binary classification problem with **5% positive class**, your model gets **97% accuracy**. Which is the honest reading?

- A) Excellent — 97% accuracy is great.
- B) Suspicious — the always-predict-negative baseline gets 95% accuracy. Report PR-AUC, precision, recall, and F1; accuracy alone is misleading here.
- C) Use the model — accuracy is the standard binary-classification metric.
- D) Switch to a different algorithm; logistic regression is too simple.

---

**Q9.** Why is **PR-AUC** preferred over **ROC-AUC** on heavily imbalanced datasets?

- A) PR-AUC is faster to compute.
- B) ROC-AUC is mathematically incorrect for imbalanced data.
- C) ROC-AUC is dominated by the (easy) majority class on imbalanced data; PR-AUC's denominator is `precision`, which makes positive-class performance the focus.
- D) PR-AUC handles missing data better.

---

**Q10.** You have a `Pipeline([("scaler", StandardScaler()), ("model", Ridge())])`. You run `cross_val_score(pipe, X, y, cv=5)`. Inside each fold, the scaler is:

- A) Fit once on all of `X` and reused across folds. (Leaky.)
- B) Re-fit on the training portion of that fold, then used to transform the validation portion. (Leak-free.)
- C) Skipped because cross-validation already handles standardization.
- D) Fit on `y` instead of `X`.

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — The validation set exists so that picking hyperparameters and comparing models does not "spend" the test set. Every comparison leaks a little optimism into the validation score; the test set is held back precisely to catch that optimism at the end. Option A is false (`train_test_split` returns one split per call). Option C is wrong (two-way splits are fine for very large datasets). Option D is unrelated to the existence of a validation set.

2. **C** — `GroupKFold(groups=customer_id)` is the honest answer. If the same customer's transactions appear in both train and validation, the model memorizes the customer, not the fraud signal. `KFold` (A) does not respect grouping. `StratifiedKFold` (B) addresses class imbalance but does not address grouping. `TimeSeriesSplit` (D) is the right answer if transactions are time-ordered and the deployment use predicts the future from the past, but the question does not give that structure; grouping is the dominant concern here.

3. **B** — `total error = bias² + variance + irreducible noise`. Bias is the systematic error from the model being too rigid; variance is the sensitivity of the model to the training sample; noise is the irreducible part of `y` that no model can predict. The decomposition is approximate (exact for squared error and certain assumptions) but the three-piece reading is universal. ESL chapter 2.

4. **B** — A large train/validation gap that is flat with training-set size is the textbook variance signature: the model has enough flexibility to fit the training set but not to generalize. Remedies: add regularization (ridge / lasso), reduce model capacity (fewer features), or collect more data. Option A would show *both* curves high and close together. Option C would show both curves converging at a high value (which we see, but the *gap* is the diagnostic). Option D is not a diagnosis.

5. **C** — The slogan version of the theorem is "no learner is universally best." The practical reading is to match the inductive bias to the problem: linear models for smooth, low-curvature relationships; trees for tabular data with categorical features; convolutional nets for images; transformers for sequences. Option A is the *literal* reading and the wrong practical conclusion; real problems are not uniformly drawn from "all problems." B and D are the absolute claims the theorem is meant to puncture.

6. **C** — A funnel is heteroscedasticity. The textbook remedies are: log-transform `y` (especially good for prices, counts, multiplicative phenomena); use `HuberRegressor` (downweights large residuals); use weighted regression with weights proportional to `1/variance(x)`. Option A is unrelated. Option B switches to a model that does not have the same assumption, but the *diagnosis* the residual plot gives you is heteroscedasticity, and the cleanest fix stays in the linear-model family. Option D does not address the issue.

7. **B** — That is exactly what lasso does. The L1 penalty has corners at the axes; the optimum often sits on a corner, where one (or several) coefficient is exactly zero. Variables with zero coefficients have been deselected by the model at this `alpha`. Increasing `alpha` deselects more variables; decreasing `alpha` keeps more. The behavior is implicit feature selection; it is what lasso is for.

8. **B** — The mean predictor (always-predict-negative) gets 95% accuracy on a 5%-positive dataset. A 97% accuracy is only 2 percentage points above that baseline, which corresponds to catching some — but not many — of the positives. The honest metrics are precision, recall, F1, and PR-AUC. Accuracy as a headline on imbalanced data is the metric that lies most often. Lecture 3, Section 7.1.

9. **C** — ROC-AUC's denominator includes the true negative rate, which is trivially high on imbalanced data (most negatives are easy). The curve hugs the upper-left corner even for mediocre classifiers. PR-AUC's denominator is precision (`TP / (TP + FP)`), which is hard to inflate on imbalanced data because false positives are abundant. The baseline PR-AUC for a random model is the positive rate, so PR-AUC has a direct "above random?" reading on imbalanced data. Option B is wrong as a math claim; ROC-AUC is still well-defined, just less informative.

10. **B** — That is the entire reason `Pipeline` exists. `cross_val_score` calls `fit` on the pipeline inside each fold, which re-fits every step (scaler included) on that fold's training portion before applying it to the validation portion. The leak-free behavior is automatic — and is exactly why fitting the scaler outside the pipeline is the canonical leak. Lecture 2, Section 9.

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
