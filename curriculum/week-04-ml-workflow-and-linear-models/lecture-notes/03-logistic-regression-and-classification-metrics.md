# Lecture 3 — Logistic Regression and Classification Metrics

> **Outcome:** You can write down the logistic-regression model (a linear function inside a sigmoid), name its loss (cross-entropy) and why we use it, fit it with scikit-learn, read off the coefficient as an effect on the log-odds, pick a decision threshold deliberately (not by accepting 0.5), and report the four classification metrics that matter — precision, recall, F1, ROC-AUC — and one bonus (PR-AUC) that is the right answer on imbalanced data. You leave knowing that "accuracy" is the metric that lies most often.

Linear regression predicts a number. Logistic regression predicts a *probability* — and, by thresholding, a class. The architecture is the same one-liner you saw in Lecture 2, with one nonlinearity wrapped around it. The difficulty is not the model. The difficulty is the metric. We will spend the back half of this lecture on metrics, because that is where most classification projects go wrong.

We target **scikit-learn 1.5+**. The `LogisticRegression`, `roc_auc_score`, `average_precision_score`, `precision_recall_curve`, `confusion_matrix`, and `classification_report` APIs are stable and will be the right answer for the rest of the decade.

---

## 1. The model in one line

A logistic regression for binary classification predicts the probability of the positive class as:

```text
P(y = 1 | x)  =  sigmoid(x · β)  =  1 / (1 + exp(−x · β))
```

The "linear" is the same `x · β` from linear regression (we include the bias column, so `β_0` is the intercept). The "logistic" is the **sigmoid** function that squashes the real number `x · β` into the open interval `(0, 1)`.

The sigmoid:

```python
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))
```

Plot it once. The shape is an "S": flat near 0 for very negative inputs, flat near 1 for very positive inputs, with a quick transition through 0.5 at `z = 0`.

The inverse of the sigmoid is the **logit** (or **log-odds**):

```text
logit(p)  =  log(p / (1 − p))
```

So we can write the model two equivalent ways:

```text
P(y = 1 | x)  =  sigmoid(x · β)        # the model in probability space
logit(P(y = 1 | x))  =  x · β           # the model in log-odds space
```

The right-hand side of the second equation is **linear**. That is why logistic regression is "linear": it is linear in the log-odds, not in the probability.

---

## 2. The loss: cross-entropy (a.k.a. negative log-likelihood)

You cannot use squared error for binary classification. Squared error against a 0/1 target gives a non-convex loss after the sigmoid, and the gradients are tiny near 0 and 1 (the "saturation problem"). The correct loss is **binary cross-entropy**:

```text
L(β)  =  −(1/n) · sum_i [ y_i · log(p_i)  +  (1 − y_i) · log(1 − p_i) ]
where p_i  =  sigmoid(x_i · β)
```

That is the negative log-likelihood of the data under the Bernoulli model "the i-th label is 1 with probability `p_i`." Two things to know:

1. **It is convex in `β`** for the sigmoid. There is one minimum. No local-minima trap.
2. **It has no closed-form minimizer.** Unlike linear regression, you must solve numerically. Scikit-learn uses L-BFGS (a quasi-Newton method) by default in 2026; for tiny problems it uses `liblinear`; for huge problems with L1, the `saga` solver.

The gradient (one application of the chain rule):

```text
∇_β L  =  (1/n) · Xᵀ (p − y)
```

That is the same shape as the linear-regression gradient — `Xᵀ` times "predicted minus actual" — with the sigmoid baked into `p`. Scikit-learn does the optimization for you; you almost never write this loop. But you should be able to derive it on paper in five minutes; it is the gateway to understanding the gradient of every classification model that follows.

---

## 3. Fitting logistic regression in scikit-learn 1.5+

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=1000, random_state=42)),
])
pipe.fit(X_train, y_train)
```

Three parameters that matter:

- **`penalty`** — `"l2"` (default, ridge-style), `"l1"` (lasso-style), `"elasticnet"`, or `None`. Almost always `"l2"`.
- **`C`** — the *inverse* regularization strength. `C` large → almost no regularization (like `alpha=0`). `C` small → heavy regularization (like `alpha=∞`). This is sklearn's historical convention, inherited from `liblinear`; it confuses everyone exactly once.
- **`solver`** — `"lbfgs"` (default; works for L2 and unpenalized), `"liblinear"` (small datasets, supports L1), `"saga"` (large datasets, supports L1 and elasticnet).

**Always scale features before fitting logistic regression with a penalty.** Same reason as ridge: the penalty is per-coefficient, and unscaled coefficients are incomparable.

The two prediction methods:

```python
y_pred = pipe.predict(X_val)              # 0 / 1, using a 0.5 threshold
p_pred = pipe.predict_proba(X_val)[:, 1]  # probability of class 1
```

`predict_proba` returns a 2-column array: column 0 is `P(y=0)`, column 1 is `P(y=1)`. You almost always want column 1 for binary classification.

> **EXPERIMENT — see the log-odds linearity.** Load `sklearn.datasets.load_breast_cancer()`. Fit a logistic regression on two features (say `mean radius` and `mean texture`). Plot `predict_proba(:, 1)` against `mean radius` colored by `mean texture` — the curve is sigmoidal. Now plot the **logit** of `predict_proba(:, 1)` against `mean radius`: it is a straight line for each fixed `mean texture`. That straightness *is* the model.

---

## 4. Interpreting coefficients: log-odds and odds ratios

The coefficient `β_j` from logistic regression is

> *"The change in log-odds of `y=1` for a one-unit increase in feature `j`, holding all other features fixed."*

Most humans do not think in log-odds. Exponentiate: `exp(β_j)` is the **odds ratio** for a one-unit increase. If `β_age = 0.05`, then `exp(0.05) ≈ 1.05`: a one-year increase in age multiplies the odds of `y=1` by 1.05 (a 5% relative increase). If `β_smoker = 1.20`, then `exp(1.20) ≈ 3.32`: being a smoker is associated with 3.32× the odds.

If you scaled features first (you should have), the coefficient is on the standardized feature: "a one-standard-deviation increase in age changes the log-odds by 0.05." Standardized coefficients are comparable across features; raw coefficients are not.

```python
import pandas as pd

coefs = pd.Series(pipe["logreg"].coef_[0], index=feature_names)
coefs_sorted = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
print(coefs_sorted.head(10))
```

---

## 5. The threshold dial

`predict()` defaults to a **0.5 threshold**: if the predicted probability is at least 0.5, predict class 1. That is the wrong threshold for almost every real-world problem.

Two examples of why 0.5 is wrong:

- **Imbalanced classes.** On a 5%-positive dataset, the optimal accuracy threshold may be 0.95 (always predict 0 unless you are very confident). The optimal F1 threshold is usually much lower than 0.5 (you need to catch positives even when uncertain).
- **Asymmetric costs.** A false positive on "is this transaction fraud" costs the price of a follow-up phone call. A false negative costs the cost of the fraud. The thresholds those two costs imply are nowhere near 0.5.

Pick the threshold deliberately. The procedure:

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

p_val = pipe.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, p_val)
f1 = 2 * precision * recall / (precision + recall + 1e-12)

best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]
print(f"best threshold for F1: {best_threshold:.3f}")
```

Then at deployment:

```python
y_pred = (pipe.predict_proba(X_new)[:, 1] >= best_threshold).astype(int)
```

The threshold is a **business decision**, not a hyperparameter. The model gives you a probability; you decide where to cut it.

---

## 6. The confusion matrix: where every metric comes from

The confusion matrix is a 2×2 table of `(true class) × (predicted class)`:

```text
                 predicted 0    predicted 1
actual 0          TN             FP
actual 1          FN             TP
```

- **TP** (true positive): predicted 1, actually 1.
- **FP** (false positive): predicted 1, actually 0. "False alarm."
- **FN** (false negative): predicted 0, actually 1. "Missed positive."
- **TN** (true negative): predicted 0, actually 0.

Every classification metric is a function of these four counts. Memorize the four names; the metrics fall out.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]

ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"]).plot()
```

---

## 7. The metrics, in the order to learn them

### 7.1 Accuracy — `(TP + TN) / total`

The fraction of predictions that were right. Intuitive, often wrong as a headline metric.

**When accuracy lies:** on imbalanced data. If 99% of your data is class 0, predicting "always 0" gets 99% accuracy. The metric tells you nothing about whether the model is useful.

Report accuracy as one number among several, never as the headline on imbalanced data.

### 7.2 Precision — `TP / (TP + FP)`

Of the things the model **said were positive**, how many actually were?

- High precision → few false alarms.
- Use as the headline when **false positives are expensive**: spam filters (you do not want to flag real email), fraud alerts (you do not want to phone every customer).

### 7.3 Recall (a.k.a. sensitivity, true positive rate) — `TP / (TP + FN)`

Of the things that **were actually positive**, how many did the model catch?

- High recall → few missed positives.
- Use as the headline when **false negatives are expensive**: cancer screening (you do not want to miss a tumor), security (you do not want to miss an intruder).

Precision and recall trade off against each other: lowering the threshold catches more positives (recall up) but also more false alarms (precision down). The trade-off is visible in the precision-recall curve.

### 7.4 F1 — `2 · (precision · recall) / (precision + recall)`

The harmonic mean of precision and recall. One number when you must report one. F1 punishes large gaps between the two: a precision of 1.0 and recall of 0.01 gives F1 ≈ 0.02, not 0.50.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

print(f"precision: {precision_score(y_val, y_pred):.3f}")
print(f"recall:    {recall_score(y_val, y_pred):.3f}")
print(f"F1:        {f1_score(y_val, y_pred):.3f}")
```

For multiclass, F1 has flavors: `average="macro"` (unweighted mean over classes), `average="weighted"` (class-frequency-weighted), `average="micro"` (global TP / FP / FN). On imbalanced multiclass, **macro F1** is the honest default.

### 7.5 ROC-AUC

The **ROC curve** plots **true positive rate** (recall) on the y-axis against **false positive rate** (`FP / (FP + TN)`) on the x-axis, as you sweep the threshold from 0 to 1.

The **AUC** (area under the curve) is one number:

- AUC = 1.0 → perfect classifier (at some threshold, you separate all positives from all negatives).
- AUC = 0.5 → random.
- AUC < 0.5 → worse than random (invert the predictions).

The most useful interpretation of ROC-AUC: it is the **probability that the model ranks a random positive higher than a random negative**. That is threshold-free; it says how well the model *orders* examples, not how well a single threshold cuts them.

```python
from sklearn.metrics import roc_auc_score, roc_curve

print(f"ROC-AUC: {roc_auc_score(y_val, p_val):.3f}")
fpr, tpr, _ = roc_curve(y_val, p_val)
# plot fpr vs tpr; one diagonal reference line for the random model.
```

ROC-AUC is **insensitive to class imbalance**. That sounds like a feature; on a 1% positive class, it is a bug. The curve is dominated by the easy negatives. Use **PR-AUC** instead in that regime.

### 7.6 PR-AUC (the imbalanced-data metric)

The **precision-recall curve** plots precision (y) against recall (x) as the threshold sweeps. The area under it is **average precision** (sklearn calls it `average_precision_score`).

- For balanced classes, PR-AUC tracks ROC-AUC closely.
- For imbalanced classes, **PR-AUC is the honest metric**. The baseline (random model) PR-AUC equals the positive rate, so a 0.05 PR-AUC on a 5%-positive dataset is no better than random — even though the ROC-AUC might look respectable.

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

print(f"PR-AUC (average precision): {average_precision_score(y_val, p_val):.3f}")
```

> **EXPERIMENT — ROC-AUC vs PR-AUC on imbalanced data.** Generate a 10,000-row dataset with 1% positives, with `y` weakly correlated with a single feature. Fit logistic regression. The ROC-AUC is around 0.80 (looks decent). The PR-AUC is around 0.05 (a hair above the 0.01 random baseline). Both numbers describe the same model; one is a flattering metric, one is honest.

---

## 8. `classification_report` — every metric in three lines

```python
from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred, target_names=["neg", "pos"]))
```

This prints precision, recall, F1, and support (count of true samples) for each class, plus macro and weighted averages. It is the right one-shot summary for a binary or multiclass model — and it is exactly the table to paste into the mini-project's write-up.

---

## 9. Common pitfalls in classification

1. **Reporting accuracy on imbalanced data.** "99% accuracy" on a 99% negative dataset is the always-predict-zero baseline. Report PR-AUC or per-class F1.
2. **Accepting the 0.5 threshold blindly.** The threshold is a business choice. Pick it explicitly.
3. **Using ROC-AUC to compare models on imbalanced data.** ROC-AUC is dominated by easy negatives. Use PR-AUC.
4. **Forgetting to set `class_weight="balanced"` when the classes are skewed.** sklearn weights the loss inversely to class frequency; without it, the optimizer maximizes accuracy, which is the wrong target.
5. **`predict_proba` for a multiclass model and then taking `[:, 1]`.** That assumes binary. Use `.predict_proba(X)` and read off the column index for your positive class via `model.classes_`.
6. **Evaluating a calibration-sensitive downstream use (cost calculations, expected value) on an uncalibrated probability.** `LogisticRegression`'s probabilities are reasonably well calibrated; tree ensembles' are not. Wrap with `CalibratedClassifierCV` when you need calibrated `predict_proba` outputs.

---

## 10. Multiclass logistic regression

For more than two classes, scikit-learn 1.5+ defaults to **multinomial** logistic regression (a.k.a. softmax regression), not the older one-vs-rest:

```text
P(y = k | x)  =  exp(x · β_k) / sum_j exp(x · β_j)
```

You get one coefficient vector `β_k` per class. The `LogisticRegression` API is the same; just pass a multiclass `y`:

```python
mdl = LogisticRegression(max_iter=1000).fit(X_train, y_multiclass_train)
mdl.predict_proba(X_val)         # shape (n, n_classes), rows sum to 1
mdl.predict(X_val)               # argmax over classes
```

`multi_class` was a parameter in scikit-learn ≤ 1.4 (`"ovr"` vs `"multinomial"`); in 1.5+ it is **deprecated**, and the multinomial behavior is the default. If you find a tutorial in 2026 that still passes `multi_class="ovr"`, the tutorial is stale.

For multiclass metrics, lean on `classification_report` with `target_names=...` and report **macro F1** and **macro ROC-AUC** unless there is a class you specifically care about.

---

## 11. When logistic regression is the right answer

The honest list, in 2026:

- **Binary classification with structured numeric / categorical features**, especially when you need probability outputs and reasonably calibrated coefficients.
- **The baseline for any classification problem.** Like its regression sibling, logistic is the floor every more-complex model has to beat.
- **High-stakes interpretability contexts** (credit risk, medical decision support, regulated industries). Logistic regression is a model regulators understand and statisticians have stress-tested for 80 years.
- **Linear separability** that you can verify with a 2-D scatter of any two features colored by class.

When logistic regression is **not** the right answer:

- **Wildly nonlinear decision boundaries** with strong feature interactions (image pixels, audio). Tree ensembles or neural nets win.
- **Text without embedding** at modern scale. Bag-of-words logistic regression is still a respectable baseline, but transformers ate the headline metrics in 2018.
- **Very high-cardinality categorical features** where the one-hot blow-up is computationally infeasible. Trees handle this directly; logistic regression needs target encoding or embeddings.

---

## 12. The checklist for a logistic regression that ships

- [ ] **Features scaled.** `StandardScaler` in the pipeline.
- [ ] **Class imbalance addressed.** `class_weight="balanced"` or sampling.
- [ ] **Threshold picked deliberately.** From the precision-recall curve, with a written justification.
- [ ] **`classification_report` printed.** Precision, recall, F1 per class.
- [ ] **PR-AUC reported** for imbalanced data; ROC-AUC for balanced.
- [ ] **Confusion matrix included.** A picture is faster than a paragraph.
- [ ] **Coefficient table.** Standardized; sorted by absolute value; signs sense-checked.
- [ ] **Calibration checked** if downstream uses the probability (`CalibrationDisplay.from_estimator`).
- [ ] **Test set scored once.** If the test number is far from CV, look for a leak.

---

## 13. Where this leaves you

You can now fit a logistic regression with calibrated probabilities, pick a threshold deliberately, report the metrics that match the problem, and read coefficients as effects on the log-odds. You can defend the choice of PR-AUC over ROC-AUC on imbalanced data without consulting Stack Overflow.

The week's mini-project is regression (Ames Housing), so logistic regression is not on the test there; the metrics-and-threshold discipline transfers directly to Week 5's classification problems and to every Week 11 production model where the "what is your business threshold" question is the first thing a stakeholder asks.

The model is one line. The metric, the threshold, and the confusion matrix are the next twenty. Like the rest of the week's work, the model is the smallest part of the work.
