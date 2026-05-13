# Week 4 — Homework

Six problems, about six hours total. Commit each in your Week 4 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — A learning curve, read honestly (1 hour)

Pick one of `sklearn.datasets.fetch_california_housing()` or `sklearn.datasets.load_diabetes()`. Build a `Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])` and plot a learning curve using `sklearn.model_selection.learning_curve` (Lecture 1, Section 7).

Save the plot as `homework/01-learning-curve.png`. In `homework/01-learning-curve.md` (~150 words):

1. What is the train RMSE at the largest training-set size?
2. What is the validation RMSE at the largest training-set size?
3. Is the gap closing, stable, or growing as you add data?
4. Do you have high bias, high variance, both, or neither?
5. What is the one move you would make next based on this plot?

**Acceptance.**

- PNG saved with axes labeled and a sentence-form title (Week 3's publication checklist applies).
- `.md` is honest about which of the four diagnoses fits.

---

## Problem 2 — Three residual plots, one model (1 hour)

Fit `LinearRegression` on the California Housing dataset (no feature engineering, just `StandardScaler` in a pipeline). Save three residual plots to `homework/02-residuals/`:

1. `residuals-vs-predicted.png` — residuals vs predicted value.
2. `residual-histogram.png` — histogram of residuals.
3. `residuals-vs-MedInc.png` — residuals vs the `MedInc` feature.

In `homework/02-residuals/notes.md` (~200 words), name which classical assumption each plot would let you check, and write a one-sentence verdict on each. The honest answer for California Housing is *"linearity is mildly violated, residuals are heavy-tailed, and there is heteroscedasticity"* — your plots should show evidence for or against that.

---

## Problem 3 — Ridge alpha by cross-validation (45 min)

Using the same pipeline as Problem 2 with `Ridge` instead of `LinearRegression`, sweep `alpha` over `np.logspace(-3, 3, 13)` using either:

- A manual loop with `cross_val_score`, recording the mean RMSE at each alpha. Plot the validation-curve (RMSE vs alpha on a log-x axis).
- `RidgeCV` with the same alpha grid, which is the production way.

Save the validation-curve plot as `homework/03-ridge-alpha.png` and report the chosen `alpha_` and the corresponding CV RMSE in `homework/03-ridge-alpha.md` (~100 words).

The shape of the curve is the lesson: at very small alpha, the model is essentially OLS and the RMSE is the OLS number; at very large alpha, the model is essentially the mean predictor and the RMSE is the baseline number; the minimum is somewhere in the middle. The width of the minimum tells you how sensitive the model is to the choice of alpha.

---

## Problem 4 — A logistic regression with deliberate threshold (1 hour)

Load `sklearn.datasets.load_breast_cancer(as_frame=True)`. Build a pipeline `Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=5000))])`. Split 80/20 with `random_state=42`. Fit on train.

In `homework/04-classification.py`:

1. Predict with the **default** 0.5 threshold; print `classification_report` on the test set. Save the printed report as `homework/04-classification/default.txt`.
2. Compute `precision_recall_curve` and pick the threshold that maximizes F1 on the **validation** portion. (Carve a small validation slice off the training set with another `train_test_split`.)
3. Predict on test with the F1-optimal threshold; print `classification_report` again. Save as `homework/04-classification/tuned.txt`.

In `homework/04-classification/notes.md` (~150 words), report both classification reports side by side and answer: did the tuned threshold improve precision, recall, F1, or all of the above? On this dataset, the default 0.5 is usually close to optimal — that is fine, and is the point. You are practicing the *procedure*, not chasing a number.

---

## Problem 5 — A leak demonstration (45 min)

In `homework/05-leak.py`, build two versions of the same pipeline on the same synthetic dataset:

- **Leaky.** Apply `StandardScaler.fit_transform(X)` *before* splitting. Then split. Then fit `Ridge(alpha=1.0)` on train. Report test RMSE.
- **Honest.** Split first. Build a `Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])`. Fit on train. Report test RMSE.

On synthetic i.i.d. data the gap is small — but the *pattern* is the lesson. To make the gap visible, construct a dataset where one feature has wildly different scale in train vs test (say, multiply the test feature by 1000 after the split). The leaky pipeline ignores this; the honest one shows the error.

In `homework/05-leak.md` (~150 words), explain in plain English why the leaky version is wrong even when the gap is small. The C5 answer: *"the leaky version's reported number is not an estimate of generalization; it includes information from the test set in the scaler's mean and standard deviation. The point is the pattern, not the magnitude on this particular dataset."*

---

## Problem 6 — Reflection (30 min)

Write `homework/06-reflection.md` (250–400 words) answering:

1. Which idea from Lecture 1 changed how you think about ML projects? The three splits, bias-variance, no-free-lunch, or something else?
2. Did the closed-form / gradient-descent / sklearn agreement in Exercise 2 surprise you? Why or why not?
3. Which of the five linear-regression assumptions (Lecture 2, Section 5) feels most like one you have, in retrospect, ignored on past projects?
4. After this week, which classification metric do you reach for first — accuracy, F1, ROC-AUC, PR-AUC — and why? Be specific about the kind of problem you are imagining.
5. What is the one habit from this week you want to keep when you move to Week 5 (trees and gradient boosting)? "Always start with a baseline" is the C5 answer; you may have a different one.

Honest is more valuable than polished.

---

## Time budget

| Problem | Time |
|--------:|----:|
| 1 | 1 h |
| 2 | 1 h |
| 3 | 45 min |
| 4 | 1 h |
| 5 | 45 min |
| 6 | 30 min |
| **Total** | **~5 h** |

(The schedule allocates 6h for homework; the remaining ~1h is buffer for reading the residual plots carefully and writing the reflection without rushing.)

When done, push your Week 4 repo and start (or finish) the [challenge](./challenges/) and the [mini-project](./mini-project/README.md).
