# Week 4 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **scikit-learn — "Getting started"** (the canonical first hour; the `fit` / `predict` / `score` API in fifteen minutes):
  <https://scikit-learn.org/stable/getting_started.html>
- **scikit-learn — "Common pitfalls and recommended practices"** (twenty minutes; prevents an entire class of leaderboard disasters):
  <https://scikit-learn.org/stable/common_pitfalls.html>
- **scikit-learn — "Cross-validation: evaluating estimator performance"**:
  <https://scikit-learn.org/stable/modules/cross_validation.html>
- **scikit-learn — "Linear Models"** (the user guide; long, but the section on ridge and lasso is the canonical one-page summary):
  <https://scikit-learn.org/stable/modules/linear_model.html>
- **scikit-learn — "Metrics and scoring: quantifying the quality of predictions"**:
  <https://scikit-learn.org/stable/modules/model_evaluation.html>

## The official docs you will bounce between all week

- **scikit-learn API reference** (the source of truth for every estimator's parameters; bookmark it):
  <https://scikit-learn.org/stable/api/index.html>
- **scikit-learn `Pipeline` and `ColumnTransformer`** (the only honest way to preprocess without leakage):
  <https://scikit-learn.org/stable/modules/compose.html>
- **scikit-learn glossary** (one line each on `fit_transform`, `random_state`, `n_jobs`, `cv`, and three dozen other names):
  <https://scikit-learn.org/stable/glossary.html>
- **numpy.linalg** reference (for the closed-form normal equation in Exercise 2):
  <https://numpy.org/doc/stable/reference/routines.linalg.html>
- **scipy.stats** reference (for the residual-normality checks):
  <https://docs.scipy.org/doc/scipy/reference/stats.html>

## Free textbooks (the canon)

- **Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*** (free PDF from Stanford; the reference text for the entire classical-ML phase):
  <https://hastie.su.domains/ElemStatLearn/>
  Read **chapter 2** for bias-variance and the curse of dimensionality, and **chapter 3** for linear regression.
- **James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning*, 2nd edition** (free PDF; the gentle, lab-flavored counterpart to ESL):
  <https://www.statlearning.com/>
  Read **chapter 3** (linear regression) and **chapter 4** (classification) this week.
- **Aurélien Géron — *Hands-On Machine Learning*** (not free in book form, but the [notebooks](https://github.com/ageron/handson-ml3) on GitHub are MIT-licensed and walk through every concept we cover).
- **Kevin Murphy — *Probabilistic Machine Learning: An Introduction*** (free PDF, MIT Press, 2022):
  <https://probml.github.io/pml-book/book1.html>
  Chapter 11 is the rigorous version of "logistic regression done properly."

## The data sources we use this week

All public, all free to download:

- **Ames Housing dataset** (Dean De Cock, 2011; the modern replacement for the older Boston Housing data, which has well-known ethical issues and has been removed from scikit-learn):
  <https://jse.amstat.org/v19n3/decock.pdf> (paper with data card)
  Download CSV: <https://raw.githubusercontent.com/dancsoderberg/ames-housing/main/AmesHousing.csv> *or* the Kaggle "House Prices — Advanced Regression Techniques" copy: <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>
- **scikit-learn's `fetch_california_housing`** (a clean numeric regression target; useful for warm-ups):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>
- **scikit-learn's `load_breast_cancer`** (a 569-row binary classification dataset; canonical for logistic regression demos):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>
- **scikit-learn's `load_diabetes`** (a tiny regression dataset that is perfect for closed-form sanity checks):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>

A note on the Boston Housing dataset: it is **deprecated** in scikit-learn 1.2+ and **removed** in 1.5+. The `B` feature in particular has a well-documented racist construction. Use Ames Housing or California Housing instead. If you find a tutorial in 2026 that still uses `load_boston`, that tutorial has not been updated since 2022.

## The math you need this week

Less than you might expect:

- **Matrix multiplication, transpose, inverse** — for the closed-form normal equation `β̂ = (XᵀX)⁻¹Xᵀy`. NumPy does all the actual work; you only need to recognize the shape of the formula.
- **Gradient of a scalar function of a vector** — for gradient descent. The MSE gradient is `(2/n) Xᵀ(Xβ − y)`; we derive it in Lecture 2.
- **The chain rule** in one place: for logistic regression's gradient. One line.
- **Logarithms and the logit / sigmoid pair** — `sigmoid(x) = 1 / (1 + e⁻ˣ)`, `logit(p) = log(p / (1 − p))`. Inverses of each other.
- **No calculus of variations, no measure theory, no Lagrangians.** Save them for the elective.

If you want the math properly written out, ESL chapter 3 has the canonical treatment in fifteen pages. Read it if you want; skip it if you do not. The lecture notes here are self-contained.

## Tools you will use this week

- **`scikit-learn`** ≥ 1.5: `pip install "scikit-learn>=1.5,<2"`.
- **`pandas`** ≥ 2.2 and **`numpy`** ≥ 2 (from prior weeks).
- **`matplotlib`** ≥ 3.8 and **`seaborn`** ≥ 0.13 (from Week 3).
- **`scipy`** ≥ 1.11 (a transitive dependency of scikit-learn; we import `scipy.stats` directly for one residual check).
- **`pytest`** ≥ 8 for the exercises.

A `requirements.txt` snippet for the week:

```text
pandas>=2.2,<3
numpy>=2,<3
scipy>=1.11,<2
scikit-learn>=1.5,<2
matplotlib>=3.8,<4
seaborn>=0.13,<1
pytest>=8
```

## Videos (free, no signup)

- **Andrew Ng — *Machine Learning Specialization* (Coursera, audit track free)**. The first course covers linear and logistic regression in a careful, pedagogical sequence. The audit track gives you the videos; only the assignments are paywalled.
- **StatQuest with Josh Starmer** — short, clear, animated explainers. The episodes on **bias-variance**, **regularization (ridge/lasso)**, **logistic regression**, and **ROC and AUC** are each under fifteen minutes and surprisingly correct.
- **3Blue1Brown — "The essence of linear algebra"** (free YouTube). Episodes 9 (dot products) and 10 (cross products) are the geometric intuition for what `XᵀX` is doing.

## Open-source projects to read (in this order)

You can learn more from one hour reading sklearn source than from three hours of tutorials.

1. **scikit-learn `LinearRegression`** — the canonical least-squares wrapper around `numpy.linalg.lstsq`: <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_base.py>
2. **scikit-learn `Ridge`** — closed-form ridge with a small numerical-stability shim: <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_ridge.py>
3. **scikit-learn `LogisticRegression`** — wraps `liblinear` or `lbfgs` solvers: <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py>
4. **scikit-learn `Pipeline`** — the most-used 200 lines of code in the library: <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/pipeline.py>

## Cheat sheets (one-page references)

- **scikit-learn — "Choosing the right estimator"** (the official flowchart; print it):
  <https://scikit-learn.org/stable/machine_learning_map.html>
- **scikit-learn — "scoring parameter" cheat sheet** (every string you can pass to `cv=` and `scoring=`):
  <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter>
- **The C5 ML-workflow checklist** (in [lecture 1, section 11](./lecture-notes/01-the-ml-workflow.md)) — one printable page; tape it to your monitor.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Train set** | The rows the model sees during `fit`. Usually 60–80% of the data. |
| **Validation set** | The rows you use to pick hyperparameters and compare models. 10–20%. |
| **Test set** | The rows you check at the end. Once. Twice if you must. 10–20%. |
| **Cross-validation** | Repeat train/val on multiple folds; average the score. The honest way to use a small dataset. |
| **Leakage** | When information from the test set (or the future) sneaks into the training process. The single most common cause of "great" scores that do not reproduce. |
| **Baseline** | The score of a deliberately dumb predictor (mean, mode, "always 0"). Every model is measured against the baseline first. |
| **Bias** | Error from the model being too rigid to fit the truth. High bias = under-fitting. |
| **Variance** | Error from the model being too sensitive to the training set. High variance = over-fitting. |
| **No-free-lunch** | Wolpert's theorem: averaged over all problems, every learner is equally bad. Practical reading: match the inductive bias to the problem. |
| **MSE / RMSE / MAE** | Mean squared error / its square root / mean absolute error. Three regression losses with three different sensitivities to outliers. |
| **R²** | Coefficient of determination. 1.0 is perfect; 0.0 is the mean predictor; negative is worse than the mean. |
| **Residual** | `y_true − y_pred`. The diagnostic plot is residuals vs predicted; you want a noisy band, not a curve. |
| **Heteroscedasticity** | Residual variance changes with the predicted value. Violates a linear-regression assumption; consider a log transform or weighted regression. |
| **Multicollinearity** | Two predictors are highly correlated. Coefficients become unstable. Diagnose with VIF or the condition number. |
| **L1 / lasso** | Regularize by the sum of absolute coefficients. Drives some coefficients to exactly zero. Implicit feature selection. |
| **L2 / ridge** | Regularize by the sum of squared coefficients. Shrinks all coefficients toward zero; rarely to exactly zero. |
| **Elastic net** | A weighted mix of L1 and L2. Inherits both behaviours. |
| **Logit / sigmoid** | `logit(p) = log(p/(1−p))`; `sigmoid(x) = 1/(1+e⁻ˣ)`. Each is the inverse of the other. |
| **Cross-entropy** | The loss for logistic regression. Equivalent to negative log-likelihood under the Bernoulli model. |
| **Precision** | `TP / (TP + FP)`. Of the things you said were positive, how many were? |
| **Recall** | `TP / (TP + FN)`. Of the things that were positive, how many did you catch? |
| **F1** | Harmonic mean of precision and recall. One number when you need one. |
| **ROC-AUC** | Area under the ROC curve. Threshold-free; reads as "probability the model ranks a random positive above a random negative." |
| **PR-AUC** | Area under the precision-recall curve. The honest metric on imbalanced data. |

---

*If a link 404s, please open an issue so we can replace it.*
