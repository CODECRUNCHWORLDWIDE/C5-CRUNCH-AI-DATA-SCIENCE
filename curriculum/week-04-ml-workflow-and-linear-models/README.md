# Week 4 — The ML Workflow and Linear Models

> *A model is a sentence about the world. Most of this week is the punctuation: which rows it was fit on, which it was tested on, which it has never seen, and what the residuals look like when it is wrong. The model itself — a linear regression, a logistic regression — fits on one slide. The workflow around it is the rest of the course.*

Welcome to week four of **C5 · Crunch AI / Data Science**. The first three weeks gave you the table: a NumPy array, a labeled pandas DataFrame, and a chart you can defend. This week you build the first **model**. It is not a deep network. It is not a gradient-boosted tree. It is a **linear regression** and its sibling, **logistic regression**, fit and evaluated through the **ML workflow** that the rest of the course will plug models into.

Three warnings before we start, because Week 4 is where most "ML courses" go wrong:

1. **The workflow is the lesson, not the model.** A correctly cross-validated linear regression on a clean dataset beats a sloppily-evaluated neural network on the same dataset every time. The score on the leaderboard is the *output* of the workflow; this week is the workflow.
2. **Linear regression is not a "simple" model.** It is the model with the most assumptions, the most diagnostic tools, the most interpretation hooks, and the longest record of fooling people who ignored its assumptions. We will fit it by hand once (closed-form, then gradient descent), then with scikit-learn, then critique each fit against its assumptions.
3. **`model.fit(X, y)` is the easiest line in the file.** Splitting the data so the score means something, building features that do not leak the answer, picking a regularization strength that generalizes — these are the hours. The fit itself is one millisecond.

We target **scikit-learn 1.5+** (current stable in 2026 is 1.6), **numpy 2.x**, and **pandas 2.2+**. The APIs are stable; the `Pipeline`, `ColumnTransformer`, and `train_test_split` interfaces you learn here are the ones you will use in 2030.

---

## Learning objectives

By the end of this week, you will be able to:

- **Split** a dataset into train / validation / test with `train_test_split` and `KFold`, and explain in one sentence why each split exists. Recognize a leak before it bites you.
- **State** the bias-variance decomposition in plain English and identify whether a learning curve shows high bias, high variance, or "you ran out of data."
- **Quote** the no-free-lunch theorem and use it to explain why "the best algorithm" is a meaningless phrase without a problem and a metric.
- **Fit** a linear regression three ways: closed-form (normal equation), gradient descent (your own loop), and `sklearn.linear_model.LinearRegression`. Verify all three agree.
- **Audit** a linear-regression fit against its five assumptions (linearity, independence, normality of residuals, homoscedasticity, no perfect multicollinearity). Read a residual plot.
- **Apply** L1 (`Lasso`) and L2 (`Ridge`) regularization. Pick the regularization strength with cross-validation, not by feel.
- **Fit** a logistic regression and report classification metrics: precision, recall, F1, ROC-AUC, PR-AUC. Know which one to lead with for an imbalanced dataset.
- **Build** an end-to-end `Pipeline` with `ColumnTransformer` so the same preprocessing applies to train and test without leakage.
- **Beat** a strong baseline on the **Ames Housing** dataset in the Week 4 mini-project. Document every feature. Justify every choice.
- **Pass** every `pytest` case on the Week 4 exercises.

---

## Prerequisites

- **Weeks 1, 2, and 3 complete.** You have a labeled DataFrame on disk from the Week 2 mini-project and at least one publication-quality chart in your portfolio.
- A working **Python 3.11+** install (we use 3.12).
- scikit-learn 1.5+: `pip install "scikit-learn>=1.5,<2"`. We rely on `Pipeline`, `ColumnTransformer`, `train_test_split`, and the `linear_model` module.
- pandas 2.2+ and numpy 2.x (from Weeks 1–2).
- matplotlib 3.8+ and seaborn 0.13+ (from Week 3 — we plot residuals and learning curves).
- `pytest` for the exercise smoke tests.

No GPU. No notebook required for the exercises (they are `.py` files and save artifacts to disk); the mini-project ships as a notebook.

---

## Topics covered

- The **ML workflow**: framing the problem, building the dataset, splitting, baselining, fitting, evaluating, iterating, freezing.
- The three splits: **train**, **validation**, **test**. Why three, not two. Why the test set is checked at most twice in the entire project.
- **Cross-validation**: `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`. Pick the one whose assumption matches your data.
- The **bias-variance decomposition** in plain English: a high-bias model is too rigid to fit the truth; a high-variance model is too flexible to ignore the noise. The remedy depends on which one you have.
- **The no-free-lunch theorem** (Wolpert, 1996): averaged over all possible problems, no learner beats another. The practical corollary is "match the inductive bias to the problem," not "pick the latest model."
- **Linear regression** done properly: the model, the loss, the closed-form solution, gradient descent, and the five assumptions. Residual analysis. Heteroscedasticity. Multicollinearity. The condition number.
- **Regularization**: ridge (L2), lasso (L1), elastic net. The geometric intuition. Picking the penalty with `RidgeCV` / `LassoCV`.
- **Logistic regression**: the link function, the loss (cross-entropy), why "linear" lives in the log-odds, not the probability.
- **Classification metrics**: accuracy (and when it lies), precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, the threshold dial.
- **Pipelines and `ColumnTransformer`**: the only honest way to apply scaling, encoding, and imputation without leaking the test set into the training set.
- **Learning curves and validation curves**: the two diagnostic plots you should draw before believing any model score.

---

## Weekly schedule

Target about **38 hours**. Some sections click in twenty minutes; some take three hours. Treat the table as a budget, not a contract.

| Day       | Focus                                                  | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|--------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | The ML workflow; train/val/test; bias-variance         |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | Linear regression by hand; assumptions; residuals      |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Wednesday | Regularization (L1/L2); cross-validation; pipelines    |   2h     |   2h      |     2h     |   0.5h    |   1h     |     0h       |   0h       |   7.5h      |
| Thursday  | Logistic regression; classification metrics            |   3h     |   1h      |     0h     |   0.5h    |   1h     |     2h       |   0.5h     |    8h       |
| Friday    | Mini-project: build features, fit, evaluate            |   0h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    6h       |
| Saturday  | Mini-project polish; write-up; learning-curve plot     |   0h     |   0h      |     0h     |   0h      |   1h     |     3h       |   0h       |    4h       |
| Sunday    | Quiz, review, push to portfolio repo                   |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                        | **11h**  | **8h**    | **2h**     | **3h**    | **6h**   | **8h**       | **2h**     |  **40h**    |

(The schedule overshoots 38h by 2h on purpose — Week 4 is the first week where things get tight, and the buffer is realistic.)

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Free docs for scikit-learn 1.5+, the Ames Housing data card, ESL / ISLR free PDFs |
| [lecture-notes/01-the-ml-workflow.md](./lecture-notes/01-the-ml-workflow.md) | The workflow; train/val/test; cross-validation; bias-variance; no-free-lunch |
| [lecture-notes/02-linear-regression-properly.md](./lecture-notes/02-linear-regression-properly.md) | Closed-form, gradient descent, sklearn; the five assumptions; residual plots; L1/L2 |
| [lecture-notes/03-logistic-regression-and-classification-metrics.md](./lecture-notes/03-logistic-regression-and-classification-metrics.md) | Log-odds, cross-entropy, threshold; precision / recall / F1 / ROC-AUC / PR-AUC |
| [exercises/README.md](./exercises/README.md) | Index of exercises |
| [exercises/exercise-01-train-test-split.py](./exercises/exercise-01-train-test-split.py) | Split correctly; cross-validate; spot a leak; report the right number |
| [exercises/exercise-02-linear-regression-by-hand.py](./exercises/exercise-02-linear-regression-by-hand.py) | Fit linear regression three ways and verify they agree to 1e-6 |
| [exercises/exercise-03-sklearn-pipeline.py](./exercises/exercise-03-sklearn-pipeline.py) | Build a `Pipeline` + `ColumnTransformer` that does not leak |
| [challenges/README.md](./challenges/README.md) | Index of weekly challenges |
| [challenges/challenge-01-beat-the-baseline-on-housing.md](./challenges/challenge-01-beat-the-baseline-on-housing.md) | Beat a published baseline on a real housing dataset |
| [quiz.md](./quiz.md) | 10 multiple-choice questions with an answer key |
| [homework.md](./homework.md) | Six practice problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Full spec for the Ames Housing price-prediction mini-project |

---

## Stretch goals

- Read **chapter 2 of *The Elements of Statistical Learning*** (Hastie, Tibshirani, Friedman — free PDF, Stanford). The bias-variance decomposition is on pages 24–26. Annotate the figure.
- Read **chapter 3 of *An Introduction to Statistical Learning*** (free PDF). The linear-regression chapter is the gentlest correct introduction in print. Skim the lab sections; the R code is incidental.
- Implement **ridge regression by hand**: the closed-form is `β̂ = (XᵀX + λI)⁻¹Xᵀy`. Verify it matches `Ridge(alpha=λ)` from scikit-learn on the diabetes dataset.
- Replicate **Andrew Gelman's "secret weapon" plot** (a forest plot of coefficients across many subgroups). It is one chart that captures most of what regression interpretation looks like in the wild.
- Read **scikit-learn's "Common pitfalls and recommended practices"** (free, on the official site). It is twenty minutes of reading that prevents twenty hours of wrong leaderboards.

---

## What you will *not* do this week

You will not:

- Build a deep neural network (Weeks 7 and 8).
- Train a gradient-boosted tree (Week 5 — and it will be the same dataset, on purpose).
- Use AutoML, AutoGluon, or "throw seventy models at it" frameworks. The point of this week is *understanding* one family of models well enough to defend the result.
- Use any model whose name contains the word "transformer." We are in 2026; the temptation is real. Resist for one more week.
- Write a Streamlit app, an API, or a dashboard around the model. Deployment lives in Week 11.
- Train on the test set. Not once. Not even "just to see." The test set is checked at most twice in the entire project, and once is fine.

That is deliberate. A correctly-evaluated linear model is the baseline every more-complex model will be measured against for the rest of the course. Without this week's discipline, every later "improvement" is unfalsifiable.

---

## A note on the EXPERIMENT cards

Starting this week, lectures contain `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The cards are not graded and they are not optional. They are the difference between reading about a model and having it in your fingers. Every concept that has fooled a generation of data scientists has an `EXPERIMENT` card that would have saved them an afternoon.

---

## Up next

[Week 5 — Trees, Forests, and Gradient Boosting](../week-05/) — once you have pushed your Ames Housing notebook to your `crunch-ai-portfolio-<yourhandle>` repo. Week 5 reuses the same dataset; the goal is to beat the linear-model score you ship this week by ≥10% relative error, *and* to explain why the trees won.
