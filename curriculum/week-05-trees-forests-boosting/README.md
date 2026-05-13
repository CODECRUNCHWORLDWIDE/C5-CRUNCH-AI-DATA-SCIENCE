# Week 5 — Trees, Forests, and Gradient Boosting

> *A linear model draws a single hyperplane through the feature space and asks every row to fall on the right side of it. A tree draws axis-aligned rectangles and asks each row which box it belongs in. Most tabular datasets — especially ones with thresholds, interactions, and missing values — are full of rectangles. That is the entire reason gradient-boosted trees won the 2010s of applied ML.*

Welcome to week five of **C5 · Crunch AI / Data Science**. Week 4 ended with a defended `RidgeCV` baseline on Ames Housing: cross-validated, residual-checked, coefficient-interpreted. That number — your dollar-RMSE on the held-out test set — is the number this week has to beat.

The plan is concrete: build a **decision tree** by hand so you know what `DecisionTreeRegressor` is doing under the hood, then a **random forest** so you understand why bagging helps, then **gradient-boosted trees** with the three libraries every working data scientist uses (scikit-learn's `HistGradientBoosting*`, XGBoost, LightGBM). The mini-project re-runs the Week 4 Ames pipeline with a boosted model and asks for a **≥10% relative RMSE improvement** with diagnostics that explain *why* the trees won.

Three warnings before we start, because Week 5 is where the temptation to "just use XGBoost" becomes overwhelming:

1. **Trees are not magic. They are histograms with branching.** Every gradient-boosted tree fit on tabular data is, at heart, a long sequence of "split the rows on a threshold, predict the mean of each side, then fit the residuals with another tree." Understanding the recursive split is the lesson; the library is a fast implementation of it.
2. **The default hyperparameters are not your friend.** `XGBRegressor()` with no arguments will overfit a small dataset before lunch. `LightGBM` with `num_leaves=31` and `n_estimators=100` is calibrated for medium-sized problems, not for the 1,460 rows of Ames. We will spend a deliberate hour on the four hyperparameters that matter (`learning_rate`, `n_estimators`, `max_depth` / `num_leaves`, `min_samples_leaf`) and ignore the eighty-seven others.
3. **A win on the leaderboard does not buy you interpretation for free.** Gradient-boosted trees do not have coefficients you can read like a linear model. They have feature importances, SHAP values, and partial-dependence plots — three different lenses, each of which can mislead in a different way. Lecture 3 and Challenge 1 spend serious time on the difference between Gini importance, permutation importance, and SHAP.

We target **scikit-learn 1.5+**, **XGBoost 2.0+**, **LightGBM 4.0+**, **numpy 2.x**, and **pandas 2.2+**. The APIs are stable; the `HistGradientBoostingRegressor`, `XGBRegressor`, and `LGBMRegressor` interfaces you learn here are the ones you will use in 2030.

---

## Learning objectives

By the end of this week, you will be able to:

- **Derive** the Gini impurity and the entropy of a split from their definitions, and recognize them as two slightly different ways of measuring "how mixed are the labels in this node."
- **Implement** a decision-tree regressor *from scratch* in pure NumPy (the recursive split, the best-feature-and-threshold search, the leaf prediction) and verify it agrees with `sklearn.tree.DecisionTreeRegressor` to floating-point tolerance on a small problem.
- **Read** a fitted decision tree as a sequence of axis-aligned thresholds. Recognize when a tree is overfitting from its depth and leaf count alone.
- **Explain** *bagging* (bootstrap aggregating) in one paragraph: why averaging many high-variance trees reduces variance without inflating bias. State the random-forest extra wrinkle (column subsampling at each split) and what it buys you.
- **Tune** the four hyperparameters that actually matter for tree ensembles: `n_estimators`, `max_depth` / `num_leaves`, `learning_rate`, and `min_samples_leaf` / `min_child_weight`. Ignore the rest until they are necessary.
- **Write down** the gradient-boosting algorithm in five lines: "fit a tree to the residuals, add a shrunken version of its predictions, repeat." Recognize the connection to functional gradient descent.
- **Choose** between `HistGradientBoostingRegressor`, XGBoost, and LightGBM with a one-paragraph justification, and know that for ≤100k tabular rows the three give RMSE numbers that agree to a few percent.
- **Compute** three flavors of feature importance — Gini / split-gain importance, permutation importance, and SHAP values — and name the failure mode of each.
- **Beat** the Week 4 linear-model RMSE on Ames Housing by **≥10% relative** with a boosted-tree pipeline, with the lift attributable to specific, named choices.
- **Pass** every `pytest` case on the Week 5 exercises.

---

## Prerequisites

- **Weeks 1, 2, 3, and 4 complete.** In particular, you have the Week 4 `ames.ipynb` checked into your `crunch-ai-portfolio-<yourhandle>` repo, with a documented test RMSE in dollars. That number is the bar this week.
- A working **Python 3.11+** install (we use 3.12).
- scikit-learn 1.5+ (`pip install "scikit-learn>=1.5,<2"`). We use `tree`, `ensemble.RandomForestRegressor`, `ensemble.HistGradientBoostingRegressor`, `inspection.permutation_importance`, plus the `Pipeline` / `ColumnTransformer` from Week 4.
- XGBoost 2.0+ (`pip install "xgboost>=2.0,<3"`). The 2.0 release shipped categorical-feature support and the modern Python API; pre-2.0 tutorials use a different interface.
- LightGBM 4.0+ (`pip install "lightgbm>=4.0,<5"`). 4.0 is the current stable line; the API is identical to 3.x for the parts we use.
- SHAP 0.45+ (`pip install "shap>=0.45,<1"`). Optional for the lectures, required for Challenge 1.
- pandas 2.2+ and numpy 2.x (from prior weeks).
- matplotlib 3.8+ (from Week 3 — we plot feature importances and partial-dependence curves).
- `pytest` for the exercise smoke tests.

No GPU. XGBoost and LightGBM both have GPU backends, but on Ames-sized data CPU is 5–20× *faster* than GPU because of the launch overhead. We stay on CPU all week.

---

## Topics covered

- **The recursive split.** A decision tree is the answer to "what is the single best (feature, threshold) split right now?" applied recursively until a stopping rule fires. Greedy. Local. Fast.
- **Impurity measures.** Gini, entropy, and mean-squared-error for regression. The first two agree on most splits; the choice rarely matters in practice, but the definitions matter for the exam.
- **Stopping rules.** `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`. The five knobs that turn a tree from "memorizing every row" to "predicting the mean."
- **Decision trees from scratch.** A 200-line NumPy implementation that agrees with sklearn to floating-point tolerance on regression problems. Exercise 1.
- **Bagging.** Bootstrap a training set, fit a tree, repeat, average. The variance reduction from `n` independent estimators is `Var(estimator) / n`; the catch is that bootstrap-sampled trees are not independent, only de-correlated.
- **Random forests.** Bagging plus column-subsampling at each split. The column-subsampling decorrelates the trees further, which is why a forest beats a bagged-trees-only ensemble in practice.
- **The out-of-bag (OOB) estimate.** Free cross-validation: each tree was fit without ~37% of the rows; predicting those rows from that tree gives an unbiased generalization estimate without a separate validation set.
- **Gradient boosting.** Fit a tree to the residuals of the previous ensemble. Shrink. Add. Repeat. The algorithm in five lines.
- **`HistGradientBoostingRegressor`** (scikit-learn's histogram-binned boosted-trees implementation; the in-library answer for tabular data since 1.0).
- **XGBoost** (Chen and Guestrin, 2016). The implementation that defined gradient boosting on Kaggle for ten years. We learn the four hyperparameters that matter.
- **LightGBM** (Microsoft, 2017). Faster than XGBoost on most datasets because it splits leaf-wise rather than level-wise, and bins continuous features into 256 buckets up front. Same accuracy, different defaults.
- **Categorical features in trees.** `HistGradientBoosting` and LightGBM natively split on categorical features; XGBoost has supported them since 2.0. The native-categorical path is usually a percentage point of RMSE better than one-hot encoding for high-cardinality features.
- **Feature importance, three ways.** Gini / split-gain importance (free from the fit), permutation importance (model-agnostic, costs `n_features` predict passes), SHAP values (theoretically principled, computationally expensive). Each has a failure mode.
- **The hyperparameter four.** `n_estimators`, `learning_rate`, `max_depth` (or `num_leaves`), `min_samples_leaf` (or `min_child_weight`). Sweep these four and ignore the rest until they are necessary.
- **Early stopping.** The "free hyperparameter" — let the boosting library tell you when adding more trees stops helping by watching a validation-set score. Saves a coarse grid-search over `n_estimators`.

---

## Weekly schedule

Target about **38 hours**. Some sections click in twenty minutes; some take three hours. Treat the table as a budget, not a contract.

| Day       | Focus                                                  | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|--------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Decision trees from scratch; Gini, entropy, MSE        |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | Random forests; bagging; OOB; feature subsampling      |   2.5h   |   2h      |     0h     |   0.5h    |   1h     |     0h       |   1h       |    7h       |
| Wednesday | Gradient boosting; XGBoost; LightGBM; HistGBR          |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0h       |   6.5h      |
| Thursday  | Feature importance; SHAP; categorical handling         |   2h     |   1h      |     2h     |   0.5h    |   1h     |     2h       |   0h       |    8.5h     |
| Friday    | Mini-project: tune the boosted model on Ames           |   0h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    6h       |
| Saturday  | Mini-project: SHAP + report; write-up                  |   0h     |   0h      |     0h     |   0h      |   1h     |     3h       |   0h       |    4h       |
| Sunday    | Quiz, review, push to portfolio repo                   |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                        | **10.5h**| **8h**    | **2h**     | **3h**    | **6h**   | **8h**       | **2h**     |  **39.5h**  |

The schedule overshoots 38h by 1.5h on purpose — Week 5 has more libraries to install and a SHAP plot to debug, and the buffer is realistic.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Free docs for scikit-learn 1.5+, XGBoost 2.0+, LightGBM 4.0+, SHAP, the original tree-ensemble papers |
| [lecture-notes/01-decision-trees-from-scratch.md](./lecture-notes/01-decision-trees-from-scratch.md) | Gini, entropy, MSE-split; the recursive algorithm; build one from scratch and check it against sklearn |
| [lecture-notes/02-random-forests-and-bagging.md](./lecture-notes/02-random-forests-and-bagging.md) | Bagging math; random forests as bagging + column subsampling; out-of-bag estimates; when forests beat single trees |
| [lecture-notes/03-gradient-boosting-xgboost-lightgbm.md](./lecture-notes/03-gradient-boosting-xgboost-lightgbm.md) | Functional gradient descent; the four hyperparameters that matter; XGBoost vs LightGBM vs HistGBR; early stopping |
| [exercises/README.md](./exercises/README.md) | Index of exercises |
| [exercises/exercise-01-tree-from-scratch.py](./exercises/exercise-01-tree-from-scratch.py) | Implement a regression tree in pure NumPy; verify it matches sklearn |
| [exercises/exercise-02-random-forest-sklearn.py](./exercises/exercise-02-random-forest-sklearn.py) | Fit a random forest; read the OOB estimate; sweep `max_features`; compare to a single tree |
| [exercises/exercise-03-xgboost-vs-lightgbm.py](./exercises/exercise-03-xgboost-vs-lightgbm.py) | Fit XGBoost and LightGBM on the same data with matched hyperparameters; compare RMSE and wall-clock |
| [challenges/README.md](./challenges/README.md) | Index of weekly challenges |
| [challenges/challenge-01-feature-importance-vs-shap.md](./challenges/challenge-01-feature-importance-vs-shap.md) | Where Gini importance, permutation importance, and SHAP disagree — and which one to trust when |
| [quiz.md](./quiz.md) | 10 multiple-choice questions with an answer key |
| [homework.md](./homework.md) | Six practice problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Beat Week 4's Ames RMSE by ≥10% with a boosted-tree pipeline |

---

## Stretch goals

- Read **Breiman's "Random Forests" (2001)** — the original paper, fifteen pages, free PDF. The bagging argument, the column-subsampling rationale, and the OOB estimate are all there in the prose, in plainer English than the textbooks.
- Read **Chen and Guestrin's "XGBoost: A Scalable Tree Boosting System" (KDD 2016)** — also free. The second-order Taylor expansion of the loss is the entire reason XGBoost was faster than scikit-learn's `GradientBoostingRegressor` in 2016.
- Read **Friedman's "Greedy Function Approximation: A Gradient Boosting Machine" (2001)** — the founding paper for gradient boosting. The "functional gradient descent" framing is from this paper.
- Read **Lundberg and Lee's "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)** — the SHAP paper. The game-theory derivation matters; understanding why SHAP values are the unique solution to a small list of axioms is the difference between using SHAP and quoting SHAP.
- Read **the scikit-learn user guide on "Decision Trees"** end-to-end (free, on the official site). It is one hour of reading that prevents one weekend of debugging.

---

## What you will *not* do this week

You will not:

- Build a deep neural network (Weeks 7 and 8).
- Train an LLM. Train a transformer. Train anything whose name contains the word "attention." It is 2026; the temptation is real; resist for two more weeks.
- Use AutoML (`AutoGluon`, `H2O AutoML`, `FLAML`) to brute-force the mini-project. The point is to *choose* a model and a hyperparameter and *defend* the choice, not to wrap a Bayesian optimizer around the entire library.
- Build a Streamlit app, an API, or a Docker container around the model. Deployment lives in Week 11.
- Re-train on the test set. Not once. Not even "just to see." The test set is checked at most twice, total.
- Tune past three or four hyperparameters per model. The XGBoost and LightGBM docs list dozens; on Ames-sized data, two thirds of them have negligible effect.

That is deliberate. The point of Week 5 is not to win Kaggle. The point is to ship a defended gradient-boosted-trees model with diagnostics that explain *why* it won, paired with a Week 4 baseline you fit yourself. Without that pairing, every "GBM beats linear" claim is unfalsifiable.

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The cards are not graded and they are not optional. They are the difference between reading about gradient boosting and being able to debug a stalled training loss. Every claim that has fooled a generation of Kagglers has an `EXPERIMENT` card that would have saved them an afternoon.

---

## Up next

[Week 6 — Unsupervised Learning and Dimensionality Reduction](../week-06/) — once you have pushed your Week 5 Ames notebook to your portfolio repo with a documented test RMSE that beats Week 4 by ≥10% relative. Week 6 leaves supervised learning for the unsupervised side of the map: k-means, hierarchical clustering, PCA, UMAP. The same discipline (baseline first, evaluate honestly, interpret out loud) carries over; only the metrics change.
