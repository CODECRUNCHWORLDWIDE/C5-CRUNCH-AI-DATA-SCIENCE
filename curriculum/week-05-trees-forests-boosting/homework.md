# Week 5 — Homework

Six problems, about six hours total. Commit each in your Week 5 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — A tree-depth sweep (1 hour)

Load `sklearn.datasets.fetch_california_housing(as_frame=True)`. Build a pipeline `Pipeline([("model", DecisionTreeRegressor(random_state=42, min_samples_leaf=5))])`. Sweep `max_depth` over `[2, 4, 6, 8, 10, None]` and record train RMSE and 5-fold-CV RMSE at each setting.

Save the plot as `homework/01-tree-depth.png`. In `homework/01-tree-depth.md` (~150 words):

1. At what depth does train RMSE first drop below 0.5?
2. At what depth does CV RMSE bottom out, and what is the CV RMSE there?
3. Where does the train-CV gap blow open? That is the depth at which the single tree starts to memorize.
4. What does this curve tell you that a single number cannot?

**Acceptance.**

- PNG saved with axes labeled and a sentence-form title.
- `.md` is honest about where train and CV diverge.

---

## Problem 2 — A forest, with OOB tracking n_estimators (1 hour)

On the same California Housing dataset, fit a `RandomForestRegressor(max_features="sqrt", oob_score=True, min_samples_leaf=5, n_jobs=-1, random_state=42)` and record the OOB R² as `n_estimators` ramps from 10 to 1000 (in 10–20 steps). Use `warm_start=True` to add trees incrementally rather than re-fitting from scratch each time.

Save the curve as `homework/02-forest-n-estimators.png`. In `homework/02-forest-n-estimators.md` (~150 words):

1. Where does the OOB R² plateau?
2. What value of `n_estimators` is "enough"? Is 500 enough? 1000?
3. Briefly explain why more trees cannot overfit a random forest (Lecture 2, Section 5).
4. How would this plot change shape for a gradient-boosted-trees model? (Hint: it would not be monotonic.)

---

## Problem 3 — Gradient boosting, learning-rate vs n_estimators (1 hour)

Fit `HistGradientBoostingRegressor` on California Housing for four `(learning_rate, max_iter)` pairs:

- `(0.5, 100)`
- `(0.1, 500)`
- `(0.05, 1000)`
- `(0.01, 5000)`

Use `early_stopping=True` and `n_iter_no_change=50` for all four. Record the test RMSE *and* the wall-clock training time.

Save the table as `homework/03-lr-vs-iter.md` (~100 words). The headline question: does the smaller learning rate win, and by how much, and how much extra wall-clock did it cost?

Acceptance: the smallest learning rate should win on RMSE by at least 0.005 (in units of `$100k` for California Housing), at a cost of 5–20× the wall-clock of the largest learning rate.

---

## Problem 4 — Beat a single tree with a forest on diabetes (45 min)

Load `sklearn.datasets.load_diabetes()`. Split 80/20 with `random_state=42`.

In `homework/04-tree-vs-forest.py`:

1. Fit `DecisionTreeRegressor(max_depth=None, min_samples_leaf=5, random_state=42)`. Print test RMSE.
2. Fit `RandomForestRegressor(n_estimators=500, max_features="sqrt", min_samples_leaf=5, oob_score=True, n_jobs=-1, random_state=42)`. Print OOB R² and test RMSE.
3. Print the train R² and test R² for both. Compute the train-test gap for each.

In `homework/04-tree-vs-forest.md` (~150 words):

1. By what percent did the forest beat the single tree on test RMSE?
2. By how much did the train-test gap narrow when moving from tree to forest?
3. The C5 answer: the gap narrows because bagging averages away the per-tree variance. Confirm that observation in your own data.

---

## Problem 5 — Three importances on a dataset of your choosing (45 min)

Pick one of: California Housing, diabetes, or a Kaggle / OpenML dataset you find interesting. Fit a `HistGradientBoostingRegressor` (or XGBoost, or LightGBM) with sensible hyperparameters and early stopping.

In `homework/05-importances.py`:

1. Compute `mdl.feature_importances_` (Gini).
2. Compute `sklearn.inspection.permutation_importance(mdl, X_test, y_test, n_repeats=10, random_state=42)`.
3. (Optional but encouraged) Compute SHAP values with `shap.TreeExplainer`.
4. Print all three rankings side by side, sorted by permutation rank.

In `homework/05-importances.md` (~200 words):

1. Which features do all three methods agree are important?
2. Where do the rankings disagree, and what is the likely cause? (Cardinality bias, correlation, training-vs-holdout — Challenge 1 walks through these.)
3. Which ranking would you lead with in a stakeholder presentation, and why?

If you do the [challenge](./challenges/challenge-01-feature-importance-vs-shap.md), this problem becomes a 15-minute extension of it on a different dataset.

---

## Problem 6 — Reflection (30 min)

Write `homework/06-reflection.md` (250–400 words) answering:

1. Did the from-scratch tree in Exercise 1 surprise you in how short the algorithm is? Why or why not? Where did the implementation get harder than you expected?
2. The bagging variance argument (`Var(mean) = Var / B`) is one of the few ML claims that has a real one-line derivation. Did seeing it on paper change how you think about ensembles?
3. Which of the four hyperparameters that matter (Lecture 3, Section 3) feels most like the one you have, in retrospect, ignored on past projects?
4. After this week, which model do you reach for first on a new tabular regression — linear, random forest, or gradient boosting — and why? Be specific about the situation.
5. What is the one habit from this week you want to keep when you move to Week 6 (unsupervised learning)? "Always start with a baseline" carries over; "always plot feature importances after every fit" might be your new one.

Honest is more valuable than polished.

---

## Time budget

| Problem | Time |
|--------:|----:|
| 1 | 1 h |
| 2 | 1 h |
| 3 | 1 h |
| 4 | 45 min |
| 5 | 45 min |
| 6 | 30 min |
| **Total** | **~5 h** |

(The schedule allocates 6h for homework; the remaining ~1h is buffer for installing XGBoost / LightGBM / SHAP without rushing.)

When done, push your Week 5 repo and start (or finish) the [challenge](./challenges/) and the [mini-project](./mini-project/README.md).
