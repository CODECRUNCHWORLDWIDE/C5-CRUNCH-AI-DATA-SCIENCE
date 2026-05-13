# Week 5 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **scikit-learn — "Decision Trees" user guide** (the canonical first hour on impurity, stopping rules, and the recursive split):
  <https://scikit-learn.org/stable/modules/tree.html>
- **scikit-learn — "Ensemble methods" user guide** (bagging, random forests, AdaBoost, gradient boosting, `HistGradientBoosting*`):
  <https://scikit-learn.org/stable/modules/ensemble.html>
- **scikit-learn — "Permutation feature importance"** (twenty minutes; prevents misuse of `feature_importances_`):
  <https://scikit-learn.org/stable/modules/permutation_importance.html>
- **XGBoost — "Introduction to Boosted Trees"** (the official one-page derivation; the second-order Taylor expansion sits at the top):
  <https://xgboost.readthedocs.io/en/stable/tutorials/model.html>
- **LightGBM — "Features"** (one page; the leaf-wise growth and histogram-binning innovations in plain prose):
  <https://lightgbm.readthedocs.io/en/stable/Features.html>
- **SHAP — "An introduction to explainable AI with Shapley values"** (notebook tutorial; the right place to start; sober about caveats):
  <https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html>

## The official docs you will bounce between all week

- **scikit-learn API reference** (the source of truth for every estimator's parameters; bookmark it):
  <https://scikit-learn.org/stable/api/index.html>
- **scikit-learn `DecisionTreeRegressor`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html>
- **scikit-learn `RandomForestRegressor`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>
- **scikit-learn `HistGradientBoostingRegressor`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html>
- **XGBoost Python API**:
  <https://xgboost.readthedocs.io/en/stable/python/python_api.html>
- **LightGBM Python API**:
  <https://lightgbm.readthedocs.io/en/stable/Python-API.html>
- **SHAP API reference** (the `Explainer`, `TreeExplainer`, and the plotting functions):
  <https://shap.readthedocs.io/en/latest/api.html>

## Free textbooks (the canon)

- **Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*** (free PDF from Stanford):
  <https://hastie.su.domains/ElemStatLearn/>
  Read **chapter 9** for trees, **chapter 10** for boosting (Friedman is one of the chapter authors and the inventor of the gradient-boosting machine), **chapter 15** for random forests.
- **James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning*, 2nd edition** (free PDF):
  <https://www.statlearning.com/>
  Read **chapter 8** end-to-end. The trees chapter is the gentlest correct introduction in print; the boosting section is shorter than ESL's but covers the four hyperparameters that matter.
- **Christoph Molnar — *Interpretable Machine Learning*** (free online, CC-BY):
  <https://christophm.github.io/interpretable-ml-book/>
  Read **chapters 5.5 (Permutation Feature Importance)**, **5.10 (SHAP)**, and **5.1 (Partial Dependence Plot)**. It is the one-stop reference for the interpretability tools we use in Lecture 3 and Challenge 1.
- **Kevin Murphy — *Probabilistic Machine Learning: An Introduction*** (free PDF, MIT Press, 2022):
  <https://probml.github.io/pml-book/book1.html>
  Chapter 18 (trees, forests, boosting) is the rigorous Bayesian-flavored treatment if you want it.

## The data sources we use this week

All public, all free to download:

- **Ames Housing dataset** (Dean De Cock, 2011 — the same one from Week 4):
  Kaggle CSV: <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>
  Data card: <https://jse.amstat.org/v19n3/decock.pdf>
- **scikit-learn's `fetch_california_housing`** (regression warm-up; 20,640 rows, 8 features):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>
- **scikit-learn's `load_diabetes`** (small enough that an exercise tree fits in milliseconds):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html>
- **OpenML's "adult" / "credit-g" / "kick" datasets** (for the tabular-classification stretch goals, all free via `fetch_openml`):
  <https://www.openml.org/>

## The papers (free PDFs, the originals)

The tree-ensemble field has the unusual property that the founding papers are all free, all readable, and mostly under twenty pages. Read at least the first one this week.

- **Leo Breiman — "Random Forests"** (2001). The founding paper. Fifteen pages. The bagging argument, the column-subsampling rationale, and the out-of-bag estimate are all in plain English.
  <https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf>
- **Jerome Friedman — "Greedy Function Approximation: A Gradient Boosting Machine"** (2001). The founding paper for gradient boosting. The "fit a tree to the residuals, shrink, add" loop and the functional-gradient-descent framing are here.
  <https://jerryfriedman.su.domains/ftp/trebst.pdf>
- **Tianqi Chen and Carlos Guestrin — "XGBoost: A Scalable Tree Boosting System"** (KDD 2016). The second-order Taylor expansion that made XGBoost fast.
  <https://arxiv.org/abs/1603.02754>
- **Guolin Ke et al. — "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"** (NeurIPS 2017). The histogram-binning + leaf-wise-growth + GOSS sampling story.
  <https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>
- **Scott Lundberg and Su-In Lee — "A Unified Approach to Interpreting Model Predictions"** (NeurIPS 2017). The SHAP paper. The game-theory derivation matters.
  <https://arxiv.org/abs/1705.07874>

## The math you need this week

Less than you might expect:

- **Weighted averages.** Every leaf prediction in a regression tree is a weighted mean. Every random-forest prediction is a mean of leaf predictions.
- **Variance of a mean.** `Var(mean of n iid estimators) = Var(one estimator) / n`. The argument for bagging in one line. (The catch is that bootstrap-resampled trees are not iid, only de-correlated.)
- **The chain rule** in one place: deriving the gradient-boosting update for arbitrary differentiable losses. We only do it in one footnote; if you want the full derivation, ESL chapter 10.
- **A second-order Taylor expansion** of the loss around the previous ensemble's prediction. This is the XGBoost trick; we sketch it but do not require you to re-derive it.
- **Shapley values** from cooperative game theory. The definition is one summation; the *uniqueness* argument (the four axioms — efficiency, symmetry, dummy, additivity — uniquely determine the Shapley value) is the part worth reading.
- **No measure theory, no convex optimization, no functional analysis.** Save them for the elective.

If you want the math properly written out, ESL chapters 9 and 10 are forty pages and cover every formula we use in this week's lectures.

## Tools you will use this week

- **`scikit-learn`** ≥ 1.5: `pip install "scikit-learn>=1.5,<2"`.
- **`xgboost`** ≥ 2.0: `pip install "xgboost>=2.0,<3"`. The 2.0 release shipped categorical-feature support and the modern Python sklearn-compatible API.
- **`lightgbm`** ≥ 4.0: `pip install "lightgbm>=4.0,<5"`. On macOS you may need `brew install libomp` first for the OpenMP runtime.
- **`shap`** ≥ 0.45: `pip install "shap>=0.45,<1"`. Optional for the lectures, required for the challenge.
- **`pandas`** ≥ 2.2, **`numpy`** ≥ 2, **`matplotlib`** ≥ 3.8 (from prior weeks).
- **`pytest`** ≥ 8 for the exercises.

A `requirements.txt` snippet for the week:

```text
pandas>=2.2,<3
numpy>=2,<3
scipy>=1.11,<2
scikit-learn>=1.5,<2
xgboost>=2.0,<3
lightgbm>=4.0,<5
shap>=0.45,<1
matplotlib>=3.8,<4
seaborn>=0.13,<1
pytest>=8
```

### Installation gotchas

- On **macOS arm64**, LightGBM needs `libomp` from Homebrew (`brew install libomp`) or it raises `OSError: dlopen ... libomp.dylib`. The pip wheel does not bundle OpenMP.
- On **Windows**, both XGBoost and LightGBM ship working wheels; no extra setup.
- On **Linux**, both work out of the box on Ubuntu 22.04+.
- **SHAP** has a transitive dependency on `numba` for some explainers; the `TreeExplainer` (the one we use) does not need it, so a minimal `pip install shap` is fine.

## Videos (free, no signup)

- **StatQuest with Josh Starmer** — the episodes on **decision trees**, **random forests**, **AdaBoost**, **gradient boosting (regression)**, **gradient boosting (classification)**, and **XGBoost (Part 1: Regression)** are each ten to fifteen minutes and unusually rigorous for video. Recommended in this order.
- **Tianqi Chen's KDD 2016 talk on XGBoost** (free on YouTube). The first ten minutes are the second-order Taylor expansion in slides; the rest is the engineering. Worth twenty minutes.
- **3Blue1Brown** has nothing on trees, but his **"backpropagation calculus"** episode (free, fifteen minutes) is the cleanest single explanation of the chain rule you will find. The same chain rule shows up once in Lecture 3 when we derive the gradient-boosting update.

## Open-source projects to read (in this order)

You can learn more from one hour reading the scikit-learn tree code than from three hours of tutorials.

1. **scikit-learn `DecisionTreeRegressor`** — the Cython-accelerated tree builder lives in `sklearn/tree/_classes.py` and `_tree.pyx`. The Python wrapper is what you read first:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py>
2. **scikit-learn `RandomForestRegressor`** — the bagging-of-trees implementation, including the OOB-prediction logic:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py>
3. **scikit-learn `HistGradientBoostingRegressor`** — the histogram-binned boosted-trees implementation:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_hist_gradient_boosting/>
4. **XGBoost `xgboost.sklearn.XGBRegressor`** — the sklearn-compatible wrapper around the underlying C++ booster:
   <https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py>
5. **SHAP `TreeExplainer`** — the fast-tree-specific Shapley-value algorithm; the docstring at the top is a one-page summary of the trick (TreeSHAP runs in polynomial time instead of exponential):
   <https://github.com/shap/shap/blob/master/shap/explainers/_tree.py>

## Cheat sheets (one-page references)

- **scikit-learn — "Choosing the right estimator"** (the official flowchart; trees and forests are in the right-hand branch):
  <https://scikit-learn.org/stable/machine_learning_map.html>
- **XGBoost — "Parameters"** (the full hyperparameter reference; long, but only four entries matter on small data):
  <https://xgboost.readthedocs.io/en/stable/parameter.html>
- **LightGBM — "Parameters"** (same shape; same four entries):
  <https://lightgbm.readthedocs.io/en/stable/Parameters.html>
- **The C5 "four hyperparameters that matter" card** — Lecture 3, Section 7. One printable page; tape it to your monitor.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Decision tree** | A model that predicts by routing a row through a sequence of axis-aligned threshold tests, returning the mean (regression) or majority (classification) of the leaf it lands in. |
| **Impurity** | A scalar summary of "how mixed are the labels in this node." Gini, entropy, MSE are three common choices; the first two are for classification, MSE is for regression. |
| **Gini impurity** | `1 − sum_k p_k²`. Zero when one class has all the rows; maximal when classes are evenly mixed. |
| **Entropy** | `− sum_k p_k log p_k`. Zero when one class has all the rows; maximal at uniform. Differs from Gini by less than a percent of split scores in practice. |
| **Split gain** | The drop in impurity (weighted by node size) from the best (feature, threshold) split. Trees grow by maximizing split gain greedily. |
| **`max_depth`** | The longest root-to-leaf path. The single most effective regularizer for a tree. Default in sklearn is `None` (unlimited); the right answer on small data is usually 4–8. |
| **`min_samples_leaf`** | A leaf must contain at least this many training rows. Bigger values smooth the tree; smaller values let it memorize. |
| **Bagging** | Bootstrap aggregating. Sample `n` rows with replacement to make a new training set, fit a tree, repeat, average the predictions. Reduces variance. |
| **Bootstrap sample** | A sample of size `n` drawn with replacement from a dataset of size `n`. About 63% of rows appear at least once; the other ~37% are "out-of-bag" for that tree. |
| **OOB estimate** | The prediction for each row, averaged over the trees that did *not* see that row in their bootstrap sample. Free cross-validation. |
| **Random forest** | Bagging plus column-subsampling at each split. `max_features` controls the column-subsampling. |
| **Gradient boosting** | Fit a tree to the residuals of the previous ensemble, shrink its predictions by a learning rate, add. Repeat for `n_estimators` rounds. |
| **`learning_rate`** | The shrinkage applied to each new tree's predictions before adding it to the ensemble. Smaller `learning_rate` + larger `n_estimators` is almost always better; the tradeoff is wall-clock. |
| **`n_estimators`** | The number of boosting rounds (or trees in a forest). Use early stopping for boosting; just pick "many" for a forest. |
| **`max_depth`** (in boosting) | The depth of each individual booster tree. 3–8 is the usual range; deeper trees overfit faster. |
| **`num_leaves`** (LightGBM) | The number of leaves per tree. LightGBM grows leaf-wise, so `num_leaves` is the more natural knob than `max_depth`. Default 31 is a reasonable starting point. |
| **Early stopping** | Stop adding trees when the validation-set score has not improved for `early_stopping_rounds`. The hyperparameter that picks `n_estimators` for you. |
| **Feature importance (Gini / split-gain)** | The total impurity reduction attributable to each feature, summed over splits in the ensemble. Free from the fit. Biased toward high-cardinality features. |
| **Permutation importance** | Permute one feature's values in the validation set, measure the score drop. Model-agnostic. Costs `n_features` predict passes. |
| **SHAP value** | The Shapley value of a feature for one prediction: the average over all feature orderings of the marginal contribution. Theoretically principled. `TreeExplainer` makes it polynomial-time on trees. |
| **Partial dependence plot** | The model's average prediction as one feature varies and all others are marginalized. Good for "what does the model think this feature does?" Misleading when features are correlated. |
| **Histogram binning** | Pre-bucket each continuous feature into ~256 bins before training. Replaces a sort over continuous values with a histogram scan. The single trick that makes `HistGradientBoostingRegressor`, XGBoost (with `tree_method="hist"`), and LightGBM fast. |

---

*If a link 404s, please open an issue so we can replace it.*
