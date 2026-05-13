# Mini-Project — Beat the Week 4 Linear Model on Ames Housing

> Fit a gradient-boosted-trees model on the Ames Housing dataset and **beat the Week 4 `RidgeCV` test RMSE by ≥10% relative**. Document every choice. Explain why the trees won in two paragraphs of prose backed by a SHAP plot. The deliverable is a single notebook plus a two-page report that an engineering manager could read in five minutes and conclude "this person knows when to use which model and why."

This is the fifth artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 4 was the linear baseline; Week 5 is the comparison. Recruiters with taste read Weeks 4 and 5 side by side — they want to see whether you can move past linear models with discipline and whether you can explain the move in plain English.

**Estimated time:** 8 hours, spread across Thursday–Sunday.

---

## What you will build

A Jupyter notebook `ames_boosted.ipynb` plus a rendered `report.md` that:

1. **Loads** the Ames Housing dataset (the 1,460-row Kaggle training file — the same one Week 4 used).
2. **Splits** the data 80/20 with `random_state=42` (identical split to Week 4). Carves a small validation slice off the training set for early stopping.
3. **Loads** the Week 4 RidgeCV test RMSE as the **headline number to beat**. The W5 headline number is the *relative improvement* over W4 in test RMSE.
4. **Builds** a `Pipeline` + `ColumnTransformer` that handles numeric columns (impute, no scaling — trees don't need it) and categorical columns (impute, optional one-hot or native categorical). Tree models route differently than linear models; the pipeline reflects that.
5. **Fits** `HistGradientBoostingRegressor` (or XGBoost, or LightGBM — your choice) with early stopping. Tunes the four hyperparameters that matter (Lecture 3, Section 3) and *only* those four.
6. **Evaluates** with 5-fold cross-validation in `log(SalePrice)` units, then back-transforms to dollar RMSE for the headline.
7. **Diagnoses** with a partial-dependence plot for the top three features and a SHAP summary plot.
8. **Compares** to the Week 4 RidgeCV result in a one-row "experiment card" table — W4 RMSE, W5 RMSE, percent improvement.
9. **Reports** in a 600–900 word executive summary that a non-technical reviewer can finish in five minutes.

The notebook is the working artifact. The `report.md` is the executive summary.

---

## The dataset

The **Ames Housing** dataset (Dean De Cock, 2011) is the same one Week 4 used:

- **1,460 rows** in the training set.
- **80 features.** 38 numeric, 23 nominal categorical, 23 ordinal categorical.
- **Substantial missing data.** Most "missing" values are not actually missing (`PoolQC=NaN` means "no pool"). Tree models handle this gracefully — `HistGradientBoostingRegressor`, XGBoost, and LightGBM all support missing values natively, so you can skip imputation for the trees-only path.
- **A right-skewed target.** As in Week 4, predicting `log(SalePrice)` is dramatically better-behaved than predicting `SalePrice` directly. Trees do not strictly need the log-transform (they are scale-invariant), but the log-transform usually still helps by 2–5% relative RMSE on this dataset because the squared-error loss is more symmetric in log-space.

Download from the Kaggle "House Prices — Advanced Regression Techniques" page (free): <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>. Use `train.csv`.

---

## Acceptance criteria

- [ ] A new directory `week-05/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `pandas>=2.2,<3`, `numpy>=2,<3`, `scipy>=1.11,<2`, `scikit-learn>=1.5,<2`, `xgboost>=2.0,<3` *or* `lightgbm>=4.0,<5` (or both), `shap>=0.45,<1`, `matplotlib>=3.8,<4`, `jupyter`.
- [ ] `jupyter nbconvert --to notebook --execute week-05/ames_boosted.ipynb` runs end-to-end without errors on a fresh clone.
- [ ] The notebook contains the **same 80/20 split with `random_state=42`** that Week 4 used. The trees see the same training set and are scored on the same test set.
- [ ] The notebook uses a **`ColumnTransformer`** (or the native categorical path) that handles numeric and categorical columns explicitly. No fit-then-split.
- [ ] The model uses **early stopping** with an explicit validation set. The `best_iteration_` / `n_iter_` is printed.
- [ ] The target is **`log1p(SalePrice)`**. Predictions are exponentiated back with `expm1` before computing RMSE in dollars.
- [ ] The **headline number** is the held-out test RMSE in dollars (rounded to the nearest $100). The CV RMSE is reported alongside ± SD.
- [ ] **Beats Week 4's test RMSE by ≥10% relative.** If W4 was \$28,500, W5 must be ≤ \$25,650. If W4 was \$32,000, W5 must be ≤ \$28,800.
- [ ] The hyperparameter choices are **documented in a small table**: `learning_rate`, `max_depth` (or `num_leaves`), `min_samples_leaf` (or `min_child_weight`), `n_estimators_chosen_by_early_stopping`.
- [ ] A **SHAP summary plot** is saved in `week-05/images/shap_summary.png`. The top three features by mean absolute SHAP are read in prose ("OverallQual, GrLivArea, and Neighborhood dominate the model's variance").
- [ ] A **partial-dependence plot** for the top three features is saved in `week-05/images/pdp.png`. The shapes — monotonic? threshold-like? interaction-revealing? — are read in prose.
- [ ] A **residual plot** (three panels, as in Week 4) is saved in `week-05/images/residuals.png`. The residual structure of the tree model is *different* from the linear model's; name the differences in prose.
- [ ] A `report.md` (~2 pages, 600–900 words) summarizes the problem, the method, the result (with the percent-improvement over W4 as the headline), and one honest limitation. No code.
- [ ] A `README.md` in `week-05/` explains: setup, data download, file layout, how to reproduce.
- [ ] The W4 and W5 numbers are summarized in a one-row table in the W5 `README.md` so a reviewer can see the comparison without opening either notebook.

---

## Suggested layout

```text
crunch-ai-portfolio-<yourhandle>/
├── README.md                    ← portfolio root
├── week-04/                     ← from Week 4
│   └── (RidgeCV pipeline, test RMSE in commit message)
└── week-05/
    ├── README.md                ← week-5 explainer with the W4 vs W5 table
    ├── requirements.txt
    ├── data/
    │   └── ames_train.csv       ← Kaggle 'train.csv'
    ├── images/
    │   ├── shap_summary.png
    │   ├── pdp.png
    │   ├── residuals.png
    │   └── feature_importance.png   ← Gini, permutation, SHAP side by side
    ├── ames_boosted.ipynb
    ├── ames_boosted.html        ← rendered preview
    └── report.md                ← 2-page executive summary
```

---

## Suggested order of operations

### Phase 1 — Reload the W4 numbers (30 min)

1. Open `ames_boosted.ipynb`. The first markdown cell is the **project header** plus a one-line citation of the Week 4 number you are trying to beat. ("W4 RidgeCV: test RMSE \$28,400 in dollars, log-RMSE 0.131 ± 0.012.")
2. Load `train.csv`. Use the *same* split as Week 4: `train_test_split(X, y, test_size=0.20, random_state=42)`. Verify the `y_test` is identical to the W4 one (e.g., by hashing it).

### Phase 2 — Untuned boosted model (1 hour)

Fit `HistGradientBoostingRegressor` (or your library of choice) with default hyperparameters and early stopping. This is your "lazy baseline" — what you get with one line of code. On Ames it is often already within 5% of the W4 linear model; sometimes already beating it. Record the number.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr_full, y_tr_full, test_size=0.15, random_state=42)

mdl = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=2000,
    max_depth=6,
    early_stopping=True,
    n_iter_no_change=50,
    random_state=42,
)
mdl.fit(X_tr_full, np.log1p(y_tr_full))   # log target, tree handles missing values
test_rmse = root_mean_squared_error(y_te, np.expm1(mdl.predict(X_te)))
print(f"test RMSE: ${test_rmse:,.0f}, n_iter chosen: {mdl.n_iter_}")
```

### Phase 3 — Tune the four (2 hours)

Sweep the four hyperparameters one at a time. Do not grid-search all four simultaneously — that wastes wall-clock and obscures which knob mattered.

1. **`learning_rate`** ∈ `{0.01, 0.03, 0.05, 0.1}`. Pick the one with the lowest CV RMSE.
2. **`max_depth`** ∈ `{3, 4, 5, 6, 8}` at the chosen LR.
3. **`min_samples_leaf`** ∈ `{5, 10, 20, 50}` at the chosen LR + depth.
4. **`n_estimators`** is chosen by early stopping. Set the upper bound to 5000.

Cross-validate each choice with 5-fold CV in `log(SalePrice)` units. Pick the combination with the lowest CV RMSE; *do not* score on the test set until the very end.

### Phase 4 — Diagnose (2 hours)

Three plots, in `week-05/images/`:

1. **`shap_summary.png`** — `shap.summary_plot(shap_values, X_test)`. The canonical SHAP figure; one feature per row, one dot per test row, colored by feature value. Lecture 3, Section 6 and Challenge 1.
2. **`pdp.png`** — `sklearn.inspection.PartialDependenceDisplay.from_estimator(mdl, X_test, features=top_three)`. Three panels, one per top-importance feature. Read the shape in prose.
3. **`residuals.png`** — three-panel residual plot (residuals vs predicted, histogram, Q-Q) on the dollar scale. Compare to the W4 residual plot. The shapes will differ; name the differences.
4. (Bonus) **`feature_importance.png`** — Gini, permutation, SHAP side by side, as in Challenge 1.

For each plot, write one paragraph in the notebook *interpreting* it.

### Phase 5 — Compare to W4 and freeze (1.5 hours)

Build the comparison table:

```text
┌─────────────────────────────────────────────────────────┐
│  EXPERIMENT  W05-ames-boosted                           │
│                                                         │
│  Goal:     Predict house prices (beat W4 by ≥10%)       │
│  Data:     Ames housing, n=1460                         │
│  Method:   HistGradientBoostingRegressor, log target    │
│  CV:       5-fold KFold                                 │
│  Metric:   RMSE in dollars                              │
│  Result:   $24,300 (W4 RidgeCV: $28,400) -- 14.4% better │
│  Status:   ✓ better than baseline                        │
└─────────────────────────────────────────────────────────┘
```

Final test-set scoring is **once**. If the test RMSE disagrees with the CV RMSE by more than 1–2 standard deviations, you have a leak. Find it. Otherwise, freeze.

Write `report.md`: 600–900 words. Sections: **Problem**, **Method** (with the hyperparameter table), **Result** (with the comparison-to-W4 numbers), **Why the trees won** (your two paragraphs of prose backed by the SHAP plot), **Limitations**. No code.

### Phase 6 — Export (30 min)

```bash
jupyter nbconvert --to html week-05/ames_boosted.ipynb
```

Push to the portfolio repo.

---

## What "great" looks like

A "great" mini-project hits all of the following:

- The model **beats W4 by ≥15% relative**, not just the 10% floor. A well-tuned `HistGradientBoostingRegressor` on Ames routinely lands in the \$22,000–\$25,000 RMSE range, comfortably below a W4 RidgeCV in the \$28,000–\$32,000 range. The 10% floor is the lower bound; 15%+ is the median.
- The **"why the trees won" prose is specific**, not generic. Generic: "trees capture nonlinearities." Specific: "the SHAP plot shows that for `OverallQual ≥ 8`, the model's prediction shifts upward by \$30k irrespective of `GrLivArea` — a step-function the linear model could not represent without an explicit interaction term."
- The **partial-dependence plots are read honestly**. The PDP for `GrLivArea` will show a near-monotonic curve with a kink around the median (a real Ames effect); the PDP for `Neighborhood` is a step function with the high-end neighborhoods on top. The plots are diagnostics, not decorations.
- The **residual plot is read as a comparison to W4's residual plot**, not in isolation. The tree model's residuals are usually less heteroscedastic than the linear model's (the log-target was already doing most of that work) but show different patches of high-error houses — name them.
- The **SHAP summary plot is in the report**, not just the notebook. A non-technical reader looking only at the report should see the SHAP plot and at least one sentence interpreting it.

A "good but not great" project beats W4 by 10–14%, includes the SHAP plot but does not interpret it, and lists hyperparameters without explaining how they were chosen.

A "needs work" project does not beat W4 (or beats it by less than 10%), does not include a SHAP plot, or compares to a different W4 number than the one in the committed Week 4 repo.

---

## Stretch goals

- **Compare three implementations.** Fit `HistGradientBoostingRegressor`, `XGBRegressor`, and `LGBMRegressor` with matched hyperparameters and report the three test RMSEs. They should agree to within a few percent; if they do not, your hyperparameters are not matched.
- **Run `permutation_importance` on the test set** and put it side by side with the SHAP ranking. The two should agree on the top 5 features and may disagree below. Challenge 1 walks through the comparison.
- **Add an interaction-feature ablation.** Drop the top SHAP feature and re-fit. By how much does the RMSE worsen? The drop is a rough lower bound on the feature's contribution; it disagrees with both SHAP and permutation importance in interesting ways.
- **Train a `StackingRegressor`** with `Ridge`, `RandomForestRegressor`, and `HistGradientBoostingRegressor` as base estimators and `Ridge` as the final estimator. Compare to the boosted-only model. On Ames the stack usually wins by ~1% and is rarely worth shipping — but it is the canonical "everything together" baseline.
- **Profile the deploy-time predict latency.** How many predictions per second does your tuned model handle? Compare to the W4 RidgeCV's predict speed. (The linear model will be ~100× faster; for some applications that matters.)

---

## Rubric

| Criterion | Weight | "Great" looks like |
|-----------|------:|--------------------|
| It runs | 15% | Fresh clone → `pip install -r requirements.txt` → `jupyter nbconvert --execute` → no errors |
| Pipeline + early stopping discipline | 15% | `ColumnTransformer` (or native categorical) end-to-end; early stopping on an explicit validation set; `best_iteration_` printed |
| Beats W4 by ≥10% | 20% | Headline percent-improvement ≥ 10%, ideally ≥ 15%; the comparison-to-W4 table is in `README.md` |
| Hyperparameter tuning | 10% | The four hyperparameters are tuned one at a time; each is justified in prose |
| Diagnostics (SHAP, PDP, residuals) | 20% | All three plots saved; each read in prose; SHAP plot is in the report |
| "Why the trees won" prose | 10% | Two paragraphs that cite specific SHAP / PDP observations, not generic ML talk |
| Report readability | 10% | The 2-page `report.md` could go to a hiring manager without explanation |

---

## What this exercises (and what comes next)

This mini-project exercises every concept from Week 5:

- The recursive tree split as the building block (Lecture 1, foundational).
- Bagging and out-of-bag estimation (Lecture 2; the stretch-goal `StackingRegressor` uses this internally).
- The gradient-boosting algorithm and the four hyperparameters that matter (Lecture 3, Sections 1 and 3).
- Early stopping as the free hyperparameter (Lecture 3, Section 5).
- Three flavors of feature importance, with the SHAP plot as the headline (Lecture 3, Section 6 and Challenge 1).
- The C5 discipline of comparing against a defended baseline (the W4 RidgeCV).

Week 6 leaves supervised learning for the unsupervised side of the map: clustering, dimensionality reduction, anomaly detection. The same discipline carries over — baseline first, evaluate honestly, interpret out loud — but the metrics change. After Week 6 you will have shipped six portfolio artifacts and be halfway through C5.

---

## Submission

Push your `crunch-ai-portfolio-<yourhandle>` repo. Open the `week-05/ames_boosted.html` link from the repo README so a reviewer with no Python install can read your work. Paste the comparison in the commit message:

```text
Week 5 mini-project: Ames Housing, boosted trees.

W4 RidgeCV test RMSE:                $28,400
W5 HistGradientBoosting test RMSE:   $24,100
relative improvement:                15.1%
hyperparameters: lr=0.05, max_depth=6, min_samples_leaf=20, n_iter=482 (early stop)
SHAP top-3: OverallQual, GrLivArea, Neighborhood
```

That is the artifact recruiters will see. The percent-improvement number is the one the W4 / W5 pairing is built around.
