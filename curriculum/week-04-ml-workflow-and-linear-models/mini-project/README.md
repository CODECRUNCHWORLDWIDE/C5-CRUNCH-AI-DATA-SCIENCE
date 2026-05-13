# Mini-Project — Predict House Prices on the Ames Housing Dataset

> Fit a regularized linear model that predicts `SalePrice` on the Ames Housing dataset. Beat a published baseline. Document every feature you build. Justify every choice. The deliverable is a single notebook plus a two-page report that an engineering manager could read in five minutes and conclude "this person has the workflow."

This is the fourth artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 1 was an image notebook; Week 2 was an EDA; Week 3 was a chart-recreation; Week 4 is the first **modeling** artifact. Recruiters with taste read this one carefully — it is the closest portfolio piece to a Kaggle submission with a defended write-up, and it is the model Week 5's gradient-boosted tree will be measured against.

**Estimated time:** 8 hours, spread across Thursday–Sunday.

---

## What you will build

A Jupyter notebook `ames.ipynb` plus a rendered `report.md` that:

1. **Loads** the Ames Housing dataset (the 1,460-row Kaggle training file).
2. **Splits** the data 80/20 (train + validation via cross-validation, then a held-out test).
3. **Baselines** the prediction with `DummyRegressor` and a strong baseline (linear regression on raw numeric features only).
4. **Builds** a `Pipeline` + `ColumnTransformer` that handles numeric columns (impute, scale), categorical columns (impute, one-hot encode), and produces an estimator-ready matrix.
5. **Fits** `RidgeCV` (and a `LassoCV` comparison) on the pipeline.
6. **Evaluates** with 5-fold cross-validation in `log(SalePrice)` units, then back-transforms to dollar RMSE for the headline.
7. **Diagnoses** with three residual plots and the standardized-coefficient table.
8. **Reports** in a 600–900 word executive summary that a non-technical reviewer can finish in five minutes.

The notebook is the working artifact. The `report.md` is the executive summary.

---

## The dataset

The **Ames Housing** dataset (Dean De Cock, 2011) is the modern replacement for the older Boston Housing dataset, which has well-known ethical issues and was removed from scikit-learn in 1.2. Ames is:

- **1,460 rows** in the training set, **1,459 rows** in the test set (which has no public `SalePrice` — we use cross-validation on the training set instead).
- **80 features.** A mix of:
  - **38 numeric**: `LotArea`, `GrLivArea`, `TotalBsmtSF`, `OverallQual`, `YearBuilt`, ...
  - **23 nominal categorical**: `Neighborhood`, `HouseStyle`, `RoofStyle`, ...
  - **23 ordinal categorical**: `ExterQual` (`Ex`/`Gd`/`TA`/`Fa`/`Po`), `BsmtCond`, ...
- **Substantial missing data.** Many "missing" values are not actually missing: `PoolQC = NaN` means "no pool," not "unknown pool quality." The data card matters.
- **A right-skewed target.** `SalePrice` ranges from $34,900 to $755,000 with a long right tail. Predicting `log(SalePrice)` is dramatically better-behaved than predicting `SalePrice` directly. The Kaggle leaderboard scores in `log(SalePrice)` RMSE for exactly this reason.

Download from <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data> (`train.csv`). The data card lives at <https://jse.amstat.org/v19n3/decock.pdf> (free PDF, the original paper).

---

## Acceptance criteria

- [ ] A new directory `week-04/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `pandas>=2.2,<3`, `numpy>=2,<3`, `scipy>=1.11,<2`, `scikit-learn>=1.5,<2`, `matplotlib>=3.8,<4`, `seaborn>=0.13,<1`, `jupyter`.
- [ ] `jupyter nbconvert --to notebook --execute week-04/ames.ipynb` runs end-to-end without errors on a fresh clone.
- [ ] The notebook contains an **80/20 split with `random_state=42`** and a **5-fold CV** on the train set. No leakage. All preprocessing is inside the `Pipeline`.
- [ ] The notebook uses a **`ColumnTransformer`** that handles numeric and categorical columns explicitly. Imputation for numerics. `OneHotEncoder(handle_unknown="ignore")` for categoricals.
- [ ] The model is **`RidgeCV`** (or `LassoCV`, or both for comparison). The chosen `alpha_` is printed.
- [ ] The target is **`log1p(SalePrice)`**. Predictions are exponentiated back with `expm1` before computing RMSE in dollars. This single move is worth ~15% off the RMSE.
- [ ] The **headline number** is the held-out test RMSE in dollars (rounded to the nearest $100). The CV RMSE is reported alongside ± SD.
- [ ] **Beats $32,000** test RMSE. (The strong baseline — linear regression on raw numeric features, no log transform — is around $45,000 on this dataset; ridge with a proper pipeline and log-target beats $32,000 routinely.)
- [ ] A **residual plot** with three panels (residuals vs predicted, histogram, Q-Q plot) is saved in `week-04/images/residuals.png`.
- [ ] A **standardized-coefficient table** of the top 15 features (by absolute value) is in the notebook. Signs are sense-checked in prose.
- [ ] A **learning curve** (`sklearn.model_selection.learning_curve`) is saved in `week-04/images/learning_curve.png` and read in the notebook for bias vs variance.
- [ ] A `report.md` (~2 pages, 600–900 words) summarizes the problem, the method, the result, and one honest limitation. No code.
- [ ] A `README.md` in `week-04/` explains: setup, data download, file layout, how to reproduce.

---

## Suggested layout

```text
crunch-ai-portfolio-<yourhandle>/
├── README.md                    ← portfolio root
└── week-04/
    ├── README.md                ← week-4 explainer
    ├── requirements.txt
    ├── data/
    │   └── ames_train.csv       ← Kaggle 'train.csv' (1,460 rows)
    ├── images/
    │   ├── residuals.png
    │   ├── learning_curve.png
    │   └── coefficients.png
    ├── ames.ipynb
    ├── ames.html                ← rendered preview
    └── report.md                ← 2-page executive summary
```

---

## Suggested order of operations

### Phase 1 — Setup and EDA (1 hour)

1. Open `ames.ipynb`. The first markdown cell is the **project header**: your handle, the dataset URL with access date, license notes (the Kaggle dataset is freely available; cite De Cock 2011).
2. Load the CSV. Print `df.shape`, `df.dtypes.value_counts()`, `df.isna().sum().sort_values(ascending=False).head(20)`.
3. Read the **data card** in De Cock's paper. The "missing" values that are not missing (`PoolQC`, `FireplaceQu`, `GarageType`, `BsmtQual`) are the most common stumble.
4. Look at the **target distribution**: `df["SalePrice"].plot.hist(bins=40)`. See the right skew. Plot `np.log1p(df["SalePrice"])`. See the bell.
5. Decide on the feature set. The C5 default for the first model: keep every feature, let `ColumnTransformer` route them to the right preprocessor, let `RidgeCV` shrink the unhelpful ones.

### Phase 2 — Pipeline and first fit (2 hours)

Build the pipeline:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

pipe = Pipeline([
    ("preprocess", preprocess),
    ("ridge",      RidgeCV(alphas=np.logspace(-3, 3, 13))),
])
```

Fit on `np.log1p(y_train)`. Cross-validate. Score on test (in dollars, after `np.expm1`).

```python
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)
y_tr_log = np.log1p(y_tr)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_log = -cross_val_score(
    pipe, X_tr, y_tr_log, cv=cv,
    scoring="neg_root_mean_squared_error",
)
print(f"CV log-RMSE: {cv_log.mean():.4f} ± {cv_log.std():.4f}")

pipe.fit(X_tr, y_tr_log)
y_te_pred = np.expm1(pipe.predict(X_te))
test_rmse = root_mean_squared_error(y_te, y_te_pred)
print(f"test RMSE (dollars): ${test_rmse:,.0f}")
print(f"chosen alpha: {pipe.named_steps['ridge'].alpha_}")
```

### Phase 3 — Diagnose (1.5 hours)

Three plots, in `week-04/images/`:

1. **`residuals.png`** — three panels (residuals vs predicted, histogram, Q-Q plot). Lecture 2, Section 6.
2. **`coefficients.png`** — horizontal bar of the top 15 standardized coefficients by absolute value. One focal color (brand violet) for the top 5, gray for the rest.
3. **`learning_curve.png`** — train RMSE and validation RMSE vs training-set size. Read it for bias vs variance.

For each plot, write one paragraph in the notebook *interpreting* it. The plots are diagnostics, not decorations.

### Phase 4 — Iterate (1.5 hours)

Pick **one** improvement and try it. Re-evaluate. The C5 short list, in order of expected gain:

1. **Drop outliers** in `GrLivArea` (De Cock's paper notes a few clear outliers with sale prices far below market — these are the famous Ames partial-sale records). Removing them is worth ~$1,000–2,000 of RMSE.
2. **Add interaction features** between `OverallQual` and `GrLivArea`. The interaction term captures "a large house in a high-quality build is worth more than the sum of the parts" — a real effect in housing data.
3. **Switch `RidgeCV` to `LassoCV`** and report which features got dropped to zero. Useful as an interpretability win even if the score is similar.
4. **Engineer "TotalSF"** = `GrLivArea + TotalBsmtSF + 1stFlrSF`. A single composite is sometimes worth more than the three components separately.

One iteration is the requirement. Two is the stretch.

### Phase 5 — Report and freeze (1.5 hours)

Final test-set scoring is **once**. If the test RMSE disagrees with the CV RMSE by more than 1–2 standard deviations, you have a leak. Find it. Otherwise, freeze.

Write `report.md`: 600–900 words. Sections: **Problem**, **Method**, **Result**, **Limitations**. No code. Imagine an engineering manager (or a hiring manager) who has five minutes and will not open the notebook.

### Phase 6 — Export (30 min)

```bash
jupyter nbconvert --to html week-04/ames.ipynb
```

Push to the portfolio repo.

---

## What "great" looks like

A "great" mini-project hits all of the following:

- The model **beats $30,000** test RMSE. Not because the leaderboard says so, but because at that level the diagnostic plots make sense and the coefficients tell a coherent story.
- The **log-target choice is justified in prose**, not just performed. ("The right-skewed target and the multiplicative structure of housing prices — a $10k swing on a $50k house is qualitatively different from a $10k swing on a $500k house — both argue for predicting log price.")
- The **residual plot is read honestly**. The plots will show some heteroscedasticity even after the log transform, especially at the high end. Naming it is the C5 brand voice; pretending it is not there is the bad voice.
- The **coefficient table is sense-checked**. "OverallQual is the largest positive coefficient (expected). YearBuilt has a positive coefficient that is small relative to GrLivArea (expected for Ames, where space matters more than age in the 1900s–2010s range)."
- The `report.md` reads like an analyst's memo, not a tutorial. Numbers cited; signs interpreted; one limitation named.

A "good but not great" project beats the baseline but does not log-transform; the report restates the score without interpreting the residual plot; the coefficient table is included but not read.

A "needs work" project skips the test split, reports the train RMSE as the headline, or fits the scaler outside the pipeline.

---

## Stretch goals

- **Fit `LassoCV` alongside `RidgeCV`** and write a paragraph on which one wins and why. Lasso usually trails ridge by a hair on this dataset because most of the 80 features carry *some* signal, but it produces a much shorter coefficient list — sometimes a useful tradeoff.
- **Add `ElasticNetCV`** with `l1_ratio` swept over `[0.1, 0.5, 0.9]`. Report the chosen `l1_ratio_` and `alpha_`. On Ames the elastic-net solution usually has 15–30 nonzero coefficients and a test RMSE within $500 of ridge — a feature-selection win, not a score win.
- **Plot `np.exp(predictions) - actual` against `Neighborhood`** in a strip plot. Identify neighborhoods where the model is systematically optimistic or pessimistic. The plot is one of the most interpretable model-fairness diagnostics in residential pricing.
- **Run `permutation_importance` on the test set.** Save a chart of the top 15 features by permutation importance, side-by-side with the standardized coefficients. They should agree in broad strokes; where they disagree is the interesting part.
- **Write a second notebook `ames_lasso.ipynb`** that uses only the features lasso did not drop. Beat your ridge RMSE on the reduced feature set. The win is interpretability, not score.

---

## Rubric

| Criterion | Weight | "Great" looks like |
|-----------|------:|--------------------|
| It runs | 15% | Fresh clone → `pip install -r requirements.txt` → `jupyter nbconvert --execute` → no errors |
| Pipeline discipline | 20% | `ColumnTransformer` end-to-end; no fit-then-split; all preprocessing inside `Pipeline` |
| Score | 20% | Test RMSE ≤ $30,000 (or a defended close-miss with reasons) |
| Diagnostics | 15% | Three residual panels, coefficient table, learning curve — each read in prose |
| Log-target reasoning | 10% | The decision to predict `log(SalePrice)` is justified, not just performed |
| Iteration | 10% | At least one engineered feature or one outlier-removal step, with before/after numbers |
| Report readability | 10% | The 2-page `report.md` could go to a hiring manager without explanation |

---

## What this exercises (and what comes next)

This mini-project exercises every concept from Week 4:

- The three-way split and 5-fold CV (Lecture 1).
- The baseline-then-model discipline (Lecture 1, Section 6).
- The five linear-regression assumptions and their residual-plot signatures (Lecture 2, Section 5).
- L2 regularization and `RidgeCV` (Lecture 2, Section 8).
- The `Pipeline` + `ColumnTransformer` no-leak pattern (Lecture 2, Section 9).
- Standardized coefficients and interpretation (Lecture 2, Section 11).
- The publication-quality charts you built in Week 3.

Week 5 will revisit the same dataset with gradient boosting (`HistGradientBoostingRegressor`, XGBoost, LightGBM). The goal there is to beat your linear-model score by ≥10% relative error. The point of *this* week's mini-project is to set a strong, defended baseline — so that next week's "the trees won" or "the trees did not win" claim is a meaningful one.

---

## Submission

Push your `crunch-ai-portfolio-<yourhandle>` repo. Open the `week-04/ames.html` link from the repo README so a reviewer with no Python install can read your work. Paste the headline numbers in the commit message:

```text
Week 4 mini-project: Ames Housing.

baseline (mean) test RMSE: $82,...
strong baseline test RMSE: $45,...
RidgeCV CV log-RMSE: 0.131 ± 0.012
RidgeCV test RMSE (dollars): $28,...
chosen alpha: 10.0
```

That is the artifact recruiters will see. It is also the number Week 5 has to beat.
