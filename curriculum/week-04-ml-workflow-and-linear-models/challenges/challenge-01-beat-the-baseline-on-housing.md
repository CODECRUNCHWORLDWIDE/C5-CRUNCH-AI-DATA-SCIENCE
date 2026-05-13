# Challenge 1 — Beat the baseline on housing

**Time estimate:** 2 hours.

## Problem statement

Fit a linear regression (ridge or lasso, your choice) on a real housing dataset and **beat a strong published baseline** in cross-validated RMSE. The two acceptable datasets are:

1. **California Housing** (ships with scikit-learn; one call): `sklearn.datasets.fetch_california_housing(as_frame=True)`. The target is median house value in units of $100,000.
2. **Ames Housing** (a CSV download): the same dataset Dean De Cock published in 2011. The target is `SalePrice` in dollars.

You will write a single script `beat_the_baseline.py` and a short markdown file `notes.md` that:

1. Loads the dataset.
2. Splits 80/20 train/test with `random_state=42`.
3. Reports the **mean-predictor baseline** RMSE on the test set.
4. Reports a **strong baseline** RMSE: linear regression with no preprocessing on the raw numeric columns only. (This is the "published baseline" you have to beat.)
5. Builds a `Pipeline` with `ColumnTransformer` (Lecture 2, Section 9) that scales numeric features, one-hot-encodes categorical features (Ames only — California has none), and feeds the result into `RidgeCV(alphas=np.logspace(-3, 3, 13))`.
6. Reports the cross-validated RMSE (5-fold) and the held-out test RMSE.
7. Saves a residual plot to `residuals.png` (Lecture 2, Section 6).

The challenge is to ship a pipeline whose test RMSE is meaningfully better than the strong baseline, with diagnostics that defend the choice.

## The data

### Option A — California Housing (recommended for a 2-hour budget)

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target          # y is in units of $100k
```

The dataset is 20,640 rows, 8 numeric features, no missing values. It is the cleanest realistic regression dataset in scikit-learn.

### Option B — Ames Housing (recommended if you want the mini-project warm-up)

Download from <https://jse.amstat.org/v19n3/decock.pdf> (paper + data card) or use the Kaggle copy at <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>. The dataset is 1,460 rows in `train.csv`, 81 features, mixed numeric and categorical, with substantial missing data.

Ames is the harder challenge; the mini-project uses the same dataset, so doing the challenge on Ames means you arrive at the mini-project with the data already understood.

## What "beat the baseline" means

For **California Housing**, the strong-baseline RMSE (linear regression on raw features, no scaling, no transformation) is about **0.73** (in units of $100k). Your `RidgeCV` pipeline should reach **0.70** or better.

For **Ames Housing**, the strong-baseline RMSE (linear regression on raw numeric features only, no categorical features, no transformation) is about **$45,000**. With a proper pipeline (numeric scaling, one-hot encoding of categoricals, RidgeCV) you should reach **$32,000** or better. Top Kaggle leaderboard submissions land around **$15,000** — those use gradient boosting (Week 5) and feature engineering well past two hours of work.

The numbers are floors, not ceilings. The point is to ship a model whose test RMSE *is* better than the strong baseline and to know *why*.

## Acceptance criteria

- [ ] A single script `beat_the_baseline.py` runs end-to-end with `python beat_the_baseline.py` and prints all four RMSE numbers (baseline, strong baseline, CV, test).
- [ ] `python -m py_compile beat_the_baseline.py` succeeds.
- [ ] The pipeline uses `Pipeline` + `ColumnTransformer` end-to-end. No fit-then-split. No `StandardScaler.fit_transform(X)` outside the pipeline.
- [ ] The RidgeCV `alpha_` (the chosen regularization strength) is printed.
- [ ] A `residuals.png` file is saved next to the script with three panels:
  - Residuals vs predicted (want a noisy horizontal band).
  - Histogram of residuals (want a bell).
  - Residuals vs the most-important feature.
- [ ] The test RMSE beats the strong baseline RMSE by at least the floor (10% relative for California, 20% relative for Ames).
- [ ] A `notes.md` file (~200 words) covering the prompt below.

## The notes prompt

Open `notes.md` and answer in roughly 200 words.

1. **What is the model?** One sentence. "Ridge regression on N features, with alpha chosen by 5-fold CV."
2. **What did the strong baseline get?** Number, units. Why is this the "strong" baseline rather than the mean predictor? (Hint: the mean predictor on Ames gets ~$80k.)
3. **What does your pipeline beat it by?** Absolute and relative. State both numbers.
4. **What feature transform mattered most?** Be specific: scaling, one-hot encoding, log-transforming `SalePrice`, dropping outliers. The most surprising win on this dataset, historically, is `log(SalePrice)` for Ames.
5. **What does the residual plot say?** One observation that suggests "the linear-model assumptions are/are not violated."
6. **What would a reviewer ask next?** One sentence on the obvious follow-up — usually "what happens if you log the target" (Ames) or "is the model worse in San Francisco than in the Central Valley" (California).

A "great" `notes.md` is honest about the residual plot and specific about the comparison. A "bad" one is "I beat the baseline, here is my code."

## Suggested layout

```python
"""Beat the housing baseline."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PLOT_VIOLET = "#7C3AED"
GRAY = "#9CA3AF"


def main() -> None:
    # 1. Load.
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    # 2. Split.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    # 3. Baselines.
    mean_rmse = root_mean_squared_error(
        y_te, DummyRegressor(strategy="mean").fit(X_tr, y_tr).predict(X_te)
    )
    strong_rmse = root_mean_squared_error(
        y_te, LinearRegression().fit(X_tr, y_tr).predict(X_te)
    )

    # 4. Build the pipeline.
    pipe = Pipeline([
        ("preprocess", ColumnTransformer([
            ("num", StandardScaler(), X.columns.tolist()),
            # add ("cat", OneHotEncoder(handle_unknown="ignore"), [...]) for Ames.
        ])),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])

    # 5. Cross-validate on train.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = -cross_val_score(
        pipe, X_tr, y_tr, cv=cv,
        scoring="neg_root_mean_squared_error",
    )

    # 6. Fit on train, score on test.
    pipe.fit(X_tr, y_tr)
    test_rmse = root_mean_squared_error(y_te, pipe.predict(X_te))

    # 7. Report.
    print(f"mean-predictor RMSE: {mean_rmse:.4f}")
    print(f"strong-baseline RMSE: {strong_rmse:.4f}")
    print(f"CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"test RMSE: {test_rmse:.4f}")
    print(f"chosen alpha: {pipe.named_steps['ridge'].alpha_}")

    # 8. Residual plot.
    resid = y_te - pipe.predict(X_te)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].scatter(pipe.predict(X_te), resid, s=6, alpha=0.4, color=PLOT_VIOLET)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Residuals vs predicted")
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("residual")
    axes[1].hist(resid, bins=40, color=GRAY)
    axes[1].set_title("Residual histogram")
    axes[2].scatter(X_te.iloc[:, 0], resid, s=6, alpha=0.4, color=PLOT_VIOLET)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title(f"Residuals vs {X_te.columns[0]}")
    fig.savefig("residuals.png", dpi=150)


if __name__ == "__main__":
    main()
```

## Hints

<details>
<summary>If your CV RMSE is much better than your test RMSE</summary>

You have a leak. Walk Section 5 of Lecture 1 — the five common leaks. The most likely on this dataset: you fit `StandardScaler` outside the `Pipeline`, or you computed a target-encoding feature outside the cross-validation folds.

</details>

<details>
<summary>If your test RMSE is essentially identical to the strong baseline</summary>

Two likely fixes:

1. You forgot to scale features before ridge. `RidgeCV` on un-scaled features penalizes coefficients on different scales differently, which is roughly equivalent to no regularization.
2. You did not include the categorical features (Ames only). Encoding `neighborhood` and `quality` features alone is usually worth $5k of RMSE on Ames.

</details>

<details>
<summary>If your residual plot has a clear funnel shape</summary>

That is heteroscedasticity — residual variance grows with predicted value. The classical fix on housing data is to predict `log(price)` rather than `price`. The model becomes a multiplicative one (a 10% change in features predicts a multiplicative change in price), which fits housing prices well. Re-fit, then exponentiate predictions when computing RMSE on the original scale.

</details>

## Stretch goals

- **Compare Ridge to Lasso.** Fit `LassoCV` with the same alpha grid. Report which features lasso drives to exactly zero. Compare test RMSE and `n_features_with_nonzero_coef` between the two.
- **Predict `log(SalePrice)` on Ames.** Fit the same pipeline on `np.log1p(y)` and exponentiate predictions before computing RMSE. The improvement is usually 10–20%.
- **Plot the learning curve** (Lecture 1, Section 7). Diagnose whether the model is bias-limited or variance-limited. The answer guides whether feature engineering or more data is the next move.
- **Run `permutation_importance`** (`sklearn.inspection.permutation_importance`) on the test set. Identify the three most important features. Cross-check against the standardized coefficients — they should agree.

## Why this matters

This challenge is a miniature of every regression project. Load. Split. Baseline. Pipeline. Regularize. Diagnose. Report. The mini-project is the same five hours of work with the portfolio polish on it. Doing the challenge first means the mini-project feels like an extension of one habit, not a new task.

## Submission

Commit `beat_the_baseline.py`, `residuals.png`, and `notes.md` to your Week 4 repo. Paste the four RMSE numbers in the commit message:

```text
Challenge 1 -- beat the baseline.

mean-predictor RMSE: ...
strong-baseline RMSE: ...
CV RMSE: ... ± ...
test RMSE: ...
chosen alpha: ...
```

Anyone reading the commit message should know whether you beat the baseline in two seconds.
