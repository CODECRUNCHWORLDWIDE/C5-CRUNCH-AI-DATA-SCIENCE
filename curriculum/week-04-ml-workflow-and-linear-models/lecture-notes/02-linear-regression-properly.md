# Lecture 2 — Linear Regression, Properly

> **Outcome:** You can write down the linear-regression model and its loss, derive the closed-form solution, fit the same model three ways (normal equation, gradient descent, scikit-learn) and verify all three agree to 1e-6, audit the fit against the five classical assumptions, read a residual plot, recognize multicollinearity from a condition number, and pick a regularization strength (ridge or lasso) by cross-validation. After this lecture, "linear regression" stops being a one-liner and becomes a small toolbox.

Linear regression is the model with the most assumptions, the most diagnostic tools, the most interpretation hooks, and the longest history of fooling people who ignored its assumptions. Every more-complex model in the rest of the course will be compared to a linear regression as the first baseline. This lecture is the one that decides whether that comparison is meaningful.

We target **scikit-learn 1.5+**, **numpy 2.x**, and **scipy ≥ 1.11**. The APIs (`LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `RidgeCV`, `LassoCV`) are stable and have been since 2017.

---

## 1. The model in one line

A linear regression predicts a scalar `y` from a row vector of features `x` as:

```text
y_hat = x · β + β_0
      = β_0 + β_1·x_1 + β_2·x_2 + … + β_p·x_p
```

`β` is a column vector of `p` coefficients; `β_0` is the intercept. By convention we **augment** `x` with a column of ones so the intercept becomes the first element of `β`:

```text
x = [1, x_1, x_2, …, x_p]
y_hat = x · β
```

The augmented form is what every textbook uses, what scikit-learn does internally (`fit_intercept=True`), and what we will use for the closed-form solution. The cost is one extra column of all-ones; the benefit is that every formula is one term shorter.

The matrix form over `n` rows:

```text
X is n × (p+1)       # n rows, p features plus the bias column
β is (p+1) × 1
y_hat = X β          # column vector of length n
```

---

## 2. The loss: mean squared error, written out

The standard loss is the **mean squared error (MSE)**:

```text
L(β) = (1/n) · sum_i (y_i − x_i · β)²
     = (1/n) · ||y − Xβ||²
```

`||·||` is the Euclidean (L2) norm: square each element, sum, take the square root. We are minimizing the *squared* norm because the math is easier and the minimizer is the same.

Three things to know about MSE before we fit anything:

1. **It is convex in `β`.** There is exactly one minimum. Gradient descent will not get stuck in a local minimum.
2. **It has a closed-form minimizer.** Unlike most ML losses, you can solve for `β` analytically. We do that next.
3. **It penalizes large errors quadratically.** A single outlier with residual 10 contributes the same as one hundred residuals of magnitude 1. If your `y` has outliers, consider MAE (mean absolute error) or Huber loss.

---

## 3. The normal equation (the closed-form solution)

Take the gradient of `L` with respect to `β` and set it to zero:

```text
∇_β L  =  (2/n) · Xᵀ (Xβ − y)  =  0
   ⟹   Xᵀ X β  =  Xᵀ y
   ⟹   β̂  =  (Xᵀ X)⁻¹ Xᵀ y          # the normal equation
```

That is the entire derivation. One line of calculus, one line of linear algebra. The minimizer is

```text
β̂ = (Xᵀ X)⁻¹ Xᵀ y
```

In NumPy:

```python
import numpy as np

X = np.c_[np.ones(len(X_raw)), X_raw]   # add the bias column
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
```

In production, **do not** compute the inverse explicitly. Use `np.linalg.lstsq` (or `solve(X.T @ X, X.T @ y)`); both are numerically more stable on near-singular matrices:

```python
beta_hat, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)
```

`np.linalg.lstsq` uses SVD under the hood, which gracefully handles rank deficiency (when `XᵀX` is singular because two features are perfectly collinear).

> **EXPERIMENT — three ways to the same answer.** Load `sklearn.datasets.load_diabetes()`. Compute `β̂` three ways: (1) `np.linalg.inv(X.T @ X) @ X.T @ y` (after adding the bias column), (2) `np.linalg.lstsq(X, y, rcond=None)`, (3) `sklearn.linear_model.LinearRegression(fit_intercept=True).fit(X_raw, y)` and read off `.intercept_` and `.coef_`. All three should match to 1e-10. The lstsq version is what sklearn calls internally.

---

## 4. Gradient descent: the same answer, the hard way

The normal equation works for any size dataset that fits in memory. For datasets where `XᵀX` is too large to invert (millions of features), or for the conceptual continuity with neural networks, we use **gradient descent**:

```text
β  ←  β − η · ∇_β L
   =  β − η · (2/n) · Xᵀ (Xβ − y)
```

`η` is the **learning rate**. The full loop:

```python
def gradient_descent(X, y, lr=0.01, n_iter=1000):
    n, p = X.shape
    beta = np.zeros(p)
    history = []
    for _ in range(n_iter):
        y_hat = X @ beta
        grad  = (2.0 / n) * X.T @ (y_hat - y)
        beta  = beta - lr * grad
        history.append(np.mean((y - y_hat) ** 2))
    return beta, history
```

Three things that go wrong if you do not scale the features first:

- The features have wildly different magnitudes (square footage in thousands, bathrooms in single digits). The gradient is dominated by the large-magnitude feature, and convergence is glacial.
- The fix is `StandardScaler` (Section 9). After scaling, every feature is mean 0, standard deviation 1, and the learning rate that works for one feature works for all of them.

A working learning rate for scaled data is around `lr=0.01` with a few thousand iterations on the diabetes dataset. If your loss is going up, the learning rate is too high. If your loss has not moved after 1000 iterations, the learning rate is too low. There is no formula; there is `np.logspace(-5, 0, 6)` and `for lr in lrs: fit_and_plot(lr)`.

---

## 5. The five classical assumptions

The linear regression *model* makes no assumptions — it is just a line. The standard *inference* and the standard *interpretation* make five:

1. **Linearity.** `E[y | X] = Xβ`. The conditional expectation of `y` given `X` is a linear function of `X`. Violated when the true relationship is curved, exponential, or step-function.
2. **Independence of errors.** The residuals are independent of each other. Violated by time-series data (today's residual correlates with yesterday's) and grouped data (residuals within a household correlate).
3. **Normality of errors.** The residuals are approximately Gaussian. Violated by heavy-tailed errors (income, returns), bimodal errors, or strong skew.
4. **Homoscedasticity.** The residuals have constant variance across the range of predicted values. Violated when the spread of errors grows with `y` (housing prices: ±$10k on a $50k house, ±$100k on a $5M house).
5. **No perfect multicollinearity.** No feature is an exact linear combination of the others. Violated when you accidentally include both a feature and its sum / mean with another.

The five assumptions matter to different degrees for different uses of the model:

| Use | Linearity | Independence | Normality | Homoscedasticity | Multicollinearity |
|-----|:---------:|:------------:|:---------:|:----------------:|:-----------------:|
| Predict `y` | important | mild | unimportant | mild | mild |
| Interpret `β` | important | important | important | important | **critical** |
| Compute p-values | important | important | important | important | **critical** |

If you only want predictions, assumptions 3 and 4 are loose suggestions. If you want to say "a 1 sqft increase is associated with a $X increase in price, holding all else equal," assumption 5 is non-negotiable.

> **EXPERIMENT — break each assumption on purpose.** Generate 500 rows of `y = 2*x + noise`. Fit a linear regression; the R² is ~0.95. Now: (a) replace the relationship with `y = 2*x² + noise` — R² drops, residuals show a parabola. (b) Make the noise scale with `x` — residuals show a funnel. (c) Add a feature `z = 2*x + 0.001*runif` — coefficient on `x` becomes unstable across re-fits. Each broken assumption has a signature in the residual plot.

---

## 6. Residual analysis: the diagnostic you should draw every time

After every linear-regression fit, draw three plots:

1. **Residuals vs predicted.** Should look like noise. Curvature → broken linearity. Funnel → broken homoscedasticity.
2. **Histogram (or Q-Q plot) of residuals.** Should look approximately Gaussian. Heavy tails → consider Huber regression or log-transform `y`.
3. **Residuals vs each feature.** One panel per feature. Curvature in any panel → that feature needs a transform or a polynomial term.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

y_hat = mdl.predict(X_val)
resid = y_val - y_hat

fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")

axes[0].scatter(y_hat, resid, s=8, alpha=0.5, color="#7C3AED")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("predicted y")
axes[0].set_ylabel("residual")
axes[0].set_title("Residuals vs predicted — want a noisy band")

axes[1].hist(resid, bins=30, color="#9CA3AF")
axes[1].set_xlabel("residual")
axes[1].set_ylabel("count")
axes[1].set_title("Residual histogram — want a bell")

# Q-Q plot
stats.probplot(resid, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q plot — want points on the line")

fig.savefig("residuals.png", dpi=150)
```

What the three plots tell you:

- A horizontal noisy band in plot 1 → linearity and homoscedasticity look fine.
- A U-shape or arc in plot 1 → linearity is broken; add a polynomial term or transform a feature.
- A funnel (wider on the right) in plot 1 → heteroscedasticity; log-transform `y` or use `HuberRegressor`.
- A bell in plot 2 → normality is fine.
- Heavy tails in plot 2 / S-curve in plot 3 → consider a robust regression.

The Shapiro-Wilk test on residuals is available (`scipy.stats.shapiro`), but on `n > 5000` rows it rejects normality for trivially-non-normal residuals. Use the *plot*, not the p-value.

---

## 7. Multicollinearity and the condition number

When two features are highly correlated, the matrix `XᵀX` becomes nearly singular. `(XᵀX)⁻¹` is then numerically unstable: small changes to `y` produce large changes to `β̂`. The classical diagnostic is the **variance inflation factor (VIF)** per feature:

```text
VIF_j = 1 / (1 − R²_j)
```

where `R²_j` is the R² from regressing feature `j` on all the other features. A VIF above ~5 is a warning; above ~10 is a problem.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def vif(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    vifs = np.empty(p)
    for j in range(p):
        mask = np.ones(p, dtype=bool); mask[j] = False
        r2 = LinearRegression().fit(X[:, mask], X[:, j]).score(X[:, mask], X[:, j])
        vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vifs
```

A faster, dataset-level diagnostic is the **condition number** of `X` (the ratio of its largest to smallest singular value):

```python
cond = np.linalg.cond(X)
```

A condition number above ~30 is a warning; above ~1000 is a problem. Cures, in order of preference:

1. **Drop one of the collinear features.** Cheapest. Almost always the right move.
2. **Add a small ridge penalty.** `Ridge(alpha=1.0)` is regularization; it adds a small constant to the diagonal of `XᵀX` before inverting, which stabilizes the inverse. (Section 8.)
3. **Use principal components** as features instead of the raw columns. Decorrelates by construction. Comes at the cost of interpretability.

---

## 8. Regularization: ridge (L2), lasso (L1), elastic net

Regularization adds a penalty on the size of the coefficients to the loss:

```text
Ridge   :  L(β) = (1/n) ||y − Xβ||²  +  α · ||β||₂²        # sum of squared coefficients
Lasso   :  L(β) = (1/n) ||y − Xβ||²  +  α · ||β||₁          # sum of absolute coefficients
Elastic :  L(β) = (1/n) ||y − Xβ||²  +  α · (ρ · ||β||₁ + (1−ρ) · ||β||₂²)
```

`α` (sklearn calls it `alpha`) controls the strength. `α=0` recovers ordinary least squares. As `α` grows, coefficients shrink toward zero.

Two behaviours that differ between L1 and L2:

- **Ridge (L2) shrinks all coefficients smoothly.** Coefficients rarely become *exactly* zero; they become small. Good for stabilizing collinear designs.
- **Lasso (L1) drives many coefficients to exactly zero.** This is *implicit feature selection*. Good when you suspect many features are irrelevant.

The geometric intuition: the L2 penalty is a circle, and the OLS "loss minimum" contour usually touches the circle on a tangent line (no coefficient is zero). The L1 penalty is a diamond, and the contour usually touches the diamond on a corner — a corner of the diamond is exactly an axis (some coefficient is zero).

In scikit-learn:

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
enet  = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)

# Pick alpha by cross-validation:
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5).fit(X_train, y_train)
lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 13), cv=5, max_iter=10_000).fit(X_train, y_train)
print("ridge best alpha:", ridge_cv.alpha_)
print("lasso best alpha:", lasso_cv.alpha_)
```

Two things every beginner gets wrong:

1. **You must scale features before regularizing.** A feature in dollars and a feature in millions of dollars get penalized at hugely different scales unless you `StandardScaler` first. Put the scaler in a `Pipeline` so it never gets forgotten.
2. **The intercept is not regularized.** sklearn handles this for you. If you implement it by hand, do not penalize `β_0`.

> **EXPERIMENT — see lasso select features.** Load the diabetes dataset (10 features). Fit `LinearRegression` and print the coefficients — all ten are non-zero. Fit `Lasso(alpha=0.5)` — three or four coefficients become exactly zero. Increase `alpha` to 1.0 — more zeros. At `alpha=10.0`, every coefficient is zero (the model is the mean predictor). The path from "all non-zero" to "all zero" is the lasso solution path; sklearn's `lasso_path` plots it for you.

---

## 9. Pipelines: the only honest way to preprocess

We have used `Pipeline` casually in the lecture. Here is the case for it explicitly.

Without a pipeline, the preprocessing-then-cross-validate dance looks like this:

```python
# WRONG — leaks the validation set into the scaler.
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
scores = cross_val_score(LinearRegression(), X_scaled, y, cv=5)
```

The scaler was fit on **all** of `X`, including the rows that will end up in each fold's validation set. Information leaks across the fold boundary. The scores will be optimistic.

With a pipeline:

```python
# RIGHT — the scaler is re-fit on the training portion of each fold.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=1.0)),
])
scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")
```

`cross_val_score` re-fits the entire pipeline on each fold's training portion. The scaler sees only train, transforms validation, the model fits on the transformed train, predicts on the transformed validation. Leak-free by construction.

For mixed-type data (some numeric columns, some categorical), use `ColumnTransformer`:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_cols = ["sqft", "bedrooms", "age"]
categorical_cols = ["neighborhood", "style"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0)),
])
pipe.fit(X_train, y_train)
```

Exercise 3 builds exactly this pattern on the Ames Housing data.

`OneHotEncoder(handle_unknown="ignore")` is the production-safe default in scikit-learn 1.5+. It silently encodes unseen categorical levels as the all-zeros vector at test time, which is the right behaviour when a new neighborhood shows up in production.

---

## 10. Choosing the metric: RMSE, MAE, R², MAPE

Four common regression metrics, with their tradeoffs:

| Metric | Formula | Units | Sensitive to outliers | When to use |
|--------|---------|-------|:--------------------:|-------------|
| **RMSE** | `√(mean((y − ŷ)²))` | same as `y` | very | default; most contexts; differentiable |
| **MAE** | `mean(\|y − ŷ\|)` | same as `y` | mildly | when outliers should not dominate |
| **R²** | `1 − SS_res/SS_tot` | unitless | depends | model comparison on the same data |
| **MAPE** | `mean(\|y − ŷ\| / \|y\|)` | percent | depends | when relative error matters and `y ≠ 0` |

R² is the default in `sklearn.score()` for regressors, but it is **not** the metric you should report. R² depends on the variance of your particular test set; the same model on a different test set can have wildly different R². Report RMSE or MAE in the units of `y` ("median absolute error of $12,400"); it is more interpretable, more comparable, and more honest.

A common mistake: optimizing for one metric in cross-validation and reporting another at test time. Pick one metric, optimize it, report it. If you must report several, fine — but the one in your loss function and the one in your headline number should be the same.

---

## 11. Interpretation: what the coefficients mean

If the linearity assumption holds and the features are not too collinear, the coefficient `β_j` is

> *"The expected change in `y` for a one-unit increase in feature `j`, holding all other features fixed."*

That sentence is the entire point of linear regression as an interpretation tool. Three caveats:

1. **"Holding all other features fixed"** is hypothetical. If `bedrooms` and `sqft` are correlated, you cannot, in practice, increase `bedrooms` while holding `sqft` fixed. The coefficient still has a mathematical meaning; the causal claim is only as strong as the design.
2. **Standardized coefficients** (i.e. coefficients after `StandardScaler`) are comparable in magnitude *to each other*. Raw coefficients are not — a feature in dollars and a feature in counts have incomparable coefficient scales.
3. **The sign matters.** A negative coefficient on `bedrooms` after controlling for `sqft` and `bathrooms` is suspicious — it usually means the model has decided "for fixed sqft, more bedrooms means smaller rooms, which is worse." That is a real effect in housing data; learn to read the sign in context.

`statsmodels.api.OLS` gives you standard errors, t-statistics, p-values, and confidence intervals. scikit-learn does not — its `LinearRegression` is the prediction-side library. If you want inference, use `statsmodels` (also free, also pip-installable: `pip install statsmodels`).

> **EXPERIMENT — collinearity scrambles signs.** Generate `x1 = N(0, 1)`, `x2 = x1 + 0.01·N(0, 1)` (nearly identical), `y = x1 + 0·x2 + N(0, 0.1)`. Fit `LinearRegression`. The coefficient on `x1` is huge and positive; the coefficient on `x2` is huge and negative. Together they predict `y` fine; individually they are nonsense. Now add `Ridge(alpha=1.0)`. Both coefficients are sensible and roughly equal. This is the regularization-stabilizes-collinearity story in code.

---

## 12. The checklist for a linear regression that ships

Before you ship a linear regression, walk this list:

- [ ] **Features scaled** (especially if regularized). `StandardScaler` in a `Pipeline`.
- [ ] **No perfect collinearity**. Condition number under 1000; VIF under 10 for each feature.
- [ ] **Residuals plotted**. No curvature, no funnel.
- [ ] **Residual histogram**. Approximately bell-shaped, or you have a justification for what's not.
- [ ] **R² *and* RMSE reported**. R² for the comparison-on-this-test-set claim; RMSE in `y`-units for the interpretation.
- [ ] **Baseline delta reported**. "RMSE 12,400 vs mean-predictor 38,200" tells the reader your model is doing real work.
- [ ] **Regularization strength picked by CV**. `RidgeCV` / `LassoCV`, not by feel.
- [ ] **Coefficients sense-checked**. Sign and magnitude match your prior. If not, write down why.
- [ ] **Test set scored once**. If the test number is far worse than CV, you have a leak.

If a reviewer asks "what does this model assume," you should be able to answer with section 5 from memory. That is what "linear regression done properly" buys you over `LinearRegression().fit(X, y)`.

---

## 13. Where this leaves you

You can now fit a linear regression three ways, audit it against five assumptions, draw three residual plots, recognize multicollinearity, and pick a regularization strength by cross-validation. You have the diagnostic vocabulary to defend "I shipped a linear model" against a colleague who wanted to start with XGBoost.

Lecture 3 carries the same discipline to **classification**: logistic regression as "linear regression on the log-odds," then the metrics you actually report (precision, recall, F1, ROC-AUC, PR-AUC) and the threshold dial that the metric depends on.

The mini-project at the end of the week drops the Ames Housing dataset into the workflow from Lecture 1 with the modeling from Lecture 2. By Sunday, you ship a defended linear-regression baseline on a real housing dataset — and Week 5 has its starting score to beat.
