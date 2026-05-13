# Lecture 1 — The ML Workflow: Train, Validate, Test, Iterate

> **Outcome:** You can split a dataset into train / validation / test without leaking, pick the right cross-validation strategy for the data you have, draw a learning curve, read it for bias vs variance, and quote the no-free-lunch theorem when a colleague tells you XGBoost is "the best model." You leave this lecture with the C5 ML-workflow checklist memorized.

Week 1 left you with the `ndarray`. Week 2 left you with the `DataFrame`. Week 3 left you with a chart you can defend. This lecture leaves you with a **workflow** — the scaffolding into which every model in the rest of the course will be dropped.

We target **scikit-learn 1.5+** (current stable in 2026 is 1.6). The `train_test_split`, `KFold`, and `cross_val_score` APIs you learn here have been stable since 2017 and will still be the right answer in 2030.

---

## 1. The workflow, in one paragraph

Most professional ML projects, regardless of model family, follow the same nine steps:

1. **Frame** the problem as a prediction task. What is `y`? What is the unit of `X`? What does "right" mean?
2. **Build** the dataset. Collect rows, define columns, document the source.
3. **Split** train / validation / test, *before* you look at the data again. Random or stratified or grouped or time-aware — pick the split whose assumption matches your data.
4. **Baseline.** Predict the mean (regression) or the majority class (classification). Score it. Every model you ship is measured against this number.
5. **Fit** a simple model. Linear regression. Logistic regression. Nothing fancy.
6. **Evaluate** on the validation set. Compute the metric you committed to in step 1.
7. **Iterate.** Try a feature, a transform, a different model, a different regularization strength. Re-validate. Repeat until validation score stops improving.
8. **Freeze.** Pick the model you will ship. Re-fit it on train + validation. Score once on the test set.
9. **Write it up.** What worked, what didn't, what your model assumes, what would break it.

The model itself — `model.fit(X_train, y_train)` — is one line in step 5. The other eight steps are the work. A correctly executed step 3 alone explains more "Kaggle gold medals" than every architecture choice in step 5 combined.

---

## 2. Why three splits, not two

The naive answer is "train and test." That works for one model with no hyperparameters. The moment you have a knob to turn — the regularization strength, the polynomial degree, the learning rate — you need a **third** set to turn it on.

The three sets:

- **Train.** The rows the model sees during `fit`. Coefficients, weights, splits — everything is learned from train.
- **Validation.** The rows you use to compare candidate models or pick hyperparameters. The validation set is "leaked into" by every comparison you make. By the end of a long iteration, the validation score is an optimistic estimate of generalization.
- **Test.** The rows you check **at most twice** in the entire project. Ideally once: at the end, on the frozen model. Test exists so that the validation-set optimism does not become a deployment surprise.

The typical proportions in 2026:

| Dataset size | Train | Validation | Test |
|--------------|------:|----------:|----:|
| Tiny (< 1k rows) | 60% | use k-fold CV instead | 20–40% |
| Small (1k–100k) | 60–70% | 15% | 15–25% |
| Medium (100k–10M) | 80% | 10% | 10% |
| Large (> 10M) | 98% | 1% | 1% |

For tiny datasets, you use **cross-validation** (next section) in place of a fixed validation set; one fold at a time is the validation set, and you average across folds. For everything else, fixed validation and test sets are fine.

> **EXPERIMENT — feel the test-set discipline.** Load `sklearn.datasets.fetch_california_housing()`. Split 60/20/20 with `train_test_split` and `random_state=42`. Fit `LinearRegression`. Print the train, validation, and test RMSE. The three numbers are usually within 5% of each other on this dataset — that is what "no leakage, no overfit, clean baseline" looks like. Now refit a `PolynomialFeatures(degree=3) → LinearRegression` pipeline and watch the test number diverge from the train number.

---

## 3. `train_test_split` in scikit-learn 1.5+

The API is one line per split:

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)
# X_train: 60%, X_val: 20%, X_test: 20%.
```

The keywords that matter:

- `test_size` — the fraction of `X` that goes into the second return tuple. A float (0.20) is a fraction; an int (200) is a row count.
- `random_state` — the seed. **Set it.** A reproducible split is the difference between a colleague who can reproduce your numbers and one who cannot. The C5 convention is `random_state=42` for tutorials, `random_state=0` for production code that should never depend on the seed.
- `shuffle=True` (default) — randomize row order before splitting. The one time you set `shuffle=False` is **time series**, where you must respect ordering.
- `stratify=y` — for classification with imbalanced classes. The split preserves the class proportions. Without it, you can get a test set with zero positives, which produces a useless metric.

A worked example with stratification:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("train positive rate:", y_train.mean().round(3))
print("test  positive rate:", y_test.mean().round(3))
# Should print the same number twice (to 1%).
```

Without `stratify=y`, on a 5%-positive dataset, a 20% test set will sometimes contain 3% positives, sometimes 7%, and your "test set ROC-AUC" becomes a number with a standard deviation of its own. Stratify.

---

## 4. Cross-validation: when one split is not enough

For datasets under about 10,000 rows, a fixed validation set is too noisy. A single 20% slice has too few rows for the validation score to be a stable estimate. The fix is **k-fold cross-validation**: split the data into k folds, train on k-1 of them, validate on the remaining one, rotate. Average the k scores.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

cv  = KFold(n_splits=5, shuffle=True, random_state=42)
mdl = LinearRegression()
scores = cross_val_score(mdl, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
print(f"CV RMSE: {-scores.mean():.3f} ± {scores.std():.3f}")
```

A few notes that bite people:

- `scoring="neg_root_mean_squared_error"` is **negative** by sklearn convention: `cross_val_score` always returns "higher is better," so error metrics are negated. Flip the sign back when reporting.
- `shuffle=True` matters. Many real datasets are sorted (by date, by group, by class). Without shuffling, fold 1 is the first 20% of the file, which may be systematically different from fold 5.
- `random_state` — set it, for the same reproducibility reason as in `train_test_split`.

The four cross-validation strategies you should know, and when each applies:

| Strategy | When to use | What it preserves |
|----------|-------------|--------------------|
| `KFold` | i.i.d. data, no structure | Random splits |
| `StratifiedKFold` | Classification, imbalanced classes | Class proportions per fold |
| `GroupKFold` | Rows clustered by an entity (user, patient, store) | No entity appears in both train and test |
| `TimeSeriesSplit` | Time-ordered data | Train always precedes validation in time |

`GroupKFold` is the one beginners skip. If you have ten observations per patient and ten thousand patients, a random k-fold will put some observations from each patient into both train and validation — the model trivially memorizes the patient. Use `GroupKFold(groups=patient_id)` instead.

`TimeSeriesSplit` is the one beginners get wrong. A random k-fold on a stock-price prediction dataset means the model "trains" on Friday and "validates" on Tuesday — three days in the past. It will look great in cross-validation and be useless in production.

> **EXPERIMENT — when k-fold lies.** Generate a synthetic dataset of 1000 rows, with a column `customer_id` that takes 100 unique values. Each customer's `y` is `customer_id * 0.1 + noise`. Fit a model that uses `customer_id` as a feature, with `KFold(n_splits=5, shuffle=True)`. You will see a near-perfect score. Now switch to `GroupKFold(n_splits=5)` with `groups=customer_id`. The score drops to near-baseline. Both numbers are from the same data; one is a lie. This is the bug at the heart of every "great CV score, terrible production" story.

---

## 5. The single largest source of bad scores: leakage

A **leak** is any path by which information from the test set (or the future) reaches the model during training. Leakage produces "great" scores that do not reproduce. Five common leaks:

1. **Fit-then-split.** You fit `StandardScaler` on the whole dataset, then split. The scaler now knows the test set's mean and standard deviation. Fix: split first, fit the scaler on train only, transform both. Better fix: use a `Pipeline` (Lecture 2, Section 9).
2. **Target encoding without folds.** You compute the mean of `y` for each level of a categorical feature, then use it as a numeric feature. The mean for the test rows includes the test rows. Fix: target-encode within cross-validation folds.
3. **Using the future to predict the past.** A "rolling 30-day average" feature computed without respecting the time index includes future days. Fix: every time-aware feature is computed using only data prior to each row's timestamp.
4. **Duplicate rows across splits.** The same record appears in both train and test (often because of an `id` that should have been a `GroupKFold` key). Fix: dedupe before splitting; split by group.
5. **The validation set, used too many times.** Not a leak in the formal sense, but a slow-motion overfit. Every model comparison "leaks" a tiny bit of validation info into your choices. Fix: separate test set, checked at most twice.

The discipline that prevents all five: **split first, then look at the data**. Then every preprocessing step is fit on train, applied to validation and test.

---

## 6. Baselines: the only honest first model

Before you fit your first "real" model, fit a baseline. For regression: predict the mean of `y_train` for every row. For classification: predict the majority class. Scikit-learn ships `DummyRegressor` and `DummyClassifier` for exactly this:

```python
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import root_mean_squared_error, accuracy_score

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
rmse = root_mean_squared_error(y_val, baseline.predict(X_val))
print(f"baseline RMSE: {rmse:.3f}")
```

The point is not the baseline's score; it is the **delta** between baseline and your model. "My model gets 0.82 R²" is a fact about your model. "My model gets 0.82 R² where the mean predictor gets 0.00 and the median predictor gets −0.01" is information.

`root_mean_squared_error` is the sklearn 1.4+ function; on older versions use `mean_squared_error(..., squared=False)`. We pin sklearn ≥ 1.5 so `root_mean_squared_error` is always available.

---

## 7. The bias-variance decomposition, in plain English

A model's expected error on a new data point decomposes (approximately, for squared-error loss) into three pieces:

```text
expected error  =  bias²  +  variance  +  irreducible noise
```

- **Bias** — how far off the model's *average* prediction is from the truth, across many possible training sets. A model with high bias is too rigid to fit the truth. Linear regression on a clearly nonlinear relationship has high bias.
- **Variance** — how much the model's predictions change when you re-train it on a different sample of the same size. A model with high variance is too flexible to ignore the noise in any one training set. A 20-degree polynomial on 30 data points has high variance.
- **Noise** — error you cannot reduce, because `y` is not a deterministic function of `X` in the first place. "How many of these patients will recover" is not predictable from a CBC panel alone; there is structural noise.

The practical reading:

- High bias → the model is too simple. Add features, raise the polynomial degree, switch to a more flexible model family.
- High variance → the model is too flexible. Add regularization, reduce features, gather more data.

You diagnose which one you have with a **learning curve**: train and validation score as a function of training-set size.

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

sizes, train_scores, val_scores = learning_curve(
    estimator=mdl, X=X_train, y=y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="neg_root_mean_squared_error",
    random_state=42,
)

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(sizes, -train_scores.mean(axis=1), label="train", color="#7C3AED")
ax.plot(sizes, -val_scores.mean(axis=1),   label="validation", color="#5B21B6")
ax.set_xlabel("training set size")
ax.set_ylabel("RMSE")
ax.set_title("Learning curve — gap between train and val is variance")
ax.legend(frameon=False)
fig.savefig("learning_curve.png", dpi=150)
```

Reading the plot:

- **Both curves high and close together** → high bias. The model is too rigid; more data will not help; switch families.
- **Train low, validation high, large gap** → high variance. The model has memorized the training set; more data will help, and so will regularization.
- **Both curves dropping together** → keep going. You have not hit the wall yet.

---

## 8. The no-free-lunch theorem

Wolpert (1996) proved a formal claim that is widely misquoted and worth quoting correctly:

> *"For any two learning algorithms A and B, averaged over all possible problems with respect to a uniform prior over target functions, the expected generalization error of A equals the expected generalization error of B."*

In English: **no learner is universally best**. If A beats B on one problem, there is a different problem where B beats A by exactly as much.

The practical corollary is **not** "all models are equal" — that is the wrong reading. Real-world problems are not uniformly drawn from "all possible problems." They have structure: smoothness, sparsity, locality, hierarchy. The right reading is:

> **Match the inductive bias of the model to the structure of the problem.**

A linear model assumes a smooth, low-curvature relationship between features and target. When that assumption holds, linear models are hard to beat (housing prices, ad click-through, polling). When it does not (image pixels → object class, audio waveform → phoneme), a deep network's translation- and locality-bias wins. Trees assume axis-aligned, low-interaction splits, which is the right bias for tabular data with categorical features and sharp boundaries.

The slogan in 2026 is "tabular goes to gradient boosting; vision and language go to transformers." That slogan is the no-free-lunch theorem applied: pick the family whose inductive bias matches the data, not the family that won the most leaderboards in the abstract.

> **EXPERIMENT — no-free-lunch in twenty lines.** Generate a 1000-row dataset where `y = 3*x1 + 2*x2 + noise`. Fit linear regression, a 1-nearest-neighbor regressor, and a random forest. Linear wins. Now generate one where `y` is a checkerboard pattern of `x1` and `x2`. Refit all three. Linear is hopeless; the forest wins. Same code, different data, different best model.

---

## 9. The C5 ML-workflow checklist

Print this. Tape it to your monitor.

1. **Frame.** One sentence: "predict `y` from `X` on unit `Z`." If you cannot write that sentence, do not run code.
2. **Define the metric.** RMSE, R², F1, ROC-AUC. Decide *before* you fit. Changing the metric to fit the result is the cardinal sin.
3. **Split.** `train_test_split(..., random_state=42, stratify=y)` or `KFold(..., shuffle=True, random_state=42)` or `GroupKFold` or `TimeSeriesSplit`. Pick by structure.
4. **Baseline.** `DummyRegressor` or `DummyClassifier`. Record the number.
5. **Simplest model first.** `LinearRegression` or `LogisticRegression`. With a `Pipeline` for preprocessing.
6. **Evaluate.** On validation (or via cross-validation). Print the metric, the standard deviation, and the baseline delta.
7. **Diagnose.** Residual plot for regression. Confusion matrix for classification. Learning curve for either.
8. **Iterate.** One change at a time. Re-evaluate. Keep a log.
9. **Freeze.** Pick the model whose validation score is best *and* whose diagnostics look honest.
10. **Test.** Once. Report the number. If it disagrees with validation by more than the validation SD, you have a leak — go back to step 3.
11. **Write up.** What the model assumes, what would break it, how to retrain it. One page.

If you find yourself reaching for step 5 before step 3, you have already lost.

---

## 10. The vocabulary you need fluently

| Term | One-sentence definition |
|------|--------------------------|
| **`fit(X, y)`** | Learn parameters from the training data. |
| **`predict(X)`** | Apply the learned parameters to new data. |
| **`score(X, y)`** | Default metric for the estimator (R² for regressors, accuracy for classifiers). Convenient; not what you should report. |
| **`fit_transform(X)`** | Learn parameters *and* transform `X` in one call. Common on scalers and encoders. |
| **`Pipeline`** | A list of `(name, estimator)` steps that compose into one estimator. The final step's `fit` / `predict` is the pipeline's `fit` / `predict`. |
| **`ColumnTransformer`** | Apply different transformers to different columns in parallel, then concatenate. |
| **`cross_val_score`** | One-line cross-validation. Returns an array of `cv` scores. |
| **`GridSearchCV`** | Cross-validate every combination of a hyperparameter grid. The exhaustive option. |
| **`RandomizedSearchCV`** | Sample combinations from a distribution. The practical option when the grid is large. |
| **`refit=True`** | After grid search, re-fit the best estimator on the full training set. Default; almost always what you want. |
| **`n_jobs=-1`** | Use all CPU cores. Set it on `cross_val_score`, `GridSearchCV`, and the slow estimators. |

---

## 11. Where this leaves you

You can now split a dataset honestly, pick a cross-validation strategy whose assumption matches your data, compute a baseline, fit a simple model, draw a learning curve, and read it for bias versus variance. You can quote the no-free-lunch theorem and not abuse it.

Lecture 2 plugs **linear regression** into this workflow. We will fit it three ways (closed-form, gradient descent, sklearn), audit the five assumptions, draw residual plots, and add ridge and lasso regularization. Lecture 3 does the same for **logistic regression** on the classification side.

By Friday, the mini-project drops the Ames Housing dataset into this workflow. The model will be linear; the workflow will be the work; the score you ship will be the baseline that Week 5's trees have to beat.
