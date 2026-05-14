# Mini-Project — A NumPy MLP Classifier on a Non-MNIST Tabular Dataset

> Build a two-layer multi-layer perceptron in pure NumPy and train it to classify a **real tabular dataset** — not MNIST. Use the same forward / backward / SGD machinery you wrote during the exercises; no PyTorch, no TensorFlow. The deliverable is a notebook plus a two-page report that an engineering manager could read in five minutes and conclude "this person understands what a framework is doing on their behalf."

This is the seventh artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 4 was the linear baseline; Week 5 was the boosted-trees lift; Week 6 was the unsupervised clustering counterpart; this week is the first **deep-learning** artifact. Recruiters with taste read Week 7 as the moment a junior data scientist either learns what backpropagation actually is or quietly imports PyTorch. The artifact you commit this week is the answer.

**Estimated time:** 8 hours, spread across Thursday-Sunday.

---

## What you will build

A Jupyter notebook `mlp_tabular.ipynb` plus a rendered `report.md` that:

1. **Loads** a non-MNIST tabular dataset (one of the three suggestions below, or your own).
2. **Splits** into training and test sets (`test_size = 0.2`, `stratify=y`, `random_state = 42`).
3. **Standardizes** the numeric features with `StandardScaler`. Mandatory; the MLP is sensitive to feature scale, as is every gradient-based learner.
4. **Trains a baseline**: a `sklearn.linear_model.LogisticRegression(max_iter=1000)`. Records the test accuracy. The MLP must beat the baseline; otherwise the deep learning is not earning its keep.
5. **Trains the MLP** in your from-scratch NumPy code (the same `forward`, `backward`, `sgd_step`, `train` functions from the exercises, with the input dimension changed to match the dataset). Record the test accuracy.
6. **Verifies the gradient** on a small batch via the gradient check from Lecture 2 Section 10. Print the relative errors per parameter; they must all be `< 1e-5`.
7. **Plots the training curves**: training loss and test accuracy per epoch, two panels.
8. **Computes a confusion matrix** for the MLP on the test set. Identify the worst-confused class pair.
9. **Reports** in a 600-900 word executive summary that a non-technical reader can finish in five minutes.

The notebook is the working artifact. The `report.md` is the executive summary.

---

## The dataset

Three default options, ordered by difficulty:

### Option A — Wisconsin Breast Cancer (~570 rows; binary; gentle)

Available as `sklearn.datasets.load_breast_cancer()`. 569 rows, 30 numeric features (cell-nucleus measurements), binary target (malignant vs. benign). The classic baseline is logistic regression at ~96%; a two-layer MLP with `h = 16` typically reaches 97-98%. A clean Week 7 result.

### Option B — UCI Adult / Census Income (~48,000 rows; binary; medium)

Available as `sklearn.datasets.fetch_openml("adult", version=2)`. Predicts whether a person earns `> $50K/year` from 14 demographic and employment features. Mixed numeric and categorical; you will one-hot encode the categoricals (~100 features after encoding). Binary target.

The Adult dataset has a well-known class imbalance (~75% earn < $50K). The C5 minimum is **accuracy ≥ 0.85** and **F1 score on the minority class ≥ 0.65**. Logistic regression gets ~0.85 / 0.65; a two-layer MLP should match or slightly exceed.

Download: <https://www.openml.org/d/1590> (free, no signup).

### Option C — UCI Iris (~150 rows; 3-class; *very* gentle, for time-pressed students)

Available as `sklearn.datasets.load_iris()`. 150 rows, 4 features, 3 classes. The smallest "real" dataset that an MLP can train on. The MLP should reach 95% test accuracy; the LR baseline is 97%. The MLP does not gain anything over LR on Iris (and that is the lesson — Iris is linearly separable in the standardized feature space; a nonlinear MLP is overkill).

If you pick this, the C5 rubric will *penalize a too-trivial choice*. The honest report says "MLP does not beat LR on Iris; this is consistent with Iris being linearly separable; for a fair test we would need more interesting data." That paragraph earns more credit than an over-fit 99% accuracy claim.

### Option D — your own dataset (must be ≥500 rows and ≥4 numeric features)

If you have access to a real dataset from work, a public Kaggle dataset, or a domain you care about, use it. The acceptance criteria below are the same. Provide a data card paragraph in `report.md` explaining the source and the licensing.

The mini-project rubric does *not* penalize you for picking option A over option B; the rubric *does* penalize you for picking option C without acknowledging its limitations.

---

## Acceptance criteria

- [ ] A new directory `week-07/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `numpy>=2,<3`, `scikit-learn>=1.5,<2`, `matplotlib>=3.8,<4`, `jupyter`.
- [ ] `jupyter nbconvert --to notebook --execute week-07/mlp_tabular.ipynb` runs end-to-end without errors.
- [ ] The notebook **standardizes** all numeric features with `StandardScaler` before any training. No training on unscaled features.
- [ ] The notebook trains the **logistic-regression baseline** and records its test accuracy.
- [ ] The notebook trains the **NumPy MLP** with the same forward / backward / SGD code from the exercises. The model definition is in `week-07/mlp.py`; the notebook imports it.
- [ ] The MLP **beats the logistic-regression baseline** in test accuracy (or matches it; ties are acceptable if defended in the report).
- [ ] The notebook **runs the gradient check** on a 5-example batch and reports the maximum relative error per parameter. All errors `< 1e-5`.
- [ ] The notebook **plots the training curves** (training loss and test accuracy per epoch) and saves them as `week-07/images/training_curves.png`.
- [ ] The notebook **computes the confusion matrix** for the MLP on the test set and saves it as `week-07/images/confusion_matrix.png`.
- [ ] A `report.md` (~2 pages, 600-900 words) summarizes the problem, the data, the baseline, the MLP, the training curves, the confusion matrix, and **the honest paragraph** about what the model does and does not generalize. No code.
- [ ] A `README.md` in `week-07/` explains: setup, data download (if applicable), file layout, how to reproduce.

---

## Suggested layout

```text
crunch-ai-portfolio-<yourhandle>/
├── README.md
├── week-04/                     ← from Week 4
├── week-05/                     ← from Week 5
├── week-06/                     ← from Week 6
└── week-07/
    ├── README.md                ← week-7 explainer with key result
    ├── requirements.txt
    ├── mlp.py                   ← the NumPy MLP module (your forward/backward/train)
    ├── data/                    ← (only if Option D with bundled data)
    ├── images/
    │   ├── training_curves.png  ← loss + test accuracy per epoch
    │   └── confusion_matrix.png ← n_classes x n_classes heatmap
    ├── mlp_tabular.ipynb        ← the working notebook
    ├── mlp_tabular.html         ← rendered preview
    └── report.md                ← 2-page executive summary
```

The `starter.py` in this folder is a runnable scaffold for `mlp.py`; copy it into `week-07/mlp.py` in your portfolio repo and fill in the data loader for your chosen dataset.

---

## Suggested order of operations

### Phase 1 — Load and split (30 min)

1. Open `mlp_tabular.ipynb`. The first cell is the **project header** plus a one-line citation of the dataset.
2. Load the data (`sklearn.datasets.load_breast_cancer()` or `fetch_openml("adult", ...)`).
3. For Adult or any dataset with categoricals: one-hot encode them with `pd.get_dummies` or `sklearn.preprocessing.OneHotEncoder`.
4. Split with `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
5. Standardize: `scaler = StandardScaler().fit(X_train); X_train = scaler.transform(X_train); X_test = scaler.transform(X_test)`.

### Phase 2 — The logistic-regression baseline (30 min)

1. Fit `LogisticRegression(max_iter=1000, random_state=42)`.
2. Record test accuracy. For binary problems, also record precision, recall, and F1 score on the positive class.
3. The baseline number is the bar your MLP must clear. Write the baseline number in the report.

### Phase 3 — The MLP (3 hours)

1. Open `mlp.py` (copied from `starter.py`). The module exports `init_params`, `forward`, `backward`, `predict`, `cross_entropy_loss`, `softmax`, `relu`, `relu_grad`, `train`. Fill in any TODOs.
2. In the notebook, import the module: `from mlp import init_params, train, predict, gradient_check`.
3. **Run the gradient check** on a tiny batch (5 examples, 4 features, 3 classes; or whatever fits your dataset). Print the max relative error per parameter. All must be `< 1e-5`. If any fail, re-derive the backward pass before continuing.
4. Train: `params, history = train(X_train, y_train, X_test, y_test, n_hidden=64, learning_rate=0.05, batch_size=64, n_epochs=30)`. Tune `n_hidden`, `learning_rate`, and `batch_size` as needed for your dataset; the C5 defaults are starting points, not final values.
5. Plot the training curves. Save as `images/training_curves.png`.
6. Record the final test accuracy.

### Phase 4 — Confusion matrix and error analysis (1.5 hours)

1. Compute the confusion matrix: `cm = sklearn.metrics.confusion_matrix(y_test, predict(X_test, params))`.
2. Plot as a heatmap with class indices on both axes and the value annotated. Save as `images/confusion_matrix.png`.
3. Identify the worst-confused class pair.
4. Look at five misclassified examples. Describe one in the report: "the model predicted class X with confidence 0.6, but the true class was Y; the feature pattern was ..." If your dataset has interpretable features, this paragraph is essential.

### Phase 5 — The report (1.5 hours)

`report.md` (600-900 words). Structure:

1. **Problem** — one paragraph. What is the dataset? What is the prediction task? Why is this a reasonable problem to put an MLP on?
2. **Data** — one paragraph. How many rows? How many features? Class distribution? Did you do any preprocessing beyond standardization?
3. **Method** — one paragraph. "We trained a two-layer ReLU MLP with He initialization, mini-batch SGD, and cross-entropy loss. Hidden width `h = X`; learning rate `η = X`; batch size `m = X`; trained for `N` epochs." One paragraph; the details are in the notebook.
4. **Result** — two paragraphs. Test accuracy of the MLP; test accuracy of the LR baseline; the gap. The confusion matrix and the worst confusion pair.
5. **Honest paragraph** — one paragraph. Did the MLP genuinely outperform LR on this task? By how much? Is the improvement statistically meaningful or in the noise? If the MLP only matches LR (which is common for small tabular datasets), say so honestly.
6. **Limitations** — one paragraph. What is the model's failure mode? On Adult: the F1 on the minority class is the weakest metric. On Breast Cancer: the dataset is small and the test set has only ~115 examples; the test accuracy is noisy. On any dataset: the model is not interpretable; if interpretability is required, you would prefer the LR.
7. **Recommendation** — one short paragraph. Would you ship this model? Or would you prefer the LR for the simpler explanation? On most small tabular datasets, the honest recommendation is "ship the LR; the MLP gains are too small to justify the complexity."

### Phase 6 — Export (30 min)

```bash
jupyter nbconvert --to html week-07/mlp_tabular.ipynb
```

Push to the portfolio repo.

---

## What "great" looks like

A "great" mini-project hits all of the following:

- The **MLP beats the LR baseline** (even if by 0.5 percentage points). The defense in the report is *specific*: "the LR reached 96.5% test accuracy; the MLP reached 97.8% (a 1.3 percentage-point gain, or a 37% reduction in test error). The gap survives reseeding (3 seeds, mean 97.6% ± 0.2%)."
- The **gradient check passes** with all errors below `1e-5`, and the report mentions running it. A submission that does not run the gradient check is a submission that may be wrong. Mentioning it in the report is the C5 signal of competence.
- The **training curves are clean.** Monotonically decreasing loss; smooth rising test accuracy plateauing in the last few epochs. If the curve is noisy, the report explains why (small batch size, high learning rate, small dataset).
- The **confusion matrix is annotated** and the worst confusion is named with one English sentence: "the model confuses class 1 (income > $50K) for class 0 (income ≤ $50K) more often than the reverse, suggesting it is conservative about predicting the high-income class — consistent with the class-imbalance pattern in the training data."
- The **honest paragraph is specific.** Generic: "the MLP did well." Specific: "on this dataset the MLP outperforms LR by 1.3 percentage points; the gain is real but small; on a different dataset with more nonlinear structure (e.g. an image task or a sequence task) the gap would be much larger; on tabular data with few rows and well-engineered features, LR is often the right choice."
- The **`report.md` could go to a hiring manager** without explanation. No code; no jargon beyond two or three terms (each defined inline); a clear "what the model does" / "how well it does it" / "where it fails" / "would I ship it" arc.

A "good but not great" project trains the MLP, gets a reasonable accuracy, plots the loss curve, and writes a generic paragraph. It is fine; it is not what we are aiming for.

A "needs work" project has any of the following: skipped the gradient check; did not standardize the features; did not run the LR baseline; failed to beat the baseline and did not explain why; used `np.exp` without the max trick and got `nan` losses; pushed a notebook that does not execute.

---

## Stretch goals

- **Add weight decay** (L2 regularization) to the SGD update: `params[k] -= learning_rate * (grads[k] + weight_decay * params[k])`. Train with `weight_decay ∈ [0, 1e-4, 1e-3, 1e-2]` and report the best test accuracy.
- **Implement early stopping.** Track the test accuracy per epoch and save the best parameters; use those for the final reported number instead of the last-epoch parameters.
- **Add momentum to SGD.** `v ← 0.9 v + g; params -= lr * v`. The training curve should be smoother and reach the same accuracy in fewer epochs.
- **Try a three-layer MLP.** Add a second hidden layer. On most small tabular datasets, the third layer does not help and sometimes hurts (more capacity to overfit, no compensating gain in expressive power). The honest paragraph for this stretch goal is: "deeper does not always mean better."
- **Train an MLP and compare to GBM (`HistGradientBoostingClassifier` from Week 5).** On most tabular datasets, gradient boosting beats MLPs. This is the lesson of Friedman et al. (2001) repeated by every empirical study since: tabular data is the home turf of boosted trees, not deep networks. Document this in the report.

---

## Rubric

| Criterion | Weight | "Great" looks like |
|-----------|------:|--------------------|
| It runs | 10% | Fresh clone → `pip install -r requirements.txt` → `jupyter nbconvert --execute` → no errors |
| Standardization + preprocessing | 10% | Numeric features standardized; categoricals (if any) handled; missing values addressed |
| LR baseline | 10% | Trained, scored, reported; the number is the bar to clear |
| MLP architecture | 15% | 2-layer MLP from your `mlp.py`; correct shapes; He init; ReLU + softmax |
| Gradient check | 15% | Run; max relative error reported per parameter; all < 1e-5 |
| Training curves | 15% | 2-panel figure (loss + test accuracy); both saved; both readable |
| MLP outperforms baseline | 10% | Or: matches baseline + the honest paragraph explains why that is acceptable |
| Confusion matrix and error analysis | 10% | Heatmap saved; worst-confused class named; one misclassified example described |
| Report readability | 5% | The 2-page `report.md` could go to a hiring manager without explanation |

---

## What this exercises (and what comes next)

This mini-project exercises every concept from Week 7:

- The forward pass for a 2-layer MLP (Lecture 1, all sections).
- The backward pass derived from the chain rule (Lecture 2, all sections).
- The gradient check (Lecture 2, Section 10).
- The mini-batch SGD loop (Lecture 3, Sections 1-3).
- The training-curve discipline (Lecture 3, Section 7).
- The honest paragraph about what the model can and cannot do (Lecture 3, Section 8).
- The C5 discipline of "the baseline is the bar; if you do not beat it, you have to explain why."

Week 8 leaves NumPy for PyTorch. The same architecture compresses to 25 lines; the same training loop compresses to 10 lines. The Friday-of-Week-7 takeaway you carry into Week 8 is: *I know what the framework is doing on my behalf; I know how to read its training curve; I know what `loss.backward()` is computing; I will trust it.* That is the C5 voice and the C5 discipline.

---

## Submission

Push your `crunch-ai-portfolio-<yourhandle>` repo. Open the `week-07/mlp_tabular.html` link from the repo README so a reviewer with no Python install can read your work. Paste the summary in the commit message:

```text
Week 7 mini-project: 2-layer NumPy MLP on Wisconsin Breast Cancer.

dataset:                Wisconsin Breast Cancer (569 rows, 30 features, 2 classes)
preprocessing:          StandardScaler on the 30 numeric features
baseline (LR):          test_acc 0.965 (115 test examples)
MLP:                    test_acc 0.974 (same 115 test examples)
                        h=32, lr=0.05, batch=32, 50 epochs
gradient check:         max relative error = 4.2e-7  (PASSED)
confusion (MLP):        [[ 41,   2],   true negatives, false positives
                         [  1,  71]]   false negatives, true positives
worst confusion:        2 benign predicted as malignant; 1 malignant predicted as benign
honest paragraph:       MLP beats LR by 0.9 percentage points;
                        a 26% reduction in test error;
                        but on this small dataset that gap is plausibly noise
                        (the test set is only 115 examples).
```

That is the artifact recruiters will see. The "MLP beat LR by 0.9 pp / 26% reduction in test error / gradient check passed" three-liner is the headline.
