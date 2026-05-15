# Lecture 2 — Evaluation, fairness audits, and model cards

> *Reading time: about 80 minutes. Companion experiments at the end take another 30. Required reading for this lecture: Mitchell et al. 2019 ("Model Cards for Model Reporting", <https://arxiv.org/abs/1810.03993>), all of it; Buolamwini and Gebru 2018 ("Gender Shades", <https://proceedings.mlr.press/v81/buolamwini18a.html>), at least the abstract and section 4.*

---

## 1. The metric question, restated

Lecture 1 named the metric question: *which number says the model is good enough to deploy*. Lecture 2 expands the question. Picking the right metric is half the job; reporting it correctly is the other half; auditing it across slices of the population is the work that the field has decided distinguishes a defensible model from an undefensible one.

The three commitments of this lecture:

1. **Report metrics on the test set, once, at the end.** Cross-validation, hyperparameter search, threshold tuning, calibration — all on training and validation. The test set is the unbiased estimator of generalization performance; touching it more than once inflates the estimate.
2. **Report metrics per slice, not just aggregate.** A model with 90 percent aggregate accuracy and 50 percent accuracy on a 5-percent minority slice is not the same model as one with 88 percent aggregate accuracy and 87 percent on that minority slice. The Mitchell template grades the second model higher.
3. **Write the model card as a one-page document, following the Mitchell et al. 2019 template.** The card is the production interface between the model and the rest of the world. It is the artifact most-read by hiring managers; it is the artifact that, in industry, gates the deploy decision.

---

## 2. Metrics that fit the problem

The Week 4 lecture introduced the metric vocabulary; the capstone makes the choice load-bearing.

### 2.1 Binary classification with class imbalance

The default temptation is to report accuracy. Resist it. For an imbalanced binary problem with 20 percent positive rate, `DummyClassifier(strategy="most_frequent")` achieves 80 percent accuracy by predicting `negative` for everything. Accuracy is not informative.

The metrics to report:

- **Confusion matrix.** Always. Two rows, two columns, four numbers — true positives (TP), false positives (FP), true negatives (TN), false negatives (FN). Every other binary-classification metric is a function of these four.
- **Precision.** `TP / (TP + FP)`. "Of the instances the model predicted positive, what fraction were actually positive."
- **Recall** (also called sensitivity, or true positive rate). `TP / (TP + FN)`. "Of the instances that were actually positive, what fraction did the model find."
- **F1.** `2 * P * R / (P + R)`. The harmonic mean of precision and recall.
- **ROC curve.** Plots TPR against FPR (`FP / (FP + TN)`) for every threshold. The area under the ROC curve (ROC-AUC) is a single-number summary; ROC-AUC = 0.5 is chance, 1.0 is perfect.
- **Precision-recall curve.** Plots precision against recall for every threshold. PR-AUC is the area; the no-skill baseline is the positive rate, not 0.5. PR-AUC dominates ROC-AUC as a summary metric when the positive rate is below 5 percent.

The C5 default reporting for binary classification: confusion matrix + precision + recall + F1 + ROC-AUC + PR-AUC + a precision-recall curve plot saved to `figures/pr_curve.png`. Six numbers and one plot.

The reporting on the model card is *not* the cross-validation mean; it is the held-out test-set number. The cross-validation mean lives in the training notebook for hyperparameter selection; the test-set number lives in the card.

### 2.2 Threshold selection

A binary classifier with continuous scores has a threshold. The default threshold of 0.5 is rarely the right one for an imbalanced problem; the right threshold is the one that maximizes the metric you care about on the validation set.

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

# probs is the validation-set predicted-positive probabilities.
# y_val is the validation-set ground truth.
precision, recall, thresholds = precision_recall_curve(y_val, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = int(np.argmax(f1_scores[:-1]))  # last entry of precision/recall has no threshold.
best_threshold: float = float(thresholds[best_idx])
print(f"Best threshold: {best_threshold:.3f}")
print(f"F1 at best threshold: {f1_scores[best_idx]:.3f}")
```

The threshold goes into the model card; the threshold is part of the model. A model with threshold 0.27 and a model with threshold 0.5 are operationally different models even if the weights are identical.

### 2.3 Regression metrics

The C5 default for regression: RMSE + MAE + R-squared + a residual plot. The residual plot reveals what the scalar metrics hide.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

rmse: float = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae: float = float(mean_absolute_error(y_test, y_pred))
r2: float = float(r2_score(y_test, y_pred))

print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R^2:  {r2:.3f}")
```

The residual plot:

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(0.0, color="black", linewidth=0.5)
plt.xlabel("Predicted")
plt.ylabel("Residual (true - predicted)")
plt.title("Residuals vs predicted")
plt.savefig("figures/residuals.png", dpi=120)
```

If the residual plot has a fan shape (variance grows with prediction), the model has heteroskedasticity and RMSE is a misleading summary; switch to MAPE or quantile regression. If the residual plot has a U or inverted-U shape, the model is non-linear and a linear regression is misspecified; switch to a tree-based model.

### 2.4 Multi-class metrics

Macro F1 (the average of per-class F1) and micro F1 (the global F1 ignoring class structure) are both reported. The per-class confusion matrix is plotted as a heatmap with `seaborn.heatmap(confusion_matrix(y_test, y_pred), annot=True)`. The two classes with the highest mutual confusion are named in the model card with a one-paragraph hypothesis for why.

### 2.5 Sequence metrics

For sequence-to-label NLP: same as multi-class classification. For sequence-to-sequence (translation, summarization): BLEU-4 with `sacrebleu`, ROUGE-L with `rouge-score`, exact-match accuracy. Report all three.

### 2.6 Time-series metrics

MAPE, sMAPE, MASE, plus a walk-forward backtest plot showing predicted vs actual for the held-out future window. The plot is more informative than any scalar metric; the plot goes in the model card.

---

## 3. The slice-based fairness audit

The audit is the work that catches the failure mode named "the model is great on average but terrible for a specific population". Buolamwini and Gebru 2018 demonstrated it empirically on commercial face-classification systems: the IBM, Microsoft, and Face++ APIs had error rates of 0.0 to 0.8 percent on light-skinned men, 1.5 to 12.0 percent on light-skinned women, 0.0 to 6.0 percent on dark-skinned men, and 20.8 to 34.7 percent on dark-skinned women. Aggregate accuracy was high; the worst-slice accuracy was catastrophic. The paper is fifteen pages; read it before drafting your audit.

### 3.1 The audit, mechanically

For every protected attribute the dataset carries (sex, race, age bracket, geography, native language, etc.), compute the headline metric per slice on the test set:

```python
import pandas as pd
from sklearn.metrics import f1_score

# y_test is the test-set ground truth; y_pred is the model predictions; X_test is the test features.
# The protected attribute is a column in X_test, e.g. X_test["sex"].

audit_rows: list[dict[str, float | str]] = []
for slice_name, slice_idx in X_test.groupby("sex").groups.items():
    slice_y_true = y_test.loc[slice_idx]
    slice_y_pred = pd.Series(y_pred, index=y_test.index).loc[slice_idx]
    n: int = len(slice_idx)
    f1: float = float(f1_score(slice_y_true, slice_y_pred))
    audit_rows.append({"slice": str(slice_name), "n": n, "f1": f1})

audit = pd.DataFrame(audit_rows)
print(audit.to_string(index=False))
```

The output is a table with one row per slice, columns for slice name, slice size, and the headline metric. The same table is computed for every protected attribute and saved to `figures/fairness_audit.csv` and rendered in the model card.

### 3.2 The threshold question

If the gap between the best-performing slice and the worst-performing slice exceeds a threshold, the audit flags it. The C5 default threshold is **5 percentage points on the headline metric**, but the right threshold is dataset-specific and is named in the model card.

Why the threshold matters: a 1-point gap is probably noise. A 5-point gap is probably real. A 20-point gap is the Buolamwini-and-Gebru failure mode and is a reason to either not deploy the model or to deploy it with a written acknowledgement of the limitation in the user-facing UI.

### 3.3 Mitigation, in three rungs

If the audit reveals a gap, the model card states the gap and the chosen mitigation. The C5 rubric accepts any of three rungs of effort:

- **Rung 1: document.** Acknowledge the gap in the model card and warn against deploying the model for the affected population.
- **Rung 2: rebalance and retrain.** Resample the training data to give the underrepresented population a larger weight; retrain; re-audit. The Fairlearn `Reweighing` algorithm is the canonical reference.
- **Rung 3: constrained optimization.** Train the model with an explicit fairness constraint (equalized odds, demographic parity, etc.). The Fairlearn `ExponentiatedGradient` reduction is the canonical reference.

The C5 capstone rewards Rung 1 (the documentation rung) as the floor; Rungs 2 and 3 are bonus credit. The reason the rubric does not require Rung 2 is that for many problems Rung 2 makes the worst-slice metric better while making the aggregate metric worse, and the trade-off requires a stakeholder decision that the C5 capstone does not have a stakeholder for. Document the gap honestly.

### 3.4 The case where the dataset has no protected attributes

A regression on Austin home prices has no obvious protected attribute. The audit still applies; you slice by:

- **Geography.** ZIP code, neighborhood, urban / suburban / rural.
- **Time.** First half of training data vs second half (the most common "fairness" issue for non-demographic data is *concept drift*, where the model is worse on data from the period it was trained least on).
- **Magnitude.** Bottom decile of the target distribution vs top decile (the model is often much worse on the rare extremes).

The audit's purpose is to find the populations the model serves badly. There is always *some* such population. The model card names them.

> **EXPERIMENT 2.1 — audit your model.** After training, run the per-slice audit code above for every column in `X_test` that could be a protected attribute. Print the table. Identify the worst-performing slice. Write one paragraph explaining the gap; commit to one of the three mitigation rungs.

---

## 4. The Mitchell model card

Mitchell et al. 2019 (<https://arxiv.org/abs/1810.03993>) is eight pages. Section 4 is the template. The template's nine sections are:

1. **Model details.** Name, version, type, training algorithm, parameters, fairness constraints, license, contact, citation.
2. **Intended use.** Primary intended use, primary intended users, out-of-scope use cases.
3. **Factors.** Relevant factors (groups, instrumentation, environments) that affect the model's performance; evaluation factors actually used.
4. **Metrics.** Model performance measures, decision thresholds, variation approaches.
5. **Evaluation data.** Datasets, motivation for the choice, preprocessing.
6. **Training data.** May mirror the dataset card. Or summarize.
7. **Quantitative analyses.** The aggregate metrics; the disaggregated (per-slice) metrics.
8. **Ethical considerations.** The decisions made about ethical issues during model development; the limitations.
9. **Caveats and recommendations.** What additional testing or work is needed before the model can be deployed in a given context.

The C5 `MODEL_CARD_TEMPLATE.md` keeps all nine sections at the cost of brevity — each section is two to four sentences or bullets. The total length is one printed page.

### 4.1 A worked example: the Adult Income binary-classification model

**Model details.** `lightgbm` `LGBMClassifier` with `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`, all other defaults. Trained 2026-05-09 on UCI Adult Income (1994 US Census; CC BY 4.0). Model artifact `model.joblib` SHA256 `a1b2c3...`. License: MIT. Contact: <jeanstephane@aloyd.com>.

**Intended use.** Predict whether an adult's annual income exceeds 50,000 US dollars from demographic and employment features. Intended for benchmarking and capstone teaching. *Not* intended for any real income-classification application; the 1994 data does not reflect current labor-market patterns.

**Factors.** Relevant factors: sex (Male/Female), race (5 census categories), age, education level, occupation, native country. Evaluation slices: sex and race.

**Metrics.** F1 at threshold 0.27 (tuned on validation). Decision threshold reported. Variation: standard deviation across 5-fold CV.

**Evaluation data.** 16,281-row held-out test split from UCI Adult Income.

**Training data.** 32,561-row UCI Adult Income train split. See `data/DATASET_CARD.md`.

**Quantitative analyses.** Aggregate: F1 = 0.71, ROC-AUC = 0.92, PR-AUC = 0.79, threshold = 0.27. Baseline (`DummyClassifier`): F1 = 0.0, accuracy = 0.762. Disaggregated by sex: F1(Male) = 0.74, F1(Female) = 0.61; gap = 13 points, exceeds 5-point threshold. Disaggregated by race: F1(White) = 0.72, F1(Black) = 0.65, F1(Asian-Pac-Islander) = 0.72, F1(Amer-Indian-Eskimo) = 0.55, F1(Other) = 0.60; gap = 17 points, exceeds 5-point threshold.

**Ethical considerations.** The fairness gap on sex and race is large; this reflects 1994 labor-market patterns and the limited size of minority slices in the data. Deploying this model for any individual-decision use case would propagate historical discrimination. The recommended use is benchmarking only.

**Caveats and recommendations.** Do not use for income screening, credit decisions, insurance underwriting, or any other individual consequential decision. The data is 30 years old. The slice-level performance gaps are documented above (Rung 1 mitigation). A production deploy would require Rung-2 or Rung-3 mitigation and re-collection on current data.

That is the entire card, in roughly half a printed page. The rubric grades against this density. Verbose cards score lower than concise cards.

### 4.2 The model card in your repo

The card lives at the repo root as `MODEL_CARD.md`. The Hugging Face Hub also expects a model card; if you push the model to the Hub, the `README.md` in the model repo serves the same purpose. The two cards should be the same content with format adjustments.

> **EXPERIMENT 2.2 — draft your model card before training is done.** Open `MODEL_CARD_TEMPLATE.md`, fill in everything you can — sections 1, 2, 5, 6, 8, 9 — before the model has finished training. Sections 4 and 7 fill in after the metrics are computed. The discipline of writing the card before the metrics arrive prevents the post-hoc rationalization that "the model is great" — you commit to the intended-use language before you know the headline number.

---

## 5. Versioning datasets and models

Reproducibility is a graded artifact in the rubric. Three pieces:

1. **The dataset is versioned.** Either via DVC against a free remote, or by pinning a Hugging Face Datasets URL with a commit hash, or by committing a small CSV directly (if < 50 MB).
2. **The model is versioned.** Either by SHA256 in the model card, or by uploading to the Hugging Face Hub with a tag, or by pushing the `.joblib`/`.pt` file to a DVC remote.
3. **The training code is versioned.** Via git, on the capstone repo's `main` branch. The model card records the git commit SHA of the training run.

### 5.1 DVC, the data-version-control tool

DVC (<https://dvc.org/doc/start>) tracks large data files in git without storing the bytes in git. The workflow is:

```bash
# One-time setup.
pip install dvc dvc-s3       # or dvc-gs, dvc-azure, dvc-webdav, ...
dvc init
git add .dvc .dvcignore
git commit -m "init dvc"

# Track a dataset.
dvc add data/raw.csv          # creates data/raw.csv.dvc and updates .gitignore.
git add data/raw.csv.dvc data/.gitignore
git commit -m "track raw.csv with dvc"

# Configure a free remote (Cloudflare R2 example).
dvc remote add -d storage s3://my-bucket/dvc-store
dvc remote modify storage endpointurl https://<account-id>.r2.cloudflarestorage.com
dvc remote modify storage access_key_id <id>
dvc remote modify storage secret_access_key <secret>
dvc push
```

The `.dvc` file is a small YAML containing the file's MD5 hash; it lives in git. The bytes live in the remote. A `git clone` of the repo plus `dvc pull` reproduces the dataset exactly. The Cloudflare R2 free allowance covers small projects (10 GB free egress per month, 10 GB free storage).

### 5.2 LakeFS, the alternative

LakeFS (<https://docs.lakefs.io/>) is "git for data" at a different level — it sits in front of an object store and lets you branch and merge entire datasets. The LakeFS Cloud free tier (<https://lakefs.io/pricing/>) covers small projects. The C5 rubric accepts LakeFS as an alternative to DVC; the Lecture 2 default is DVC because the workflow is closer to git that students already know.

### 5.3 The Hugging Face Hub fallback

If the dataset lives on the Hugging Face Hub already, the simplest versioning is to pin the revision in the loading code:

```python
from datasets import load_dataset

ds = load_dataset(
    "imdb",
    revision="e6281661ce1c48d982bc483cf8a173c1bbeb5d31",  # a specific commit hash.
)
```

The commit hash makes the dataset version reproducible. The model card records the hash.

### 5.4 The model SHA256

The trained model's hash is computed once at the end of training and recorded in the model card:

```python
import hashlib
import joblib

joblib.dump(model, "model.joblib")

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

model_hash: str = sha256_of_file("model.joblib")
print(f"Model SHA256: {model_hash}")
```

The hash goes in `MODEL_CARD.md` section 1. If anyone wants to verify they have the same model you trained, they recompute the hash and compare.

### 5.5 The determinism caveat

"Reproducible" is a sliding scale. Setting `random_state=42` on the train/test split and on the model is the first rung. Setting `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, and `torch.use_deterministic_algorithms(True)` is the second rung. The third rung (true bit-for-bit determinism across machines and CUDA versions) is impossible in 2026 for non-trivial models; PyTorch documents the limits at <https://pytorch.org/docs/stable/notes/randomness.html>. The C5 capstone rubric grades the first rung as table stakes; the second rung as bonus credit; the third rung is not graded.

---

## 6. Logging the training run

Every training run produces a `runs/run-<datetime>/` directory with:

- `config.yaml` — every hyperparameter used, including the random seed.
- `metrics.json` — every metric on validation and test.
- `model.joblib` (or `model.pt`) — the trained model.
- `model_sha256.txt` — the hash of the model file.
- `git_commit.txt` — the output of `git rev-parse HEAD`.
- `requirements.txt` — the output of `pip freeze`.

The C5 starter `train.py` writes this directory automatically. The structure makes the run reproducible: a second run with the same `config.yaml` and `requirements.txt` should produce a model with the same SHA256 (modulo the determinism caveat above).

A minimal `config.yaml` for the LightGBM Adult Income example:

```yaml
seed: 42
data:
  source: uci/adult
  hash: 9c2bdec5e2c8a18d4ab46bb6ce5d0e4f3f7c8c2c2b3d8e8f8f8f8f8f8f8f8f8f
  test_size: 0.15
  val_size: 0.15
model:
  type: lightgbm.LGBMClassifier
  n_estimators: 200
  learning_rate: 0.05
  num_leaves: 31
threshold: 0.27
metrics:
  val_f1: 0.71
  val_roc_auc: 0.92
git_commit: 4f3c8c2c2b3d8e8f8f8f8f8f8f8f8f8f4f3c8c2c
```

---

## 7. The "no test-set peeking" mechanical check

The rubric grader runs a programmatic check on `train.py`:

```bash
grep -c "X_test\|y_test" train.py  # should be <= 4.
grep -c "test_set\|TEST" train.py  # should be 0 in hyperparameter search blocks.
```

If the count is over 4, the grader looks for the offending lines. A `print(f"Test shape: {X_test.shape}")` for debugging is acceptable but cleanly counted; a `model.score(X_test, y_test)` inside the cross-validation loop is a fail. The rubric is mechanical here on purpose — it is the single most-common cause of inflated metrics.

> **EXPERIMENT 2.3 — run the grep on your `train.py`.** Before submitting, count the `X_test`/`y_test` references. If over 4, refactor.

---

## 8. The C5 metric reporting block

Every capstone `train.py` ends with a block that looks like:

```python
def final_eval(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    """The only function in train.py that touches X_test / y_test."""
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    metrics: dict[str, float] = {
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "threshold": float(threshold),
    }
    return metrics


if __name__ == "__main__":
    # ... (training, validation, threshold tuning) ...
    test_metrics = final_eval(model, X_test, y_test, threshold=best_threshold)
    print(test_metrics)
    with open("runs/latest/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
```

The `final_eval` function is called exactly once. The output is saved to a JSON file. The model card pulls the numbers from the JSON file. The grader greps for `final_eval(`; the grader counts the occurrences.

---

## 9. Recap and next

- Pick the metric family that fits the problem; never report only accuracy on imbalanced data.
- Tune the threshold on validation; report the threshold in the model card.
- Run a slice-based fairness audit on every protected attribute the dataset carries; report the per-slice metric in the model card.
- Commit to a mitigation rung (document, rebalance, constrained-train) and report which one.
- Write the model card as a one-page document following the Mitchell template; draft it before training finishes.
- Version the dataset with DVC, LakeFS, or a pinned Hub revision. Version the model with SHA256 in the card.
- Log every training run to `runs/run-<datetime>/`.
- Touch the test set once.

Lecture 3 closes the loop: packaging the model behind a thin inference layer (`predict.py`, `serve.py`), deploying to a free public URL on Hugging Face Spaces, Streamlit Community Cloud, or Fly.io, and setting up basic drift monitoring on SQLite. The capstone is the integration.
