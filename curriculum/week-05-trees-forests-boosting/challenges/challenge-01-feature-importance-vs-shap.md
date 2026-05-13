# Challenge 1 — Feature Importance vs SHAP

**Time estimate:** 2 hours.

## Problem statement

Fit a gradient-boosted-trees regressor on the California Housing dataset, compute three different feature-importance rankings — **Gini / split-gain importance** (free from the fit), **permutation importance** (model-agnostic, costs `n_features × n_repeats` predict passes), and **SHAP values** (game-theoretic, exact for tree models via `TreeExplainer`) — and **find at least one feature where the three rankings disagree**. Write a short note explaining why.

The point is not that one method is right and the others are wrong. The point is that the three methods measure *different things*, and the differences are interpretable. A working data scientist knows which question each method answers and which one to lead with for a given audience.

## What you will produce

A single script `importances.py` and a short markdown file `notes.md` (~200 words) that:

1. Loads `sklearn.datasets.fetch_california_housing(as_frame=True)`.
2. Splits 80/20 train/test with `random_state=42`.
3. Fits `HistGradientBoostingRegressor` (or `XGBRegressor`, or `LGBMRegressor` — your choice) with sensible hyperparameters and early stopping. The point of this challenge is the *importance* methods, not the tuning.
4. Computes three importance rankings:
   - `mdl.feature_importances_` (split-gain).
   - `sklearn.inspection.permutation_importance(mdl, X_test, y_test, n_repeats=10)`.
   - `shap.TreeExplainer(mdl).shap_values(X_test)` and take `np.abs(...).mean(axis=0)` as the per-feature SHAP magnitude.
5. Combines the three into a single comparison DataFrame, ranks each, and prints the ranks side by side.
6. Saves a figure `importances.png` showing all three rankings as a grouped horizontal bar chart.
7. Identifies **at least one feature** where the three rankings disagree (different position in the ranked list).

## Acceptance criteria

- [ ] `importances.py` runs end-to-end with `python importances.py` and prints a comparison DataFrame plus a "rankings disagree on feature X" summary line.
- [ ] `python -m py_compile importances.py` succeeds.
- [ ] `importances.png` is saved with axes labeled and a sentence-form title.
- [ ] `notes.md` (~200 words) explains *why* the three rankings disagree on the feature(s) you identified. The C5 voice: specific, sober, no hand-waving.
- [ ] The script uses `random_state=42` everywhere randomness is involved (the model, `permutation_importance`, the train/test split). A second run with the same seed produces identical numbers.

## The notes prompt

Open `notes.md` and answer in roughly 200 words:

1. **Which features rank highly in all three methods?** These are the unambiguously-important features. Name them.
2. **Which feature do the three methods rank most differently?** Be specific: feature name, rank under each method.
3. **What is the most likely reason for the disagreement?** The C5 short list:
   - **Cardinality bias.** Gini importance is biased toward features with many candidate thresholds; continuous features outrank binary features even when the binary feature is more informative. (Lecture 2, Section 7.)
   - **Correlated features.** Permutation importance underestimates importance for correlated features because permuting one creates impossible rows. SHAP's `TreeExplainer` handles this with the conditional-expectation framing, but the result can still feel counterintuitive.
   - **Training-set vs held-out measurement.** Gini importance measures impurity reduction on the *training* set inside each tree; permutation and SHAP measure the effect on a held-out set. Overfitting inflates Gini.
4. **Which ranking would you lead with in a stakeholder presentation, and why?** The honest C5 answer is usually "permutation importance for the headline, SHAP for the per-row explanations, Gini importance only as a sanity check." Your answer may differ; back it up.

## Hints

<details>
<summary>If the SHAP install fails</summary>

`pip install "shap>=0.45,<1"`. On some macOS setups, SHAP's transitive dependency on `numba` fails to build; the fast `TreeExplainer` does *not* need numba, so you can skip the failed dependency by `pip install --no-deps shap` and then `pip install matplotlib pandas slicer cloudpickle` (the runtime dependencies that `TreeExplainer` actually needs).

</details>

<details>
<summary>If the three rankings agree perfectly</summary>

They will not, on a real dataset. If your script reports identical rankings, you have probably:

1. Compared only the top three features (which usually agree). Look at the full ranked list.
2. Ranked by raw importance instead of by relative-to-the-best (a feature that is "0.05 in Gini, 0.05 in permutation, 0.04 in SHAP" looks identical at three-decimal precision but the ranks may differ).
3. Used a single-feature decision-stump dataset. California Housing has 8 features with measurable correlation; the rankings will differ on at least one.

</details>

<details>
<summary>If SHAP takes too long</summary>

`shap.TreeExplainer` runs in polynomial time on tree models; on California Housing it should finish in under 10 seconds. If it does not, you may have accidentally used the model-agnostic `shap.KernelExplainer`, which is exponential in `n_features` and effectively unusable here. Use `shap.TreeExplainer(mdl)` — the `Tree` part is what makes it fast.

</details>

## Suggested layout

```python
"""Three feature-importance methods on California Housing."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


PLOT_VIOLET = "#7C3AED"
GRAY = "#9CA3AF"


def main() -> None:
    # 1. Load and split.
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    # 2. Fit.
    mdl = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=2000,
        max_depth=6,
        early_stopping=True,
        n_iter_no_change=50,
        random_state=42,
    ).fit(X_tr, y_tr)

    # 3. Three importances.
    gini = pd.Series(mdl.feature_importances_, index=X.columns, name="gini")
    perm = permutation_importance(mdl, X_te, y_te, n_repeats=10,
                                  random_state=42, n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=X.columns, name="permutation")

    explainer = shap.TreeExplainer(mdl)
    shap_values = explainer.shap_values(X_te)
    shap_s = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns, name="shap")

    # 4. Combine, rank, print.
    df = pd.concat([gini, perm_s, shap_s], axis=1)
    df["gini_rank"] = df["gini"].rank(ascending=False).astype(int)
    df["perm_rank"] = df["permutation"].rank(ascending=False).astype(int)
    df["shap_rank"] = df["shap"].rank(ascending=False).astype(int)
    df = df.sort_values("perm_rank")
    print(df.round(4))

    # 5. Disagreement?
    rank_cols = ["gini_rank", "perm_rank", "shap_rank"]
    disagree = df[df[rank_cols].nunique(axis=1) > 1]
    if len(disagree) > 0:
        print(f"\nRankings disagree on: {disagree.index.tolist()}")
    else:
        print("\nAll three methods agree on the ranking (rare).")

    # 6. Save the chart.
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    n = len(df)
    ypos = np.arange(n)
    h = 0.27
    ax.barh(ypos - h, df["gini"].to_numpy(),         height=h, label="Gini",        color=GRAY)
    ax.barh(ypos,     df["permutation"].to_numpy(),  height=h, label="Permutation", color=PLOT_VIOLET)
    ax.barh(ypos + h, df["shap"].to_numpy(),         height=h, label="SHAP",        color="#5B21B6")
    ax.set_yticks(ypos)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_xlabel("importance (un-normalized; ranks matter, not units)")
    ax.set_title("Three feature-importance methods rarely agree at the bottom")
    ax.legend()
    fig.savefig("importances.png", dpi=150)


if __name__ == "__main__":
    main()
```

## Stretch goals

- **Repeat on the Ames Housing dataset** (Week 4 / Week 5 mini-project). The three rankings disagree more dramatically there because of the high-cardinality categorical features (`Neighborhood`, 25 levels) and the multiplicative target structure.
- **SHAP summary plot.** Replace the per-feature SHAP magnitude with the full `shap.summary_plot(shap_values, X_test)`, which is the canonical SHAP figure (one row per feature, one dot per row of test data, colored by feature value). It is the chart that gets pasted into Kaggle write-ups.
- **A partial-dependence plot** for the disagreement feature. `sklearn.inspection.PartialDependenceDisplay.from_estimator` draws it in one line. Read it for "does the model use this feature in a smooth way or in a step-function way?"
- **A SHAP force plot** for one specific prediction. `shap.plots.force(explainer.expected_value, shap_values[0], X_test.iloc[0])` is the canonical per-row explanation chart. Save it as HTML and link from `notes.md`.

## Why this matters

In production, "why did the model predict X for this row?" is a question stakeholders ask weekly. The right answer is rarely "look at `feature_importances_`." Permutation importance is the right answer at the model level; SHAP is the right answer at the per-row level. Knowing which one to pull up for which question is the skill this challenge develops.

The mini-project at the end of the week uses the same `TreeExplainer` pattern on Ames Housing. Doing the challenge first means the SHAP plot in your portfolio piece is a known quantity, not a five-hour debugging session on Saturday.

## Submission

Commit `importances.py`, `importances.png`, and `notes.md` to your Week 5 repo. Paste the disagreement summary in the commit message:

```text
Challenge 1 -- feature importance vs SHAP.

top-3 unanimous: ...
disagreement feature(s): ...
likely cause: cardinality bias / correlation / training-vs-holdout
chosen leading method for stakeholder use: ...
```

Anyone reading the commit message should know which features the three methods disagree on in two seconds.
