# Challenge 1 — Justify the K

**Time estimate:** 2 hours.

## Problem statement

Fit k-means on the **Iris dataset** (or the **Wine dataset** — your choice; both ship with scikit-learn and have known class structure), produce three independent "what is the right `k`?" diagnostics — the **WCSS elbow plot**, the **silhouette curve**, and the **gap statistic** — and write a short honest paragraph defending your choice of `k`. Compare your chosen `k` to the dataset's known class count: if your `k` does not match the known number of classes, write the second honest paragraph that explains why.

The point is not to recover the known `k` — sometimes the diagnostics disagree with the labels, and the disagreement is informative. The point is to *defend a choice* using evidence and to write the defense in plain English.

## What you will produce

A single script `justify_k.py` and a short markdown file `notes.md` (~200 words) that:

1. Loads `sklearn.datasets.load_iris(as_frame=True)` (or `load_wine`). Standardize with `StandardScaler`.
2. Computes the three diagnostics over `k` in `[2, 3, ..., 10]`:
   - **WCSS** (`KMeans(n_clusters=k).inertia_`).
   - **Silhouette score** (`sklearn.metrics.silhouette_score`).
   - **Gap statistic** (Lecture 3, Section 4; or your implementation from Exercise 3).
3. Saves a three-panel figure `justify_k.png` showing all three curves side by side, with the chosen `k` annotated on each.
4. Picks a `k` and prints both the chosen value and the silhouette score at that `k`.
5. Writes the honest paragraph in `notes.md`.

## Acceptance criteria

- [ ] `justify_k.py` runs end-to-end with `python justify_k.py` and prints the chosen `k`, the silhouette at that `k`, and the gap statistic at that `k`.
- [ ] `python -m py_compile justify_k.py` succeeds.
- [ ] `justify_k.png` is saved with axes labeled and a sentence-form title.
- [ ] `notes.md` (~200 words) explains the chosen `k` using *all three* diagnostics. The C5 voice: specific, sober, no hand-waving.
- [ ] The script uses `random_state=42` everywhere randomness is involved.
- [ ] If the diagnostics disagree, the disagreement is named in `notes.md` and one is chosen with a reason.

## The notes prompt

Open `notes.md` and answer in roughly 200 words:

1. **Which `k` does the elbow plot suggest?** Look for the bend in the WCSS curve. On Iris, the bend is usually at `k = 3`; on Wine, at `k = 3`. If the curve is smooth, say so.
2. **Which `k` maximizes the silhouette score?** State the value and the score.
3. **Which `k` does the gap statistic suggest?** Use the rule from Lecture 3 Section 4: the smallest `k` such that `Gap(k) >= Gap(k+1) - se(k+1)`.
4. **Do the three agree?** Often two of the three agree on a value of `k` and the third disagrees. State which two and why.
5. **What is the known number of classes in the dataset, and does it match?** Iris has 3 classes; Wine has 3. If your chosen `k` matches, that is encouraging. If it does not, name a reason (the algorithm sees natural clusters that do not correspond to the human-labeled classes; or, conversely, the human-labeled classes are not well-separated in the feature space).

## Hints

<details>
<summary>If the gap statistic is slow</summary>

The gap statistic with `B=10` reference samples and a `k` range of `[2..10]` requires `9 * 11 = 99` k-means fits. On Iris (150 rows), this is under 5 seconds. If yours is much slower, reduce `n_init` from 10 to 5 — the reference data is uniform, so the gap statistic does not benefit much from multiple restarts.

</details>

<details>
<summary>If the three diagnostics give three different answers on Iris</summary>

This is the *expected* outcome on Iris. The WCSS elbow on Iris is at `k = 3` (or sometimes `k = 2`, depending on the standardization). The silhouette score on standardized Iris is actually higher at `k = 2` (the setosa class is so far from the other two that grouping versicolor and virginica together gives a higher silhouette than separating them). The gap statistic typically picks `k = 2` for the same reason. The "right" answer depends on what you mean: `k = 3` matches the species labels; `k = 2` matches the dominant cluster structure in the feature space. Both are defensible. The honest paragraph is the one that recognizes the tradeoff.

</details>

<details>
<summary>If your script is hard to read</summary>

Stick to four functions: one for each diagnostic, plus a `main()` that calls all three and saves the figure. Use the `numpy.random.default_rng(seed)` pattern from Exercise 3. Do not over-engineer; this is a 2-hour challenge.

</details>

## Suggested layout

```python
"""Three k-selection diagnostics on the Iris dataset."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PLOT_VIOLET = "#7C3AED"
GRAY = "#9CA3AF"
RANDOM_STATE = 42


def wcss_curve(X, k_range):
    return np.array([
        KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X).inertia_
        for k in k_range
    ])


def silhouette_curve(X, k_range):
    out = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        labels = KMeans(n_clusters=k, n_init=10,
                        random_state=RANDOM_STATE).fit(X).labels_
        out[i] = silhouette_score(X, labels)
    return out


def gap_statistic(X, k_range, B=10):
    rng = np.random.default_rng(RANDOM_STATE)
    n, p = X.shape
    lo, hi = X.min(axis=0), X.max(axis=0)
    gap = np.zeros(len(k_range))
    se = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        real = KMeans(n_clusters=k, n_init=10,
                      random_state=RANDOM_STATE).fit(X).inertia_
        log_real = np.log(real)
        log_refs = np.zeros(B)
        for b in range(B):
            X_ref = rng.uniform(lo, hi, size=(n, p))
            log_refs[b] = np.log(
                KMeans(n_clusters=k, n_init=10, random_state=b).fit(X_ref).inertia_
            )
        gap[i] = log_refs.mean() - log_real
        se[i] = log_refs.std() / np.sqrt(B)
    return gap, se


def pick_k_gap(gap, se):
    """Smallest k such that Gap(k) >= Gap(k+1) - se(k+1)."""
    for i in range(len(gap) - 1):
        if gap[i] >= gap[i + 1] - se[i + 1]:
            return i  # index, not the value of k
    return len(gap) - 1


def main():
    data = load_iris(as_frame=True)
    X = StandardScaler().fit_transform(data.data)
    k_range = list(range(2, 11))

    wcss = wcss_curve(X, k_range)
    sils = silhouette_curve(X, k_range)
    gap, se = gap_statistic(X, k_range, B=10)

    chosen_idx_silhouette = int(np.argmax(sils))
    chosen_idx_gap = pick_k_gap(gap, se)

    print(f"silhouette peaks at k = {k_range[chosen_idx_silhouette]}, "
          f"score = {sils[chosen_idx_silhouette]:.3f}")
    print(f"gap statistic suggests k = {k_range[chosen_idx_gap]}, "
          f"gap = {gap[chosen_idx_gap]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].plot(k_range, wcss, "o-", color=GRAY)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("WCSS")
    axes[0].set_title("Elbow plot")

    axes[1].plot(k_range, sils, "o-", color=PLOT_VIOLET)
    axes[1].axvline(k_range[chosen_idx_silhouette], color=PLOT_VIOLET, ls="--", alpha=0.4)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("mean silhouette")
    axes[1].set_title("Silhouette curve")

    axes[2].errorbar(k_range, gap, yerr=se, fmt="o-", color="#5B21B6")
    axes[2].axvline(k_range[chosen_idx_gap], color="#5B21B6", ls="--", alpha=0.4)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("gap")
    axes[2].set_title("Gap statistic")

    fig.suptitle("Three diagnostics for choosing k on standardized Iris")
    fig.savefig("justify_k.png", dpi=150)


if __name__ == "__main__":
    main()
```

## Stretch goals

- **Repeat on a dataset that does not cluster.** Use `sklearn.datasets.make_classification(n_samples=300, n_features=8, n_informative=3, random_state=42)` and discard the labels. The three diagnostics should disagree more dramatically; the gap statistic should be near zero across the range.
- **Add the column-permutation baseline.** Run the diagnostics on `X` and on a column-permuted `X_perm`. Plot all three curves for both. The "real" curves should diverge meaningfully from the "permuted" curves on clustered data and lie on top of each other on noise.
- **Add a 2D PCA + UMAP overlay.** Plot the four-panel "Iris in PCA(2)" and "Iris in UMAP(2)" colored by the cluster labels at your chosen `k`. The visual separation should match the silhouette score.
- **Repeat on Mall Customers.** The "Mall Customer Segmentation" dataset (Kaggle, linked in `resources.md`) is small and gentle. The three diagnostics on it usually agree on `k = 5`. The mini-project will revisit this in more depth.

## Why this matters

The mini-project asks for the same defense paragraph on a real customer dataset. Doing the challenge first means the "defend your `k`" section of the mini-project report is a known quantity, not a five-hour Saturday struggle.

In production, "we segmented our customers into 5 groups" is a sentence that follows hundreds of project hours and shows up on slides. The data scientist who wrote it has either: (a) defended the 5 with evidence, or (b) picked the first elbow they saw. The difference is the difference between a segmentation that survives the next quarter and one that gets quietly abandoned.

## Submission

Commit `justify_k.py`, `justify_k.png`, and `notes.md` to your Week 6 repo. Paste the chosen `k` and the disagreement summary in the commit message:

```text
Challenge 1 -- justify the k on Iris.

elbow plot suggests:       k = 3
silhouette score peaks at: k = 2 (score = 0.581)
gap statistic suggests:    k = 2

chosen k: 3 (matches species labels; silhouette penalty 0.04)
honest note: gap and silhouette both pick k=2 because setosa is
extremely far from the other two species; k=3 still matches the
known biology and is the defensible operational choice.
```

Anyone reading the commit message should know which `k` you chose and why in five seconds.
