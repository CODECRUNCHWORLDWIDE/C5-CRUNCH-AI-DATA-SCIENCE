# Week 6 — Homework

Six problems, about six hours total. Commit each in your Week 6 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — K-means on Gaussian blobs, varying `cluster_std` (1 hour)

Generate four datasets with `sklearn.datasets.make_blobs(n_samples=400, centers=4, random_state=42)` for `cluster_std` ∈ `[0.3, 0.6, 1.0, 2.0]`. For each:

1. Fit `KMeans(n_clusters=4, n_init=10, random_state=42)`.
2. Compute the silhouette score.
3. Record `adjusted_rand_score` against the known cluster labels.

Save the 2×2 figure as `homework/01-blob-std.png` showing all four scatter plots colored by predicted labels, each titled with `cluster_std` and the silhouette score. In `homework/01-blob-std.md` (~150 words):

1. How does the silhouette score change as `cluster_std` grows?
2. At what `cluster_std` does the ARI drop below 0.8?
3. The honest C5 reading: "as the clusters overlap more, k-means recovers them with progressively less fidelity, and the silhouette score honestly reflects this." Confirm in your own data.

---

## Problem 2 — DBSCAN on two-moons; pick `eps` from the k-distance plot (1 hour)

Generate `X, _ = make_moons(n_samples=500, noise=0.06, random_state=42)`. Standardize with `StandardScaler`.

1. Draw the **k-distance plot** for `min_samples=5` (distance to the 5th-nearest neighbor for each row, sorted). Save as `homework/02-k-distance.png`.
2. Identify the "knee" of the curve visually; record the `eps` value at the knee.
3. Fit `DBSCAN(eps=<the value>, min_samples=5)`. Count the number of clusters and noise points.
4. Save the scatter plot as `homework/02-dbscan-moons.png`, colored by predicted labels (noise in gray, clusters in violet shades).

In `homework/02-dbscan.md` (~150 words):

1. Where did the knee land in your k-distance plot?
2. How many clusters did DBSCAN return? Did it recover both moons?
3. What happens if you double `eps`? Halve it? Record both.

---

## Problem 3 — PCA on `load_digits`; pick `k` for 90% variance (1 hour)

Load `sklearn.datasets.load_digits()`. Standardize. Fit `PCA(n_components=64)` (the full set of components).

1. Plot the scree plot (explained variance ratio per component) and the cumulative explained variance, side by side. Save as `homework/03-scree.png`.
2. Find the smallest `k` such that the cumulative explained variance is ≥ 0.90. Print that `k`.
3. Reconstruct the data from the top `k` components (`X_reconstructed = pca.inverse_transform(pca.transform(X))` with `n_components=k`) and compute the reconstruction error in Frobenius norm.

In `homework/03-pca.md` (~150 words):

1. How many components capture 90% of the variance?
2. What does the scree plot's shape tell you about the data — is the structure concentrated in a few directions or spread across many?
3. The honest reading: PCA reduces dimensionality without preserving the *task* of digit classification — the reconstruction error can be small while the downstream accuracy drops if the discarded components carried digit-discriminating information.

---

## Problem 4 — UMAP vs PCA on the same dataset (45 min)

Pick **one** of: `load_digits`, `load_wine`, or the **20-newsgroups** subset of your choice (use `fetch_20newsgroups_vectorized(subset='train', remove=('headers', 'footers', 'quotes'))` and take the first 2000 rows).

In `homework/04-umap-vs-pca.py`:

1. Fit `PCA(n_components=2)`. Compute the explained variance.
2. Fit `umap.UMAP(n_components=2, random_state=42)`.
3. Plot both as a side-by-side scatter, colored by the known labels (digit, wine class, or newsgroup). Save as `homework/04-umap-vs-pca.png`.

In `homework/04-umap-vs-pca.md` (~150 words):

1. Which method shows clearer separation of the known classes?
2. PCA's "preserved" structure is *linear variance*; UMAP's is *local neighborhood*. State which property your data benefits from more.
3. Would you use either embedding as features for a downstream classifier? Why or why not? Lecture 2, Section 11.

---

## Problem 5 — The honest "is `k` real?" protocol on Iris (45 min)

Load `sklearn.datasets.load_iris(as_frame=True)`. Standardize.

In `homework/05-is-k-real.py`:

1. Run the five-step protocol from Lecture 3, Section 7 for `k=3`:
   a. Elbow plot of WCSS for `k ∈ [2..10]`.
   b. Silhouette diagram at `k=3`.
   c. Column-permutation reshuffle baseline: `real_silhouette / perm_silhouette`.
   d. Re-fit with three random seeds; compute pairwise `adjusted_rand_score`.
   e. 2D PCA scatter colored by the predicted labels at `k=3`.
2. Save a four-panel figure `homework/05-protocol.png` with the elbow, the silhouette diagram, the reshuffle bars, and the PCA scatter.

In `homework/05-is-k-real.md` (~200 words):

1. Report the five numbers (elbow location, mean silhouette, reshuffle ratio, mean pairwise ARI across seeds, visual separation yes/no).
2. The classic Iris result: silhouette is higher at `k=2` than at `k=3` (setosa is far from the others), but `k=3` matches the species labels. Reseeding stability is essentially perfect (ARI > 0.99). The reshuffle baseline gives a ratio of ~2.5–3.0. PCA shows clear separation. Five strong "real" signals.
3. Did your numbers match the C5 expectations? If not, name the discrepancy.

---

## Problem 6 — Reflection (30 min)

Write `homework/06-reflection.md` (250–400 words) answering:

1. The from-scratch k-means in Exercise 1 is forty lines. Did its brevity surprise you, given how famous the algorithm is? Where did the implementation get harder than you expected — the k-means++ initialization, the empty-cluster handling, or somewhere else?
2. The four signs from Lecture 3 Section 2 (smooth elbow, low silhouette, instability under reseeding, no 2D separation) all point to "the data does not cluster." Have you reviewed a project (yours or someone else's) where one or more of these signs were present but were ignored? What would the honest report have said?
3. PCA, t-SNE, and UMAP each preserve a different property of the data. Which one's failure mode worries you most for the kinds of data you typically work with, and why?
4. After this week, which question do you ask first when someone says "we should cluster this": "what is the right `k`?" or "does this even cluster?" Be specific about why.
5. What is the one habit from this week you want to keep when you move to Week 7 (neural networks from scratch)? "Always plot before drawing conclusions" carries over; "always check the reshuffle baseline" might be your new one.

Honest is more valuable than polished.

---

## Time budget

| Problem | Time |
|--------:|----:|
| 1 | 1 h |
| 2 | 1 h |
| 3 | 1 h |
| 4 | 45 min |
| 5 | 45 min |
| 6 | 30 min |
| **Total** | **~5 h** |

(The schedule allocates 6h for homework; the remaining ~1h is buffer for installing `umap-learn` and `numba` on macOS without rushing.)

When done, push your Week 6 repo and start (or finish) the [challenge](./challenges/) and the [mini-project](./mini-project/README.md).
