# Lecture 1 — K-means, Hierarchical, DBSCAN, and Gaussian Mixtures

> **Outcome:** You can write down the k-means objective and Lloyd's algorithm in five lines of pseudocode, implement it in pure NumPy, and verify it agrees with `sklearn.cluster.KMeans`. You can name what hierarchical agglomerative clustering, DBSCAN, and Gaussian mixture models *each* do that k-means does *not*, and pick between the four with a one-paragraph justification. After this lecture, "clustering" stops being a single technique and becomes a small family of techniques with non-overlapping strengths.

Five weeks of supervised learning have trained you to write `model.fit(X, y)` and check a number against the test set. Unsupervised learning has no `y`; the fit takes only `X`, and the "result" is a partition of the rows (clustering) or an embedding of the rows (dimensionality reduction; Lecture 2). The discipline shifts: you can no longer measure success by RMSE or accuracy. You measure it by *plotting* the result, *probing* it under reseeding and reshuffling, and *writing a paragraph* about whether the result is real or is an artifact of the algorithm.

This lecture is about the four clustering algorithms you actually use in 2026. The next lecture is about three dimensionality-reduction algorithms. The third lecture is the honest critique — when clustering returns clusters that do not exist.

We target **scikit-learn 1.5+** (the `cluster` and `mixture` modules), **numpy 2.x**, **scipy 1.11+** (for `scipy.cluster.hierarchy.linkage` in the hierarchical section).

---

## 1. The unsupervised setting

A clustering algorithm takes a feature matrix `X` of shape `(n, p)` and returns a vector `labels` of shape `(n,)` whose entries are integer cluster IDs in `{0, 1, ..., k-1}` (or `-1` for "noise," in DBSCAN). The integer labels carry no inherent meaning: cluster 0 and cluster 1 are no more related than cluster 0 and cluster 7. The algorithm has *partitioned* the rows; it has not *named* the partitions.

The first practical consequence is that **two runs of the same algorithm can swap cluster IDs**. If you fit k-means twice with two different random seeds, the row that was "cluster 0" in run A might be "cluster 2" in run B. The partitions can still be identical up to a label permutation. Use `sklearn.metrics.adjusted_rand_score` to compare clusterings without being confused by relabelings; do not use `accuracy_score`, which is meaningless here.

The second consequence is that **the algorithm always returns clusters**. K-means with `k=5` returns five clusters whether or not the data has any natural group structure. If you ask "is the resulting partition meaningful?", the algorithm has no answer; that is your job, with the help of the silhouette score (Section 7), the gap statistic (Lecture 3), the 2D embedding (Lecture 2), and your judgment.

---

## 2. K-means: the objective in one line

K-means asks: given `n` rows and a target cluster count `k`, find the partition that minimizes the total squared Euclidean distance from each row to its assigned cluster's mean.

Formally, the **within-cluster sum of squares** is:

```text
WCSS  =  sum over clusters c
         sum over rows i in c
         || x_i  −  μ_c ||²
```

where `μ_c` is the mean of the rows in cluster `c`. The k-means problem is:

```text
minimize_{partition P, centroids μ}   WCSS(P, μ)
```

The minimization is over both the partition and the centroids. Given a partition, the optimal centroids are the cluster means (this is a calculus exercise — take the gradient of WCSS with respect to `μ_c` and set it to zero; you get `μ_c = mean(x_i : i in c)`). Given centroids, the optimal partition assigns each row to its nearest centroid. The two updates alternate; this is **Lloyd's algorithm**.

> **EXPERIMENT — derive the optimal centroid.** Write out WCSS for a single cluster `c`: `f(μ_c) = sum_{i in c} (x_i − μ_c)^T (x_i − μ_c)`. Take the gradient with respect to `μ_c`: `∇f = −2 · sum_{i in c} (x_i − μ_c) = 0` implies `μ_c = (1/|c|) · sum_{i in c} x_i`. That is the cluster mean. Two lines of calculus; the entire centroid update.

---

## 3. Lloyd's algorithm

The algorithm in five lines:

```text
initialize:  pick k centroids (k-means++ or uniform random)
repeat:
    assign:  for each row i, label_i = argmin_c || x_i − μ_c ||²
    update:  for each c, μ_c = mean(x_i : label_i = c)
until labels do not change between iterations
```

That is the entire algorithm. Three remarks:

1. **It converges in finitely many steps.** There are only `k^n` possible partitions; each iteration strictly decreases WCSS or leaves it unchanged; once WCSS is unchanged, the algorithm has converged. (In practice, "finitely many" is ≤ 300 iterations; sklearn's default `max_iter=300` is rarely reached.)
2. **It converges to a *local* minimum.** Lloyd's algorithm is coordinate descent on a non-convex objective. Different initializations converge to different partitions. The k-means++ initialization (Section 5) reduces but does not eliminate this.
3. **The Euclidean distance is hard-coded.** K-means is *only* sensible for features where Euclidean distance is meaningful — that is, continuous features on comparable scales. Categorical features and unscaled features both break k-means.

---

## 4. The minimal Python implementation

The following 40 lines fit k-means in pure NumPy. It is what you build in Exercise 1, with two simplifications: uniform-random initialization, and a fixed `max_iter` rather than convergence detection.

```python
"""A 40-line k-means in pure NumPy."""

import numpy as np


def kmeans(X: np.ndarray, k: int, *, max_iter: int = 300,
           random_state: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """K-means with uniform-random init.

    Returns:
        labels   -- (n,) int array of cluster IDs
        centroids -- (k, p) array of cluster means
        inertia  -- float, the final WCSS
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape

    # Initialize: pick k rows uniformly at random as centroids.
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign: nearest centroid per row.
        # || x - μ ||² = ||x||² - 2 x·μ + ||μ||²; ||x||² is constant per row.
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update: cluster means.
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = X[mask].mean(axis=0)
            # If a cluster goes empty, leave its centroid alone; sklearn
            # re-initializes by picking the farthest-from-any-centroid row.

    inertia = float(((X - centroids[labels]) ** 2).sum())
    return labels, centroids, inertia
```

That is the entire algorithm. Test it against `sklearn.cluster.KMeans` on a `make_blobs` dataset and the two agree on the partition (up to a label permutation). Exercise 1 makes this rigorous.

The pairwise-distance computation in the assignment step is `O(n · k · p)` per iteration. For `n` in the millions, this is the wall-clock bottleneck and `MiniBatchKMeans` is the answer (sample a mini-batch each iteration; converges to a slightly worse WCSS but is `10–100×` faster).

---

## 5. K-means++: why initialization matters

Uniform-random initialization gives bad local minima often enough to matter. The pathological case is two clusters far apart; if both initial centroids land in the same cluster, Lloyd's algorithm splits that one cluster in half and never recovers.

**K-means++** (Arthur and Vassilvitskii, 2007) fixes this with a smarter sampling scheme:

```text
1. Pick the first centroid uniformly at random.
2. For each subsequent centroid:
   - Compute D_i = distance from x_i to the nearest already-chosen centroid.
   - Sample the next centroid with probability proportional to D_i².
3. Run Lloyd's algorithm from these initial centroids.
```

The squared distance in step 2 is the only trick. It biases the next centroid toward rows far from the existing ones — that is, toward the cluster the existing centroids have not covered. The resulting WCSS is `O(log k)` times the optimum *in expectation*, a guarantee that uniform initialization does not have.

In practice, sklearn runs k-means++ initialization followed by Lloyd's algorithm `n_init` times (default 10) and returns the partition with the lowest WCSS. The extra initializations cost wall-clock but reduce the variance of the result substantially.

> **EXPERIMENT — initialization matters.** Generate three Gaussian blobs in 2D with `make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42)`. Fit `KMeans(n_clusters=3, init="random", n_init=1, random_state=0)` ten times with different `random_state` values. Print `inertia_` for each. The values will vary by 20–50% between runs. Now repeat with `init="k-means++", n_init=10`. The variation collapses to under 1%. That is the value of k-means++ in one experiment.

---

## 6. Three failure modes of k-means

K-means assumes that clusters are:

1. **Round (isotropic).** The Euclidean-distance metric treats all directions equally. A cluster that is long and thin will be cut in half by k-means; it sees the elongated cluster as two round ones. The fix is to use a Mahalanobis-distance variant or a Gaussian mixture model (Section 9).
2. **Equal-density.** The WCSS objective sums squared distances; a cluster with `1000` rows contributes 1000 squared-distance terms while a cluster with `100` rows contributes 100. The optimal partition tends to break the dense cluster in half to reduce the large total. The fix is hierarchical or density-based clustering (Sections 8 and 9).
3. **Equally scaled (Euclidean-comparable).** If feature 1 is dollars (range 0–1,000,000) and feature 2 is years (range 0–100), the dollar feature dominates the distance metric and k-means effectively clusters on dollars alone. The fix is to standardize every feature (`StandardScaler`) before fitting.

The three failure modes are easy to see on synthetic data:

- **Non-convex clusters.** `make_moons(n_samples=300)` returns two interlocking half-moons. K-means cuts each moon in half and assigns the four pieces to two clusters that look like vertical halves. DBSCAN, in contrast, recovers the moons exactly.
- **Unequal density.** Generate three blobs with `cluster_std=[0.5, 0.5, 3.0]`. The third blob's variance is six times the others. K-means splits the wide cluster.
- **Unequal scale.** Generate two clusters where feature 1 differs by 1000 between clusters and feature 2 differs by 5. The unscaled distance is dominated by feature 1; k-means recovers the two clusters trivially. Now scale feature 2 to a range of `[0, 1000]` and the result rotates; k-means now clusters on feature 2.

> **EXPERIMENT — k-means on two moons.** `from sklearn.datasets import make_moons; X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)`. Fit `KMeans(n_clusters=2, random_state=42)` and plot the predicted labels colored on the scatter of `X`. The two predicted clusters are vertical halves of the plot, not the two moons. Now fit `DBSCAN(eps=0.2, min_samples=5)` on the same `X` and re-plot. The two moons appear as the two predicted clusters. Same data, two algorithms; only one was a sensible choice. The same story has played out on a dozen real datasets in the literature.

---

## 7. Evaluating a clustering: the silhouette score

Without a `y`, you cannot compute accuracy or RMSE. The closest analogue is the **silhouette score**:

For each row `i`:
- `a_i` = mean distance from `i` to other rows *in the same cluster*.
- `b_i` = mean distance from `i` to rows in the *nearest other cluster*.
- `silhouette_i = (b_i − a_i) / max(a_i, b_i)`.

The score is in `[−1, +1]`:
- `+1` means `i` is much closer to its own cluster than to any other (good).
- `0` means `i` is on the boundary between two clusters.
- `−1` means `i` is closer to another cluster than to its own (bad; usually a sign of misassignment).

The mean silhouette over all rows is a single scalar summary; the per-row values are the input to the **silhouette diagram**, a horizontal bar chart sorted by cluster with one bar per row. The silhouette diagram is the single most useful clustering diagnostic. It shows which clusters are tight and which are loose, which clusters have rows on the wrong side, and roughly how big each cluster is.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

mdl = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X)
print(f"mean silhouette = {silhouette_score(X, mdl.labels_):.3f}")
# Per-row silhouettes for the silhouette diagram:
sil = silhouette_samples(X, mdl.labels_)
```

**Reading a silhouette score:**
- `> 0.5` — "strong" clustering. The clusters are well-separated.
- `0.25 – 0.5` — "weak" clustering. The clusters have meaningful structure but with overlap.
- `< 0.25` — no real cluster structure. The data is continuous; the partition is an artifact.

On Lecture 3 we revisit the "below 0.25" case at length. It is more common on real data than the textbook examples suggest.

---

## 8. Hierarchical agglomerative clustering

Hierarchical clustering takes a different approach: instead of asking "what is the best partition into `k` groups?", it builds the entire tree of partitions from `n` singleton clusters up to one all-rows cluster, and lets you cut the tree at any height.

The algorithm in three lines:

```text
1. Start with n singleton clusters (each row in its own cluster).
2. Find the two closest clusters; merge them into one.
3. Repeat until one cluster remains. Record every merge.
```

The output is a **dendrogram** — a tree drawn with merge heights on the y-axis. A horizontal cut of the dendrogram at height `h` recovers the clustering at the moment where all clusters within distance `h` had been merged. The "right" `k` is suggested by the height of the largest *gap* between successive merges; large gaps indicate that the algorithm had to climb a long way to merge across that boundary, which is a sign of meaningful cluster structure.

The **linkage** rule is the "closest" definition. Four standard choices:

| Linkage | "Distance between clusters A and B" | Use when |
|---------|-------------------------------------|----------|
| **Single** | `min_{a in A, b in B} d(a, b)` | The clusters are connected by chains; finds elongated structures. Vulnerable to "chaining" — one stray row can bridge two distinct clusters. |
| **Complete** | `max_{a in A, b in B} d(a, b)` | The clusters should be tight. Tends to find compact spherical clusters; the opposite extreme from single. |
| **Average** | `mean_{a in A, b in B} d(a, b)` | A reasonable middle ground; rarely the best but rarely the worst. |
| **Ward** | The increase in within-cluster variance from the merge | The right default for Euclidean data with round clusters. Behaves similarly to k-means but returns a full hierarchy. |

For most data, **Ward linkage** is the right choice. Use single linkage only when you have a specific reason to find chain-like structure (which is rare; DBSCAN does this better).

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Fit and predict at k=4:
mdl = AgglomerativeClustering(n_clusters=4, linkage="ward").fit(X)
labels = mdl.labels_

# Draw the full dendrogram (use scipy for the plotting):
Z = sch.linkage(X, method="ward")
fig, ax = plt.subplots(figsize=(10, 4))
sch.dendrogram(Z, ax=ax, truncate_mode="level", p=5)
ax.set_title("Ward linkage on X")
plt.tight_layout()
plt.savefig("dendrogram.png")
```

> **EXPERIMENT — read the dendrogram before picking `k`.** Generate four blobs in 2D and run agglomerative clustering with Ward linkage. Plot the dendrogram. The four largest merge heights are dramatically taller than the rest — that gap is the dendrogram telling you `k=4` is the right cut. Now generate two blobs that overlap heavily (`cluster_std=2.0` instead of `0.6`). The dendrogram has no clear gap; the algorithm has produced a tree but the data does not strongly suggest any specific `k`.

The hierarchical algorithm has two practical limits:
1. **`O(n²)` memory** for the pairwise distance matrix (and `O(n² log n)` to `O(n³)` time depending on linkage). On `n > 30,000` it gets slow; on `n > 100,000` it becomes unusable. For larger data, fit hierarchical on a sample.
2. **No "predict" method.** Agglomerative clustering does not generalize to new rows; it is purely a partition of the rows you fit on. K-means and GMM both have `predict`, which is useful when new data arrives.

---

## 9. DBSCAN: density-based clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise; Ester et al., 1996) flips the question. Instead of "what partition minimizes the total squared distance?", DBSCAN asks "where are the regions of high density, and which points are in them?"

Two parameters:
- **`eps`** — a radius. Two points are "neighbors" if they are within `eps` of each other.
- **`min_samples`** — a count. A point is a **core point** if it has at least `min_samples` neighbors within `eps`.

The algorithm:
1. Label each point as core, border, or noise:
   - **Core** — at least `min_samples` neighbors within `eps`.
   - **Border** — within `eps` of a core point, but not core itself.
   - **Noise** — neither core nor reachable from a core point.
2. Form clusters as connected components of core points, plus the borders attached to each component.
3. Noise points are not assigned to any cluster (`label = -1`).

DBSCAN's strengths:
- **Finds arbitrary-shape clusters.** The two-moons example that defeats k-means works trivially with DBSCAN: each moon is a connected chain of core points.
- **No `k`.** DBSCAN decides the cluster count from the data. The number of clusters is an output, not a hyperparameter.
- **Handles noise.** Points in low-density regions are explicitly labeled as outliers, not forced into the nearest cluster.

DBSCAN's weaknesses:
- **The two parameters are delicate.** The right `eps` and `min_samples` depend on the data's scale and density. The standard recipe (Section 5 of the original paper) is the **k-distance plot**: for each row, compute the distance to its `k`-th nearest neighbor; plot the sorted distances; the "knee" of the curve is the suggested `eps`. The recipe works but requires inspection.
- **Variable density defeats it.** If one cluster is dense and another is sparse, no single `eps` correctly captures both. The standard workaround is **OPTICS** (`sklearn.cluster.OPTICS`), a related algorithm that uses a variable-density notion of reachability.
- **Scale sensitive.** As with k-means, unstandardized features break DBSCAN. Always `StandardScaler` first.

```python
from sklearn.cluster import DBSCAN

mdl = DBSCAN(eps=0.3, min_samples=5).fit(X_scaled)
labels = mdl.labels_           # -1 = noise; 0, 1, 2, ... = cluster IDs
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"found {n_clusters} clusters, {n_noise} noise points")
```

The k-distance plot for picking `eps`:

```python
from sklearn.neighbors import NearestNeighbors

nbr = NearestNeighbors(n_neighbors=5).fit(X_scaled)
dists, _ = nbr.kneighbors(X_scaled)
k_dists = np.sort(dists[:, 4])     # distance to the 5th-nearest neighbor
plt.plot(k_dists)
plt.ylabel("distance to 5th-nearest neighbor")
plt.xlabel("rows sorted by k-distance")
plt.title("Pick eps at the 'knee' of this curve")
```

> **EXPERIMENT — k-distance plot to pick `eps`.** Generate three blobs in 2D, standardize, and draw the k-distance plot for `k = min_samples = 5`. The curve has a sharp knee; `eps` should be set just below the knee. Pick `eps` at the knee, fit DBSCAN, and verify it recovers the three blobs. Now intentionally pick `eps` ten times larger; DBSCAN merges all three blobs into one cluster. Pick `eps` ten times smaller and almost everything is labeled noise. The k-distance plot lets you find the right value in one figure.

---

## 10. Gaussian mixture models

K-means is the limiting case of a more general model: a **Gaussian mixture model** (GMM) treats the data as drawn from a mixture of `k` Gaussian distributions, each with its own mean, covariance, and mixing weight.

The model:

```text
P(x)  =  sum_{c=1..k}  π_c · N(x | μ_c, Σ_c)
```

where `π_c` are the mixing weights (`sum_c π_c = 1`), `μ_c` are the means, and `Σ_c` are the covariance matrices. Fitting the GMM is the **expectation-maximization (EM)** algorithm:

```text
E-step (expectation):
    for each row i and cluster c:
        γ_ic = π_c · N(x_i | μ_c, Σ_c) / sum_j π_j · N(x_i | μ_j, Σ_j)
    γ_ic is the posterior probability that row i was drawn from cluster c.

M-step (maximization):
    π_c = (1/n) sum_i γ_ic                                 # mixing weight
    μ_c = (sum_i γ_ic · x_i) / (sum_i γ_ic)               # mean
    Σ_c = (sum_i γ_ic · (x_i - μ_c)(x_i - μ_c)^T) / (sum_i γ_ic)  # covariance

repeat until log-likelihood stops increasing.
```

GMM generalizes k-means in three ways:

1. **Soft assignments.** Each row gets a probability vector over clusters (the `γ_ic`), not a hard label. A row halfway between two clusters has `γ ≈ [0.5, 0.5]`. K-means is the limiting case where all probabilities are 0 or 1.
2. **Elliptical clusters.** The covariance matrix `Σ_c` can be arbitrary (`covariance_type="full"`), shared across clusters (`"tied"`), diagonal (`"diag"`), or isotropic (`"spherical"`). K-means is the limiting case of spherical equal-variance covariances.
3. **A probabilistic interpretation.** Because GMM is a generative model, you can compute the likelihood of new data, the BIC (Bayesian Information Criterion) for picking `k`, and use the model as a density estimate. K-means has none of these.

```python
from sklearn.mixture import GaussianMixture

mdl = GaussianMixture(n_components=4, covariance_type="full",
                      random_state=42).fit(X)
labels = mdl.predict(X)              # hard assignments (argmax of γ)
probs  = mdl.predict_proba(X)        # soft assignments (the γ matrix)
print(f"log-likelihood: {mdl.score(X):.3f}")
print(f"BIC: {mdl.bic(X):.1f}")      # smaller is better; useful for picking k
```

**When to use GMM over k-means:** when clusters are *elliptical* (not round), when you want *soft* assignments, when you want to *score* new data under the cluster model, or when you want to pick `k` with the BIC.

**When not to use GMM:** when clusters are not Gaussian at all (use DBSCAN), when the data has more than a few hundred features (the covariance matrices have `p²` parameters each and become hard to estimate), or when you simply want a fast partition (k-means is `5–10×` faster).

> **EXPERIMENT — elliptical clusters defeat k-means.** Generate two elongated clusters in 2D by drawing from two highly-correlated Gaussians: `cov = [[1.0, 0.9], [0.9, 1.0]]`; one centered at `(0, 0)`, one at `(5, 5)`. Fit `KMeans(n_clusters=2)` and `GaussianMixture(n_components=2, covariance_type="full")`. K-means returns a partition that cuts perpendicular to the line `y=x`; GMM recovers the two elongated clusters. The covariance matrices in the fitted GMM reveal the elongation.

---

## 11. Choosing between the four

A one-paragraph decision rule:

- **K-means** — when clusters are round, equal-density, and standardized; when `k` is known or can be elbow-plotted; when speed matters. The default for "I just need to cluster this; what should I try first?"
- **Hierarchical (Ward linkage)** — when `k` is unknown and you want to inspect the dendrogram; when the data has nested or hierarchical structure (e.g., taxonomic data, document hierarchies); when `n < 30,000` and you want to read the merge sequence.
- **DBSCAN** — when clusters have arbitrary shapes; when you expect noise / outliers that should not be forced into a cluster; when the data has roughly uniform density inside clusters. Avoid when cluster densities differ substantially.
- **GMM** — when clusters are elliptical (correlated features within a cluster); when you want soft assignments or per-cluster densities; when the BIC for model selection is appealing. Avoid past `p ≈ 50` without a covariance regularization scheme.

A more visual rule of thumb is sklearn's "Comparison of clustering algorithms" figure (linked in `resources.md`). The figure runs every clustering algorithm on six synthetic datasets and shows where each succeeds and where each fails. Print it; tape it to your monitor.

---

## 12. The clustering checklist

Before you ship a clustering result, walk this list:

- [ ] **`X` was standardized.** `StandardScaler` before `KMeans` / `DBSCAN` / `GMM` unless features are already comparable.
- [ ] **`random_state` was set.** K-means and GMM are stochastic; without `random_state`, two runs disagree.
- [ ] **`n_init` was raised** for k-means and GMM (10–20 instead of 1). Reduces the chance of converging to a bad local minimum.
- [ ] **The mean silhouette score is reported.** Above 0.5 is good; below 0.25 is a warning.
- [ ] **The clustering was inspected in 2D** (PCA / UMAP — Lecture 2). If the embedding shows no separation, neither does the data.
- [ ] **The number of clusters was justified** by at least two of the three (elbow plot, silhouette score, gap statistic). The honest paragraph from Lecture 3 follows.
- [ ] **The result was compared to a reshuffling baseline.** Cluster the same data with a permuted `X` (column-wise) and compare. If the silhouette score barely drops, the original "clusters" were noise.
- [ ] **At least one row was *labeled by hand* with its cluster ID and probed.** "Cluster 3 contains 230 customers whose median age is 41 and whose median spend is $87" is a sentence that earns a paragraph in the report.

---

## 13. Where this leaves you

You can now write the k-means objective in one line, derive the centroid update in two lines of calculus, implement Lloyd's algorithm in 40 lines of NumPy, and verify it agrees with sklearn. You can name three failure modes of k-means and pick a more appropriate algorithm for each one — hierarchical for nested structure, DBSCAN for arbitrary shapes with noise, GMM for elliptical clusters with soft assignments. You can read a silhouette score and a dendrogram.

The next lecture is about *seeing* the clusters: PCA for linear structure, t-SNE and UMAP for nonlinear structure. Without a 2D plot of the result, every clustering report has a hole in it. After Lecture 2 you will have the plot. After Lecture 3 you will have the honest paragraph about whether the clusters in the plot are real.

The mini-project at the end of the week takes a real customer or product dataset and asks: **find meaningful clusters, justify the `k`, visualize the result in 2D, and write one paragraph about whether the clusters are real**. That is the test the rest of the week prepares you for.
