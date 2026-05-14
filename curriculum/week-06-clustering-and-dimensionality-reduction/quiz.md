# Week 6 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** The k-means objective minimizes:

- A) The total Euclidean distance from each row to the cluster centroid.
- B) The within-cluster sum of squared Euclidean distances from each row to its assigned centroid (WCSS).
- C) The pairwise distance between every pair of rows in the same cluster.
- D) The negative log-likelihood under a Gaussian mixture with shared variance.

---

**Q2.** Lloyd's algorithm converges:

- A) To the global minimum of the WCSS objective.
- B) To a local minimum of the WCSS objective in finitely many steps. The local minimum depends on the initialization, which is why k-means++ exists and why sklearn defaults to `n_init=10`.
- C) After exactly `k` iterations.
- D) Only when the data is linearly separable.

---

**Q3.** The k-means++ initialization picks the next centroid:

- A) Uniformly at random from the data.
- B) As the row farthest from any already-chosen centroid.
- C) With probability proportional to the squared distance from the nearest already-chosen centroid. The squared distance is the key — linear distance would bias too softly.
- D) From the rows at the cluster boundaries of the previous run.

---

**Q4.** The single biggest preprocessing decision for k-means is:

- A) Whether to one-hot encode categorical features.
- B) Whether to standardize features so that each contributes comparably to Euclidean distance. K-means on a feature in dollars (range 0–1,000,000) and a feature in years (range 0–100) is effectively clustering on dollars alone.
- C) Whether to log-transform skewed features.
- D) Whether to remove outliers before fitting.

---

**Q5.** On the two-moons dataset (`sklearn.datasets.make_moons`), which clustering algorithm is most likely to recover the two crescents as the two clusters?

- A) `KMeans(n_clusters=2)`.
- B) `GaussianMixture(n_components=2)`.
- C) `DBSCAN(eps=0.2, min_samples=5)` on standardized data. DBSCAN finds connected components of dense regions, which is exactly the shape of each moon.
- D) `AgglomerativeClustering(linkage="ward")`.

---

**Q6.** A mean silhouette score of 0.18 on your best `k` choice means:

- A) The clusters are reasonable; silhouette around 0.2 is the typical industry average.
- B) The data does not contain real cluster structure at this set of features; the partition is essentially arbitrary. The honest paragraph from Lecture 3 applies.
- C) The wrong impurity function was used; switch from MSE to Gini.
- D) DBSCAN would do better; switch algorithms.

---

**Q7.** PCA's first principal component is:

- A) The direction in which the data varies the least.
- B) The direction in which the data varies the most. Formally, the eigenvector of the centered covariance matrix `X^T X / (n-1)` with the largest eigenvalue, or equivalently the first right-singular vector of the centered `X`.
- C) The mean of the data.
- D) The orthogonal projection of `X` onto the first two columns.

---

**Q8.** You apply PCA to a dataset and find that PC1 + PC2 together explain 12% of the variance. The honest interpretation is:

- A) The first two principal components are sufficient; project to 2D for downstream learning.
- B) The structure of the data is genuinely high-dimensional — most of the variance is spread across many dimensions, not concentrated in a few. A 2D PCA scatter is informative for visualization but will be very lossy. Either keep more components, switch to a nonlinear method like UMAP, or accept the 88% of variance you are not visualizing.
- C) PCA is broken; the data has no covariance.
- D) The features were not standardized — re-fit with `StandardScaler` first.

---

**Q9.** A 2D t-SNE scatter of your dataset shows four visible clusters. Which statement is **most likely false**?

- A) The local neighborhoods of each cluster are preserved in the high-dimensional space.
- B) The relative sizes of the four clusters in the scatter accurately reflect the number of rows in each cluster.
- C) Two clusters that appear close to each other in the scatter may be far apart in the high-dimensional space.
- D) Running t-SNE with a different `random_state` will produce a different scatter, possibly with a different visible cluster count.

---

**Q10.** Your stakeholder asks for "five customer segments" from a dataset of three continuous features (frequency, recency, monetary value). The honest first move is:

- A) Run k-means with `n_clusters=5` and present the five clusters.
- B) Check whether the data clusters at all — run the elbow plot, silhouette curve, and 2D embedding — and if it does not, propose a *quantile-based segmentation* (low / medium / high on each of the three features, giving 27 rule-based segments) as a more honest alternative. The user often wants "actionable segments," not "algorithmic clusters." Lecture 3, Section 5.
- C) Run UMAP and use the 2D coordinates as features for a downstream classifier.
- D) Run hierarchical clustering with Ward linkage at `n_clusters=5`.

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — K-means minimizes the within-cluster sum of *squared* distances (WCSS), `sum_c sum_{i in c} ||x_i - μ_c||²`. The squared form is important — it makes the optimal per-cluster representative the *mean* (taking the derivative gives a closed-form solution). With un-squared distance, the optimal representative would be the *geometric median*, and the algorithm would be the k-medians algorithm, a different beast. Lecture 1, Section 2.

2. **B** — Lloyd's algorithm is coordinate descent on a non-convex objective. Each iteration strictly decreases WCSS or leaves it unchanged; once unchanged, the algorithm has converged to a local minimum. There are finitely many partitions, so convergence is finite. Different initializations give different local minima, which is why k-means++ (initialization) and `n_init=10` (multiple restarts) matter so much. Lecture 1, Sections 3 and 5.

3. **C** — k-means++ samples each subsequent centroid with probability proportional to *squared* distance from the nearest already-chosen centroid. The squared distance is what gives the `O(log k)` approximation-ratio guarantee. Lecture 1, Section 5.

4. **B** — Euclidean-distance algorithms (k-means, DBSCAN, hierarchical, GMM with isotropic covariance) are dominated by the highest-range feature on unscaled data. Standardization with `StandardScaler` is the default first step for *every* clustering pipeline unless you have a specific reason otherwise. Lecture 1, Section 6.

5. **C** — DBSCAN recovers the two moons trivially because each moon is a connected chain of dense neighbors and DBSCAN finds connected components of core points. K-means and GMM both cut each moon in half because they prefer round / elliptical clusters. Hierarchical with Ward linkage behaves like k-means here for the same reason. Lecture 1, Section 9.

6. **B** — Silhouette below 0.25 across all `k` is the textbook signature of a continuous distribution with no real cluster structure. The partition exists (the algorithm always returns one) but is essentially arbitrary; reseeding will move many rows; the 2D embedding will show no separation. The honest paragraph from Lecture 3 Section 9 follows: "the data does not cluster at this set of features." Lecture 3, Section 2.

7. **B** — PC1 is the direction of maximal variance, formally the top eigenvector of `S = X^T X / (n-1)` (the sample covariance of centered `X`). Equivalently, the first right-singular vector of the centered `X`. PCA finds the *high*-variance directions; the *low*-variance directions are PC `p`, PC `p-1`, .... Lecture 2, Sections 1 and 2.

8. **B** — A scree plot where the top two components capture only 12% means the dataset's variance is spread across many directions. This is honest information: the structure is high-dimensional. A 2D PCA scatter will show a very lossy summary; switch to UMAP for visualization, or keep more components for downstream learning. Standardization is a reasonable thing to check (A) but does not change the *concentration* of variance, only its allocation across original features. Lecture 2, Sections 3 and 5.

9. **B** — t-SNE *does not* preserve cluster sizes or inter-cluster distances. A large blob in the scatter does not mean a sparse cluster in the original space; the algorithm normalizes per-point neighborhoods, expanding sparse regions and contracting dense ones. Local neighborhoods *are* preserved (option A is true); inter-cluster distances are *not* (option C is true); the result *is* stochastic (option D is true). Only B is false. Distill 2016 article in `resources.md` and Lecture 2 Section 7.

10. **B** — When the stakeholder says "segments" and the data is continuous, the right answer is often *not* clustering at all. Quantile-based or rule-based segmentation gives the user actionable, defensible segments without depending on a clustering algorithm's hyperparameters. K-means may also work — but only if the elbow plot, silhouette score, and 2D embedding all show real structure. Running k-means without first checking is the data-science move that produces marketing decks built on imaginary segments. Lecture 3, Section 5.

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
