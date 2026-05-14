# Week 6 — Clustering, Dimensionality Reduction, and Unsupervised Methods

> *Five weeks of supervised learning have trained you to ask "what is the right answer for this row?" Clustering asks a different and more delicate question: "if I had no answers at all, would the rows fall into recognizable groups?" The honest answer is "sometimes." Half of this week is teaching you the algorithms; the other half is teaching you when to put them down.*

Welcome to week six of **C5 · Crunch AI / Data Science**. Weeks 1–5 trained four supervised models on labeled targets — first a NumPy linear regressor, then `RidgeCV`, then a from-scratch decision tree, then a `HistGradientBoostingRegressor` that beat the Ridge baseline by ≥10% on Ames. Every one of those models had a `y` to compare predictions against. The validation story was the same every week: split the data, fit on train, score on test, defend the number.

This week, the `y` goes away.

K-means, hierarchical agglomerative clustering, DBSCAN, and Gaussian mixture models look at `X` alone and propose a partitioning of the rows. PCA, t-SNE, and UMAP look at `X` alone and propose a lower-dimensional embedding. None of these methods has a ground-truth error to score against. That changes the discipline more than the math: the *evaluation* of an unsupervised result is half-statistical, half-rhetorical, and you have to be honest about which half is doing the work.

Three warnings before we start, because Week 6 is where the temptation to "just cluster everything" gets out of hand:

1. **Clustering is a hypothesis, not a discovery.** When you run k-means with `k=5` and get five blobs, you have *not* discovered that the data has five clusters. You have asked k-means "what is the best 5-blob partition of these rows?" and it has answered. The same data with `k=7` will give you seven blobs that look just as plausible. Lecture 3 is the honest critique of this: the algorithm always returns clusters; the question is whether they *mean* anything.
2. **Most data does not cluster.** Real customer records, real product features, real sensor logs — most of them have continuous structure with no clear gaps. The silhouette score, the gap statistic, and the elbow plot will all give you a "best k," but the gap between the best k and the second-best k is small. Compare to truly clustered synthetic data (three Gaussian blobs in 2D) and the difference is stark. You will see this on real data this week and have to write it down.
3. **t-SNE and UMAP make pretty pictures that are easy to over-interpret.** Both are nonlinear dimensionality-reduction methods. Both can take 50-dimensional data and produce a 2D scatter that *looks* like four clusters. The clusters in the scatter may not correspond to clusters in the original space; the distances in the scatter may not correspond to distances in the original space. Lecture 2 walks through the cases where t-SNE fools you and the cases where it does not.

We target **scikit-learn 1.5+** (the `cluster`, `decomposition`, and `mixture` modules), **umap-learn 0.5+**, **numpy 2.x**, **pandas 2.2+**, **matplotlib 3.8+**. The APIs are stable; `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `GaussianMixture`, `PCA`, and `TSNE` have not moved since sklearn 1.0.

---

## Learning objectives

By the end of this week, you will be able to:

- **Derive** the k-means objective (within-cluster sum of squares) and the Lloyd's-algorithm update rule from first principles. State the three failure modes (non-convex clusters, unequal density, scale sensitivity) without looking them up.
- **Implement** k-means in pure NumPy in under 50 lines. Verify it agrees with `sklearn.cluster.KMeans` on synthetic Gaussian-blob data to floating-point tolerance after the same initialization.
- **Choose** between k-means, hierarchical agglomerative, DBSCAN, and Gaussian mixtures with a one-paragraph justification: k-means for round equal-density blobs, hierarchical for nested structure or unknown `k`, DBSCAN for arbitrary shapes with noise, GMM for soft assignments and elliptical clusters.
- **Justify a value of `k`** using the elbow plot (within-cluster SSE), the silhouette score, and the gap statistic — and recognize when none of the three gives a clean answer, in which case the right move is to question whether the data clusters at all.
- **Explain** PCA in one paragraph: the eigendecomposition of the centered covariance matrix returns orthogonal axes ranked by variance explained. Know what the scree plot is and what "explained variance ratio" means.
- **Decide** when PCA is the right preprocessing step (correlated features, dimensionality reduction for downstream supervised learning, visualization of linear structure) and when it is the wrong one (already-low dimension, nonlinear structure, the variance you are throwing away is the signal).
- **Use** t-SNE and UMAP for 2D visualization with the caveats: tune `perplexity` (t-SNE) or `n_neighbors` (UMAP), do not over-interpret cluster sizes or inter-cluster distances, never use t-SNE or UMAP coordinates as input to a downstream supervised model unless you have a very specific reason.
- **Recognize** the cases where clustering is not the answer: the data is continuous, the "clusters" are an artifact of preprocessing, the business question is actually a classification problem in disguise, the stakeholder wants segments but you are looking at a single distribution.
- **Ship** a mini-project on real customer (or product) records that finds clusters, justifies the `k` honestly, visualizes the result in 2D, and writes one paragraph about whether the clusters are real or are a story the algorithm told.
- **Pass** every `pytest` case on the Week 6 exercises.

---

## Prerequisites

- **Weeks 1, 2, 3, 4, and 5 complete.** In particular, you have the Week 5 `ames_boosted.ipynb` checked into your `crunch-ai-portfolio-<yourhandle>` repo with a documented test RMSE that beat Week 4 by ≥10%. That is the signal you can ship a defended supervised model end-to-end. Week 6 is the unsupervised counterpart.
- A working **Python 3.11+** install (we use 3.12).
- scikit-learn 1.5+ (`pip install "scikit-learn>=1.5,<2"`). We use `cluster.KMeans`, `cluster.DBSCAN`, `cluster.AgglomerativeClustering`, `mixture.GaussianMixture`, `decomposition.PCA`, `manifold.TSNE`, `metrics.silhouette_score`, plus `preprocessing.StandardScaler` from Week 4.
- umap-learn 0.5+ (`pip install "umap-learn>=0.5,<1"`). Optional for the lectures, required for Exercise 2 and the mini-project's bonus.
- pandas 2.2+ and numpy 2.x (from prior weeks).
- matplotlib 3.8+ (from Week 3 — we plot 2D embeddings, silhouette diagrams, scree plots, dendrograms).
- `pytest` for the exercise smoke tests.

No GPU. UMAP has an optional `rapidsai/cuml` GPU backend, but on the dataset sizes this week (<100k rows) CPU is faster than GPU once you account for the data transfer. We stay on CPU.

---

## Topics covered

- **The k-means objective.** Minimize the within-cluster sum of squares (WCSS) over a partition of the rows into `k` groups. Lloyd's algorithm is coordinate descent: alternate "assign each row to its nearest centroid" and "set each centroid to the mean of its rows" until the assignments stop changing. Converges in finitely many steps to a local minimum.
- **The k-means++ initialization.** Picking initial centroids uniformly at random gives bad local minima often. k-means++ picks them with probability proportional to squared distance from the nearest already-chosen centroid; the resulting WCSS is `O(log k)` times the optimum in expectation. This is the default in sklearn 1.0+.
- **Hierarchical agglomerative clustering.** Start with `n` singleton clusters; at each step, merge the two closest clusters under a *linkage* rule (single, complete, average, Ward). Returns the full dendrogram; cut at a chosen height to recover any `k`. Ward linkage minimizes within-cluster variance and is the most common choice for Euclidean data.
- **DBSCAN.** Density-based: a "core point" has at least `min_samples` neighbors within distance `eps`; clusters are connected components of core points plus their reachable neighbors; points reachable from no core point are labeled noise. Finds arbitrary-shape clusters; has no `k` parameter; *does* have two delicate parameters (`eps` and `min_samples`) and a notorious sensitivity to scale.
- **Gaussian mixture models.** Model the data as a mixture of `k` Gaussians, each with its own mean and covariance. Fit by EM: the E-step computes soft-assignment probabilities, the M-step re-estimates the means, covariances, and mixing weights. Generalizes k-means (which is a GMM with isotropic equal-variance Gaussians and hard assignments).
- **Evaluating clusterings.** Silhouette score (per-row contrast between within-cluster distance and nearest-other-cluster distance, averaged), Davies-Bouldin index (lower is better), Calinski-Harabasz (higher is better), the gap statistic (compares observed WCSS to WCSS on uniformly random data of the same shape). All three have failure modes; none of them is a substitute for plotting the result.
- **PCA from the covariance matrix.** Center `X`, compute `X^T X / (n-1)`, take its eigendecomposition (or the SVD of `X` directly), keep the top `k` eigenvectors. The eigenvectors are the principal directions; the eigenvalues are the variance explained along each direction. The "scree plot" of eigenvalues tells you where the signal stops and the noise starts.
- **t-SNE.** A nonlinear embedding that preserves *local* neighborhoods at the cost of *global* distances. The single hyperparameter `perplexity` (effective neighborhood size, default 30) controls the scale at which local structure is preserved. Stochastic; runs of t-SNE on the same data give different 2D coordinates. Use for visualization, not for input to downstream models.
- **UMAP.** A nonlinear embedding with a manifold-learning foundation (the "Uniform Manifold Approximation and Projection" name is literal — there is a real differential-geometry derivation behind it). Faster than t-SNE on large data, often preserves global structure better, has more hyperparameters (`n_neighbors`, `min_dist`, `metric`) but most can be left at defaults.
- **When clustering is not the answer.** Continuous distributions, the "segments" your stakeholder wants are really a classification problem, the elbow plot has no elbow, the silhouette score is below 0.2 across all `k`, the clusters move under reseeding, the 2D embedding shows no separation. Lecture 3 walks through the honest critique.

---

## Weekly schedule

Target about **38 hours**. Some sections click in twenty minutes; some take three hours. Treat the table as a budget, not a contract.

| Day       | Focus                                                       | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|-------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | k-means; Lloyd's algorithm; from-scratch                    |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | Hierarchical; DBSCAN; GMM; how to choose                    |   2.5h   |   2h      |     0h     |   0.5h    |   1h     |     0h       |   1h       |    7h       |
| Wednesday | PCA, t-SNE, UMAP; what each is for                          |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0h       |   6.5h      |
| Thursday  | When clustering is not the answer; honest evaluation        |   2h     |   1h      |     2h     |   0.5h    |   1h     |     2h       |   0h       |    8.5h     |
| Friday    | Mini-project: cluster real customer data, justify `k`       |   0h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    6h       |
| Saturday  | Mini-project: 2D viz + report; honest limitations           |   0h     |   0h      |     0h     |   0h      |   1h     |     3h       |   0h       |    4h       |
| Sunday    | Quiz, review, push to portfolio repo                        |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                             | **10.5h**| **8h**    | **2h**     | **3h**    | **6h**   | **8h**       | **2h**     |  **39.5h**  |

The schedule overshoots 38h by 1.5h on purpose — Week 6 has one new install (umap-learn) and one new visual habit (silhouette diagrams) and the buffer is realistic.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Free docs for scikit-learn 1.5+ cluster/decomposition/manifold/mixture, umap-learn, the t-SNE and UMAP papers |
| [lecture-notes/01-kmeans-hierarchical-dbscan-gmm.md](./lecture-notes/01-kmeans-hierarchical-dbscan-gmm.md) | The four clustering algorithms; Lloyd's algorithm in 5 lines; when to use which |
| [lecture-notes/02-pca-tsne-umap.md](./lecture-notes/02-pca-tsne-umap.md) | Dimensionality reduction, linear and nonlinear; what each is for; when to use which |
| [lecture-notes/03-when-clustering-is-not-the-answer.md](./lecture-notes/03-when-clustering-is-not-the-answer.md) | The honest critique: when the algorithm returns clusters that do not exist |
| [exercises/README.md](./exercises/README.md) | Index of exercises |
| [exercises/exercise-01-kmeans-from-scratch.py](./exercises/exercise-01-kmeans-from-scratch.py) | Implement Lloyd's algorithm in pure NumPy; verify against sklearn |
| [exercises/exercise-02-pca-and-umap.py](./exercises/exercise-02-pca-and-umap.py) | PCA on the digits dataset; UMAP as a follow-up; read the scree plot |
| [exercises/exercise-03-choose-k-honestly.py](./exercises/exercise-03-choose-k-honestly.py) | Elbow, silhouette, and gap statistic on one dataset; recognize when they disagree |
| [challenges/README.md](./challenges/README.md) | Index of weekly challenges |
| [challenges/challenge-01-justify-the-k.md](./challenges/challenge-01-justify-the-k.md) | Pick a `k`, defend it three ways, write the honest paragraph |
| [quiz.md](./quiz.md) | 10 multiple-choice questions with an answer key |
| [homework.md](./homework.md) | Six practice problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Cluster real customer/product records; justify the `k`; visualize in 2D |

---

## Stretch goals

- Read **MacQueen's "Some methods for classification and analysis of multivariate observations" (1967)** — the original k-means paper. Six pages. The algorithm has not changed.
- Read **Lloyd's "Least squares quantization in PCM" (1957, published 1982)** — the algorithm that became k-means, derived from a signal-processing problem. Twelve pages.
- Read **van der Maaten and Hinton's "Visualizing Data using t-SNE" (JMLR 2008)** — the t-SNE paper. The Kullback-Leibler-on-conditional-distributions derivation is the part worth reading; the rest is engineering.
- Read **McInnes, Healy, Melville's "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" (2018, arXiv)** — the UMAP paper. The differential-geometry framing is more rigorous than t-SNE's; the practical recipe is similar.
- Read **the scikit-learn user guide on "Clustering"** end-to-end (free, on the official site). The comparison table at the bottom — algorithm, geometry, scale, use case — is the single most useful one-page reference for this week.

---

## What you will *not* do this week

You will not:

- Train a self-supervised model (BYOL, SimCLR, masked language modeling). Those are unsupervised in a different sense — the "task" is a pretext task — and they belong with Weeks 7–9 on deep learning.
- Use a clustering result as a feature in a downstream supervised model unless you have a specific defended reason. The "cluster ID" feature is the canonical leaky feature when the clusters are fit on data that includes the test set.
- Cluster on the raw, unscaled feature matrix. K-means and DBSCAN are both Euclidean-distance-based; a feature measured in dollars and a feature measured in years will dominate or vanish based on the unit, not the information content.
- Run t-SNE with default `perplexity=30` on a dataset of 200 rows and report the result. The default is calibrated for `n` in the thousands; on small data the result is noise.
- Build a Streamlit app or a dashboard around the clusters. The mini-project's deliverable is a notebook and a 2-page report, like Weeks 4 and 5.
- Pretend that finding `k=5` clusters means there are five kinds of customer. There are five regions of the `X` space that k-means labeled with five different integers. Whether those regions correspond to anything meaningful is a separate question, and the entire point of Lecture 3.

That is deliberate. The point of Week 6 is not to cluster something for the sake of clustering it. The point is to ship a defended unsupervised analysis with a 2D visualization, a justified `k`, and an honest paragraph about whether the clusters mean anything — paired with the discipline of writing that paragraph even when the answer is "no, they probably do not."

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The cards are not graded and they are not optional. They are the difference between reading about k-means and being able to debug a silhouette plot that does not match the elbow plot. Every claim that has fooled a generation of analysts has an `EXPERIMENT` card that would have saved them an afternoon.

---

## Up next

[Week 7 — Neural Networks from Scratch](../week-07/) — once you have pushed your Week 6 clustering notebook to your portfolio repo with a defended `k` and a 2D visualization. Week 7 leaves both supervised tabular learning and unsupervised learning behind for the first deep-learning week: a 2-layer NumPy neural network on MNIST, ≥95% test accuracy, backprop derived by hand. The discipline carries over (baseline first, evaluate honestly, interpret out loud); the model changes shape entirely.
