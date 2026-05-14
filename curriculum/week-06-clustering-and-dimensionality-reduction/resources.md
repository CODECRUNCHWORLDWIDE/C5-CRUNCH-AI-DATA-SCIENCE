# Week 6 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **scikit-learn — "Clustering" user guide** (the canonical comparison table; the right place to start the week):
  <https://scikit-learn.org/stable/modules/clustering.html>
- **scikit-learn — "Decomposing signals in components"** (PCA, ICA, NMF, dictionary learning; we use the PCA section):
  <https://scikit-learn.org/stable/modules/decomposition.html>
- **scikit-learn — "Manifold learning"** (Isomap, LLE, t-SNE; sober about when each works and when it does not):
  <https://scikit-learn.org/stable/modules/manifold.html>
- **scikit-learn — "Gaussian mixture models"** (the EM derivation in two pages plus the API; read it twice):
  <https://scikit-learn.org/stable/modules/mixture.html>
- **scikit-learn — "Silhouette analysis"** (the silhouette-plot recipe; twenty minutes; the figure in this guide is the one to recreate every time you cluster):
  <https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html>
- **UMAP documentation — "Basic UMAP Parameters"** (the official walkthrough of `n_neighbors`, `min_dist`, `n_components`, `metric`):
  <https://umap-learn.readthedocs.io/en/latest/parameters.html>
- **Distill — "How to Use t-SNE Effectively" (Wattenberg, Viégas, Johnson, 2016)** (interactive; the single best reference on t-SNE pitfalls; required reading):
  <https://distill.pub/2016/misread-tsne/>

## The official docs you will bounce between all week

- **scikit-learn API reference**:
  <https://scikit-learn.org/stable/api/index.html>
- **scikit-learn `KMeans`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
- **scikit-learn `MiniBatchKMeans`** (for `n` in the millions):
  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html>
- **scikit-learn `DBSCAN`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>
- **scikit-learn `AgglomerativeClustering`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>
- **scikit-learn `GaussianMixture`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html>
- **scikit-learn `PCA`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>
- **scikit-learn `TSNE`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>
- **scikit-learn `silhouette_score` / `silhouette_samples`**:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html>
- **umap-learn API reference**:
  <https://umap-learn.readthedocs.io/en/latest/api.html>
- **scipy `dendrogram` and linkage matrix**:
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html>

## Free textbooks (the canon)

- **Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*** (free PDF from Stanford):
  <https://hastie.su.domains/ElemStatLearn/>
  Read **chapter 14** end-to-end. The clustering section (14.3) is the densest correct introduction in print; the PCA section (14.5) is the cleanest derivation; section 14.4 covers self-organizing maps which we skip but are worth knowing about.
- **James, Witten, Hastie, Tibshirani — *An Introduction to Statistical Learning*, 2nd edition** (free PDF):
  <https://www.statlearning.com/>
  Read **chapter 12** end-to-end. The k-means and hierarchical clustering treatments are gentler than ESL's and the figures are worth the read alone. The PCA section is one chapter, also gentle.
- **Christoph Molnar — *Interpretable Machine Learning*** (free online, CC-BY):
  <https://christophm.github.io/interpretable-ml-book/>
  No clustering chapter, but the **chapter on PCA-based proxy models** (in the "Global Model-Agnostic Methods" section) is worth twenty minutes for the broader point about when dimensionality reduction supports interpretability.
- **Kevin Murphy — *Probabilistic Machine Learning: An Introduction*** (free PDF, MIT Press, 2022):
  <https://probml.github.io/pml-book/book1.html>
  Chapter 21 (clustering) is the rigorous Bayesian-flavored treatment if you want it. Chapter 20 covers PCA, factor analysis, and ICA in one place; the unified probabilistic framing is the part worth reading.
- **Aurélien Géron — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd edition** (the early chapters are free as a preview from O'Reilly; the clustering chapter is not, but the [author's notebooks are public and CC-BY on GitHub](https://github.com/ageron/handson-ml3)):
  Chapter 9 (unsupervised learning) is the most practical single chapter on this week's material in print.

## The data sources we use this week

All public, all free to download:

- **scikit-learn's `load_digits`** (the 8x8 handwritten-digit dataset; 1,797 rows, 64 features; the standard PCA / t-SNE / UMAP demo):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>
- **scikit-learn's `make_blobs`** (the synthetic Gaussian-blob generator; the right tool for "show me what k-means does on data it was designed for"):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>
- **scikit-learn's `make_moons` / `make_circles`** (the synthetic non-convex shapes; the right tool for "show me where k-means fails and DBSCAN wins"):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html>
- **UCI / Kaggle "Online Retail" dataset** (the mini-project default; 540k transactions from a UK retailer, public since 2011):
  <https://archive.ics.uci.edu/dataset/352/online+retail>
- **UCI "Mall Customer Segmentation"** (a smaller, 200-row segmentation dataset that is gentler if Online Retail is overwhelming; CSV on Kaggle):
  <https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python>
- **scikit-learn's `fetch_20newsgroups_vectorized`** (a high-dimensional sparse dataset; the right tool for "show me PCA on something that needs it"):
  <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups_vectorized.html>

## The papers (free PDFs, the originals)

The unsupervised-learning field has a longer history than the boosted-trees field — the founding papers go back to the 1950s and 1960s — but the modern reference points are all free.

- **Stuart Lloyd — "Least squares quantization in PCM"** (Bell Labs technical memo, 1957; published in IEEE Trans. Inf. Theory, 1982). The original k-means algorithm, framed as a signal-quantization problem. Twelve pages.
  <https://ieeexplore.ieee.org/document/1056489> (the 1982 published version; the 1957 memo is harder to find but content-identical)
- **James MacQueen — "Some methods for classification and analysis of multivariate observations"** (Berkeley Symposium, 1967). The paper that named "k-means" and the algorithm in roughly its modern form. Six pages.
  <https://projecteuclid.org/euclid.bsmsp/1200512992>
- **Arthur and Vassilvitskii — "k-means++: The Advantages of Careful Seeding"** (SODA 2007). The initialization that fixes Lloyd's algorithm's habit of converging to bad local minima. Nine pages.
  <https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf>
- **Ester, Kriegel, Sander, Xu — "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise"** (KDD 1996). The original DBSCAN paper. Eight pages. The figure that compares DBSCAN to k-means on two-moons is the one every textbook reuses.
  <https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf>
- **Karl Pearson — "On Lines and Planes of Closest Fit to Systems of Points in Space"** (Philosophical Magazine, 1901). The original PCA paper, predating the eigendecomposition formalism by twenty-five years. Sixteen pages.
  <https://www.tandfonline.com/doi/abs/10.1080/14786440109462720> (the journal abstract; the full paper is in many archives, free)
- **Harold Hotelling — "Analysis of a complex of statistical variables into principal components"** (Journal of Educational Psychology, 1933). The paper that gave PCA its modern formulation, ten years after Pearson.
  Free PDF: <https://psycnet.apa.org/record/1934-00645-001>
- **van der Maaten and Hinton — "Visualizing Data using t-SNE"** (JMLR, 2008). The t-SNE paper. The Kullback-Leibler-on-conditional-distributions derivation is the part worth reading.
  <https://jmlr.org/papers/v9/vandermaaten08a.html>
- **McInnes, Healy, Melville — "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"** (2018, arXiv). The UMAP paper. The differential-geometry framing is more rigorous than t-SNE's; the practical recipe is similar.
  <https://arxiv.org/abs/1802.03426>
- **Tibshirani, Walther, Hastie — "Estimating the number of clusters in a data set via the gap statistic"** (J. Royal Statistical Society B, 2001). The gap statistic, the most principled "what is the right `k`?" criterion. Twelve pages.
  <https://hastie.su.domains/Papers/gap.pdf>

## The math you need this week

Less than you might expect:

- **Euclidean distance** in `R^p`. `d(x, y) = sqrt(sum_j (x_j - y_j)^2)`. K-means and DBSCAN are both Euclidean by default. The fact that "Euclidean" is the default — not "the right thing for your data" — is a surprisingly important lecture point.
- **Within-cluster sum of squares.** `WCSS = sum_c sum_{i in c} ||x_i - μ_c||²`. The k-means objective in one line. Lloyd's algorithm is coordinate descent on this objective.
- **The variance-of-a-projection identity.** For a unit vector `v` and centered data `X`, `Var(X v) = v^T S v` where `S = X^T X / (n-1)` is the sample covariance. PCA maximizes this over unit vectors; the answer is the top eigenvector of `S`.
- **The singular value decomposition** `X = U Σ V^T`. The columns of `V` are the principal directions; the singular values squared (divided by `n-1`) are the variances. `sklearn.decomposition.PCA` uses the SVD path, not an eigendecomposition, because the SVD is numerically more stable on tall-skinny `X`.
- **Bayes' rule** for the GMM E-step: `P(cluster k | x) = π_k · N(x | μ_k, Σ_k) / sum_j π_j · N(x | μ_j, Σ_j)`. Two lines; the entire EM update follows from "take the expectation of the complete-data log-likelihood under this posterior, then maximize over parameters."
- **The Kullback-Leibler divergence** `KL(p || q) = sum_x p(x) log(p(x) / q(x))`. The t-SNE objective is `KL(P || Q)` where `P` is the high-D pairwise neighborhood distribution and `Q` is the low-D one. We sketch this but do not require you to re-derive it.
- **No measure theory, no functional analysis, no convex optimization** beyond "coordinate descent converges for smooth, separable objectives." Save them for the elective.

If you want the math properly written out, ESL chapter 14 covers every formula we use this week in forty pages.

## Tools you will use this week

- **`scikit-learn`** ≥ 1.5: `pip install "scikit-learn>=1.5,<2"`.
- **`umap-learn`** ≥ 0.5: `pip install "umap-learn>=0.5,<1"`. Optional for the lectures, required for Exercise 2 and the mini-project bonus.
- **`scipy`** ≥ 1.11: `pip install "scipy>=1.11,<2"`. We use `scipy.cluster.hierarchy.linkage` and `dendrogram` for the dendrogram plots.
- **`pandas`** ≥ 2.2, **`numpy`** ≥ 2, **`matplotlib`** ≥ 3.8 (from prior weeks).
- **`pytest`** ≥ 8 for the exercises.

A `requirements.txt` snippet for the week:

```text
pandas>=2.2,<3
numpy>=2,<3
scipy>=1.11,<2
scikit-learn>=1.5,<2
umap-learn>=0.5,<1
matplotlib>=3.8,<4
seaborn>=0.13,<1
pytest>=8
```

### Installation gotchas

- **`umap-learn`** has a transitive dependency on **`numba`**, which is large (~50 MB) and slow to install on macOS arm64 the first time (it compiles LLVM wheels). The install succeeds without further setup but takes 2–5 minutes the first time.
- **`scipy.cluster.hierarchy`** is part of scipy itself; you do not need a separate install. The `dendrogram` figure can be slow on `n > 5000` — for large datasets, use `linkage(..., method="ward")` and the `truncate_mode="level"` argument to `dendrogram`.
- **`sklearn.manifold.TSNE`** is single-threaded by default; the multi-threaded backend (`method="barnes_hut"`, `n_jobs > 1`) is set automatically when `n > 1000`. For `n > 10_000`, t-SNE becomes painfully slow; switch to UMAP.
- On **macOS arm64**, `numba` (a transitive dependency of `umap-learn`) sometimes fails to find LLVM. If you see an `llvmlite` build error: `pip install --upgrade pip` and re-try; the binary wheels for Python 3.12+ are stable.

## Videos (free, no signup)

- **StatQuest with Josh Starmer** — the episodes on **k-means**, **hierarchical clustering**, **DBSCAN**, **PCA Step-by-Step**, and **t-SNE** are each ten to fifteen minutes and unusually rigorous for video. Recommended in this order.
- **Andrew Ng's machine-learning course (Coursera, free to audit)** — the Week 8 lectures cover k-means and PCA at the right level for this week. Two hours total.
- **3Blue1Brown's "Essence of Linear Algebra" series** — the **eigenvectors and eigenvalues** episode and the **change of basis** episode are the two cleanest visual explanations of what PCA is doing. Twenty-five minutes total.
- **Leland McInnes' SciPy 2018 talk on UMAP** (free on YouTube). The first ten minutes are the differential-geometry derivation in slides; the rest is the engineering. Worth thirty minutes if you want to know why UMAP works.

## Open-source projects to read (in this order)

You can learn more from one hour reading the scikit-learn cluster code than from three hours of tutorials.

1. **scikit-learn `KMeans`** — the Python wrapper is `sklearn/cluster/_kmeans.py`; the Cython-accelerated inner loops are in `_k_means_lloyd.pyx` and `_k_means_elkan.pyx`. The Python wrapper is what you read first:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py>
2. **scikit-learn `DBSCAN`** — `sklearn/cluster/_dbscan.py`; the core idea is in 200 lines:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_dbscan.py>
3. **scikit-learn `PCA`** — `sklearn/decomposition/_pca.py`; the SVD path and the eigendecomposition path are both there:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_pca.py>
4. **scikit-learn `TSNE`** — `sklearn/manifold/_t_sne.py`; the Barnes-Hut acceleration is the heart of the file:
   <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_t_sne.py>
5. **umap-learn `UMAP`** — the entry point is `umap/umap_.py`; the optimization is in `umap/layouts.py`; both are heavily commented:
   <https://github.com/lmcinnes/umap/blob/master/umap/umap_.py>

## Cheat sheets (one-page references)

- **scikit-learn — "Choosing the right estimator"** (the official flowchart; clustering and dimensionality reduction are in the left-hand branch):
  <https://scikit-learn.org/stable/machine_learning_map.html>
- **scikit-learn — "Comparison of clustering algorithms"** (the visual one-page comparison; the figure that every textbook reuses):
  <https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html>
- **The C5 "when clustering is not the answer" checklist** — Lecture 3, Section 8. One printable page; tape it to your monitor.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Unsupervised learning** | A learning task without a target `y`. The model is asked to find structure in `X` alone — clusters, low-dimensional manifolds, density estimates. |
| **K-means** | A clustering algorithm that partitions `n` rows into `k` groups by minimizing the sum of squared Euclidean distances from each row to its assigned centroid. Iterates "assign to nearest centroid, move centroid to cluster mean" until convergence. |
| **Centroid** | The mean of the rows assigned to a k-means cluster. The cluster representative; `predict(x)` returns the centroid index nearest `x`. |
| **WCSS** | Within-cluster sum of squares. `sum_c sum_{i in c} ||x_i - μ_c||²`. The k-means objective; the y-axis of an elbow plot; `KMeans.inertia_` in sklearn. |
| **Elbow plot** | WCSS as a function of `k`. Reads like an "elbow" when there is a clear `k`; reads like a smooth curve when there is not. The latter is more common on real data. |
| **k-means++** | An initialization that picks centroids with probability proportional to squared distance from already-chosen centroids. Achieves WCSS within `O(log k)` of optimal in expectation. The default since sklearn 1.0. |
| **Lloyd's algorithm** | The coordinate-descent algorithm for k-means: alternate assignment (E-step) and centroid update (M-step). Converges in finitely many steps to a local minimum. |
| **Hierarchical (agglomerative) clustering** | Start with `n` singleton clusters; merge the two closest at each step. Returns the full dendrogram. Cut at a chosen height to recover any `k`. |
| **Linkage** | The rule for "closest" between two clusters: single (min pairwise distance), complete (max), average (mean), Ward (within-cluster-variance minimizer). |
| **Dendrogram** | The tree of merges from agglomerative clustering. Reading the dendrogram is part of choosing `k`; the height of the largest gap is one heuristic for where to cut. |
| **DBSCAN** | Density-based clustering. Core points have `≥ min_samples` neighbors within `eps`; clusters are connected components of core points; non-core unreachable points are noise. No `k`; two delicate parameters. |
| **Core point / noise point** | DBSCAN labels: core has `min_samples` neighbors within `eps`; noise has no path of core points reaching it. |
| **Gaussian mixture model** | Models the data as `k` Gaussians with means, covariances, and mixing weights. Fit by EM. Gives soft (probabilistic) assignments and elliptical clusters. |
| **EM algorithm** | Alternates E-step (compute posterior cluster probabilities given current parameters) and M-step (re-estimate parameters given posteriors). Generalizes Lloyd's algorithm to GMMs. |
| **Silhouette score** | `(b − a) / max(a, b)` per row, where `a` is the mean within-cluster distance and `b` is the mean distance to the nearest *other* cluster. Averaged over rows. Range −1 to 1; > 0.5 is "strong," 0.25–0.5 is "weak," < 0.25 is "no real cluster." |
| **Gap statistic** | Compare observed WCSS at `k` to WCSS on uniformly random data of the same shape; pick the smallest `k` where the gap exceeds the random-data gap at `k+1`. The most principled choice criterion. |
| **PCA** | Principal Component Analysis. Find the orthogonal directions of maximal variance in the centered data. Implemented as the SVD of `X` or the eigendecomposition of `X^T X`. |
| **Explained variance ratio** | `eigenvalue_k / sum_j eigenvalue_j`. The fraction of total variance captured by the `k`-th principal component. The y-axis of a scree plot. |
| **Scree plot** | Explained variance (or eigenvalue) versus principal component index. The "elbow" suggests how many components to keep. |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding. A nonlinear embedding that preserves local neighborhoods at the cost of global distances. Single hyperparameter `perplexity` (effective neighborhood size; default 30). Stochastic. |
| **Perplexity** | The "effective number of neighbors" each point in t-SNE pretends to have. Smaller = tighter local clusters; larger = smoother embedding. Rule of thumb: 5–50, with 30 the default. |
| **UMAP** | Uniform Manifold Approximation and Projection. A nonlinear embedding with a differential-geometry foundation. Faster than t-SNE on large data; often preserves global structure better; hyperparameters `n_neighbors` and `min_dist`. |
| **Curse of dimensionality** | In high dimensions, all pairwise distances become similar (the ratio of max to min nearest-neighbor distance approaches 1). Euclidean clustering becomes useless past `p ≈ 50`; use PCA, UMAP, or domain-specific features first. |
| **Standardization** | Subtract the mean, divide by the standard deviation, per feature. The right preprocessing for Euclidean clustering when feature scales differ. `sklearn.preprocessing.StandardScaler`. |

---

*If a link 404s, please open an issue so we can replace it.*
