# Lecture 2 — PCA, t-SNE, and UMAP

> **Outcome:** You can derive PCA from "find the direction of maximal variance," implement it in five lines of NumPy with the SVD, and read a scree plot. You can describe what t-SNE and UMAP each optimize and why the resulting 2D scatter is *only* a visualization — never a feature for a downstream model unless you have a specific reason. You can pick between the three with a one-paragraph justification and avoid the four standard mistakes of over-interpreting a 2D embedding.

Clustering (Lecture 1) returns labels. Dimensionality reduction returns *coordinates*. You start with a feature matrix `X` of shape `(n, p)` where `p` is too large to plot or too high to cluster meaningfully, and you produce a new matrix `Z` of shape `(n, d)` where `d` is small — usually 2 (for visualization) or `10–50` (for downstream learning). The transformation `X → Z` is the job of a dimensionality-reduction algorithm.

There are three you actually use in 2026:

1. **PCA** — linear; fast; principled; the right default. Captures the directions of maximal variance.
2. **t-SNE** — nonlinear; slow; stochastic; for visualization only. Preserves local neighborhoods.
3. **UMAP** — nonlinear; fast on large data; more globally faithful than t-SNE; the modern default for 2D plots of high-dimensional data.

We target **scikit-learn 1.5+** (the `decomposition` and `manifold` modules) and **umap-learn 0.5+** (the standalone UMAP library).

---

## 1. What PCA actually computes

PCA asks: among all unit-length directions `v` in `R^p`, which one captures the most variance of the data?

Formally, with `X` of shape `(n, p)` centered (each column's mean is zero), the variance of the projection `X v` is:

```text
Var(X v)  =  (1 / (n-1))  ·  || X v ||²
         =  v^T  ·  S  ·  v
```

where `S = X^T X / (n-1)` is the sample covariance matrix. PCA maximizes this over unit vectors `v`. By the Rayleigh quotient theorem, the maximizer is the eigenvector of `S` with the largest eigenvalue — and the value of the maximum *is* that eigenvalue.

The first principal component (PC1) is that top eigenvector. The second PC is the top eigenvector of `S` restricted to directions orthogonal to PC1 (it is the eigenvector with the *second*-largest eigenvalue). And so on. The `p` principal components form an orthonormal basis of `R^p` ordered by the variance captured along each direction.

The **eigenvalues themselves** are the variances. They satisfy:

```text
sum_j λ_j  =  total variance of X  =  trace(S)
```

so each eigenvalue's fraction of the total is the **explained variance ratio** of that principal component.

---

## 2. The SVD path (what sklearn actually does)

Computing `S = X^T X / (n-1)` and then its eigendecomposition is numerically unstable when `X` is tall and skinny (`n >> p`). The robust path is the **singular value decomposition** of `X` directly:

```text
X  =  U  Σ  V^T
```

where `U` is `(n, n)` orthogonal, `Σ` is `(n, p)` diagonal with non-negative entries `σ_1 ≥ σ_2 ≥ ...`, and `V` is `(p, p)` orthogonal. The columns of `V` are the principal directions; the singular values squared, divided by `n-1`, are the eigenvalues of `S`:

```text
λ_j  =  σ_j²  /  (n - 1)
```

The projection of `X` onto the first `k` principal components is:

```text
Z  =  X · V[:, :k]  =  U · Σ[:, :k]
```

`sklearn.decomposition.PCA` uses the SVD path by default. The five-line implementation:

```python
import numpy as np

def pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via SVD. Returns (Z, components, explained_variance_ratio)."""
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
    Z = U[:, :k] * sigma[:k]
    components = Vt[:k]
    var = (sigma ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = var[:k] / var.sum()
    return Z, components, explained_variance_ratio
```

That is the entire algorithm. Five lines plus the centering and the variance ratios. The numpy `svd` does the heavy lifting; on a `(1000, 64)` matrix it returns in milliseconds.

> **EXPERIMENT — PCA on the digits dataset.** Load `sklearn.datasets.load_digits()` (1797 rows, 64 features — 8×8 pixel images flattened). Fit `PCA(n_components=10)` and inspect `pca.explained_variance_ratio_`. The first two components capture ~28%; the first ten capture ~60%; the first 30 capture ~90%. The scree plot has a gentle slope rather than a sharp elbow — typical of natural-image data, where many dimensions carry signal.

---

## 3. The scree plot and the choice of `k`

The **scree plot** shows the explained variance ratio (or cumulative variance) versus the principal-component index:

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

mdl = PCA(n_components=20).fit(X)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax1.plot(np.arange(1, 21), mdl.explained_variance_ratio_, "o-")
ax1.set_xlabel("principal component")
ax1.set_ylabel("variance ratio (per component)")
ax2.plot(np.arange(1, 21), np.cumsum(mdl.explained_variance_ratio_), "o-")
ax2.set_xlabel("principal component")
ax2.set_ylabel("cumulative variance ratio")
ax2.axhline(0.9, ls="--", color="#9CA3AF")
```

Three heuristics for picking `k`:

1. **The elbow.** The component index where the per-component variance ratio's slope flattens. Often subjective.
2. **A variance threshold.** "Keep the smallest `k` such that the cumulative variance ratio is ≥ 0.90 (or 0.95, or 0.99)." Easy to defend; the choice of threshold is itself an assumption.
3. **Cross-validation in a downstream task.** If PCA is preprocessing for a supervised model, sweep `k` and pick the value that maximizes CV score. The most rigorous; the most expensive.

For *visualization*, `k = 2` is forced (one PC per axis). For *downstream learning*, the right `k` is often "the smallest that doesn't hurt CV score" — which is usually substantially smaller than the dimension at which the cumulative variance reaches 0.99.

---

## 4. When PCA is the right tool

PCA is the right tool when:

- **The features are correlated.** PCA's whole job is to find a rotated basis in which the new features are uncorrelated. If your 50 features include 30 that are linear combinations of the other 20, PCA reveals the 20 latent directions.
- **You need to visualize linear structure.** A 2D PCA scatter of a dataset where the variance is concentrated in a few directions is informative. (A 2D PCA scatter of MNIST is also useful, but it captures only ~17% of the variance — the global structure rather than the digit-by-digit clusters.)
- **You need to preprocess for downstream learning.** A `Pipeline([("pca", PCA(n_components=30)), ("model", Ridge())])` is a defensible recipe for high-dimensional regression. The PCA is a regularizer.
- **You need to denoise.** Keeping the top `k` components and reconstructing `X ≈ Z · V[:k]^T` projects out the small-variance directions, which often correspond to noise. This is the trick behind many image-denoising and missing-data-imputation recipes.

PCA is the **wrong** tool when:

- **The structure is nonlinear.** PCA finds linear directions only. A dataset shaped like a Swiss roll has all its interesting structure in a nonlinear manifold; PCA flattens it badly. Use Isomap, t-SNE, or UMAP.
- **The interesting signal is in low-variance directions.** PCA assumes "high variance = signal." Sometimes the high-variance directions are uninteresting (overall brightness in images; overall vocabulary frequency in text) and the *low*-variance directions carry the signal. This is when supervised methods like LDA (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`) win.
- **The data is already low-dimensional.** PCA on `p = 4` features is pointless; you cannot reduce below 4 without losing information, and 4 features can be inspected directly.
- **The features are not on comparable scales.** PCA on unscaled features is dominated by the highest-variance feature (which is usually the one with the largest unit, not the most information). Always `StandardScaler` first.

> **EXPERIMENT — scale matters for PCA.** Load `fetch_california_housing(as_frame=True)`. Without scaling, fit `PCA(n_components=2)` and inspect `pca.components_`. The first PC is dominated by `Population` (range 0–35,000) and `AveOccup`. Now apply `StandardScaler` before PCA and re-fit. The first PC is now a mix of all 8 features, weighted by their information content rather than their unit. The scaled PCA is the only one that makes sense.

---

## 5. PCA's two failure modes you must know

**Failure mode 1: nonlinear manifolds.** Generate the Swiss roll: `from sklearn.datasets import make_swiss_roll; X, t = make_swiss_roll(n_samples=2000, random_state=42)`. The data sits on a 2D manifold (the "unrolled" sheet) embedded in 3D. PCA projects it to 2D and you get a flattened pancake — the rolling structure is lost. Isomap, t-SNE, and UMAP each find the unrolled sheet.

**Failure mode 2: high variance is not signal.** Generate two classes where class 1 is at `x = (0, 0)` plus tiny noise and class 2 is at `x = (1, 0)` plus tiny noise, with both classes having huge noise in the second dimension. PCA's PC1 will be the second dimension (high variance, no class signal); the *class separation* is in PC2 (low variance, all the signal). This is the textbook case for LDA over PCA.

---

## 6. t-SNE: preserving local neighborhoods

PCA is linear; many real datasets are not. **t-SNE** (t-distributed Stochastic Neighbor Embedding; van der Maaten and Hinton, 2008) is a nonlinear embedding that tries to preserve the local neighborhood structure of the high-dimensional data when projecting to (usually) 2D.

The intuition:
1. In the high-dimensional space, define a conditional probability `P(j | i)` that "point `i` would pick point `j` as a neighbor" using a Gaussian kernel centered at `i`. The kernel width is set per-point so that each point has the same **effective number of neighbors**, called the **perplexity**.
2. In the low-dimensional embedding `Y`, define an analogous conditional probability `Q(j | i)` using a *Student t* distribution with one degree of freedom (the t-distribution has heavier tails than the Gaussian, which avoids "crowding" in low dimensions).
3. Move the low-dimensional points `y_i` to minimize the Kullback-Leibler divergence `KL(P || Q)`. The optimization is gradient descent; it is stochastic and non-convex.

The single hyperparameter that matters is **`perplexity`** (default 30 in sklearn). Loosely, perplexity is the "size of the local neighborhood" t-SNE is preserving:

- **Small perplexity (5–15)** — tight local structure; many small clusters.
- **Default perplexity (30)** — balanced; usually works.
- **Large perplexity (50–100)** — more global structure visible; the embedding looks smoother but local detail is lost.

```python
from sklearn.manifold import TSNE

embedding = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",            # better than the random init for reproducibility
    learning_rate="auto",  # the modern default in sklearn 1.2+
).fit_transform(X)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=4)
```

---

## 7. The four mistakes of over-interpreting t-SNE

The Distill 2016 article "How to Use t-SNE Effectively" (required reading; see `resources.md`) catalogs the standard mistakes. The four you must internalize:

1. **Cluster sizes in the t-SNE plot do not match cluster densities in `X`.** A large blob in the embedding does not mean a sparse cluster in the original space; the algorithm normalizes the per-point neighborhoods, which expands sparse regions and contracts dense ones. Do not read "cluster A is larger than cluster B" off a t-SNE plot.
2. **Inter-cluster distances in the t-SNE plot do not match distances in `X`.** Two clusters that appear far apart in the t-SNE plot may be neighbors in the original space; two that overlap in the plot may be miles apart. Only the *local* structure is preserved.
3. **The number of visible clusters depends on `perplexity`.** Run t-SNE with `perplexity=5`, then `30`, then `100` on the same data and you will see three different cluster counts. None of them is "the right one." Run multiple perplexities and report all.
4. **Random initializations give different embeddings.** t-SNE is stochastic. Two runs with different `random_state` produce different 2D coordinates, sometimes dramatically. Use `init="pca"` for reproducibility (the PCA-initialized embedding is much more stable across reseeding).

The most dangerous failure mode is **t-SNE on random noise**. Run t-SNE on a `(1000, 50)` matrix of pure standard-normal noise — *no signal whatsoever* — and you will see what looks like clusters in the 2D embedding. The "clusters" are the algorithm's response to local crowding in 50-dimensional Gaussian noise; they are not a property of the data. The lesson: t-SNE always produces a picture that *looks* clustered; only a real signal makes those clusters meaningful.

> **EXPERIMENT — t-SNE on random noise.** `X = np.random.default_rng(42).standard_normal((1000, 50))`. Run `TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(X)` and plot. The embedding is not uniformly scattered; it has visible structure that looks like clusters. There are no clusters in `X`. The "clusters" are an artifact of the algorithm. This is the single most important t-SNE experiment to have run yourself.

---

## 8. UMAP: faster, more global, more parameters

**UMAP** (Uniform Manifold Approximation and Projection; McInnes, Healy, Melville, 2018) is a more recent nonlinear embedding that addresses several of t-SNE's weaknesses:

1. **Faster on large data.** UMAP scales to millions of rows where t-SNE becomes painfully slow past 50,000.
2. **More globally faithful.** UMAP preserves more of the global structure of the high-dimensional data than t-SNE does. Inter-cluster distances in UMAP are still not perfectly meaningful, but they are *less* meaningless than in t-SNE.
3. **Better-behaved on small datasets.** UMAP's local-vs-global tradeoff is controlled by `n_neighbors`, which scales gracefully with dataset size where t-SNE's `perplexity` does not.

The two main hyperparameters:

- **`n_neighbors`** (default 15) — the local-neighborhood size; analogous to t-SNE's `perplexity`. Small values emphasize local structure; large values emphasize global structure. Range 5–100.
- **`min_dist`** (default 0.1) — the minimum distance between points in the embedding. Small values pack points tightly within clusters; large values spread them out. Range 0.0–0.99.

```python
import umap

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
)
embedding = reducer.fit_transform(X)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=4)
```

UMAP shares **the same four interpretive mistakes** as t-SNE: cluster sizes are not meaningful, inter-cluster distances are not meaningful, the result depends on hyperparameters, and the result depends on the random seed. The mistakes are slightly less severe (UMAP preserves more global structure, and is less stochastic in practice) but they are present.

UMAP has one extra feature t-SNE lacks: **`transform`**. After fitting UMAP on a training set, you can embed *new* data into the same 2D coordinate system. `sklearn.manifold.TSNE` does not have `transform`; you have to re-fit on the augmented data. This makes UMAP usable as preprocessing in a `Pipeline`; t-SNE cannot be.

> **EXPERIMENT — UMAP vs t-SNE on digits.** Load `load_digits()`. Fit `PCA(n_components=2)`, `TSNE(n_components=2, init="pca")`, and `UMAP(n_components=2)`. Plot all three side by side, colored by digit label (0–9). The PCA scatter shows blobs that mostly overlap; the t-SNE scatter shows 10 mostly-separated clusters with arbitrary inter-cluster arrangement; the UMAP scatter shows 10 mostly-separated clusters with a more legible global arrangement (4 and 9 cluster near each other because they look similar; 0 sits apart from the rest because it does not). The three plots tell complementary stories.

---

## 9. PCA vs t-SNE vs UMAP: a one-paragraph decision rule

- **Use PCA** when: the structure is linear; you want a transformation that generalizes to new data (`transform` works); you need a defensible preprocessing step for downstream learning; you need a fast result; you want to report explained-variance ratios. PCA is the default; the burden of proof is on the nonlinear methods.
- **Use t-SNE** when: you want a *visualization* of nonlinear structure on a dataset of `n < 50,000`; you are willing to run multiple perplexities and accept that the result is stochastic; you do not need to apply the transform to new data. Use `init="pca"` for reproducibility.
- **Use UMAP** when: you want the same visualization t-SNE gives but on larger data, faster, and with somewhat better-preserved global structure; you need a `transform` method for new data (though using UMAP coordinates as features for a downstream model is still suspect — see Section 11). Use `random_state` for reproducibility.

For the mini-project, the recipe is usually: **PCA(2) for the first look; UMAP(2) for the headline visualization; t-SNE(2) only if you want to confirm a UMAP result with a second method**.

---

## 10. The four-line "see-all-three" recipe

For any high-dimensional clustering result, the visual sanity check is to plot the cluster assignment in three reductions side by side. Four lines plus a few `plt.subplot` calls:

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

X_pca = PCA(n_components=2).fit_transform(X)
X_tsne = TSNE(n_components=2, init="pca", random_state=42).fit_transform(X)
X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
```

Plot the three side by side, colored by the cluster labels you produced in Lecture 1. If the clusters are well-separated in all three, the clustering is robust. If they are well-separated in t-SNE / UMAP but not in PCA, the structure is nonlinear and PCA was a misleading preprocessing choice. If they are well-separated only in t-SNE and disappear under reseeding, the clusters may be a t-SNE artifact.

The three-panel "PCA, t-SNE, UMAP" figure is the single most useful visualization for an unsupervised analysis. Every Week 6 mini-project includes one.

---

## 11. Never use t-SNE / UMAP coordinates as features (almost never)

A tempting recipe: fit UMAP on the training set, get 2D coordinates, use those coordinates as features in a downstream supervised model. *Do not do this without a specific reason.*

Three problems:

1. **The embedding is non-invertible and lossy.** You have thrown away `p − 2` dimensions of information. A downstream model has only the embedding's local-neighborhood structure to work with, which is rarely enough.
2. **The embedding does not generalize cleanly to new data.** UMAP's `transform` works but is approximate; t-SNE has no `transform` at all. The coordinates of a new row depend on the training-set neighborhood, which means the embedding is not a fixed feature transformation.
3. **The embedding optimizes for visualization, not prediction.** PCA's "preserve maximal variance" objective is a defensible regularizer for downstream models. t-SNE's "preserve local neighborhoods in 2D for plotting" is not.

The right tool for "dimensionality reduction as preprocessing for supervised learning" is PCA (linear) or **kernel PCA** / **autoencoder** (nonlinear; Weeks 7–9 will revisit). t-SNE and UMAP are for plots.

The exception: when the *visualization* itself is the deliverable. A 2D map of customer segments used to drive a marketing campaign — where the map is the artifact stakeholders are looking at — is a legitimate use of UMAP as the final output, with cluster labels overlaid.

---

## 12. The dimensionality-reduction checklist

Before you ship a 2D embedding, walk this list:

- [ ] **`X` was standardized.** `StandardScaler` before PCA; the input features should also be standardized for t-SNE and UMAP, though they are somewhat less sensitive.
- [ ] **The dimensionality before reduction was reported.** "We embedded a 50-dimensional feature space into 2D."
- [ ] **PCA's explained variance was reported.** "PC1 + PC2 capture 47% of the total variance."
- [ ] **For t-SNE / UMAP, the embedding was run with at least two `random_state` values** and the results compared. If they disagree dramatically, the embedding is not robust.
- [ ] **For t-SNE, at least two `perplexity` values were tried** (e.g., 10 and 30). The structure should look similar at the two perplexities; if not, report both.
- [ ] **The plot has axis labels and a legend.** A scatter without "PC1 (28% variance)" on the x-axis is unfinished.
- [ ] **The colors correspond to something** — cluster labels (Lecture 1), a known categorical, or a target variable. A monochrome 2D scatter is rarely useful.
- [ ] **The honest paragraph is in the notebook.** "The embedding shows three visible clusters in UMAP; the same clusters are visible (though more diffuse) in PCA and in t-SNE; this triangulation supports the conclusion that `k=3` reflects real structure."

---

## 13. Where this leaves you

You can now derive PCA from "find the direction of maximal variance," implement it in five lines via the SVD, and read a scree plot. You can name PCA's two failure modes (nonlinear manifolds, low-variance signal) and pick a nonlinear alternative when appropriate. You know that t-SNE preserves local neighborhoods at the cost of global distances; that the cluster sizes and inter-cluster distances in a t-SNE plot are not meaningful; that t-SNE on random noise still produces a picture that looks clustered. You know UMAP is faster, more globally faithful, and has a usable `transform` — and you know the four interpretive mistakes still apply.

The mini-project at the end of the week takes a real customer or product dataset, finds clusters with one of the algorithms from Lecture 1, and visualizes the result with one (or more) of the embeddings from this lecture. The **three-panel "PCA, t-SNE, UMAP" figure** is the headline visualization. The **honest paragraph** about whether the clusters are real is the headline writing.

Lecture 3 is the honest critique: when the clusters are not real, what does that look like, and how do you say so in writing?
