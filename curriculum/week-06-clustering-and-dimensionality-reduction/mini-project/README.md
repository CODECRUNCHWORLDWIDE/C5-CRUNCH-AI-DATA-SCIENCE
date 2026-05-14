# Mini-Project — Cluster Real Customer (or Product) Data

> Cluster a real-world customer or product dataset. Justify the number of clusters with three independent diagnostics. Visualize the result in 2D. Write one honest paragraph about whether the clusters are real, or whether the data does not cluster at this set of features. The deliverable is a notebook plus a two-page report that an engineering or marketing manager could read in five minutes and conclude "this person knows when clusters mean something and when they don't."

This is the sixth artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 4 was the linear baseline; Week 5 was the boosted-trees lift; this week is the unsupervised counterpart. Recruiters with taste read Week 6 as the moment a junior data scientist either learns when *not* to cluster or doubles down on producing clusters that do not exist. The artifact you commit this week is the answer.

**Estimated time:** 8 hours, spread across Thursday–Sunday.

---

## What you will build

A Jupyter notebook `segmentation.ipynb` plus a rendered `report.md` that:

1. **Loads** a real customer or product dataset (one of the suggestions below, or your own — see "the data" section).
2. **Cleans and standardizes** the features. Numeric features get `StandardScaler`; categorical features get one-hot or are dropped from the clustering (clustering on Euclidean distance plus one-hot categoricals is fragile; the C5 default is to use the numeric features).
3. **Computes the three `k`-selection diagnostics** for `k ∈ [2..10]`:
   - The WCSS elbow plot.
   - The silhouette score curve.
   - The gap statistic.
4. **Picks `k`** with a one-paragraph defense.
5. **Fits four clustering algorithms** at the chosen `k` (or its natural analogue for DBSCAN, which has no `k`):
   - `KMeans`
   - `AgglomerativeClustering(linkage="ward")`
   - `DBSCAN` (with `eps` chosen from the k-distance plot)
   - `GaussianMixture(covariance_type="full")`
6. **Computes the silhouette score** for each. Compares the four partitions with pairwise `adjusted_rand_score`.
7. **Visualizes** the chosen clustering in 2D via **PCA(2), t-SNE(2), and UMAP(2)** — the three-panel figure from Lecture 2 Section 10.
8. **Profiles** each cluster: for each cluster, the row count and the mean of each numeric feature. The table that answers "what kind of customer is in cluster 3?"
9. **Reports** in a 600–900 word executive summary that a non-technical reader can finish in five minutes.

The notebook is the working artifact. The `report.md` is the executive summary.

---

## The dataset

Three default options, ordered by difficulty:

### Option A — Mall Customer Segmentation (~200 rows; gentle)

Kaggle's "Mall Customer Segmentation" dataset is the smallest and friendliest: 200 rows, 4 features (Age, Annual Income, Spending Score, Gender). The classic result is `k=5` with five clusters that loosely correspond to {high-income high-spend, high-income low-spend, low-income high-spend, low-income low-spend, middle-income middle-spend}. All three diagnostics agree, the silhouette score is around 0.55, and the UMAP scatter shows five visible blobs. A clean Week 6 win.

Download: <https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python> (free, CSV, no signup required).

### Option B — UCI Online Retail (~540k transactions; recommended default)

The "Online Retail" dataset from the UCI Machine Learning Repository (540k transactions from a UK retailer, 2010–2011). The standard workflow:

1. Aggregate per customer to compute **RFM features** (Recency, Frequency, Monetary value).
2. Log-transform Frequency and Monetary (right-skewed).
3. Standardize.
4. Cluster.

The result is typically four clusters loosely corresponding to {churned, occasional, regular, VIP}. The diagnostics may agree or disagree; this is the dataset where the "is `k` real?" question gets interesting.

Download: <https://archive.ics.uci.edu/dataset/352/online+retail>

### Option C — your own dataset (must be ≥500 rows and ≥4 numeric features)

If you have access to a real dataset from work, a public Kaggle dataset, or a domain you care about, use it. The acceptance criteria below are the same. Provide a data card paragraph in `report.md` explaining the source and the licensing.

The mini-project rubric does not penalize you for picking option A over option B; the rubric *does* penalize you for picking a tiny dataset (under 200 rows) where the diagnostics become noisy.

---

## Acceptance criteria

- [ ] A new directory `week-06/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `pandas>=2.2,<3`, `numpy>=2,<3`, `scipy>=1.11,<2`, `scikit-learn>=1.5,<2`, `umap-learn>=0.5,<1`, `matplotlib>=3.8,<4`, `jupyter`.
- [ ] `jupyter nbconvert --to notebook --execute week-06/segmentation.ipynb` runs end-to-end without errors on a fresh clone.
- [ ] The notebook **standardizes** all numeric features with `StandardScaler` before any clustering. No clustering on unscaled features.
- [ ] The notebook computes the **three `k`-selection diagnostics** (WCSS elbow, silhouette curve, gap statistic) and saves them as a single 3-panel figure `week-06/images/k_selection.png`.
- [ ] The chosen `k` is **justified in prose** referencing all three diagnostics. If they disagree, the disagreement is named.
- [ ] **Four algorithms are fit** at the chosen `k`: KMeans, AgglomerativeClustering (Ward), DBSCAN (with `eps` chosen from the k-distance plot — also saved as `week-06/images/k_distance.png`), GaussianMixture.
- [ ] A **silhouette diagram** is saved for the chosen-algorithm-and-`k` as `week-06/images/silhouette_diagram.png`.
- [ ] The **three-panel "PCA / t-SNE / UMAP" figure** is saved as `week-06/images/embeddings.png`, colored by the chosen clustering's labels.
- [ ] A **cluster-profile table** is in the notebook: one row per cluster, columns for cluster size and the mean of each numeric feature. The table answers "who is in each cluster?"
- [ ] The **column-permutation reshuffle baseline** is computed: `real_silhouette / perm_silhouette`. The ratio is reported in the notebook.
- [ ] A `report.md` (~2 pages, 600–900 words) summarizes the problem, the data, the chosen `k`, the algorithm comparison, the cluster profiles, and **the honest paragraph** from Lecture 3 Section 9 (either "yes, the clusters are real, here's how we know" or "no, the data does not cluster; we recommend quantile segmentation instead"). No code.
- [ ] A `README.md` in `week-06/` explains: setup, data download, file layout, how to reproduce.

---

## Suggested layout

```text
crunch-ai-portfolio-<yourhandle>/
├── README.md                    ← portfolio root
├── week-04/                     ← from Week 4
├── week-05/                     ← from Week 5
└── week-06/
    ├── README.md                ← week-6 explainer with chosen k summary
    ├── requirements.txt
    ├── data/
    │   └── customers.csv        ← the dataset (gitignore if large)
    ├── images/
    │   ├── k_selection.png      ← elbow + silhouette + gap, 3 panels
    │   ├── k_distance.png       ← DBSCAN eps selection
    │   ├── silhouette_diagram.png
    │   ├── embeddings.png       ← PCA + t-SNE + UMAP, 3 panels
    │   └── cluster_profile.png  ← bar chart of feature means per cluster
    ├── segmentation.ipynb
    ├── segmentation.html        ← rendered preview
    └── report.md                ← 2-page executive summary
```

---

## Suggested order of operations

### Phase 1 — Load and standardize (30 min)

1. Open `segmentation.ipynb`. The first markdown cell is the **project header** plus a one-line citation of the dataset.
2. Load the data. Inspect with `df.describe()` and `df.dtypes`.
3. Drop or impute missing values. Select the numeric features for clustering.
4. Standardize with `StandardScaler`. The standardized matrix `X` is what every subsequent step uses.

### Phase 2 — Pick `k` honestly (1.5 hours)

1. Compute the **WCSS curve** for `k ∈ [2..10]`. Plot the elbow.
2. Compute the **silhouette score** for the same range.
3. Compute the **gap statistic** for the same range (`B = 10` reference samples).
4. Combine into a 3-panel figure `images/k_selection.png`.
5. Pick `k`. Write the one-paragraph defense in the notebook. If the three diagnostics agree, the defense is short; if they disagree, the defense explains the disagreement.

### Phase 3 — Four algorithms (1.5 hours)

1. Fit `KMeans(n_clusters=chosen_k, n_init=10, random_state=42)`.
2. Fit `AgglomerativeClustering(n_clusters=chosen_k, linkage="ward")`.
3. Pick `eps` for DBSCAN from the k-distance plot (`images/k_distance.png`); fit `DBSCAN(eps=eps_value, min_samples=5)`. Note the number of clusters DBSCAN returns — it may differ from your chosen `k`.
4. Fit `GaussianMixture(n_components=chosen_k, covariance_type="full", random_state=42)`.
5. Compute the silhouette score for each algorithm.
6. Compute pairwise `adjusted_rand_score` between the four label vectors.

### Phase 4 — Visualize in 2D (1.5 hours)

1. Fit `PCA(n_components=2)` on the standardized `X`. Note the explained variance ratio of the first two components.
2. Fit `TSNE(n_components=2, init="pca", random_state=42)`.
3. Fit `umap.UMAP(n_components=2, random_state=42)`.
4. Plot all three side by side, colored by the chosen clustering's labels. Save as `images/embeddings.png`.
5. Save the silhouette diagram for the chosen clustering as `images/silhouette_diagram.png`.

### Phase 5 — Profile the clusters (1 hour)

1. For each cluster, compute the row count and the mean of each numeric feature.
2. Create a "cluster profile" table in the notebook.
3. Write one sentence per cluster: *"Cluster 0 (n=312): customers with median age 41, median spending score 28, low frequency. Likely the 'occasional shopper' segment."*
4. Save a bar chart of feature means by cluster as `images/cluster_profile.png`.

### Phase 6 — Diagnose and write (1.5 hours)

1. Compute the **reshuffle baseline**: `real_silhouette / perm_silhouette`. Record.
2. Write the **honest paragraph** in `report.md`. Use Template A from Lecture 3 Section 9 if your clusters are robust, Template B if they are not. The template is the structure; fill it in with your numbers and your cluster descriptions.
3. The rest of `report.md`: **Problem**, **Data**, **Method**, **Result** (with the chosen `k`, the algorithm comparison, the cluster profiles), **Honest paragraph**, **Limitations**, **What we'd recommend**. 600–900 words, no code.

### Phase 7 — Export (30 min)

```bash
jupyter nbconvert --to html week-06/segmentation.ipynb
```

Push to the portfolio repo.

---

## What "great" looks like

A "great" mini-project hits all of the following:

- The **chosen `k` is defended with all three diagnostics**, including the case where they disagree (which is common). Generic: "we chose `k=4` because the elbow was clearest there." Specific: "the elbow plot bends at `k=4` but is also reasonable at `k=3`; the silhouette score peaks at `k=4` (0.51 vs 0.46 at `k=3`); the gap statistic is positive and increasing through `k=4` and flat thereafter. We chose `k=4` on the silhouette and gap-statistic evidence."
- The **four algorithms agree** at the chosen `k` (pairwise ARI > 0.75 across the four). If they disagree (ARI < 0.5 between, say, KMeans and DBSCAN), the disagreement is *named* in the report. "DBSCAN at `eps=0.4` returned 3 clusters and 47 noise points, suggesting the data has 3 well-defined cores plus a tail; KMeans, AgglomerativeClustering, and GaussianMixture at `k=4` agree on a 4-cluster partition (pairwise ARI 0.91)."
- The **three 2D embeddings show separation**, and the report says so. "The four clusters are visually separated in UMAP and t-SNE; PCA(2) shows the clusters mostly overlap because the first two principal components capture only 38% of the variance." This is the kind of sentence that earns the report a second read.
- The **cluster profiles read as English**. Each cluster gets one sentence describing its members in domain terms ("frequent low-basket buyers", "infrequent high-basket loners", etc.), not generic ML terms.
- The **honest paragraph is in the report**, not just the notebook. A non-technical reader looking only at `report.md` should learn whether the clusters are real and what to do with them.
- The **reshuffle ratio is in the report**. "The silhouette on column-permuted data is 0.19; the real-vs-permuted ratio is 2.7, indicating the clustering captures joint structure rather than marginal-distribution artifacts."

A "good but not great" project picks `k` with the elbow plot alone, runs only KMeans, and produces a 2D plot without the silhouette diagram.

A "needs work" project clusters without standardizing, picks `k` from a single diagnostic, ships only the KMeans partition with no algorithm comparison, or — most commonly — fails to write the honest paragraph and presents weak clusters (silhouette 0.21) as strong findings.

---

## Stretch goals

- **Try the column-permutation baseline at every `k`** in your range, not just at the chosen `k`. The real-vs-permuted silhouette curves should diverge at the right `k`; if they do not diverge, the data is not clustered at any `k`.
- **Add a hierarchical dendrogram.** Plot the full dendrogram (`scipy.cluster.hierarchy.dendrogram` with `truncate_mode="level"`) and annotate the cuts at `k=2`, `k=3`, `k=4`. The cut heights tell you which `k` corresponds to large merge-distance gaps and which is mid-merge.
- **Compute the BIC for GMM at several `k`** with `gmm.bic(X)`. The BIC penalizes model complexity; it is the GMM analogue of the gap statistic. Plot BIC vs `k`. Does it pick the same `k` as silhouette and gap?
- **Run the quantile-segmentation baseline.** On the same features, compute the 33rd and 67th percentiles per feature and build the `3^p` rule-based segments. Report the segment-size distribution. Which segmentation — algorithmic or quantile — would you recommend to a stakeholder for an actual marketing campaign?
- **Profile clusters by an external variable.** If your dataset has a variable you did *not* cluster on (e.g., for Online Retail, the country of the customer), compute the distribution of that variable per cluster. A 4-cluster RFM segmentation that also segments by country (without being told to) is the strongest evidence of meaningful structure.

---

## Rubric

| Criterion | Weight | "Great" looks like |
|-----------|------:|--------------------|
| It runs | 10% | Fresh clone → `pip install -r requirements.txt` → `jupyter nbconvert --execute` → no errors |
| Standardization + preprocessing | 10% | All numeric features standardized; categorical handling is explicit; missing values are addressed |
| `k`-selection: three diagnostics | 20% | Elbow, silhouette, gap statistic all computed and plotted; chosen `k` is justified in prose |
| Four algorithms compared | 15% | KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture all fit; silhouette and pairwise ARI reported |
| 2D embeddings | 15% | PCA + t-SNE + UMAP side by side; the three-panel figure is in the report |
| Cluster profiles | 10% | One sentence per cluster in domain terms; bar chart of feature means by cluster |
| Honest paragraph | 15% | Template A or B from Lecture 3 Section 9; specific numbers; honest about ambiguity |
| Report readability | 5% | The 2-page `report.md` could go to a hiring manager without explanation |

---

## What this exercises (and what comes next)

This mini-project exercises every concept from Week 6:

- K-means and the three other clustering algorithms (Lecture 1, all sections).
- The four ways k-means fails and how to recognize them (Lecture 1, Section 6).
- PCA, t-SNE, and UMAP for visualization (Lecture 2, all sections).
- The "is `k` real?" five-step protocol (Lecture 3, Section 7).
- The honest paragraph, in either Template A or Template B form (Lecture 3, Section 9).
- The C5 discipline of writing what the diagnostics actually say, not what you wish they said.

Week 7 leaves both supervised tabular learning and unsupervised analysis for the first deep-learning week: a 2-layer neural network in pure NumPy on MNIST, ≥95% test accuracy, with backprop derived by hand. The discipline carries over — baseline first, evaluate honestly, write the honest paragraph — but the model takes a new shape.

---

## Submission

Push your `crunch-ai-portfolio-<yourhandle>` repo. Open the `week-06/segmentation.html` link from the repo README so a reviewer with no Python install can read your work. Paste the summary in the commit message:

```text
Week 6 mini-project: Online Retail RFM segmentation.

dataset:                UCI Online Retail (540k transactions → 4338 customers)
features clustered on:  log(frequency), log(monetary), recency_days
chosen k:               4
defense:                elbow at k=4; silhouette 0.51 (peak); gap stat positive through k=4
algorithm agreement:    pairwise ARI 0.89 across KMeans / Ward / GMM; DBSCAN found 3 clusters + 312 noise
reshuffle ratio:        2.7 (real 0.51, permuted 0.19) -- clusters are real

cluster profiles:
  0  churned (n=1834)      -- recency > 200 days, frequency = 1-2
  1  occasional (n=1521)   -- recency 60-150, frequency 3-8
  2  regular (n=748)       -- recency < 60, frequency 10-30
  3  VIP (n=235)           -- recency < 30, frequency > 30, monetary top 5%
```

That is the artifact recruiters will see. The "chosen k + defense + cluster profiles" three-liner is the headline.
