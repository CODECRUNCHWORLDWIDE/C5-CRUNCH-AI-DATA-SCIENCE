# Week 6 — Exercises

Three drills. Each is a `.py` file with TODOs you fill in. Do them in order — they walk you from k-means built by hand through dimensionality reduction and into the honest "is `k` real?" diagnostics.

1. **[Exercise 1 — K-Means from Scratch](exercise-01-kmeans-from-scratch.py)** — implement Lloyd's algorithm in pure NumPy. The assignment step, the centroid-update step, the k-means++ initialization. Verify it agrees with `sklearn.cluster.KMeans` on Gaussian-blob data via `adjusted_rand_score > 0.9`. (~70 min)
2. **[Exercise 2 — PCA and UMAP on Digits](exercise-02-pca-and-umap.py)** — implement PCA via the SVD, verify the explained-variance ratios match sklearn to three decimals, and produce a side-by-side 2D embedding of the digits dataset via PCA, t-SNE, and (optionally) UMAP. (~60 min)
3. **[Exercise 3 — Choose `k` Honestly](exercise-03-choose-k-honestly.py)** — implement the WCSS elbow curve, the silhouette curve, the gap statistic, and the column-permutation baseline. Run them on a clustered dataset and a noise dataset and verify the four diagnostics agree on each. (~60 min)

## Workflow

- Type, do not paste.
- Each file builds its own deterministic synthetic data (or uses a scikit-learn bundled dataset) so the tests never depend on the network.
- Each file has built-in `assert`-style checks at the bottom. Run with `python exercise-XX.py` and watch it print `OK`.
- Hint blocks live at the very bottom of each file, commented out. Read them only after fifteen minutes of trying a problem.
- Every clustering you fit should be inspected in 2D before you decide whether it is real. Lecture 3 makes this explicit.

## Self-grading

After each exercise, ask: *could I drop this code into the mini-project tomorrow?* If yes, move on. If no, the gap is the lesson.

## Running with `pytest`

A minimal `pytest`-style smoke test is at the bottom of each file. From the `exercises/` directory:

```bash
pytest exercise-01-kmeans-from-scratch.py
pytest exercise-02-pca-and-umap.py
pytest exercise-03-choose-k-honestly.py
```

All three should pass before you move to the [homework](../homework.md), the [challenge](../challenges/), and the [mini-project](../mini-project/README.md).

Exercise 2 skips the UMAP-specific shape check if `umap-learn` is not installed. To run the full comparison:

```bash
pip install "umap-learn>=0.5,<1"
```

On macOS arm64, `umap-learn`'s transitive dependency on `numba` is large (~50 MB) and slow to install the first time. See [resources.md](../resources.md) for the installation notes.

## Compilation check

Each file must compile cleanly with `python -m py_compile exercise-XX.py`. The CI runs that check.
