# Week 1 — Homework

Six problems, about six hours total. Commit each in your Week 1 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics are below.

---

## Problem 1 — Time the Python loop versus the NumPy sum (45 min)

Write `homework/01-list-vs-array-timing.py` that, for each size `N` in `{10_000, 100_000, 1_000_000, 10_000_000}`:

1. Builds `xs = [float(i) for i in range(N)]` and `a = np.arange(N, dtype=np.float64)`.
2. Times summing `xs` with a Python `for` loop.
3. Times summing `xs` with built-in `sum(xs)`.
4. Times summing `a` with `a.sum()`.

Use `time.perf_counter` and run each timing **at least three times**, reporting the minimum. (The minimum is more honest than the mean — slow runs are usually contention, not algorithmic.)

**Acceptance.**

- The script runs in under 60 seconds end-to-end.
- It prints a small table with the four sizes and three timings each.
- Commit a `homework/01-results.md` containing the table and a one-paragraph interpretation: at what `N` does the gap matter? Where does NumPy stop being worth it?

---

## Problem 2 — Predict the shape, then check (45 min)

For each of the following arrays, write down the **expected shape on paper**, then confirm with code.

```python
a = np.arange(24).reshape(2, 3, 4)
```

What is the shape of:

1. `a[0]`
2. `a[:, 0]`
3. `a[..., 0]`
4. `a[1, 1, 1]`
5. `a.sum(axis=0)`
6. `a.sum(axis=1)`
7. `a.sum(axis=2)`
8. `a.sum(axis=(0, 2))`
9. `a.transpose(2, 0, 1)`
10. `a[:, None]`

Commit `homework/02-shape-predictions.md`. For every entry, write your *predicted* shape, the *actual* shape, and a one-line explanation if they differed.

---

## Problem 3 — From-scratch standardization (45 min)

In `homework/03-standardize.py`, implement two functions:

```python
def standardize_columns(X: np.ndarray) -> np.ndarray:
    """Return X with each column shifted to mean 0, std 1 (ddof=0)."""

def standardize_rows(X: np.ndarray) -> np.ndarray:
    """Return X with each row shifted to mean 0, std 1 (ddof=0)."""
```

Each function must be **one expression** (chained calls and arithmetic are fine; an explicit `for` loop is not).

Include a `test_standardize()` function (under `if __name__ == "__main__":`) that:

- Builds a `(100, 4)` matrix from `rng.normal(loc=5, scale=2, size=(100, 4))` with `rng = np.random.default_rng(0)`.
- Calls both functions and asserts the appropriate axis has mean ≈ 0 and std ≈ 1.

**Acceptance.** `python homework/03-standardize.py` prints `OK` and nothing else.

---

## Problem 4 — Vectorize a pairwise distance matrix (1 h)

Given `X` of shape `(N, D)`, the **pairwise distance matrix** `D` of shape `(N, N)` has `D[i, j] = ‖X[i] − X[j]‖₂`. The naïve loop is `O(N²)` Python operations:

```python
def slow_pairwise(X: np.ndarray) -> np.ndarray:
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.sqrt(((X[i] - X[j]) ** 2).sum())
    return D
```

In `homework/04-pairwise.py`, write a vectorized `fast_pairwise(X)` using one of these patterns:

- Broadcasting: `X[:, None, :] - X[None, :, :]` yields shape `(N, N, D)`, then `**2`, sum over the last axis, and `sqrt`.
- The `(a - b)² = a² − 2ab + b²` trick: compute `||X||² + ||X||²ᵀ − 2 X Xᵀ` and `sqrt`. This is the trick scikit-learn uses internally.

Time both `slow_pairwise(X)` and `fast_pairwise(X)` for `N = 500, D = 10`. Commit `homework/04-results.md` with the two timings and the speedup factor (single number, with one decimal place). Confirm they agree with `np.allclose`.

---

## Problem 5 — Read and explain a NumPy snippet (45 min)

Find one of the following in real NumPy or scikit-image source code (links below):

- `numpy/_core/numeric.py` — pick any function ≤ 30 lines.
- `skimage/color/colorconv.py` — function `rgb2gray`.
- `numpy-100` repo, any "hard" exercise's solution.

Links:

- <https://github.com/numpy/numpy/tree/main/numpy/_core>
- <https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py>
- <https://github.com/rougier/numpy-100>

In `homework/05-snippet-walkthrough.md`, paste the snippet (with attribution and a permalink) and write a paragraph (≈150 words) explaining what each line does, what shape each intermediate has, and which broadcasting rule each line relies on.

---

## Problem 6 — Reflection (30 min)

Write `homework/06-reflection.md` (250–400 words) answering:

1. What did you most expect to know about NumPy that turned out to be wrong?
2. Which lecture moment (a paragraph, a diagram, a code example) changed your mental model the most?
3. Where in your previous Python code (C1 projects, side projects) did you write a `for` loop that you would now refuse to write?
4. What is the one NumPy feature you want to go deeper on before Week 2?

Honest is more valuable than polished. If your answer to (1) is "I thought `a.sum()` and `np.sum(a)` were different functions and now I know they aren't," say that.

---

## Time budget

| Problem | Time |
|--------:|----:|
| 1 | 45 min |
| 2 | 45 min |
| 3 | 45 min |
| 4 | 1 h |
| 5 | 45 min |
| 6 | 30 min |
| **Total** | **~4 h 30 min** |

When done, push your Week 1 repo and start the [mini-project](./mini-project/README.md).
