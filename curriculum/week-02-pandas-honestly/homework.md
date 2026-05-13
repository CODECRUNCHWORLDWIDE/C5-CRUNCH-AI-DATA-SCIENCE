# Week 2 — Homework

Six problems, about six hours total. Commit each in your Week 2 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — Time `.apply` versus the vectorized form (45 min)

Write `homework/01-apply-vs-vectorized.py` that, for each row count `N` in `{10_000, 100_000, 1_000_000, 10_000_000}`:

1. Builds a DataFrame `df` with two `float64` columns `a` and `b` of length `N`, drawn from `np.random.default_rng(0).normal(size=N)`.
2. Times three implementations of `c = a + b`:
   - `df["a"] + df["b"]` — operator on `Series`.
   - `df.apply(lambda r: r["a"] + r["b"], axis=1)` — the bad spelling.
   - `df["a"].to_numpy() + df["b"].to_numpy()` — pure NumPy.
3. Reports the wall-clock time for each, taking the minimum of three runs.

Use `time.perf_counter`.

**Acceptance.**

- The script finishes in under five minutes total. (At `N = 10M` the `apply` row will take ~60 s; that is part of the lesson.)
- It prints a small table of `N` vs the three timings and the speed-up factor of vectorized vs `apply`.
- Commit `homework/01-results.md` with the table and a one-paragraph interpretation: at what `N` does the gap start to matter for *you*? What is the speed-up at `N = 1M`?

---

## Problem 2 — `.loc` vs `.iloc` predict-and-check (45 min)

For each of the following expressions on `df`, predict on paper what is returned, then run and confirm.

```python
import pandas as pd
df = pd.DataFrame(
    {"x": [10, 20, 30, 40], "y": [1.1, 2.2, 3.3, 4.4]},
    index=["a", "b", "c", "d"],
)
```

Expressions:

1. `df.loc["b"]`
2. `df.iloc[1]`
3. `df.loc["b":"c"]`
4. `df.iloc[1:3]`
5. `df.loc[:, "x"]`
6. `df.iloc[:, 0]`
7. `df.loc["b", "y"]`
8. `df.iloc[1, 1]`
9. `df.loc[["a", "c"], "y"]`
10. `df.loc[df["x"] > 15]`

For each, in `homework/02-loc-vs-iloc.md`, write: the predicted return type and shape, the actual value, and (if you missed) a one-line note on what you got wrong.

---

## Problem 3 — Clean one real public dataset, end to end (1 h 15 min)

Pick **one** of:

- A subset of the NYC 311 service-requests CSV: <https://data.cityofnewyork.us/Social-Services/311-Service-Requests/erm2-nwe9>
- A single month of NYC green-taxi Parquet: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- The Our World in Data COVID dataset: <https://github.com/owid/covid-19-data/tree/master/public/data>

In `homework/03-clean.py`:

1. Load the file with the right `read_csv` / `read_parquet` arguments.
2. Print the output of the five-minute first look (`shape`, `info`, `head`, missing counts per column).
3. Convert every column to a sensible dtype (datetimes, categoricals, nullable ints). No `object`-typed columns should remain except free-text fields you intend to keep as text — and those should be `string[pyarrow]`, not `object`.
4. Drop columns you do not need for any downstream analysis (be honest).
5. Save the cleaned result as a Parquet file (`cleaned.parquet`) and print the size on disk before and after cleaning.

In `homework/03-notes.md`, write 200 words on: what was the worst dtype problem you found? Which column did you have to use `errors="coerce"` for, and how much data did that drop?

---

## Problem 4 — Group-by + merge end to end (1 h)

In `homework/04-groupby-merge.py`, working from the cleaned Parquet of Problem 3:

1. Compute a per-group summary: pick a natural grouping key for your dataset (borough, country, complaint type, etc.) and a natural metric (count of rows, sum of fares, mean of cases). Use named aggregation.
2. Compute a second per-group summary at a different grain (e.g., per group per month, or per group per day-of-week).
3. Merge the two summaries back together on the shared key, picking `how=` and `validate=` deliberately. State your choices in a comment.
4. Save the merged summary as `summary.parquet`.

In `homework/04-notes.md`, paste the head of your final table and one paragraph (≈150 words) on what jumps out from the numbers. "Things that jumped out" can be small ("Manhattan has 3× the trips of Brooklyn at all hours"); they should be specific.

---

## Problem 5 — Refactor an `.apply` you found in the wild (45 min)

Find an `.apply` call in real code. You may use:

- Any GitHub search for `df.apply(lambda` in a `.py` file: <https://github.com/search?q=df.apply%28lambda+language%3Apython&type=code>
- An old C1 project of your own.
- A Stack Overflow answer with `.apply` in the snippet.

Paste the snippet (with attribution and a permalink) into `homework/05-refactor.md`. Write a paragraph on:

1. What it does.
2. Which vectorized rewrite from Lecture 3 applies (`np.where`, `.str`, `.dt`, `.map`, `.merge`, arithmetic).
3. The rewritten code.
4. A one-line note on whether you actually believe the rewrite is correct, or whether you would need test cases.

The point is the habit of reading other people's code and asking "is this the slow shape or the fast shape?"

---

## Problem 6 — Reflection (30 min)

Write `homework/06-reflection.md` (250–400 words) answering:

1. What did you most expect to know about pandas that turned out to be wrong?
2. Which lecture moment (a paragraph, a code example, a diagram) changed your mental model the most?
3. In the cleaning exercise (Exercise 2 or Problem 3), which dtype repair did you find ugliest? Why?
4. After Lecture 3, where in your own past code (C1 projects, side projects, work scripts) did you write an `.apply` that you now know how to vectorize?
5. What is the one pandas feature you want to go deeper on before Week 3?

Honest is more valuable than polished.

---

## Time budget

| Problem | Time |
|--------:|----:|
| 1 | 45 min |
| 2 | 45 min |
| 3 | 1 h 15 min |
| 4 | 1 h |
| 5 | 45 min |
| 6 | 30 min |
| **Total** | **~5 h** |

When done, push your Week 2 repo and start the [mini-project](./mini-project/README.md).
