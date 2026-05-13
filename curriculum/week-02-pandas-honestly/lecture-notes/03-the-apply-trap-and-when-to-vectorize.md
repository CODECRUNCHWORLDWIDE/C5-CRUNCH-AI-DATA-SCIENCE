# Lecture 3 — The `apply` Trap and When to Vectorize

> **Outcome:** You can spot an `.apply()` call that is secretly a Python `for` loop, rewrite the common cases with `.str` / `.dt` / arithmetic / `np.where` / `merge`, and recognize the small set of situations where `apply` is honestly the right tool.

In Week 1 you learned that `for i in range(len(a)): out[i] = a[i] ** 2` is a code smell. This lecture is the pandas version: `DataFrame.apply` and `Series.apply` and `groupby.apply` are *all* Python `for` loops, dressed up in method syntax. They are 10×–100× slower than the vectorized rewrite for the cases that come up 95% of the time. Eliminate them on sight.

The other 5% — where `apply` is honest — we will name, too.

---

## 1. What `apply` is actually doing

`Series.apply(func)` calls `func` once per element, in Python, in a loop. `DataFrame.apply(func, axis=1)` calls `func` once per row, building a temporary `Series` per row, in a loop. `groupby.apply(func)` calls `func` once per group, which is fine if the *body* is vectorized, and a disaster if it is not.

A microbenchmark, the kind people share on Twitter without context. Sum two columns, ten million rows:

```python
import time
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
N = 10_000_000
df = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})

# Vectorized
t0 = time.perf_counter()
out = df["a"] + df["b"]
print(f"vectorized:  {time.perf_counter() - t0:.3f} s")

# Series.apply on one column (silly, but instructive)
t0 = time.perf_counter()
out = df["a"].apply(lambda x: x * 2)
print(f"Series.apply: {time.perf_counter() - t0:.3f} s")

# DataFrame.apply, axis=1 (the worst one)
t0 = time.perf_counter()
out = df.apply(lambda row: row["a"] + row["b"], axis=1)
print(f"DataFrame.apply axis=1: {time.perf_counter() - t0:.3f} s")
```

On a 2022 laptop with pandas 2.3, the rough numbers look like:

```
vectorized:           0.03 s
Series.apply:         2.0  s
DataFrame.apply axis=1: 60+ s
```

That is two orders of magnitude between `df["a"] + df["b"]` and `df.apply(lambda r: r["a"] + r["b"], axis=1)`. They compute the same thing.

Why? Because `df.apply(..., axis=1)` does, for every one of ten million rows:

1. Build a temporary `Series` of length 2 (`{"a": ..., "b": ...}`).
2. Hand it to your `lambda`.
3. Capture the return.
4. Box it into a new `Series` at the end.

That is ten million Python function calls, ten million small allocations, ten million GIL acquisitions. None of which the vectorized `df["a"] + df["b"]` does — it dispatches once to `np.add` in C.

`Series.apply` is faster than `DataFrame.apply` because it skips the per-row `Series` construction, but it is still a Python loop. Treat both as slow until proven otherwise.

> **Important.** "Slow" here means seconds vs minutes on tens of millions of rows. On a thousand-row DataFrame, `apply` is fine and you should not over-engineer it. The discipline of vectorizing is for data sizes where the difference matters. We teach it on small data so you have the reflex when the data grows.

---

## 2. The five common `apply` patterns and their vectorized rewrites

These cover most of what beginners reach for `apply` for. Memorize the pattern, not the line.

### Pattern A — element-wise math on a Series

```python
# Slow
df["squared"] = df["x"].apply(lambda v: v ** 2)

# Fast
df["squared"] = df["x"] ** 2
```

If the function is arithmetic, the operator works.

### Pattern B — element-wise math on multiple columns

```python
# Slow
df["bmi"] = df.apply(lambda r: r["weight_kg"] / r["height_m"] ** 2, axis=1)

# Fast
df["bmi"] = df["weight_kg"] / df["height_m"] ** 2
```

Arithmetic on `Series` is element-wise and broadcasts naturally. Multi-column "row-wise math" is just column-arithmetic; nothing more.

### Pattern C — string operations

```python
# Slow
df["upper"] = df["name"].apply(str.upper)
df["len"]   = df["name"].apply(len)
df["domain"] = df["email"].apply(lambda s: s.split("@")[1] if "@" in s else None)

# Fast — the .str accessor
df["upper"]  = df["name"].str.upper()
df["len"]    = df["name"].str.len()
df["domain"] = df["email"].str.split("@").str[1]
```

The `.str` accessor on a `Series` of strings dispatches to vectorized C-level implementations (PyArrow under `string[pyarrow]`, NumPy character-array under `object`). Browse the `.str` docs — almost every Python `str` method is there.

### Pattern D — datetime operations

```python
# Slow
df["hour"] = df["pickup_at"].apply(lambda d: d.hour)
df["dow"]  = df["pickup_at"].apply(lambda d: d.strftime("%A"))

# Fast — the .dt accessor
df["hour"] = df["pickup_at"].dt.hour
df["dow"]  = df["pickup_at"].dt.day_name()
```

Same pattern as `.str`. Same speed-up.

### Pattern E — conditional / "if x then y else z"

```python
# Slow
df["bucket"] = df["age"].apply(
    lambda a: "minor" if a < 18 else "adult" if a < 65 else "senior"
)

# Fast — np.select for multi-branch
import numpy as np
df["bucket"] = np.select(
    [df["age"] < 18, df["age"] < 65],
    ["minor",        "adult"],
    default="senior",
)

# Or, two-branch — np.where
df["adult"] = np.where(df["age"] >= 18, True, False)
```

`np.where(cond, x, y)` is the vectorized ternary. `np.select([cond1, cond2, ...], [val1, val2, ...], default=...)` is the vectorized chained-`elif`. Both run in C; both broadcast.

The pandas-native version of the same pattern is `pd.cut`:

```python
df["bucket"] = pd.cut(
    df["age"],
    bins=[-np.inf, 18, 65, np.inf],
    labels=["minor", "adult", "senior"],
)
```

`pd.cut` is the one to reach for when the buckets are intervals; `np.select` is the one for arbitrary boolean conditions.

### Pattern F — "look up a value in a small table"

This is the most under-vectorized pattern.

```python
# Slow
tax_rate = {"NY": 0.08, "CA": 0.0725, "TX": 0.0625}
df["tax"] = df["state"].apply(lambda s: tax_rate.get(s, 0.0))

# Fast — map() is vectorized
df["tax"] = df["state"].map(tax_rate).fillna(0.0)

# Even better — merge with a small DataFrame
tax = pd.DataFrame(
    {"state": list(tax_rate), "rate": list(tax_rate.values())}
)
df = df.merge(tax, on="state", how="left").fillna({"rate": 0.0})
```

`Series.map` accepts a dict, a Series, or a function. With a dict or a Series it is vectorized in C. Use `map` for small lookups; use `merge` when the lookup table is itself a DataFrame (it gives you multiple result columns for free).

---

## 3. The 5% case: when `apply` is honest

Three situations where `apply` is genuinely the right tool:

**(a) The body of the function is itself a non-trivial pandas / NumPy call.** A `groupby.apply` whose body does a regression on each group is not slow because of `apply`; it is slow because regressions are slow. The Python loop overhead is dwarfed by the work inside.

```python
def fit_slope(group):
    x = group["t"].values
    y = group["y"].values
    return np.polyfit(x, y, 1)[0]

slopes = df.groupby("user_id").apply(fit_slope, include_groups=False)
```

`fit_slope` does real work per call. `apply` here is fine.

> **pandas 2.2 note.** `include_groups=False` is the new default behavior recommended in pandas 2.2+ (the group columns are no longer passed into the function). Pass it explicitly to silence the deprecation warning and avoid relying on the old behavior.

**(b) The function has side effects you cannot vectorize.** Calling an external API per row, writing to a database, hashing with a cryptographic library — these are inherently per-call. (Even here, batching is usually possible. But the per-row Python overhead is the cheap part.)

**(c) The expression genuinely cannot be vectorized.** Iterative refinement (each row depends on the previous result of the same column in a way `shift` cannot express); a chain of conditions across more than a handful of branches; calls to a third-party library that takes scalars only. Real but rare. When you think you have one of these, ask first: is `shift` enough? Can I rewrite it as a `groupby` over a windowed key? More often than you expect, the answer is yes.

**Two notes on signals you are *not* in the 5% case:**

- If your `apply` body is "if-elif-elif-else" with simple comparisons, you are not in the 5%. That is `np.select`.
- If your `apply` body calls `.split()`, `.lower()`, `.strip()`, `.startswith()` — anything from `str`'s standard methods — you are not in the 5%. That is `.str`.

---

## 4. The two-line tour of Polars

When pandas runs out of headroom, the most common next stop in 2026 is **Polars**.

Polars is a DataFrame library written in Rust, with a columnar memory layout (Apache Arrow under the hood), full multi-threading by default, and a *lazy* execution mode that lets it plan a whole query before running it. The API is similar enough to pandas that a pandas user can read most Polars code without help.

The same per-region revenue example from Lecture 2:

```python
import polars as pl

trips = pl.read_csv("trips.csv")

rev = (
    trips.join(regions, on="region_id", how="left")
         .group_by(["region_name", "month"])
         .agg(
             pl.col("fare").sum().alias("revenue"),
             pl.col("trip_id").n_unique().alias("n_trips"),
         )
         .sort(["region_name", "month"])
)
```

Three things to notice:

1. **Method chaining is the dominant style.** Polars encourages it; pandas allows it.
2. **`pl.col("fare").sum().alias("revenue")` replaces the named-aggregation form** — same idea, slightly different spelling.
3. **There is no index.** Polars has only columns. The pandas `Index` / `MultiIndex` does not exist; if you need group keys back, they are columns.

When to reach for Polars:

- Your pandas script is the bottleneck and the work is mostly group-by / join / filter.
- You want predictable parallelism (Polars uses all your cores by default).
- You are starting a new project and the team is open to it.

When to *stay* in pandas:

- The rest of your stack is pandas (scikit-learn, statsmodels, plotnine, seaborn all expect pandas as input).
- The dataset fits comfortably in RAM and the script runs in seconds.
- You are reading existing pandas code that you do not own.

For C5 we stay in pandas throughout. The Polars pointer is so you know the off-ramp exists.

---

## 5. The two-line tour of DuckDB

The other 2026 escape hatch: **DuckDB**. An in-process analytical SQL database, no server, that reads CSV / Parquet directly and joins faster than pandas can `merge`.

```python
import duckdb

revenue = duckdb.sql("""
    SELECT region_name, month, SUM(fare) AS revenue, COUNT(DISTINCT trip_id) AS n_trips
    FROM 'trips.csv' AS t
    LEFT JOIN 'regions.csv' AS r USING (region_id)
    GROUP BY region_name, month
    ORDER BY region_name, month
""").df()      # returns a pandas DataFrame
```

That is one SQL string. DuckDB streams the CSV, planning and executing the join and aggregation in parallel, and returns a pandas DataFrame at the end.

When to reach for DuckDB:

- You have CSV / Parquet on disk larger than RAM, and the analysis is expressible as SQL.
- Your team reads SQL more comfortably than pandas.
- You are joining 10 GB of data and the pandas `merge` is taking minutes.

When to stay in pandas: same as Polars — the rest of the stack is pandas.

The honest summary: **in 2026, a senior data engineer reaches for DuckDB or Polars when the pandas script is too slow, not as a first move.** pandas is still the lingua franca. Both alternatives interoperate cleanly via Arrow; you can swap mid-pipeline.

---

## 6. The pre-flight checklist before you write `.apply`

Before you type `.apply` in real code, run this in your head:

1. **Is this arithmetic on `Series`?** If yes, use the operator (`+`, `*`, `**`).
2. **Is this a string operation?** If yes, use `.str.<method>`.
3. **Is this a datetime operation?** If yes, use `.dt.<attribute>`.
4. **Is this `if cond then x else y`?** If yes, use `np.where` or `np.select` or `pd.cut`.
5. **Is this "look up a value"?** If yes, use `.map` or `.merge`.
6. **Is this aggregation per group?** If yes, use `.groupby(...).agg(...)`.
7. **None of the above?** Then `apply` may be honest. Verify with a microbenchmark on 100K rows.

The first six bullets cover almost every case you will see for the rest of C5.

---

## 7. The mental model to walk away with

```
   ┌──────────────────────────────────────────────────────────┐
   │ .apply(...)  ≅  for row in df: func(row)                 │
   │              ≅  for v in series: func(v)                 │
   │                                                          │
   │ Default assumption: slow. Replace it.                    │
   │ Exceptions: regression-per-group, API-call-per-row,      │
   │   genuinely sequential logic. Name them when you use them.│
   └──────────────────────────────────────────────────────────┘

   Replacement menu:
       arithmetic         →  operators on Series
       string             →  .str.<method>
       datetime           →  .dt.<attribute>
       conditional        →  np.where / np.select / pd.cut
       lookup             →  .map / .merge
       per-group aggregate →  .groupby(...).agg(...)
       per-group window   →  .groupby(...).transform(...) / shift / rolling
```

Three habits, in priority order:

1. **`.apply` is a code smell.** Read it as "this is a Python loop." Replace it unless you have named the exception.
2. **`.str` and `.dt` exist. Use them.** Every Python `str` and `datetime` method has a vectorized counterpart.
3. **`np.where`, `np.select`, `.map`, and `.merge` cover the rest.** When the rewrite is not one of those four, ask whether you are really in the 5% case.

---

## 8. Self-check

Without re-reading:

1. Estimate the speed-up of `df["a"] + df["b"]` over `df.apply(lambda r: r["a"] + r["b"], axis=1)` on 10M rows. Order of magnitude.
2. You have `df["name"]` of type `string[pyarrow]` and want a new column with the part after `"@"` in each email. Write it without `apply`.
3. You have `df["age"]` and want a `bucket` column with `"minor" / "adult" / "senior"`. Two acceptable vectorized forms — name both.
4. You have a `state → tax_rate` dict and a `df["state"]` column. Write the vectorized lookup.
5. Name two situations where `apply` is the right tool.
6. What does Polars give up that pandas has? What does it gain?
7. What kind of query is DuckDB obviously faster at than pandas?
8. Why is `df.apply(..., axis=1)` slower than `df["col"].apply(...)` even when the function is the same?

---

## Further reading

- **pandas — Group by: `apply` and the alternatives**: <https://pandas.pydata.org/docs/user_guide/groupby.html#flexible-apply>
- **pandas — `.str` accessor methods**: <https://pandas.pydata.org/docs/user_guide/text.html>
- **pandas — Enhancing performance** (Cython, Numba, `eval`): <https://pandas.pydata.org/docs/user_guide/enhancingperf.html>
- **Polars user guide**: <https://docs.pola.rs/user-guide/getting-started/>
- **DuckDB — Python integration**: <https://duckdb.org/docs/api/python/overview>
- **Tom Augspurger, *Modern Pandas*, part 4 ("Performance")**: <https://tomaugspurger.net/posts/modern-4-performance/>

Next: take the [quiz](../quiz.md), do the [exercises](../exercises/README.md), and start the [mini-project](../mini-project/README.md).
