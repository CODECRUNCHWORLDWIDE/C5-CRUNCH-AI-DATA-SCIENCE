# Lecture 2 — Merge, Group-by, Aggregate

> **Outcome:** You can join two DataFrames with the correct `how=`, explain the row-count change without surprise, group by one or more keys and apply multiple aggregations, and choose between long and wide layouts on purpose.

Lecture 1 was about one DataFrame. This lecture is about two or more — and about asking questions of one DataFrame that the dtype machinery alone cannot answer ("revenue per region per month"). Two operations carry most of this weight: **`merge`** (joining tables) and **`groupby`** (split-apply-combine). They are also the operations that, done wrong, silently produce wrong numbers. Spend the time.

We continue on **pandas 2.x** (2.3 in 2026).

---

## 1. The three combiner verbs: `merge`, `concat`, `join`

pandas has three functions that combine DataFrames, and the differences between them are not academic.

| Function | What it does | When to reach for it |
|----------|--------------|----------------------|
| **`merge`** | SQL-style join on one or more **column** values (or on the index) | "I have orders and customers; give me one row per order with the customer's city." |
| **`concat`** | Stack DataFrames along an axis (rows by default) | "I have January, February, March files with the same columns — give me the year." |
| **`join`** | A convenience wrapper around `merge` that joins on the **index** | "I have two tables that already share an index; merge on it." |

In practice you will use `merge` and `concat` constantly and `join` occasionally. We will spend most of this lecture on `merge`.

### `concat` first, because it is simpler

```python
>>> jan = pd.DataFrame({"date": ["2025-01-15"], "amt": [100]})
>>> feb = pd.DataFrame({"date": ["2025-02-10"], "amt": [120]})
>>> pd.concat([jan, feb], ignore_index=True)
         date  amt
0  2025-01-15  100
1  2025-02-10  120
```

`ignore_index=True` discards the original row indices (which were both `0`) and gives a fresh `RangeIndex`. Almost always the right choice when stacking files.

`concat(axis=1)` stacks columns side by side, aligning on the index. If your two DataFrames share an index, this is a quick `join`-by-index. If they don't, you get `NaN`s.

### `merge` second, because it is everything

The minimal call:

```python
>>> merged = orders.merge(customers, on="customer_id", how="left")
```

That is: take `orders`, find the matching row in `customers` for each `customer_id`, attach the customer columns. `how="left"` means "keep every order, even if the customer row is missing." We will spend §3 on the four values of `how`.

---

## 2. The four joins — once with a Venn diagram, three times with code

Imagine two tables, **L** (left) and **R** (right), with a key column. Each value of the key may appear in L only, R only, or both.

```
   ┌─────── L ───────┐    ┌─────── R ───────┐
   │ L-only │ both    │ │ both   │ R-only   │
   └────────┼─────────┘ └────────┼──────────┘
            └───── intersection ─┘
```

The four `how=` values pick which slice you want:

| `how=` | What you get |
|--------|--------------|
| `"inner"` | Only the intersection — rows where the key matched on both sides |
| `"left"` | Every row of L; R columns are `NaN` where no match |
| `"right"` | Every row of R; L columns are `NaN` where no match |
| `"outer"` | The union — every row of L *and* every row of R; `NaN` on the side that did not match |

In code:

```python
>>> L = pd.DataFrame({"k": ["a", "b", "c"],     "x": [1, 2, 3]})
>>> R = pd.DataFrame({"k": ["b", "c", "d"],     "y": [20, 30, 40]})

>>> L.merge(R, on="k", how="inner")
   k  x   y
0  b  2  20
1  c  3  30

>>> L.merge(R, on="k", how="left")
   k  x     y
0  a  1   NaN
1  b  2  20.0
2  c  3  30.0

>>> L.merge(R, on="k", how="right")
   k    x   y
0  b  2.0  20
1  c  3.0  30
2  d  NaN  40

>>> L.merge(R, on="k", how="outer")
   k    x     y
0  a  1.0   NaN
1  b  2.0  20.0
2  c  3.0  30.0
3  d  NaN  40.0
```

Read those four results carefully. Notice that `inner` is the smallest, `outer` is the largest, and `left` and `right` are between. The row count is the easiest thing to get wrong and the easiest thing to verify.

### The `validate=` and `indicator=` arguments — use them

`merge` quietly accepts duplicate keys on either side. A many-to-many merge can multiply your row count and you will not notice until the analysis is broken.

Two arguments protect you:

```python
merged = orders.merge(
    customers,
    on="customer_id",
    how="left",
    validate="many_to_one",   # every order has at most one customer
    indicator=True,           # adds a "_merge" column: 'left_only' / 'right_only' / 'both'
)
```

`validate="many_to_one"` raises if `customers` has duplicate `customer_id` rows. `indicator=True` adds a categorical column that tells you, for every output row, where it came from. After a `left` join, the `_merge == "left_only"` rows are the orders with no matching customer — exactly the rows you want to investigate.

> **Rule:** for any merge you intend to ship, pass `validate=` and `indicator=`. They are free. They turn a silent class of bugs into a loud one.

### Different column names on each side

The `on=` argument requires the same column name on both sides. When the keys are named differently:

```python
merged = orders.merge(
    customers,
    left_on="customer_id",
    right_on="id",
    how="left",
)
```

`merged` will then have both `customer_id` and `id` columns. Drop one with `.drop(columns="id")`.

### Multi-column keys

```python
merged = trips.merge(
    weather,
    on=["pickup_date", "pickup_borough"],
    how="left",
)
```

Both columns must match for a row to join. This is the common pattern for any "by day, by location" or "by user, by month" join.

### Joining on the index

If both DataFrames already use the same column as their index, `df1.join(df2)` is the cleaner spelling. It defaults to a left join on the index.

---

## 3. `groupby`: split-apply-combine

This is the single most expressive pandas operation. Once you understand it, half of your analyses get one line shorter.

The model is three steps:

1. **Split** the rows into groups by one or more keys.
2. **Apply** a function to each group.
3. **Combine** the per-group results back into a single object.

```python
>>> df = pd.DataFrame({
...     "region":   ["NE", "NE", "SE", "SE", "NE", "SE"],
...     "salesrep": ["Ada", "Bo", "Ada", "Cy",  "Bo", "Cy"],
...     "amount":   [100, 200, 150, 250, 300, 350],
... })
>>> df.groupby("region")["amount"].sum()
region
NE    600
SE    750
Name: amount, dtype: int64
```

That is one line. The equivalent in `for`-loop pseudocode:

```python
totals = {}
for region, group in df.groupby("region"):
    totals[region] = group["amount"].sum()
```

You almost never want to write the loop. The `.sum()` happens in C across all groups at once.

### Common aggregations

Every reduction you used in NumPy is here, applied per group:

```python
df.groupby("region")["amount"].sum()
df.groupby("region")["amount"].mean()
df.groupby("region")["amount"].std()
df.groupby("region")["amount"].count()      # non-null count
df.groupby("region")["amount"].size()       # row count (including nulls)
df.groupby("region")["amount"].nunique()    # distinct values
df.groupby("region")["amount"].first()      # first row per group
df.groupby("region")["amount"].last()       # last row per group
df.groupby("region")["amount"].quantile(0.95)
```

`count` versus `size` is a beginner trap: `count` excludes `NaN`, `size` does not. If you want "how many rows are in this group", you want `size`.

### Multiple aggregations at once with `.agg`

```python
>>> df.groupby("region")["amount"].agg(["sum", "mean", "count"])
        sum   mean  count
region
NE      600  200.0      3
SE      750  250.0      3
```

Different aggregations per column:

```python
>>> df.groupby("region").agg(
...     total_amount=("amount", "sum"),
...     mean_amount =("amount", "mean"),
...     n_reps      =("salesrep", "nunique"),
... )
        total_amount  mean_amount  n_reps
region
NE               600        200.0       2
SE               750        250.0       2
```

That is the **named-aggregation** form: `new_col=(source_col, aggfunc)`. It is the cleanest spelling pandas offers and the one you should default to.

### Multiple group keys

```python
df.groupby(["region", "salesrep"])["amount"].sum()
```

Returns a `Series` with a `MultiIndex` (`region`, `salesrep`). `.unstack()` will pivot the inner level into columns; we will see that in §6.

### `transform`, `filter`, `apply` — the three escape hatches

- **`transform(func)`** returns a `Series` the same length as `df`, with each row replaced by the per-group result. Useful for "subtract the group mean from every row":

  ```python
  df["amount_centered"] = (
      df["amount"] - df.groupby("region")["amount"].transform("mean")
  )
  ```

- **`filter(func)`** keeps or drops *whole groups* based on a predicate:

  ```python
  big_regions = df.groupby("region").filter(lambda g: g["amount"].sum() > 700)
  ```

- **`apply(func)`** is the most flexible and the slowest. Lecture 3 is mostly about not reaching for it.

### `groupby` with `as_index=False`

By default `groupby` returns a result with the group keys as the index. Sometimes you want them as columns:

```python
df.groupby("region", as_index=False)["amount"].sum()
#   region  amount
# 0     NE     600
# 1     SE     750
```

If you are going to immediately `merge` the result back onto the original DataFrame, `as_index=False` saves a `reset_index()`.

---

## 4. A worked example: per-region revenue with growth

Tie merge and group-by together. You have a `trips` table (one row per ride) and a `regions` lookup table (region code → region name).

```python
trips = pd.DataFrame({
    "trip_id":   range(1, 9),
    "region_id": [10, 20, 10, 30, 20, 20, 10, 30],
    "month":     ["2025-08", "2025-08", "2025-09", "2025-08",
                  "2025-09", "2025-09", "2025-09", "2025-09"],
    "fare":      [12.5, 30.0, 14.0, 22.0, 28.0, 31.0, 15.0, 24.0],
})

regions = pd.DataFrame({
    "region_id":   [10, 20, 30],
    "region_name": ["Manhattan", "Brooklyn", "Queens"],
})

# 1) attach the human-readable region name
enriched = trips.merge(
    regions,
    on="region_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)
assert (enriched["_merge"] == "both").all(), "found trips with no region"
enriched = enriched.drop(columns="_merge")

# 2) revenue per region per month
rev = enriched.groupby(["region_name", "month"], as_index=False).agg(
    revenue=("fare", "sum"),
    n_trips=("trip_id", "nunique"),
)

# 3) month-over-month growth, per region
rev = rev.sort_values(["region_name", "month"])
rev["revenue_prev"] = rev.groupby("region_name")["revenue"].shift(1)
rev["growth_pct"]   = (rev["revenue"] / rev["revenue_prev"] - 1) * 100
```

Five steps. No loops. The `merge` brings the region name in; the `groupby` aggregates fares and counts trips; `shift(1)` within a group looks at the prior month; arithmetic gives the growth percentage. Notice how `shift` inside `groupby` is the pandas idiom for "previous row, but reset at each group boundary."

---

## 5. `pivot`, `pivot_table`, `melt`: long vs wide

Real datasets come in **long** form (one observation per row, with the type-of-observation in a column) or **wide** form (one row per entity, with each measurement in its own column). You will reshape between them all week.

### Long → wide with `pivot_table`

```python
>>> long = pd.DataFrame({
...     "region": ["NE", "NE", "SE", "SE"],
...     "month":  ["Aug", "Sep", "Aug", "Sep"],
...     "rev":    [100, 110, 200, 220],
... })
>>> long.pivot_table(index="region", columns="month", values="rev")
month     Aug  Sep
region
NE        100  110
SE        200  220
```

`pivot_table` aggregates when there are duplicate `(index, columns)` pairs; the default is `mean`. Pass `aggfunc=` to override:

```python
long.pivot_table(index="region", columns="month", values="rev", aggfunc="sum")
```

There is also a plain `pivot()` that errors instead of aggregating when there are duplicates. Use `pivot_table` by default — it is more forgiving and the error message when duplicates exist is clearer.

### Wide → long with `melt`

```python
>>> wide = pd.DataFrame({
...     "region": ["NE", "SE"],
...     "Aug":    [100, 200],
...     "Sep":    [110, 220],
... })
>>> wide.melt(id_vars="region", var_name="month", value_name="rev")
  region month  rev
0     NE   Aug  100
1     SE   Aug  200
2     NE   Sep  110
3     SE   Sep  220
```

`id_vars=` are the columns to keep as-is; the rest become two columns: one for the old column name (`var_name`), one for the value (`value_name`).

### Why you reshape

- **Wide** is what humans read in a spreadsheet.
- **Long** is what almost every pandas / seaborn / SQL operation expects.

The pattern in real code: load whatever you got, **melt to long** for analysis, **pivot to wide** at the end if a human is reading the output.

---

## 6. `stack` and `unstack` — the index-level versions of pivot/melt

`pivot_table` and `melt` work on column values. `stack` / `unstack` work on `MultiIndex` *levels*. They are exactly the same idea, just in index space.

```python
>>> df = pd.DataFrame({
...     "rev": [100, 110, 200, 220],
... }, index=pd.MultiIndex.from_tuples(
...     [("NE","Aug"),("NE","Sep"),("SE","Aug"),("SE","Sep")],
...     names=["region","month"],
... ))
>>> df.unstack("month")        # move the 'month' level from index to columns
        rev
month   Aug  Sep
region
NE      100  110
SE      200  220
>>> df.unstack("month").stack(future_stack=True)   # back to long
```

> **pandas 2.x note.** The old `DataFrame.stack()` will be replaced in pandas 3.0 by the `future_stack=True` semantics, which handle missing data more sensibly. Pass `future_stack=True` today in new code.

---

## 7. Time-series basics (a brief detour)

You will not escape this week without one datetime column. The big three idioms:

```python
# Parse strings to datetimes — once, at load time.
df["pickup_at"] = pd.to_datetime(df["pickup_at"], errors="coerce", utc=True)

# The .dt accessor — like .str but for datetimes.
df["pickup_hour"]    = df["pickup_at"].dt.hour
df["pickup_weekday"] = df["pickup_at"].dt.day_name()
df["pickup_month"]   = df["pickup_at"].dt.to_period("M")

# Resample: like groupby, but for time bins.
hourly = (
    df.set_index("pickup_at")["fare"]
      .resample("1h").sum()
)
```

`resample` requires a `DatetimeIndex`. The frequency strings (`"1h"`, `"1D"`, `"1ME"` for month-end, `"1W"`) are listed in the [pandas offset alias docs](https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases).

> **pandas 2.2 note.** Several frequency aliases were renamed: `"M"` → `"ME"` (month-end), `"Q"` → `"QE"` (quarter-end), `"Y"` → `"YE"` (year-end). The old aliases still work in 2.x and warn; in 3.0 they will be removed. Update your code now.

---

## 8. A debugging checklist for merges and group-bys

Every join or aggregation you ship goes through this checklist. Five seconds each, half a minute total, hours of bugs avoided.

**Before the merge:**

- [ ] What is the row count of each input?
- [ ] What is the unique-key count on each side? (`df[key].nunique()`)
- [ ] Is the key really unique on the right side, or are you about to many-to-many by accident?

**The merge itself:**

- [ ] Did I pass `validate=`?
- [ ] Did I pass `how=` explicitly?
- [ ] Did I pass `indicator=True`?

**After the merge:**

- [ ] Is the row count what I expected? (If `left`, equals the left's row count. If `inner`, ≤ each side's row count.)
- [ ] What does `merged["_merge"].value_counts()` say? How many rows fell into `left_only`?

**Before the groupby:**

- [ ] Are the group keys what I think they are? (`df[key].value_counts()`)
- [ ] Does the column I am aggregating have the dtype I think? (Aggregating an `object` column silently does string concatenation, not addition.)

**After the groupby:**

- [ ] Does the number of output rows equal the number of unique keys?
- [ ] Are there `NaN`s in the aggregated column? Did I want to skip them, or fail loudly?

This is the kind of checklist a senior engineer runs internally without saying so. Write it down for a month and then you will.

---

## 9. The mental model to walk away with

```
   merge:   ┌── L ──┐    ┌── R ──┐
            │       │ ⋈  │       │   on=key, how=inner/left/right/outer
            └───────┘    └───────┘
                       │
                       ▼
                  combined table
                  validate= protects you
                  indicator= explains what happened


   groupby: rows  ──► [split by key]  ──► group_1, group_2, group_3
                                                  │
                                                  ▼  agg/transform/filter
                                         per-group results
                                                  │
                                                  ▼
                                            combined back
```

Three habits, in priority order:

1. **Pass `how=` and `validate=` explicitly on every merge.** If you don't, the next person on call (which might be you) cannot tell what you meant.
2. **Reach for `.agg(name=(col, func))` whenever you want multiple aggregations.** It is the only spelling that reads well a month from now.
3. **Reshape on purpose.** Long for analysis, wide for humans. `melt` and `pivot_table` are the two verbs.

Every later week uses these exact verbs. Feature engineering in Week 4 is a stack of group-bys and merges. Time-series cross-validation in Week 10 is a `groupby` over time bins. Get this lecture right and the rest is shorter.

---

## 10. Self-check

Without re-reading:

1. Two tables `L` (3 rows) and `R` (4 rows) share a key with 2 matching values. What is the row count for `L.merge(R, on='k', how='inner')`? `how='left'`? `how='outer'`?
2. Name two arguments to `merge` you should pass on every production merge.
3. `df.groupby('region').size()` versus `df.groupby('region').count()`: how do they differ?
4. Rewrite this with **named aggregation**: `df.groupby("region")["amount"].agg(["sum", "mean"])`.
5. You have a long-format `(region, month, rev)` table. Write one expression that produces a wide table with regions as rows and months as columns.
6. What does `df.groupby("region")["x"].transform("mean")` return? How does that differ from `df.groupby("region")["x"].mean()`?
7. Your `merge` returned 100,000 rows when you expected 50,000. What is the most likely cause?
8. You have a `DatetimeIndex` and want the sum of `fare` per calendar day. Write the one-liner.

---

## Further reading

- **pandas — Merging, joining, and concatenating**: <https://pandas.pydata.org/docs/user_guide/merging.html>
- **pandas — Group by: split-apply-combine**: <https://pandas.pydata.org/docs/user_guide/groupby.html>
- **pandas — Reshaping and pivot tables**: <https://pandas.pydata.org/docs/user_guide/reshaping.html>
- **Wes McKinney, *Python for Data Analysis*, chapters 8 and 10**: <https://wesmckinney.com/book/>
- **Tom Augspurger, *Modern Pandas*, part 4 ("Performance")**: <https://tomaugspurger.net/posts/modern-4-performance/>

Next: [Lecture 3 — The `apply` trap and when to vectorize](./03-the-apply-trap-and-when-to-vectorize.md).
