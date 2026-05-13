# Week 2 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** Which of the following best describes a pandas `DataFrame`?

- A) A 2-D NumPy array with an index attached.
- B) A dict of `Series`, each potentially a different dtype, all sharing one row index.
- C) A list of rows, each row a dict.
- D) A wrapper around an SQL table held in memory.

---

**Q2.** Given `df` with an index `[100, 200, 300]` and one column `x` with values `[10, 20, 30]`, what does `df.loc[0]` do?

- A) Returns the row with `x = 10`.
- B) Returns the row at integer position 0 (`x = 10`).
- C) Raises `KeyError` — `0` is not a label in the index.
- D) Returns `NaN`.

---

**Q3.** Which idiom correctly sets `salary` to `0` for all rows where `age > 30`, without triggering `SettingWithCopyWarning`?

- A) `df["salary"][df["age"] > 30] = 0`
- B) `df.loc[df["age"] > 30, "salary"] = 0`
- C) `subset = df[df["age"] > 30]; subset["salary"] = 0`
- D) `df[df["age"] > 30]["salary"] = 0`

---

**Q4.** You have a column read from CSV that came back as `dtype: object` and contains strings like `"85,000"`, `"NA"`, and `"unknown"`. Which is the most appropriate cleaning step before `pd.to_numeric`?

- A) `df["salary"].astype(float)` directly.
- B) `df["salary"].apply(lambda s: float(s.replace(",", "")))`.
- C) Re-read the file with `na_values=["NA", "unknown"]`, then `.str.replace(",", "")`, then `pd.to_numeric(..., errors="coerce")`.
- D) Drop the column entirely; it cannot be repaired.

---

**Q5.** Two tables `L` (3 rows, keys `a, b, c`) and `R` (3 rows, keys `b, c, d`) are merged on the shared key. What is the row count for `L.merge(R, on="k", how="outer")`?

- A) 2
- B) 3
- C) 4
- D) 6

---

**Q6.** You write `df.merge(other, on="user_id", how="left")` and the resulting DataFrame has 1.5× the row count of `df`. What is the most likely cause?

- A) The merge is broken in pandas 2.x; use `concat`.
- B) `other` has duplicate `user_id` rows, producing a many-to-many join.
- C) `how="left"` always inflates rows; use `how="inner"`.
- D) The `user_id` columns have different dtypes on the two sides.

---

**Q7.** Which `groupby` form produces a DataFrame with three columns named `revenue`, `n_orders`, and `n_reps`?

- A) `df.groupby("region")["revenue", "n_orders", "n_reps"].sum()`
- B) `df.groupby("region").agg({"revenue": "sum", "n_orders": "sum", "n_reps": "sum"})`
- C) `df.groupby("region").agg(revenue=("revenue", "sum"), n_orders=("order_id", "count"), n_reps=("salesrep", "nunique"))`
- D) `df.groupby("region").transform("sum")`

---

**Q8.** You want, for each row of `df`, the *total* revenue of the row's region (so you can compute "share of region"). Which call gives a Series the same length as `df`?

- A) `df.groupby("region")["revenue"].sum()`
- B) `df.groupby("region")["revenue"].transform("sum")`
- C) `df.groupby("region")["revenue"].apply(sum)`
- D) `df.groupby("region")["revenue"].agg("sum")`

---

**Q9.** You write `df.apply(lambda r: r["a"] + r["b"], axis=1)` on a 10-million-row DataFrame and it takes 60 seconds. What is the right rewrite?

- A) Use `df.swifter.apply(...)` — it parallelizes `apply`.
- B) `df["a"] + df["b"]` — operator-level arithmetic on `Series`.
- C) `df[["a", "b"]].sum(axis=1)` is the same thing in pandas.
- D) Both B and C are valid; B is the clearest.

---

**Q10.** Which of these is **honest** use of `.apply` (not a slow loop in disguise)?

- A) `df["age"].apply(lambda a: "minor" if a < 18 else "adult")`
- B) `df["email"].apply(lambda s: s.split("@")[1])`
- C) `df.groupby("user_id").apply(lambda g: np.polyfit(g["t"], g["y"], 1)[0])`
- D) `df["price"].apply(lambda p: p * 1.08)`

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — A `DataFrame` is a dict of `Series` (one per column), each with its own dtype, all sharing one row `Index`. It is *not* a 2-D NumPy array (A); that would force a single dtype across all columns. It is not a list of dicts (C); the storage is columnar. It is not an in-memory SQL table (D), though the analogy is close enough for marketing.

2. **C** — `.loc` is **label-based**. The labels in the index are `100, 200, 300`. `0` is not among them, so it raises `KeyError`. If you wanted "the row at position 0", you would write `df.iloc[0]`. Mixing the two is the most common beginner crash.

3. **B** — `df.loc[mask, col] = value` is one operation, unambiguous, and the canonical spelling. The other three are *chained indexing* — two operations where the first might return a view or a copy — which is exactly what `SettingWithCopyWarning` warns about. Even when they happen to work, they are forbidden style.

4. **C** — Encode the missing tokens at read time (`na_values=`), then handle the format with `.str.replace`, then convert with `errors="coerce"` so any remaining garbage becomes `NaN`. Option A would raise on the first comma; option B is a slow `apply` and crashes on `NaN`; option D throws away data unnecessarily.

5. **C** — `outer` is the union of the two key sets: `{a, b, c} ∪ {b, c, d} = {a, b, c, d}` = 4 rows. `inner` would be the intersection (`{b, c}` = 2 rows); `left` would be 3; `right` would be 3. Knowing the row count *before* you run the merge is the discipline that saves the day.

6. **B** — A many-to-many merge multiplies row counts. The single most common cause is a duplicate row on the *right* side of a `left` merge. Pass `validate="many_to_one"` to make pandas raise when this happens, and `indicator=True` to count slices afterwards. (Option D — dtype mismatch — would yield zero matches, not extra rows.)

7. **C** — Named aggregation (`new_col=(source_col, func)`) is the clean spelling for "different aggregations on different columns, with explicit output names". Option A's syntax does not work the way it looks. Option B works but loses the `n_orders` and `n_reps` distinctions because it would aggregate columns of those names if they existed, not compute them from `order_id` and `salesrep`. Option D returns a same-shape DataFrame, not a per-group summary.

8. **B** — `transform("sum")` returns a `Series` the same length as the input, with each row replaced by the per-group result. `sum()` (A) returns one value per group. `apply(sum)` (C) returns one value per group (and is slower). `agg("sum")` (D) is the same as `sum()` for a single function.

9. **D** — Both B (`df["a"] + df["b"]`) and C (`df[["a", "b"]].sum(axis=1)`) are vectorized and return the same numbers; B is the canonical spelling because it expresses the intent in a single line of arithmetic. Option A (swifter) is a real library but is not the right answer here — the underlying operation is so cheap when vectorized that there is no parallelism to extract.

10. **C** — Fitting a regression per group does real work per call; the `apply` overhead is negligible compared to `np.polyfit`. Options A, B, D are all classic slow-`apply` patterns that have one-line vectorized rewrites (`np.select` / `pd.cut`; `.str.split("@").str[1]`; `df["price"] * 1.08`).

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
