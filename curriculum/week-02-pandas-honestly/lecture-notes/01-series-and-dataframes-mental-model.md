# Lecture 1 — Series, DataFrame, and the Mental Model that Saves You

> **Outcome:** You can describe a `Series` as a NumPy array plus an index and a `DataFrame` as a dict of `Series` that share one index; you pick `.loc` vs `.iloc` without thinking; you know what `SettingWithCopyWarning` is warning about and how to write the code that does not trigger it.

Week 1 left you with one object: the `ndarray`. This lecture adds two more — `Series` and `DataFrame` — and we are going to be careful to say exactly what they add on top of the array, and what they do not add.

We target **pandas 2.x** throughout (2.0 shipped 2023-04, made `pyarrow`-backed strings opt-in; the current minor in 2026 is 2.3, with `copy_on_write=True` the default from 3.0 onward). Where 1.x / 2.x behavior differs we will say so.

---

## 1. The `Series`: a NumPy array, plus a label per element

Open a REPL. Build one of each.

```python
>>> import numpy as np
>>> import pandas as pd
>>> pd.__version__
'2.3.0'

>>> a = np.array([10, 20, 30, 40])
>>> a
array([10, 20, 30, 40])

>>> s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"], name="x")
>>> s
a    10
b    20
c    30
d    40
Name: x, dtype: int64
```

What did `Series` add over `array`? Three things:

1. An **index** — one label per element. The labels can be strings, ints, datetimes, anything hashable.
2. A **name** — a single string the `Series` carries with it. When you put many `Series` side by side, the name becomes the column name.
3. A **set of methods** that respect the index: alignment on arithmetic, `.loc`, `.rename`, the `.str` and `.dt` accessors.

Under the hood:

```python
>>> s.values            # the underlying NumPy array
array([10, 20, 30, 40])
>>> s.index             # the index object
Index(['a', 'b', 'c', 'd'], dtype='object')
>>> s.dtype
dtype('int64')
```

That `s.values` is the same buffer you would have in NumPy. Arithmetic operations on `Series` go through the same C ufuncs you used last week. The label work is in Python; the math is in C.

### Index alignment — the feature pandas wears on its sleeve

Two `Series` with different indices added together align on labels, not positions:

```python
>>> s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
>>> s2 = pd.Series([10, 20, 30], index=["b", "c", "d"])
>>> s1 + s2
a     NaN
b    12.0
c    23.0
d     NaN
dtype: float64
```

Where the labels matched, addition happened. Where one side was missing, the result is `NaN`. This is the feature pandas was built to deliver — the same idea as a SQL join, scaled down to one column. It is also the feature that bites beginners who assume positional addition.

If you want positional addition, drop the index: `s1.values + s2.values`. Use that as a forcing function — if you can write the same expression cleanly without the index, the index was not doing work.

---

## 2. The `DataFrame`: a dict of `Series` that share one index

```python
>>> df = pd.DataFrame({
...     "name":   ["Ada", "Bo", "Cy", "Dee"],
...     "age":    [34, 27, 41, 22],
...     "salary": [85000.0, 62000.0, 110000.0, 48000.0],
...     "active": [True, True, False, True],
... })
>>> df
  name  age    salary  active
0  Ada   34   85000.0    True
1   Bo   27   62000.0    True
2   Cy   41  110000.0   False
3  Dee   22   48000.0    True
```

This object is a **dict of `Series` that all share the same index** (here, the default `RangeIndex(0, 4)`). Each column has its own `dtype`. Each column is one contiguous C buffer of that dtype. The index is shared.

You can verify:

```python
>>> df.dtypes
name       object
age         int64
salary    float64
active       bool
dtype: object
>>> df["age"].values is df["age"].to_numpy()    # depends on storage; both reach the same buffer
>>> df.columns
Index(['name', 'age', 'salary', 'active'], dtype='object')
>>> df.index
RangeIndex(start=0, stop=4, step=1)
```

That `dtype: object` on the `name` column is one of the things this week will repair: in pandas 2.x there is a much better choice (`string[pyarrow]`), discussed in §4.

The mental picture to keep:

```
DataFrame
├── index      (one shared Index of length N)
├── columns    (an Index of length C — the column labels)
└── columns dict
    ├── "name"   →  Series of object,        length N
    ├── "age"    →  Series of int64,         length N
    ├── "salary" →  Series of float64,       length N
    └── "active" →  Series of bool,          length N
```

A `DataFrame` is **not** a 2-D array. It is one shared `Index` plus a per-column array, each with its own dtype. That distinction is what lets one column be `datetime64[ns, UTC]` while the next is `category` and the next is `float64`. NumPy cannot do that; pandas can.

This is also the reason `df.values` (which would force everything to one dtype, usually `object`) is almost always the wrong call. Use `df.to_numpy()` if you must, with `dtype=` chosen explicitly.

---

## 3. The five-minute first look

Whenever a DataFrame lands on your machine, run this exact sequence before anything else.

```python
df = pd.read_csv("trips.csv")
df.shape                          # (1_482_312, 19)
df.info(memory_usage="deep")      # dtypes + memory
df.head()                         # first 5 rows
df.tail()                         # last 5 rows — does it look the same as head?
df.describe(include="all")        # summary stats, including object columns
df.isna().sum().sort_values()     # missing counts per column
df.dtypes.value_counts()          # how many cols of each dtype
```

What you are looking for, in five minutes:

- **Shape.** Did you load the right file? Off by 10×? Wrong file.
- **Memory.** `info(memory_usage="deep")` includes the actual size of object-dtype columns (the cheap version uses pointer size). If your 200 MB CSV loaded to 2 GB, something is wrong.
- **`object` columns.** Any column you expected to be numeric or datetime that came back as `object` has dirty data inside (a stray "N/A", a comma in a number, a date in two formats).
- **Missing.** Which columns are missing how much? Often missingness is *the* signal.
- **Range.** Does `age` go up to 230? Does `salary` include `-1`? Negative ages and sentinel salaries are how real datasets encode missing data.

Five minutes. Every dataset, every time. The cost of skipping this step is hours of debugging later.

---

## 4. dtypes in pandas 2.x

This is the section that changes between books. pandas 1.x defaults left a lot of dtypes wrong (text as `object`, ints with missing values promoted to `float64`). pandas 2.x added PyArrow-backed dtypes that fix most of this. By 2026, real code should be using them.

### The dtypes you will see

| Dtype | Lives in column when |
|-------|-----------------------|
| `int64` | Pure integers, no missing values |
| `Int64` (capital I) | Integers with missing values (nullable integer) |
| `float64` | Real numbers; also integers that *had* missing values, in older code |
| `bool` | True/False, no missing values |
| `boolean` | True/False with missing values (nullable boolean) |
| `datetime64[ns]` | Naive datetimes (no timezone) |
| `datetime64[ns, UTC]` | Tz-aware datetimes |
| `timedelta64[ns]` | Durations |
| `category` | A fixed, small set of values (states, country codes, gender labels) |
| `string[pyarrow]` | Text, the modern default (pandas 2.x) |
| `object` | Anything else — the dtype of last resort |

Three rules that catch 90% of real bugs:

**(a) `object` is almost always a bug.** A column of city names should be `string[pyarrow]`, not `object`. A column of `datetime`s should be `datetime64[ns]`, not `object`. When you see `object` in `df.dtypes`, ask why.

```python
>>> df["city"] = df["city"].astype("string[pyarrow]")
>>> df["signup_at"] = pd.to_datetime(df["signup_at"], errors="coerce")
```

**(b) Integers + missing values force a choice.** Classic pandas converts `[1, 2, NaN]` to `float64`, silently. That is fine for a model input; it is a disaster for an ID column. Use the nullable integer dtype:

```python
>>> pd.Series([1, 2, pd.NA], dtype="Int64")
0       1
1       2
2    <NA>
dtype: Int64
```

Note the capital `I` — that is the nullable integer extension dtype, not the underlying NumPy `int64`. They print very similarly, which is annoying. `df.dtypes` will tell you which one you have.

**(c) Categoricals save real memory.** A column with 10 million rows but only 50 distinct state codes:

```python
>>> df["state"] = df["state"].astype("category")
>>> df["state"].nbytes
# typically 10–50× smaller than the string version
```

The `category` dtype stores each row as an `int8`/`int16` code into a small lookup table. Group-bys over a `category` column are also much faster.

> **pandas 2.x note.** The `string[pyarrow]` dtype is opt-in via `astype` today and will become the default in pandas 3.0 (date TBD). To opt in globally today: `pd.options.future.infer_string = True`. It cuts memory roughly in half versus `object` strings and is faster for nearly every string operation.

### The `errors=` keyword

When converting dtypes, pandas gives you three choices for "what if a value won't convert":

```python
pd.to_numeric(s, errors="raise")     # default: raise on failure
pd.to_numeric(s, errors="coerce")    # set to NaN on failure
pd.to_numeric(s, errors="ignore")    # return the original Series, deprecated in 2.2
```

`errors="coerce"` is the move 95% of the time during cleaning: convert what you can, set the rest to missing, then look at what is missing and decide. We will use it repeatedly in Lecture 2 and the exercises.

---

## 5. `.loc` vs `.iloc`: label vs position, and why everyone gets this wrong

This is the single most common pandas beginner crash. We will spell it out so it never happens to you.

There are two indexers:

- **`.loc[row_label, col_label]`** — by **label**. The row labels are whatever is in the index (strings, datetimes, ints — whatever). The column labels are the column names.
- **`.iloc[row_position, col_position]`** — by **integer position**. Always 0-based, like NumPy.

The confusion comes when the index *is* integers but **not** `0, 1, 2, ...`:

```python
>>> df = pd.DataFrame({"x": [10, 20, 30]}, index=[100, 200, 300])
>>> df
       x
100   10
200   20
300   30

>>> df.loc[100]      # row labelled 100
x    10
Name: 100, dtype: int64

>>> df.iloc[0]       # row at position 0
x    10
Name: 100, dtype: int64

>>> df.loc[0]        # KeyError: 0 is not a label in the index
```

Same row, two ways to find it. Get this wrong on a DataFrame whose index is a `DatetimeIndex` and you will spend an afternoon debugging.

### The naked `df[...]` form

`df[col]` and `df[col_list]` work on columns. `df[slice]` works on rows. But mixing them is a slow path to confusion. The rule for new code:

> **For setting and for ambiguous reads, always use `.loc` or `.iloc`.**

```python
df["age"]                          # one column, returns Series — fine
df[["age", "salary"]]              # two columns, returns DataFrame — fine
df[df["age"] > 30]                 # row filter, returns DataFrame — fine
df.loc[df["age"] > 30, "salary"]   # filter rows AND pick column — use .loc
df.loc[0, "age"] = 99              # set one cell — use .loc
```

The shape this rule has: **reads of one column** with `df["col"]` are fine. **Anything with a mask plus a column** uses `.loc`. **Anything setting a value** uses `.loc`. Stop here and the next section never bites you.

### Slicing semantics: a pandas gotcha

`.loc` slices are **inclusive of the end**. `.iloc` slices are **half-open**, like Python.

```python
>>> df = pd.DataFrame({"x": [10, 20, 30, 40]}, index=["a", "b", "c", "d"])
>>> df.loc["a":"c"]            # INCLUDES 'c'
    x
a  10
b  20
c  30

>>> df.iloc[0:3]               # EXCLUDES position 3
    x
a  10
b  20
c  30
```

Why? Because for label-based slicing it would be very strange to ask for "from `a` to `c`" and not get `c`. Different convention, same library. Live with it.

---

## 6. `SettingWithCopyWarning`: the warning everyone hits, and what it is actually saying

You will see this:

```
SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer, col_indexer] = value instead.
```

It is not a bug. It is pandas telling you: "the operation you wrote *might* not modify the DataFrame you think it is modifying, because it is ambiguous whether your slice was a view or a copy."

The classic pattern that triggers it:

```python
>>> # chained indexing: df["col"] returns a Series; [mask] indexes into that Series.
>>> df["salary"][df["age"] > 30] = 0
SettingWithCopyWarning: ...
```

This is two separate operations: first `df["salary"]` (which pandas might return as a view or a copy of the salary column; you cannot tell from outside), then `[df["age"] > 30] = 0` on whatever that was. The semantics are ambiguous; the warning is real.

The fix is a single combined `.loc` call:

```python
>>> df.loc[df["age"] > 30, "salary"] = 0
```

This is one operation. pandas knows exactly which rows and which column you mean. No ambiguity, no warning, and it does what you intended.

The general rule:

| Bad | Good |
|-----|------|
| `df["col"][mask] = value` | `df.loc[mask, "col"] = value` |
| `df[df.x > 0]["y"] = 1` | `df.loc[df.x > 0, "y"] = 1` |
| `subset = df[df.x > 0]; subset["y"] = 1` | `df.loc[df.x > 0, "y"] = 1` |

> **pandas 3.0 note (coming).** In pandas 3.0, `copy_on_write=True` is the default. That eliminates the warning's *cause*: every "slice" is conceptually its own copy until you write to it, at which point pandas does the right thing. Until you upgrade — and most teams will be on 2.x for a while yet — write `.loc[mask, col] = value` and you will never see the warning.

---

## 7. The index, and why you should care about it

Many pandas operations are *much* faster when the index is the thing you are looking up by. Setting the index well is half the trick.

```python
>>> df = df.set_index("user_id")        # row labels are now user_id values
>>> df.loc[42_000]                       # O(1) ish lookup
```

You can have a `MultiIndex` (multiple columns as the row label):

```python
>>> sales = sales.set_index(["region", "month"])
>>> sales.loc[("Northeast", "2025-08")]  # both levels
```

Three honest cautions:

1. **The default `RangeIndex` is fine** for most exploratory work. Don't set an index unless you have a reason.
2. **Indices must be unique for `loc` lookups to behave like dictionary lookups.** A non-unique index returns a `Series` or `DataFrame` of all matching rows, not a single row. `df.index.is_unique` will tell you.
3. **`reset_index()` undoes it.** When in doubt, reset and start over.

---

## 8. The mental model to walk away with

After this lecture you should be able to draw this:

```
┌──────────────────────────────────────────────────────┐
│ DataFrame                                             │
│                                                       │
│  index  (Index of length N — shared by every column) │
│                                                       │
│  ┌──────────┬──────────┬──────────┬──────────┐       │
│  │ col_a    │ col_b    │ col_c    │ col_d    │       │
│  │ Series   │ Series   │ Series   │ Series   │       │
│  │ int64    │ float64  │ string   │ datetime │       │
│  │  …N rows │  …N rows │  …N rows │  …N rows │       │
│  └──────────┴──────────┴──────────┴──────────┘       │
└──────────────────────────────────────────────────────┘
```

One shared index. Each column its own typed array. Reads with `df["col"]` or `df.loc[mask, "col"]`. Writes with `df.loc[mask, "col"] = value`. Dtypes are honest — `object` is a sign something is wrong.

Every later week of C5 — joining tables in Lecture 2, group-by in Lecture 2, the `apply` trap in Lecture 3, feature engineering in Week 4 — sits on this picture. Get this picture right and the rest is vocabulary.

---

## 9. Self-check

Without re-reading:

1. State, in one sentence each, what a `Series` adds on top of an `ndarray`, and what a `DataFrame` adds on top of a dict of `Series`.
2. Given `s1 = pd.Series([1,2,3], index=["a","b","c"])` and `s2 = pd.Series([10,20,30], index=["b","c","d"])`, what is `s1 + s2`?
3. You see `df["city"]` has `dtype: object`. Name two reasons that is bad and one fix.
4. The index of `df` is `[100, 200, 300]`. What does `df.loc[0]` do? What does `df.iloc[0]` do?
5. `df.loc["a":"c"]` versus `df.iloc[0:3]` on a four-row DataFrame `["a","b","c","d"]`: which includes the row at position 2 and which includes the row labelled `"c"`? (Trick question.)
6. Rewrite `df["salary"][df["age"] > 30] = 0` without triggering `SettingWithCopyWarning`.
7. You have 10 million rows and a column that takes 50 distinct string values. What dtype reduces memory and speeds up group-by?
8. Why does `pd.to_numeric(s, errors="coerce")` show up in nearly every cleaning script?

---

## Further reading

- **pandas — Indexing and selecting data**: <https://pandas.pydata.org/docs/user_guide/indexing.html>
- **pandas — Working with missing data**: <https://pandas.pydata.org/docs/user_guide/missing_data.html>
- **pandas — Copy-on-Write** (the 3.0 default): <https://pandas.pydata.org/docs/user_guide/copy_on_write.html>
- **PyArrow-backed dtypes**: <https://pandas.pydata.org/docs/user_guide/pyarrow.html>
- **Tom Augspurger, *Modern Pandas*, part 1 ("Indexing")**: <https://tomaugspurger.net/posts/modern-1-intro/>

Next: [Lecture 2 — Merge, group-by, aggregate](./02-merge-groupby-aggregate.md).
