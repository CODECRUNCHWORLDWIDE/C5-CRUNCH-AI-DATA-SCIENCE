# Week 2 — pandas, Honestly

> *A `DataFrame` is a dict of `Series`. A `Series` is a NumPy array plus an index. Everything else — `merge`, `groupby`, `pivot`, `apply` — is a consequence of those two sentences.*

Welcome to week two of **C5 · Crunch AI / Data Science**. Last week you spent five days on the `ndarray`. This week we put an index on it, give it a name, hand it a label per column, and call the result a `DataFrame`. Almost nothing new happens at the bottom of the stack — the data is still a contiguous typed buffer. What changes is the *vocabulary* you have to ask questions of it.

Three caveats up front, because pandas is the library most over-sold in the data-science world:

1. **pandas is slower than NumPy** for raw numerical work. If you spend a week here and walk away thinking pandas replaced NumPy, you missed the point. pandas adds *labels* and *I/O* and *joins*. It does not replace the array.
2. **pandas is slower than DuckDB or Polars** for joins, group-bys, and analytical SQL on large data. We will use pandas anyway because it is the lingua franca; the team you join *will* hand you pandas code. We touch Polars and DuckDB at the end of Lecture 3 so you know the off-ramps exist.
3. **`DataFrame.apply` is a Python `for` loop in disguise.** Half this week is teaching you to recognize it and the other half is teaching you not to write it. By Friday, "let me just `.apply` it" should sound to you the way `for i in range(len(a)): out[i] = a[i] ** 2` sounded by the end of Week 1.

We target **pandas 2.x** throughout (pandas 2.0 shipped 2023-04, switched to PyArrow-backed strings, and made `copy_on_write=True` the default in 3.0). The current minor in 2026 is 2.3. Where 1.x / 2.x behavior differs we will flag it.

---

## Learning objectives

By the end of this week, you will be able to:

- **Describe** a `Series` as "a NumPy array plus an index" and a `DataFrame` as "a dict of `Series` that share an index" — and act on that mental model.
- **Read** a CSV / JSON / Parquet file into a `DataFrame`, inspect it with `info()`, `head()`, `describe(include="all")`, and know within five minutes what is wrong with it.
- **Choose** `.loc` (label-based) vs `.iloc` (position-based) reliably, and never write `df["col"][idx] = value` again.
- **Handle missing data** explicitly: `isna`, `dropna`, `fillna` — and know when each is the wrong choice.
- **Convert dtypes safely**: object→numeric, object→datetime (with `errors=` and timezones), the new PyArrow-backed string dtype.
- **Join two or more tables** with the right `merge(how=...)` and explain the row-count change without surprise.
- **Group, aggregate, and reshape** with `groupby`, `agg`, `pivot_table`, `melt` — and pick long vs wide on purpose.
- **Recognize an `apply`** as a Python loop and rewrite it with `.str`, `.dt`, `np.where`, arithmetic, or a `merge`.
- **Pass** every `pytest` case on the Week 2 exercises and ship the 2-page EDA mini-project.

---

## Prerequisites

- **Week 1 complete.** You should not still be Googling `.shape` vs `.size`. The image-filter mini-project should be pushed to your portfolio.
- A working **Python 3.11+** install (we use 3.12).
- pandas 2.x: `pip install "pandas>=2.2,<3"`.
- pyarrow (for the new string dtype and Parquet I/O): `pip install "pyarrow>=15"`.
- matplotlib (for one quick chart in the mini-project): `pip install matplotlib`.
- `pytest` for the exercise smoke tests.

No GPU. No notebook required for the exercises (they are `.py` files); the mini-project ships as a notebook.

---

## Topics covered

- The `Series`: NumPy array + index + name. Why the index matters.
- The `DataFrame`: a dict of `Series` that share an index.
- dtypes in pandas 2.x: `int64`, `float64`, `bool`, `datetime64[ns, tz]`, `category`, and the **new** PyArrow-backed `string[pyarrow]`. Why mixed dtypes (`object`) silently hide bugs.
- `.loc` vs `.iloc`: label vs position. The single most common beginner crash.
- `SettingWithCopyWarning`: what it warns about, why it exists, and how `df.loc[mask, "col"] = value` makes it go away correctly.
- Reading the world: `read_csv`, `read_json`, `read_parquet`, `read_excel`. The five `read_csv` arguments you will use every day (`dtype=`, `parse_dates=`, `na_values=`, `usecols=`, `chunksize=`).
- Missing data: `NaN`, `NaT`, `pd.NA`. `isna`, `dropna`, `fillna`. Why `df == np.nan` is always `False`.
- `merge`, `concat`, `join`: how each differs, when to reach for which.
- The four joins — `inner`, `outer`, `left`, `right` — drawn as Venn diagrams once, in code three times.
- `groupby`: split-apply-combine. `.agg`, `.transform`, `.filter`.
- Reshaping: `pivot`, `pivot_table`, `melt`, `stack`, `unstack`. Long vs wide on purpose.
- Time-series basics: `pd.to_datetime`, `.dt` accessor, `resample`, time-zone awareness.
- The **`apply` trap**: why it is slow, vectorized alternatives for 95% of cases, the 5% where `apply` is honest.
- A taste of **Polars** and **DuckDB** for when pandas is the wrong tool.

---

## Weekly schedule

Target about **36 hours**. Some sections click in fifteen minutes; some take three hours. Treat the table as a budget, not a contract.

| Day       | Focus                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Series, DataFrame, dtype, `.loc` vs `.iloc`    |    3h    |    2h     |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     7h      |
| Tuesday   | Reading messy CSV; missing data; dtype repair  |    0h    |    2h     |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     4h      |
| Wednesday | `merge`, `concat`, the four joins              |    3h    |    2h     |     2h     |    0.5h   |   1h     |     0h       |    0h      |     8.5h    |
| Thursday  | `groupby`, `agg`, `pivot`/`melt`               |    0h    |    2h     |     2h     |    0.5h   |   1h     |     2h       |    0.5h    |     8h      |
| Friday    | The `apply` trap; Polars/DuckDB taste; mini    |    2h    |    0h     |     0h     |    0.5h   |   1h     |     2h       |    0.5h    |     6h      |
| Saturday  | Mini-project polish                            |    0h    |    0h     |     0h     |    0h     |   1h     |     3h       |    0h      |     4h      |
| Sunday    | Quiz, review, push to portfolio repo           |    0h    |    0h     |     0h     |    0.5h   |   0h     |     0h       |    0h      |     0.5h    |
| **Total** |                                                | **8h**   | **8h**    | **4h**     | **3h**    | **6h**   | **7h**       | **2h**     | **38h**     |

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Free books, official docs, the Polars/DuckDB pointers |
| [lecture-notes/01-series-and-dataframes-mental-model.md](./lecture-notes/01-series-and-dataframes-mental-model.md) | Series, DataFrame, dtype, `.loc` vs `.iloc`, `SettingWithCopyWarning` |
| [lecture-notes/02-merge-groupby-aggregate.md](./lecture-notes/02-merge-groupby-aggregate.md) | `merge` / `concat` / `join`, the four joins, `groupby`, `agg`, `pivot` / `melt` |
| [lecture-notes/03-the-apply-trap-and-when-to-vectorize.md](./lecture-notes/03-the-apply-trap-and-when-to-vectorize.md) | Why `apply` is slow; vectorized rewrites; Polars and DuckDB |
| [exercises/README.md](./exercises/README.md) | Index of exercises |
| [exercises/exercise-01-load-and-inspect.py](./exercises/exercise-01-load-and-inspect.py) | Load a real public CSV; inspect with `info` / `describe`; pick the right `dtype` |
| [exercises/exercise-02-clean-the-messy.py](./exercises/exercise-02-clean-the-messy.py) | Mixed dtypes, missing data, encoding errors — repair them |
| [exercises/exercise-03-groupby-aggregate.py](./exercises/exercise-03-groupby-aggregate.py) | Group-by, multiple aggregations, pivot to wide |
| [challenges/README.md](./challenges/README.md) | Index of weekly challenges |
| [challenges/challenge-01-merge-three-tables.md](./challenges/challenge-01-merge-three-tables.md) | A three-way merge with realistic key mismatches |
| [quiz.md](./quiz.md) | 10 multiple-choice questions with an answer key |
| [homework.md](./homework.md) | Six practice problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Full spec for the 2-page EDA report on a messy public dataset |

---

## Stretch goals

- Read the [pandas 2.0 release notes](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html). Pick three breaking changes from 1.x and write each as a one-sentence migration note. The PyArrow string transition is the one most likely to bite real code.
- Skim [Modin](https://modin.readthedocs.io/) and [Polars](https://docs.pola.rs/) homepages. Write a 200-word note on when you would reach for each instead of pandas, and what you would lose.
- Reproduce one figure from [*Python for Data Analysis* by Wes McKinney](https://wesmckinney.com/book/) (the pandas author's book, free online, 3rd edition uses pandas 2.x). Chapter 10 on groupby is exactly this week.
- Look at one pandas function's source on GitHub — say, [`DataFrame.merge`](https://github.com/pandas-dev/pandas/blob/main/pandas/core/frame.py). You do not need to follow every line; just notice how many keyword arguments a real production function has, and why each exists.

---

## What you will *not* do this week

You will not:

- Train a model (Week 4).
- Make a chart for an executive (Week 3, mostly).
- Use any "AI" or "auto-EDA" library that produces pre-baked reports. We are building the muscle of looking at the data ourselves.
- Use `apply` once you understand the alternatives.

That is deliberate. Half the bugs in real ML pipelines are bad data joins and silent dtype coercions. The week you spend on pandas now is the year you do not spend debugging silent corruption later.

---

## Up next

[Week 3 — Visualization that Doesn't Lie](../week-03/) — once you have pushed your EDA report to your `crunch-ai-portfolio-<yourhandle>` repo.
