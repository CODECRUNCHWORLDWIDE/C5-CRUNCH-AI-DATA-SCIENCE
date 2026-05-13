# Week 2 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **pandas — "10 minutes to pandas"** (the canonical first hour):
  <https://pandas.pydata.org/docs/user_guide/10min.html>
- **pandas — Indexing and selecting data** (`.loc` vs `.iloc` lives here):
  <https://pandas.pydata.org/docs/user_guide/indexing.html>
- **pandas — Merge, join, concatenate** (the chapter the whole world should read once):
  <https://pandas.pydata.org/docs/user_guide/merging.html>
- **pandas — Group by: split-apply-combine**:
  <https://pandas.pydata.org/docs/user_guide/groupby.html>
- **pandas — Working with missing data**:
  <https://pandas.pydata.org/docs/user_guide/missing_data.html>

## The official docs you will bounce between all week

- **pandas 2.x user guide**: <https://pandas.pydata.org/docs/user_guide/index.html>
- **pandas API reference**: <https://pandas.pydata.org/docs/reference/index.html>
- **pandas 2.0 release notes** (the PyArrow string transition is the big one): <https://pandas.pydata.org/docs/whatsnew/v2.0.0.html>
- **`read_csv` reference page** (it has dozens of arguments; you will use ~six of them weekly): <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>
- **`merge` reference page** (every keyword you will ever need is here): <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html>

## Free books (full texts, legally free)

- **Wes McKinney — *Python for Data Analysis*, 3rd ed.** (the pandas author; 3rd edition is free online and uses pandas 2.x):
  <https://wesmckinney.com/book/>
  Chapters 5 (intro), 7 (cleaning), 8 (joins, group-by), and 10 (group-by deep dive) are exactly this week.
- **Jake VanderPlas — *Python Data Science Handbook*** (free online, MIT-licensed):
  <https://jakevdp.github.io/PythonDataScienceHandbook/>
  Chapter 3 is "Data Manipulation with pandas." Lighter touch than McKinney, good for a second pass.
- **Tom Augspurger — *Modern Pandas*** (a seven-part blog series; free):
  <https://tomaugspurger.net/posts/modern-1-intro/>
  Written by a pandas core dev. Opinionated, current, focused on idioms over features.

## The data sources we use this week

All public, all free to download:

- **NYC Taxi & Limousine Commission** trip records (Parquet, monthly partitions):
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
  We use the green-taxi files because they are smaller (~10–20 MB per month) than yellow.
- **NYC Open Data — 311 Service Requests** (CSV, free, very messy):
  <https://data.cityofnewyork.us/Social-Services/311-Service-Requests/erm2-nwe9>
- **City of Chicago — Crimes** (CSV, large):
  <https://data.cityofchicago.org/Public-Safety/Crimes-One-year-prior-to-present/x2n5-8w5q>
- **Our World in Data — COVID dataset** (clean, well-documented, multi-table — perfect for the merge challenge):
  <https://github.com/owid/covid-19-data/tree/master/public/data>
- **Palmer Penguins** (small, clean, ships with seaborn — useful for groupby drills):
  <https://github.com/allisonhorst/palmerpenguins>

## The math you do not need this week

You do not need new math this week. Week 2 is library proficiency. If you want to keep linear algebra warm, the MIT 18.06 lectures from Week 1 stay relevant for Week 4.

## Tools you will use this week

- **`pandas`** ≥ 2.2: `pip install "pandas>=2.2,<3"`.
- **`pyarrow`** ≥ 15 (for the PyArrow-backed string dtype and Parquet I/O): `pip install "pyarrow>=15"`.
- **`numpy`** ≥ 2 (still): `pip install "numpy>=2,<3"`.
- **`matplotlib`** for one or two quick charts: `pip install matplotlib`.
- **`pytest`** for the exercises.
- **`polars`** and **`duckdb`** are optional (Lecture 3 references them but does not require them).

A `requirements.txt` snippet for the week:

```text
pandas>=2.2,<3
pyarrow>=15
numpy>=2,<3
matplotlib>=3.8
pytest>=8
# optional, mentioned in Lecture 3:
# polars>=1.0
# duckdb>=1.0
```

## Videos (free, no signup)

- **Wes McKinney — "10 things I hate about pandas"** (PyData, ~30 min): the pandas author lists his own library's footguns. You will recognize half of them by Friday.
- **Matt Harrison — "pandas idioms"** (PyData, multiple talks): tight 20-minute clinics on `.loc`, `.assign`, and chained-method style.
- **Sebastiaan Mathôt or Corey Schafer — pandas walkthroughs** (YouTube, free): general-audience tutorials at 1.25× speed make a good background companion.

## Open-source projects to read (in this order)

You can learn more from one hour reading other people's pandas than from three hours of tutorials.

1. **pandas itself** — `pandas-dev/pandas` on GitHub. Read `pandas/core/frame.py` for `DataFrame.__init__` and `merge`:
   <https://github.com/pandas-dev/pandas>
2. **seaborn** — every plotting function does data manipulation. Skim `seaborn/_oldcore.py`:
   <https://github.com/mwaskom/seaborn>
3. **`pandera`** — schema validation for `DataFrame`s. Reading the README teaches you what people wish pandas enforced by default:
   <https://github.com/unionai-oss/pandera>

## Faster pandas alternatives (mentioned in Lecture 3)

- **Polars** — Rust-backed, columnar, lazy. The API is similar enough that pandas users pick it up in a day:
  <https://docs.pola.rs/>
- **DuckDB** — an in-process analytical SQL database that reads CSV / Parquet directly and joins faster than pandas can `merge`:
  <https://duckdb.org/docs/>
- **Modin** — a near-drop-in pandas replacement that uses Ray / Dask under the hood:
  <https://modin.readthedocs.io/>

In 2026 the rule of thumb is: write the analysis in pandas first because that is what the team reads; if the same notebook needs to be a nightly job over 50 GB, port the hot path to DuckDB or Polars.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **`Series`** | A 1-D labeled array: a NumPy array, an index, and an optional name |
| **`DataFrame`** | A 2-D labeled table: a dict of `Series` that share an index |
| **Index** | The row labels of a `Series` or `DataFrame` (numbers, strings, datetimes — anything hashable) |
| **`MultiIndex`** | A hierarchical index with multiple levels (e.g. `(year, month)`) |
| **dtype** | The element type of a `Series`: `int64`, `float64`, `bool`, `datetime64[ns]`, `category`, `string[pyarrow]`, `object` |
| **`object` dtype** | The "we gave up and stored Python objects" dtype. Slow. A sign of mixed types or unparsed strings |
| **`NaN`** | "Not a Number", the missing value for float columns |
| **`NaT`** | "Not a Time", the missing value for datetime columns |
| **`pd.NA`** | The new, dtype-agnostic missing-value sentinel (pandas 1.0+) |
| **`.loc`** | Label-based indexing: `df.loc["row_label", "col_label"]` |
| **`.iloc`** | Position-based indexing: `df.iloc[0, 0]` |
| **`merge`** | A SQL-style join of two DataFrames on one or more keys |
| **`concat`** | Stack two DataFrames vertically (rows) or horizontally (columns) |
| **`groupby`** | Split rows by one or more keys, apply a function, combine the results |
| **`agg`** | Apply one or more aggregation functions to each group |
| **`pivot` / `pivot_table`** | Reshape from long to wide |
| **`melt`** | Reshape from wide to long |
| **`apply`** | Run a Python function on each row, column, or group. Usually slow — see Lecture 3 |

---

*If a link 404s, please open an issue so we can replace it.*
