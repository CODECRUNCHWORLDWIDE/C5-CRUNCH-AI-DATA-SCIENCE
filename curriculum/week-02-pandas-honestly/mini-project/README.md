# Mini-Project — A 2-Page EDA Report on a Messy Public Dataset

> Clean and explore a real, messy public dataset and produce a two-page exploratory data analysis report (a Jupyter notebook exported to HTML and to a PDF-suitable Markdown). The deliverable is a reviewer-readable artifact, not a stack of cells.

This is the second artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 1 was an image notebook; Week 2 is an EDA. Recruiters skim the EDA carefully — it is the closest of any portfolio piece to what an analyst job actually looks like.

**Estimated time:** 7 hours, spread across Thursday–Saturday.

---

## What you will build

A Jupyter notebook `eda.ipynb` plus a rendered `report.md` that, on one real public dataset:

1. **Loads** the dataset with deliberate `read_csv` / `read_parquet` arguments.
2. **Documents** the data's provenance, license, and known quirks in a header cell.
3. **Runs the five-minute first look** — shape, dtypes, memory, missing-value counts.
4. **Cleans** every column to an honest dtype. Handles missing values, encoded sentinels, and mixed formats. No `object` columns left except free-text fields stored as `string[pyarrow]`.
5. **Asks and answers three specific questions** of the data using `groupby` / `merge` / `pivot`. Each answer is one or two paragraphs of prose and one small table or chart.
6. **Ends with a "what this dataset cannot tell you" section** — three honest limitations of the data and the analysis.

The notebook is the working artifact. The two-page `report.md` is the executive summary a non-technical reader can finish in five minutes.

---

## Pick a dataset

Choose **one** of these public datasets. All are free, all are real, all have realistic dirt.

- **NYC 311 service requests** (CSV; multi-year, ~10–20 GB total; pick one borough × one month for a manageable subset):
  <https://data.cityofnewyork.us/Social-Services/311-Service-Requests/erm2-nwe9>
- **NYC TLC green-taxi trip records** (Parquet, monthly; ~10–20 MB per month):
  <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- **City of Chicago crimes** (CSV; the "one year prior to present" subset is manageable):
  <https://data.cityofchicago.org/Public-Safety/Crimes-One-year-prior-to-present/x2n5-8w5q>
- **Our World in Data — COVID-19** (clean, multi-table — good for the merge work):
  <https://github.com/owid/covid-19-data/tree/master/public/data>
- **UK Office for National Statistics — UK road safety data** (CSV; classic teaching dataset):
  <https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data>

If you pick something else (you may), it must be **public**, **free**, **≥ 10 columns and ≥ 50,000 rows**, and have at least one of: missing values, mixed-format dates, encoded sentinels, multi-table joins. Document the choice and the URL in the notebook's first cell.

---

## Acceptance criteria

- [ ] A new directory `week-02/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `pandas>=2.2,<3`, `pyarrow>=15`, `matplotlib>=3.8`, `jupyter`.
- [ ] `jupyter nbconvert --to notebook --execute week-02/eda.ipynb` runs end-to-end without errors on a fresh clone. (If the raw dataset is too big to ship, include a small sampled copy in `week-02/data/` and have the notebook fall back to it; document this in the notebook.)
- [ ] **Zero** uses of `.apply` in the notebook. Every transformation is vectorized.
- [ ] **Every** column in the cleaned DataFrame has an honest dtype: datetimes, categoricals, nullable ints, `string[pyarrow]`. No bare `object` columns.
- [ ] **Every** `merge` call passes `how=`, `validate=`, and `indicator=`.
- [ ] The notebook prints, for every transition cell, the shape and the dtype map *or* a short comment naming them. Discipline: name your shapes.
- [ ] The notebook is committed with **outputs cleared**, and a separate `eda.html` export is committed for preview.
- [ ] A `report.md` (~2 pages, 600–900 words) summarizes the dataset, the three questions, the three answers, and the three limitations. A non-technical reader should be able to finish it in five minutes.
- [ ] A `README.md` in `week-02/` explains: setup, data source URL and license, file layout, how to reproduce.

---

## Suggested layout

```
crunch-ai-portfolio-<yourhandle>/
├── README.md                    ← portfolio root
└── week-02/
    ├── README.md                ← week-2 explainer
    ├── requirements.txt
    ├── data/
    │   └── sample.csv           ← small subset, committed
    ├── eda.ipynb
    ├── eda.html                 ← rendered preview
    └── report.md                ← 2-page executive summary
```

---

## Suggested order of operations

### Phase 1 — Setup and provenance (30 min)

1. Open `week-02/eda.ipynb`. The first markdown cell is the **dataset card**, with: name, source URL, license, the date you downloaded it, the size on disk, and a one-sentence statement of what each row represents.
2. The next code cell does the imports and prints `pd.__version__`. Pin pandas 2.2+ in your `requirements.txt`.
3. Save a small representative sample (10,000 rows is plenty) into `data/sample.csv` so the notebook is reproducible from a fresh clone without re-downloading gigabytes. Document this.

### Phase 2 — The five-minute first look (45 min)

A section titled "First look." One code cell that runs, in this order:

```python
df.shape
df.head()
df.info(memory_usage="deep")
df.describe(include="all")
df.isna().sum().sort_values(ascending=False)
df.dtypes.value_counts()
```

A markdown cell after each block of output, in your own words, naming what you found. Be specific. "10% of `dropoff_latitude` is missing — those are the trips that ended outside the geofence" beats "there is some missing data."

### Phase 3 — Cleaning (1.5 h)

A section titled "Cleaning." For each column that needs work, one code cell and one markdown cell explaining what you did and why. Cover at minimum:

- Datetimes: `pd.to_datetime` with `errors="coerce"` and a timezone if applicable.
- Numerics with encoded missing values: `pd.to_numeric(errors="coerce")` after stripping thousand-separators or other formatting.
- Categoricals: `astype("category")` for any column with a small, fixed value set.
- Strings: `astype("string[pyarrow]")` for free-text columns you keep.
- Nullable integers: `astype("Int64")` (capital I) for ID columns that have missing values.

End the cleaning section with a single cell that prints the final dtype map and asserts no column is `object`:

```python
assert not (df.dtypes == "object").any(), (
    "object columns remain: "
    + df.dtypes[df.dtypes == "object"].index.tolist().__str__()
)
print(df.dtypes)
```

If your dataset has more than one table (e.g. trips + zones), do a clean `merge` here. Pass `how=`, `validate=`, `indicator=`. Drop the indicator column once you have logged its `value_counts()`.

### Phase 4 — Three questions (2 h)

The heart of the project. Pick **three** specific questions you can answer with this dataset. They should be:

- **Concrete** — not "is the data interesting?" but "is the average trip distance in Manhattan higher than in Brooklyn at 8 am on weekdays?"
- **Answerable** — the data has the columns you need to answer them.
- **Independent** — the three answers should illuminate three different aspects of the data.

For each question:

1. A markdown cell with the question, two sentences on why it matters, and (honestly) what you expect the answer to be before you compute it.
2. A code cell that computes the answer with `groupby` / `merge` / `pivot_table` / `resample`. No `.apply`.
3. A code cell that produces a small matplotlib chart (one chart, simply labeled — Week 3 will deepen this).
4. A markdown cell with the answer in plain English, including the numbers. Numbers cited, not asserted.

Example question pool (use these or come up with your own):

- Which complaint type in 311 has the worst response time, and by how much?
- Is the tip percentage on green-taxi trips higher for credit-card payments than for cash? By how much?
- Did COVID case-fatality rates in country X improve between the first and the second wave? By how much?
- Which Chicago district has the highest theft rate per capita? (You will need a population join.)
- Does road accident severity correlate with time of day in the UK dataset?

### Phase 5 — Limitations, and export (1 h)

A final section titled "What this dataset cannot tell you." Three honest limitations, one paragraph each. Examples:

- "311 data records *reports*, not *incidents*. Underreporting in low-income neighborhoods has been documented (see Levy and Mattsson, 2023) and we make no correction for it."
- "Trip distance in the TLC data is the meter's running total, which includes detours. A missing dropoff GPS does not mean a missing trip; it means an off-grid trip."
- "COVID counts are *reported* cases, dependent on a country's testing capacity. Cross-country comparisons of case-fatality ratios conflate disease severity with testing policy."

This is the C5 brand voice: "Acknowledge bias and limitations every time we ship a model." Even though this is EDA, not a model, the discipline is the same.

Then export:

```bash
jupyter nbconvert --to html eda.ipynb
```

Write `report.md`: 600–900 words. Sections: **Dataset**, **Cleaning**, **Q1**, **Q2**, **Q3**, **Limitations**. No code. Cite numbers, not adjectives. Imagine an executive who has five minutes and will not open the notebook.

---

## Stretch goals

- Re-run the same `groupby` and `merge` in DuckDB (one SQL string) and confirm the numbers match. Note the wall-clock difference. This is the Lecture 3 comparison made real.
- Re-run a single hot path in Polars and report the speed-up.
- Add a `pandera` schema for your cleaned table and run it as a validation step. Future-you on a different dataset version will thank present-you.
- Publish `report.md` as a GitHub Pages page so the link in your portfolio README points directly at a readable HTML version.

---

## Rubric

| Criterion | Weight | "Great" looks like |
|-----------|------:|--------------------|
| It runs | 20% | Fresh clone → `pip install -r requirements.txt` → `jupyter nbconvert --execute` → no errors |
| Cleaning discipline | 25% | Zero `object` columns; every dtype is honest; cleaning is documented step by step |
| Vectorization discipline | 15% | Zero `.apply`; every transformation is `.str` / `.dt` / arithmetic / `np.where` / `merge` |
| Joins | 10% | Every merge has `how=`, `validate=`, `indicator=` and a row-count assertion afterward |
| Questions & answers | 15% | Three concrete questions, three concrete numerical answers, one chart each |
| Report readability | 10% | The 2-page `report.md` could go to an executive without explanation |
| Limitations | 5% | Three honest limitations naming specific biases or data gaps |

---

## What this exercises (and what comes next)

This mini-project exercises every concept from Week 2:

- The five-minute first look.
- dtype discipline (pandas 2.x styles).
- `.loc` / `.iloc` for safe writes.
- `merge` with `validate=` and `indicator=`.
- `groupby` with named aggregation and `transform`.
- `pivot_table` / `melt` to reshape on purpose.
- The `apply` trap — by writing none.

Week 3 will take the three charts you made here and ask whether they are honest. The same cleaning artifact you ship this week will power Week 4's regression. Doing the EDA cleanly now saves a week of debugging downstream.

---

## Submission

Push your `crunch-ai-portfolio-<yourhandle>` repo. Open the `week-02/eda.html` link from the repo README so a reviewer with no Python install can read your work. Paste the URL into your Week 2 review thread (or the C5 community channel if you are doing this in a cohort). That is the artifact recruiters will see.
