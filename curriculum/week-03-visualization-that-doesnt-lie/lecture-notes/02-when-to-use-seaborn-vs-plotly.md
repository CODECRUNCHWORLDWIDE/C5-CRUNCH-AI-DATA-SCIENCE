# Lecture 2 — When to Use Seaborn vs Plotly (and When matplotlib Alone Is Enough)

> **Outcome:** You can name, for any chart task you face this week, which of matplotlib / seaborn / plotly is the right tool, with a one-sentence reason. You stop reaching for plotly because it is shinier and start reaching for it when interactivity earns its keep.

Lecture 1 taught you matplotlib. matplotlib is the foundation of nearly every Python plotting library — seaborn calls into it, pandas' `.plot` method dispatches to it, even older Bokeh examples wrap it. The question this lecture answers is: **given you know matplotlib, when should you reach for something else?**

We target **seaborn 0.13+** and **plotly 5.x**. Both are free, open-source, pip-installable. Both are stable APIs; seaborn 0.13 (released late 2023) introduced the `seaborn.objects` interface that we touch only briefly; plotly's `plotly.express` interface has been the recommended high-level API since 4.8 (2020).

---

## 1. The honest three-library landscape

| Library | What it is | Output | Best at | Worst at |
|--------|----|----|----|----|
| **matplotlib** | The low-level drawing API | static (PNG, SVG, PDF) | every chart on Earth, with effort | quick "give me a faceted bar chart of X by Y" |
| **seaborn** | A statistical-defaults wrapper over matplotlib | static (PNG, SVG, PDF) | distribution and relationship charts of a tidy DataFrame | non-tabular data; deep customization |
| **plotly** | An interactive web charting library | HTML / JSON | charts that need hover, zoom, pan, or 3-D | static reports; PDF export; small file size |

Three rules of thumb that hold in almost every case:

1. **If the chart is for a report or a slide deck, use matplotlib (with or without seaborn for setup).** Static images, predictable file size, vector output for print.
2. **If the chart is for a web page where readers will explore, use plotly.** Hover tooltips and zoom are why plotly exists.
3. **If the chart is a quick "show me the distribution of X by Y" during EDA, use seaborn.** One line, sensible defaults, axis labels guessed from the DataFrame.

Most of this week's charts go into reports. Most of this week we use matplotlib and seaborn. Plotly gets one focused exercise.

---

## 2. seaborn in twenty minutes

Seaborn is a thin layer over matplotlib that does three things matplotlib does not:

1. **Knows about tidy DataFrames.** You pass `data=df, x="col_a", y="col_b", hue="col_c"` and seaborn figures out the rest.
2. **Picks reasonable statistical defaults.** A scatter plot includes a regression line if you ask; a histogram includes a KDE if you ask; a categorical plot stacks or dodges based on the data.
3. **Returns a matplotlib `Axes`.** Every seaborn function gives you back the Axes object so you can customize with the OO API you already know.

The five seaborn functions you use this week (and most weeks):

```python
import seaborn as sns

sns.histplot(data=df, x="trip_distance", bins=40)            # distribution of one variable
sns.boxplot(data=df, x="borough", y="fare_amount")           # distribution per group
sns.scatterplot(data=df, x="trip_distance", y="fare_amount") # relationship of two variables
sns.regplot(data=df, x="trip_distance", y="fare_amount")     # scatter + linear fit + CI
sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0)  # correlation matrix
```

Each one writes to the current matplotlib Axes (or one you pass with `ax=...`) and returns it. You can then style it like any matplotlib chart:

```python
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
sns.histplot(data=df, x="trip_distance", bins=40, color="#7C3AED", ax=ax)
ax.set_xlabel("Trip distance (miles)")
ax.set_ylabel("Trips")
ax.set_title("Most green-taxi trips are under three miles")
fig.savefig("trip_distance.png", dpi=150)
```

That `ax=ax` argument is the single most important seaborn habit. Without it, seaborn picks the "current" Axes from pyplot, which is fine for the REPL and a footgun in a multi-chart script. **Pass `ax` explicitly. Every time.**

### When seaborn earns its keep

- **`sns.pairplot(df)`** — small-multiples scatter matrix of every numeric column. Three lines in seaborn; thirty in matplotlib.
- **`sns.catplot(..., col=, row=)`** — faceted small multiples. Same idea: short in seaborn, tedious in matplotlib.
- **`sns.lmplot`** — scatter + per-group regression line. Try writing that in matplotlib by hand and you will appreciate the wrapper.

### When seaborn costs you control

- Anything where you need to draw on the chart that is not "another seaborn call." Annotations, custom guide lines, anything mathematical. Drop to matplotlib OO on the returned Axes.
- Anything where seaborn's default style fights yours. `sns.set_style("white")` and `sns.set_context("notebook")` get you close; the rest you do in matplotlib.

The honest read of seaborn: it is a **convenience** library. You can do everything seaborn does in matplotlib; seaborn is the right call for the 80% of charts where the convenience is real and the right call to skip when you need control.

---

## 3. The seven seaborn charts that cover most EDA

A `DataFrame`-first cheat sheet. Each row is "the question you have" and "the seaborn function that answers it":

| Question | Function | Notes |
|----|----|----|
| What is the distribution of one numeric column? | `sns.histplot` | Add `kde=True` for a smoothed version. |
| What is the distribution of one column, per group? | `sns.boxplot` / `sns.violinplot` | Violin shows shape; box shows quartiles + outliers. |
| What is the relationship of two numeric columns? | `sns.scatterplot` | Color points by a third column with `hue=`. |
| What is the relationship, plus a linear fit? | `sns.regplot` / `sns.lmplot` | `regplot` is one chart; `lmplot` does facets. |
| How are *all* the numeric columns related? | `sns.pairplot` | Scatter matrix. Slow for >10 columns. |
| What is the count of one categorical column? | `sns.countplot` | A bar chart, but with seaborn's tidy-data interface. |
| Are two categorical columns correlated? | `sns.heatmap(pd.crosstab(...))` | Or pass a correlation matrix. |

Drill: take the Palmer Penguins dataset (`sns.load_dataset("penguins")`, ships with seaborn) and produce one chart for each row of the table. That is the seaborn vocabulary in one notebook.

---

## 4. plotly in twenty minutes

Plotly produces **interactive** HTML charts. Hover tooltips, zoom, pan, toggle traces in the legend. The output is a JSON specification and an embedded JavaScript renderer.

Two API tiers, like seaborn vs matplotlib:

- **`plotly.express` (high-level).** Takes a `DataFrame` and a few column names, returns a `Figure`. `px.scatter`, `px.line`, `px.bar`, `px.histogram`, `px.box`, `px.choropleth`. This is what you use 90% of the time.
- **`plotly.graph_objects` (low-level).** Build a `go.Figure` and add `go.Scatter`, `go.Bar`, etc. as traces. Needed for fine control; rarely needed for everyday charts.

A one-line plotly chart:

```python
import plotly.express as px
fig = px.scatter(df, x="trip_distance", y="fare_amount",
                 color="payment_type", hover_data=["pickup_borough"])
fig.write_html("scatter.html")
```

That opens in a browser. Hover over a point: trip distance, fare, payment type, borough. Zoom in: the points stay sharp; the axis ticks re-fit. None of this is possible with matplotlib because matplotlib draws raster output.

### When plotly earns its keep

- **Dashboards and reports embedded in web pages.** Streamlit, Dash, FastAPI HTML responses, JupyterLab outputs that get shared.
- **Maps.** `px.choropleth`, `px.scatter_mapbox`, `px.line_geo`. Matplotlib's `cartopy` integration is fine; plotly's is shorter and interactive.
- **3-D scatter where rotation actually matters.** A static 3-D matplotlib chart is almost always misleading; an interactive 3-D plotly chart at least lets the reader rotate it.
- **Exploratory work where hover is the question.** "What is this outlier?" plotly answers in one mouse hover; matplotlib needs an annotation per point.

### When plotly costs you

- **Reports printed on paper.** Plotly's PDF export works but is heavy and not vector by default.
- **File size.** A plotly HTML file is 3 MB minimum (the renderer is bundled). Six plotly charts on one page is 20 MB. matplotlib PNGs are 50 KB.
- **Style consistency with the brand.** Plotly's theme system exists (`plotly.io.templates`) but the default look does not match the FT / OWID editorial style we are aiming for. Customizing it to look quiet is fighting the library.
- **Reproducibility.** A static PNG is bit-for-bit the same forever. An HTML file rendered by a future plotly.js version may render differently.

For C5 the rule is: **the mini-project's charts are matplotlib + seaborn (static). Plotly shows up in Exercise 3's stretch goal and in Week 11 when we serve a model behind an API.** If you ever ship a customer-facing dashboard, plotly is a great call. For a portfolio report, static is the right answer.

---

## 5. A four-question picker

Given a chart task, ask:

1. **Is the output a report, a slide deck, or a print medium?** If yes, matplotlib (with seaborn for setup).
2. **Is the chart a one-line "give me the distribution of X by Y" during EDA?** If yes, seaborn.
3. **Will a reader hover, zoom, or interact?** If yes, plotly.
4. **Does the chart need a level of customization the library does not offer?** If yes, matplotlib. If you started with seaborn, drop to matplotlib on the returned Axes. If you started with plotly, you are usually stuck.

A worked picker example. "I need a chart of CO2 emissions per country over time, colored by region, that goes in our portfolio README."

- Static? Yes (it goes in a markdown file). → matplotlib or seaborn.
- One-line? No — there are five regions and we want a legend with regional totals. → matplotlib OO.
- Interactive? No (README.md is rendered to HTML but plotly embeds are heavy). → matplotlib.
- Customization? We want a small annotation on the 2020 pandemic dip. → matplotlib's `ax.annotate`.

Answer: matplotlib, with maybe `sns.set_style("whitegrid")` at the top for a faint grid.

Same question but: "I need the same chart for the FT's data desk web page, where readers will hover for exact numbers."

- Static? No (web page, readers will interact). → plotly.
- Answer: `plotly.express.line` with `hover_data=["population", "gdp"]`.

The picker is short because the answers are usually short. Most charts in a data-science workflow are static; one in five is interactive.

---

## 6. The chart-type ↔ question mapping (the part that matters more than the library)

A chart is a tool. Tools work when matched to the question. The question is one of these five — pick the right one before picking the library.

### Distribution: "How is one variable spread out?"

- **Histogram** (`sns.histplot`, `ax.hist`). Default first choice for one numeric column. Bin choice matters; 30 bins is a good starting point, then adjust.
- **KDE** (`sns.kdeplot`). A smoothed histogram. Good for overlaying multiple distributions; bad when the data is sharply discrete.
- **Box plot** (`sns.boxplot`). Quartile-and-whiskers; compact when you have many groups.
- **Violin plot** (`sns.violinplot`). A box plot plus a KDE on each side. Shows the *shape*, not just the quartiles. Better when groups are few and their shapes matter.
- **Strip plot / swarm plot** (`sns.stripplot`, `sns.swarmplot`). One point per row, jittered. Good for `n < 200`; useless for `n > 5000`.

### Relationship: "How do two variables co-vary?"

- **Scatter plot** (`sns.scatterplot`, `ax.scatter`). Default first choice. Add transparency (`alpha=0.3`) for `n > 1000`.
- **Hexbin** (`ax.hexbin`). 2-D histogram. The right answer when scatter becomes a solid blob.
- **2-D KDE** (`sns.kdeplot` with two variables). Smoothed version of hexbin.
- **Regression plot** (`sns.regplot`). Scatter plus a linear fit and a 95% CI band.

### Composition: "How does a whole break into parts?"

- **Stacked bar** (`ax.bar(..., bottom=)`). The honest version of a pie chart.
- **Treemap** (`squarify` library, optional). For hierarchical composition.
- **Pie chart** (`ax.pie`). Avoid except when there are two or three slices and you have a *brand* reason.

### Change over time: "How does this evolve?"

- **Line chart** (`ax.plot`). Default. One series per line.
- **Area chart** (`ax.fill_between`). When you want to emphasize the total.
- **Slope chart** (two columns connected by lines). Two time points; emphasizes the change.
- **Small multiples**. Many series, one panel each. The Tufte answer when a line chart has too many lines.

### Ranking: "Who is biggest?"

- **Horizontal bar, sorted** (`ax.barh` with `sort_values`). The right answer for "top 10 by X." Always.
- **Dot plot** (`ax.plot(x, y, "o")` with categories on y). Like a bar chart but emphasizes the value, not the bar.

The book to read on this is the [FT Visual Vocabulary PDF](https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/Visual-vocabulary.pdf) — it lays out every chart family on one fold-out and recommends one or two for each question. Print it. Tape it to your monitor.

---

## 7. A worked task-by-task picker

**Task A.** EDA: "What is the distribution of `trip_distance` in the cleaned taxi DataFrame?"

- Question: distribution of one numeric column.
- Library: seaborn (one line).
- Code: `sns.histplot(data=df, x="trip_distance", bins=40)`.

**Task B.** Slide deck: "How many trips happened per borough, August 2025?"

- Question: ranking of categorical groups.
- Library: matplotlib (you want full control of the title and caption).
- Chart type: horizontal bar, sorted.
- Worked in Lecture 1, Section 12.

**Task C.** Web report: "Let readers explore CO2 emissions by country and year."

- Question: change over time, with interactivity.
- Library: plotly.
- Code: `px.line(co2, x="year", y="co2", color="country")`.

**Task D.** Mini-project critique: "Recreate the FT's '2020 was the warmest year on record' chart."

- Question: change over time, one variable.
- Library: matplotlib (the FT chart is print; static reproduction is the right comparison).
- Chart type: line with annotation on the 2020 maximum.

**Task E.** "Which Chicago districts have the highest theft rate?"

- Question: ranking.
- Library: matplotlib + seaborn (seaborn for `barplot` is fine here; matplotlib's `barh` is fine too).
- Chart type: horizontal bar, sorted.

For every task: name the question first; then the library; then the chart type. Three sentences before you write code.

---

## 8. The seaborn `objects` interface (brief)

In seaborn 0.12 (2022), the maintainers shipped a new `seaborn.objects` interface that is more compositional, less stateful. It looks like:

```python
import seaborn.objects as so
(
    so.Plot(df, x="trip_distance", y="fare_amount", color="payment_type")
      .add(so.Dots())
      .add(so.Line(), so.PolyFit(order=1))
      .label(x="Trip distance (mi)", y="Fare (USD)")
)
```

It is closer to ggplot2 (the R library) than the older seaborn API. As of 2026 it is stable but not yet the dominant seaborn idiom in the wild. We mention it for completeness; this week we use the older function-style API (`sns.histplot`, `sns.scatterplot`) because that is what 95% of seaborn code in real projects looks like.

If you have used ggplot2, the `seaborn.objects` syntax will feel like home and is a fine choice for new code. The pickability of "which function do I call" is slower to learn, which is why we did not lead with it.

---

## 9. Pandas' built-in `.plot` method: when it is enough

Pandas wraps matplotlib in a `DataFrame.plot` method. For a quick chart on a DataFrame you have already loaded:

```python
df["trip_distance"].plot(kind="hist", bins=40)
df.plot(x="pickup_datetime", y="fare_amount")
df.groupby("borough")["fare_amount"].mean().plot(kind="bar")
```

`.plot` returns a matplotlib `Axes`, so you can customize after the fact. The defaults are matplotlib's defaults — austere. For a one-line EDA chart in a notebook, `.plot` is the shortest spelling. For anything that ships, you outgrow it within a day.

The full kinds (`kind="line" | "bar" | "barh" | "hist" | "box" | "kde" | "area" | "pie" | "scatter" | "hexbin"`) overlap with seaborn. The rule of thumb:

- **`df["col"].plot(...)`** for a one-line REPL chart.
- **`sns.something(data=df, x=..., y=...)`** for a one-line EDA chart that you might keep.
- **`fig, ax = plt.subplots(); ax.something(...)`** for any chart that goes in a report.

Three lines of progression. Pick the right one for the moment.

---

## 10. A decision flowchart

When you sit down to make a chart, ask in this order:

```text
What is the question?
    ├── Distribution of one var?         → histplot / boxplot / violinplot (seaborn)
    ├── Relationship of two vars?        → scatter / hexbin / regplot
    ├── Composition (parts of a whole)?  → stacked bar; avoid pie
    ├── Change over time?                → line; small multiples if many series
    └── Ranking?                         → horizontal sorted bar

Will the reader interact (hover, zoom)?
    ├── Yes → plotly.express
    └── No  → matplotlib + seaborn

Do I need full control of the look (annotations, fine layout)?
    ├── Yes → matplotlib OO end-to-end
    └── No  → seaborn one-liner is fine
```

Run that flowchart for every chart this week, out loud the first few times. By the mini-project it should be silent.

---

## 11. Self-check (five-minute drill)

For each task, name **the library** and **the chart type** before looking at the answer:

1. "Show me the distribution of trip fares for each borough."
2. "I need a web page where readers can explore COVID cases by country over time."
3. "What is the relationship between `trip_distance` and `tip_amount`, with `payment_type` as a third dimension?"
4. "I want a single chart for our README showing total trips per month in 2025."
5. "Which complaint types in 311 have the worst response times?"
6. "How are all six numeric columns of the penguins dataset related?"

Answers:

1. **seaborn**, `boxplot` (or `violinplot` if you want shape, not just quartiles).
2. **plotly**, `line` (interactivity is the reason).
3. **seaborn**, `scatterplot` with `hue="payment_type"`.
4. **matplotlib**, line chart (one series, README-bound, no interactivity).
5. **matplotlib**, horizontal sorted bar (ranking, going in a report).
6. **seaborn**, `pairplot`.

If you got 5+ correct on the first pass, you have internalized the picker.

---

## Recap

- matplotlib draws static charts with full control. It is the foundation; everything else compiles down to it.
- seaborn is a convenience layer for tidy DataFrames with statistical defaults. Use it when the convenience is real and the customization stays shallow.
- plotly is for interactive HTML output. Use it for dashboards, maps, and exploration; avoid it for print.
- The chart-type-to-question mapping matters more than the library choice. Pick the chart type first, the library second.
- Pandas' `.plot` is a useful third option for one-line EDA in a notebook.

Next: Lecture 3 — perceptual encoding, color, axis tricks, and the four common ways a chart can lie even when every label is correct.

---

## Further reading (optional, free)

- Seaborn's [tutorial](https://seaborn.pydata.org/tutorial.html) — the first three pages are exactly this lecture in twice the words.
- Plotly's [express overview](https://plotly.com/python/plotly-express/) — every function on one page.
- Mike Bostock's "[A Better Way to Code](https://medium.com/@mbostock/a-better-way-to-code-2b1d2876a3a0)" — about Observable / D3, but the principles about chart-as-program are universal.
