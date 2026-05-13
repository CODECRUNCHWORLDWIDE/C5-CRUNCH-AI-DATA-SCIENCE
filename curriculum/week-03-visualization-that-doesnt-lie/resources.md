# Week 3 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **matplotlib — "Quick start"** (the canonical first hour; reads the Figure/Axes model into you):
  <https://matplotlib.org/stable/users/explain/quick_start.html>
- **matplotlib — "Anatomy of a figure"** (one diagram, thirty named parts; print it):
  <https://matplotlib.org/stable/gallery/showcase/anatomy.html>
- **seaborn — "An introduction to seaborn"**:
  <https://seaborn.pydata.org/tutorial/introduction.html>
- **seaborn — "Choosing color palettes"**:
  <https://seaborn.pydata.org/tutorial/color_palettes.html>
- **plotly — "Plotly Express in Python"** (the high-level interface; skim before deciding plotly is for you):
  <https://plotly.com/python/plotly-express/>

## The official docs you will bounce between all week

- **matplotlib user guide**: <https://matplotlib.org/stable/users/index.html>
- **matplotlib API reference**: <https://matplotlib.org/stable/api/index.html>
- **matplotlib "Choosing colormaps"** (read once, refer to forever): <https://matplotlib.org/stable/users/explain/colors/colormaps.html>
- **seaborn API reference**: <https://seaborn.pydata.org/api.html>
- **plotly Python reference**: <https://plotly.com/python-api-reference/>

## Free chart-style guides

- **Financial Times — Visual Vocabulary** (free PDF; the most useful single page on chart selection in the field):
  <https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/Visual-vocabulary.pdf>
- **Financial Times — Chart Doctor case studies** (free, GitHub-hosted; one chart per case):
  <https://github.com/Financial-Times/chart-doctor>
- **BBC Visual and Data Journalism cookbook** (R-flavored but the rationale is language-independent):
  <https://bbc.github.io/rcookbook/>
- **Our World in Data — chart style** (every chart on the site is open-source; the GitHub repo holds the standard):
  <https://github.com/owid/owid-grapher>
- **Datawrapper Academy — "What to consider when making charts"** (free; their checklist is short and sober):
  <https://academy.datawrapper.de/>

## Free books and long-form reading

- **Claus O. Wilke — *Fundamentals of Data Visualization*** (free online HTML, MIT-licensed):
  <https://clauswilke.com/dataviz/>
  Read chapters 4 (color), 17 (visualizing distributions), and 25 (handling overlap).
- **Kieran Healy — *Data Visualization: A Practical Introduction*** (free online HTML):
  <https://socviz.co/>
  The R code is incidental; the design principles are universal.
- **Jake VanderPlas — *Python Data Science Handbook*, Chapter 4 (Visualization with Matplotlib)** (free, MIT-licensed):
  <https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html>

## The data sources we use this week

All public, all free to download:

- **Our World in Data — CO2 and Greenhouse Gas Emissions** (CSV; the FT and NYT pull from the same upstream):
  <https://github.com/owid/co2-data>
- **FRED — Federal Reserve Economic Data** (CSV via the website; classic for time-series practice):
  <https://fred.stlouisfed.org/>
- **NYC Open Data — 311 Service Requests** (from Week 2; we plot it this week):
  <https://data.cityofnewyork.us/Social-Services/311-Service-Requests/erm2-nwe9>
- **Palmer Penguins** (small, clean, ships with seaborn; perfect for the relationship-chart drill):
  <https://github.com/allisonhorst/palmerpenguins>
- **Gapminder** (Hans Rosling's bubble-chart dataset; lifespan, GDP, population by country and year):
  <https://www.gapminder.org/data/>

## The math you do not need this week

You do not need new math this week. Week 3 is library proficiency and design discipline. The one piece of statistics that helps — kernel density estimation — has a one-sentence definition: "a smoothed histogram where each data point contributes a small Gaussian bump." That is enough.

## Tools you will use this week

- **`matplotlib`** ≥ 3.8: `pip install "matplotlib>=3.8,<4"`.
- **`seaborn`** ≥ 0.13: `pip install "seaborn>=0.13,<1"`.
- **`plotly`** ≥ 5.20 (optional but recommended): `pip install "plotly>=5.20,<6"`.
- **`pandas`** ≥ 2.2 and **`numpy`** ≥ 2 (from prior weeks).
- **`pytest`** for the exercises.

A `requirements.txt` snippet for the week:

```text
pandas>=2.2,<3
numpy>=2,<3
matplotlib>=3.8,<4
seaborn>=0.13,<1
plotly>=5.20,<6
pytest>=8
```

## Videos (free, no signup)

- **Nicolas Rougier — "Scientific Visualization: Python and Matplotlib"** (free PDF + companion videos):
  <https://github.com/rougier/scientific-visualization-book>
- **Cole Nussbaumer Knaflic — *Storytelling with Data*** talks (free YouTube): tight, executive-facing critiques of real corporate charts.
- **John Burn-Murdoch — Financial Times** charts walkthroughs (free, on the FT YouTube channel): the same data journalist who built the FT's COVID dashboards, talking through his chart choices.

## Open-source projects to read (in this order)

You can learn more from one hour reading other people's charts than from three hours of tutorials.

1. **matplotlib gallery** — every example with full source: <https://matplotlib.org/stable/gallery/index.html>
2. **seaborn gallery** — same: <https://seaborn.pydata.org/examples/index.html>
3. **Our World in Data — Grapher** — the open-source chart engine behind every chart on the site: <https://github.com/owid/owid-grapher>
4. **FT Chart Doctor** — one repo per chart, with the data and the R/Python source: <https://github.com/Financial-Times/chart-doctor>

## Style files (provided)

A small `crunch.mplstyle` matplotlib stylesheet ships with this week (see Exercise 1). It sets the brand violet `#7C3AED` as the primary color, JetBrains Mono for tick labels, EB Garamond for titles, and a quiet gray gridline. Drop it into `~/.config/matplotlib/stylelib/` and use `plt.style.use("crunch")` to enable.

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **Figure** | The whole image / page / canvas. One Figure can hold many Axes. |
| **Axes** | One coordinate system (one chart). `fig, ax = plt.subplots()` makes one of each. |
| **Artist** | Anything drawn on an Axes: lines, points, patches, text, ticks. |
| **Spine** | One of the four edges of an Axes: top, right, bottom, left. |
| **Tick / tick label** | A mark on an axis / the text under that mark. |
| **Pyplot API** | `plt.plot(...)`, `plt.title(...)` — a stateful wrapper for the REPL. |
| **OO API** | `ax.plot(...)`, `ax.set_title(...)` — the API you ship. |
| **Sequential colormap** | One direction, light → dark (e.g. `viridis`). For ordered values. |
| **Diverging colormap** | Two directions from a midpoint (e.g. `RdBu`). For deviations from a baseline. |
| **Qualitative palette** | A small set of distinct colors for categories (e.g. `tab10`). |
| **KDE** | Kernel density estimate — a smoothed histogram. |
| **Hexbin** | A 2-D histogram on a hexagonal grid; the right answer for dense scatter plots. |
| **Slope chart** | Two columns connected by lines — "before / after" for ranked items. |
| **Small multiples** | A grid of the same chart drawn many times, one per category. Tufte's invention. |
| **Data-ink ratio** | Tufte: the proportion of ink that encodes data. Higher is better. |
| **Chartjunk** | Tufte: any ink that does not encode data (3-D effects, drop shadows, decorative borders). |

---

*If a link 404s, please open an issue so we can replace it.*
