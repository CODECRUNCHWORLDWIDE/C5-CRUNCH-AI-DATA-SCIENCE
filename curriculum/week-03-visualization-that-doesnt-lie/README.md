# Week 3 — Visualization That Doesn't Lie

> *A chart is an argument. Every default — the y-axis range, the color ramp, the chart type, the aspect ratio — is a sentence in that argument. This week we learn to read the argument and to write it on purpose.*

Welcome to week three of **C5 · Crunch AI / Data Science**. Week 1 gave you the `ndarray`. Week 2 put an index and a column name on it and called it a `DataFrame`. This week we take that cleaned table and turn it into a picture an executive can act on — and, more importantly, a picture that does not silently lie.

Three caveats up front, because visualization is the part of data work that is most often graded on aesthetics and least often on honesty:

1. **A pretty chart is not a true chart.** Most "data viz courses" are taught as design. We will spend roughly one third of the week on what is true, one third on what is readable, and one third on the matplotlib / seaborn / plotly mechanics. The order is deliberate.
2. **The default matplotlib style is a working draft, not a deliverable.** You will leave this week knowing how to ship a chart that goes into a report — labeled axes, sane font sizes, a title that is a *sentence*, not a column name. The default `plt.plot(x, y)` figure is fine for the REPL and nowhere else.
3. **A chart can be honest *and* wrong.** Honest means the encoding does not deceive. Wrong means the underlying data was bad, the question was bad, or the conclusion does not follow. We work on the encoding here; the rest is Weeks 1, 2, 10, and 12.

We target **matplotlib 3.8+**, **seaborn 0.13+**, and **plotly 5.x** (current minor in 2026 is 5.24). All three are free, open-source, pip-installable, and have stable APIs in 2026.

---

## Learning objectives

By the end of this week, you will be able to:

- **Describe** matplotlib's Figure/Axes mental model: one `Figure`, one or more `Axes`, each Axes a coordinate system. Use the object-oriented API, not `plt.plot`-and-pray.
- **Choose** between matplotlib, seaborn, and plotly for a given task, with one-sentence reasons each.
- **Pick** a chart type that matches the question (distribution, relationship, composition, change-over-time, ranking) rather than the chart type you remember first.
- **Encode** values perceptually well: position beats length beats area beats color hue beats color saturation. Know which encoding you are using and why.
- **Read** a y-axis. Notice when it does not start at zero. Notice when it is logarithmic. Notice when there are two of them. Decide whether each choice is honest.
- **Recreate** a published chart from a real news source (FT, NYT Upshot, Our World in Data) using only matplotlib + seaborn, and write a critique of the original.
- **Avoid** the four most common chart lies: truncated y-axis on a bar chart, dual-axis with two unrelated scales, rainbow colormaps on ordinal data, 3-D pie charts.
- **Pass** every `pytest` case on the Week 3 exercises and ship the chart-recreation mini-project.

---

## Prerequisites

- **Weeks 1 and 2 complete.** You should have a cleaned `DataFrame` on disk from the Week 2 mini-project — we will plot it.
- A working **Python 3.11+** install (we use 3.12).
- matplotlib 3.8+: `pip install "matplotlib>=3.8,<4"`.
- seaborn 0.13+: `pip install "seaborn>=0.13,<1"`.
- plotly 5.x (optional but recommended): `pip install "plotly>=5.20,<6"`.
- pandas 2.2+ and numpy 2.x (from Weeks 1–2).
- `pytest` for the exercise smoke tests.

No GPU. No notebook required for the exercises (they are `.py` files and save PNGs to disk); the mini-project ships as a notebook.

---

## Topics covered

- The matplotlib **Figure / Axes / Artist** model. Why `fig, ax = plt.subplots()` is the only spelling you should ship.
- The two matplotlib APIs: the **pyplot state-machine** (`plt.plot`, `plt.title`) and the **object-oriented** (`ax.plot`, `ax.set_title`). Pick OO and never look back.
- Anatomy of a chart: figure, axes, spines, ticks, tick labels, gridlines, legend, title, caption.
- **Seaborn** as a "statistical-defaults wrapper around matplotlib." When `sns.scatterplot` saves you twenty lines and when it costs you control.
- **Plotly** as the "interactive HTML output" option. When you want hover, zoom, and a chart that lives on a web page — and when interactivity is a distraction.
- The five canonical questions: distribution (histogram, KDE, box, violin), relationship (scatter, hexbin, regression), composition (stacked bar, treemap — and why pies usually lose), change over time (line, area, slope), ranking (sorted bar, dot plot).
- **Perceptual encoding ordering** (Cleveland and McGill, 1984): position > length > angle > area > volume > color hue. Pick encodings up the ladder, not down.
- **Color**: sequential (one-ended), diverging (two-ended), qualitative (categorical). Why `viridis` won and `jet` lost. Color-blind safety in 2026.
- **Axis choices**: zero baseline on bars (non-negotiable), free baselines on line charts (often fine), log scale (when the data spans orders of magnitude), the dual-axis trap.
- **The four common lies**: truncated bar y-axis, dual-axis correlation theater, rainbow on ordinal, 3-D anything.
- **Reading a published chart**: the FT, NYT Upshot, and Our World in Data house styles. What they encode, what they label, what they cite.
- The matplotlib **stylesheet** mechanism: ship one `crunch.mplstyle` and your charts look like the brand. We provide a sober violet-accented one.

---

## Weekly schedule

Target about **36 hours**. Some sections click in fifteen minutes; some take three hours. Treat the table as a budget, not a contract.

| Day       | Focus                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Figure/Axes; the OO API; first labeled chart   |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | Seaborn for distributions and relationships    |   0h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    4h       |
| Wednesday | When plotly; interactivity vs print            |   2h     |   1h      |     2h     |   0.5h    |   1h     |     0h       |   0h       |   6.5h      |
| Thursday  | Perceptual encoding; color; axis honesty       |   3h     |   0h      |     2h     |   0.5h    |   1h     |     2h       |   0.5h     |    9h       |
| Friday    | Time-series charts; recreating a published one |   0h     |   2h      |     0h     |   0.5h    |   1h     |     2h       |   0.5h     |    6h       |
| Saturday  | Mini-project polish; critique writing          |   0h     |   0h      |     0h     |   0h      |   1h     |     3h       |   0h       |    4h       |
| Sunday    | Quiz, review, push to portfolio repo           |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                | **8h**   | **7h**    | **4h**     | **3h**    | **6h**   | **7h**       | **2h**     |  **37h**    |

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Free docs for matplotlib / seaborn / plotly; the FT chart-style notes |
| [lecture-notes/01-matplotlib-fundamentals.md](./lecture-notes/01-matplotlib-fundamentals.md) | Figure/Axes model; pyplot vs OO; first publication-quality chart |
| [lecture-notes/02-when-to-use-seaborn-vs-plotly.md](./lecture-notes/02-when-to-use-seaborn-vs-plotly.md) | Task-by-task picker between matplotlib, seaborn, and plotly |
| [lecture-notes/03-perceptual-encoding-and-not-lying.md](./lecture-notes/03-perceptual-encoding-and-not-lying.md) | Cleveland and McGill; color; axis honesty; the four common lies |
| [exercises/README.md](./exercises/README.md) | Index of exercises |
| [exercises/exercise-01-figure-and-axes.py](./exercises/exercise-01-figure-and-axes.py) | Build a 2×2 subplot grid; label every axis; ship a PNG |
| [exercises/exercise-02-distributions-and-relationships.py](./exercises/exercise-02-distributions-and-relationships.py) | Histogram, KDE, scatter, hexbin — pick the right one for the question |
| [exercises/exercise-03-time-series.py](./exercises/exercise-03-time-series.py) | A line chart with a sensible date axis, annotation, and a caption |
| [challenges/README.md](./challenges/README.md) | Index of weekly challenges |
| [challenges/challenge-01-recreate-an-FT-chart.md](./challenges/challenge-01-recreate-an-FT-chart.md) | Recreate a Financial Times chart with synthetic data, side-by-side |
| [quiz.md](./quiz.md) | 10 multiple-choice questions with an answer key |
| [homework.md](./homework.md) | Six practice problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Full spec for the chart-recreation-and-critique project |

---

## Stretch goals

- Read the [matplotlib "Anatomy of a figure"](https://matplotlib.org/stable/gallery/showcase/anatomy.html) page. Annotate it in your own notes. The thirty terms on that figure are the matplotlib vocabulary.
- Skim Edward Tufte's *The Visual Display of Quantitative Information* (1983). The book is not free, but the [Tufte CSS gallery](https://edwardtufte.github.io/tufte-css/) reproduces the visual ideas. Tufte's "data-ink ratio" is the single best principle in the field.
- Read three pages of the [Financial Times Visual Vocabulary](https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/Visual-vocabulary.pdf) (free PDF). Pick one chart family per page and write a one-sentence note on when you would reach for it.
- Reproduce one figure from Hans Rosling's *Factfulness* — usually a bubble chart of GDP vs life expectancy, sized by population. The data is on [Gapminder](https://www.gapminder.org/data/) (free).
- Look at the source of one seaborn function (say [`sns.histplot`](https://github.com/mwaskom/seaborn/blob/master/seaborn/distributions.py)) and notice how much of it is data manipulation, not drawing. Seaborn is mostly pandas with a small matplotlib epilogue.

---

## What you will *not* do this week

You will not:

- Build a dashboard (a different skill — Streamlit / Dash / Grafana belong in a deployment week, not here).
- Train a model (Week 4).
- Make charts that need D3, Observable, or front-end Javascript. Python is enough for everything Week 3 asks of you.
- Use `chart.js`, `bokeh`, `altair`, or any of the dozen other Python plotting libraries. Three is enough to be fluent in; six is a way to be fluent in none.
- Use any "AI-generated chart" or "auto-EDA" library. Defaults you did not choose are defaults you cannot defend.

That is deliberate. The week you spend learning *why* a chart works is the year you do not spend producing executive decks that get quietly ignored.

---

## Up next

[Week 4 — The ML Workflow and Linear Models](../week-04/) — once you have pushed your chart-recreation report to your `crunch-ai-portfolio-<yourhandle>` repo.
