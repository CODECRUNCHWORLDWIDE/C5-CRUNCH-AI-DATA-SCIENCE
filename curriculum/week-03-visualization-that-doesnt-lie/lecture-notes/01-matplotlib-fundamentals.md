# Lecture 1 — matplotlib Fundamentals: Figure, Axes, and the API That Ships

> **Outcome:** You can name every part of a matplotlib figure; you write `fig, ax = plt.subplots()` reflexively; you can take a `DataFrame` and produce a publication-quality chart with labeled axes, a sentence-form title, a caption with the data source, and no warnings.

Week 1 left you with the `ndarray`. Week 2 left you with the `DataFrame`. This week's first object is the **`Figure`**, and its child the **`Axes`**. Everything else in matplotlib — lines, bars, scatter points, titles, ticks, gridlines, legends — is an `Artist` placed on an `Axes`.

We target **matplotlib 3.8+** (current stable in 2026 is 3.9). The Figure / Axes model has been the matplotlib data model since version 1.0 in 2008; you are learning a 17-year-old API that will outlive most of the libraries you use this year.

---

## 1. Why matplotlib feels confusing for the first hour

Most beginners' first matplotlib experience goes like this:

```python
>>> import matplotlib.pyplot as plt
>>> plt.plot([1, 2, 3], [4, 1, 5])
>>> plt.title("My chart")
>>> plt.xlabel("x")
>>> plt.show()
```

That works. It also teaches you nothing about what is happening. There is a hidden `Figure`, a hidden `Axes`, and a hidden "current axes" that `plt.title` writes to. The `plt` namespace is a stateful wrapper around an object model. The stateful wrapper is fine for the REPL. It is not the API you ship.

The object model:

```python
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 1, 5])
>>> ax.set_title("My chart")
>>> ax.set_xlabel("x")
>>> fig.savefig("chart.png", dpi=150, bbox_inches="tight")
```

Same chart. Now `fig` and `ax` are named objects you can introspect, modify, save, and pass around. When you have multiple subplots, `fig, axes = plt.subplots(2, 2)` gives you a 2-D array of `Axes` and the stateful API breaks down completely. Start with the OO API. There is no reason to teach the other.

The rule for this course is:

> **Use `fig, ax = plt.subplots(...)`. Never use `plt.plot(...)` outside the REPL.**

If you find yourself writing `plt.title(...)` in a script, you have already lost.

---

## 2. The Figure / Axes / Artist tree

A matplotlib figure is a tree:

```text
Figure
├── Axes (one or more — each is one chart)
│   ├── XAxis
│   │   ├── ticks
│   │   ├── tick labels
│   │   └── axis label
│   ├── YAxis
│   │   └── (same)
│   ├── spines (the four edges of the chart box)
│   ├── Artists (plotted data)
│   │   ├── Line2D       ← from ax.plot
│   │   ├── PathCollection ← from ax.scatter
│   │   ├── Rectangle   ← from ax.bar
│   │   ├── Text         ← from ax.annotate or ax.text
│   │   └── ...
│   ├── Legend
│   └── title
└── suptitle (a Figure-level title above all Axes)
```

Memorize these five names: **Figure, Axes, Spines, Ticks, Artists**. Every method call in this lecture is one of "make an Axes," "draw an Artist on an Axes," or "configure the Axes' spines or ticks."

A useful interactive drill:

```python
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots()
>>> ax.plot([1, 2, 3], [4, 1, 5])
>>> ax.get_children()  # every artist on this Axes
[<matplotlib.lines.Line2D ...>,
 <matplotlib.spines.Spine ...>,   # left
 <matplotlib.spines.Spine ...>,   # right
 <matplotlib.spines.Spine ...>,   # bottom
 <matplotlib.spines.Spine ...>,   # top
 <matplotlib.axis.XAxis ...>,
 <matplotlib.axis.YAxis ...>,
 <matplotlib.text.Text ...>]      # title (empty)
```

Every element is an object. Every object has a setter. The chart is just data plus configuration of those objects.

---

## 3. The four ways to make a Figure (and which to pick)

```python
# 1. Single Axes (the most common):
fig, ax = plt.subplots(figsize=(7, 4))

# 2. A grid:
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
#   axes is a 2-D numpy array: axes[0, 0], axes[1, 2], etc.

# 3. Uneven layout (when subplots is too rigid):
fig = plt.figure(figsize=(10, 4))
gs = fig.add_gridspec(nrows=2, ncols=3)
ax_big   = fig.add_subplot(gs[:, 0:2])       # left two columns, both rows
ax_top   = fig.add_subplot(gs[0, 2])         # top right
ax_bot   = fig.add_subplot(gs[1, 2])         # bottom right

# 4. subplot_mosaic — string-art layout (matplotlib 3.3+, very readable):
fig, axes = plt.subplot_mosaic("""
    AAB
    AAC
""", figsize=(10, 4))
# axes is a dict: axes["A"], axes["B"], axes["C"]
```

For 90% of weekly work, option 1 (single Axes) or 2 (rectangular grid). For the mini-project's three-chart layout, option 4 (`subplot_mosaic`) reads like ASCII art. We will use 1 and 2 in this lecture and meet 4 in Lecture 3.

`figsize` is in **inches**. The default is 6.4 × 4.8, which is tiny on a modern screen. For reports, 7 × 4 (landscape) or 8 × 6 (presentation) is a good starting point. The DPI (`fig.savefig(..., dpi=150)`) controls pixel count; 150 is sharp for the web, 300 for print.

---

## 4. Ten lines that ship: the publication-quality chart template

```python
import matplotlib.pyplot as plt
import numpy as np

# data
x = np.linspace(0, 4 * np.pi, 200)
y = np.sin(x)

# chart
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(x, y, color="#7C3AED", linewidth=2, label="sin(x)")
ax.set_xlabel("x (radians)")
ax.set_ylabel("sin(x)")
ax.set_title("One period of sin is roughly 6.28 radians")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", frameon=False)

# annotation
ax.axhline(0, color="black", linewidth=0.8)
ax.text(0.99, 0.02, "source: synthetic — y = sin(x)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="gray")

fig.savefig("sin.png", dpi=150)
```

Walk through every line; nothing here is decorative.

- `figsize=(7, 4)` — landscape, fits a report page.
- `layout="constrained"` — replaces the old `tight_layout` hack. Matplotlib 3.6+ figures out the margins.
- `color="#7C3AED"` — brand violet (see [BRAND.md](../../../assets/branding/BRAND.md)). Default matplotlib blue is fine but identifying.
- `linewidth=2` — the default 1.5 is too thin for a deck slide.
- `label="sin(x)"` — required for the legend to find this line.
- `set_xlabel` / `set_ylabel` — *always*. A chart without axis labels is a chart you should not ship.
- `set_title` — a **sentence**, not a column name. "Daily revenue, Q3 2025" is OK; "One period of sin is roughly 6.28 radians" is better — it tells the reader what to take away.
- `grid(True, alpha=0.3)` — a faint gridline helps the eye; default off is too austere for tabular reading.
- `legend(frameon=False)` — the frame around a legend is chartjunk most of the time.
- `text(..., transform=ax.transAxes, ...)` — the caption with the data source. `transAxes` means coordinates 0..1 across the Axes, not data units. This is the right way to place text in "corner" positions.

The output is a labeled, captioned, sourced chart. If a reviewer cannot tell what it is showing in five seconds, you have failed. Every line above is fighting for those five seconds.

---

## 5. The state of pyplot in 2026

You will see `plt.plot(...)` in a thousand tutorials. Two reasons it survives, neither sufficient to make it the API you ship:

1. **Tradition.** matplotlib copied MATLAB's stateful API in 2003. Twenty-three years later, that legacy is the reason most beginners' first chart is `plt.plot`.
2. **Interactive REPL.** `plt.plot; plt.title; plt.show()` is genuinely shorter than the OO version for a one-off REPL chart. We won't fight that — at the REPL, use pyplot if you like. In any script, notebook cell that ships, or function, use OO.

You can mix them. `plt.subplots()` is pyplot; it returns OO objects. `fig.savefig(...)` is OO; you call it after composing with pyplot. The rule is:

- `plt.subplots`, `plt.style.use`, `plt.show`, `plt.savefig` — *constructors and global commands.* Pyplot wins here. Short and clear.
- `plt.plot`, `plt.title`, `plt.xlabel`, `plt.legend` — *operations on a chart.* OO wins here. Call them on `ax`.

This is the entire matplotlib style policy for the week.

---

## 6. The seven verbs you will use every day

Most charts you write this week are one of these:

```python
ax.plot(x, y)              # 1. line
ax.scatter(x, y)           # 2. scatter
ax.bar(categories, heights)# 3. bar (vertical)
ax.barh(categories, lengths) # 4. bar (horizontal — better for long labels)
ax.hist(values, bins=30)   # 5. histogram
ax.fill_between(x, lo, hi) # 6. shaded band (e.g., confidence interval)
ax.imshow(matrix)          # 7. heatmap (also useful for images)
```

Every method has dozens of keyword arguments. You will look them up by name your whole career. Memorize the verbs and the rough call shape; the keywords are a docs-tab away.

A two-line drill: produce a chart for each verb on the same synthetic data.

```python
rng = np.random.default_rng(0)
x   = np.linspace(0, 10, 100)
y   = np.sin(x) + 0.1 * rng.normal(size=100)

fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
axes[0, 0].plot(x, y);                  axes[0, 0].set_title("plot")
axes[0, 1].scatter(x, y, s=10);         axes[0, 1].set_title("scatter")
axes[0, 2].bar(np.arange(10), y[::10]); axes[0, 2].set_title("bar")
axes[0, 3].barh(np.arange(10), y[::10]);axes[0, 3].set_title("barh")
axes[1, 0].hist(y, bins=20);            axes[1, 0].set_title("hist")
axes[1, 1].fill_between(x, y-0.2, y+0.2, alpha=0.4)
axes[1, 1].plot(x, y);                  axes[1, 1].set_title("fill_between + plot")
axes[1, 2].imshow(rng.random((20, 20))); axes[1, 2].set_title("imshow")
axes[1, 3].axis("off")
fig.savefig("verbs.png", dpi=150)
```

Run that file. The output is the matplotlib vocabulary in one image. Pin it to your desk.

---

## 7. Spines, ticks, and the "publication look"

The default matplotlib spines are all four edges of the chart box. That is the engineer's default — a chart in a box. The editorial look (FT, NYT, Our World in Data) removes the top and right spines:

```python
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
```

Two lines. The chart immediately looks like a newspaper. There is no statistical reason; it is a style convention. Pick one and stick with it.

Ticks are the small marks on each axis. By default, matplotlib chooses tick locations automatically using a `Locator`. You override them with `set_xticks` / `set_xticklabels`:

```python
ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
ax.set_xticklabels(["0", "π", "2π", "3π", "4π"])
```

For dates on the x-axis, matplotlib has `matplotlib.dates` with `DateFormatter` and `DateLocator`. We will use them in Lecture 3 and Exercise 3.

The font size of tick labels defaults to 10pt, which is small. For presentations, bump everything:

```python
plt.rcParams.update({
    "axes.titlesize":  14,
    "axes.labelsize":  12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})
```

`rcParams` is matplotlib's global config dict. Setting it at the top of a notebook gives every chart the same look. The next step up is a **stylesheet**.

---

## 8. Stylesheets: ship the brand once

A matplotlib stylesheet is a text file with one `key: value` per line, in the same vocabulary as `rcParams`. You drop it into `~/.config/matplotlib/stylelib/` and use it with `plt.style.use("name")`. The C5 stylesheet (which Exercise 1 writes for you):

```text
# crunch.mplstyle
font.family:        sans-serif
font.sans-serif:    JetBrains Mono, DejaVu Sans
axes.titlesize:     13
axes.labelsize:     11
axes.spines.top:    False
axes.spines.right:  False
axes.grid:          True
grid.alpha:         0.25
grid.linestyle:     -
xtick.labelsize:    10
ytick.labelsize:    10
legend.frameon:     False
figure.figsize:     7, 4
figure.dpi:         110
savefig.dpi:        150
savefig.bbox:       tight
axes.prop_cycle:    cycler("color", ["7C3AED", "5B21B6", "111111", "9CA3AF"])
```

Use it:

```python
plt.style.use("crunch")
```

Every chart in this week's exercises and the mini-project should set the style once at the top of the file. The brand violet `#7C3AED` and its deep variant `#5B21B6` are the primary chart colors; black and gray fill in for additional series.

The benefit is consistency: any chart you ship looks like every other chart you ship. The cost is one line at the top of every script. The tradeoff is obvious.

---

## 9. Saving: png, svg, pdf — pick one per purpose

`fig.savefig("chart.ext", dpi=150)` infers format from the extension:

- **PNG**: raster, good for web, lossless. Use 150 dpi for the web, 300 for print.
- **SVG**: vector, infinitely scalable, edits cleanly in Inkscape or Illustrator. Use when the chart goes into a publication.
- **PDF**: vector, the print standard. Use when the chart goes into a LaTeX document.
- **JPG**: do not save charts as JPG. The compression artifacts on sharp lines are objectionable. PNG instead.

Two common arguments to always pass: `dpi=150` (or 300) and `bbox_inches="tight"` (trim whitespace). With the C5 stylesheet, both are defaults; without it, set them explicitly.

`fig.savefig` is also how you regression-test charts — compare a saved PNG against a reference. Exercise 1 includes a simple version of this.

---

## 10. Reading a chart from the docs: a worked example

Open the matplotlib gallery: <https://matplotlib.org/stable/gallery/index.html>. Click any example. Every page has the source. Pick the *Slope chart* example (or any other) and read it line by line. The pattern is always:

1. Imports.
2. Data (often synthetic in the gallery; you bring real data).
3. `fig, ax = plt.subplots(...)`.
4. One or more Artist creations: `ax.plot`, `ax.bar`, `ax.scatter`, ...
5. Axis configuration: `set_xlabel`, `set_ylabel`, `set_title`, `set_xticks`, ...
6. Spine and grid cleanup.
7. `fig.savefig(...)` or `plt.show()`.

Every chart you write this week is some variation on those seven steps. The library has hundreds of functions; the patterns have seven.

---

## 11. Common defaults to override (the publication checklist)

Before you ship a chart, run down this list:

- [ ] **Title** is a sentence ("Revenue grew 12% Q3 to Q4"), not a column name ("revenue").
- [ ] **X label** present, in plain English with units ("Quarter, 2024–2025"), not the column name ("qtr").
- [ ] **Y label** present, with units ("USD millions"), not "value."
- [ ] **Axis ranges** make sense. Bar charts start at zero (non-negotiable). Line charts can start elsewhere if you say why.
- [ ] **Tick labels** are readable. Numeric ticks for currency use a thousands separator (`"{x:,.0f}"`); dates use a date formatter, not a Unix timestamp.
- [ ] **Legend** present if there is more than one series, absent if there is one.
- [ ] **Color** is intentional. Brand color for the main series; gray for context series.
- [ ] **Caption** with data source. "Source: NYC TLC, 2025-08." This is the single most ignored detail and the one that separates a sketch from a chart.
- [ ] **No 3-D**. No drop shadows. No animated GIFs unless the medium is animated.

If you can tick every box, the chart is ready for a slide deck or a report. If you cannot, the chart is a draft.

---

## 12. A worked example end-to-end: NYC borough trip counts

Take the Week 2 mini-project artifact (the cleaned NYC taxi or 311 DataFrame) and produce one publication-quality chart:

```python
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("crunch")  # the stylesheet from Section 8

df = pd.read_parquet("week-02/data/taxi_clean.parquet")

# aggregate
by_borough = (
    df.groupby("pickup_borough", observed=True)
      .size()
      .sort_values(ascending=True)   # ascending so the longest bar is at the top
)

# chart
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.barh(by_borough.index, by_borough.values, color="#7C3AED")
ax.set_xlabel("Trips, August 2025")
ax.set_ylabel("")  # the borough name is its own label; no axis label needed
ax.set_title("Manhattan dominates pickup volume — by a factor of two over Brooklyn")
ax.xaxis.set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
)
ax.text(0.99, -0.18, "Source: NYC TLC green-taxi trip records, August 2025",
        transform=ax.transAxes, ha="right", fontsize=8, color="gray")
fig.savefig("trips_by_borough.png", dpi=150)
```

Notice every choice:

- **`barh`** instead of `bar`. Borough names are long; horizontal bars give them room without rotating tick labels.
- **`sort_values(ascending=True)`**. The longest bar at the top is the most-read position.
- **No y-axis label.** The category names *are* the label.
- **Brand violet `#7C3AED`.**
- **Thousands separator on the x-axis.** `15832` is unreadable; `15,832` is not.
- **Sentence-form title.** "Trips by borough" would be a column name. "Manhattan dominates pickup volume — by a factor of two over Brooklyn" is what you actually want the reader to know.
- **Caption with source.** Two seconds of typing; ten minutes of credibility.

That is the workflow for the week. Cleaning was Week 2; turning the clean table into the picture is Week 3.

---

## 13. Self-check (five-minute drill)

Without looking anything up, write code that:

1. Creates a 2×2 grid of subplots, 10 inches wide, 6 inches tall, with constrained layout.
2. On the top-left axes, plots `np.sin(x)` for `x` in `[0, 4π]` in the brand violet.
3. On the top-right, plots a histogram of 1000 standard-normal samples with 30 bins.
4. On the bottom-left, draws a horizontal bar chart of `["A", "B", "C", "D"]` with heights `[3, 1, 4, 1]`.
5. On the bottom-right, draws a scatter plot of 200 points with random `x, y` in `[0, 1]`.
6. Sets a Figure-level title `"Four chart families"` (`fig.suptitle`).
7. Saves the figure to `four_charts.png` at 150 dpi.

If you can do it without the docs in under five minutes, the chapter is in your fingers. If not, redo it once with the docs open; that is enough.

---

## Recap

- One `Figure`, one or more `Axes`, many `Artist`s.
- `fig, ax = plt.subplots(figsize=(W, H), layout="constrained")` is the spelling you ship.
- The pyplot API is fine at the REPL; the OO API is the API of every script and notebook.
- Seven verbs do most of the work: `plot`, `scatter`, `bar`, `barh`, `hist`, `fill_between`, `imshow`.
- A `crunch.mplstyle` stylesheet ships the brand in one line.
- Every shipped chart has: labeled axes (with units), a sentence-form title, a caption with source, no top/right spines, no chartjunk.

Next: Lecture 2 — when matplotlib is not enough, when seaborn saves you twenty lines, and when plotly is the right (or wrong) call.

---

## Further reading (optional, free)

- The matplotlib [anatomy diagram](https://matplotlib.org/stable/gallery/showcase/anatomy.html) — print it.
- The matplotlib [style sheets reference](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html) — pick a base to override.
- Nicolas Rougier's [*Scientific Visualization* book](https://github.com/rougier/scientific-visualization-book) — free, deep, opinionated. Chapter 3 on "Figure design" is exactly this lecture in twice the words.
