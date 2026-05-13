# Lecture 3 — Perceptual Encoding and Not Lying With Charts

> **Outcome:** You can rank visual encodings by perceptual accuracy, pick a color scheme on purpose (sequential / diverging / qualitative), name the four most common ways a chart misleads, and decide on the y-axis baseline, the scale (linear vs log), and the number of y-axes with a sentence of reasoning each.

A chart is a picture you make from numbers. The numbers might be honest, the labels might be correct, and the chart might still mislead the reader — because the *encoding* (the mapping from number to visual property) is doing work the reader cannot see. This lecture is about reading and writing that encoding on purpose.

We will reference two classics:

- **Cleveland and McGill (1984).** "Graphical Perception: Theory, Experimentation, and Application." The paper that put a partial order on visual encodings. The order is the spine of this lecture.
- **Tufte (1983).** *The Visual Display of Quantitative Information.* The vocabulary of "chartjunk," "data-ink ratio," and "small multiples." Not free, but the [Tufte CSS gallery](https://edwardtufte.github.io/tufte-css/) reproduces the visual ideas.

---

## 1. The perceptual ladder (Cleveland and McGill, abridged)

When you encode a number into a chart, you can use:

1. **Position along a common scale** (e.g. y-axis on a scatter plot). Most accurate.
2. **Position along non-aligned scales** (small multiples).
3. **Length** (bar chart).
4. **Angle / slope** (line chart, pie chart — angle).
5. **Area** (bubble chart, treemap).
6. **Volume** (3-D anything).
7. **Color saturation** (heatmap, choropleth).
8. **Color hue** (categorical — but only for distinction, not magnitude).

Position is the most accurately decoded by human vision. The reader can compare two y-axis values within a few percent. Length comes second; bar charts are very legible because the lengths share a baseline. Angle is third — which is why pie charts are not the disaster they are sometimes claimed to be, but they are still worse than bars. Area is fourth and falls off fast: humans systematically underestimate large circles compared to small ones. Volume is a disaster — never encode a magnitude as a 3-D volume. Color saturation is fine for *ordered* magnitudes but only when the differences are large. Color hue (red vs blue vs green) is for *categories*, not magnitudes.

The single most useful rule that falls out of this:

> **When you have a magnitude to encode, prefer position to length to area to color. Move up the ladder whenever you can.**

A "share of total" question encoded as a pie chart (angle) is fine for two or three slices. The same question encoded as a stacked bar (length on a common baseline) is better. A horizontal bar of "share by category" (position) is better still.

---

## 2. The data-ink ratio

Tufte defines **data ink** as the ink on the page that encodes data; everything else is **non-data ink**. The data-ink ratio is the fraction:

```text
data-ink ratio = data ink / total ink
```

You want it close to 1. In practice:

- Remove the chart border (the top and right spines).
- Remove drop shadows.
- Remove 3-D effects.
- Remove decorative gridlines (or fade them to `alpha=0.2`).
- Remove the legend if there is one series.
- Remove the axis label if the column name *is* the label.

Every line of ink should defend its existence: "I encode a number" or "I help the reader read a number." If neither, delete it.

The reverse — **chartjunk** — is ink that defends nothing. The 3-D pie chart in your last manager's deck has approximately 12% data ink and 88% chartjunk. The data-ink ratio is the metric that explains why it bothers you.

---

## 3. The four common lies

You will see all four this year. Three of the four are produced by software defaults, not by intent — which makes them more common, not less harmful. Learn to spot them; learn to not produce them.

### Lie 1: Truncated y-axis on a bar chart

Bars encode magnitude as **length from the baseline**. If the baseline is not zero, the lengths lie. A bar that goes from 100 to 105 looks five times as tall as a bar that goes from 100 to 101 if the y-axis starts at 100 — and the differences are 5% and 1% respectively, not 500%.

```python
# WRONG: truncated baseline on a bar
fig, ax = plt.subplots()
ax.bar(["A", "B"], [105, 101])
ax.set_ylim(100, 106)   # <- the lie

# RIGHT: zero baseline on a bar
fig, ax = plt.subplots()
ax.bar(["A", "B"], [105, 101])
ax.set_ylim(0, 110)
```

The rule for bars is non-negotiable: **bar charts start at zero**. If your data has a small dynamic range relative to its absolute value (e.g., "interest rate moved from 4.1% to 4.3%"), do not draw a bar. Draw a dot plot or a line — encodings whose magnitude is read as position, not length.

### Lie 2: Dual-axis correlation theater

Two series on the same chart with two y-axes, each scaled independently, "showing" that the series move together. The reader sees correlation that does not exist — because you scaled the axes to make it look like one.

```python
# WRONG: two y-axes, scaled to look correlated
fig, ax1 = plt.subplots()
ax1.plot(years, gdp, color="C0")
ax1.set_ylabel("GDP", color="C0")

ax2 = ax1.twinx()
ax2.plot(years, ice_cream_sales, color="C1")
ax2.set_ylabel("Ice cream sales", color="C1")
```

Three problems. First, you are picking the scaling — you can make any two series look correlated by squinting at the y-axes. Second, the reader cannot compare slopes across two scales. Third, the implied causal claim is uncontrolled.

Three honest alternatives:

- **Normalize.** Convert both series to "% change from year 0" (an index, both ending up on the same y-axis). The reader compares slopes truthfully.
- **Two small multiples** stacked. Same x-axis, two separate Axes. The reader sees both series; the comparison is honest.
- **Scatter plot.** If you actually want to claim correlation, draw the scatter (`x = series 1, y = series 2`). The R^2 makes the claim explicit.

Dual-axis charts are not always lies; the BBC and the FT use them carefully. The carefulness comes from explicit scale relationships ("right axis is 1000× left axis, fixed") and obvious unit differences ("temperature in °C and rainfall in mm"). The casual dual-axis is the dangerous one.

### Lie 3: Rainbow colormaps on ordinal data

The matplotlib `jet` colormap and its cousins (`hsv`, `rainbow`, `nipy_spectral`) cycle through hue: blue → cyan → green → yellow → red. Used on an ordered magnitude, they introduce false structure: yellow (in the middle) appears brighter than its neighbors, so the eye sees a band in the middle of an otherwise smooth gradient.

```python
# WRONG: jet on an ordered magnitude
ax.imshow(temperature_field, cmap="jet")

# RIGHT: viridis (or any perceptually uniform colormap)
ax.imshow(temperature_field, cmap="viridis")
```

The matplotlib team made `viridis` the default in 2015 for exactly this reason. `viridis` is **perceptually uniform** (equal steps in the data are equal steps in perceived lightness) and **color-blind safe**. For diverging data (deviations from a baseline) use `RdBu` or `coolwarm`. For sequential data use `viridis`, `plasma`, `magma`, or `cividis`. Never `jet`.

The full taxonomy:

- **Sequential** (one direction; low → high): `viridis`, `plasma`, `magma`, `cividis`, `Blues`, `Greys`.
- **Diverging** (two directions from a midpoint): `RdBu`, `coolwarm`, `PuOr`, `BrBG`. Use when the data has a meaningful midpoint (zero, a baseline, a forecast).
- **Qualitative** (distinct, unordered categories): `tab10`, `Set2`, `Paired`. Use for up to ~8 categories. Beyond that, the eye cannot distinguish them.

The [matplotlib colormap reference](https://matplotlib.org/stable/users/explain/colors/colormaps.html) shows every option. Read it once; bookmark it.

### Lie 4: 3-D anything

A 3-D pie chart, a 3-D bar chart, a 3-D surface plot of two-variable data. The added dimension does not encode data; it encodes "I wanted this to look impressive." Perspective distorts magnitudes in ways the reader cannot correct for. The bars in the back of a 3-D bar chart look smaller than equivalent bars in the front. The "back" slice of a 3-D pie looks larger because of foreshortening.

The fix is one word: **don't**. If you have three numeric dimensions, use a 2-D chart with the third dimension on color (heatmap) or size (bubble chart). If you have a genuine 3-D surface, draw a 2-D contour plot or several small multiples at fixed slices. Static 3-D charts on paper are almost always wrong. Interactive 3-D (plotly, rotatable) is the only honest 3-D, and even then, ask first whether 2-D would do.

---

## 4. Linear vs log: the scale-choice decision

You have two y-axis scale options for a quantitative variable:

- **Linear.** Equal vertical steps mean equal additive differences. The default.
- **Log.** Equal vertical steps mean equal multiplicative differences (factors of 10, 2, *e*, etc.). Used when the data spans many orders of magnitude or when ratios matter more than absolute values.

When to switch to log:

- **The data spans 3+ orders of magnitude.** "Country GDP in USD" ranges from $10^9$ (small countries) to $10^{13}$ (US, China). On a linear axis only the top two countries are visible; on a log axis all countries are visible.
- **The relationship of interest is multiplicative.** Exponential growth (epidemic case counts, compound interest) is a straight line on a log y-axis. Reading the slope reads the growth rate directly.
- **The reader will compare *ratios*** between values. "How much more does the US emit per capita than India?" The eye reads ratios on a log scale; it reads differences on a linear scale.

When to stay linear:

- **The data is bounded** (probabilities, percentages, scores 0–100).
- **The reader needs to read the absolute value at a glance** (revenue in dollars, headcount in people).
- **The story is additive, not multiplicative.** "Revenue increased by $5M" is a linear-axis story; "Revenue doubled" is a log-axis story.

A common mistake: using log on bars. Bar length encodes magnitude as a difference from zero, but on a log axis "zero" is at $-\infty$ — there is no baseline. A bar chart on a log y-axis is one of the dishonest figures in the wild. If you find yourself reaching for one, switch to a dot plot or a line.

Annotate the axis. The reader should know in two seconds whether they are reading linear or log:

```python
ax.set_yscale("log")
ax.set_ylabel("Cases per day (log scale)")
```

The "(log scale)" is non-optional.

---

## 5. Color, in detail

Color is the encoding most often used badly. Three rules.

### Rule 1: pick the right family

- **Sequential** for an ordered magnitude (temperature, density, count). One color, varying in lightness.
- **Diverging** for deviations from a baseline (anomaly, surplus/deficit, residual). Two colors meeting at the midpoint.
- **Qualitative** for categories with no order (region, payment method, species).

Picking the wrong family is half the color mistakes you will see. "Region" is qualitative; encoding it with `viridis` is a mistake. "Temperature anomaly" is diverging; encoding it with `viridis` is also a mistake (the midpoint should be visually neutral, not green).

### Rule 2: pick a color-blind-safe palette

Roughly 5% of men have some form of color-vision deficiency. The most common is red-green. Palettes that distinguish red from green only by hue are unreadable for that 5%.

Color-blind-safe defaults:

- `viridis`, `plasma`, `magma`, `cividis` (sequential) — all designed to be CB-safe.
- `tab10` — the default qualitative palette in matplotlib 2.0+, CB-safe.
- `RdBu`, `coolwarm` — diverging; red and blue are distinguishable to all forms of CB.

Avoid: `rainbow`, `jet`, `Spectral`, and any custom red-green diverging palette. The matplotlib [colormaps page](https://matplotlib.org/stable/users/explain/colors/colormaps.html) is the canonical reference; pick from there.

### Rule 3: pick a brand color, use it sparingly

For our work, **Plot Violet `#7C3AED`** is the brand color (see [BRAND.md](../../../branding/BRAND.md)). Use it for the **focal series** — the one the reader should look at first. Use gray (`#9CA3AF`) for context series. Other categories get other CB-safe hues, but the focal series is always violet.

```python
# focal series in brand violet, context in gray
ax.plot(years, our_company, color="#7C3AED", linewidth=2, label="Our company")
ax.plot(years, competitor_1, color="#9CA3AF", linewidth=1, label="Competitor 1")
ax.plot(years, competitor_2, color="#9CA3AF", linewidth=1, label="Competitor 2")
```

This is the FT / OWID house move: one bright color for the story, gray for the context.

---

## 6. Axis ranges, baselines, and the "fair frame"

The chart's framing — what range of x and y the reader sees — is part of the argument. Three honest defaults:

- **Bar charts: y starts at 0.** Non-negotiable.
- **Line charts: y starts wherever the data starts, with some padding (5–10%).** Truncating a line chart's y-axis is acceptable because the encoding is position-and-slope, not length-from-zero. But say so: a line going from 99.8% to 99.9% over a y-range of 99.5%–100% reads differently than the same line on a 0–100% range; either can be the right choice for a given story; the writer is responsible for picking on purpose.
- **Scatter charts: pad the data by ~5%.** matplotlib does this by default. Do not force `(0, max)` unless zero is meaningful.

The "fair frame" question is: does the reader, seeing the chart for two seconds, draw the same conclusion you would draw from the underlying data? If yes, the frame is fair. If the chart implies a story that the data does not support (or hides one that it does), the frame is unfair.

The "two-second test" is more useful than any rule. Show the chart to a colleague, count to two, ask them what they remember. If they remember the wrong thing, the encoding is wrong, regardless of whether every label is correct.

---

## 7. Annotation as the single highest-leverage chart improvement

Most charts under-annotate. The reader is left to derive the takeaway from the line. Adding one annotation — an arrow, a number, a sentence — moves the chart from "data" to "story."

```python
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(years, co2, color="#7C3AED", linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Global CO2 emissions (Gt)")
ax.set_title("Global CO2 emissions briefly fell during the 2020 pandemic")

# the annotation
peak_year = years[co2.argmax()]
dip_year  = 2020
ax.annotate(
    "2020: −5.4% (lockdowns)",
    xy=(dip_year, co2[years.tolist().index(dip_year)]),
    xytext=(dip_year - 8, co2.max() * 0.8),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=10, color="black",
)
```

`annotate` takes `xy` (the point being annotated, in data coordinates), `xytext` (where the label sits, also in data coordinates), and an optional arrow. Two extra lines for an entire layer of clarity. The Cleveland-and-McGill paper does not cover this; the FT and NYT habit does. **Annotate the takeaway.**

---

## 8. Small multiples: the right answer when one chart is too crowded

When you have one chart with too many lines (say, "CO2 emissions for 50 countries over 60 years"), the right answer is rarely "make the lines thinner" or "use more colors." The right answer is **small multiples**: a grid of the same chart, one panel per category.

```python
import seaborn as sns
g = sns.FacetGrid(co2_long, col="country", col_wrap=5, height=2, aspect=1.5)
g.map_dataframe(sns.lineplot, x="year", y="co2", color="#7C3AED")
g.set_titles("{col_name}")
g.savefig("co2_small_multiples.png", dpi=150)
```

Tufte called small multiples "the most powerful design device in statistical graphics." They convert a single crowded chart into many readable ones; the reader's eye does the comparison across panels. The seaborn `FacetGrid` and `relplot` / `catplot` (with `col=` / `row=`) make small multiples one line of code.

The matplotlib equivalent is `plt.subplots(nrows=5, ncols=10, sharex=True, sharey=True)` plus a loop. Slightly more code, full control. For five panels or fewer, matplotlib is fine; for fifty, use seaborn's facet machinery.

---

## 9. A worked critique: spotting all four lies on one chart

You may have seen this chart (or one like it):

> A 3-D pie chart labeled "Q3 sales by region," with five wedges. The legend uses six colors (one extra is for "other"). The slices are exploded outward by 10%. There is a drop shadow.

What is wrong, in order:

1. **3-D.** The depth distorts wedge sizes. The slice at the front looks larger.
2. **Pie chart.** Five wedges is at the limit of acceptable; an exploded pie makes the angles harder to read.
3. **Six colors.** Categorical with six colors near the upper end of perceptual distinguishability; if any pair is similar (red and orange, blue and purple), the reader cannot tell them apart in the wedge.
4. **Drop shadow.** Chartjunk. Encodes nothing.
5. **No data labels in the wedges.** The reader must triangulate "slice color" → "legend entry" → "value."
6. **No source.** Where does the data come from? Nobody knows.

The honest version: a horizontal sorted bar chart with one bar per region, brand violet, value labels at the end of each bar, source in a small caption. Same data, four lies removed.

The point of this lecture is to make that diagnosis automatic. Look at a chart; spot the lies; propose the honest replacement. By the mini-project critique you should be able to do it in ninety seconds.

---

## 10. The full publication checklist (revisited)

Compose Lectures 1, 2, and 3 into one checklist for every chart you ship:

- [ ] **Question.** What is the chart answering? Distribution / relationship / composition / time / ranking?
- [ ] **Chart type.** Matches the question (see Lecture 2, Section 6).
- [ ] **Library.** matplotlib + seaborn for static; plotly for interactive (see Lecture 2, Section 5).
- [ ] **Title.** A sentence with the takeaway, not a column name.
- [ ] **Axis labels.** Both axes, in English, with units.
- [ ] **Ticks.** Readable: thousands separators, date formatters, sensible rotation.
- [ ] **Color.** Right family (sequential / diverging / qualitative). Brand violet for focal series. CB-safe.
- [ ] **Baseline.** Bars start at zero. Lines start with padding. Decision conscious.
- [ ] **Scale.** Linear unless data spans 3+ OoM or ratios matter. "(log scale)" in the label if log.
- [ ] **Annotation.** At least one — the takeaway, the inflection, the outlier — with an arrow.
- [ ] **Legend.** Present when needed, absent when redundant; no frame.
- [ ] **Spines.** Top and right hidden (editorial style).
- [ ] **Gridlines.** Faint (`alpha=0.2–0.3`) or absent.
- [ ] **Caption with source.** Lower-right corner, gray, 8pt.
- [ ] **No chartjunk.** No 3-D, no shadows, no decorative borders.
- [ ] **No dual y-axis** unless you can justify it in one sentence.
- [ ] **File format.** PNG 150dpi for the web, SVG for print, plotly HTML for interactive.

That is the entire ship-list. The first three weeks of this course produce charts you can defend to a senior data scientist; this checklist is the way.

---

## 11. Self-check (five-minute drill)

For each chart, name the encoding mistake and the fix:

1. A bar chart of "average customer satisfaction" with a y-axis from 4.5 to 5.0.
2. A 3-D pie chart of "browser market share."
3. A line chart with two y-axes showing "S&P 500" and "Number of pirate attacks per year."
4. A heatmap of "monthly temperature by city" using the `jet` colormap.
5. A line chart of "country GDP, 1960–2024" with 50 lines, one per country, none labeled.
6. A scatter plot of "income vs life expectancy" with the dot size proportional to a country's population, in cubic units (volume).

Answers:

1. **Truncated baseline on a bar.** Fix: start the y-axis at 0 (or switch to a dot plot if dynamic range is small).
2. **3-D and pie.** Fix: a horizontal sorted bar.
3. **Dual-axis correlation theater.** Fix: normalize both series to "% change from start" and put them on the same axis, or two small multiples.
4. **Rainbow colormap on ordered data.** Fix: `viridis` (sequential) or `RdBu` (diverging, if there is a meaningful midpoint).
5. **Too many lines, no annotation.** Fix: small multiples (one panel per country, or per region group), or highlight 2–3 of interest and gray out the rest.
6. **Magnitude as volume.** Fix: encode population as area (radius proportional to √population) or, better, as bar length next to the scatter.

If you got 5+ correct on the first pass, you have internalized the lecture.

---

## Recap

- Perceptual encodings rank: position > length > angle > area > volume > color saturation > color hue.
- Pick the colormap family on purpose: sequential / diverging / qualitative.
- The four lies: truncated bar baseline, dual-axis theater, rainbow on ordinal, 3-D anything.
- Log vs linear is a deliberate choice; bars on a log scale are dishonest.
- Annotation is the highest-leverage upgrade to most charts.
- Small multiples beat crowded single charts almost every time.

Next: the exercises and the mini-project, where you take all three lectures and produce charts you would put in front of an executive.

---

## Further reading (optional, free)

- Cleveland and McGill (1984). "Graphical Perception" — the seminal paper, free PDFs are widely available; search the title.
- The matplotlib [colormaps page](https://matplotlib.org/stable/users/explain/colors/colormaps.html).
- The FT [Visual Vocabulary PDF](https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/Visual-vocabulary.pdf).
- Claus Wilke's *Fundamentals of Data Visualization*, chapters 4 (color), 17 (distributions), 19 (proportions). [Free online](https://clauswilke.com/dataviz/).
