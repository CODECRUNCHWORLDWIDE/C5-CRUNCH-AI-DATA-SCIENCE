# Week 3 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** Which spelling is the API you should ship?

- A) `plt.plot([1, 2, 3]); plt.title("X"); plt.show()`
- B) `fig, ax = plt.subplots(); ax.plot([1, 2, 3]); ax.set_title("X")`
- C) `pd.DataFrame({"x": [1, 2, 3]}).plot()`
- D) Either — matplotlib's two APIs are equivalent in modern usage.

---

**Q2.** A `Figure` in matplotlib is best described as:

- A) A single coordinate system on which lines and points are drawn.
- B) The container that holds one or more `Axes`, plus the suptitle.
- C) A synonym for the entire pyplot module.
- D) The output image file written by `savefig`.

---

**Q3.** You are making a chart that ranks ten product categories by revenue, with one bar per category. Which layout choice is best?

- A) Vertical bars, alphabetical order, default colors.
- B) Vertical bars sorted by revenue descending, all bars the same color.
- C) Horizontal bars sorted by revenue ascending (largest at the top), one focal-color bar for the leader.
- D) A pie chart with each category as a wedge.

---

**Q4.** A bar chart shows `[A: 105, B: 103, C: 101]` with the y-axis ranging from 100 to 106. What is wrong?

- A) Nothing — choosing the y-axis range to fit the data is good practice.
- B) The bars are encoding **length** from the baseline, but the baseline is not zero — small differences look much larger than they are.
- C) The colors are not branded.
- D) Bar charts cannot show three categories.

---

**Q5.** You have a chart with one line for each of 200 country-level series over 60 years. The result is a hairball. What is the right fix?

- A) Make the lines thinner.
- B) Use the `rainbow` colormap so every country gets a distinct hue.
- C) Switch to **small multiples** — a grid of small charts, one per country (or per region).
- D) Switch to a 3-D line chart with country as the third axis.

---

**Q6.** Which colormap is the right choice for a heatmap of monthly average temperature (a single-direction ordered magnitude)?

- A) `jet`
- B) `viridis`
- C) `tab10`
- D) `Paired`

---

**Q7.** A colleague shows you a chart with two y-axes: "S&P 500 close" on the left, "Total wine consumption in France" on the right, both plotted as lines over 1990–2024. They claim the two move together. The honest critique is:

- A) "Looks great — two-axis charts are the standard for showing co-movement."
- B) "Two independent scales make any two series look correlated when you pick the scaling carefully. Normalize both to '% change from 1990' and put them on one axis, or draw a scatter."
- C) "The lines should be thicker."
- D) "Add a third axis for context."

---

**Q8.** When does plotly earn its keep over matplotlib + seaborn?

- A) Always — plotly is the modern standard for all Python visualization.
- B) When the chart is going into a printed report.
- C) When the chart will live on a web page and readers will hover, zoom, or pan to read it.
- D) When you need the highest possible image resolution.

---

**Q9.** Which chart type is the right answer for "what is the joint distribution of two numeric variables when n ≈ 100,000 and a scatter plot is a solid blob?"

- A) A bar chart.
- B) A 3-D surface plot.
- C) A hexbin (or 2-D KDE).
- D) A pie chart for each value of x.

---

**Q10.** You finished a chart. Which **single item** from the publication checklist is the most commonly skipped — and therefore the highest-leverage thing to add before you ship it?

- A) A drop shadow.
- B) A short caption with the data source ("Source: …, accessed YYYY-MM-DD").
- C) A 3-D effect on the focal series.
- D) A second y-axis for a related metric.

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — The object-oriented API (`fig, ax = plt.subplots(); ax.plot(...); ax.set_title(...)`) is the spelling that scales from one chart to many, makes the Figure and Axes objects nameable, and produces the same output every time. Option A is the pyplot state-machine — fine at the REPL, wrong in a script. Option C is pandas' `.plot` method, a useful one-liner that returns to (A) or (B) the moment you need more than the default look. Option D is wrong: the APIs are not equivalent in usage, even though they share an underlying object model.

2. **B** — A `Figure` is the container; an `Axes` is the chart. One Figure can hold many Axes. The `suptitle` is a Figure-level title above all Axes; per-Axes titles are set with `ax.set_title`. Option A confuses Figure with Axes; option C confuses Figure with the module; option D confuses Figure with the file written by `savefig`.

3. **C** — Horizontal bars give the long category labels room. Sorting ascending (largest at the top) is the convention because the eye reads top-down. One focal-color bar (brand violet) for the leader; gray for the rest is the FT / OWID move and makes the takeaway visible in two seconds. Option A throws away the ranking; option B is fine but vertical bars rotate long labels; option D is the wrong chart type for ten categories (use bars).

4. **B** — Bar charts encode magnitude as length from the baseline. A baseline that is not zero turns small differences into big visual differences. The rule is non-negotiable: bars start at zero. If the dynamic range is small relative to the absolute value, switch to a dot plot or a line — encodings whose magnitude is read as position, not length-from-baseline. Lecture 3, Section 3, Lie 1.

5. **C** — Small multiples (a grid of small versions of the same chart, one panel per category) is Tufte's "the most powerful design device in statistical graphics." Seaborn's `FacetGrid` / `relplot(col=, row=)` makes this one line. Option A does not fix the underlying overcrowding; option B makes it worse (rainbow is a perceptual-encoding mistake); option D adds visual noise without adding clarity.

6. **B** — `viridis` is a sequential, perceptually uniform, color-blind-safe colormap. It is the default in matplotlib 2.0+ for exactly the case in the question. `jet` (A) is the classic rainbow mistake — non-uniform perceived lightness, not CB-safe. `tab10` (C) and `Paired` (D) are qualitative palettes for categorical data, not for an ordered magnitude.

7. **B** — Two independently scaled y-axes are the textbook example of correlation theater. The reader cannot compare slopes across two scales, and the analyst is implicitly picking the scaling that makes the two series look related. The honest move is to normalize both series to percentage change from a baseline year and put them on one axis, *or* draw a scatter plot if a co-movement claim is the actual point. Lecture 3, Section 3, Lie 2.

8. **C** — Plotly's reason to exist is interactivity: hover tooltips, zoom, pan, legend toggling. Use it when the output is HTML and a reader will explore. For static reports (A is wrong as a blanket claim; B is the wrong direction), matplotlib + seaborn are smaller, sharper, and more reproducible. Option D confuses image resolution with file format — matplotlib will happily save SVG at any resolution.

9. **C** — A hexbin is a 2-D histogram on a hexagonal grid; it counts points per cell and encodes the count as color (sequential, with `viridis`). It is the right answer when scatter becomes a solid blob — the density structure is now visible. Option A is the wrong encoding (bars are for 1-D magnitude, not 2-D density). Option B is 3-D anything, almost always misleading. Option D is nonsensical.

10. **B** — The data-source caption is the single most-skipped item on the publication checklist and the one that, when missing, hurts credibility the most. Two seconds of typing; a long-term reputation for sourced work. The other three options are the *opposite* of what to ship — drop shadows, 3-D effects, and gratuitous second axes are chartjunk and dishonesty respectively.

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
