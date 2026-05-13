# Challenge 1 — Recreate an FT chart

**Time estimate:** 2 hours.

## Problem statement

Pick one chart from the *Financial Times* — either from their open-source [Chart Doctor repo](https://github.com/Financial-Times/chart-doctor) or from any FT data story you can read for free (the FT puts a few graphics per week outside the paywall, and the Chart Doctor's [Visual Vocabulary PDF](https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/Visual-vocabulary.pdf) reproduces dozens at thumbnail size). Reproduce it from scratch in matplotlib with synthetic data that has the same *shape* as the original. Then write a one-page critique of the original chart.

You will write a single script `recreate_ft_chart.py` plus a markdown file `critique.md` that:

1. Identifies the chart you picked, with a stable link.
2. Generates synthetic data with the same dimensions and broad shape.
3. Renders a matplotlib reproduction in the FT visual style (sans-serif, top/right spines off, one accent color on the focal series, gray on context).
4. Saves a side-by-side: your reproduction at `reproduction.png` and a screenshot of the original at `original.png`.
5. Writes the one-page critique.

This is the realistic version of half the day-to-day of a data journalist or analyst-with-taste. Get it right and the mini-project's three reproductions feel like more of the same.

## Which charts work well

Good chart families for this exercise:

- **Line charts with one focal series**: "How has X changed over time?" The FT's COVID dashboards (still archived on the GitHub repo) are textbook examples.
- **Horizontal sorted bars with a category highlight**: "Which countries are above the average?"
- **Slope charts**: "Before vs after for a ranked list of items."
- **Small multiples**: "X for many groups, one panel each."

Avoid for this challenge:

- Choropleth maps (too much extra machinery for two hours).
- Interactive charts (the point is static reproduction).
- 3-D anything (you should not be reproducing chartjunk).

## The data

You will generate the data yourself. Two reasons:

1. The FT's underlying data is often paywalled or behind a sign-up.
2. The point of the exercise is the **encoding**, not the data engineering. We did the data engineering in Week 2.

Sketch:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
# Example: a line chart of "global CO2 emissions, 1990-2024"
years = np.arange(1990, 2025)
co2 = 22.0 + 0.5 * (years - 1990) + rng.normal(scale=0.5, size=len(years))
co2[years == 2020] -= 1.5     # the pandemic dip
co2[years > 2020]  += 0.2     # the post-pandemic rebound
df = pd.DataFrame({"year": years, "co2_gt": co2})
```

Eyeball the FT chart, estimate the start value, the end value, the slope, any inflections. You are not asked to match the numbers exactly. You are asked to match the shape.

## Acceptance criteria

- [ ] A single script `recreate_ft_chart.py` runs end-to-end with `python recreate_ft_chart.py` and writes `reproduction.png` to the current directory.
- [ ] `python -m py_compile recreate_ft_chart.py` succeeds.
- [ ] The matplotlib chart uses the OO API end-to-end (no `plt.plot`).
- [ ] The chart passes the publication checklist from Lecture 3, Section 10:
  - Title is a sentence with the takeaway.
  - Both axis labels in English with units.
  - Top and right spines hidden.
  - At least one annotation pointing at the takeaway.
  - Source caption in the lower-right corner.
  - One brand-violet (`#7C3AED`) focal series; everything else gray.
- [ ] A file `original.png` next to the reproduction — either a screenshot or, for Chart Doctor entries, the PNG that ships in the repo.
- [ ] A `critique.md` file (~250 words, see the prompt below).
- [ ] In the commit message, paste the URL of the FT chart you picked.

## The critique prompt

Open `critique.md` and answer these in roughly 250 words. The goal is to think like an editor, not a fan.

1. **What does the chart claim?** One sentence. Read the title; read the lede paragraph if there is one. State the argument the chart is making.
2. **What is the encoding?** Position / length / angle / area / color hue / color saturation. Cite Lecture 3, Section 1.
3. **What does it do well?** Be specific. "The annotation on 2020 makes the pandemic dip obvious in under a second" is specific; "it's clear" is not.
4. **Where could it mislead?** Walk down the four common lies from Lecture 3 (truncated y, dual-axis, rainbow on ordinal, 3-D) and check each. If none apply, name a more subtle issue: data-source choice, framing, missing context.
5. **What would your editor change?** One concrete suggestion. The FT's charts are mostly excellent, so the criticisms will be subtle. That is fine — subtle criticism is the harder skill.

A "great" critique is honest about what works (most FT charts work) and specific about what could be tighter. A "bad" critique is either uncritical ("amazing chart, no notes") or contrarian for the sake of it ("the FT is overrated"). The FT is not overrated; they are the modern reference standard. The exercise is to look at a reference and find one thing to learn from it.

## Suggested layout

```python
"""Recreate an FT chart with synthetic data."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOT_VIOLET = "#7C3AED"
GRAY = "#9CA3AF"


def make_data():
    rng = np.random.default_rng(0)
    # ... fill in synthetic data matching the FT chart's shape.
    ...


def main() -> None:
    df = make_data()
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    # ... draw the chart in FT style.
    ax.set_title("The single-sentence takeaway")
    ax.set_xlabel("...")
    ax.set_ylabel("...")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # annotation, caption, save.
    fig.savefig("reproduction.png", dpi=150)


if __name__ == "__main__":
    main()
```

## Hints

<details>
<summary>If you cannot decide which FT chart to pick</summary>

Open the [Chart Doctor README](https://github.com/Financial-Times/chart-doctor/) and scroll to the "case studies" list. Each case study is a self-contained example with the chart and the data. The "Sahel groundwater" line chart and the "wages vs inflation" slope chart are both classic and well within two hours of work.

If you want to pull from the live FT site, the data desk's [@ftdata Twitter / Bluesky feed](https://www.ft.com/visual-and-data-journalism) often surfaces free-to-read graphics.

</details>

<details>
<summary>If your reproduction does not look "FT enough"</summary>

The FT visual style is mostly four things:

1. Top and right spines off. (`ax.spines["top"].set_visible(False)` etc.)
2. Sans-serif body, often Metric (a paid font; DejaVu Sans is a free near-match).
3. One bright accent color for the focal series; gray for context series.
4. Annotations *inside* the chart, pointing at the takeaway, not in a legend at the bottom.

You do not need the FT's actual brand palette. The C5 brand violet is fine. The point is the *structure*, not the pixel-exact reproduction.

</details>

<details>
<summary>If your annotation overlaps the line</summary>

Move the `xytext` of `ax.annotate(...)` further from the data, and lengthen the arrow. The matplotlib annotation docs show `connectionstyle="arc3,rad=0.2"` for a curved arrow, which often helps thread between data points. Or annotate just one extreme (the max, the min, the inflection) and trust the reader to find the rest.

</details>

## Stretch goals

- Add a small inset chart (`fig.add_axes([...])`) showing the same data on a different scale — say a log-axis inset on a linear main chart. This is a real FT technique for "the headline is linear but the structure is multiplicative."
- Save both the `reproduction.png` and a `reproduction.svg`. Diff them visually; the SVG is what would print in the newspaper.
- Find one other reproduction of the same chart on the web (there are blog reproductions for many famous FT charts) and write one paragraph comparing your version to theirs.
- Add a second chart to the script: a `subplot_mosaic` with the main chart on the left and a small inset summary on the right. The challenge is the layout, not the data.

## Why this matters

Most charts you will be asked to make in a real job are derivatives of charts a senior person already likes. "Make it look like the one in the FT" is half the design brief in a working data team. The skill is to look at a published reference, name the encoding choices, and reproduce them on purpose. The FT and a handful of other reference outlets (NYT Upshot, Our World in Data, Reuters Graphics) are the modern canon. An hour of looking at any of them teaches more than three hours of tutorial.

## Submission

Commit `recreate_ft_chart.py`, `reproduction.png`, `original.png`, and `critique.md` to your Week 3 repo. Paste the URL of the original FT chart in the commit message. In your portfolio README, link to the reproduction next to the original so a reviewer can see both side by side.
