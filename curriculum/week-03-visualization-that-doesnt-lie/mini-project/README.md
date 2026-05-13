# Mini-Project — Recreate Three Published Charts, and Critique Each

> Pick three published charts from real news sources (Financial Times, New York Times *Upshot*, Our World in Data, Reuters Graphics, BBC, *The Economist*). Recreate each from scratch in matplotlib with synthetic or downloaded data. Then write a one-page critique of each original. The deliverable is a single notebook that contains six images (three reproductions, three originals as screenshots), three critiques, and a short reflection on what you learned.

This is the third artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 1 was an image notebook; Week 2 was an EDA; Week 3 is the chart-recreation-and-critique. Recruiters with taste — that is, the kind you want — read this one carefully. It is the closest portfolio piece to the actual day-to-day of a data analyst with chart literacy.

**Estimated time:** 7 hours, spread across Thursday–Saturday.

---

## What you will build

A Jupyter notebook `charts.ipynb` plus a rendered `report.md` that, for three published charts:

1. **Identifies** each chart: source, title, stable URL, date you accessed it.
2. **Reproduces** each chart from scratch in matplotlib + seaborn, with synthetic or downloaded data, in the editorial style described in Lecture 3 (top/right spines hidden, one focal color, sentence-form title, data-source caption).
3. **Saves** each reproduction next to a screenshot of the original.
4. **Critiques** each original using the publication checklist (Lecture 3, Section 10) and the four-lies framework (Section 3) — what works, what could mislead, what your editor would change.
5. **Concludes** with a one-page reflection on the three charts as a set.

The notebook is the working artifact. The `report.md` is the executive summary a non-technical reader can finish in five minutes.

---

## Pick three charts

The three should illustrate **three different chart families** — a line chart and a horizontal bar and a small-multiples, for instance, not three line charts. Constraints:

- **Free to view.** No paywalled article. If you find a perfect chart on a paywalled FT story, find a similar one in the [FT Chart Doctor](https://github.com/Financial-Times/chart-doctor/) repo, which is open-source.
- **Recent.** Charts from 2020 or later use modern style conventions. Pre-2010 charts are valid as historical references but less useful as targets.
- **Reputable.** Sources that count: Financial Times, New York Times (*Upshot* especially), Our World in Data, Reuters Graphics, BBC Visual and Data Journalism, *The Economist*, *The Pudding*, Bloomberg Graphics, ProPublica. If you pick something else, the chart should be from an organization that has a named data desk.

Suggested chart pool (use these or your own):

- **Our World in Data — "CO2 emissions per capita by country"** (line chart with many countries; small multiples is the honest version): <https://ourworldindata.org/co2-emissions>.
- **FT Chart Doctor — "Wages vs inflation"** (slope chart): <https://github.com/Financial-Times/chart-doctor>.
- **FT Chart Doctor — "How rising temperatures affect grape harvests"** (line + annotation).
- **NYT Upshot — Election forecast dial** (this one is interesting *because* it is contested; the dial encoding was widely criticized).
- **Reuters — "How many trees can the world afford to lose?"** (small multiples, choropleth).
- **The Economist — daily chart series**, easily searchable on their site.

Document your three picks in the notebook's first cell. The point is the chart, not the data — for the reproduction you may estimate the numbers from the chart itself.

---

## Acceptance criteria

- [ ] A new directory `week-03/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `pandas>=2.2,<3`, `numpy>=2,<3`, `matplotlib>=3.8,<4`, `seaborn>=0.13,<1`, `jupyter`.
- [ ] `jupyter nbconvert --to notebook --execute week-03/charts.ipynb` runs end-to-end without errors on a fresh clone.
- [ ] The notebook contains **three reproductions** and **three screenshots of originals**, six images in total, saved as PNGs in `week-03/images/`.
- [ ] All three reproductions use the **object-oriented matplotlib API** end-to-end. No `plt.plot(...)` in the notebook outside `plt.subplots`, `plt.style.use`, and `plt.show`.
- [ ] All three reproductions pass the **publication checklist** (Lecture 3, Section 10): sentence-form title, labeled axes with units, top/right spines off, one focal-color series, source caption, no chartjunk.
- [ ] Each critique addresses each of the four common lies — "this chart does/does not exhibit lie N because …" — and adds at least one subtler observation per chart.
- [ ] A `report.md` (~2 pages, 600–900 words) summarizes the three originals, the three critiques, and the meta-reflection. No code. A non-technical reader should be able to finish it in five minutes.
- [ ] A `README.md` in `week-03/` explains: setup, the three chart URLs (with access dates and license notes), file layout, how to reproduce.

---

## Suggested layout

```text
crunch-ai-portfolio-<yourhandle>/
├── README.md                    ← portfolio root
└── week-03/
    ├── README.md                ← week-3 explainer
    ├── requirements.txt
    ├── images/
    │   ├── chart-1-original.png
    │   ├── chart-1-reproduction.png
    │   ├── chart-2-original.png
    │   ├── chart-2-reproduction.png
    │   ├── chart-3-original.png
    │   └── chart-3-reproduction.png
    ├── data/                    ← any small CSVs you used
    ├── charts.ipynb
    ├── charts.html              ← rendered preview
    └── report.md                ← 2-page executive summary
```

---

## Suggested order of operations

### Phase 1 — Setup and chart selection (45 min)

1. Open `week-03/charts.ipynb`. The first markdown cell is the **project header**: name, your handle, the three chart URLs, the dates you accessed them, any license notes (most reputable outlets allow fair-use educational reproduction; cite them).
2. The next code cell does the imports, the brand-color constants, and `plt.style.use("crunch")` if you have the stylesheet installed. Pin `matplotlib>=3.8`, `seaborn>=0.13` in your `requirements.txt`.
3. Save the three screenshots of the originals into `images/chart-N-original.png`. The screenshots are committed; the data is yours.

### Phase 2 — Three reproductions (3 hours, ~1 hour each)

For each of the three charts, a section with this structure:

```text
## Chart 1: <name of the chart and source>

[markdown] Brief paragraph: what is the chart, who made it, what is its claim.

[markdown] An <img src="images/chart-1-original.png"> with the source caption.

[code] Generate / load the data.

[code] Make the reproduction with fig, ax = plt.subplots(...).
       Save to images/chart-1-reproduction.png.

[markdown] An <img src="images/chart-1-reproduction.png"> next to the original
           (use a Markdown table or two side-by-side <img> tags).

[markdown] One sentence on how close the reproduction is to the original
           and what you simplified.

[markdown] ## Critique
           - Encoding: position / length / angle / area / color hue.
           - What it does well: 2-3 specific items.
           - The four lies check: truncated bar / dual axis / rainbow / 3-D.
             For each, "does/does not apply because ...".
           - One subtler observation (data source, framing, missing context).
           - One concrete edit your editor would make.
```

For the data: if the chart is from Our World in Data, the data is open and you can use the actual CSV. If the chart is from the FT or NYT, you will usually need to estimate the values from the chart itself (eyeball it; a 5% reading error is fine for an educational reproduction). Document which approach you took in the notebook.

### Phase 3 — Meta-reflection (45 min)

A final section titled "What I learned from three charts as a set." One page of prose covering:

1. **Style commonalities.** What do the three originals do that you would copy in your own work?
2. **One difference.** Where do they diverge in convention? (e.g., the FT uses inline annotations; OWID uses a sidebar with explainer text.)
3. **The trickiest reproduction.** Which one was hardest to recreate and why?
4. **One thing you will steal.** The single most concrete habit from these reference outlets you will use this week.

This is the C5 brand voice: numbers cited, not asserted; acknowledge what you do not know. "The FT uses inline annotation 3 times in this chart; my reproduction has 1; the FT's is denser-but-readable, which is the harder craft." That is what you are aiming for.

### Phase 4 — Export and report (1.5 hours)

```bash
jupyter nbconvert --to html week-03/charts.ipynb
```

Write `report.md`: 600–900 words. Sections: **Project**, **Chart 1**, **Chart 2**, **Chart 3**, **Reflection**. No code. Each section is one paragraph on the original and one on your critique. Imagine an executive (or a hiring manager) who has five minutes and will not open the notebook.

---

## What "great" looks like

A "great" mini-project hits all of the following:

- The reproductions are recognizably the same chart as the originals — same chart type, same broad shape, comparable annotations.
- The reproductions are *cleaner* than the originals on one or two checklist items. (You are allowed to improve on the FT. They make mistakes too.)
- Each critique identifies at least one thing the original does well and at least one thing it could do better. Neither uncritical nor contrarian.
- The meta-reflection makes a claim about the three charts as a set that you could not have made about any one of them.
- The `report.md` reads like an editor's column, not a tutorial.

A "good but not great" project has reproductions that match the originals but no improvements; critiques that list the four-lies framework without finding subtler observations; a reflection that summarizes the three charts without synthesis.

A "needs work" project has reproductions that miss the chart type, critiques that praise without criticizing, or any chart-junk in the reproductions themselves.

---

## Stretch goals

- For one of your three charts, produce a **second reproduction** in plotly. Side-by-side with the matplotlib version; comment on which is more appropriate for the original publication context.
- Add a fourth "anti-chart" — your single least favorite published chart from anywhere on the web — and write the critique. The point is that the muscles you built work in both directions.
- Find the actual source data for one of the three originals (most OWID, FT Chart Doctor, and BBC charts are open-source on GitHub) and reproduce with the real numbers. Note any numeric discrepancies between your eyeballed reproduction and the real-data version.
- Convert one of your three reproductions to an SVG (`fig.savefig("chart.svg")`) and edit one element in Inkscape or a text editor (an SVG is XML). Useful skill for the day a designer needs you to ship to a CMS.

---

## Rubric

| Criterion | Weight | "Great" looks like |
|-----------|------:|--------------------|
| It runs | 15% | Fresh clone → `pip install -r requirements.txt` → `jupyter nbconvert --execute` → no errors |
| Reproductions | 25% | Three charts, three different chart families, each recognizably the same as the original |
| Style discipline | 15% | Every reproduction passes the publication checklist; OO API end-to-end |
| Critique depth | 25% | Each critique handles the four lies and adds a subtler observation; not uncritical, not contrarian |
| Reflection | 10% | A synthesis claim about the three charts as a set |
| Report readability | 10% | The 2-page `report.md` could go to a hiring manager without explanation |

---

## What this exercises (and what comes next)

This mini-project exercises every concept from Week 3:

- The Figure / Axes / Artist model.
- The OO matplotlib API.
- The seaborn / plotly picker (most reproductions will be matplotlib; one stretch may be plotly).
- Perceptual encoding choices.
- The four lies.
- Color family selection.
- Annotation as story.
- The publication checklist.

Week 4 will start with linear models, but the chart literacy you build this week shows up in every model-evaluation plot, every residual diagnostic, every learning-rate curve, and every fairness audit you will draw for the rest of the course. The discipline of "this chart will be defended by a sentence and a source line" is one habit; it generalizes.

---

## Submission

Push your `crunch-ai-portfolio-<yourhandle>` repo. Open the `week-03/charts.html` link from the repo README so a reviewer with no Python install can read your work. Paste the URL into your Week 3 review thread (or the C5 community channel if you are doing this in a cohort). That is the artifact recruiters will see — and the chart literacy it shows is one of the highest-signal items in a data-scientist portfolio.
