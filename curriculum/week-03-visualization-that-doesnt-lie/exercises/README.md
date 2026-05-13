# Week 3 — Exercises

Three drills. Each is a `.py` file with TODOs you fill in. Do them in order — they get progressively closer to ship-quality chart work.

1. **[Exercise 1 — Figure and Axes](exercise-01-figure-and-axes.py)** — build a 2×2 subplot grid, label every axis, ship a PNG. The drill that puts the OO API in your fingers. (~40 min)
2. **[Exercise 2 — Distributions and Relationships](exercise-02-distributions-and-relationships.py)** — histogram, KDE, scatter, hexbin. Pick the right chart for the question on a Palmer-Penguins-style dataset. (~50 min)
3. **[Exercise 3 — Time Series](exercise-03-time-series.py)** — a line chart with a sensible date axis, one annotation, and a caption. (~40 min)

## Workflow

- Type, do not paste.
- Each file builds its own deterministic synthetic data so the tests never depend on the network.
- Each file has built-in `assert`-style checks at the bottom and saves a PNG to disk. Run with `python exercise-XX.py` and watch it print `OK`.
- Hint blocks live at the very bottom of each file, commented out. Read them only after fifteen minutes of trying a problem.
- Every chart you save should pass the publication checklist from Lecture 3 (Section 10) — labeled axes, sentence-form title, source caption, no chartjunk.

## Self-grading

After each exercise, open the saved PNG and ask: *could I drop this into a report tomorrow?* If yes, move on. If no, the chart is missing something on the publication checklist.

## Running with `pytest`

A minimal `pytest`-style smoke test is at the bottom of each file. From the `exercises/` directory:

```bash
pytest exercise-01-figure-and-axes.py
pytest exercise-02-distributions-and-relationships.py
pytest exercise-03-time-series.py
```

All three should pass before you move to the [homework](../homework.md) and the [mini-project](../mini-project/README.md).
