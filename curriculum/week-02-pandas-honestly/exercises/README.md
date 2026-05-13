# Week 2 — Exercises

Three drills. Each is a `.py` file with TODOs you fill in. Do them in order — they get progressively closer to real cleaning and analysis work.

1. **[Exercise 1 — Load and inspect](exercise-01-load-and-inspect.py)** — load a small NYC-taxi-style public CSV, run the five-minute first look, pick the right dtype for every column. (~40 min)
2. **[Exercise 2 — Clean the messy](exercise-02-clean-the-messy.py)** — a deliberately ugly synthetic CSV: mixed dtypes, encoded missing values, inconsistent date formats. Repair it. (~50 min)
3. **[Exercise 3 — Group-by, aggregate, pivot](exercise-03-groupby-aggregate.py)** — group by one and two keys, run named aggregations, pivot to wide. (~40 min)

## Workflow

- Type, do not paste.
- Each file has built-in `assert`-style checks at the bottom. Run with `python exercise-XX.py` and watch it print `OK`.
- Hint blocks live at the very bottom of each file, commented out. Read them only after fifteen minutes of trying a problem.
- Exercises 1 and 2 build their own deterministic synthetic data (so the tests never flake on a network failure). Exercise 1 also documents how to fetch a real NYC-taxi Parquet if you want the real-world flavor.

## Self-grading

After each exercise, ask: *could I explain this to a junior engineer in three minutes?* If yes, move on. If no, re-read the lecture material that covered it.

## Running with `pytest`

A minimal `pytest`-style smoke test is at the bottom of each file. From the `exercises/` directory:

```bash
pytest exercise-01-load-and-inspect.py
pytest exercise-02-clean-the-messy.py
pytest exercise-03-groupby-aggregate.py
```

All three should pass before you move to the [homework](../homework.md) and the [mini-project](../mini-project/README.md).
