# Week 4 — Exercises

Three drills. Each is a `.py` file with TODOs you fill in. Do them in order — they build the ML workflow piece by piece.

1. **[Exercise 1 — Train / Validation / Test Split](exercise-01-train-test-split.py)** — split a dataset three ways, cross-validate, deliberately introduce a leak and watch the score lie, fix it. (~45 min)
2. **[Exercise 2 — Linear Regression by Hand](exercise-02-linear-regression-by-hand.py)** — fit `LinearRegression` three ways (normal equation, gradient descent, scikit-learn) and verify all three agree to 1e-6. (~50 min)
3. **[Exercise 3 — scikit-learn Pipeline](exercise-03-sklearn-pipeline.py)** — build a `Pipeline` + `ColumnTransformer` over mixed numeric and categorical features without leakage; tune the regularization with `RidgeCV`. (~50 min)

## Workflow

- Type, do not paste.
- Each file builds its own deterministic synthetic data (or uses a scikit-learn bundled dataset) so the tests never depend on the network.
- Each file has built-in `assert`-style checks at the bottom. Run with `python exercise-XX.py` and watch it print `OK`.
- Hint blocks live at the very bottom of each file, commented out. Read them only after fifteen minutes of trying a problem.
- Every model you fit should be compared to a `DummyRegressor` / `DummyClassifier` baseline. Lecture 1, Section 6.

## Self-grading

After each exercise, ask: *could I drop this code into the mini-project tomorrow?* If yes, move on. If no, the gap is the lesson.

## Running with `pytest`

A minimal `pytest`-style smoke test is at the bottom of each file. From the `exercises/` directory:

```bash
pytest exercise-01-train-test-split.py
pytest exercise-02-linear-regression-by-hand.py
pytest exercise-03-sklearn-pipeline.py
```

All three should pass before you move to the [homework](../homework.md), the [challenge](../challenges/), and the [mini-project](../mini-project/README.md).

## Compilation check

Each file must compile cleanly with `python -m py_compile exercise-XX.py`. The CI runs that check.
