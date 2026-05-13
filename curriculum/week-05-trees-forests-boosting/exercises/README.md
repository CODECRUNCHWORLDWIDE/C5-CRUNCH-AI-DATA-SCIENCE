# Week 5 — Exercises

Three drills. Each is a `.py` file with TODOs you fill in. Do them in order — they walk you from a tree built by hand through the modern boosted-trees libraries.

1. **[Exercise 1 — A Decision Tree from Scratch](exercise-01-tree-from-scratch.py)** — implement a regression tree in pure NumPy. The recursive split, the incremental running sums, the MSE-impurity criterion, two stopping rules. Verify it agrees with `sklearn.tree.DecisionTreeRegressor` to ~15% on the diabetes dataset. (~70 min)
2. **[Exercise 2 — Random Forest with sklearn](exercise-02-random-forest-sklearn.py)** — build a `RandomForestRegressor` with Breiman's regression defaults, read the OOB R², sweep `max_features`, compare to a single tree, compute split-gain and permutation importance. (~60 min)
3. **[Exercise 3 — XGBoost vs LightGBM](exercise-03-xgboost-vs-lightgbm.py)** — fit `HistGradientBoostingRegressor`, `XGBRegressor`, and `LGBMRegressor` on the California Housing dataset with matched hyperparameters and early stopping. Compare test RMSE and wall-clock. (~60 min)

## Workflow

- Type, do not paste.
- Each file builds its own deterministic synthetic data (or uses a scikit-learn bundled dataset) so the tests never depend on the network — with one exception: Exercise 3 uses `fetch_california_housing`, which downloads ~600 KB on first use and caches afterward.
- Each file has built-in `assert`-style checks at the bottom. Run with `python exercise-XX.py` and watch it print `OK`.
- Hint blocks live at the very bottom of each file, commented out. Read them only after fifteen minutes of trying a problem.
- Every model you fit should be compared to a `DummyRegressor` baseline. Lecture 1, Section 12.

## Self-grading

After each exercise, ask: *could I drop this code into the mini-project tomorrow?* If yes, move on. If no, the gap is the lesson.

## Running with `pytest`

A minimal `pytest`-style smoke test is at the bottom of each file. From the `exercises/` directory:

```bash
pytest exercise-01-tree-from-scratch.py
pytest exercise-02-random-forest-sklearn.py
pytest exercise-03-xgboost-vs-lightgbm.py
```

All three should pass before you move to the [homework](../homework.md), the [challenge](../challenges/), and the [mini-project](../mini-project/README.md).

Exercise 3 skips the XGBoost and LightGBM tests if those libraries are not installed (it falls back to `HistGradientBoostingRegressor` only). To run the full comparison:

```bash
pip install "xgboost>=2.0,<3" "lightgbm>=4.0,<5"
```

On macOS arm64, LightGBM additionally needs `brew install libomp`. See [resources.md](../resources.md) for the installation notes.

## Compilation check

Each file must compile cleanly with `python -m py_compile exercise-XX.py`. The CI runs that check.
