# Week 1 — Exercises

Three drills. Each is a `.py` file with TODOs you fill in. Do them in order — they get progressively further from "the docs example" and closer to real code.

1. **[Exercise 1 — Shape, dtype, strides](exercise-01-shape-dtype-strides.py)** — create, inspect, reshape, index; predict view vs copy and confirm with `np.shares_memory`. (~35 min)
2. **[Exercise 2 — Broadcasting drills](exercise-02-broadcasting-drills.py)** — six small problems, each replacing a Python loop with one or two vectorized lines. (~40 min)
3. **[Exercise 3 — Image as array](exercise-03-image-as-array.py)** — load a real image as a 3-D `uint8` array, slice channels, threshold, flip, build a naive grayscale by hand. (~35 min)

## Workflow

- Type, do not paste.
- Each file has a small built-in checker (`assert` statements) at the bottom. Run the file with `python exercise-XX.py` and watch it print `OK`.
- For Exercise 1 and 2, hint blocks live at the very bottom of each file, commented out. Read them only after fifteen minutes of trying.
- Exercise 3 requires `pip install imageio matplotlib` and a sample image. The file downloads a small Creative Commons image automatically; if you are offline, point it at any local PNG/JPG.

## Self-grading

After each exercise, ask yourself: *could I explain this to a junior engineer in three minutes?* If yes, move on. If no, re-read the lecture material that covered it.

## Running with `pytest`

A minimal `pytest`-style smoke test is included at the bottom of each file. From the `exercises/` directory:

```bash
pytest exercise-01-shape-dtype-strides.py
pytest exercise-02-broadcasting-drills.py
pytest exercise-03-image-as-array.py
```

All three should pass before you move to the [homework](../homework.md) and the [mini-project](../mini-project/README.md).
