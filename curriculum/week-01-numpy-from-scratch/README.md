# Week 1 — NumPy from Scratch

> *Before pandas, before scikit-learn, before PyTorch — there is an array of numbers in contiguous memory. Everything else is built on top of it.*

Welcome to **C5 · Crunch AI / Data Science**. Week 1 is the foundation week. We will not train a model. We will not load a dataset that ships with seaborn. We will spend five days on one object — the NumPy `ndarray` — until you can describe its shape, dtype, strides, and memory layout without thinking. Every later week (pandas, scikit-learn, the from-scratch neural net in Week 7, PyTorch in Week 8) treats this object as the primitive. If you understand it now, the rest of C5 stops feeling like a sequence of disconnected libraries and starts feeling like one continuous numerical stack.

This is also the week where we draw the line that defines this whole track: **vectorize, or stop**. A `for` loop over array elements in Python is a bug, not a style choice. By Friday you will have internalized that.

We target **NumPy 2.x** (released 2024-06; the current major series in 2026). Where the 1.x / 2.x API differs, we flag it.

---

## Learning objectives

By the end of this week, you will be able to:

- **Explain** in three sentences why a Python list of floats is the wrong data structure for numerical work — and what an `ndarray` does differently.
- **Create** arrays with `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, `np.full`, and `np.random.default_rng(...).normal(...)`, and choose an appropriate `dtype` (`int32`, `int64`, `float32`, `float64`, `bool_`).
- **Inspect** any array's `shape`, `dtype`, `strides`, `ndim`, `size`, `itemsize`, `nbytes`, and `flags`.
- **Distinguish** a *view* from a *copy* and use `np.shares_memory(a, b)` to verify which you have.
- **Reshape, transpose, slice, and fancy-index** an array, and predict whether each operation returns a view or a copy before running it.
- **Broadcast** two arrays of different shapes by reciting the three broadcasting rules.
- **Vectorize** a numerical computation: replace a Python `for` loop with a ufunc or a reduction along an axis.
- **Manipulate an image as a 3D array** (`H × W × 3` for RGB) — convert to grayscale, threshold, slice channels, flip axes — using nothing but NumPy.
- **Pass** all `pytest` cases on the Week 1 exercises.

---

## Prerequisites

- **C1 Weeks 1–14** completed, or equivalent. You should be comfortable with functions, classes, list comprehensions, exceptions, and `pip` / `venv`.
- A working **Python 3.11+** install. We use 3.12 throughout.
- A `numpy` 2.x install: `pip install "numpy>=2.0,<3"`.
- `matplotlib` and `imageio` for image work (Mini-Project): `pip install matplotlib imageio`.
- Some memory of high-school linear algebra. Vectors and matrices, dot products. We do not yet need eigenvalues; that arrives in Week 6.

No GPU is required this week. We are not doing deep learning yet. Everything runs on a 2018 laptop in under a second.

---

## Topics covered

- Why Python `list` is slow for numerical work: per-element `PyObject` headers, pointer indirection, no SIMD.
- The `ndarray`: one contiguous block of typed memory plus `(shape, dtype, strides)`.
- `dtype` matters: `int32` vs `int64`, `float32` vs `float64` — memory cost, precision cost, overflow risk.
- Strides: how a multi-dimensional shape maps onto a 1-D memory buffer.
- C-contiguous (row-major) vs Fortran-contiguous (column-major). When it matters, when it does not.
- View vs copy. Why `a[::2]` shares memory and `a[[0, 2, 4]]` does not.
- Vectorization: write the math, not the loop. Why this is 10–100× faster.
- Broadcasting: the three rules, with worked examples.
- Universal functions (ufuncs): `np.sin`, `np.add`, the `out=` parameter, the `where=` parameter.
- Reductions along an axis: `sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax`.
- A first taste of `np.einsum` for arbitrary tensor contractions.
- The "no `for`-loop" discipline as the foundation of every later week.
- An image as a 3-D `uint8` array: a worked example all week.

---

## Weekly schedule

The schedule below adds up to about **36 hours**. Treat it as a target; some sections will click in fifteen minutes, others will take three hours. That is fine.

| Day       | Focus                                          | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Why lists are slow; the `ndarray`; dtype       |    3h    |    2h     |     0h     |    0.5h   |   1h     |     0h       |    0.5h    |     7h      |
| Tuesday   | Shape, strides, view vs copy, C/F order        |    0h    |    2h     |     1h     |    0.5h   |   1h     |     0h       |    0.5h    |     5h      |
| Wednesday | Vectorization, ufuncs, broadcasting            |    3h    |    2h     |     1h     |    0.5h   |   1h     |     0h       |    0h      |     7.5h    |
| Thursday  | Reductions, axes, an image as an array         |    0h    |    2h     |     1h     |    0.5h   |   1h     |     2h       |    0.5h    |     7h      |
| Friday    | `einsum` taster; mini-project deep work        |    0h    |    0h     |     1h     |    0.5h   |   1h     |     2h       |    0.5h    |     5h      |
| Saturday  | Mini-project polish                            |    0h    |    0h     |     0h     |    0h     |   1h     |     3h       |    0h      |     4h      |
| Sunday    | Quiz, review, push to portfolio repo           |    0h    |    0h     |     0h     |    0.5h   |   0h     |     0h       |    0h      |     0.5h    |
| **Total** |                                                | **6h**   | **8h**    | **4h**     | **3h**    | **6h**   | **7h**       | **2h**     | **36h**     |

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Free books, official docs, MIT linear algebra OCW |
| [lecture-notes/01-the-ndarray-and-why-it-exists.md](./lecture-notes/01-the-ndarray-and-why-it-exists.md) | The array object: memory, dtype, strides, views |
| [lecture-notes/02-vectorization-broadcasting-views.md](./lecture-notes/02-vectorization-broadcasting-views.md) | Vectorization, broadcasting, ufuncs, reductions, `einsum` |
| [exercises/README.md](./exercises/README.md) | Index of exercises |
| [exercises/exercise-01-shape-dtype-strides.py](./exercises/exercise-01-shape-dtype-strides.py) | Inspect and reshape arrays; predict view vs copy |
| [exercises/exercise-02-broadcasting-drills.py](./exercises/exercise-02-broadcasting-drills.py) | Six broadcasting drills, each replacing a Python loop |
| [exercises/exercise-03-image-as-array.py](./exercises/exercise-03-image-as-array.py) | Load an image; manipulate it as a 3-D array |
| [challenges/README.md](./challenges/README.md) | Index of weekly challenges |
| [challenges/challenge-01-image-filter-pure-numpy.md](./challenges/challenge-01-image-filter-pure-numpy.md) | Grayscale, box blur, Sobel edge — all pure NumPy |
| [quiz.md](./quiz.md) | 10 multiple-choice questions with an answer key |
| [homework.md](./homework.md) | Six practice problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Full spec for the vectorized image-processing notebook |

---

## Stretch goals

- Read NumPy 2.x's [release notes](https://numpy.org/doc/stable/release.html) and pick three breaking changes from 1.x → 2.0. Write each as a one-sentence migration note.
- Watch [Travis Oliphant's "The History of Array Programming"](https://www.youtube.com/results?search_query=travis+oliphant+history+numpy) talk (or any retrospective on the origins of NumPy).
- Skim chapters 1–3 of [*From Python to NumPy* by Nicolas Rougier](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) (free). The book's whole point is the discipline this week tries to install.
- Reproduce one figure from chapter 4 ("Code vectorization") of Rougier — pick any worked example and re-implement it from scratch.

---

## What you will *not* do this week

You will not:

- Train a model (Week 4).
- Open a `.csv` (Week 2).
- Touch pandas, scikit-learn, or PyTorch.
- Use any "AI" library.

That is deliberate. The data-science stack is a stack precisely because each layer hides the one beneath it. We are at the bottom layer. Acquire it cleanly.

---

## Up next

[Week 2 — pandas, Honestly](../week-02-pandas-honestly/) — once you have pushed your vectorized image notebook to your `crunch-ai-portfolio-<yourhandle>` repo.
