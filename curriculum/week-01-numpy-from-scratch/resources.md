# Week 1 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **NumPy — Absolute beginners' tutorial** (official):
  <https://numpy.org/doc/stable/user/absolute_beginners.html>
- **NumPy — Quickstart** (the canonical "first hour" guide):
  <https://numpy.org/doc/stable/user/quickstart.html>
- **NumPy — Broadcasting** (the single most important page in the docs):
  <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- **NumPy — Indexing** (basic vs advanced):
  <https://numpy.org/doc/stable/user/basics.indexing.html>
- **NumPy — Internal memory layout** (strides, contiguity, views):
  <https://numpy.org/doc/stable/dev/internals.html>

## The official docs you will bounce between all week

- **NumPy 2.x user guide**: <https://numpy.org/doc/stable/user/index.html>
- **NumPy API reference**: <https://numpy.org/doc/stable/reference/index.html>
- **NumPy 2.0 release notes** — what changed from 1.x: <https://numpy.org/doc/stable/release/2.0.0-notes.html>
- **NEP-50 — promotion rules in NumPy 2.x** — the new type-promotion behavior: <https://numpy.org/neps/nep-0050-scalar-promotion.html>
- **`np.einsum` — the most powerful function you have never used**: <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>

## Free books (full texts, legally free)

- **Nicolas Rougier — *From Python to NumPy*** (CC-BY-NC-SA, full text online):
  <https://www.labri.fr/perso/nrougier/from-python-to-numpy/>
  This is the single best book on the vectorization mindset. Chapters 1–4 cover this week's material.
- **Jake VanderPlas — *Python Data Science Handbook*** (free online, MIT-licensed):
  <https://jakevdp.github.io/PythonDataScienceHandbook/>
  Chapter 2 is "Introduction to NumPy." Skim it; it is short and excellent.
- **Robert Johansson — *Numerical Python*** companion site (free notebooks):
  <https://github.com/jrjohansson/numerical-python-book>
- **Eli Bressert — *SciPy and NumPy* (O'Reilly, free PDF)**:
  <https://www.oreilly.com/library/view/scipy-and-numpy/9781449361600/> — title-page preview is free; chapter 2 is the relevant section.

## The math you need this week (free open courseware)

You do not need to *take* a linear-algebra course this week, but you should know where to revisit it.

- **MIT 18.06 — Linear Algebra (Gilbert Strang)** — the canonical free linear-algebra course:
  <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/>
  Lectures 1–3 (vectors, dot products, matrix multiplication) are exactly the operations you will vectorize.
- **3Blue1Brown — *Essence of Linear Algebra*** (15 short videos, free, YouTube):
  <https://www.3blue1brown.com/topics/linear-algebra>
  Visual intuition for vectors, matrices, and transformations.
- **Khan Academy — Linear algebra** (free, exercises included):
  <https://www.khanacademy.org/math/linear-algebra>

## Tools you will use this week

- **`numpy`** ≥ 2.0: `pip install "numpy>=2.0,<3"`.
- **`matplotlib`** for visualizing arrays and images: `pip install matplotlib`.
- **`imageio`** for reading PNG/JPEG into a NumPy array: `pip install imageio`.
- **`pytest`** to run the test suite on the exercises: `pip install pytest`.
- **Jupyter or VS Code notebooks** for the mini-project. The mini-project ships as a notebook so reviewers can see your arrays inline.

A `requirements.txt` snippet for the week:

```text
numpy>=2.0,<3
matplotlib>=3.8
imageio>=2.34
pytest>=8.0
```

## Videos (free, no signup)

- **NumPy's official "What is NumPy?" intro** (30 min, on the docs site).
- **Sentdex — "NumPy Tutorial"** (YouTube, ~1 h): a no-nonsense walkthrough; you can play it at 1.5×.
- **SciPy 2019 — Aaron Meurer, "Beyond `np.array`: an introduction to NumPy"** (search YouTube; ~25 min).
- **PyCon 2021 — Sebastian Berg, "How does NumPy actually work?"** (search YouTube; ~30 min). Optional but excellent for the strides material.

## Open-source projects to read (in this order)

You can learn more from one hour reading other people's NumPy than from three hours of tutorials.

1. **NumPy itself** — `numpy/numpy` on GitHub. Start at `numpy/_core/`:
   <https://github.com/numpy/numpy>
2. **scikit-image** — every function in `skimage` is a NumPy exercise. Pick `skimage.color.rgb2gray`:
   <https://github.com/scikit-image/scikit-image/blob/main/skimage/color/colorconv.py>
3. **`numpy-100`** — 100 short exercises with answers. After Week 1 you should be able to solve 60+:
   <https://github.com/rougier/numpy-100>

## Glossary cheat sheet

Keep this open in a tab.

| Term | Plain English |
|------|---------------|
| **`ndarray`** | NumPy's N-dimensional array object; a contiguous block of typed memory plus metadata |
| **`dtype`** | The element type and width: `int32`, `float64`, `bool_`, `uint8`, etc. |
| **shape** | A tuple giving the size of the array along each axis: `(3, 4)` is 3 rows × 4 columns |
| **`ndim`** | The number of axes (the length of `shape`) |
| **`size`** | The total number of elements (`prod(shape)`) |
| **strides** | A tuple of byte offsets to step along each axis in the underlying buffer |
| **view** | A new `ndarray` that shares memory with another; mutating one mutates both |
| **copy** | A new `ndarray` with its own memory |
| **C-contiguous** | The last axis is contiguous in memory (row-major; the default) |
| **F-contiguous** | The first axis is contiguous in memory (column-major; what Fortran and MATLAB use) |
| **ufunc** | A "universal function" — element-wise operation written in C (e.g. `np.sin`) |
| **broadcasting** | The set of rules NumPy uses to align arrays of different shapes for element-wise operations |
| **axis** | A direction along which a reduction or operation runs; axis 0 is "rows," axis 1 is "columns" for a 2-D array |
| **`einsum`** | A mini-language for arbitrary tensor contractions in one string |

---

*If a link 404s, please open an issue so we can replace it.*
