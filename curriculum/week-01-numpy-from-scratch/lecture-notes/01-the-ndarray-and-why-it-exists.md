# Lecture 1 — The `ndarray` and Why It Exists

> **Outcome:** You can explain why a Python list of floats is the wrong structure for numerical work; create and inspect arrays; predict their `shape`, `dtype`, and `strides`; and tell a view from a copy with `np.shares_memory`.

We begin with a question that sounds like a complaint: *why does Python need NumPy at all?* Python already has lists. Lists already hold numbers. So what does NumPy give us that justifies a separate library, a separate type, and (in 2024) a backwards-incompatible 2.0 release that some downstream projects took a year to absorb?

The answer is three words: **contiguous typed memory**. That is what the `ndarray` is. The rest of NumPy — every ufunc, every broadcasting rule, every reduction — falls out of that choice.

We target **NumPy 2.x** throughout (the 2.0 release shipped 2024-06; current minor in 2026 is 2.2+). Where the 1.x / 2.x API differs, we will say so.

---

## 1. Why a Python list of floats is slow

Open a REPL and time a sum.

```python
import time

xs = [float(i) for i in range(10_000_000)]

t0 = time.perf_counter()
s = 0.0
for x in xs:
    s += x
t1 = time.perf_counter()
print(f"Python loop:   {t1 - t0:.3f} s,  sum = {s}")
```

On a 2020-era laptop with CPython 3.12, this prints something like:

```
Python loop:   0.46 s,  sum = 4.999999e+13
```

Now the NumPy equivalent:

```python
import numpy as np

a = np.arange(10_000_000, dtype=np.float64)

t0 = time.perf_counter()
s = a.sum()
t1 = time.perf_counter()
print(f"NumPy sum:     {t1 - t0:.3f} s,  sum = {s}")
```

```
NumPy sum:     0.008 s,  sum = 4.999999e+13
```

That is a **50–60× speedup** on a single line. Where does it come from?

### What a Python `list` actually is

In CPython, a `list` is an array of *pointers* to `PyObject` headers. Each element of `[1.0, 2.0, 3.0]` is **not** a `double` sitting in a row of memory. It is a pointer to a separately-allocated `PyFloatObject`, somewhere on the heap, that contains:

- A reference count (one 64-bit word).
- A pointer to the type object `float` (one 64-bit word).
- The actual `double` value (one 64-bit word).

Per element: **24 bytes of metadata for 8 bytes of data**. Plus the pointer in the list (another 8 bytes). That is a 4× memory overhead for *every single number*. And because the floats are scattered across the heap, accessing them is cache-hostile: each `+=` triggers a pointer dereference, a type check, and likely an L1 cache miss.

The arithmetic itself is also slow. `s += x` is not a C `+=` on a `double`; it is a method call (`float.__iadd__`), which in turn calls `float.__add__`, which constructs a new `PyFloatObject` for the result, which then gets bound back to `s`. We are paying interpreter overhead per element.

### What an `ndarray` is

A NumPy `ndarray` is **one contiguous C buffer** of raw machine numbers, plus a small Python header describing how to interpret it. The buffer holds `float64`s as actual IEEE-754 64-bit values, side by side, with **zero per-element overhead**.

```
list   →  [ptr] [ptr] [ptr] [ptr] ...    ╲
              ↓     ↓     ↓     ↓        ╲
            box   box   box   box   <- 24 B each, somewhere on the heap
            8.0   1.5   3.1   2.7

ndarray → │ 8.0 │ 1.5 │ 3.1 │ 2.7 │ ... ←  one contiguous block of float64
          └───┘ └───┘ └───┘ └───┘
            8 B   8 B   8 B   8 B
```

`a.sum()` walks that contiguous buffer in a tight C loop, with SIMD on modern CPUs. There is no per-element type check, no allocation, no boxing. That is where the 50× comes from.

The catch: **every element must have the same type** (the `dtype`). You cannot mix a `float64` and a string in one `ndarray` and expect speed. NumPy will accept it, but it falls back to `dtype=object`, which is just a list of pointers in disguise — and slow again. This is the price of the speedup, and it is a price worth paying.

---

## 2. Creating and inspecting an `ndarray`

The fundamental constructor is `np.array`:

```python
>>> import numpy as np
>>> a = np.array([[1, 2, 3], [4, 5, 6]])
>>> a
array([[1, 2, 3],
       [4, 5, 6]])
>>> a.shape
(2, 3)
>>> a.dtype
dtype('int64')        # platform-dependent; on Windows you may see int32
>>> a.ndim
2
>>> a.size
6
>>> a.itemsize
8                     # bytes per element
>>> a.nbytes
48                    # total bytes
```

Every `ndarray` has these attributes. You will read them constantly.

| Attribute | Meaning |
|-----------|---------|
| `shape` | Tuple of axis lengths, e.g. `(2, 3)` for 2 rows × 3 columns |
| `dtype` | Element type: `int64`, `float32`, `bool_`, `uint8`, etc. |
| `ndim` | Number of axes (= `len(shape)`) |
| `size` | Total elements (= `prod(shape)`) |
| `itemsize` | Bytes per element |
| `nbytes` | Total bytes (= `size * itemsize`) |
| `strides` | Bytes to step along each axis (see §4) |
| `flags` | Flags including `C_CONTIGUOUS`, `F_CONTIGUOUS`, `WRITEABLE`, `OWNDATA` |

### Other common constructors

```python
np.zeros((3, 4))                  # all zeros, default float64
np.ones((3, 4), dtype=np.float32) # all ones, float32
np.full((2, 2), 7.5)              # all 7.5
np.arange(10)                     # 0, 1, 2, ..., 9
np.arange(0, 1, 0.1)              # 0.0, 0.1, ..., 0.9 (avoid for floats!)
np.linspace(0, 1, 11)             # 0.0, 0.1, ..., 1.0 — preferred for floats
np.eye(4)                         # 4×4 identity
np.empty((3, 3))                  # uninitialized; faster but unsafe to read

rng = np.random.default_rng(seed=42)   # NumPy 2.x preferred API
rng.normal(size=(3, 4))                # Gaussian samples
rng.integers(0, 10, size=(3, 4))       # uniform integers
```

> **NumPy 2.x note.** Always use `np.random.default_rng()` instead of the legacy `np.random.seed(); np.random.randn(...)` API. The default RNG is a 64-bit PCG64 with no global state, and the legacy module is now considered a compatibility shim.

---

## 3. `dtype` matters: precision, memory, overflow

A `dtype` is two facts: *what kind of number* (int, unsigned int, float, bool) and *how wide* (in bits). The full table is at <https://numpy.org/doc/stable/reference/arrays.scalars.html>, but the ones you will see all year are:

| `dtype`     | Bytes | Range / precision |
|-------------|------:|-------------------|
| `bool_`     | 1     | `True` / `False` |
| `int8`      | 1     | −128 to 127 |
| `uint8`     | 1     | 0 to 255 (image pixel channels live here) |
| `int32`     | 4     | ±2.1 × 10⁹ |
| `int64`     | 8     | ±9.2 × 10¹⁸ (default on 64-bit Linux/macOS) |
| `float32`   | 4     | ~7 decimal digits (GPU tensors and most deep-learning weights) |
| `float64`   | 8     | ~15–16 decimal digits (default; "double precision") |
| `complex128`| 16    | Two `float64`s |

Three lessons:

**Memory.** A `(1000, 1000)` `float64` matrix is 8 MB. The same `float32` is 4 MB. The same `uint8` is 1 MB. For an image dataset of 50,000 images at `256 × 256 × 3`, that is 9.4 GB in `float32` and 9.4 GB in `uint8` becomes 9.4 / 4 ≈ 2.4 GB. Storage on disk and RAM in training scale with `itemsize`. Always pick the narrowest dtype that does not lose information.

**Precision.** `float32` has about 7 decimal digits of precision. For most NN training that is fine. For scientific computing (numerical integration, stiff ODEs, finance) it is often *not* fine; you want `float64`. The default in NumPy is `float64`; in PyTorch it is `float32`. Know which world you are in.

**Overflow.** Integer overflow is silent in NumPy. This trips up everyone exactly once:

```python
>>> a = np.array([200, 200, 200], dtype=np.uint8)
>>> a + 100
array([ 44, 44, 44], dtype=uint8)    # wrapped around mod 256
```

`uint8` can hold 0 to 255. `200 + 100 = 300`, which wraps to `300 mod 256 = 44`. NumPy does not warn. If you are doing arithmetic on a `uint8` image, **cast to `int16` or `float32` first**:

```python
>>> a.astype(np.int16) + 100
array([300, 300, 300], dtype=int16)
```

> **NumPy 2.x note.** Promotion rules changed in 2.0 via [NEP-50](https://numpy.org/neps/nep-0050-scalar-promotion.html). Previously, `np.uint8(200) + 100` would have *promoted* to `int64` and given you 300 with no overflow. In 2.x, scalar-array operations preserve the array's dtype. The new rule is more predictable; the old one occasionally hid bugs. Read NEP-50 once and move on.

---

## 4. Shape and strides: how a 1-D buffer becomes N-dimensional

This is the section that turns NumPy from a black box into a mental model.

An `ndarray` stores its data as **one flat 1-D byte buffer**. The `shape` and `strides` attributes tell NumPy how to *index* into that buffer.

`strides` is a tuple, one entry per axis, giving the number of bytes you must step in the buffer to advance by one along that axis.

```python
>>> a = np.arange(12, dtype=np.int64).reshape(3, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> a.shape
(3, 4)
>>> a.strides
(32, 8)
```

Reading that: to step **one row** (axis 0), move 32 bytes in the buffer (= 4 columns × 8 bytes per `int64`). To step **one column** (axis 1), move 8 bytes (= one element).

This is **row-major** or **C-contiguous** layout: the last axis varies fastest in memory.

The same buffer with the same shape could be laid out **column-major** (Fortran-contiguous), where the *first* axis varies fastest:

```python
>>> b = np.asfortranarray(a)
>>> b.strides
(8, 32)
>>> b.flags['F_CONTIGUOUS']
True
```

Same `shape`, same values, different memory order. Why care?

Two reasons:

1. **Cache locality.** Iterating over `a[i, :]` in row-major is a contiguous read; iterating over `a[:, j]` in row-major skips by 32 bytes each step — slower because of cache-line and prefetcher behavior. For most NumPy code the speed difference is small (NumPy's C loops handle both), but the difference matters when interoperating with LAPACK / BLAS, which originated in Fortran and assume column-major.
2. **Interop.** Some libraries (older Fortran code, MATLAB) assume column-major. When passing arrays across the boundary you may need `np.asfortranarray(a)` or `np.ascontiguousarray(a)`.

For C5, the rule is: **let NumPy default to C-contiguous, do not fight it**, and only switch when a function's docs tell you to.

### Strides let you build views without copying

The deep reason strides exist: many array operations can be expressed as *new shape + new strides on the same buffer*. No data is copied.

```python
>>> a = np.arange(12).reshape(3, 4)
>>> a.T               # transpose
array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])
>>> a.T.strides       # original (32, 8) became (8, 32)
(8, 32)
>>> np.shares_memory(a, a.T)
True
```

`a.T` is a *view*: same data buffer, swapped strides. Zero allocation. The same is true for most slices, reshapes, and broadcasts.

---

## 5. View vs copy: the single most important distinction in Week 1

A *view* shares its buffer with another array. Mutating one mutates the other. A *copy* has its own buffer.

```python
>>> a = np.arange(10)
>>> b = a[::2]                # every other element — a VIEW
>>> b[0] = 999
>>> a
array([999,   1,   2,   3,   4,   5,   6,   7,   8,   9])
>>> np.shares_memory(a, b)
True

>>> c = a[[0, 2, 4]]          # fancy indexing — a COPY
>>> c[0] = -1
>>> a
array([999,   1,   2,   3,   4,   5,   6,   7,   8,   9])   # unchanged
>>> np.shares_memory(a, c)
False
```

The general rule:

| Operation | Returns |
|-----------|---------|
| Basic slicing (`a[1:3]`, `a[::2]`, `a[:, 0]`) | **view** |
| `reshape`, `transpose`, `np.expand_dims`, `np.squeeze` | **view** (when possible) |
| Boolean indexing (`a[a > 0]`) | **copy** |
| Fancy/integer-array indexing (`a[[0, 2, 4]]`, `a[idx]` where `idx` is an array) | **copy** |
| `a.copy()` | **copy** |
| `astype` | **copy** (almost always) |
| Arithmetic (`a + b`, `a * 2`) | **copy** (new buffer) |

When you are not sure, **ask NumPy**:

```python
>>> np.shares_memory(a, b)
True
```

This question — "did I just make a view or a copy?" — recurs in every Week 1 bug. Always be able to answer it.

### Why views matter

Views are the mechanism that makes vectorized code cheap. When you write

```python
sub = image[100:200, 100:200, :]   # a 100×100 view into a larger image
sub *= 2                            # mutates the original image in place
```

no data was copied. The slice is a (shape, strides, offset) triple into the original buffer. This is good for performance and a source of bugs when you did not realize you wrote into the original.

If you want independence, ask for it: `sub = image[100:200, 100:200, :].copy()`.

---

## 6. Reshaping: the same data, a different view

```python
>>> a = np.arange(12)
>>> a.reshape(3, 4)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> a.reshape(2, 6)
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
>>> a.reshape(2, -1)              # -1 means "figure it out"
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
```

`reshape` returns a view when it can. If the requested shape cannot be expressed with new strides (e.g., reshaping a transposed array into something that needs the data re-laid-out), NumPy falls back to a copy. The function `ndarray.reshape(...)` *will* copy silently if needed; `ndarray.ravel(order='C')` is the same. If you want to **enforce** no copy, use `a.shape = (...)` (the attribute setter), which raises `AttributeError` if a copy would be required. This is a debugging tool, not an everyday API.

---

## 7. Indexing: basic vs advanced

NumPy has two distinct indexing modes, and the difference is exam material.

**Basic indexing** — slices and integers along each axis. Returns a *view*.

```python
a[0]                # first row;          view
a[0, :]             # first row;          view
a[:, 1]             # second column;      view
a[1:3, 0:2]         # 2×2 sub-block;      view
a[..., -1]          # last along last axis; view
```

**Advanced (fancy) indexing** — boolean masks or integer arrays. Returns a *copy*.

```python
a[a > 0]                                  # boolean mask;       copy
a[[0, 2, 4]]                              # integer array;      copy
a[[0, 1, 2], [0, 1, 0]]                   # element-wise pairs; copy
```

Mixing the two is allowed and surprisingly subtle; read the [indexing docs](https://numpy.org/doc/stable/user/basics.indexing.html) for the rules. The 90% case: stick to basic indexing for views, use advanced indexing when you need to gather scattered elements (and remember it copies).

---

## 8. The mental model to walk away with

After this lecture you should be able to draw this on a napkin:

```
   shape:  (3, 4)        ← Python-visible metadata
   dtype:  float64
   strides: (32, 8)
            │
            ▼
   data:  [0.0 │ 1.0 │ 2.0 │ 3.0 │ 4.0 │ 5.0 │ 6.0 │ 7.0 │ 8.0 │ 9.0 │ 10.0 │ 11.0]
          ───── one contiguous 96-byte C buffer ─────
```

An `ndarray` is a thin Python wrapper around a `(buffer, shape, strides, dtype)` quadruple. Everything else — reshape, transpose, slice — is metadata manipulation on the same buffer. Vectorized operations (Lecture 2) iterate the buffer in C with no per-element Python overhead. That is the whole game.

---

## 9. Self-check

Without re-reading:

1. Why is a Python `list` of one million `float`s slower to sum than a NumPy `float64` array of the same size?
2. What does `a.strides` of `(32, 8)` tell you about `a`?
3. Given `a = np.arange(12).reshape(3, 4)`, will `a.T` allocate new memory? How do you confirm?
4. `b = a[a > 5]` — view or copy?
5. `b = a[1:3, :]` — view or copy?
6. You have a `uint8` image and want to subtract 10 from every pixel without underflowing. What do you do?
7. What does `np.linspace(0, 1, 11)` produce, and why is it usually preferable to `np.arange(0, 1, 0.1)` for floats?

---

## Further reading

- **NumPy — Internal memory layout**: <https://numpy.org/doc/stable/dev/internals.html>
- **NumPy — The N-dimensional array**: <https://numpy.org/doc/stable/reference/arrays.ndarray.html>
- **NEP-50 — Promotion rules in NumPy 2.x**: <https://numpy.org/neps/nep-0050-scalar-promotion.html>
- **Rougier, *From Python to NumPy*, chapter 2 ("Anatomy of an array")**:
  <https://www.labri.fr/perso/nrougier/from-python-to-numpy/#anatomy-of-an-array>

Next: [Lecture 2 — Vectorization, Broadcasting, Views](./02-vectorization-broadcasting-views.md).
