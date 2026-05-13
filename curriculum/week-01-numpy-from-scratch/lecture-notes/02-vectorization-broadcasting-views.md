# Lecture 2 — Vectorization, Broadcasting, Views

> **Outcome:** You can replace a Python `for` loop over an array with a vectorized expression; broadcast two arrays of different shapes by reciting the three rules; choose the right reduction along the right axis; and read a small `np.einsum` string without panicking.

Lecture 1 was the static picture: what an `ndarray` *is*. This lecture is the dynamic picture: what you *do* with one. We are after a single skill: **write the math, not the loop**.

By the end of this week, an inner Python `for` loop in your numerical code is a code smell. There will be exceptions (and we will name them), but the default is: *if you wrote a loop over array elements in Python, you wrote a bug*.

---

## 1. Vectorization, the discipline

A vectorized operation is one expressed on whole arrays rather than on elements. The contrast:

```python
# Pythonic, slow. Per-element interpreter overhead.
def add_lists(xs, ys):
    out = [0.0] * len(xs)
    for i in range(len(xs)):
        out[i] = xs[i] + ys[i]
    return out

# Vectorized, fast. Two C pointers walking two contiguous buffers.
c = a + b
```

Where the speed comes from is the same as Lecture 1: the second version walks two `float64` buffers in a C loop with SIMD, no boxing, no per-element type check, no `PyFloat_FromDouble`. Empirically on a 10-million-element array of `float64`s, the Python version takes ~1.5 seconds; `a + b` takes ~30 milliseconds. **A 50× speedup, for less typing.**

Vectorization is not (yet) about correctness; it is about the *style*. The cognitive shift: instead of describing how to update each element, describe **the operation on the whole array**.

### What "vectorize" means in NumPy vs in maths

The word is overloaded. In NumPy, "vectorized" means "implemented in C with no Python-level loop." In linear algebra, "vectorize" means "stack the columns of a matrix into one tall vector" (the `vec(·)` operator). These are different concepts; context disambiguates. In Week 1 we always mean the NumPy sense.

### When loops are still acceptable

A loop in Python is acceptable when:

- The loop body is itself a non-trivial NumPy operation (a few thousand or more elements per iteration). The Python overhead is negligible.
- You need state across iterations that NumPy cannot express (some PDE solvers, some iterative refinement loops, MCMC).
- You are calling into an external library whose interface is per-row.

The danger pattern is "for each element of an array, do scalar math, write back." That is what we are training out of you.

---

## 2. Universal functions (ufuncs)

A **ufunc** is a function that operates element-wise on arrays, written in C. They are the atoms of vectorized NumPy.

```python
>>> a = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
>>> np.sin(a)
array([0.        , 0.47942554, 0.84147098, 0.99749499, 0.90929743])
>>> np.exp(a)
array([1.        , 1.64872127, 2.71828183, 4.48168907, 7.3890561 ])
>>> a ** 2
array([0.  , 0.25, 1.  , 2.25, 4.  ])
```

The full list is in the [ufunc docs](https://numpy.org/doc/stable/reference/ufuncs.html). The ones you will use this week:

| Family | Examples |
|--------|----------|
| Arithmetic | `+`, `-`, `*`, `/`, `%`, `**`, `np.add`, `np.multiply`, `np.power` |
| Comparisons | `>`, `>=`, `==`, `!=`, `np.greater`, `np.equal` |
| Trig / exp / log | `np.sin`, `np.cos`, `np.exp`, `np.log`, `np.log1p` |
| Rounding | `np.round`, `np.floor`, `np.ceil`, `np.trunc` |
| Sign / abs | `np.abs`, `np.sign`, `np.copysign` |
| Logic | `np.logical_and`, `np.logical_not`, `&`, `|`, `~` (on bool arrays) |

Every operator (`+`, `*`, `>`, etc.) on `ndarray` dispatches to a ufunc under the hood. They all share three useful keyword arguments:

```python
np.add(a, b, out=c)            # write result into c, no new allocation
np.add(a, b, where=mask)       # only compute where mask is True
np.multiply(a, b, dtype=np.float32)  # force the output dtype
```

`out=` is the trick for tight memory budgets: you can reuse a pre-allocated buffer for thousands of operations and not allocate at all.

### `np.where` and `np.clip` — the two most-asked vectorization patterns

The two things you will reach for repeatedly:

```python
>>> a = np.array([-2, -1, 0, 1, 2])
>>> np.where(a > 0, a, 0)               # ReLU in one line
array([0, 0, 0, 1, 2])
>>> np.clip(a, -1, 1)                   # saturate
array([-1, -1,  0,  1,  1])
```

`np.where(cond, x, y)` is the vectorized ternary: "where `cond`, take `x`, else `y`." `np.clip(a, lo, hi)` is `max(lo, min(hi, a))`.

---

## 3. Broadcasting

Broadcasting is how NumPy applies a ufunc to two arrays of *different shapes* without making you write the loop yourself.

The simplest case is a scalar:

```python
>>> a = np.arange(6).reshape(2, 3)
>>> a + 10
array([[10, 11, 12],
       [13, 14, 15]])
```

The scalar `10` was "broadcast" to a `(2, 3)` array of tens. No memory was allocated for that virtual array — NumPy just iterates as if it were there.

The interesting case is when both operands are arrays:

```python
>>> a = np.arange(6).reshape(2, 3)        # shape (2, 3)
>>> b = np.array([10, 20, 30])            # shape (3,)
>>> a + b
array([[10, 21, 32],
       [13, 24, 35]])
```

`b` was broadcast across the rows of `a`. The shape `(3,)` lined up with the last axis of `(2, 3)`. The result has shape `(2, 3)`.

### The three broadcasting rules

When operating on two arrays, NumPy compares their shapes **right-to-left**. Two dimensions are *compatible* when:

1. They are **equal**, or
2. One of them is **1**, or
3. One of them is **missing** (treated as `1` on the left).

If both arrays are compatible along every axis, broadcasting succeeds. The output shape is, axis by axis, the larger of the two dimensions.

Examples (✓ = compatible, ✗ = error):

```
A      (2, 3, 4)
B         (3, 4)
align: (2, 3, 4) vs (1, 3, 4)   ✓ → result (2, 3, 4)

A      (8, 1, 6, 1)
B         (7, 1, 5)
align: (8, 1, 6, 1) vs (1, 7, 1, 5)   ✓ → result (8, 7, 6, 5)

A      (3, 4)
B      (2, 4)
align: (3, 4) vs (2, 4)   ✗ axis 0: 3 vs 2
```

### A worked example: centering rows and columns

You want to subtract the mean of each row from every element in that row:

```python
>>> a = np.array([[1, 2, 3],
...               [4, 5, 6],
...               [7, 8, 9]])           # shape (3, 3)
>>> row_means = a.mean(axis=1)          # shape (3,)
>>> row_means
array([2., 5., 8.])
>>> a - row_means.reshape(3, 1)         # shape (3, 1) broadcasts across columns
array([[-1.,  0.,  1.],
       [-1.,  0.,  1.],
       [-1.,  0.,  1.]])
```

We had to `reshape` the means from `(3,)` to `(3, 1)` to align with the **rows** of `a`. Without the reshape, NumPy would have aligned the trailing `3` of `row_means` with the trailing `3` of `a` — meaning subtract the column means, which is *not* what we wanted. This is the most common broadcasting bug.

A cleaner spelling using `None` as a "new axis" marker:

```python
>>> a - row_means[:, None]              # equivalent to .reshape(3, 1)
```

`None` (or its alias `np.newaxis`) inserts an axis of length 1 at that position.

### Broadcasting does not allocate the broadcast array

This is the practical reason broadcasting exists. When `b` of shape `(3,)` is broadcast across `a` of shape `(1000, 3)`, NumPy does **not** materialize a `(1000, 3)` copy of `b`. It iterates as if `b` were repeated, with a stride of `0` along the broadcast axis. You get the convenience of "as if I had a giant array" without the memory cost.

You can see this with `np.broadcast_to`:

```python
>>> b = np.array([10, 20, 30])
>>> view = np.broadcast_to(b, (1000, 3))
>>> view.shape
(1000, 3)
>>> view.strides
(0, 8)                # stride 0 along the broadcast axis — same row repeated
>>> view.flags['WRITEABLE']
False                 # broadcast views are read-only
```

---

## 4. Reductions along an axis

A reduction collapses one or more axes by aggregating along them.

```python
>>> a = np.arange(12).reshape(3, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> a.sum()              # over all elements
66
>>> a.sum(axis=0)        # sum down each column — collapses axis 0
array([12, 15, 18, 21])
>>> a.sum(axis=1)        # sum across each row — collapses axis 1
array([ 6, 22, 38])
>>> a.sum(axis=1, keepdims=True)
array([[ 6],
       [22],
       [38]])
```

The mental rule: **`axis=k` means "the axis that disappears."** `a.sum(axis=0)` removes axis 0; the result has shape `(4,)`. `a.sum(axis=1)` removes axis 1; the result has shape `(3,)`. `keepdims=True` keeps the reduced axis as size-1, which is useful for further broadcasting.

The reductions you will use this week:

| Function | What it does |
|----------|--------------|
| `sum`, `prod` | Sum, product |
| `mean`, `std`, `var` | Mean, std, variance (Bessel correction via `ddof=1`) |
| `min`, `max` | Element-wise min, max |
| `argmin`, `argmax` | Index of the min, max |
| `any`, `all` | Boolean reductions over a bool array |
| `cumsum`, `cumprod` | Cumulative versions (return same shape) |

A composite example. You have an array `X` of shape `(N, D)` representing `N` data points in `D` dimensions, and you want each column's z-score:

```python
mu    = X.mean(axis=0)              # shape (D,)
sigma = X.std(axis=0)               # shape (D,)
Z     = (X - mu) / sigma            # broadcasts (D,) across N rows; shape (N, D)
```

Three lines. No Python loops. The same scikit-learn `StandardScaler` does internally exactly this.

---

## 5. The "no `for`-loop" discipline — five worked transformations

We will refactor five loops into one-line vectorized expressions. Internalize the pattern.

**(a)** Square each element.

```python
# Loop
out = np.empty_like(a)
for i in range(len(a)):
    out[i] = a[i] ** 2

# Vectorized
out = a ** 2
```

**(b)** Sum of squared differences (a foundational ML quantity).

```python
# Loop
total = 0.0
for i in range(len(y)):
    total += (y[i] - yhat[i]) ** 2

# Vectorized
total = ((y - yhat) ** 2).sum()
```

**(c)** Pairwise Euclidean distance from a point `p` to each row of a matrix `X`.

```python
# Loop
d = np.empty(len(X))
for i in range(len(X)):
    d[i] = np.sqrt(((X[i] - p) ** 2).sum())

# Vectorized — broadcasting subtracts p (shape (D,)) from every row of X
d = np.sqrt(((X - p) ** 2).sum(axis=1))
```

**(d)** Replace negatives with zero (ReLU).

```python
# Loop
for i in range(len(a)):
    if a[i] < 0:
        a[i] = 0

# Vectorized
np.maximum(a, 0, out=a)
```

**(e)** Build a 100×100 grid of `(x, y)` coordinates and compute `sin(x) * cos(y)` at every point.

```python
# Loop
out = np.empty((100, 100))
for i in range(100):
    for j in range(100):
        out[i, j] = np.sin(i / 10) * np.cos(j / 10)

# Vectorized — np.meshgrid + broadcasting
x = np.arange(100) / 10
y = np.arange(100) / 10
out = np.sin(x[:, None]) * np.cos(y[None, :])   # outer product, in one line
```

Notice how the vectorized versions read like *the math*, not like *a procedure*. That is the win.

---

## 6. `np.einsum` — the most powerful function you have never used

Einstein summation notation, in one function. The string you pass to `einsum` says: "these are the input axes; these are the output axes; sum over the rest." It captures matrix multiplication, dot products, outer products, traces, transposes, batched matrix multiplication, tensor contractions — all in one mini-language.

Three examples to get a feel for it.

**Dot product** of two 1-D vectors:

```python
>>> a = np.array([1, 2, 3])
>>> b = np.array([4, 5, 6])
>>> np.einsum('i,i->', a, b)        # sum over i
32
>>> # equivalent to a @ b
```

The string `'i,i->'` says: input 1 has one axis (call it `i`); input 2 has one axis (also `i`, so they must match in length); output has no axes — sum over `i`.

**Matrix multiplication**:

```python
>>> A = np.arange(6).reshape(2, 3)
>>> B = np.arange(6).reshape(3, 2)
>>> np.einsum('ij,jk->ik', A, B)
array([[10, 13],
       [28, 40]])
>>> # equivalent to A @ B
```

Here `'ij,jk->ik'` says: `A` is indexed by `(i, j)`, `B` by `(j, k)`, output is indexed by `(i, k)`, and `j` is summed over (because it appears on the left but not on the right). That *is* matrix multiplication.

**Batched matrix multiplication** — many small matrix-vector products at once:

```python
>>> X = np.random.default_rng(0).normal(size=(100, 32, 32))  # 100 matrices
>>> v = np.random.default_rng(0).normal(size=(100, 32))      # 100 vectors
>>> result = np.einsum('bij,bj->bi', X, v)                   # shape (100, 32)
```

You will not master `einsum` this week. Just *recognize* it when you see it; it shows up in every PyTorch codebase by Week 8. The single best reference is the [official docs](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html); the second-best is Tim Rocktäschel's blog post "Einsum is all you need" (free, easily searchable).

---

## 7. An image is an `ndarray`

This is where Week 1 becomes concrete. An RGB image is a 3-D `uint8` array:

```python
>>> import imageio.v3 as iio
>>> img = iio.imread('cat.png')
>>> img.shape
(512, 768, 3)             # (height, width, channels)
>>> img.dtype
dtype('uint8')
>>> img[0, 0]
array([180, 92, 45], dtype=uint8)   # the top-left pixel: R=180, G=92, B=45
```

Every operation we have introduced now has an image meaning:

| Operation | What it does to an image |
|-----------|--------------------------|
| `img[:, :, 0]` | Just the red channel — a 2-D `uint8` array |
| `img[::-1, :, :]` | Flip vertically (a view, no copy) |
| `img[:, ::-1, :]` | Flip horizontally (a view) |
| `img.mean(axis=2)` | Per-pixel average across channels — a naive grayscale |
| `img.transpose(2, 0, 1)` | Swap to `(channels, H, W)` — PyTorch order |
| `img[100:300, 200:400]` | A cropped view |
| `img > 200` | A boolean mask: pixels brighter than 200 |
| `np.where(mask, 255, 0)` | Threshold to binary |

A correct grayscale conversion uses the [ITU-R BT.601 luma coefficients](https://en.wikipedia.org/wiki/Luma_(video)) — different weights for R, G, B because the eye is more sensitive to green:

```python
weights = np.array([0.299, 0.587, 0.114])
gray = (img * weights).sum(axis=2)         # broadcasting + reduction
gray = gray.clip(0, 255).astype(np.uint8)  # back into uint8 range
```

That is your first real bit of image processing. No PIL, no OpenCV.

A box blur is a 3×3 averaging filter — and you will write it in the mini-project, in pure NumPy, using `np.pad` and slicing.

---

## 8. The mental model to walk away with

Three habits, in priority order:

1. **No `for` over elements.** When you reach for a loop, ask: can a ufunc, a broadcast, or a reduction do this? The answer is "yes" 95% of the time.
2. **Match shapes before you operate.** Read the shapes off the arrays involved; broadcast deliberately; use `None` / `np.newaxis` to align. If `a.shape == (N, D)` and `b.shape == (D,)`, then `a - b` works; if `b.shape == (N,)` you need `b[:, None]`.
3. **Pick the right axis.** Reductions and operations take `axis=`. The axis you name is the axis that *disappears*. Get this wrong and you compute the wrong quantity silently.

Every later week of C5 — pandas vectorization in Week 2, batched gradient descent in Week 7, PyTorch tensor ops in Week 8 — uses the same three habits. The libraries change. The discipline does not.

---

## 9. Self-check

Without re-reading:

1. State the three broadcasting rules in plain English.
2. Given `a.shape == (5, 3)` and `b.shape == (3,)`, what is the shape of `a + b`? What if `b.shape == (5,)`?
3. You want to subtract the mean of each *column* from every element in that column. Write the one-line expression.
4. `a.sum(axis=0)` versus `a.sum(axis=1)` on a `(3, 4)` array: which is `(4,)` and which is `(3,)`?
5. Why does `np.broadcast_to(b, (1000, 3))` return a read-only array? Why is its `strides[0]` zero?
6. Translate `np.einsum('ij,jk->ik', A, B)` into a NumPy operator and into a maths statement.
7. An `(H, W, 3)` `uint8` image. Write a single expression to flip it vertically without copying.

---

## Further reading

- **NumPy — Broadcasting**: <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- **NumPy — `np.einsum`**: <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
- **Rougier, *From Python to NumPy*, chapter 3 ("Code vectorization") and chapter 4 ("Problem vectorization")**:
  <https://www.labri.fr/perso/nrougier/from-python-to-numpy/>
- **Jake VanderPlas, *Python Data Science Handbook*, chapter 2 (free):
  <https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html>
- **Tim Rocktäschel, "Einsum is all you need"** (search the title; classic blog post):
  free, MIT-licensed code.

Next: take the [quiz](../quiz.md), do the [exercises](../exercises/README.md), and start the [mini-project](../mini-project/README.md).
