# Week 1 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** Why is a Python `list` of one million floats slower to sum than a NumPy `float64` array of the same size?

- A) Python's `sum()` is implemented in pure Python; NumPy's `sum()` is in C.
- B) The list stores pointers to `PyFloatObject` boxes; the `ndarray` stores raw `double`s contiguously in memory.
- C) NumPy uses multiple threads by default; Python is single-threaded.
- D) The GIL is released for NumPy operations.

---

**Q2.** Given `a = np.arange(12).reshape(3, 4)` with `dtype=np.int64`, what are `a.strides`?

- A) `(1, 4)`
- B) `(4, 1)`
- C) `(32, 8)`
- D) `(8, 32)`

---

**Q3.** Which of the following operations is guaranteed to return a **view** (shared memory) rather than a copy?

- A) `a[a > 0]`
- B) `a[[0, 2, 4]]`
- C) `a[::2]`
- D) `a.astype(np.float32)`

---

**Q4.** Given `a.shape == (5, 3)` and `b.shape == (3,)`, what is the shape of `a + b`?

- A) `(5, 3)`
- B) `(3,)`
- C) `(5,)`
- D) ValueError — shapes are incompatible.

---

**Q5.** You have `X` of shape `(5, 3)` and want to subtract the mean of each **row** from every element of that row. Which expression is correct?

- A) `X - X.mean(axis=0)`
- B) `X - X.mean(axis=1)`
- C) `X - X.mean(axis=1)[:, None]`
- D) `X - X.mean(axis=1).reshape(1, 5)`

---

**Q6.** What does `a.sum(axis=0)` return for `a` of shape `(3, 4)`?

- A) A scalar.
- B) An array of shape `(3,)`.
- C) An array of shape `(4,)`.
- D) An array of shape `(3, 4)`.

---

**Q7.** What does `np.einsum('ij,jk->ik', A, B)` compute?

- A) Element-wise multiplication of `A` and `B`.
- B) The outer product of `A` and `B`.
- C) The matrix product `A @ B`.
- D) The trace of `A @ B`.

---

**Q8.** Given `a = np.array([200, 200, 200], dtype=np.uint8)`, what is `a + np.uint8(100)` in NumPy 2.x?

- A) `array([300, 300, 300], dtype=int64)` — auto-promoted.
- B) `array([44, 44, 44], dtype=uint8)` — silent wrap-around.
- C) Raises an `OverflowError`.
- D) `array([255, 255, 255], dtype=uint8)` — saturated.

---

**Q9.** Which of the following is the **NumPy 2.x preferred** way to seed and use a random generator?

- A) `np.random.seed(42); x = np.random.randn(5)`
- B) `rng = np.random.default_rng(42); x = rng.normal(size=5)`
- C) `import random; random.seed(42); x = [random.gauss(0, 1) for _ in range(5)]`
- D) `x = np.random.RandomState(42).randn(5)`

---

**Q10.** A `(H, W, 3)` `uint8` image, with H = 480, W = 640. How many bytes does its underlying buffer occupy?

- A) ~150 KB
- B) ~300 KB
- C) ~900 KB
- D) ~7 MB

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — A `list` of floats is an array of pointers to `PyFloatObject` boxes scattered on the heap (24 B of metadata per 8 B `double`). The `ndarray` is one contiguous C buffer of `double`s. The 50–100× speedup is the elimination of per-element boxing and the use of SIMD-friendly contiguous reads.

2. **C** — `int64` is 8 bytes. A `(3, 4)` row-major array steps 8 bytes per column (axis 1) and `4 * 8 = 32` bytes per row (axis 0). `strides = (32, 8)`. Option D `(8, 32)` would be Fortran (column-major) order.

3. **C** — Basic slicing `[::2]` returns a view. Boolean masks (A) and integer-array indices (B) return copies; `astype` (D) almost always copies. `np.shares_memory(a, a[::2])` is `True`; for the others it is `False`.

4. **A** — Broadcasting compares shapes right-to-left. `(5, 3)` vs `(3,)` → align as `(5, 3)` vs `(1, 3)` (the missing leading axis is treated as 1). Result shape is `(5, 3)`.

5. **C** — `X.mean(axis=1)` collapses axis 1, returning shape `(5,)`. To broadcast across **rows**, we need to align it with axis 0 of `X`, so we reshape to `(5, 1)` via `[:, None]` (equivalently `.reshape(5, 1)`). Option B fails because `(5, 3) - (5,)` aligns `(5,)` with the *last* axis and complains. Option A subtracts column means.

6. **C** — `axis=k` collapses axis `k`. `(3, 4).sum(axis=0)` removes axis 0, leaving `(4,)`. The rule of thumb: *the axis you name is the axis that disappears.*

7. **C** — `'ij,jk->ik'`: `A` is indexed by `(i, j)`, `B` by `(j, k)`, output by `(i, k)`. `j` appears only on the left → sum over it. That is the definition of matrix multiplication.

8. **B** — Silent wrap-around. `uint8` holds 0–255; `200 + 100 = 300 mod 256 = 44`. NEP-50 in NumPy 2.x preserves the array's dtype in scalar-array ops, so no auto-promotion to `int64` happens. Cast up (`.astype(np.int16)`) when you need range.

9. **B** — `np.random.default_rng()` is the 2.x preferred API. It returns a `Generator` instance backed by PCG64, with no global state. The legacy `np.random.seed` / `randn` API is a compatibility shim and is documented as such.

10. **C** — `480 * 640 * 3 = 921,600 bytes ≈ 900 KB`. `uint8` is 1 byte per element, no metadata overhead. (As `float32` the same image would be 3.7 MB; as `float64` it would be 7.4 MB.)

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
