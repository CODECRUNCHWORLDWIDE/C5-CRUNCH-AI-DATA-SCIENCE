"""
Exercise 1 — Shape, dtype, strides; view vs copy.

Goal: build the mental model from Lecture 1 with your fingers. You will
create arrays, inspect their attributes, reshape and transpose them, and
in each case predict whether the result shares memory with the original
before checking with np.shares_memory.

Estimated time: 35 minutes.

Run with:    python exercise-01-shape-dtype-strides.py
Or test:     pytest exercise-01-shape-dtype-strides.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK — exercise 1" at the end with no AssertionError.
- The five pytest functions all pass.

Do not look at the HINT block at the bottom until you have tried for
fifteen minutes.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Part A — Build and inspect
# -----------------------------------------------------------------------------

def make_matrix() -> np.ndarray:
    """Return a 3-row, 4-column array of int64 holding the numbers 0..11.

    Use np.arange and reshape. Do NOT type the literal [[0,1,2,3],...].
    """
    # TODO: implement
    raise NotImplementedError("Part A — make_matrix")


def describe(a: np.ndarray) -> dict[str, object]:
    """Return a dict of the array's shape, dtype, ndim, size, itemsize,
    nbytes, strides, and whether it is C-contiguous.

    Keys must match exactly:
      'shape', 'dtype', 'ndim', 'size', 'itemsize', 'nbytes',
      'strides', 'c_contiguous'.
    """
    # TODO: implement. Use a.flags['C_CONTIGUOUS'] for the boolean.
    raise NotImplementedError("Part A — describe")


# -----------------------------------------------------------------------------
# Part B — View vs copy
# -----------------------------------------------------------------------------

def basic_slice_view(a: np.ndarray) -> np.ndarray:
    """Return the second row of `a` as a basic slice (a view).

    For a (3, 4) input this should have shape (4,).
    """
    # TODO: implement using basic slicing (NOT fancy indexing).
    raise NotImplementedError("Part B — basic_slice_view")


def fancy_index_copy(a: np.ndarray) -> np.ndarray:
    """Return rows 0 and 2 of `a` as a fancy-indexed copy.

    For a (3, 4) input this should have shape (2, 4).
    """
    # TODO: implement using an integer-array index, e.g. a[[..., ...]].
    raise NotImplementedError("Part B — fancy_index_copy")


def transpose_view(a: np.ndarray) -> np.ndarray:
    """Return the transpose of `a`. For (3, 4) input, output is (4, 3)."""
    # TODO: implement (one short expression).
    raise NotImplementedError("Part B — transpose_view")


# -----------------------------------------------------------------------------
# Part C — Dtype matters
# -----------------------------------------------------------------------------

def uint8_overflow_demo() -> np.ndarray:
    """Return np.array([200, 200, 200], dtype=np.uint8) PLUS 100.

    Do NOT cast first. Return the (wrapped) result. The test below checks
    that you got the silent-overflow behavior.
    """
    # TODO: implement
    raise NotImplementedError("Part C — uint8_overflow_demo")


def uint8_safe_add() -> np.ndarray:
    """Same arithmetic as uint8_overflow_demo, but cast UP before adding so
    you do not lose information. Return an int16 array of [300, 300, 300].
    """
    # TODO: implement using .astype(np.int16) first.
    raise NotImplementedError("Part C — uint8_safe_add")


# -----------------------------------------------------------------------------
# Pytest-style checks (also run when the file is executed directly).
# -----------------------------------------------------------------------------

def test_make_matrix_shape_and_values() -> None:
    a = make_matrix()
    assert a.shape == (3, 4), f"expected (3, 4), got {a.shape}"
    assert a.dtype == np.int64 or a.dtype == np.int32, \
        f"expected an int dtype, got {a.dtype}"
    assert a.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_describe_keys_and_types() -> None:
    a = make_matrix()
    d = describe(a)
    for k in ("shape", "dtype", "ndim", "size",
              "itemsize", "nbytes", "strides", "c_contiguous"):
        assert k in d, f"describe missing key: {k}"
    assert d["ndim"] == 2
    assert d["size"] == 12
    assert d["c_contiguous"] is True


def test_basic_slice_returns_view() -> None:
    a = make_matrix()
    v = basic_slice_view(a)
    assert v.shape == (4,)
    assert np.shares_memory(a, v), "basic slice should share memory"


def test_fancy_index_returns_copy() -> None:
    a = make_matrix()
    c = fancy_index_copy(a)
    assert c.shape == (2, 4)
    assert not np.shares_memory(a, c), "fancy index should NOT share memory"


def test_transpose_is_a_view() -> None:
    a = make_matrix()
    t = transpose_view(a)
    assert t.shape == (4, 3)
    assert np.shares_memory(a, t), "transpose should share memory"


def test_uint8_overflow_and_safe() -> None:
    wrapped = uint8_overflow_demo()
    assert wrapped.dtype == np.uint8
    assert wrapped.tolist() == [44, 44, 44], \
        "uint8 200 + 100 should wrap mod 256 to 44"
    safe = uint8_safe_add()
    assert safe.dtype == np.int16
    assert safe.tolist() == [300, 300, 300]


def _run_all_tests() -> None:
    test_make_matrix_shape_and_values()
    test_describe_keys_and_types()
    test_basic_slice_returns_view()
    test_fancy_index_returns_copy()
    test_transpose_is_a_view()
    test_uint8_overflow_and_safe()
    print("OK — exercise 1")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# make_matrix:
#     return np.arange(12).reshape(3, 4)
#
# describe:
#     return {
#         "shape": a.shape,
#         "dtype": a.dtype,
#         "ndim": a.ndim,
#         "size": a.size,
#         "itemsize": a.itemsize,
#         "nbytes": a.nbytes,
#         "strides": a.strides,
#         "c_contiguous": bool(a.flags["C_CONTIGUOUS"]),
#     }
#
# basic_slice_view:
#     return a[1, :]               # a[1] also works
#
# fancy_index_copy:
#     return a[[0, 2]]             # integer-array index => copy
#
# transpose_view:
#     return a.T
#
# uint8_overflow_demo:
#     return np.array([200, 200, 200], dtype=np.uint8) + np.uint8(100)
#
# uint8_safe_add:
#     return np.array([200, 200, 200], dtype=np.uint8).astype(np.int16) + 100
#
# -----------------------------------------------------------------------------
