"""
Exercise 2 -- ReLU, Softmax, Cross-Entropy, and the Backward Pass.

Goal: build the *primitive functions* of a 2-layer MLP and the backward pass
that ties them together. Verify the analytical gradient against a numerical
finite-difference gradient on a tiny synthetic batch -- this is the
"gradient check" of Lecture 2 Section 10, the single most important
debugging tool in deep learning.

By the end of this exercise you will have:

    (1) relu(z) and relu_grad(z) (Lecture 2, Section 7).
    (2) A re-derived stable softmax (Lecture 1, Section 5).
    (3) A cross-entropy loss with log(0) clipping (Lecture 1, Section 6).
    (4) A backward pass that computes dW1, db1, dW2, db2 (Lecture 2).
    (5) A gradient check that verifies the backward pass agrees with the
        finite-difference numerical gradient to within 1e-5 relative error.

Estimated time: 90-120 minutes. The derivation is one page; the
implementation is fifteen lines; the gradient check is what catches the bug.

Run with:    python exercise-02-implement-relu-softmax-ce.py
Or test:     pytest exercise-02-implement-relu-softmax-ce.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The gradient check passes with max relative error < 1e-5 for every parameter.
- python -m py_compile exercise-02-implement-relu-softmax-ce.py succeeds.

The exercise uses tiny synthetic data so the gradient check is exhaustive
across every parameter; this is the right scale to debug at.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Part A -- ReLU and its gradient
# -----------------------------------------------------------------------------

def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise ReLU: max(0, z).

    Parameters
    ----------
    z -- (any-shape) float array

    Returns
    -------
    a -- (same-shape) float array with negative entries zeroed
    """
    # TODO: implement element-wise ReLU.
    # See HINT block at the bottom of the file.
    raise NotImplementedError("Part A -- relu")


def relu_grad(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise derivative of ReLU.

    The derivative is 1 where z > 0, else 0. The boundary point z == 0
    has undefined derivative; we follow the convention of returning 0
    (some libraries use 0.5; the difference is measure-zero in practice).

    Parameters
    ----------
    z -- (any-shape) float array (the pre-activation; *not* the activation)

    Returns
    -------
    g -- (same-shape) float array with values in {0.0, 1.0}
    """
    # TODO: implement the ReLU derivative.
    raise NotImplementedError("Part A -- relu_grad")


# -----------------------------------------------------------------------------
# Part B -- the stable softmax (re-implemented)
# -----------------------------------------------------------------------------

def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax over the last axis.

    Identical to Exercise 1 Part B; re-implemented here so this exercise
    is self-contained.
    """
    # TODO: re-implement the stable softmax.
    raise NotImplementedError("Part B -- softmax")


# -----------------------------------------------------------------------------
# Part C -- cross-entropy loss
# -----------------------------------------------------------------------------

def cross_entropy_loss(
    probs: NDArray[np.float64],
    y: NDArray[np.int64],
) -> float:
    """Mean cross-entropy loss over a batch. Re-implementation of Exercise 1
    Part D so this exercise is self-contained.
    """
    # TODO: re-implement cross-entropy loss.
    raise NotImplementedError("Part C -- cross_entropy_loss")


# -----------------------------------------------------------------------------
# Part D -- the forward pass (used by the backward pass)
# -----------------------------------------------------------------------------

def forward(
    X: NDArray[np.float64],
    params: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Forward pass for a 2-layer ReLU + softmax MLP. Identical to Exercise 1
    Part C.
    """
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = softmax(Z2)
    return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def forward_and_loss(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    params: dict[str, NDArray[np.float64]],
) -> tuple[float, dict[str, NDArray[np.float64]]]:
    """Forward pass + loss. Returns (loss, cache).
    """
    cache = forward(X, params)
    loss = cross_entropy_loss(cache["A2"], y)
    cache["y"] = y
    return loss, cache


# -----------------------------------------------------------------------------
# Part E -- the backward pass
# -----------------------------------------------------------------------------

def backward(
    cache: dict[str, NDArray[np.float64]],
    params: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Compute the gradients of the loss with respect to W1, b1, W2, b2.

    Parameters
    ----------
    cache  -- the dict returned by forward(), with keys X, Z1, A1, A2, y
    params -- the dict containing W1, b1, W2, b2

    Returns
    -------
    grads  -- a dict with the same keys as params; same shapes as params.

    The derivation (Lecture 2, Sections 4-8):

        1. dZ2 = (A2 - Y) / m              ; softmax + CE simplification
        2. dW2 = A1.T @ dZ2                ; backprop through W2
        3. db2 = dZ2.sum(axis=0)           ; backprop through b2
        4. dA1 = dZ2 @ W2.T                ; backprop through W2 to A1
        5. dZ1 = dA1 * relu_grad(Z1)       ; backprop through ReLU
        6. dW1 = X.T @ dZ1                 ; backprop through W1
        7. db1 = dZ1.sum(axis=0)           ; backprop through b1

    Where Y is the one-hot encoding of y.
    """
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]
    y = cache["y"]
    W2 = params["W2"]
    m, k = A2.shape

    # One-hot encode the labels.
    Y = np.zeros((m, k), dtype=np.float64)
    Y[np.arange(m), y] = 1.0

    # TODO: implement the 6-line backward pass.
    # See HINT block at the bottom.
    raise NotImplementedError("Part E -- backward")


# -----------------------------------------------------------------------------
# Part F -- gradient checking
# -----------------------------------------------------------------------------

def numerical_gradient(
    params: dict[str, NDArray[np.float64]],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    param_name: str,
    idx: tuple[int, ...],
    eps: float = 1e-7,
) -> float:
    """Compute the centered finite-difference gradient of the loss with
    respect to params[param_name][idx].

    The formula (Lecture 2, Section 10):
        g_num = ( L(theta + eps*e_i) - L(theta - eps*e_i) ) / (2*eps)

    where e_i is the standard basis vector at idx. We mutate params in
    place to compute the +/- losses, then restore the original value.
    """
    p = params[param_name]
    original = float(p[idx])

    p[idx] = original + eps
    loss_plus, _ = forward_and_loss(X, y, params)

    p[idx] = original - eps
    loss_minus, _ = forward_and_loss(X, y, params)

    p[idx] = original

    return (loss_plus - loss_minus) / (2.0 * eps)


def gradient_check(
    params: dict[str, NDArray[np.float64]],
    grads: dict[str, NDArray[np.float64]],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_samples: int = 12,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compare the analytical gradients to numerical gradients at n_samples
    random elements of each parameter array. Return the max relative error
    per parameter.

    The relative-error formula (Lecture 2, Section 10):
        rel_err = | g_ana - g_num |  /  max(| g_ana |, | g_num |, 1e-12)

    Accept if max rel_err < 1e-5.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    max_errs: dict[str, float] = {}
    for k in params:
        p = params[k]
        g_a = grads[k]
        n_total = p.size
        n_to_check = min(n_samples, n_total)
        flat_idx = rng.choice(n_total, size=n_to_check, replace=False)
        max_err = 0.0
        for fi in flat_idx:
            idx = np.unravel_index(int(fi), p.shape)
            g_num = numerical_gradient(params, X, y, k, idx)
            g_ana = float(g_a[idx])
            denom = max(abs(g_ana), abs(g_num), 1e-12)
            rel_err = abs(g_ana - g_num) / denom
            max_err = max(max_err, rel_err)
        max_errs[k] = max_err
    return max_errs


# -----------------------------------------------------------------------------
# Helpers used by the tests
# -----------------------------------------------------------------------------

def init_tiny_mlp(
    n_in: int = 5,
    n_hidden: int = 4,
    n_out: int = 3,
    rng: np.random.Generator | None = None,
) -> dict[str, NDArray[np.float64]]:
    """He initialization on a tiny MLP. Used for the gradient check."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden)),
        "b1": np.zeros((n_hidden,), dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out)),
        "b2": np.zeros((n_out,), dtype=np.float64),
    }


def make_tiny_batch(
    n_samples: int = 7,
    n_features: int = 5,
    n_classes: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float64)
    y = rng.integers(0, n_classes, size=(n_samples,)).astype(np.int64)
    return X, y


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_relu_basic() -> None:
    z = np.array([[-2.0, -0.5, 0.0, 1.5, 3.0]])
    a = relu(z)
    assert np.allclose(a, [[0.0, 0.0, 0.0, 1.5, 3.0]])


def test_relu_grad_basic() -> None:
    z = np.array([[-2.0, -0.5, 0.0, 1.5, 3.0]])
    g = relu_grad(z)
    # By convention, derivative at z=0 is 0.
    assert np.allclose(g, [[0.0, 0.0, 0.0, 1.0, 1.0]])


def test_softmax_basic() -> None:
    p = softmax(np.array([[0.0, 0.0, 0.0]]))
    assert np.allclose(p, 1.0 / 3.0)


def test_softmax_no_overflow() -> None:
    p = softmax(np.array([[1000.0, 1000.0]]))
    assert np.all(np.isfinite(p))
    assert np.allclose(p, 0.5)


def test_cross_entropy_uniform_is_log_k() -> None:
    probs = np.full((10, 5), 1.0 / 5)
    y = np.zeros(10, dtype=np.int64)
    loss = cross_entropy_loss(probs, y)
    assert abs(loss - float(np.log(5))) < 1e-9


def test_forward_shapes_tiny() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_tiny_mlp(rng=rng)
    X, _ = make_tiny_batch(rng=rng)
    cache = forward(X, params)
    assert cache["Z1"].shape == (7, 4)
    assert cache["A1"].shape == (7, 4)
    assert cache["Z2"].shape == (7, 3)
    assert cache["A2"].shape == (7, 3)


def test_backward_returns_correct_shapes() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_tiny_mlp(rng=rng)
    X, y = make_tiny_batch(rng=rng)
    _, cache = forward_and_loss(X, y, params)
    grads = backward(cache, params)
    for k in params:
        assert grads[k].shape == params[k].shape, (
            f"grad[{k}] had shape {grads[k].shape}, "
            f"params[{k}] had shape {params[k].shape}"
        )


def test_backward_gradient_check_passes() -> None:
    """The most important test in this exercise.

    The analytical gradient from backward() must agree with the
    numerical (finite-difference) gradient to within 1e-5 relative
    error on a tiny batch. This is the Lecture 2 Section 10 gradient
    check; if this test passes, your derivation is almost certainly
    correct.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_tiny_mlp(rng=rng)
    X, y = make_tiny_batch(rng=rng)
    _, cache = forward_and_loss(X, y, params)
    grads = backward(cache, params)
    max_errs = gradient_check(params, grads, X, y, n_samples=12, rng=rng)

    for k, err in max_errs.items():
        assert err < 1e-5, (
            f"gradient check FAILED for {k}: max relative error = {err:.3e}. "
            f"Re-derive {k}'s gradient -- one of:\n"
            f"  - missing 1/m factor in dZ2\n"
            f"  - wrong transpose (X.T vs A1.T vs W2.T)\n"
            f"  - ReLU mask uses A1 instead of Z1\n"
            f"  - missing sum(axis=0) on the bias gradient\n"
        )


def test_backward_step_decreases_loss() -> None:
    """After one SGD step with a small learning rate, the loss should
    decrease. If it increases, the gradient has the wrong sign."""
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_tiny_mlp(rng=rng)
    X, y = make_tiny_batch(rng=rng)
    loss_before, cache = forward_and_loss(X, y, params)
    grads = backward(cache, params)
    eta = 0.01
    for k in params:
        params[k] -= eta * grads[k]
    loss_after, _ = forward_and_loss(X, y, params)
    assert loss_after < loss_before, (
        f"loss INCREASED from {loss_before:.4f} to {loss_after:.4f} after "
        f"one SGD step. Likely the sign of the gradient update is wrong: "
        f"use params -= eta * grads, NOT params += eta * grads."
    )


def _run_all_tests() -> None:
    test_relu_basic()
    test_relu_grad_basic()
    test_softmax_basic()
    test_softmax_no_overflow()
    test_cross_entropy_uniform_is_log_k()
    test_forward_shapes_tiny()
    test_backward_returns_correct_shapes()
    test_backward_gradient_check_passes()
    test_backward_step_decreases_loss()
    print("OK -- exercise 2")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# relu:
#     return np.maximum(z, 0.0)
#
# relu_grad:
#     return (z > 0).astype(np.float64)
#
# softmax:
#     z_shifted = z - z.max(axis=-1, keepdims=True)
#     exp_z = np.exp(z_shifted)
#     return exp_z / exp_z.sum(axis=-1, keepdims=True)
#
# cross_entropy_loss:
#     m = probs.shape[0]
#     safe = np.clip(probs[np.arange(m), y], 1e-12, 1.0)
#     return float(-np.log(safe).mean())
#
# backward (Lecture 2, Sections 4-8):
#     dZ2 = (A2 - Y) / m
#     dW2 = A1.T @ dZ2
#     db2 = dZ2.sum(axis=0)
#     dA1 = dZ2 @ W2.T
#     dZ1 = dA1 * relu_grad(Z1)
#     dW1 = X.T @ dZ1
#     db1 = dZ1.sum(axis=0)
#     return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
#
# -----------------------------------------------------------------------------
