"""
Exercise 1 -- Tensors and Autograd.

Goal: build a working mental model of torch.Tensor and torch.autograd by
re-implementing Week 7's NumPy MLP gradients with PyTorch's autograd and
verifying the two agree to within 1e-5 relative error.

By the end of this exercise you will have:

    (1) Created tensors with the five idiomatic patterns (Section 1 of Lecture 1).
    (2) Built the two-layer MLP forward pass with raw autograd (no nn.Module yet).
    (3) Called .backward() on the cross-entropy loss and inspected the .grad on
        every parameter.
    (4) Re-implemented the Week 7 NumPy backward pass and compared the gradients
        side by side, parameter by parameter.
    (5) Written the manual SGD update with `with torch.no_grad():` (Lecture 1
        Section 9), and verified the loss decreases on a small synthetic batch.

Estimated time: 60-90 minutes.

Run with:    python exercise-01-tensors-and-autograd.py
Or test:     pytest exercise-01-tensors-and-autograd.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-tensors-and-autograd.py succeeds.

The exercise uses synthetic data so the tests never depend on the dataset.

PyTorch reference: https://pytorch.org/docs/stable/autograd.html
Week 7 reference:  ../../week-07-neural-networks-from-scratch/lecture-notes/02-backprop-from-chain-rule.md
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Part A -- create tensors in the five idiomatic patterns
# -----------------------------------------------------------------------------

def make_tensor_from_list() -> "torch.Tensor":
    """Return a 1D float tensor containing [1.0, 2.0, 3.0].

    Use torch.tensor (the copy form).
    Reference: https://pytorch.org/docs/stable/generated/torch.tensor.html
    """
    # TODO: return torch.tensor([1.0, 2.0, 3.0])
    raise NotImplementedError("Part A -- make_tensor_from_list")


def make_tensor_from_numpy(arr: "NDArray[np.float32]") -> "torch.Tensor":
    """Return a zero-copy view of the NumPy array `arr`.

    Use torch.from_numpy; the result must share memory with `arr`.
    Reference: https://pytorch.org/docs/stable/generated/torch.from_numpy.html
    """
    # TODO: return torch.from_numpy(arr)
    raise NotImplementedError("Part A -- make_tensor_from_numpy")


def make_zeros(shape: "Tuple[int, ...]") -> "torch.Tensor":
    """Return a tensor of zeros with the given shape, dtype float32.

    Reference: https://pytorch.org/docs/stable/generated/torch.zeros.html
    """
    # TODO: return torch.zeros(*shape, dtype=torch.float32)
    raise NotImplementedError("Part A -- make_zeros")


def make_randn(shape: "Tuple[int, ...]", seed: int = RANDOM_STATE) -> "torch.Tensor":
    """Return a tensor sampled from N(0, 1) with the given shape.

    Use a torch.Generator seeded with `seed` so the test is deterministic.
    Reference: https://pytorch.org/docs/stable/generated/torch.randn.html
    """
    # TODO: build a generator with the seed and call torch.randn(..., generator=...)
    raise NotImplementedError("Part A -- make_randn")


# -----------------------------------------------------------------------------
# Part B -- autograd on a simple scalar function
# -----------------------------------------------------------------------------

def grad_of_scalar(x_value: float) -> float:
    """Return the derivative of y = sin(x) * x^2 at x = x_value, via autograd.

    Construct a leaf tensor with requires_grad=True at x_value, compute
    y = torch.sin(x) * x**2, call y.backward(), and return x.grad as a
    Python float via .item().

    The analytical answer is 2x sin(x) + x^2 cos(x); the test compares against
    it to within 1e-5.

    Reference: https://pytorch.org/docs/stable/autograd.html
    """
    # TODO: implement
    raise NotImplementedError("Part B -- grad_of_scalar")


# -----------------------------------------------------------------------------
# Part C -- two-layer MLP forward pass with raw autograd
# -----------------------------------------------------------------------------

def init_params_pytorch(
    n_in: int, n_hidden: int, n_out: int, seed: int = RANDOM_STATE,
) -> "dict[str, torch.Tensor]":
    """He initialization for the weights, zero initialization for biases.

    All four returned tensors must have requires_grad=True. He scaling:
        W1 ~ N(0, sqrt(2 / n_in)),    shape (n_in, n_hidden)
        W2 ~ N(0, sqrt(2 / n_hidden)), shape (n_hidden, n_out)
        b1, b2 = zeros

    NOTE: To match the Week 7 NumPy storage convention, we keep W1 with shape
    (n_in, n_hidden) and use Z1 = X @ W1 + b1. This is NOT the nn.Linear
    convention (which transposes); see Lecture 2 Section 2 for the difference.

    Returns dict with keys "W1", "b1", "W2", "b2".
    """
    # TODO: build the four tensors with requires_grad=True
    # HINT:
    #   g = torch.Generator().manual_seed(seed)
    #   W1 = torch.randn(n_in, n_hidden, generator=g) * (2.0 / n_in) ** 0.5
    #   W1.requires_grad_(True)
    #   ...
    raise NotImplementedError("Part C -- init_params_pytorch")


def forward_raw(
    X: "torch.Tensor", params: "dict[str, torch.Tensor]",
) -> "torch.Tensor":
    """Two-layer MLP forward pass returning raw logits Z2.

    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    return Z2   # raw logits, no softmax

    NOTE: We do NOT apply softmax here. CrossEntropyLoss does log_softmax
    internally; doing it twice is numerically wrong. See Lecture 2 Section 6.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
    """
    # TODO: implement
    raise NotImplementedError("Part C -- forward_raw")


# -----------------------------------------------------------------------------
# Part D -- the Week 7 NumPy backward pass (the reference implementation)
# -----------------------------------------------------------------------------

def numpy_softmax(z: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """Numerically stable softmax along the last axis (copied from Week 7)."""
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def numpy_backward(
    X: "NDArray[np.float64]",
    y: "NDArray[np.int64]",
    W1: "NDArray[np.float64]", b1: "NDArray[np.float64]",
    W2: "NDArray[np.float64]", b2: "NDArray[np.float64]",
) -> "dict[str, NDArray[np.float64]]":
    """Re-run the Week 7 NumPy backward pass.

    Forward:
        Z1 = X @ W1 + b1
        A1 = max(0, Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)
    Loss = mean cross-entropy.

    Backward (the chain rule, Week 7 Lecture 2):
        dZ2 = (A2 - Y_one_hot) / m
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

    Returns dict with keys "W1", "b1", "W2", "b2" (the gradients).
    """
    m, k = X.shape[0], W2.shape[1]
    Z1 = X @ W1 + b1
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ W2 + b2
    A2 = numpy_softmax(Z2)
    Y = np.zeros((m, k), dtype=np.float64)
    Y[np.arange(m), y] = 1.0

    dZ2 = (A2 - Y) / m                       # (m, k)
    dW2 = A1.T @ dZ2                         # (h, k)
    db2 = dZ2.sum(axis=0)                    # (k,)
    dA1 = dZ2 @ W2.T                         # (m, h)
    dZ1 = dA1 * (Z1 > 0).astype(np.float64)  # (m, h)
    dW1 = X.T @ dZ1                          # (n_in, h)
    db1 = dZ1.sum(axis=0)                    # (h,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# -----------------------------------------------------------------------------
# Part E -- the PyTorch backward pass via autograd
# -----------------------------------------------------------------------------

def pytorch_backward(
    X: "torch.Tensor", y: "torch.Tensor", params: "dict[str, torch.Tensor]",
) -> "dict[str, torch.Tensor]":
    """Run the forward pass, compute the cross-entropy loss, and call backward.

    Returns a dict with keys "W1", "b1", "W2", "b2" mapping to the .grad of
    each parameter tensor.

    Use torch.nn.functional.cross_entropy(logits, y) for numerical stability;
    it expects integer class indices (not one-hot) in `y` and computes the
    mean loss.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

    IMPORTANT: Before calling forward and backward, zero the .grad on every
    parameter so the gradients do not accumulate across test runs.
    """
    # TODO: implement
    # HINT:
    #   for p in params.values():
    #       if p.grad is not None:
    #           p.grad.zero_()
    #   logits = forward_raw(X, params)
    #   loss = torch.nn.functional.cross_entropy(logits, y)
    #   loss.backward()
    #   return {name: p.grad.clone() for name, p in params.items()}
    raise NotImplementedError("Part E -- pytorch_backward")


# -----------------------------------------------------------------------------
# Part F -- manual SGD step in a `with torch.no_grad():` block
# -----------------------------------------------------------------------------

def manual_sgd_step(
    params: "dict[str, torch.Tensor]", lr: float = 0.1,
) -> None:
    """Apply one in-place SGD update to each parameter:

        param -= lr * param.grad
        param.grad.zero_()

    The update must be wrapped in `with torch.no_grad():` to avoid recording
    the in-place op into the autograd graph (Lecture 1 Section 9, Section 10
    Trap #2).

    The function returns nothing; it mutates the parameters in place.
    """
    # TODO: implement
    raise NotImplementedError("Part F -- manual_sgd_step")


# -----------------------------------------------------------------------------
# Helpers used by the tests
# -----------------------------------------------------------------------------

def make_synthetic_batch(
    n_samples: int = 128,
    n_features: int = 784,
    n_classes: int = 10,
    seed: int = RANDOM_STATE,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Synthetic input + integer labels for testing. Not learnable."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, n_features, generator=g, dtype=torch.float64)
    y = torch.randint(0, n_classes, (n_samples,), generator=g, dtype=torch.int64)
    return X, y


def params_to_numpy(
    params: "dict[str, torch.Tensor]",
) -> "dict[str, NDArray[np.float64]]":
    """Detach each parameter and convert to a float64 NumPy array."""
    return {k: v.detach().cpu().double().numpy() for k, v in params.items()}


def relative_error(
    a: "NDArray[np.float64]", b: "NDArray[np.float64]",
) -> float:
    """Max relative error between two arrays; the standard gradient-check
    metric from Week 7 Lecture 2 Section 10."""
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0.0, 1.0, denom)
    return float(np.max(np.abs(a - b) / denom))


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_tensor_from_list_is_a_tensor() -> None:
    t = make_tensor_from_list()
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3,)
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))


def test_tensor_from_numpy_shares_memory() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = make_tensor_from_numpy(arr)
    t[0] = 99.0
    assert arr[0] == 99.0, "from_numpy should share memory"


def test_make_zeros_shape_and_dtype() -> None:
    t = make_zeros((2, 3))
    assert t.shape == (2, 3)
    assert t.dtype == torch.float32
    assert torch.all(t == 0.0)


def test_make_randn_deterministic() -> None:
    a = make_randn((2, 3), seed=123)
    b = make_randn((2, 3), seed=123)
    assert torch.allclose(a, b)


def test_grad_of_scalar_matches_analytical() -> None:
    # dy/dx at x=3 for y = sin(x)*x^2 is 2*3*sin(3) + 9*cos(3)
    x = 3.0
    expected = 2.0 * x * np.sin(x) + x ** 2 * np.cos(x)
    actual = grad_of_scalar(x)
    assert abs(actual - expected) < 1e-5, f"actual {actual}, expected {expected}"


def test_init_params_shapes() -> None:
    p = init_params_pytorch(784, 64, 10)
    assert p["W1"].shape == (784, 64)
    assert p["b1"].shape == (64,)
    assert p["W2"].shape == (64, 10)
    assert p["b2"].shape == (10,)
    for name in ("W1", "b1", "W2", "b2"):
        assert p[name].requires_grad, f"{name} should require grad"


def test_forward_raw_output_shape() -> None:
    X, _ = make_synthetic_batch(n_samples=8)
    X = X.float()
    p = init_params_pytorch(784, 64, 10)
    # cast params to float32 to match X
    p = {k: v.detach().float().requires_grad_(True) for k, v in p.items()}
    logits = forward_raw(X, p)
    assert logits.shape == (8, 10)


def test_pytorch_backward_matches_numpy_to_1e_5() -> None:
    """The headline test: PyTorch autograd must agree with the Week 7 NumPy
    backward to 1e-5 relative error on every parameter."""
    n_in, n_hidden, n_out = 50, 32, 5
    m = 16

    g = torch.Generator().manual_seed(0)
    X_torch = torch.randn(m, n_in, generator=g, dtype=torch.float64)
    y_torch = torch.randint(0, n_out, (m,), generator=g, dtype=torch.int64)

    params = init_params_pytorch(n_in, n_hidden, n_out, seed=0)
    # Cast to float64 to match NumPy precision.
    params = {k: v.detach().double().requires_grad_(True) for k, v in params.items()}

    np_params = params_to_numpy(params)
    X_np = X_torch.numpy()
    y_np = y_torch.numpy()

    pt_grads = pytorch_backward(X_torch, y_torch, params)
    np_grads = numpy_backward(
        X_np, y_np,
        np_params["W1"], np_params["b1"],
        np_params["W2"], np_params["b2"],
    )

    for name in ("W1", "b1", "W2", "b2"):
        pt = pt_grads[name].detach().cpu().double().numpy()
        nm = np_grads[name]
        err = relative_error(pt, nm)
        assert err < 1e-5, (
            f"gradient mismatch on {name}: relative error {err:.2e}"
        )


def test_manual_sgd_step_decreases_loss() -> None:
    """After one manual SGD step on a small problem, the loss should drop."""
    X, y = make_synthetic_batch(n_samples=32, n_features=20, n_classes=4)
    X = X.float()

    params = init_params_pytorch(20, 16, 4, seed=1)
    params = {k: v.detach().float().requires_grad_(True) for k, v in params.items()}

    # Initial loss
    logits = forward_raw(X, params)
    loss_before = torch.nn.functional.cross_entropy(logits, y).item()

    # One backward + SGD step
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()
    logits = forward_raw(X, params)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    manual_sgd_step(params, lr=0.1)

    # Loss after
    with torch.no_grad():
        logits_after = forward_raw(X, params)
        loss_after = torch.nn.functional.cross_entropy(logits_after, y).item()

    # Loss should decrease (or at worst stay roughly equal on this random problem).
    # We require a strict decrease because the synthetic problem is large enough.
    assert loss_after < loss_before, (
        f"loss did not decrease: before {loss_before:.4f}, after {loss_after:.4f}"
    )


def test_manual_sgd_step_zeros_grad() -> None:
    """After manual_sgd_step, the .grad on each parameter should be zero."""
    X, y = make_synthetic_batch(n_samples=8, n_features=10, n_classes=3)
    X = X.float()
    params = init_params_pytorch(10, 8, 3, seed=2)
    params = {k: v.detach().float().requires_grad_(True) for k, v in params.items()}
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()
    logits = forward_raw(X, params)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    manual_sgd_step(params, lr=0.01)
    for name, p in params.items():
        assert p.grad is not None, f"{name}.grad is None after sgd step"
        assert torch.all(p.grad == 0.0), f"{name}.grad was not zeroed"


def _run_all_tests() -> None:
    test_tensor_from_list_is_a_tensor()
    test_tensor_from_numpy_shares_memory()
    test_make_zeros_shape_and_dtype()
    test_make_randn_deterministic()
    test_grad_of_scalar_matches_analytical()
    test_init_params_shapes()
    test_forward_raw_output_shape()
    test_pytorch_backward_matches_numpy_to_1e_5()
    test_manual_sgd_step_decreases_loss()
    test_manual_sgd_step_zeros_grad()
    print("OK -- exercise 1")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# make_tensor_from_list:
#     return torch.tensor([1.0, 2.0, 3.0])
#
# make_tensor_from_numpy:
#     return torch.from_numpy(arr)
#
# make_zeros:
#     return torch.zeros(*shape, dtype=torch.float32)
#
# make_randn:
#     g = torch.Generator().manual_seed(seed)
#     return torch.randn(*shape, generator=g)
#
# grad_of_scalar:
#     x = torch.tensor(x_value, requires_grad=True)
#     y = torch.sin(x) * x ** 2
#     y.backward()
#     return float(x.grad.item())
#
# init_params_pytorch:
#     g = torch.Generator().manual_seed(seed)
#     W1 = (torch.randn(n_in, n_hidden, generator=g) * (2.0 / n_in) ** 0.5).requires_grad_(True)
#     b1 = torch.zeros(n_hidden, requires_grad=True)
#     W2 = (torch.randn(n_hidden, n_out, generator=g) * (2.0 / n_hidden) ** 0.5).requires_grad_(True)
#     b2 = torch.zeros(n_out, requires_grad=True)
#     return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#
# forward_raw:
#     Z1 = X @ params["W1"] + params["b1"]
#     A1 = torch.relu(Z1)
#     Z2 = A1 @ params["W2"] + params["b2"]
#     return Z2
#
# pytorch_backward:
#     for p in params.values():
#         if p.grad is not None:
#             p.grad.zero_()
#     logits = forward_raw(X, params)
#     loss = torch.nn.functional.cross_entropy(logits, y)
#     loss.backward()
#     return {k: v.grad.clone() for k, v in params.items()}
#
# manual_sgd_step:
#     with torch.no_grad():
#         for p in params.values():
#             p -= lr * p.grad
#             p.grad.zero_()
#
# -----------------------------------------------------------------------------
