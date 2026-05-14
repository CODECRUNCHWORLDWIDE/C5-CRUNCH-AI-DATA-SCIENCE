"""
Exercise 1 -- NumPy MLP Skeleton.

Goal: build the *forward-only* infrastructure for a 2-layer MLP in pure
NumPy. The backward pass is Exercise 2; the training loop is Exercise 3.

By the end of this exercise you will have:

    (1) Parameter initialization with He scaling (Lecture 1, Section 9).
    (2) A numerically stable softmax (Lecture 1, Section 5).
    (3) A forward pass that returns a cache for the backward pass.
    (4) A predict function that returns argmax class indices.
    (5) A cross-entropy loss with log(0) clipping.
    (6) A sanity check that the *untrained* initial loss is approximately
        log(k) = log(10) = 2.30 on a synthetic MNIST-like batch.

Estimated time: 60-90 minutes.

Run with:    python exercise-01-numpy-mlp-skeleton.py
Or test:     pytest exercise-01-numpy-mlp-skeleton.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-numpy-mlp-skeleton.py succeeds.

The exercise uses synthetic data so the tests never depend on the network.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Part A -- He initialization
# -----------------------------------------------------------------------------

def init_params(
    n_in: int,
    n_hidden: int,
    n_out: int,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    """He initialization for the weights, zero initialization for biases.

    Parameters
    ----------
    n_in     -- the input feature count (e.g. 784 for MNIST)
    n_hidden -- the hidden layer width (e.g. 128)
    n_out    -- the number of output classes (e.g. 10)
    rng      -- a numpy Generator; use np.random.default_rng(seed) to construct.

    Returns
    -------
    A dict with keys "W1", "b1", "W2", "b2" and the appropriate shapes:
        W1: (n_in, n_hidden),  drawn from N(0, sqrt(2 / n_in))
        b1: (n_hidden,),       zeros
        W2: (n_hidden, n_out), drawn from N(0, sqrt(2 / n_hidden))
        b2: (n_out,),          zeros

    See Lecture 1 Section 9 for the He initialization derivation
    (He et al. 2015, https://arxiv.org/abs/1502.01852).
    """
    # TODO: implement He initialization.
    # See HINT block at the bottom of the file.
    raise NotImplementedError("Part A -- init_params")


# -----------------------------------------------------------------------------
# Part B -- the softmax with the numerical-stability trick
# -----------------------------------------------------------------------------

def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax over the last axis.

    For input z of shape (..., k), returns probabilities p of shape (..., k)
    such that p.sum(axis=-1) == 1.0 (within floating-point tolerance).

    The numerical-stability trick (Lecture 1, Section 5):
        p_i = exp(z_i - max(z))  /  sum_j exp(z_j - max(z))
    is identical to the naive softmax but cannot overflow.

    Parameters
    ----------
    z -- (..., k) float array of logits

    Returns
    -------
    p -- (..., k) float array of probabilities; each row sums to 1.0
    """
    # TODO: implement the stable softmax.
    # See HINT block at the bottom.
    raise NotImplementedError("Part B -- softmax")


# -----------------------------------------------------------------------------
# Part C -- the forward pass
# -----------------------------------------------------------------------------

def forward(
    X: NDArray[np.float64],
    params: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Forward pass for a 2-layer ReLU + softmax MLP.

    Parameters
    ----------
    X      -- (m, n_in) input batch
    params -- the dict returned by init_params

    Returns
    -------
    cache  -- a dict with keys X, Z1, A1, Z2, A2. The backward pass
              (Exercise 2) needs every entry.

    Forward pass (Lecture 1, Section 4):
        Z1 = X @ W1 + b1
        A1 = max(0, Z1)               (ReLU)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)              (probabilities)
    """
    # TODO: implement the forward pass.
    raise NotImplementedError("Part C -- forward")


# -----------------------------------------------------------------------------
# Part D -- cross-entropy loss
# -----------------------------------------------------------------------------

def cross_entropy_loss(
    probs: NDArray[np.float64],
    y: NDArray[np.int64],
) -> float:
    """Mean cross-entropy loss over a batch.

    Parameters
    ----------
    probs -- (m, k) array of predicted probabilities (rows sum to 1.0)
    y     -- (m,) array of integer class labels in [0, k)

    Returns
    -------
    loss  -- the scalar mean negative-log-likelihood

    Implementation notes:
    - Select the predicted probability of the true class per row with
      fancy indexing: probs[np.arange(m), y].
    - Clip the selected probabilities to [1e-12, 1.0] to avoid log(0) = -inf.
      The clip value 1e-12 is below float32 precision; the gradient is
      essentially unaffected.
    - Return the mean of -log over the clipped probabilities.
    """
    # TODO: implement cross-entropy loss.
    raise NotImplementedError("Part D -- cross_entropy_loss")


# -----------------------------------------------------------------------------
# Part E -- predict
# -----------------------------------------------------------------------------

def predict(
    X: NDArray[np.float64],
    params: dict[str, NDArray[np.float64]],
) -> NDArray[np.int64]:
    """Return the argmax class for each row of X.

    Parameters
    ----------
    X      -- (m, n_in) input batch
    params -- the trained parameter dict

    Returns
    -------
    labels -- (m,) array of int64 class indices
    """
    # TODO: implement predict.
    # Hint: call forward and take the argmax of the final-layer activations.
    raise NotImplementedError("Part E -- predict")


# -----------------------------------------------------------------------------
# Helpers used by the tests
# -----------------------------------------------------------------------------

def make_synthetic_mnist(
    n_samples: int = 128,
    n_features: int = 784,
    n_classes: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a synthetic MNIST-like batch for testing.

    The data is uniform in [0, 1] and the labels are uniform over n_classes.
    This is not a learnable task -- it is a smoke test for shapes and
    numerical stability.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features)).astype(np.float64)
    y = rng.integers(0, n_classes, size=(n_samples,)).astype(np.int64)
    return X, y


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_init_params_shapes() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    p = init_params(n_in=784, n_hidden=64, n_out=10, rng=rng)
    assert p["W1"].shape == (784, 64), f"W1 shape was {p['W1'].shape}"
    assert p["b1"].shape == (64,), f"b1 shape was {p['b1'].shape}"
    assert p["W2"].shape == (64, 10), f"W2 shape was {p['W2'].shape}"
    assert p["b2"].shape == (10,), f"b2 shape was {p['b2'].shape}"


def test_init_params_biases_are_zero() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    p = init_params(n_in=784, n_hidden=64, n_out=10, rng=rng)
    assert np.all(p["b1"] == 0.0)
    assert np.all(p["b2"] == 0.0)


def test_init_params_He_scale() -> None:
    """The std of W1 should be approximately sqrt(2 / n_in)."""
    rng = np.random.default_rng(RANDOM_STATE)
    p = init_params(n_in=784, n_hidden=64, n_out=10, rng=rng)
    expected_std_W1 = np.sqrt(2.0 / 784.0)
    actual_std_W1 = float(p["W1"].std())
    # Allow 15% relative tolerance because we are sampling 50,176 values.
    assert abs(actual_std_W1 - expected_std_W1) / expected_std_W1 < 0.15, (
        f"W1 std was {actual_std_W1:.4f}, expected ~{expected_std_W1:.4f}"
    )


def test_softmax_sums_to_one() -> None:
    z = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    p = softmax(z)
    assert p.shape == (3, 3)
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_softmax_is_stable_on_large_inputs() -> None:
    """A naive softmax would overflow here; the stable one must not."""
    z = np.array([[1000.0, 1000.1, 999.9], [-1000.0, -999.9, -1000.1]])
    p = softmax(z)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_softmax_uniform_logits_give_uniform_probs() -> None:
    z = np.zeros((1, 7))
    p = softmax(z)
    assert np.allclose(p, 1.0 / 7.0)


def test_forward_output_shapes() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_params(n_in=784, n_hidden=64, n_out=10, rng=rng)
    X, _ = make_synthetic_mnist(n_samples=32)
    cache = forward(X, params)
    assert cache["Z1"].shape == (32, 64)
    assert cache["A1"].shape == (32, 64)
    assert cache["Z2"].shape == (32, 10)
    assert cache["A2"].shape == (32, 10)


def test_forward_activations_nonnegative() -> None:
    """ReLU activations must be >= 0."""
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_params(n_in=784, n_hidden=64, n_out=10, rng=rng)
    X, _ = make_synthetic_mnist(n_samples=32)
    cache = forward(X, params)
    assert (cache["A1"] >= 0.0).all()


def test_forward_softmax_rows_sum_to_one() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_params(n_in=784, n_hidden=64, n_out=10, rng=rng)
    X, _ = make_synthetic_mnist(n_samples=32)
    cache = forward(X, params)
    assert np.allclose(cache["A2"].sum(axis=-1), 1.0)


def test_cross_entropy_loss_uniform_probs() -> None:
    """For a uniform softmax over k classes, CE loss should equal log(k)."""
    m, k = 100, 10
    probs = np.full((m, k), 1.0 / k)
    y = np.zeros(m, dtype=np.int64)
    loss = cross_entropy_loss(probs, y)
    assert abs(loss - np.log(k)) < 1e-9, (
        f"loss was {loss:.6f}, expected {np.log(k):.6f}"
    )


def test_cross_entropy_loss_perfect_predictions() -> None:
    """If the model assigns probability 1 to the correct class, loss = 0."""
    m, k = 5, 3
    y = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    probs = np.zeros((m, k))
    probs[np.arange(m), y] = 1.0
    loss = cross_entropy_loss(probs, y)
    # With clipping at 1e-12, log(1.0) = 0; loss should be essentially 0.
    assert loss < 1e-6, f"loss was {loss}, expected ~0"


def test_initial_loss_is_log_k() -> None:
    """A randomly-initialized 2-layer MLP should produce initial loss ~log(k).

    This is the most important sanity check in the file. Lecture 1 Section 8.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_params(n_in=784, n_hidden=128, n_out=10, rng=rng)
    X, y = make_synthetic_mnist(n_samples=256, n_features=784, n_classes=10, rng=rng)
    cache = forward(X, params)
    loss = cross_entropy_loss(cache["A2"], y)
    expected = float(np.log(10))
    # On random init the loss is close to log(10) but not exactly equal.
    # Allow +/- 0.5 (the empirical range is ~[2.0, 2.6] for He init).
    assert abs(loss - expected) < 0.5, (
        f"initial loss was {loss:.4f}, expected close to log(10) = {expected:.4f}"
    )


def test_predict_returns_int_labels_in_range() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    params = init_params(n_in=784, n_hidden=64, n_out=10, rng=rng)
    X, _ = make_synthetic_mnist(n_samples=32)
    labels = predict(X, params)
    assert labels.shape == (32,)
    assert labels.dtype.kind == "i", f"dtype was {labels.dtype}"
    assert (labels >= 0).all() and (labels < 10).all()


def _run_all_tests() -> None:
    test_init_params_shapes()
    test_init_params_biases_are_zero()
    test_init_params_He_scale()
    test_softmax_sums_to_one()
    test_softmax_is_stable_on_large_inputs()
    test_softmax_uniform_logits_give_uniform_probs()
    test_forward_output_shapes()
    test_forward_activations_nonnegative()
    test_forward_softmax_rows_sum_to_one()
    test_cross_entropy_loss_uniform_probs()
    test_cross_entropy_loss_perfect_predictions()
    test_initial_loss_is_log_k()
    test_predict_returns_int_labels_in_range()
    print("OK -- exercise 1")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# init_params:
#     return {
#         "W1": rng.normal(0.0, np.sqrt(2.0 / n_in),     size=(n_in, n_hidden)),
#         "b1": np.zeros((n_hidden,), dtype=np.float64),
#         "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out)),
#         "b2": np.zeros((n_out,), dtype=np.float64),
#     }
#
# softmax:
#     z_shifted = z - z.max(axis=-1, keepdims=True)
#     exp_z = np.exp(z_shifted)
#     return exp_z / exp_z.sum(axis=-1, keepdims=True)
#
# forward:
#     Z1 = X @ params["W1"] + params["b1"]
#     A1 = np.maximum(Z1, 0.0)
#     Z2 = A1 @ params["W2"] + params["b2"]
#     A2 = softmax(Z2)
#     return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
#
# cross_entropy_loss:
#     m = probs.shape[0]
#     safe = np.clip(probs[np.arange(m), y], 1e-12, 1.0)
#     return float(-np.log(safe).mean())
#
# predict:
#     cache = forward(X, params)
#     return cache["A2"].argmax(axis=-1).astype(np.int64)
#
# -----------------------------------------------------------------------------
