"""
Exercise 3 -- Train a 2-Layer MLP on an MNIST Subset.

Goal: put forward + backward + SGD together in a training loop and reach
>= 90% test accuracy on a 10,000-example MNIST subset in 10 epochs. The
full-MNIST run (60k train, 10k test, >= 95% target) is the mini-project;
this exercise is the smaller-scale dry run.

By the end of this exercise you will have:

    (1) An MNIST loader that downloads via sklearn.datasets.fetch_openml,
        normalizes pixel values to [0, 1], and subsamples to 10,000
        training + 2,000 test examples (or uses the full set if the
        FULL_MNIST flag is set).
    (2) A mini-batch SGD training loop (Lecture 3, Section 1).
    (3) End-of-epoch test accuracy logging.
    (4) A sanity check that the initial loss is log(10) and the final
        test accuracy is >= 0.90.

Estimated time: 90-120 minutes (mostly the first run + debugging).

Run with:    python exercise-03-train-on-mnist-subset.py
Or test:     pytest exercise-03-train-on-mnist-subset.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions that DO NOT need MNIST all pass on every run.
- The end-to-end run reaches >= 0.90 test accuracy in 10 epochs (this is
  not enforced as a pytest, because the MNIST download requires network
  access; the integration test is marked optional below).
- python -m py_compile exercise-03-train-on-mnist-subset.py succeeds.

Note: the integration test calls into the network only if the MNIST
download has succeeded; on a fresh CI box without network, the slow
test is skipped automatically.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray


RANDOM_STATE = 42

# Flip this to False for the small smoke-test path (10k train / 2k test).
# Keep True for the full MNIST run (60k train / 10k test) -- which is the
# mini-project default. The exercise targets are tuned for FULL_MNIST=False.
FULL_MNIST = False


# -----------------------------------------------------------------------------
# Part A -- primitives (copied from Exercise 2)
# -----------------------------------------------------------------------------

def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(z, 0.0)


def relu_grad(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return (z > 0).astype(np.float64)


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def cross_entropy_loss(
    probs: NDArray[np.float64], y: NDArray[np.int64]
) -> float:
    m = probs.shape[0]
    safe = np.clip(probs[np.arange(m), y], 1e-12, 1.0)
    return float(-np.log(safe).mean())


def init_params(
    n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_hidden)),
        "b1": np.zeros((n_hidden,), dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out)),
        "b2": np.zeros((n_out,), dtype=np.float64),
    }


def forward(
    X: NDArray[np.float64], params: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = softmax(Z2)
    return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def backward(
    cache: dict[str, NDArray[np.float64]],
    params: dict[str, NDArray[np.float64]],
    y: NDArray[np.int64],
) -> dict[str, NDArray[np.float64]]:
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]
    W2 = params["W2"]
    m, k = A2.shape
    Y = np.zeros((m, k), dtype=np.float64)
    Y[np.arange(m), y] = 1.0
    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def predict(
    X: NDArray[np.float64], params: dict[str, NDArray[np.float64]],
) -> NDArray[np.int64]:
    return forward(X, params)["A2"].argmax(axis=-1).astype(np.int64)


# -----------------------------------------------------------------------------
# Part B -- the SGD step
# -----------------------------------------------------------------------------

def sgd_step(
    params: dict[str, NDArray[np.float64]],
    X_batch: NDArray[np.float64],
    y_batch: NDArray[np.int64],
    learning_rate: float,
) -> float:
    """Run forward, backward, and update params in place. Return the loss.

    The update rule (Lecture 3, Section 3): params -= learning_rate * grads.
    """
    # TODO: implement one SGD step.
    # See HINT block at the bottom of the file.
    raise NotImplementedError("Part B -- sgd_step")


# -----------------------------------------------------------------------------
# Part C -- the training loop
# -----------------------------------------------------------------------------

def train(
    X_train: NDArray[np.float64], y_train: NDArray[np.int64],
    X_test: NDArray[np.float64], y_test: NDArray[np.int64],
    *,
    n_hidden: int = 128,
    learning_rate: float = 0.1,
    batch_size: int = 128,
    n_epochs: int = 10,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, list[float]]]:
    """Train the 2-layer MLP with mini-batch SGD.

    See Lecture 3 Section 1 for the loop structure.

    Returns
    -------
    params  -- the trained parameters
    history -- dict with keys "train_loss" and "test_acc", lists of length n_epochs
    """
    rng = np.random.default_rng(random_state)
    n_train, n_in = X_train.shape
    n_out = int(y_train.max()) + 1

    params = init_params(n_in, n_hidden, n_out, rng)
    history: dict[str, list[float]] = {"train_loss": [], "test_acc": []}

    # TODO: implement the training loop.
    # The structure is:
    #   for each epoch:
    #     shuffle the training indices
    #     for each batch:
    #         sgd_step(params, X_batch, y_batch, learning_rate)
    #         collect the loss
    #     compute the mean train loss for the epoch
    #     compute the test accuracy
    #     append both to history
    #     if verbose, print "epoch <i>  train_loss=<l>  test_acc=<a>"
    # See HINT block at the bottom.
    raise NotImplementedError("Part C -- train")


# -----------------------------------------------------------------------------
# Part D -- MNIST loader
# -----------------------------------------------------------------------------

def load_mnist_subset(
    n_train: int = 10_000,
    n_test: int = 2_000,
    full: bool = False,
) -> tuple[
    NDArray[np.float64], NDArray[np.int64],
    NDArray[np.float64], NDArray[np.int64],
]:
    """Download MNIST via sklearn.datasets.fetch_openml, normalize, and
    optionally subsample.

    The full MNIST has 70,000 images: 60k train + 10k test (LeCun 1998).
    For the exercise smoke-test path we subsample to 10k + 2k.

    Returns
    -------
    X_train, y_train, X_test, y_test  -- the subset.

    Pixel values are normalized to [0, 1] by dividing by 255.

    This function requires network access on first call to cache MNIST.
    Subsequent calls hit the local cache at ~/scikit_learn_data/openml/.
    """
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        cache=True,
    )
    X = np.asarray(mnist.data, dtype=np.float64) / 255.0
    y = np.asarray(mnist.target, dtype=np.int64)

    # Canonical split: first 60k are train, last 10k are test.
    X_full_train, X_full_test = X[:60_000], X[60_000:]
    y_full_train, y_full_test = y[:60_000], y[60_000:]

    if full:
        return X_full_train, y_full_train, X_full_test, y_full_test

    rng = np.random.default_rng(RANDOM_STATE)
    train_idx = rng.choice(len(X_full_train), size=n_train, replace=False)
    test_idx = rng.choice(len(X_full_test), size=n_test, replace=False)
    return (
        X_full_train[train_idx], y_full_train[train_idx],
        X_full_test[test_idx], y_full_test[test_idx],
    )


# -----------------------------------------------------------------------------
# Helpers used by the tests
# -----------------------------------------------------------------------------

def make_synthetic_classification(
    n_samples: int = 200,
    n_features: int = 20,
    n_classes: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """A learnable synthetic dataset: classes are linearly-separable
    Gaussian blobs in feature space.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    centers = rng.normal(0.0, 3.0, size=(n_classes, n_features))
    X = np.empty((n_samples, n_features), dtype=np.float64)
    y = np.empty(n_samples, dtype=np.int64)
    per_class = n_samples // n_classes
    idx = 0
    for c in range(n_classes):
        n_c = per_class if c < n_classes - 1 else n_samples - idx
        X[idx : idx + n_c] = centers[c] + rng.normal(0.0, 1.0, size=(n_c, n_features))
        y[idx : idx + n_c] = c
        idx += n_c
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def _has_mnist_cached() -> bool:
    """Return True if MNIST is cached locally (so the slow test will run)."""
    candidates = [
        os.path.expanduser("~/scikit_learn_data/openml/openml.org/data/v1/download/52667.gz"),
        os.path.expanduser("~/scikit_learn_data/openml/openml.org"),
    ]
    return any(os.path.exists(p) for p in candidates)


# -----------------------------------------------------------------------------
# Pytest-style checks (no MNIST required)
# -----------------------------------------------------------------------------

def test_sgd_step_decreases_loss_on_easy_data() -> None:
    """One SGD step on synthetic data should decrease the loss."""
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_synthetic_classification(n_samples=64, n_features=20, n_classes=3, rng=rng)
    params = init_params(n_in=20, n_hidden=16, n_out=3, rng=rng)
    loss_before = cross_entropy_loss(forward(X, params)["A2"], y)
    sgd_step(params, X, y, learning_rate=0.01)
    loss_after = cross_entropy_loss(forward(X, params)["A2"], y)
    assert loss_after < loss_before


def test_train_converges_on_easy_synthetic_data() -> None:
    """A 2-layer MLP should reach >= 95% accuracy on linearly-separable
    Gaussian blobs in under 10 epochs."""
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_synthetic_classification(
        n_samples=600, n_features=20, n_classes=3, rng=rng,
    )
    X_train, y_train = X[:500], y[:500]
    X_test, y_test = X[500:], y[500:]
    params, history = train(
        X_train, y_train, X_test, y_test,
        n_hidden=32, learning_rate=0.05, batch_size=32,
        n_epochs=15, random_state=RANDOM_STATE, verbose=False,
    )
    final_acc = history["test_acc"][-1]
    assert final_acc >= 0.95, (
        f"test accuracy on synthetic data was {final_acc:.4f}; "
        f"expected >= 0.95. Check the SGD step and the loop."
    )


def test_train_returns_correct_history_keys() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_synthetic_classification(rng=rng)
    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]
    _, history = train(
        X_train, y_train, X_test, y_test,
        n_hidden=8, learning_rate=0.05, batch_size=32,
        n_epochs=3, verbose=False,
    )
    assert "train_loss" in history
    assert "test_acc" in history
    assert len(history["train_loss"]) == 3
    assert len(history["test_acc"]) == 3


def test_train_loss_monotone_nonincreasing_on_easy_data() -> None:
    """On easy data, the training loss should decrease (or at most stay
    flat) between epoch 1 and the last epoch. Hiccup in the middle is OK
    (mini-batch noise) but the overall trend must be down."""
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = make_synthetic_classification(n_samples=300, rng=rng)
    X_train, y_train = X[:250], y[:250]
    X_test, y_test = X[250:], y[250:]
    _, history = train(
        X_train, y_train, X_test, y_test,
        n_hidden=16, learning_rate=0.05, batch_size=32,
        n_epochs=8, verbose=False,
    )
    losses = history["train_loss"]
    # First-epoch loss should exceed last-epoch loss.
    assert losses[-1] < losses[0]


# -----------------------------------------------------------------------------
# Slow integration test -- only runs if MNIST is already cached
# -----------------------------------------------------------------------------

def test_mnist_subset_reaches_90_percent() -> None:
    """The full Exercise 3 acceptance criterion.

    On a 10,000-example MNIST training subset, the MLP should reach >= 90%
    test accuracy in 10 epochs. We mark this test as "skipped" if MNIST
    is not cached locally, because the test should not need network.
    """
    if not _has_mnist_cached():
        print("  ... skipped: MNIST not cached. Run load_mnist_subset() once.")
        return

    X_train, y_train, X_test, y_test = load_mnist_subset(
        n_train=10_000, n_test=2_000, full=False,
    )
    t0 = time.time()
    _, history = train(
        X_train, y_train, X_test, y_test,
        n_hidden=128, learning_rate=0.1, batch_size=128,
        n_epochs=10, random_state=RANDOM_STATE, verbose=False,
    )
    elapsed = time.time() - t0
    final_acc = history["test_acc"][-1]
    print(f"  ... MNIST subset: final test_acc = {final_acc:.4f}, took {elapsed:.1f}s")
    assert final_acc >= 0.90, (
        f"final test accuracy was {final_acc:.4f}; "
        f"expected >= 0.90 on the 10k MNIST subset."
    )


def _run_all_tests() -> None:
    test_sgd_step_decreases_loss_on_easy_data()
    test_train_returns_correct_history_keys()
    test_train_loss_monotone_nonincreasing_on_easy_data()
    test_train_converges_on_easy_synthetic_data()
    test_mnist_subset_reaches_90_percent()
    print("OK -- exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# sgd_step:
#     cache = forward(X_batch, params)
#     loss = cross_entropy_loss(cache["A2"], y_batch)
#     grads = backward(cache, params, y_batch)
#     for k in params:
#         params[k] -= learning_rate * grads[k]
#     return loss
#
# train:
#     for epoch in range(n_epochs):
#         perm = rng.permutation(n_train)
#         epoch_losses = []
#         for start in range(0, n_train, batch_size):
#             batch_idx = perm[start : start + batch_size]
#             X_batch = X_train[batch_idx]
#             y_batch = y_train[batch_idx]
#             loss = sgd_step(params, X_batch, y_batch, learning_rate)
#             epoch_losses.append(loss)
#         train_loss = float(np.mean(epoch_losses))
#         test_acc = float((predict(X_test, params) == y_test).mean())
#         history["train_loss"].append(train_loss)
#         history["test_acc"].append(test_acc)
#         if verbose:
#             print(f"epoch {epoch + 1:2d}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")
#     return params, history
#
# -----------------------------------------------------------------------------
