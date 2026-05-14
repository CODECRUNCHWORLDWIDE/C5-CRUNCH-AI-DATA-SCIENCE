"""
Mini-project starter -- a 2-layer NumPy MLP for tabular classification.

Copy this file into your portfolio repo at week-07/mlp.py and fill in
the data loader. The forward / backward / SGD code is identical to the
exercises and is provided so you do not have to re-derive it.

The notebook (mlp_tabular.ipynb in your portfolio) will:
    1. import this module
    2. load the dataset (one of: Wisconsin Breast Cancer, Adult, Iris, or your own)
    3. train both LR and the MLP
    4. plot training curves and confusion matrix
    5. write the report

This script is also runnable standalone for a quick smoke test on the
Wisconsin Breast Cancer dataset:

    python starter.py

Acceptance:
    - python3 -m py_compile starter.py succeeds.
    - The smoke test prints final test accuracy >= 0.94 on Breast Cancer.
    - The gradient check at the start prints max relative error < 1e-5
      for every parameter.

Reading: README.md (this folder), Lectures 1-3, Exercises 1-3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Primitives (identical to Exercise 2 / Exercise 3)
# -----------------------------------------------------------------------------

def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise ReLU: max(0, z)."""
    return np.maximum(z, 0.0)


def relu_grad(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivative of ReLU. 1 where z > 0, else 0."""
    return (z > 0).astype(np.float64)


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax over the last axis."""
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def cross_entropy_loss(
    probs: NDArray[np.float64], y: NDArray[np.int64]
) -> float:
    """Mean cross-entropy loss; clipping at 1e-12 avoids log(0)."""
    m = probs.shape[0]
    safe = np.clip(probs[np.arange(m), y], 1e-12, 1.0)
    return float(-np.log(safe).mean())


def init_params(
    n_in: int, n_hidden: int, n_out: int, rng: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    """He initialization for weights; zero initialization for biases."""
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / n_in),     size=(n_in, n_hidden)),
        "b1": np.zeros((n_hidden,), dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out)),
        "b2": np.zeros((n_out,), dtype=np.float64),
    }


def forward(
    X: NDArray[np.float64], params: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """Forward pass for a 2-layer ReLU + softmax MLP."""
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
    """Backward pass: returns gradients of L w.r.t. W1, b1, W2, b2."""
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


def sgd_step(
    params: dict[str, NDArray[np.float64]],
    X_batch: NDArray[np.float64],
    y_batch: NDArray[np.int64],
    learning_rate: float,
    weight_decay: float = 0.0,
) -> float:
    """One mini-batch SGD step with optional L2 weight decay.

    The weight-decay term updates per parameter as:
        params[k] -= learning_rate * (grads[k] + weight_decay * params[k])

    For weight_decay = 0 this reduces to vanilla SGD.
    """
    cache = forward(X_batch, params)
    loss = cross_entropy_loss(cache["A2"], y_batch)
    grads = backward(cache, params, y_batch)
    for k in params:
        params[k] -= learning_rate * (grads[k] + weight_decay * params[k])
    return loss


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def train(
    X_train: NDArray[np.float64], y_train: NDArray[np.int64],
    X_test: NDArray[np.float64], y_test: NDArray[np.int64],
    *,
    n_hidden: int = 64,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    n_epochs: int = 30,
    weight_decay: float = 0.0,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, list[float]]]:
    """Train a 2-layer MLP with mini-batch SGD.

    Returns:
        params  -- the trained parameter dict
        history -- {"train_loss": list, "test_acc": list, "train_acc": list}
    """
    rng = np.random.default_rng(random_state)
    n_train, n_in = X_train.shape
    n_out = int(y_train.max()) + 1
    params = init_params(n_in, n_hidden, n_out, rng)
    history: dict[str, list[float]] = {
        "train_loss": [], "test_acc": [], "train_acc": [],
    }

    for epoch in range(n_epochs):
        perm = rng.permutation(n_train)
        epoch_losses: list[float] = []
        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            loss = sgd_step(
                params, X_train[batch_idx], y_train[batch_idx],
                learning_rate, weight_decay,
            )
            epoch_losses.append(loss)
        train_loss = float(np.mean(epoch_losses))
        train_acc = float((predict(X_train, params) == y_train).mean())
        test_acc = float((predict(X_test, params) == y_test).mean())
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        if verbose:
            print(
                f"epoch {epoch + 1:3d}  "
                f"train_loss={train_loss:.4f}  "
                f"train_acc={train_acc:.4f}  "
                f"test_acc={test_acc:.4f}"
            )

    return params, history


# -----------------------------------------------------------------------------
# Gradient check
# -----------------------------------------------------------------------------

def numerical_gradient(
    params: dict[str, NDArray[np.float64]],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    param_name: str,
    idx: tuple[int, ...],
    eps: float = 1e-7,
) -> float:
    """Centered finite-difference gradient of the loss with respect to
    a single scalar element of params[param_name] at idx.
    """
    p = params[param_name]
    original = float(p[idx])
    p[idx] = original + eps
    loss_plus = cross_entropy_loss(forward(X, params)["A2"], y)
    p[idx] = original - eps
    loss_minus = cross_entropy_loss(forward(X, params)["A2"], y)
    p[idx] = original
    return (loss_plus - loss_minus) / (2.0 * eps)


def gradient_check(
    params: dict[str, NDArray[np.float64]],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_samples: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, float]:
    """Compare analytical to numerical gradients at n_samples random
    elements of each parameter. Return the max relative error per param.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    cache = forward(X, params)
    grads = backward(cache, params, y)
    max_errs: dict[str, float] = {}
    for k in params:
        p = params[k]
        g_a = grads[k]
        flat_idx = rng.choice(p.size, size=min(n_samples, p.size), replace=False)
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
# Smoke test: Wisconsin Breast Cancer
# -----------------------------------------------------------------------------

def _smoke_test() -> None:
    """Quick end-to-end demo on Wisconsin Breast Cancer (Option A).

    Replace the data-loading block with your own dataset for the
    portfolio mini-project. Acceptance: final test_acc >= 0.94 and
    gradient-check max relative error < 1e-5 for every parameter.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Logistic-regression baseline.
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_std, y_train)
    lr_acc = float(lr.score(X_test_std, y_test))
    print(f"LR baseline test_acc: {lr_acc:.4f}")

    # Gradient check on a 5-example batch.
    rng = np.random.default_rng(RANDOM_STATE)
    tiny_params = init_params(
        n_in=X_train_std.shape[1], n_hidden=8, n_out=2, rng=rng,
    )
    tiny_X = X_train_std[:5]
    tiny_y = y_train[:5]
    errs = gradient_check(tiny_params, tiny_X, tiny_y, n_samples=8, rng=rng)
    for k, err in errs.items():
        print(f"  grad_check {k}: max rel err = {err:.3e}")
        assert err < 1e-5, f"gradient check FAILED for {k}: {err}"

    # Train the MLP.
    params, history = train(
        X_train_std, y_train, X_test_std, y_test,
        n_hidden=32, learning_rate=0.05, batch_size=32,
        n_epochs=50, random_state=RANDOM_STATE, verbose=False,
    )
    mlp_acc = history["test_acc"][-1]
    print(f"MLP test_acc: {mlp_acc:.4f}")
    print(f"  delta vs LR: {(mlp_acc - lr_acc) * 100:+.2f} percentage points")

    assert mlp_acc >= 0.94, (
        f"MLP test_acc was {mlp_acc:.4f}; expected >= 0.94 on Breast Cancer."
    )
    print("OK -- mini-project starter smoke test")


if __name__ == "__main__":
    _smoke_test()
