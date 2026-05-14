"""
Exercise 2 -- nn.Module and the Bare Training Loop.

Goal: build a 2-layer MLP as a torch.nn.Module subclass and train it with the
eight-line training loop from Lecture 2 Section 7. Verify that the loss
decreases on a learnable synthetic classification task.

By the end of this exercise you will have:

    (1) A TwoLayerMLP class subclassing nn.Module with __init__ + forward.
    (2) A make_learnable_dataset helper that produces a synthetic dataset
        with a linear-then-ReLU teacher signal (so the MLP CAN learn).
    (3) The bare training loop: zero_grad, forward, loss, backward, step.
    (4) A per-epoch evaluate function that computes test loss and accuracy
        in a `with torch.no_grad():` block.
    (5) A test that trains the MLP for 20 epochs and asserts the test
        accuracy is >= 0.75 (the teacher signal is strong; this is a
        soft-but-real learnability check).

Estimated time: 90-120 minutes.

Run with:    python exercise-02-nn-module-and-training-loop.py
Or test:     pytest exercise-02-nn-module-and-training-loop.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-nn-module-and-training-loop.py succeeds.

The exercise uses synthetic data so there is no dataset download.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    https://pytorch.org/docs/stable/optim.html
    https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Part A -- the TwoLayerMLP class
# -----------------------------------------------------------------------------

class TwoLayerMLP(nn.Module):
    """A 2-layer fully connected MLP: Linear -> ReLU -> Linear.

    The hidden layer uses ReLU activation. The output layer returns raw
    logits (no softmax); pair it with nn.CrossEntropyLoss in training.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    Parameters
    ----------
    n_in     -- input feature count
    n_hidden -- hidden layer width
    n_out    -- number of output classes
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        # TODO: register self.fc1 (nn.Linear(n_in, n_hidden)),
        #               self.relu (nn.ReLU()),
        #               self.fc2 (nn.Linear(n_hidden, n_out)).
        # See Lecture 2 Section 1.
        raise NotImplementedError("Part A -- TwoLayerMLP.__init__")

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # TODO: return self.fc2(self.relu(self.fc1(x)))
        raise NotImplementedError("Part A -- TwoLayerMLP.forward")


# -----------------------------------------------------------------------------
# Part B -- a learnable synthetic dataset
# -----------------------------------------------------------------------------

def make_learnable_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    n_hidden_teacher: int = 16,
    n_classes: int = 4,
    seed: int = RANDOM_STATE,
) -> "Tuple[torch.Tensor, torch.Tensor]":
    """Generate (X, y) from a fixed teacher MLP so that an MLP student CAN
    learn the mapping.

    The teacher is itself a 2-layer ReLU MLP with random weights; its
    argmax is taken as the integer label. With sufficient student capacity
    the student should reach >=75% accuracy in 20 epochs.

    Returns
    -------
    X : float32 tensor of shape (n_samples, n_features)
    y : int64   tensor of shape (n_samples,)   with values in [0, n_classes)
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, n_features, generator=g)

    # Teacher MLP: random weights, no training, take argmax.
    W1_t = torch.randn(n_features, n_hidden_teacher, generator=g) * 0.5
    b1_t = torch.randn(n_hidden_teacher, generator=g) * 0.1
    W2_t = torch.randn(n_hidden_teacher, n_classes, generator=g) * 0.5
    b2_t = torch.randn(n_classes, generator=g) * 0.1
    with torch.no_grad():
        h = torch.relu(X @ W1_t + b1_t)
        logits = h @ W2_t + b2_t
        y = logits.argmax(dim=1).long()
    return X, y


# -----------------------------------------------------------------------------
# Part C -- the bare training step
# -----------------------------------------------------------------------------

def training_step(
    model: nn.Module,
    X: "torch.Tensor",
    y: "torch.Tensor",
    loss_fn: nn.Module,
    optimizer: "torch.optim.Optimizer",
) -> float:
    """Run one training step:

        1. optimizer.zero_grad()
        2. logits = model(X)
        3. loss = loss_fn(logits, y)
        4. loss.backward()
        5. optimizer.step()
        6. return loss.item()

    The function returns the loss as a Python float (NOT a tensor; do not
    accumulate tensors across iterations -- it keeps the autograd graph
    alive and OOMs over time; see Lecture 2 Section 8 and Lecture 3
    Section 8 Failure 6).
    """
    # TODO: implement
    raise NotImplementedError("Part C -- training_step")


# -----------------------------------------------------------------------------
# Part D -- evaluate function
# -----------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: "torch.device",
) -> "Tuple[float, float]":
    """Compute per-example mean loss and classification accuracy over `loader`.

    Must:
        1. Call model.eval().
        2. Wrap the entire loop in `with torch.no_grad():` (Lecture 1
           Section 7; halves memory and ~doubles throughput).
        3. Move each (X, y) batch to `device`.
        4. Return (mean_loss, accuracy) as Python floats.

    Reference: https://pytorch.org/docs/stable/generated/torch.no_grad.html
    """
    # TODO: implement
    raise NotImplementedError("Part D -- evaluate")


# -----------------------------------------------------------------------------
# Part E -- the full train loop wrapper
# -----------------------------------------------------------------------------

def train(
    n_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 4,
    hidden: int = 32,
    seed: int = RANDOM_STATE,
) -> "Tuple[nn.Module, float]":
    """End-to-end training:

        1. Set torch.manual_seed(seed).
        2. Build the learnable dataset; split 80/20 into train/test
           (use a simple slice; this is synthetic data, no need for
           sklearn).
        3. Wrap each split in a TensorDataset; wrap each TensorDataset
           in a DataLoader (shuffle=True for train, False for test).
        4. Instantiate the model (TwoLayerMLP), the loss (nn.CrossEntropyLoss),
           and the optimizer (torch.optim.SGD with the given lr).
        5. Loop n_epochs times; in each epoch, iterate train_loader,
           call training_step, accumulate the loss; at the end of the
           epoch call evaluate(model, test_loader, ...).
        6. Return (model, final_test_accuracy).

    Use the device returned by best_device() below.
    """
    # TODO: implement; see Lecture 2 Section 10 for the reference template.
    raise NotImplementedError("Part E -- train")


# -----------------------------------------------------------------------------
# Helper -- best available device
# -----------------------------------------------------------------------------

def best_device() -> "torch.device":
    """Return cuda > mps > cpu. See Lecture 3 Section 5."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_two_layer_mlp_has_expected_parameters() -> None:
    m = TwoLayerMLP(20, 32, 4)
    names = [n for n, _ in m.named_parameters()]
    # Standard naming for the layers we set in __init__:
    expected = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert set(names) == expected, f"got {set(names)}"


def test_two_layer_mlp_forward_shape() -> None:
    m = TwoLayerMLP(20, 32, 4)
    X = torch.randn(8, 20)
    out = m(X)
    assert out.shape == (8, 4)


def test_two_layer_mlp_weight_shapes_match_nn_linear_convention() -> None:
    """nn.Linear stores weight as (out_features, in_features) -- see
    Lecture 2 Section 2.
    """
    m = TwoLayerMLP(20, 32, 4)
    assert m.fc1.weight.shape == (32, 20)
    assert m.fc2.weight.shape == (4, 32)


def test_make_learnable_dataset_shapes_and_dtype() -> None:
    X, y = make_learnable_dataset(n_samples=200, n_features=10, n_classes=3, seed=0)
    assert X.shape == (200, 10)
    assert y.shape == (200,)
    assert X.dtype == torch.float32
    assert y.dtype == torch.int64
    assert (y >= 0).all() and (y < 3).all()


def test_training_step_decreases_loss_on_average() -> None:
    """One training step should reduce the loss on average across multiple
    batches with a non-trivial learning rate."""
    torch.manual_seed(RANDOM_STATE)
    X, y = make_learnable_dataset(n_samples=512, n_features=20, n_classes=4)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = TwoLayerMLP(20, 32, 4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    initial = loss_fn(model(X), y).item()
    for X_b, y_b in loader:
        training_step(model, X_b, y_b, loss_fn, optimizer)
    with torch.no_grad():
        after = loss_fn(model(X), y).item()
    # One full pass through 512 samples must reduce the loss.
    assert after < initial, f"initial {initial:.4f}, after {after:.4f}"


def test_evaluate_returns_floats_in_range() -> None:
    torch.manual_seed(RANDOM_STATE)
    X, y = make_learnable_dataset(n_samples=200, n_features=10, n_classes=3)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model = TwoLayerMLP(10, 16, 3)
    loss_fn = nn.CrossEntropyLoss()
    device = best_device()
    model.to(device)
    loss, acc = evaluate(model, loader, loss_fn, device)
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_train_reaches_minimum_accuracy() -> None:
    """The headline test: a 20-epoch train on the learnable dataset should
    reach test accuracy >= 0.75 on the held-out 20%. The teacher signal is
    strong enough that this is consistently achievable.
    """
    model, test_acc = train(n_epochs=20, batch_size=64, lr=0.1, seed=RANDOM_STATE)
    assert test_acc >= 0.75, (
        f"test accuracy {test_acc:.4f} below 0.75; check the training loop"
    )


def test_train_returns_an_nn_module() -> None:
    model, _ = train(n_epochs=2, batch_size=64, seed=RANDOM_STATE)
    assert isinstance(model, nn.Module)


def _run_all_tests() -> None:
    test_two_layer_mlp_has_expected_parameters()
    test_two_layer_mlp_forward_shape()
    test_two_layer_mlp_weight_shapes_match_nn_linear_convention()
    test_make_learnable_dataset_shapes_and_dtype()
    test_training_step_decreases_loss_on_average()
    test_evaluate_returns_floats_in_range()
    test_train_reaches_minimum_accuracy()
    test_train_returns_an_nn_module()
    print("OK -- exercise 2")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# TwoLayerMLP.__init__:
#     super().__init__()
#     self.fc1 = nn.Linear(n_in, n_hidden)
#     self.relu = nn.ReLU()
#     self.fc2 = nn.Linear(n_hidden, n_out)
#
# TwoLayerMLP.forward:
#     return self.fc2(self.relu(self.fc1(x)))
#
# training_step:
#     optimizer.zero_grad()
#     logits = model(X)
#     loss = loss_fn(logits, y)
#     loss.backward()
#     optimizer.step()
#     return float(loss.item())
#
# evaluate:
#     model.eval()
#     total_loss, total_correct, n = 0.0, 0, 0
#     with torch.no_grad():
#         for X, y in loader:
#             X, y = X.to(device), y.to(device)
#             logits = model(X)
#             total_loss += loss_fn(logits, y).item() * X.size(0)
#             total_correct += (logits.argmax(1) == y).sum().item()
#             n += X.size(0)
#     return total_loss / n, total_correct / n
#
# train:
#     torch.manual_seed(seed)
#     device = best_device()
#     X, y = make_learnable_dataset(n_samples, n_features, n_classes=n_classes, seed=seed)
#     n_train = int(0.8 * len(X))
#     X_tr, y_tr = X[:n_train], y[:n_train]
#     X_te, y_te = X[n_train:], y[n_train:]
#     tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
#     te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
#     model = TwoLayerMLP(n_features, hidden, n_classes).to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     for _ in range(n_epochs):
#         model.train()
#         for X_b, y_b in tr_loader:
#             X_b, y_b = X_b.to(device), y_b.to(device)
#             training_step(model, X_b, y_b, loss_fn, optimizer)
#     _, acc = evaluate(model, te_loader, loss_fn, device)
#     return model, acc
#
# -----------------------------------------------------------------------------
