"""
Exercise 3 -- FashionMNIST End-to-End.

Goal: train a 2-layer MLP on FashionMNIST via torchvision, reach >=85% test
accuracy, save the state_dict to disk, reload it into a freshly-constructed
model, and verify the reloaded model produces identical predictions.

This exercise puts together Lectures 2 and 3 in a single runnable script.
It is the Week 8 capstone exercise.

By the end of this exercise you will have:

    (1) A build_loaders function that returns train and test DataLoaders for
        FashionMNIST via torchvision.datasets.FashionMNIST, with the standard
        FashionMNIST Normalize transform.
    (2) A FashionMLP nn.Module subclass with the (flatten -> 784 -> 128 ->
        ReLU -> 10) architecture.
    (3) A train function with the eight-line loop, per-epoch validation, and
        a per-epoch print statement.
    (4) A save_model / load_model pair that uses state_dict + the
        weights_only=True flag (PyTorch 2.4+).
    (5) A test that trains for 5 epochs and asserts test accuracy >= 0.85.
    (6) A test that confirms save -> load preserves predictions exactly.

Estimated time: 90-120 minutes (longer if FashionMNIST has to download).

Run with:    python exercise-03-fashionmnist-end-to-end.py
Or test:     pytest exercise-03-fashionmnist-end-to-end.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-03-fashionmnist-end-to-end.py succeeds.

NOTE: Running the tests actually downloads FashionMNIST (~30 MB into ./data/).
Re-runs are instant after the first download.

PyTorch references:
    https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
    https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html
    https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

# torchvision is imported lazily inside build_loaders so that py_compile works
# on machines where torchvision is not installed yet. The pytest checks DO
# import it; that is expected and intentional.


RANDOM_STATE = 42
DATA_ROOT = "./data"
CHECKPOINT_PATH = "./fashion_mlp.pt"

# Standard channel statistics for FashionMNIST; computed once on the train set.
FASHION_MNIST_MEAN: float = 0.2860
FASHION_MNIST_STD: float = 0.3530


# -----------------------------------------------------------------------------
# Part A -- best device
# -----------------------------------------------------------------------------

def best_device() -> "torch.device":
    """cuda > mps > cpu. See Lecture 3 Section 5.

    Reference: https://pytorch.org/docs/stable/notes/mps.html
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Part B -- the FashionMLP module
# -----------------------------------------------------------------------------

class FashionMLP(nn.Module):
    """A 2-layer MLP on flattened 28x28 FashionMNIST images.

    Architecture:
        Flatten        : (B, 1, 28, 28) -> (B, 784)
        Linear(784, h) : -> (B, h)
        ReLU           : (elementwise)
        Linear(h, 10)  : -> (B, 10) raw logits

    Returns raw logits; pair with nn.CrossEntropyLoss.
    """

    def __init__(self, n_hidden: int = 128, n_classes: int = 10) -> None:
        super().__init__()
        # TODO: build the nn.Sequential as described above. See Lecture 3
        # Section 3 for the canonical pattern.
        raise NotImplementedError("Part B -- FashionMLP.__init__")

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # TODO: return self.net(x)
        raise NotImplementedError("Part B -- FashionMLP.forward")


# -----------------------------------------------------------------------------
# Part C -- the data loaders
# -----------------------------------------------------------------------------

def build_loaders(
    batch_size: int = 128, root: str = DATA_ROOT, num_workers: int = 0,
) -> "Tuple[DataLoader, DataLoader]":
    """Construct FashionMNIST DataLoaders.

    Use torchvision.datasets.FashionMNIST with download=True. Apply a
    transforms.v2.Compose of [ToTensor, Normalize(mean, std)] where the
    mean and std are FASHION_MNIST_MEAN and FASHION_MNIST_STD.

    Train loader: shuffle=True. Test loader: shuffle=False.

    Returns (train_loader, test_loader).

    Reference: https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
    """
    from torchvision import datasets
    from torchvision.transforms import v2

    # TODO: build the transform pipeline.
    # HINT:
    #   transform = v2.Compose([
    #       v2.ToTensor(),
    #       v2.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
    #   ])
    raise NotImplementedError("Part C -- build_loaders")


# -----------------------------------------------------------------------------
# Part D -- evaluate
# -----------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: "torch.device",
) -> "Tuple[float, float]":
    """Compute (mean_loss, accuracy) over `loader` with model in eval mode.

    Identical contract to Exercise 2's evaluate; see Lecture 2 Section 8.

    Must use `with torch.no_grad():` and `model.eval()`.
    """
    # TODO: implement
    raise NotImplementedError("Part D -- evaluate")


# -----------------------------------------------------------------------------
# Part E -- the train function
# -----------------------------------------------------------------------------

def train(
    n_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = RANDOM_STATE,
    verbose: bool = True,
) -> "Tuple[nn.Module, float]":
    """Train the FashionMLP and return (model, final_test_accuracy).

    Steps:
        1. torch.manual_seed(seed).
        2. device = best_device(), train_loader, test_loader = build_loaders(batch_size).
        3. model = FashionMLP(n_hidden=128).to(device).
        4. loss_fn = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(
           model.parameters(), lr=lr).
        5. For each epoch:
           - model.train()
           - For each (X, y) in train_loader: zero_grad, forward, loss,
             backward, step (the eight-line loop from Lecture 2 Section 7).
           - Run evaluate(model, test_loader, ...) to get test_loss, test_acc.
           - If verbose, print one summary line per epoch.
        6. Return (model, final_test_acc).
    """
    # TODO: implement. See Lecture 3 Section 6 for the full reference script.
    raise NotImplementedError("Part E -- train")


# -----------------------------------------------------------------------------
# Part F -- save and load the state_dict
# -----------------------------------------------------------------------------

def save_model(model: nn.Module, path: str = CHECKPOINT_PATH) -> None:
    """Save model.state_dict() to `path` via torch.save.

    Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    # TODO: torch.save(model.state_dict(), path)
    raise NotImplementedError("Part F -- save_model")


def load_model(path: str = CHECKPOINT_PATH, n_hidden: int = 128) -> nn.Module:
    """Construct a fresh FashionMLP and load the state_dict from `path`.

    Use weights_only=True (PyTorch 2.4+ safe-loader; see Lecture 3 Section 7).
    Place the model on best_device() and call model.eval() before returning.
    """
    # TODO: implement
    # HINT:
    #   model = FashionMLP(n_hidden=n_hidden)
    #   state = torch.load(path, map_location="cpu", weights_only=True)
    #   model.load_state_dict(state)
    #   model.to(best_device()).eval()
    #   return model
    raise NotImplementedError("Part F -- load_model")


# -----------------------------------------------------------------------------
# Pytest-style checks (the FashionMNIST download is required for these)
# -----------------------------------------------------------------------------

def test_fashion_mlp_has_expected_parameter_count() -> None:
    m = FashionMLP(n_hidden=128)
    total = sum(p.numel() for p in m.parameters())
    # 784*128 + 128 + 128*10 + 10 = 101,770
    expected = 784 * 128 + 128 + 128 * 10 + 10
    assert total == expected, f"got {total}, expected {expected}"


def test_fashion_mlp_forward_shape_on_image_batch() -> None:
    m = FashionMLP(n_hidden=64)
    X = torch.randn(8, 1, 28, 28)
    out = m(X)
    assert out.shape == (8, 10)


def test_build_loaders_returns_two_dataloaders() -> None:
    train_loader, test_loader = build_loaders(batch_size=64)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert train_loader.batch_size == 64
    assert test_loader.batch_size == 64


def test_build_loaders_yields_normalized_image_batches() -> None:
    train_loader, _ = build_loaders(batch_size=8)
    X, y = next(iter(train_loader))
    assert X.shape == (8, 1, 28, 28), f"X.shape was {X.shape}"
    assert y.shape == (8,)
    assert X.dtype == torch.float32
    # After Normalize((0.286,), (0.353,)), values should typically be in
    # roughly [-1, 3]. We allow a generous range.
    assert X.min().item() > -3.0 and X.max().item() < 5.0


def test_train_reaches_85_percent_in_five_epochs() -> None:
    """The headline assertion: a 2-layer MLP must reach >=85% test accuracy
    on FashionMNIST in 5 epochs at lr=1e-3 (Adam). This is the standard
    benchmark for the Week 8 architecture."""
    _, test_acc = train(n_epochs=5, batch_size=128, lr=1e-3, seed=RANDOM_STATE,
                        verbose=False)
    assert test_acc >= 0.85, (
        f"final test accuracy {test_acc:.4f} below 0.85; check the training loop"
    )


def test_save_and_load_preserves_predictions() -> None:
    """Train briefly; save; load into a fresh model; verify predictions match."""
    # Train for just 1 epoch -- the test is correctness, not accuracy.
    model_a, _ = train(n_epochs=1, batch_size=128, lr=1e-3, seed=RANDOM_STATE,
                       verbose=False)
    save_model(model_a, CHECKPOINT_PATH)
    try:
        model_b = load_model(CHECKPOINT_PATH, n_hidden=128)
        # Same input, both on CPU for fair comparison.
        X = torch.randn(16, 1, 28, 28)
        with torch.no_grad():
            logits_a = model_a.cpu()(X)
            logits_b = model_b.cpu()(X)
        assert torch.allclose(logits_a, logits_b, atol=1e-6), (
            "save -> load did not preserve the parameters"
        )
    finally:
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)


def _run_all_tests() -> None:
    test_fashion_mlp_has_expected_parameter_count()
    test_fashion_mlp_forward_shape_on_image_batch()
    test_build_loaders_returns_two_dataloaders()
    test_build_loaders_yields_normalized_image_batches()
    test_train_reaches_85_percent_in_five_epochs()
    test_save_and_load_preserves_predictions()
    print("OK -- exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# FashionMLP.__init__:
#     super().__init__()
#     self.net = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(28 * 28, n_hidden),
#         nn.ReLU(),
#         nn.Linear(n_hidden, n_classes),
#     )
#
# FashionMLP.forward:
#     return self.net(x)
#
# build_loaders:
#     from torchvision import datasets
#     from torchvision.transforms import v2
#     transform = v2.Compose([
#         v2.ToTensor(),
#         v2.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
#     ])
#     train_ds = datasets.FashionMNIST(root, train=True,  download=True, transform=transform)
#     test_ds  = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers)
#     test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers)
#     return train_loader, test_loader
#
# evaluate (identical to Exercise 2):
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
#     train_loader, test_loader = build_loaders(batch_size=batch_size)
#     model = FashionMLP(n_hidden=128).to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     last_acc = 0.0
#     for epoch in range(n_epochs):
#         model.train()
#         running = 0.0
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             loss = loss_fn(model(X), y)
#             loss.backward()
#             optimizer.step()
#             running += loss.item() * X.size(0)
#         train_loss = running / len(train_loader.dataset)
#         test_loss, last_acc = evaluate(model, test_loader, loss_fn, device)
#         if verbose:
#             print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
#                   f"test_loss {test_loss:.4f}  test_acc {last_acc:.4f}")
#     return model, last_acc
#
# save_model:
#     torch.save(model.state_dict(), path)
#
# load_model:
#     model = FashionMLP(n_hidden=n_hidden)
#     state = torch.load(path, map_location="cpu", weights_only=True)
#     model.load_state_dict(state)
#     model.to(best_device()).eval()
#     return model
#
# -----------------------------------------------------------------------------
