"""
Exercise 2 -- Build a CNN from scratch on CIFAR-10.

Goal: build the TinyVGG CNN from Lecture 2 Section 5, train it on CIFAR-10
for 15 epochs of Adam at lr=1e-3, and reach >=70% test accuracy.

By the end of this exercise you will have:

    (1) A TinyVGG nn.Module subclass with three Conv-BN-ReLU-Conv-BN-ReLU-Pool
        blocks, a global average pool, and a single Linear classifier.
    (2) A build_cifar10_loaders function that returns train and test
        DataLoaders for CIFAR-10 via torchvision.datasets.CIFAR10, with the
        standard CIFAR-10 normalization and RandomCrop+RandomHorizontalFlip
        augmentation on the training loader only.
    (3) A train function with the eight-line loop from Week 8, per-epoch
        validation, and a per-epoch print statement.
    (4) A parameter-count check that verifies the model has ~1.1M parameters.
    (5) A test that the model reaches >=70% test accuracy in 15 epochs
        (this test is slow; it actually trains the model).

Estimated time: 90-150 minutes (most of it watching training).

Run with:    python exercise-02-build-a-cnn.py
Or test:     pytest exercise-02-build-a-cnn.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-build-a-cnn.py succeeds.

NOTE: Running the full training test downloads CIFAR-10 (~170 MB) into
./data/ and takes 30 minutes on CPU, 5 minutes on a Colab T4 GPU, or
20 minutes on Apple Silicon mps. The shape-only tests are fast.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
"""

from __future__ import annotations

import os
from typing import Tuple

# torch and torchvision are imported lazily inside functions so this file
# compiles cleanly without them installed. The pytest functions do require
# torch; the conftest infrastructure handles the skip.


RANDOM_STATE: int = 42
DATA_ROOT: str = "./data"

# Standard channel statistics for CIFAR-10; precomputed on the train split.
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


# -----------------------------------------------------------------------------
# Part A -- device detection (same as Week 8)
# -----------------------------------------------------------------------------


def best_device() -> "object":  # type: ignore[type-arg]
    """Return cuda > mps > cpu as a torch.device.

    See Week 8 Lecture 3 Section 5. The return type is annotated as object
    so py_compile does not need torch at parse time; the actual return is
    a torch.device.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Part B -- the TinyVGG model
# -----------------------------------------------------------------------------


def build_tiny_vgg(n_classes: int = 10) -> "object":  # type: ignore[type-arg]
    """Construct the TinyVGG model from Lecture 2 Section 5.

    Architecture (input: (N, 3, 32, 32)):

        Block 1: Conv(3->64, 3x3, p=1) -> BN -> ReLU
                 Conv(64->64, 3x3, p=1) -> BN -> ReLU
                 MaxPool(2)                                   -> (N, 64, 16, 16)
        Block 2: Conv(64->128, 3x3, p=1) -> BN -> ReLU
                 Conv(128->128, 3x3, p=1) -> BN -> ReLU
                 MaxPool(2)                                   -> (N, 128, 8, 8)
        Block 3: Conv(128->256, 3x3, p=1) -> BN -> ReLU
                 Conv(256->256, 3x3, p=1) -> BN -> ReLU
                 MaxPool(2)                                   -> (N, 256, 4, 4)
        Head:    AdaptiveAvgPool2d((1, 1)) -> Flatten -> Linear(256, n_classes)

    Conv layers use bias=False because BatchNorm has its own bias. Use
    nn.Sequential to chain the body; you do not need a custom forward.

    Returns:
        A torch.nn.Module instance.

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
    """
    # Lazy import so py_compile passes without torch.
    import torch
    from torch import nn

    class TinyVGG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # TODO: build self.features as an nn.Sequential containing the
            # three blocks specified in the docstring. Each conv has
            # kernel_size=3, padding=1, bias=False; each MaxPool is 2x2.
            self.features = nn.Sequential(
                # Block 1: TODO fill in
                # Block 2: TODO fill in
                # Block 3: TODO fill in
            )
            # TODO: self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            # TODO: self.classifier = nn.Linear(256, n_classes)
            raise NotImplementedError("Part B -- TinyVGG.__init__")

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # TODO: features -> global_pool -> flatten(start_dim=1) -> classifier
            raise NotImplementedError("Part B -- TinyVGG.forward")

    return TinyVGG()


# -----------------------------------------------------------------------------
# Part C -- CIFAR-10 loaders (the canonical pipeline from Lecture 2 Section 7)
# -----------------------------------------------------------------------------


def build_cifar10_loaders(
    root: str = DATA_ROOT,
    batch_size: int = 128,
    num_workers: int = 2,
) -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Return (train_loader, test_loader) for CIFAR-10.

    Training transforms (in order):
        RandomCrop(32, padding=4), RandomHorizontalFlip,
        ToImage, ToDtype(float32, scale=True), Normalize(CIFAR10_MEAN, CIFAR10_STD)

    Test transforms (in order):
        ToImage, ToDtype(float32, scale=True), Normalize(CIFAR10_MEAN, CIFAR10_STD)

    References:
        https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
        https://pytorch.org/vision/stable/transforms.html
    """
    import torch
    import torchvision  # type: ignore
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2  # type: ignore

    # TODO: build train_tf and test_tf using v2.Compose with the transforms
    # listed in the docstring above. The order matters: augmentation (crop
    # and flip) operates on PIL images and must come before ToImage; the
    # normalize step needs a float tensor and so must come after ToDtype.
    train_tf = None  # TODO
    test_tf = None  # TODO

    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Part D -- the training loop (unchanged from Week 8)
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: "object",
    loader: "object",
    optimizer: "object",
    loss_fn: "object",
    device: "object",
) -> float:
    """One epoch of the bare PyTorch training loop. Returns mean train loss.

    The pattern from Week 8 Lecture 2 Section 7:
        model.train()
        for X, y in loader:
            X = X.to(device); y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
    """
    # TODO: implement the eight-line loop, accumulating total_loss and n_seen
    # for the per-epoch running mean. See exercise-03-fashionmnist-end-to-end.py
    # from Week 8 for a reference solution.
    raise NotImplementedError("Part D -- train_one_epoch")


def evaluate(
    model: "object",
    loader: "object",
    device: "object",
) -> float:
    """One forward pass over the loader. Returns top-1 accuracy as a float in [0, 1].

    Use:
        model.eval()
        with torch.no_grad():
            for X, y in loader: ...
    """
    # TODO: implement.
    raise NotImplementedError("Part D -- evaluate")


def train_tiny_vgg(
    n_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = RANDOM_STATE,
) -> "Tuple[object, float]":  # type: ignore[type-arg]
    """End-to-end training. Returns (trained_model, final_test_accuracy)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = best_device()

    train_loader, test_loader = build_cifar10_loaders(batch_size=batch_size)
    model = build_tiny_vgg(n_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = evaluate(model, test_loader, device)
        final_acc = test_acc
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    return model, final_acc


# -----------------------------------------------------------------------------
# Tests (require torch + torchvision; some are slow)
# -----------------------------------------------------------------------------


def test_build_tiny_vgg_shapes() -> None:
    """Forward pass on a fake CIFAR-10 batch returns shape (N, 10)."""
    import torch

    model = build_tiny_vgg(n_classes=10)
    x = torch.zeros(4, 3, 32, 32)
    y = model(x)
    assert y.shape == (4, 10), f"shape mismatch: {y.shape}"


def test_build_tiny_vgg_parameter_count() -> None:
    """TinyVGG should have between 1.0M and 1.3M parameters."""
    model = build_tiny_vgg(n_classes=10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_000_000 <= n_params <= 1_300_000, (
        f"expected ~1.1M parameters; got {n_params:,}"
    )


def test_loaders_have_expected_shapes() -> None:
    """Both loaders yield batches with (N, 3, 32, 32) X and (N,) y."""
    train_loader, test_loader = build_cifar10_loaders(batch_size=8, num_workers=0)
    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))
    assert X_train.shape == (8, 3, 32, 32), f"train X shape: {X_train.shape}"
    assert y_train.shape == (8,), f"train y shape: {y_train.shape}"
    assert X_test.shape == (8, 3, 32, 32), f"test X shape: {X_test.shape}"
    assert y_test.shape == (8,), f"test y shape: {y_test.shape}"


def test_tiny_vgg_reaches_seventy_percent_on_cifar10_slow() -> None:
    """Full 15-epoch training run; expects test_acc >= 0.70.

    This test is slow (~30 min CPU, ~5 min GPU). It is marked slow so CI
    can skip it in fast mode. Set RUN_SLOW=1 in the environment to run it.
    """
    if os.environ.get("RUN_SLOW") != "1":
        # Skip the slow test by returning early; pytest treats this as pass.
        return
    _, final_acc = train_tiny_vgg(n_epochs=15)
    assert final_acc >= 0.70, f"TinyVGG only reached {final_acc:.4f}; expected >=0.70"


# -----------------------------------------------------------------------------
# Main entry: fast sanity checks (slow training only via pytest with RUN_SLOW=1)
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the fast checks (forward pass, parameter count) without training."""
    test_build_tiny_vgg_shapes()
    test_build_tiny_vgg_parameter_count()
    # We do NOT run the slow training in main(); use pytest with RUN_SLOW=1.
    print("OK -- exercise 2 (fast checks)")


if __name__ == "__main__":
    main()
