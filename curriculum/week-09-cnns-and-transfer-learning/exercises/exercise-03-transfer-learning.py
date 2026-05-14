"""
Exercise 3 -- Transfer Learning with ResNet-18.

Goal: load torchvision.models.resnet18 with the IMAGENET1K_V1 pretrained
weights, replace the classification head with a Linear(512, 10), freeze the
backbone, and train the head on CIFAR-10 for 5 epochs of Adam at lr=1e-3.
Target test accuracy: >=85%.

By the end of this exercise you will have:

    (1) A build_feature_extractor function that loads pretrained ResNet-18,
        freezes every parameter, then replaces model.fc with a new head.
    (2) A function that counts and reports the trainable parameter count
        (should be only the new head's ~5k parameters).
    (3) A build_cifar10_loaders_224 function that resizes CIFAR-10 to 224x224
        with ImageNet normalization (the right preprocessing for the
        pretrained backbone; see Lecture 3 Section 7).
    (4) A training run for 5 epochs that reaches >=85% test accuracy.
    (5) A second function build_finetune that does the full fine-tune with
        differential learning rates (backbone at 1e-4, head at 1e-3).
    (6) Tests that verify the freeze actually froze the backbone, the head
        is the right shape, and the loaders produce 224x224 batches.

Estimated time: 60-120 minutes (training is fast for the frozen case).

Run with:    python exercise-03-transfer-learning.py
Or test:     pytest exercise-03-transfer-learning.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-03-transfer-learning.py succeeds.

Note: First call to resnet18(weights=...) downloads ~45 MB of pretrained
weights to ~/.cache/torch/hub/checkpoints/. First call to CIFAR10(...)
downloads ~170 MB to ./data/. Both are cached after that.

PyTorch references:
    https://pytorch.org/vision/stable/models.html
    https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

from __future__ import annotations

import os
from typing import Tuple

# torch and torchvision are imported lazily so this file compiles cleanly
# without them. The pytest functions require both.


RANDOM_STATE: int = 42
DATA_ROOT: str = "./data"

# ImageNet normalization (the pretrained backbone expects this; do not use
# CIFAR-10's normalization here, see Lecture 3 Section 7).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Part A -- device detection (same as Week 8 / Exercise 2)
# -----------------------------------------------------------------------------


def best_device() -> "object":  # type: ignore[type-arg]
    """Return cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Part B -- the feature-extraction recipe (frozen backbone)
# -----------------------------------------------------------------------------


def build_feature_extractor(n_classes: int = 10) -> "object":  # type: ignore[type-arg]
    """ResNet-18 with frozen backbone and a new classification head.

    Recipe from Lecture 3 Section 4:

        1. Load resnet18 with the IMAGENET1K_V1 pretrained weights.
        2. Freeze every parameter (requires_grad = False).
        3. Replace model.fc with a new nn.Linear(model.fc.in_features, n_classes).

    The order matters: freeze FIRST, then swap the head. If you swap first
    the new head defaults to requires_grad=True; then your freeze loop turns
    it back off and nothing trains.

    Returns:
        A torch.nn.Module with only the head parameters trainable.

    References:
        https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
        https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights  # type: ignore

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    # TODO Step 1: freeze every parameter with a for loop over model.parameters().
    # TODO Step 2: replace model.fc with nn.Linear(model.fc.in_features, n_classes).
    raise NotImplementedError("Part B -- build_feature_extractor")


def count_trainable_parameters(model: "object") -> int:
    """Return the number of parameters with requires_grad=True.

    For a correctly-built feature extractor on CIFAR-10, this should be
    exactly 512*10 + 10 = 5130 (the new Linear's weight and bias).
    """
    # TODO: sum p.numel() over model.parameters() where p.requires_grad.
    raise NotImplementedError("Part B -- count_trainable_parameters")


# -----------------------------------------------------------------------------
# Part C -- CIFAR-10 loaders at 224x224 with ImageNet normalization
# -----------------------------------------------------------------------------


def build_cifar10_loaders_224(
    root: str = DATA_ROOT,
    batch_size: int = 64,
    num_workers: int = 2,
) -> "Tuple[object, object]":  # type: ignore[type-arg]
    """CIFAR-10 resized to 224x224 with ImageNet normalization.

    The pretrained ResNet-18 was trained on 224x224 ImageNet images with
    ImageNet's per-channel mean and std. Feeding it 32x32 CIFAR-style input
    works but underperforms because the pretrained features assume larger
    input scale.

    Training transforms (in order):
        Resize((224, 224)), RandomHorizontalFlip,
        ToImage, ToDtype(float32, scale=True), Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Test transforms (in order):
        Resize((224, 224)),
        ToImage, ToDtype(float32, scale=True), Normalize(IMAGENET_MEAN, IMAGENET_STD)

    References:
        https://pytorch.org/vision/stable/transforms.html
        https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
    """
    import torch
    import torchvision  # type: ignore
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2  # type: ignore

    # TODO: build train_tf and test_tf with the transforms specified above.
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
# Part D -- training loop (same as Week 8 / Exercise 2)
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: "object",
    loader: "object",
    optimizer: "object",
    loss_fn: "object",
    device: "object",
) -> float:
    """One epoch of the bare loop. Returns mean train loss."""
    # TODO: implement (same as Exercise 2 Part D).
    raise NotImplementedError("Part D -- train_one_epoch")


def evaluate(
    model: "object",
    loader: "object",
    device: "object",
) -> float:
    """Top-1 accuracy on the loader."""
    # TODO: implement (same as Exercise 2 Part D).
    raise NotImplementedError("Part D -- evaluate")


def train_feature_extractor(
    n_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = RANDOM_STATE,
) -> "Tuple[object, float]":  # type: ignore[type-arg]
    """Train the frozen-backbone feature extractor. Returns (model, final_test_acc)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = best_device()

    train_loader, test_loader = build_cifar10_loaders_224(batch_size=batch_size)
    model = build_feature_extractor(n_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Only the head's parameters need an optimizer step. We could pass
    # model.parameters() (the frozen ones are no-ops) but this is cleaner.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    final_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = evaluate(model, test_loader, device)
        final_acc = test_acc
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    return model, final_acc


# -----------------------------------------------------------------------------
# Part E -- full fine-tune with differential learning rates
# -----------------------------------------------------------------------------


def build_finetune(n_classes: int = 10) -> "object":  # type: ignore[type-arg]
    """ResNet-18 with a new head, no freezing (full fine-tune)."""
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights  # type: ignore

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # TODO: replace model.fc with nn.Linear(model.fc.in_features, n_classes).
    # Do NOT freeze; every parameter stays trainable.
    raise NotImplementedError("Part E -- build_finetune")


def make_finetune_optimizer(
    model: "object",
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
) -> "object":  # type: ignore[type-arg]
    """Two parameter groups: backbone at 1e-4, head at 1e-3.

    Implementation hint (Lecture 3 Section 5): use id(p) to identify the
    parameters of model.fc and exclude them from the backbone group. Then
    construct an Adam optimizer with two param groups.
    """
    import torch

    # TODO: build head_params (list of tensors in model.fc.parameters()),
    # build a set of their ids, then build backbone_params as all other
    # parameters. Return torch.optim.Adam([
    #     {"params": backbone_params, "lr": backbone_lr},
    #     {"params": head_params, "lr": head_lr},
    # ]).
    raise NotImplementedError("Part E -- make_finetune_optimizer")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_feature_extractor_freezes_backbone() -> None:
    """Only model.fc's parameters should have requires_grad=True."""
    import torch

    model = build_feature_extractor(n_classes=10)
    head_params = list(model.fc.parameters())
    head_param_ids = {id(p) for p in head_params}
    for name, p in model.named_parameters():
        if id(p) in head_param_ids:
            assert p.requires_grad, f"head param {name} is not trainable"
        else:
            assert not p.requires_grad, f"backbone param {name} is still trainable"


def test_feature_extractor_head_shape() -> None:
    """The new head must be nn.Linear(512, 10)."""
    import torch
    from torch import nn

    model = build_feature_extractor(n_classes=10)
    assert isinstance(model.fc, nn.Linear), f"fc is {type(model.fc)}, expected nn.Linear"
    assert model.fc.in_features == 512, f"in_features = {model.fc.in_features}"
    assert model.fc.out_features == 10, f"out_features = {model.fc.out_features}"


def test_count_trainable_parameters() -> None:
    """For a frozen-backbone ResNet-18 with a 10-class head, trainable params = 5130."""
    model = build_feature_extractor(n_classes=10)
    n_trainable = count_trainable_parameters(model)
    # 512 * 10 (weight) + 10 (bias) = 5130
    assert n_trainable == 5130, f"expected 5130 trainable params; got {n_trainable}"


def test_loaders_224_shape() -> None:
    """build_cifar10_loaders_224 yields (N, 3, 224, 224) batches."""
    train_loader, test_loader = build_cifar10_loaders_224(batch_size=4, num_workers=0)
    X, y = next(iter(train_loader))
    assert X.shape == (4, 3, 224, 224), f"train X shape: {X.shape}"
    assert y.shape == (4,), f"train y shape: {y.shape}"
    X, y = next(iter(test_loader))
    assert X.shape == (4, 3, 224, 224), f"test X shape: {X.shape}"
    assert y.shape == (4,), f"test y shape: {y.shape}"


def test_finetune_has_no_frozen_params() -> None:
    """The full fine-tune model has every parameter trainable."""
    model = build_finetune(n_classes=10)
    for name, p in model.named_parameters():
        assert p.requires_grad, f"param {name} is frozen in build_finetune"


def test_finetune_optimizer_has_two_param_groups() -> None:
    """make_finetune_optimizer returns an Adam with two param groups (lr=1e-4, 1e-3)."""
    model = build_finetune(n_classes=10)
    opt = make_finetune_optimizer(model)
    assert len(opt.param_groups) == 2, f"expected 2 param groups; got {len(opt.param_groups)}"
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-4, 1e-3], f"expected lrs [1e-4, 1e-3]; got {lrs}"


def test_feature_extractor_reaches_eighty_five_percent_slow() -> None:
    """Full 5-epoch training run; expects test_acc >= 0.85.

    Slow: 5-30 minutes depending on device. Set RUN_SLOW=1 to run.
    """
    if os.environ.get("RUN_SLOW") != "1":
        return
    _, final_acc = train_feature_extractor(n_epochs=5)
    assert final_acc >= 0.85, f"feature extractor only reached {final_acc:.4f}; expected >=0.85"


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


def main() -> None:
    """Fast sanity checks; the slow training is gated behind RUN_SLOW=1."""
    test_feature_extractor_freezes_backbone()
    test_feature_extractor_head_shape()
    test_count_trainable_parameters()
    test_finetune_has_no_frozen_params()
    test_finetune_optimizer_has_two_param_groups()
    print("OK -- exercise 3 (fast checks)")


if __name__ == "__main__":
    main()
