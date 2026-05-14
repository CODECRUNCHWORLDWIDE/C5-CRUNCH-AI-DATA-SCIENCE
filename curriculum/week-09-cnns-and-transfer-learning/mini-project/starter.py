"""
Mini-project starter -- transfer-learning CIFAR-10 classifier in PyTorch 2.x.

This is a working skeleton. It runs (once torch and torchvision are installed)
and reaches ~92 percent test accuracy on CIFAR-10 in 10 epochs of fine-tuning
ResNet-18 with differential learning rates.

The C5 expectation is that you:

    1. Split this file into the multi-file layout suggested in README.md:
         - model.py:           the TinyVGG class + the pretrained ResNet-18 helpers
         - data.py:            build_cifar10_loaders_32 + build_cifar10_loaders_224
         - baseline_train.py:  the from-scratch TinyVGG training script
         - baseline_evaluate.py:
         - transfer_train.py:  the ResNet-18 fine-tune training script
         - transfer_evaluate.py:
    2. Add the confusion matrix to the evaluate scripts.
    3. Add the training-curve plot to each train script.
    4. Write report.md.

References:
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    https://pytorch.org/vision/stable/models.html
    https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
"""

from __future__ import annotations

from typing import Tuple


RANDOM_STATE: int = 42
DATA_ROOT: str = "./data"
BASELINE_CHECKPOINT: str = "./baseline_model.pt"
TRANSFER_CHECKPOINT: str = "./transfer_model.pt"

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Device detection
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
# Models (move to model.py)
# -----------------------------------------------------------------------------


def build_tiny_vgg(n_classes: int = 10) -> "object":  # type: ignore[type-arg]
    """The TinyVGG from Lecture 2 Section 5. ~1.1M params, ~70% on CIFAR-10."""
    import torch
    from torch import nn

    class TinyVGG(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 2
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Block 3
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(256, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.features(x)
            h = self.global_pool(h)
            h = torch.flatten(h, start_dim=1)
            return self.classifier(h)

    return TinyVGG()


def build_pretrained_resnet18(n_classes: int = 10) -> "object":  # type: ignore[type-arg]
    """ResNet-18 pretrained on ImageNet, head replaced for CIFAR-10."""
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights  # type: ignore

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def make_finetune_optimizer(
    model: "object",
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
) -> "object":  # type: ignore[type-arg]
    """Two parameter groups: backbone at 1e-4, head at 1e-3."""
    import torch

    head_params = list(model.fc.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )


# -----------------------------------------------------------------------------
# Data loaders (move to data.py)
# -----------------------------------------------------------------------------


def build_cifar10_loaders_32(
    root: str = DATA_ROOT,
    batch_size: int = 128,
    num_workers: int = 2,
) -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Native 32x32 CIFAR-10 with CIFAR-10 normalization. For the from-scratch baseline."""
    import torch
    import torchvision  # type: ignore
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2  # type: ignore

    train_tf = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=torch.cuda.is_available()),
    )


def build_cifar10_loaders_224(
    root: str = DATA_ROOT,
    batch_size: int = 64,
    num_workers: int = 2,
) -> "Tuple[object, object]":  # type: ignore[type-arg]
    """224x224 CIFAR-10 with ImageNet normalization. For transfer learning."""
    import torch
    import torchvision  # type: ignore
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2  # type: ignore

    train_tf = v2.Compose([
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_tf = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf,
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=torch.cuda.is_available()),
    )


# -----------------------------------------------------------------------------
# Training loop (move to a shared utility module or duplicate in each script)
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: "object",
    loader: "object",
    optimizer: "object",
    loss_fn: "object",
    device: "object",
) -> float:
    """The eight-line loop from Week 8 Lecture 2 Section 7."""
    import torch

    model.train()
    total_loss = 0.0
    n_seen = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n_seen += X.size(0)
    return total_loss / n_seen


def evaluate(
    model: "object",
    loader: "object",
    device: "object",
) -> float:
    """Top-1 accuracy on the loader."""
    import torch

    model.eval()
    n_correct = 0
    n_seen = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X).argmax(dim=1)
            n_correct += (pred == y).sum().item()
            n_seen += X.size(0)
    return n_correct / n_seen


# -----------------------------------------------------------------------------
# Save / load helpers (move to a shared utility module)
# -----------------------------------------------------------------------------


def save_model(model: "object", path: str) -> None:
    """Save state_dict (not the model object). Lecture 3 Section 8 of Week 8."""
    import torch

    torch.save(model.state_dict(), path)


def load_state_into(model: "object", path: str, device: "object") -> "object":  # type: ignore[type-arg]
    """Load state_dict into the given model in place. Uses weights_only=True."""
    import torch

    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


# -----------------------------------------------------------------------------
# Top-level training scripts (split into baseline_train.py and transfer_train.py)
# -----------------------------------------------------------------------------


def train_baseline(
    n_epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = RANDOM_STATE,
) -> None:
    """The from-scratch TinyVGG training run."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = best_device()
    print(f"baseline: training on {device}")

    train_loader, test_loader = build_cifar10_loaders_32(batch_size=batch_size)
    model = build_tiny_vgg(n_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    save_model(model, BASELINE_CHECKPOINT)
    print(f"baseline: saved {BASELINE_CHECKPOINT}")


def train_transfer(
    n_epochs: int = 10,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    seed: int = RANDOM_STATE,
) -> None:
    """The transfer-learning ResNet-18 fine-tune."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = best_device()
    print(f"transfer: training on {device}")

    train_loader, test_loader = build_cifar10_loaders_224(batch_size=batch_size)
    model = build_pretrained_resnet18(n_classes=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_finetune_optimizer(model, backbone_lr=backbone_lr, head_lr=head_lr)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    save_model(model, TRANSFER_CHECKPOINT)
    print(f"transfer: saved {TRANSFER_CHECKPOINT}")


# -----------------------------------------------------------------------------
# Entry point (run both back to back)
# -----------------------------------------------------------------------------


def main() -> None:
    """In the real project this would be two scripts; here we chain them."""
    train_baseline()
    train_transfer()


if __name__ == "__main__":
    main()
