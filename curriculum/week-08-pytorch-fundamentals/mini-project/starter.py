"""
Mini-project starter -- FashionMNIST classifier in PyTorch 2.x.

This is a working skeleton that you should copy into train.py and modify.
It runs (once torch and torchvision are installed) and reaches ~88% test
accuracy on FashionMNIST in 5 epochs of Adam at lr=1e-3.

The C5 expectation is that you:
    1. Split this file into the multi-file layout suggested in README.md:
         - model.py:   the FashionMLP class
         - data.py:    the build_loaders function
         - train.py:   the train function + main entry
         - evaluate.py: the reload + evaluate script
    2. Add the confusion matrix to evaluate.py.
    3. Add the training-curve plot to train.py.
    4. Write report.md.

References:
    https://pytorch.org/tutorials/beginner/basics/intro.html
    https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


RANDOM_STATE: int = 42
DATA_ROOT: str = "./data"
CHECKPOINT_PATH: str = "./model.pt"

# Standard FashionMNIST channel statistics; precomputed.
FASHION_MNIST_MEAN: float = 0.2860
FASHION_MNIST_STD: float = 0.3530


# -----------------------------------------------------------------------------
# Device detection (Lecture 3 Section 5)
# -----------------------------------------------------------------------------

def best_device() -> "torch.device":
    """Return cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Model (will move to model.py in your version)
# -----------------------------------------------------------------------------

class FashionMLP(nn.Module):
    """A 2-layer MLP for FashionMNIST.

    Architecture:
        Flatten        : (B, 1, 28, 28) -> (B, 784)
        Linear(784, h) : -> (B, h)
        ReLU           : (elementwise)
        Linear(h, 10)  : -> (B, 10) raw logits

    Returns raw logits; pair with nn.CrossEntropyLoss.
    """

    def __init__(self, n_hidden: int = 128, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


# -----------------------------------------------------------------------------
# Data (will move to data.py in your version)
# -----------------------------------------------------------------------------

def build_loaders(
    batch_size: int = 128, root: str = DATA_ROOT, num_workers: int = 0,
) -> "Tuple[DataLoader, DataLoader]":
    """Construct FashionMNIST DataLoaders with standard normalization.

    Imported lazily so this file py_compiles without torchvision installed.
    Reference: https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
    """
    from torchvision import datasets
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
    ])

    train_ds = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform,
    )
    test_ds = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Evaluate (will be in evaluate.py too)
# -----------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: "torch.device",
) -> "Tuple[float, float]":
    """Return (mean_loss, accuracy) over loader. Wraps in no_grad."""
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            n += X.size(0)
    return total_loss / n, total_correct / n


# -----------------------------------------------------------------------------
# Train (will be in train.py)
# -----------------------------------------------------------------------------

def train(
    n_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_hidden: int = 128,
    seed: int = RANDOM_STATE,
    checkpoint_path: str = CHECKPOINT_PATH,
    verbose: bool = True,
) -> "Tuple[nn.Module, list, list, list]":
    """End-to-end FashionMNIST training.

    Returns
    -------
    model : the trained nn.Module
    train_losses : list[float], one per epoch (per-example mean)
    test_losses  : list[float], one per epoch (per-example mean)
    test_accs    : list[float], one per epoch (in [0, 1])

    Side effects: writes model.state_dict() to checkpoint_path at the end.
    """
    torch.manual_seed(seed)
    device = best_device()
    if verbose:
        print(f"[train] device: {device}")

    train_loader, test_loader = build_loaders(batch_size=batch_size)
    model = FashionMLP(n_hidden=n_hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: "list[float]" = []
    test_losses: "list[float]" = []
    test_accs: "list[float]" = []

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
        train_loss = running / len(train_loader.dataset)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if verbose:
            print(
                f"epoch {epoch:3d}  "
                f"train_loss {train_loss:.4f}  "
                f"test_loss {test_loss:.4f}  "
                f"test_acc {test_acc:.4f}"
            )

    torch.save(model.state_dict(), checkpoint_path)
    if verbose:
        print(f"[train] saved state_dict to {checkpoint_path}")

    return model, train_losses, test_losses, test_accs


# -----------------------------------------------------------------------------
# Load (will be in evaluate.py)
# -----------------------------------------------------------------------------

def load_model(
    path: str = CHECKPOINT_PATH, n_hidden: int = 128,
) -> nn.Module:
    """Reload the FashionMLP from a state_dict on disk.

    Uses weights_only=True (PyTorch 2.4+ safe loader; Lecture 3 Section 7).
    """
    model = FashionMLP(n_hidden=n_hidden)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(best_device()).eval()
    return model


# -----------------------------------------------------------------------------
# Plot helpers (will be in train.py; matplotlib imported lazily)
# -----------------------------------------------------------------------------

def plot_training_curves(
    train_losses: "list[float]",
    test_losses: "list[float]",
    test_accs: "list[float]",
    save_path: str = "./training_curves.png",
) -> None:
    """Save a 2-panel figure: loss curves on the left, accuracy on the right."""
    import matplotlib.pyplot as plt

    epochs = list(range(len(train_losses)))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, train_losses, marker="o", label="train loss")
    ax_loss.plot(epochs, test_losses,  marker="o", label="test loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.set_title("Loss per epoch")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, test_accs, marker="o", color="tab:green",
                label="test accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_title("Test accuracy per epoch")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: "list[int]",
    y_pred: "list[int]",
    class_names: "list[str]",
    save_path: str = "./confusion_matrix.png",
) -> None:
    """Save a 10x10 confusion-matrix heatmap with class names on both axes."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color,
                    fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Confusion-matrix collection (will be in evaluate.py)
# -----------------------------------------------------------------------------

def collect_predictions(
    model: nn.Module, loader: DataLoader, device: "torch.device",
) -> "Tuple[list, list]":
    """Run model over loader and collect (y_true, y_pred) lists."""
    model.eval()
    y_true: "list[int]" = []
    y_pred: "list[int]" = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(1).cpu().tolist()
            y_true.extend(y.tolist())
            y_pred.extend(preds)
    return y_true, y_pred


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """Train, save, reload, evaluate, plot. The full mini-project pipeline."""
    print("[main] starting training")
    model, train_losses, test_losses, test_accs = train(
        n_epochs=10, batch_size=128, lr=1e-3, n_hidden=128, seed=RANDOM_STATE,
        checkpoint_path=CHECKPOINT_PATH, verbose=True,
    )

    print("[main] plotting training curves")
    plot_training_curves(train_losses, test_losses, test_accs,
                         save_path="./training_curves.png")

    print("[main] reloading model from disk")
    reloaded = load_model(CHECKPOINT_PATH, n_hidden=128)
    _, test_loader = build_loaders(batch_size=128)
    loss_fn = nn.CrossEntropyLoss()
    device = best_device()
    test_loss, test_acc = evaluate(reloaded, test_loader, loss_fn, device)
    print(f"[main] reloaded test_acc = {test_acc:.4f}")
    assert abs(test_acc - test_accs[-1]) < 1e-3, (
        "Reloaded model accuracy does not match final-epoch accuracy. "
        "Check that load_model uses weights_only=True and that you are "
        "loading into a freshly-constructed FashionMLP with the same n_hidden."
    )

    print("[main] plotting confusion matrix")
    y_true, y_pred = collect_predictions(reloaded, test_loader, device)
    class_names = list(test_loader.dataset.classes)
    plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path="./confusion_matrix.png")

    print("[main] done; see training_curves.png and confusion_matrix.png")


if __name__ == "__main__":
    main()
