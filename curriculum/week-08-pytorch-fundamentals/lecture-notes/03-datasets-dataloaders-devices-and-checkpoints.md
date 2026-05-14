# Lecture 3 — Datasets, DataLoaders, Devices, and Checkpoints

> **Outcome:** You can subclass `Dataset` for a custom data source and wrap it in a `DataLoader` with sensible shuffle / batch / worker settings. You can load FashionMNIST in three lines via `torchvision.datasets`. You can detect the best available device (`cuda`, `mps`, `cpu`) and place a model and its inputs on it without surprise. You can save and load a `state_dict` correctly, including the `weights_only=True` flag and the `map_location` argument. After this lecture, the entire data and persistence pipeline of a PyTorch project is no longer a mystery.

Lectures 1 and 2 built the model and the training loop. They both used in-memory tensors directly: `X = torch.randn(64, 784)`, `y = torch.randint(0, 10, (64,))`. Real workflows do not look like that. Real datasets live on disk, in databases, or behind APIs; real models train on data too large to fit in RAM; real GPUs sit idle while a single-threaded data loader struggles to keep up.

This lecture is the plumbing. `Dataset` is the abstraction over your data source. `DataLoader` is the abstraction over batching, shuffling, and parallel prefetch. Device placement is the abstraction over CPU and GPU. `state_dict` is the abstraction over persistence. None of these are deep — each is one or two pages of API — but together they are what separates a tutorial demo from a real PyTorch project.

We target **PyTorch 2.x** and **torchvision 0.19+**. The official data documentation is at <https://pytorch.org/docs/stable/data.html>; the torchvision datasets at <https://pytorch.org/vision/stable/datasets.html>.

---

## 1. The `Dataset` abstraction

`torch.utils.data.Dataset` is an abstract base class with two methods you must implement:

```python
from typing import Tuple
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]
```

That is the entire interface. `__len__` returns the size; `__getitem__(i)` returns the i-th example as a `(input, target)` tuple. Nothing about batching, nothing about shuffling, nothing about devices — those are the `DataLoader`'s job.

The C5 convention is to keep `Dataset` subclasses *thin*: they should return tensors (or convertible objects) and apply minimal transforms. Heavy preprocessing (StandardScaler, image augmentation, tokenization) goes in a `transform` callable passed to the constructor:

```python
class MyImageDataset(Dataset):
    def __init__(self, image_paths: list[str], labels: list[int],
                 transform=None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = load_image(self.image_paths[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]
```

This is the pattern that `torchvision.datasets.FashionMNIST` (and every `torchvision` dataset) follows. We use it ourselves in Challenge 2.

Reference: <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>.

### `TensorDataset` — for in-memory tensors

If your data already fits in memory as a pair of tensors, you do not need to subclass anything:

```python
from torch.utils.data import TensorDataset
ds = TensorDataset(X, y)
```

`TensorDataset(*tensors)` wraps any number of tensors that share their first dimension. Used in Exercise 2 and in the homework for the synthetic-data warmups.

Reference: <https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset>.

---

## 2. `DataLoader` — batching, shuffling, parallel prefetch

`torch.utils.data.DataLoader` wraps a `Dataset` and produces a Python iterator over **batches**:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset=ds,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

for X_batch, y_batch in loader:
    ...                                # each batch is a tuple of tensors
```

The seven arguments that matter:

1. **`dataset`** — the `Dataset` to draw from. Required.
2. **`batch_size`** — examples per batch. Defaults to 1; never use the default. Common choices: 32, 64, 128, 256. For FashionMNIST: 128 is the C5 default.
3. **`shuffle`** — `True` for training, `False` for evaluation. When `True`, the loader internally builds a `RandomSampler`; the order is re-shuffled at the start of each epoch. The `random_state` is the global PyTorch seed (set via `torch.manual_seed`).
4. **`num_workers`** — number of subprocess workers that prefetch batches. `0` means everything happens in the main process; `>0` spawns workers via Python multiprocessing. For Week 8 the default `0` is fine; for the Week 9 CIFAR-10 work you will want `2-4`.
5. **`pin_memory`** — when `True`, batches are placed in page-locked CPU memory before being passed to the user, which speeds up the subsequent transfer to GPU. Set `True` if and only if you are training on GPU.
6. **`drop_last`** — when `True`, the final partial batch is discarded. Useful when batch-norm misbehaves on a tiny batch; for plain MLPs, leave `False`.
7. **`persistent_workers`** — when `True`, workers are reused across epochs instead of respawning. Saves the spawn overhead; PyTorch 1.7+.

Reference: <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>.

### What `num_workers` actually does

`num_workers=0` (default) means each `next(iter(loader))` runs `dataset[i]` for each example in the batch *in the main process*, then collates the results into a batch tensor. The forward pass, backward pass, and data loading take turns. On a fast model and a slow disk, the GPU sits idle while the disk reads.

`num_workers=4` spawns four subprocess workers. Each worker holds its own copy of the dataset object, runs `dataset[i]` on assigned indices, and pushes the resulting tensors to a shared queue. The main process pulls collated batches off the queue while the GPU computes. The overlap is the speedup.

The rules of thumb:

- For tiny in-memory datasets (FashionMNIST in memory): `num_workers=0` is fine. The overhead of spawning workers exceeds the parallelism gain.
- For on-disk image datasets (CIFAR-10 with disk reads per item, image augmentation): `num_workers=2-4`.
- For huge datasets with heavy augmentation: `num_workers=8-16`. The right number is whatever feeds the GPU without saturating CPU.

On macOS, `num_workers>0` warns about the `fork` start method; PyTorch handles it but the first batch is 5-10 seconds slower. On Windows, `num_workers>0` requires the training script to be guarded by `if __name__ == "__main__":` (the multiprocessing requirement). Both are documented at <https://pytorch.org/docs/stable/data.html#multi-process-data-loading>.

> **EXPERIMENT — measure the `num_workers` speedup.** With FashionMNIST in memory, time one epoch of the loop in Lecture 2 Section 10 at `num_workers=0` and at `num_workers=4`. For this dataset, the speedup is small (the data is in memory; the workers spend most of their time waiting on the queue). Now replace `FashionMNIST` with a synthetic dataset that adds a 50-millisecond `time.sleep` to each `__getitem__` (simulating a slow disk read). Re-time. The `num_workers=4` version is now ~4x faster. The lesson: workers matter when `__getitem__` is slow, not when it is fast.

---

## 3. FashionMNIST via torchvision

`torchvision` ships a curated collection of image datasets, each one a `Dataset` subclass. For Week 8 we use `FashionMNIST`:

```python
from torchvision import datasets
from torchvision.transforms.v2 import Compose, ToTensor, Normalize

transform = Compose([
    ToTensor(),                        # PIL image -> tensor in [0, 1], CHW order
    Normalize((0.2860,), (0.3530,)),   # FashionMNIST channel mean / std
])

train_ds = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform,
)
test_ds = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform,
)
```

The first call to `FashionMNIST(..., download=True)` downloads ~30 MB into `./data/FashionMNIST/raw/` and converts the raw IDX files into a tensor-friendly format. Subsequent calls find the cache and return instantly. Reference: <https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html>.

The transforms pipeline does two things:

- **`ToTensor()`** converts a PIL image (with values in `[0, 255]`) to a `float32` tensor in `[0.0, 1.0]`, reordering from HWC (height, width, channel) to CHW (channel, height, width). PyTorch convolutions expect CHW. Reference: <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToTensor.html>.
- **`Normalize(mean, std)`** subtracts a per-channel mean and divides by a per-channel std. The FashionMNIST values `(0.2860, 0.3530)` are the empirical channel statistics. For an MLP that flattens the image, this normalization is the deep-learning equivalent of the `StandardScaler` from Week 4. Reference: <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html>.

The ten FashionMNIST classes (from the paper, <https://arxiv.org/abs/1708.07747>):

```text
0  T-shirt/top      5  Sandal
1  Trouser          6  Shirt
2  Pullover         7  Sneaker
3  Dress            8  Bag
4  Coat             9  Ankle boot
```

The benchmark numbers to know:

- **Random baseline:** 10% accuracy.
- **Logistic regression:** ~84%.
- **2-layer MLP (the Week 8 target):** ~88%.
- **Small CNN (the Week 9 target):** ~92%.
- **ResNet / DenseNet (state of the art):** ~96%.

Your Week 8 target is `≥85%` test accuracy. Anything from 85% to 89% is in range; lower and you have a bug, higher and you may have leaked the test set or overfit to a single random seed.

For the MLP, you need to **flatten** the `(1, 28, 28)` image into a `(784,)` vector. In the model:

```python
class FashionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)                 # (batch, 1, 28, 28) -> (batch, 784)
        return self.net(x)
```

`nn.Flatten()` is a `Module` that turns `(batch, *)` into `(batch, prod(*))`. Convenient; standard; covered in Lecture 2 Section 4 of the official PyTorch tutorial. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>.

---

## 4. CIFAR-10 — the stretch goal

For students who want a harder benchmark, `torchvision.datasets.CIFAR10` provides 50,000 training and 10,000 test 32×32 color images across ten classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

```python
from torchvision import datasets
from torchvision.transforms.v2 import Compose, ToTensor, Normalize

transform = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
```

The standard CIFAR-10 mean and std are documented (with citations) at <https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py>. For an MLP on CIFAR-10, the flattening turns each image into a `(3 * 32 * 32,) = (3072,)` vector. A 2-layer MLP reaches ~50% test accuracy — far below the small-CNN baseline of ~70%. The gap is the motivation for Week 9. The C5 challenge in Challenge 1 quantifies the gap by running both architectures side by side.

Reference: <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>.

---

## 5. Device placement

A model's parameters live on some device. The model's `__init__` initializes them on CPU. To train on GPU, you move them:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMLP().to(device)
```

The `.to(device)` call recursively moves every parameter and buffer registered in the module tree. It is in-place on the module (returns `self`), even though it creates new tensors on the destination device. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to>.

The batch must also be moved, every iteration:

```python
for X, y in train_loader:
    X, y = X.to(device), y.to(device)
    ...
```

Two flavors of move:

- **CPU → GPU** is a `cudaMemcpyAsync` call on CUDA; takes ~milliseconds per batch on a typical setup. With `pin_memory=True` on the loader, the transfer is overlapped with computation and is effectively free.
- **GPU → CPU** is the same in reverse, used at the end of the loop when you want to read tensor values (`loss.item()` does this implicitly).

The C5 detection idiom for 2026, supporting CUDA, Apple Silicon `mps`, and CPU:

```python
def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Reference: <https://pytorch.org/docs/stable/notes/mps.html> for the `mps` (Apple Silicon Metal Performance Shaders) backend. Some ops are still slower on `mps` than on `cpu` in PyTorch 2.4 (notably some `nn.Conv2d` configurations); the C5 default is to prefer `mps` when available but fall back to `cpu` if a specific op warns.

> **EXPERIMENT — measure the CPU vs. device speedup.** On a `(2048, 784)` batch with a `(784, 128, 10)` MLP, time 100 forward passes on `cpu`, then on `cuda` (or `mps`). For this tiny model the speedup is modest (the model is so small that the GPU's parallelism is underused); the C5 expectation is roughly 2-3x on `cuda` and 1-2x on `mps`. Now do the same experiment with a `(64, 3, 32, 32)` batch through a small CNN (a 6-layer Conv2d + Linear network); the speedup should jump to 10-20x on `cuda`. The lesson: GPU pays off when the model is large enough that the parallelism dominates the transfer cost. A 100k-parameter MLP is at the threshold; a 1M-parameter CNN is solidly on the GPU side.

---

## 6. The full FashionMNIST training script

Combining everything in Lecture 1, Lecture 2 Section 7-8, and Lecture 3 Sections 1-5, the full training script for the Week 8 FashionMNIST exercise:

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.v2 import Compose, ToTensor, Normalize


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FashionMLP(nn.Module):
    def __init__(self, n_hidden: int = 128, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    transform = Compose([ToTensor(), Normalize((0.2860,), (0.3530,))])
    train_ds = datasets.FashionMNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += loss_fn(logits, y).item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_correct / n


def train(n_epochs: int = 10, batch_size: int = 128, lr: float = 1e-3) -> nn.Module:
    torch.manual_seed(42)
    device = best_device()
    train_loader, test_loader = build_loaders(batch_size=batch_size)
    model = FashionMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
              f"test_loss {test_loss:.4f}  test_acc {test_acc:.4f}")

    return model


if __name__ == "__main__":
    train()
```

About 60 lines. Run it; in 5-10 minutes on a CPU (or under one minute on a modest GPU) it will print ten lines and finish with `test_acc` near `0.88`. That is the Week 8 MLP-on-FashionMNIST result. It is also exactly the structure your mini-project will have.

---

## 7. Saving and loading `state_dict`

Once `train()` returns a trained model, you need to save it. The right way:

```python
torch.save(model.state_dict(), "fashion_mlp.pt")
```

To load it later:

```python
model = FashionMLP()                                            # construct same class
state = torch.load("fashion_mlp.pt", weights_only=True)         # PyTorch 2.4+
model.load_state_dict(state)
model.eval()                                                    # ready for inference
```

Four things to know:

1. **You must reconstruct the model class first.** The state_dict is just the parameter values; the model architecture lives in your code. This is intentional — it means a state_dict is portable across machines as long as the class definition is available.
2. **`weights_only=True` is the safe deserialization path.** PyTorch 2.4 introduced this flag (defaulting to `False` for backward compatibility); PyTorch 2.6 plans to default it to `True`. Pass it explicitly; it disables the pickle unpickler and only loads tensor data. Reference: <https://pytorch.org/docs/stable/generated/torch.load.html>.
3. **`map_location` controls the destination device.** A checkpoint saved on GPU loads on CPU only if you pass `torch.load("...", map_location="cpu", weights_only=True)`. Without it, `torch.load` tries to materialize the tensors on the same device they were saved from, which fails if that device is not present.
4. **Do not `torch.save(model, ...)`.** The pickle-based save-the-whole-object path works in narrow scenarios (same PyTorch version, same file layout, same class definitions). It is fragile. The `state_dict` path is the production standard.

Reference: <https://pytorch.org/tutorials/beginner/saving_loading_models.html>.

### Checkpointing during training

In addition to the final save, it is good practice to save the model every few epochs and on best-validation hits:

```python
best_val_acc = 0.0
for epoch in range(n_epochs):
    # ... training loop ...
    train_loss = ...
    val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best.pt")
```

This is the "save the best checkpoint" pattern. For the mini-project it is good enough to checkpoint only at the end; for any nontrivial training run, save the best as you go.

For full reproducibility, you may also want to save the optimizer state dict (`optimizer.state_dict()`), the epoch number, and the random seed — bundled into a single dict and saved with one `torch.save`. The pattern:

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "rng_state": torch.get_rng_state(),
    "best_val_acc": best_val_acc,
}, "checkpoint.pt")
```

To resume training from this checkpoint, load and restore all five entries. The C5 mini-project rubric awards bonus credit for a working resume-from-checkpoint capability.

---

## 8. The three more training failures (continued from Lecture 2)

Lecture 2 covered the three classic in-loop failures. Here are three more that show up once you have a DataLoader and a device.

### Failure 4 — Loss is fine but `test_acc` is stuck at 10%

*Symptom:* training loss decreases as expected, but accuracy on the test loader is around 10% (random chance on 10 classes), epoch after epoch.

*Cause (almost always):* the test loader is shuffling. With `shuffle=True` on a paired-tensor dataset, the labels can desynchronize from the inputs across worker processes. Or the test transform differs from the train transform (e.g., you forgot to normalize). Or the model is on `cuda` and you forgot to move the model after `.load_state_dict`.

*Fix:* `shuffle=False` on the test loader; identical `transform` on both; `model.to(device)` immediately after construction.

### Failure 5 — DataLoader hangs at the start of the first epoch

*Symptom:* the training script prints "epoch 0..." and sits there indefinitely. CPU usage is high; nothing is printed.

*Cause:* on macOS or Windows, `num_workers>0` requires the script to be guarded by `if __name__ == "__main__":`. Without it, the worker subprocesses re-import the script, which re-runs the training loop, which spawns more workers — and the system fork-bombs itself.

*Fix:* wrap the entry point. The full pattern is in the script in Section 6 above.

### Failure 6 — GPU memory exhaustion (`CUDA out of memory`)

*Symptom:* training runs for one or two epochs and then dies with `RuntimeError: CUDA out of memory. Tried to allocate ...`.

*Causes:*

1. **Batch size too large.** *Fix:* halve it. Most models work with `batch_size=32-128`.
2. **Accumulating tensors with `requires_grad=True` across iterations.** Storing `loss` (not `loss.item()`) in a list keeps the entire autograd graph alive. *Fix:* always store `.item()` values, never tensors.
3. **A leaking validation loop.** Forgetting `with torch.no_grad():` around evaluation builds a full backward graph for the validation pass, doubling memory. *Fix:* the `no_grad` context.
4. **Genuinely too large a model for the GPU.** The 8 GB GPU in a laptop cannot train a 5B-parameter model; the 80 GB A100 can. *Fix:* use a smaller model, use gradient accumulation, or train on a beefier GPU.

The C5 rule: if you see OOM, lower the batch size first; that is the fix in 90% of cases.

---

## 9. The complete debugging checklist

Tape this to your monitor for Thursday's mini-project session.

| Symptom | First check | Reference |
|---------|-------------|-----------|
| `loss = nan` | Lower the learning rate | Lecture 2, Section 9 |
| `loss` does not decrease | `optimizer.zero_grad()` is missing | Lecture 2, Section 9 |
| `loss` decreases, accuracy is 10% | Shuffle and transform on test loader | Section 8, Failure 4 |
| `RuntimeError: ... device` | `X.to(device)` on every batch | Section 5 |
| `CUDA out of memory` | Halve the batch size | Section 8, Failure 6 |
| Loader hangs (macOS / Windows) | `if __name__ == "__main__":` guard | Section 8, Failure 5 |
| `RuntimeError: leaf var ... in-place` | Use `torch.no_grad()` around manual updates | Lecture 1, Section 10 |
| `RuntimeError: backward through graph second time` | Re-run forward; you cannot reuse the graph | Lecture 1, Section 6 |
| Loss decreases then NaN at epoch 50 | Gradient explosion; lower lr or clip gradients | Lecture 2, Section 9 |
| `model.parameters()` returns empty | Missing `super().__init__()` in your Module | Lecture 2, Section 1 |

The list covers ~95% of Week 8 debugging. The remaining 5% is genuinely new and is what you ask a teaching assistant about.

---

## 10. What you can build now

By the end of Lecture 3 you have the full bare-PyTorch toolkit:

- **Model definition** — `nn.Module` subclass with `__init__` and `forward`. Lecture 2 Section 1.
- **Loss function** — `nn.CrossEntropyLoss` for classification. Lecture 2 Section 6.
- **Optimizer** — `optim.SGD` or `optim.Adam`. Lecture 2 Section 5.
- **Data pipeline** — `Dataset` and `DataLoader`. Section 1-2.
- **Real benchmark** — `torchvision.datasets.FashionMNIST` (or CIFAR-10). Section 3-4.
- **Device support** — `cuda`, `mps`, `cpu` detection and `.to(device)`. Section 5.
- **Checkpointing** — `state_dict` save / load with `weights_only=True`. Section 7.
- **Debugging vocabulary** — the nine common failures and their fixes. Section 8-9.

That toolkit is enough to ship the mini-project: an MLP on FashionMNIST, trained end to end, checkpointed to disk, reloaded and evaluated by a separate script. It is also the toolkit you will use for the next ten weeks of the curriculum, with the model class swapped out for CNNs (Week 9), recurrent layers (Week 12), transformers (Week 13), and so on. The plumbing does not change.

---

## 11. What's next

Exercise 3 is the first full FashionMNIST training run. Read the assertions before you start; they pin down the expected test accuracy and the expected save/load behavior.

Challenge 1 quantifies the MLP-vs-CNN gap on CIFAR-10. It is preparation for Week 9.

Challenge 2 is the "write your own `Dataset`" drill — wrap a folder of JPG files into a PyTorch dataset with on-the-fly transforms. The skill transfers to every real-world image task you will see.

The mini-project is the full FashionMNIST classifier with reload-and-evaluate, training-curve plot, and a 600-word report.

By Sunday you should have a portfolio repo with a working `state_dict.pt`, a `train.py` that produces it, an `evaluate.py` that loads it, and a `report.md` that reads honestly. That is the Week 8 deliverable.

Reference reading for Lecture 3:

- The official "Datasets and DataLoaders" tutorial: <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>
- The official "Save and Load the Model" tutorial: <https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html>
- The `torch.utils.data` reference: <https://pytorch.org/docs/stable/data.html>
- The FashionMNIST class reference: <https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html>
