# Week 8 Exercise Solutions

Do not read this file until you have attempted each exercise for at least an hour. The exercises are designed so that the value comes from the attempt, not from the answer.

The solutions are written exactly as a strong C5 student should write them. Production code would add docstrings, logging, and a few more guards; the exercises are deliberately minimal.

---

## Exercise 1 — Tensors and Autograd

The headline test is `test_pytorch_backward_matches_numpy_to_1e_5` — your PyTorch autograd must produce the same gradients as the Week 7 NumPy backward to within `1e-5` relative error. If that test passes, every other test in the file is incidental.

### Part A — tensor creation

```python
def make_tensor_from_list() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0])


def make_tensor_from_numpy(arr: NDArray[np.float32]) -> torch.Tensor:
    return torch.from_numpy(arr)


def make_zeros(shape: Tuple[int, ...]) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.float32)


def make_randn(shape: Tuple[int, ...], seed: int = RANDOM_STATE) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)
```

**Commentary.** The `torch.tensor` vs. `torch.from_numpy` distinction (Lecture 1 Section 1) is the easy point — `torch.tensor` copies; `torch.from_numpy` aliases. The test `test_tensor_from_numpy_shares_memory` is the proof.

For `make_randn`, the deterministic seed is the C5 convention: every randomly-initialized object in C5 takes a `seed` keyword and uses `torch.Generator().manual_seed(seed)` rather than the global `torch.manual_seed`. The reason: tests should be independent. If two tests share global RNG state, the order of test execution matters and that is a brittleness we avoid.

### Part B — autograd on a scalar

```python
def grad_of_scalar(x_value: float) -> float:
    x = torch.tensor(x_value, requires_grad=True)
    y = torch.sin(x) * x ** 2
    y.backward()
    return float(x.grad.item())
```

**Commentary.** Three things to notice:

1. `torch.tensor(x_value, requires_grad=True)` produces a leaf tensor. `x.requires_grad` is `True`; `x.is_leaf` is `True`; `x.grad` is `None` until `backward` runs.
2. `y` inherits the requires_grad flag because it was computed from `x`. `y.is_leaf` is `False`; `y.grad_fn` points to the multiplication op.
3. After `y.backward()`, `x.grad` holds `dy/dx` at the value of `x`. The analytical answer at `x=3` is `2*3*sin(3) + 9*cos(3) = 0.8467 - 8.9100 = -8.0633`.

### Part C — He init and forward

```python
def init_params_pytorch(
    n_in: int, n_hidden: int, n_out: int, seed: int = RANDOM_STATE,
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(n_in, n_hidden, generator=g) * (2.0 / n_in) ** 0.5).requires_grad_(True)
    b1 = torch.zeros(n_hidden, requires_grad=True)
    W2 = (torch.randn(n_hidden, n_out, generator=g) * (2.0 / n_hidden) ** 0.5).requires_grad_(True)
    b2 = torch.zeros(n_out, requires_grad=True)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_raw(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = torch.relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    return Z2
```

**Commentary.** The He scaling `sqrt(2 / fan_in)` is the same as Week 7. The trailing `_` on `requires_grad_(True)` is the in-place form (Lecture 1 Section 7); the alternative would be to pass `requires_grad=True` to a tensor constructor but you cannot easily do that *after* a multiplication, hence the in-place flip.

`b1` and `b2` use the eager `torch.zeros(..., requires_grad=True)` because there is no scalar multiplication needed. Both styles are fine.

`forward_raw` returns *raw logits* — Lecture 2 Section 6, the most common new-user bug is to apply softmax here and break `CrossEntropyLoss`. Do not.

### Part E — autograd backward

```python
def pytorch_backward(X, y, params):
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()
    logits = forward_raw(X, params)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    return {name: p.grad.clone() for name, p in params.items()}
```

**Commentary.** Three rules:

1. **Zero the grads before backward.** Otherwise, on the second call, the new gradient accumulates on top of the old one. Lecture 1 Section 10, Trap 1.
2. **Use `torch.nn.functional.cross_entropy`** instead of writing your own. It takes integer class indices (not one-hot), expects raw logits, and computes the mean by default.
3. **Return `.clone()`** of the gradients. The `.grad` attribute is itself a tensor that PyTorch may reuse; cloning produces an independent snapshot the caller can hold without surprise.

### Part F — manual SGD

```python
def manual_sgd_step(params, lr=0.1):
    with torch.no_grad():
        for p in params.values():
            p -= lr * p.grad
            p.grad.zero_()
```

**Commentary.** The `with torch.no_grad():` wrapper is essential — Lecture 1 Section 10 Trap 2. Without it, the in-place `-=` is recorded into the graph (because `p` has `requires_grad=True`), polluting the next backward.

The `p.grad.zero_()` at the end is the manual equivalent of what `optimizer.zero_grad()` does. Once you start using `torch.optim.SGD`, you will not write this loop again — but you should write it once, by hand, so the optimizer is no longer a black box.

---

## Exercise 2 — `nn.Module` and the Training Loop

### Part A — TwoLayerMLP

```python
class TwoLayerMLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))
```

**Commentary.**

- `super().__init__()` is mandatory. The most common cause of a model whose `parameters()` is empty is a missing `super` call.
- `self.fc1`, `self.relu`, `self.fc2` are the parameter names. They appear in `model.named_parameters()` as `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`. `nn.ReLU` has no parameters; it is a stateless module.
- `self.relu` could just as well be `torch.relu` called as a function inside `forward`. The Module form is preferred by the PyTorch style guide because it keeps the model's architecture inspectable via `print(model)`.

### Part C — training_step

```python
def training_step(model, X, y, loss_fn, optimizer):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    return float(loss.item())
```

**Commentary.** The five-line core. Note `loss.item()` — *not* `loss`. Returning the tensor would hold the autograd graph alive past the function boundary; calling code that accumulates losses in a list would leak memory over time.

### Part D — evaluate

```python
def evaluate(model, loader, loss_fn, device):
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
```

**Commentary.** The two non-obvious lines:

1. **`loss_fn(logits, y).item() * X.size(0)`** — we multiply by the batch size so that the running sum is a sum over examples, not a sum over batches. Dividing by `n` at the end gives a per-example mean. The reason: the final batch may be partial; treating each batch as a full one over-weights the partial.
2. **`(logits.argmax(1) == y).sum().item()`** — `argmax(1)` returns the class index per row; the equality test is element-wise; `.sum()` counts the True entries; `.item()` extracts a Python int.

### Part E — train

```python
def train(n_epochs, batch_size, lr, n_samples, n_features, n_classes, hidden, seed):
    torch.manual_seed(seed)
    device = best_device()
    X, y = make_learnable_dataset(n_samples, n_features, n_classes=n_classes, seed=seed)
    n_train = int(0.8 * len(X))
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
    model = TwoLayerMLP(n_features, hidden, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(n_epochs):
        model.train()
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            training_step(model, X_b, y_b, loss_fn, optimizer)
    _, acc = evaluate(model, te_loader, loss_fn, device)
    return model, acc
```

**Commentary.** Standard end-to-end script. The 80/20 split is the simplest possible — for synthetic data with a uniform distribution, a slice is fine. For real data, use `torch.utils.data.random_split` or sklearn's `train_test_split`.

---

## Exercise 3 — FashionMNIST End-to-End

### Part B — FashionMLP

```python
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
```

**Commentary.** Identical to the Lecture 3 Section 6 reference. The `nn.Flatten()` is the part new users sometimes forget; without it the `Linear(784, 128)` errors with a shape mismatch when handed a `(B, 1, 28, 28)` image batch.

### Part C — build_loaders

```python
def build_loaders(batch_size=128, root=DATA_ROOT, num_workers=0):
    from torchvision import datasets
    from torchvision.transforms import v2
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
    ])
    train_ds = datasets.FashionMNIST(root, train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
```

**Commentary.** The lazy `from torchvision import` is so that `python -m py_compile` works on a machine without torchvision installed. The tests *do* import it, which is expected — they require a working PyTorch + torchvision install.

The Normalize values `(0.286, 0.353)` are the channel statistics of the FashionMNIST training set, computed once. Using them is not strictly required for the MLP to learn, but normalization improves convergence and is the modern default; see the torchvision docs at <https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.Normalize.html>.

### Part E — train

```python
def train(n_epochs=5, batch_size=128, lr=1e-3, seed=RANDOM_STATE, verbose=True):
    torch.manual_seed(seed)
    device = best_device()
    train_loader, test_loader = build_loaders(batch_size=batch_size)
    model = FashionMLP(n_hidden=128).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    last_acc = 0.0
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
        test_loss, last_acc = evaluate(model, test_loader, loss_fn, device)
        if verbose:
            print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
                  f"test_loss {test_loss:.4f}  test_acc {last_acc:.4f}")
    return model, last_acc
```

**Commentary.** This is the full reference. Expected output on a typical run (CPU, 5 epochs):

```text
epoch   0  train_loss 0.5174  test_loss 0.4267  test_acc 0.8479
epoch   1  train_loss 0.3814  test_loss 0.3978  test_acc 0.8579
epoch   2  train_loss 0.3415  test_loss 0.3743  test_acc 0.8663
epoch   3  train_loss 0.3179  test_loss 0.3636  test_acc 0.8714
epoch   4  train_loss 0.3002  test_loss 0.3530  test_acc 0.8731
```

The final test accuracy is consistently in `[0.86, 0.89]` for this architecture with `random_state=42`. The `>=0.85` threshold has a margin.

### Part F — save_model and load_model

```python
def save_model(model, path=CHECKPOINT_PATH):
    torch.save(model.state_dict(), path)


def load_model(path=CHECKPOINT_PATH, n_hidden=128):
    model = FashionMLP(n_hidden=n_hidden)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(best_device()).eval()
    return model
```

**Commentary.** Two things matter:

1. **`weights_only=True`** — the PyTorch 2.4+ safe loader. Suppresses the UserWarning about untrusted pickle deserialization and restricts the deserializer to a tensor-only whitelist. PyTorch 2.6 plans to make this the default.
2. **`map_location="cpu"`** — load the tensors to CPU first, then move the model to the best device. This is the safe pattern for checkpoints that may have been saved on a different device than the one currently available; without it, loading a CUDA checkpoint on a CPU-only machine would error.

---

## Common bugs across all three exercises

The C5 cohort hits these every cycle.

| Bug | Symptom | Fix |
|-----|---------|-----|
| Forgot `optimizer.zero_grad()` | Loss explodes after a few iterations | Add the zero before each forward |
| Forgot `super().__init__()` | `model.parameters()` is empty | Add super call in `__init__` |
| Returned `loss` instead of `loss.item()` | OOM after many iterations | Always `.item()` when accumulating |
| Forgot `model.eval()` in evaluate | Test accuracy depends on dropout state | Always pair `train()` / `eval()` |
| Forgot `with torch.no_grad():` in evaluate | OOM during evaluation; slow | Wrap the eval loop |
| Applied softmax before CE loss | Slow convergence; never converges | Remove the softmax; CE does it |
| Used `tensor.numpy()` on `requires_grad` tensor | `RuntimeError: Can't call numpy() on Tensor that requires grad` | Use `.detach().numpy()` |
| Forgot to move batch to device | `Expected all tensors to be on the same device` | `X.to(device)` per batch |
| Did not use `weights_only=True` | UserWarning at load time | Pass the flag |
| Did not pass `map_location` | Crash loading GPU checkpoint on CPU | Pass `map_location="cpu"` |

If your solution does not pass a test, the most likely bug is in this table.
