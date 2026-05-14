# Lecture 2 — `nn.Module`, Optimizers, and the Training Loop

> **Outcome:** You can subclass `nn.Module` to define a model, build a two-layer MLP in five lines, attach a `torch.optim` optimizer to its parameters, and write the eight-line training loop from memory. You can explain why `optimizer.zero_grad()` exists, why `loss.backward()` is the line that does the work, and why `optimizer.step()` is in-place. You can read every PyTorch training script on GitHub without surprise.

Lecture 1 gave you tensors and autograd. You can already write a complete MLP forward pass in PyTorch, run `loss.backward()`, and update parameters with a manual SGD loop. That is the "raw autograd" mode. It works; it is not how anyone writes production PyTorch.

This lecture is the **idiomatic mode**. The forward pass moves from a function into a class. The parameter update moves from a `for` loop into an `optim` object. The result is the same numerics with a much cleaner script — and, crucially, the same shape that every PyTorch codebase from `nanoGPT` to `torchvision/models/resnet.py` uses. By the end of this lecture you can read any of those files without surprise.

We target **PyTorch 2.x**. The official `nn.Module` reference is at <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>; the `torch.optim` reference is at <https://pytorch.org/docs/stable/optim.html>.

---

## 1. The model, as a class

A PyTorch model is a subclass of `torch.nn.Module`. The pattern has three parts:

```python
import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        return z2                            # raw logits
```

Three rules to remember and one trick:

1. **`super().__init__()` is mandatory.** It registers the module's internal book-keeping (submodules, parameters, hooks). Forgetting it causes a quiet failure where `model.parameters()` returns an empty iterator.
2. **Submodules are assigned to `self.<name>` in `__init__`.** PyTorch detects them via `__setattr__` magic and registers them under the names you used. `model.fc1` is a submodule; `model.fc1.weight` is its parameter; `model.parameters()` iterates over both.
3. **`forward(self, x)` defines the forward pass.** You never call `model.forward(x)` directly — you call `model(x)`, which invokes `__call__`, which runs hooks and then `forward`. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>.
4. **The trick: `nn.Linear` already manages its own `(weight, bias)`.** You do not initialize them; PyTorch does, with a sensible default (Kaiming uniform for ReLU networks, which is essentially the He initialization from Week 7). Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>. The line `self.fc1 = nn.Linear(784, 128)` is the equivalent of Week 7's `W1 = rng.normal(0, sqrt(2/784), (784, 128)); b1 = np.zeros(128)`.

The model is now a stateful object. `model.parameters()` returns an iterator over every trainable tensor; `model.state_dict()` returns a dict mapping names to tensors. We will use both in Section 4.

> **EXPERIMENT — enumerate the parameters.** In a REPL: `m = TwoLayerMLP(784, 128, 10); for name, p in m.named_parameters(): print(name, tuple(p.shape), p.numel())`. You should see four entries: `fc1.weight (128, 784) 100352`, `fc1.bias (128,) 128`, `fc2.weight (10, 128) 1280`, `fc2.bias (10,) 10`. Total: 101,770 parameters. Same as Week 7's `h=128` MLP. The numbers do not change; only the bookkeeping does.

---

## 2. `nn.Linear` and the weight-matrix convention

A note on a subtle convention difference between Week 7 and PyTorch.

In Week 7 NumPy, you wrote `Z = X @ W + b` where `W` had shape `(n_in, n_out)`. The matrix-vector product was `(batch, n_in) @ (n_in, n_out) = (batch, n_out)`. The bias broadcast over the batch dimension.

In PyTorch, `nn.Linear(in_features, out_features)` stores `weight` with shape `(out_features, in_features)` and `bias` with shape `(out_features,)`. The forward is `output = input @ weight.T + bias`. The transposed storage is a deliberate choice — it matches the math convention `y = Wx + b` more closely and it integrates cleanly with the C++ backend.

The implication: if you ever inspect `model.fc1.weight.shape`, it will be `(128, 784)`, not `(784, 128)`. Confusing if you are coming straight from Week 7; standard everywhere else. The forward pass `model(X)` handles the transpose automatically; you almost never have to think about it.

Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>.

---

## 3. `nn.Sequential` — the lightweight alternative

For models that are just a stack of layers in order, the subclass form above is overkill. PyTorch provides `nn.Sequential`:

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

The same two-layer MLP, in one expression. `nn.Sequential` is itself an `nn.Module` subclass; `model.parameters()`, `model.state_dict()`, and `model(x)` all work exactly as before.

The trade: `nn.Sequential` cannot express anything more complex than a linear chain. If your forward pass has a skip connection, a branching path, or a conditional, you need the subclass form. For Week 8's MLPs, `nn.Sequential` is the right choice. For Week 9's CNNs with skip connections, the subclass form returns.

Use `nn.Sequential` for prototyping and read-only models. Use the subclass form for anything you intend to extend.

Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>.

---

## 4. Parameters, buffers, and `state_dict`

A module's tensors come in two flavors:

- **Parameters** — tensors that should be trained. Created via `nn.Parameter(tensor)` or, more commonly, via standard layers like `nn.Linear` that wrap their tensors in `nn.Parameter` automatically. Returned by `model.parameters()`. Have `requires_grad=True` by default. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>.
- **Buffers** — tensors that are part of the model's state but not trained. The classic example is `BatchNorm`'s running mean and variance. Registered via `self.register_buffer("name", tensor)`. Saved in `state_dict` alongside parameters but excluded from `parameters()`. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer>.

For Week 8's MLP, you have only parameters. For Week 9's CNN with `nn.BatchNorm2d`, you will see buffers.

The `state_dict` is the single dict that captures both. It is a Python `OrderedDict` mapping flattened names ("fc1.weight", "fc1.bias", "bn1.running_mean", ...) to tensors. Two operations matter:

```python
torch.save(model.state_dict(), "model.pt")               # write to disk
model.load_state_dict(torch.load("model.pt",             # read from disk
                                 weights_only=True))
```

The `weights_only=True` flag (PyTorch 2.4+) restricts the deserializer to a safe whitelist of tensor types. It avoids the pickle attack surface. Always pass it when loading `state_dict`s you saved. Reference: <https://pytorch.org/docs/stable/generated/torch.load.html>.

Three things to notice:

1. **You do not save the *model class*, only the *state*.** When you load, you must first construct an instance of the same class (`m = TwoLayerMLP(784, 128, 10)`) and *then* call `load_state_dict`. The class definition lives in your code, not in the checkpoint.
2. **The flattened names must match.** If you renamed `self.fc1` to `self.linear_in` between training and loading, `load_state_dict` will error. (You can pass `strict=False` to ignore mismatches, but the right move is to keep the names stable.)
3. **The format is portable.** A `state_dict.pt` file saved in Linux loads correctly on macOS and vice versa. Same for CPU-saved files loaded on GPU: pass `map_location` to `torch.load` to choose the destination device.

This is the production checkpoint format. Use it everywhere.

---

## 5. The optimizer

A `torch.optim` optimizer takes a list of parameters and a learning rate, and provides two methods: `zero_grad()` and `step()`. The full API has more (state dict for the optimizer, parameter groups, scheduler hooks), but the two-method core is what you write 99% of the time.

The two optimizers you need this week:

### SGD — `torch.optim.SGD`

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
```

The same SGD you wrote by hand in Lecture 1 Section 9. With `momentum=0.0` (the default), it is plain mini-batch SGD: `param -= lr * param.grad` for each parameter.

With `momentum>0`, the update is a moving average:

```text
v_t = momentum * v_{t-1} + grad
param -= lr * v_t
```

The classical choice is `momentum=0.9`. It smooths the gradient estimate; in practice, SGD-with-momentum reaches a given loss in fewer epochs than vanilla SGD. We will not use momentum on the FashionMNIST MLP because it works without — but for the CIFAR-10 stretch goal, `SGD(lr=0.01, momentum=0.9)` is the right starting point.

Reference: <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>.

### Adam — `torch.optim.Adam`

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Adam is the modern default. It is **adaptive** — it maintains a per-parameter learning rate via running estimates of the first and second moments of the gradient. Kingma and Ba's 2015 paper is the reference; PyTorch's implementation is at <https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py>.

The C5 rule of thumb for 2026:

- **SGD with `lr=0.01`** is the right choice when (a) you have hand-tuned learning rate schedules from prior work, (b) you are training a small model (under 10M parameters) and want maximum control, or (c) you are reproducing a published result that used SGD.
- **Adam with `lr=1e-3` to `lr=3e-4`** is the right choice when you want a default that works on most problems with minimal tuning. Use this for the FashionMNIST mini-project.

Reference: <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>.

> **EXPERIMENT — Adam vs. SGD on FashionMNIST.** After Exercise 3 trains an MLP with SGD, change one line — `optim.SGD` to `optim.Adam`, `lr=0.01` to `lr=1e-3` — and retrain. The Adam version should reach the same test accuracy in roughly half the epochs. The lesson is the gap in convenience: Adam picks up most of the slack of a missing learning-rate schedule. The lesson is *not* that Adam is "better" — the SOTA on most benchmarks is still SGD with carefully tuned momentum, weight decay, and schedule.

---

## 6. The loss function

PyTorch ships every common loss function in `torch.nn`. The two you need this week:

### `nn.CrossEntropyLoss` — for multi-class classification

```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
```

- **Input:** raw logits of shape `(batch, num_classes)`. **Do not apply softmax first.**
- **Target:** integer class indices of shape `(batch,)`, dtype `torch.int64`.
- **Output:** scalar mean loss.

Internally, `CrossEntropyLoss` computes `log_softmax` and the negative-log-likelihood in a single numerically-stable op. This is the Week 7 "subtract the max" trick, productionized. If you apply `softmax` yourself first, you reintroduce the overflow bug.

Reference: <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>.

### `nn.MSELoss` — for regression

```python
loss_fn = nn.MSELoss()
loss = loss_fn(predictions, targets)
```

For an image classifier, you want `CrossEntropyLoss`. `MSELoss` returns for the Week 4-style regression problems that occasionally appear in homework.

Reference: <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>.

Both losses are themselves `nn.Module` instances and can be moved with `.to(device)`. For stateless losses like these the device move is a no-op but the API is uniform.

---

## 7. The eight-line training loop

Everything in Sections 1-6 assembles into the canonical PyTorch training loop. Read it once; memorize it; write it for the next ten years.

```python
import torch
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoLayerMLP(784, 128, 10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    model.train()                                    # set train mode
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)            # move batch
        optimizer.zero_grad()                        # clear gradients
        logits = model(X)                            # forward
        loss = loss_fn(logits, y)                    # compute loss
        loss.backward()                              # backward
        optimizer.step()                             # update weights
```

The eight lines inside the `for epoch` loop are the loop. Every PyTorch training script in the world is a variant of this loop with bookkeeping added (logging, checkpointing, learning-rate scheduling, validation). The core is invariant.

Three things every line earns its place:

1. **`model.train()`** sets the model into training mode. For an MLP this is a no-op, but for any model containing `nn.Dropout` or `nn.BatchNorm` it is essential — dropout zeroes activations only in training mode; batchnorm uses batch statistics in training and running statistics in eval. Get into the habit of calling it; you will save yourself a debugging hour the first time you add a dropout layer and forget.
2. **`X.to(device)` and `y.to(device)`** move the batch to the same device as the model. The `.to(device)` call is cheap when source and destination match (a no-op) and a memcpy when they differ. The most common GPU bug is forgetting one of these moves; the error message is "Expected all tensors to be on the same device" and it is unambiguous.
3. **`optimizer.zero_grad()` *before* `loss.backward()`.** The order matters: if you zero *after* backward, you clear the gradients you just computed and the step does nothing. Mnemonic: "Zero, then learn, then step." Reference: <https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer.zero_grad>.

The C5 convention is to write the loop *exactly* this way for the first month of using PyTorch. After you have written it twenty times you may add bookkeeping (TQDM progress bar, periodic logging, gradient-norm tracking), but the eight-line core stays.

---

## 8. Validation, periodically

Training without validation is overfitting waiting to happen. The full per-epoch pattern includes a validation pass:

```python
for epoch in range(n_epochs):
    # ---- train ----
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X.size(0)
    train_loss /= len(train_loader.dataset)

    # ---- validate ----
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            val_loss += loss_fn(logits, y).item() * X.size(0)
            val_correct += (logits.argmax(1) == y).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)

    print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  "
          f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")
```

Five additions to the eight-line core:

1. **`model.eval()`** sets the model to evaluation mode. Inverse of `model.train()`. Disables dropout, uses running statistics in batchnorm.
2. **`with torch.no_grad():`** wraps the entire validation pass. Disables autograd tracking; halves memory; roughly doubles throughput. Mandatory for evaluation.
3. **`.item()`** extracts a Python `float` from a single-element tensor. Use it for loss accumulation; otherwise you accumulate tensor references and the autograd graph grows without bound until you OOM.
4. **`X.size(0)`** is the batch size of the current batch. We multiply by it because we want a *per-example* average loss, not a per-batch average (the final batch may be partial).
5. **The print statement** is the entire logging system. Add `wandb.log({...})` or a `csv.writer` if you want persistence, but the print is the minimum.

For Week 8's exercises, this is the loop you write. For the mini-project, you wrap the loop in a `train(...)` function and add a periodic checkpoint save. The structure does not change.

> **EXPERIMENT — measure the speedup of `no_grad`.** On a `(1024, 784)` batch with a `(784, 128, 10)` MLP, time one forward pass with and without `with torch.no_grad():`. The no_grad version should be roughly 1.5-2x faster on CPU and use about half the memory. The reason: no graph to build, no intermediate tensor retention. The same speedup applies to your validation loop; on a 60k-example validation set the saving is a non-trivial 30 seconds per epoch.

---

## 9. The three common training failures

Three failure modes you will hit; recognize each by its symptom; the fix is mechanical once you know which one it is.

### Failure 1 — Loss is NaN

*Symptom:* loss prints as `nan` after a few iterations. Sometimes from iteration 0; sometimes after a few hundred steps.

*Causes (in order of likelihood):*

1. **Learning rate too high.** A 10x-too-large learning rate produces gradient explosions that overflow to `inf`, then `inf * 0 = nan`. *Fix:* lower the learning rate by 10x; if SGD, try `lr=0.001` instead of `lr=0.01`.
2. **Log of zero in a manually computed loss.** If you wrote `loss = -torch.log(p[y])` and any `p` is zero, the log is `-inf`. *Fix:* use `nn.CrossEntropyLoss`, which clamps internally; or use `torch.log(p.clamp_min(1e-12))`.
3. **Bad input data.** A NaN in the input tensor propagates through the network. *Fix:* `assert torch.isfinite(X).all()` at the top of each batch step.

### Failure 2 — Loss does not decrease

*Symptom:* loss starts at `log(num_classes)` (the Week 7 sanity check) and stays there, epoch after epoch.

*Causes:*

1. **`optimizer.zero_grad()` is missing.** Without it, gradients accumulate and effectively cancel after enough steps. *Fix:* add it at the start of the inner loop.
2. **Wrong learning rate (too low).** A 1000x-too-small `lr` makes updates so tiny the loss never moves. *Fix:* try `lr=1e-3` for Adam, `lr=0.1` for SGD.
3. **Device mismatch.** The model is on `cuda`, the data is on `cpu`. PyTorch errors loudly here, but if you somehow have the model on CPU and forgot to move a parameter, the model is silently not training the un-moved part. *Fix:* `model.to(device)` once, `print(next(model.parameters()).device)` to verify.
4. **`loss.backward()` missing.** Easy to delete by accident during refactoring. *Fix:* add it.
5. **Forward pass returning the same thing every time.** If you accidentally hard-coded the forward to return a constant, the loss is constant. *Fix:* print `logits[0, :5]` for two different batches; verify they differ.

### Failure 3 — Validation accuracy decreases while training accuracy rises

*Symptom:* train accuracy goes up monotonically, validation accuracy peaks and then falls. The textbook overfitting signature.

*Causes:*

1. **Model is overfitting.** Too many parameters for the dataset, no regularization. *Fix:* add dropout, add weight decay, reduce model size, get more data, or apply early stopping (stop training at the validation peak; load that checkpoint for inference).
2. **Train-test leakage.** Same data in both splits (rare; check your splits). *Fix:* `random_state` everywhere; `assert len(set(train_ids) & set(val_ids)) == 0`.
3. **Validation set too small.** A 100-example validation set has very noisy accuracy. *Fix:* use a 5,000+ example validation set; for FashionMNIST, the 10k-example test set is fine.

The three failures cover roughly 90% of "my training is broken." The list is on the cheat sheet at the end of the lecture.

---

## 10. A complete script

Putting Sections 1-9 together, the entire training script for an MLP on synthetic data — about 50 lines:

```python
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class TwoLayerMLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_dataset(n: int = 1000, seed: int = 42) -> TensorDataset:
    rng = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 784, generator=rng)
    y = torch.randint(0, 10, (n,), generator=rng)
    return TensorDataset(X, y)


def train(n_epochs: int = 5, batch_size: int = 64) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = make_synthetic_dataset(1000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TwoLayerMLP(784, 128, 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        print(f"epoch {epoch:3d}  loss {total_loss / len(train_ds):.4f}")


if __name__ == "__main__":
    train()
```

That is a complete, runnable PyTorch script. Forty-eight lines. Every line is from this lecture. No magic. No abstractions. Run it; it will print five lines of loss numbers. The loss will *not* decrease — the data is random, so there is no signal — but the script will *run*, the gradients will flow, the optimizer will step, and nothing will explode.

Exercise 2 is the same script with the synthetic dataset replaced by something learnable. Exercise 3 is the same script with FashionMNIST. Both reuse this skeleton verbatim.

---

## 11. What's next

Lecture 3 introduces:

- **`Dataset` and `DataLoader`** in depth: subclassing for custom data sources, the `num_workers` / `pin_memory` / `shuffle` / `drop_last` flags, and the `transform` pipeline.
- **`torchvision.datasets.FashionMNIST`** as the canonical example: where the data lives, what the transforms do, and how to plot a batch.
- **Device placement** in depth: detecting `cuda` vs. `mps` vs. `cpu`, the cost of CPU-GPU transfers, and when GPU is and is not worth it.
- **`state_dict` save / load** in production: best practices, the `weights_only` flag, restoring a model for inference.

Exercise 2 is the first time you build a model class from scratch. Exercise 3 puts the model class, the data loader, the optimizer, and the eight-line training loop together on FashionMNIST — the C5 Week 8 deliverable.

Reference reading for Lecture 2: <https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html> (the official "Build the Neural Network" tutorial; thirty minutes; goes through `nn.Module` with the same FashionMNIST example).
