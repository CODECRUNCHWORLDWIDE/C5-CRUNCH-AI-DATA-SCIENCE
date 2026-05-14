# Lecture 1 — Tensors and Autograd

> **Outcome:** You can create a `torch.Tensor` in five different ways and name when each one copies vs. shares memory. You can describe the autograd graph as a runtime-built DAG of operations. You can hand-derive the gradient of a scalar function of two tensors, then verify it against PyTorch's `.backward()`. You can explain — without hand-waving — why `loss.backward()` and the NumPy backward pass you wrote in Week 7 compute the same numbers. After this lecture, the framework is no longer a black box, just an automated one.

Week 7 ended with a model whose every gradient you derived from the chain rule. That model worked, reached 95% test accuracy on MNIST, and gave you the right kind of pride. It also took two days to write and would take another two days to extend to a third layer or a different activation.

Week 8 fixes both problems with one trade: you give up the manual derivation and let a framework do it. The framework is **autograd** — PyTorch's reverse-mode automatic-differentiation engine. The lecture is the technology behind that trade. By the end of it, you will know exactly what `loss.backward()` does, why it is correct, and where it can still go wrong.

We target **PyTorch 2.x** throughout. The official autograd documentation is at <https://pytorch.org/docs/stable/autograd.html> and the longer-form notes are at <https://pytorch.org/docs/stable/notes/autograd.html>. Pin both.

---

## 1. What is a tensor?

A `torch.Tensor` is a multi-dimensional array. It has:

- **A shape** — a tuple of dimensions. `tensor.shape` returns a `torch.Size`, which is a subclass of `tuple`.
- **A dtype** — the element type. Common dtypes: `torch.float32` (default), `torch.float64`, `torch.int64`, `torch.bool`. Reference: <https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype>.
- **A device** — where the memory lives. `torch.device("cpu")`, `torch.device("cuda")`, `torch.device("mps")`. Reference: <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>.
- **A `requires_grad` flag** — whether to track operations for the autograd graph. Default `False`.
- **A `.grad` attribute** — the accumulated gradient, populated by `.backward()`. Only meaningful on leaf tensors with `requires_grad=True`.

That is almost the entire interface. A NumPy array has the first two; PyTorch adds the next three.

Five ways to create a tensor:

```python
import torch
import numpy as np

# 1. From a Python list (always copies)
a = torch.tensor([1.0, 2.0, 3.0])

# 2. From a NumPy array, zero-copy view (shares memory; mutating one mutates the other)
arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = torch.from_numpy(arr)

# 3. Smart constructor (zero-copy when possible)
c = torch.as_tensor(arr)

# 4. Factory functions
d = torch.zeros(3, 4)
e = torch.ones(3, 4)
f = torch.arange(0, 10, dtype=torch.float32)
g = torch.linspace(0.0, 1.0, 100)

# 5. Random
rng = torch.Generator().manual_seed(42)
h = torch.randn(3, 4, generator=rng)
```

The `torch.tensor` vs. `torch.as_tensor` vs. `torch.from_numpy` distinction trips up everyone. The short version:

- `torch.tensor(data)` always copies. Safe but wasteful for large NumPy arrays.
- `torch.as_tensor(data)` copies when it has to (dtype/device mismatch) and aliases otherwise. The lazy default.
- `torch.from_numpy(arr)` is zero-copy and dtype-respecting; it can only consume a NumPy array, not a Python list.

Reference: <https://pytorch.org/docs/stable/torch.html#creation-ops>.

> **EXPERIMENT — observe the aliasing.** In a REPL: `arr = np.array([1.0, 2.0, 3.0]); t = torch.from_numpy(arr); t[0] = 99.0; print(arr)`. The NumPy array is mutated too — they share the buffer. Now do the same with `torch.tensor(arr)` and verify that the NumPy array is untouched. The bug this prevents: returning a `torch.from_numpy(...)` view of a temporary NumPy buffer that goes out of scope mid-training; you lose your data without a warning.

---

## 2. Shapes, broadcasting, and the basics

PyTorch's broadcasting rules are identical to NumPy's. Two tensors are broadcastable if, aligned from the right, every pair of dimensions is either equal, one of them is 1, or one of them is missing.

```python
A = torch.zeros(3, 4)
b = torch.ones(4)          # broadcasts to (3, 4)
C = A + b                  # shape (3, 4); every row is [1, 1, 1, 1]
```

Reference: <https://pytorch.org/docs/stable/notes/broadcasting.html>.

The shape-manipulation methods you will use this week:

- `tensor.reshape(*shape)` — view with a new shape, copies if not contiguous.
- `tensor.view(*shape)` — same, but errors instead of copying. Use when you know the tensor is contiguous; safer to use `reshape` unless performance matters.
- `tensor.unsqueeze(dim)` — insert a dimension of size 1 at `dim`. `t.unsqueeze(0)` turns a `(N,)` into a `(1, N)`.
- `tensor.squeeze(dim)` — remove a dimension of size 1 at `dim`. Inverse of `unsqueeze`.
- `tensor.permute(*dims)` — rearrange dimensions. `tensor.permute(2, 0, 1)` on a `(H, W, C)` image gives `(C, H, W)`, the PyTorch convention.
- `tensor.transpose(d1, d2)` — swap two dimensions. Special case of `permute`.

Matrix multiplication has three forms:

```python
torch.matmul(A, B)  # generic; broadcasts the batch dimensions
A @ B               # operator form; the Python 3.5+ syntax
torch.mm(A, B)      # restricted to 2-D inputs; faster signature but less flexible
```

For Week 8, use `@`. The `torch.matmul` API rules are explained at <https://pytorch.org/docs/stable/generated/torch.matmul.html> and they handle batched matmuls correctly (the batch dimension broadcasts).

---

## 3. The `.numpy()` and `.tolist()` escape hatches

Tensors interoperate with the rest of the Python data ecosystem:

```python
t = torch.tensor([1.0, 2.0, 3.0])

t.numpy()      # -> ndarray; shares memory if on CPU; errors if requires_grad
t.tolist()     # -> [1.0, 2.0, 3.0]; copies; works regardless of grad state
t.item()       # -> 1.0 (only works for single-element tensors)

t.detach().numpy()                # the safe pattern when the tensor has requires_grad
t.detach().cpu().numpy()          # the safe pattern when the tensor is on GPU
```

`.detach()` returns a new tensor that shares storage but is removed from the autograd graph. `.cpu()` is a no-op if the tensor is already on CPU.

The `.detach().cpu().numpy()` chain is the production-safe way to extract a tensor for logging, plotting, or saving. Memorize it; you will write it a hundred times this week.

---

## 4. Autograd, briefly

Autograd is PyTorch's reverse-mode automatic differentiation engine. The user-facing contract is small:

1. Create leaf tensors with `requires_grad=True`.
2. Compute a scalar quantity (the loss) using ops on those tensors.
3. Call `.backward()` on the scalar.
4. Read `.grad` on the leaves.

That is the entire API. The complications are in the details — what counts as a leaf, what `.backward()` does if you call it twice, why `.grad` is None until `.backward()` runs — and the lecture spends the next four sections on them.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1                # y = x^2 + 3x + 1
y.backward()                          # populate x.grad
print(x.grad)                         # tensor(7.) -- because dy/dx = 2x + 3, evaluated at x=2 is 7
```

Three things to notice:

1. `y` was *not* declared with `requires_grad=True`. It inherited it because it was computed from `x`. The flag propagates through every op.
2. `y.backward()` only works because `y` is a scalar. For a non-scalar `y`, you must pass a "gradient argument" to `backward` — but in practice we always call `.backward()` on the scalar loss, so this edge case rarely matters.
3. `x.grad` is `7.0`, not `tensor(7., grad_fn=...)`. It is itself a `requires_grad=False` tensor; the grad of a grad is not tracked unless you ask for it (the "double backward" use case; not relevant this week).

Reference: <https://pytorch.org/docs/stable/autograd.html>.

> **EXPERIMENT — compute a gradient by hand and verify it.** In a REPL: `x = torch.tensor(3.0, requires_grad=True); y = torch.sin(x) * x**2; y.backward(); print(x.grad)`. Now compute the derivative analytically: `dy/dx = 2x sin(x) + x^2 cos(x)`. At `x = 3`: `2 * 3 * sin(3) + 9 * cos(3) = 6 * 0.1411 + 9 * (-0.9900) = 0.8467 - 8.9100 = -8.0633`. The two numbers should agree to four decimal places. They do.

---

## 5. The computation graph

The graph autograd builds is a **directed acyclic graph** (DAG). Each node is one of:

- A **leaf** — a tensor that was *not* created from another op. Inputs and parameters are leaves.
- An **operation** — every PyTorch op (`+`, `*`, `matmul`, `sin`, `relu`, `linear`, etc.) creates an op node when executed on a `requires_grad=True` tensor.
- A **non-leaf intermediate** — a tensor that *was* created from an op. Its `.grad_fn` attribute points to the op that created it.

When you call `.backward()` on a leaf of the graph (typically a scalar loss `L`), autograd walks the graph in topological reverse, applying the **chain rule** at every op node, accumulating gradients into the leaves' `.grad` attributes.

Equivalently: every op in PyTorch has a "backward" function that takes the upstream gradient and returns the gradients to feed back. The `Function` class in autograd is the API; `torch.autograd.Function` is the documented base class for writing your own. We do not write one this week (PyTorch ships ~2,000 of them; we will use the existing ones), but it is useful to know they exist. Reference: <https://pytorch.org/docs/stable/autograd.html#function>.

The graph is **dynamic**: it is rebuilt on every forward pass. If you compute `y = x ** 2` once and `y = torch.sin(x)` the next iteration, the second backward pass uses the cosine, not `2x`. This is "define-by-run" or "eager mode," in contrast with TensorFlow 1.x's "define-then-run" static graphs. It makes debugging easier (Python stack traces are honest) at a small cost in optimization opportunity (the framework cannot reorder ops it has not seen yet).

Reference: <https://pytorch.org/docs/stable/notes/autograd.html>.

---

## 6. Leaves, intermediates, and `retain_graph`

The most common autograd confusion is the difference between a leaf and an intermediate:

- **Leaf** tensors: created with `torch.tensor(..., requires_grad=True)` or `torch.zeros(..., requires_grad=True)`. Parameters of a model. Have a `.grad` attribute that gets populated by `.backward()`. `.is_leaf` is `True`.
- **Intermediate** tensors: created as the output of an op on `requires_grad=True` inputs. Do *not* have `.grad` populated by default — the gradient is computed for them during backward but immediately consumed to compute the gradient for upstream leaves. `.is_leaf` is `False`.

If you actually want the gradient of an intermediate, you must opt in:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.retain_grad()           # opt in to retaining .grad on this intermediate
z = y + 1
z.backward()
print(y.grad)             # tensor(1.) -- dz/dy = 1
```

You will rarely want this. It is documented because students who do not know it spend an afternoon wondering why `.grad` is `None` on a non-leaf they thought should have it. The right read is: `.grad` lives on the leaves; everything else flows through.

A related issue: calling `.backward()` twice on the same graph errors with `RuntimeError: Trying to backward through the graph a second time`. The graph is freed after the first backward to save memory. If you really want to back-propagate twice (e.g., a double-backward for second-order methods), pass `retain_graph=True` to the first call.

For a standard training loop, you will never need either of these. The loop is: forward → backward (graph freed) → step → repeat (new forward builds a new graph).

---

## 7. `no_grad`, `detach`, and `requires_grad_`

Three ways to opt out of autograd:

```python
with torch.no_grad():
    # everything inside this block runs with requires_grad implicitly False
    y_pred = model(X_val)
    val_loss = loss_fn(y_pred, y_val)
```

`torch.no_grad()` is a context manager that disables gradient tracking in its body. The forward pass is still computed; only the graph construction is skipped. Use it for **every evaluation pass**: no graph means no memory overhead and roughly 2x speed. Reference: <https://pytorch.org/docs/stable/generated/torch.no_grad.html>.

```python
y = model(X)
y_detached = y.detach()       # new tensor; shares storage; not in graph
```

`tensor.detach()` returns a tensor that shares the underlying storage but is removed from the autograd graph. Useful when you want the *value* of a tensor for logging or plotting but not the gradient flow. Reference: <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>.

```python
param = torch.zeros(10, requires_grad=True)
param.requires_grad_(False)    # in-place flip the flag
```

The trailing underscore is the in-place version (a PyTorch convention: `add_` is in-place addition, `mul_` is in-place multiplication, `requires_grad_` is in-place setting of the flag). Used when freezing parameters (transfer learning, Week 11).

The three patterns cover the entire "stop tracking gradients here" need.

---

## 8. From NumPy MLP to PyTorch autograd: the side-by-side translation

This is the part where Week 7 pays off. Take the two-layer MLP forward pass from Lecture 1 last week:

```python
# Week 7 -- pure NumPy
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```

The same forward pass in PyTorch with autograd, no `nn.Module` yet:

```python
# Week 8 -- raw PyTorch autograd, no nn.Module
import torch

def forward(X: torch.Tensor,
            W1: torch.Tensor, b1: torch.Tensor,
            W2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    Z1 = X @ W1 + b1
    A1 = torch.relu(Z1)
    Z2 = A1 @ W2 + b2
    return Z2  # raw logits; the loss function will apply softmax
```

Differences:

1. **No softmax in the forward.** We return the raw logits `Z2` and let `nn.CrossEntropyLoss` (Lecture 2) compute the log-softmax internally for numerical stability.
2. **No `cache` return value.** The autograd graph *is* the cache. PyTorch remembers `Z1` and `A1` automatically as part of the graph used by `.backward()`. You do not maintain it; the framework does.
3. **`torch.relu`** instead of `np.maximum(Z1, 0.0)`. Same operation, idiomatic name.
4. **No bias broadcasting question.** PyTorch broadcasts the bias along the batch dimension the same way NumPy does.

The backward pass:

```python
# Week 7 -- pure NumPy, ~30 lines
def backward(cache, Y, W1, W2):
    m = cache["X"].shape[0]
    dZ2 = (cache["A2"] - Y) / m
    dW2 = cache["A1"].T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (cache["Z1"] > 0)
    dW1 = cache["X"].T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

```python
# Week 8 -- one line
loss.backward()
```

That is the trade. You spent two days deriving the Week 7 version; you spend one second writing the Week 8 version. The framework computes the same numbers — verifiable to numerical precision. The next experiment is the side-by-side comparison.

> **EXPERIMENT — verify autograd against Week 7 NumPy.** Reuse the Week 7 `backward` function. Initialize the same `W1, b1, W2, b2` as both NumPy arrays and PyTorch tensors with `requires_grad=True`. On a 5-example batch, run the Week 7 forward + backward to get `dW1_numpy`. In PyTorch, run forward, compute loss, call `loss.backward()`, read `W1.grad`. Check `np.allclose(dW1_numpy, W1.grad.detach().numpy(), atol=1e-5)`. They agree. Exercise 1 walks this through; Exercise 1's `test_pytorch_grad_matches_numpy` is the assertion.

---

## 9. The optimizer step, the manual way

We will use `torch.optim.SGD` in Lecture 2. For one lecture, it helps to see what `optimizer.step()` does under the hood. The SGD update rule, written manually:

```python
import torch

# Setup
X = torch.randn(64, 784)
y = torch.randint(0, 10, (64,))
W1 = torch.randn(784, 128, requires_grad=True) * (2.0 / 784) ** 0.5
b1 = torch.zeros(128, requires_grad=True)
W2 = torch.randn(128, 10, requires_grad=True) * (2.0 / 128) ** 0.5
b2 = torch.zeros(10, requires_grad=True)
lr = 0.1

# One training step
Z2 = forward(X, W1, b1, W2, b2)
loss = torch.nn.functional.cross_entropy(Z2, y)
loss.backward()

# The SGD update, written by hand
with torch.no_grad():
    for param in (W1, b1, W2, b2):
        param -= lr * param.grad
        param.grad.zero_()              # don't forget to clear the .grad
```

Four things to notice:

1. **The update is in a `no_grad` context.** Without it, the in-place modification of a `requires_grad=True` tensor would be recorded into the graph, polluting the next backward. The `no_grad` block tells autograd "do not track these ops."
2. **`param -= lr * param.grad`** is in-place. The `-=` operator on a tensor calls `tensor.sub_(other)` under the hood. Using `param = param - lr * param.grad` would *replace* the variable's binding without modifying the underlying parameter, and the model would never update.
3. **`param.grad.zero_()`** clears the gradient after the step. If you skip this, the next `backward()` will *accumulate* into `param.grad`, doubling the effective update. This is the bug that takes everyone an hour the first time it happens — your loss spirals and you wonder why; the fix is one line.
4. **The combined pattern — backward, step, zero — is what `optimizer.step()` and `optimizer.zero_grad()` automate.** Lecture 2 introduces both. Once you have them, you will never write this loop by hand again.

The manual version is two minutes of code that buys you the right mental model for `optim.SGD`. Run it. Verify the loss decreases.

---

## 10. The autograd traps

A short list of bugs the C5 cohort hits regularly. Read once; reread on Thursday when your loss is `NaN`.

1. **Forgetting `optimizer.zero_grad()`.** Gradients accumulate by default. After two backward passes without zeroing, the update is twice what you intended; after ten, the gradient norm is roughly ten times larger and the loss explodes. *Symptom:* loss goes up, fast. *Fix:* call `.zero_grad()` at the start of every step.
2. **In-place ops on a `requires_grad=True` tensor outside `no_grad`.** Modifying a parameter directly (e.g., `W -= lr * W.grad`) without the `no_grad` wrapper records the subtraction into the graph and corrupts the next backward. *Symptom:* `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` or, worse, silent wrong gradients. *Fix:* wrap manual updates in `with torch.no_grad():` or, better, use an `optim` object.
3. **Returning logits but computing softmax in `nn.CrossEntropyLoss`.** `CrossEntropyLoss` expects raw logits and computes `log_softmax` internally for numerical stability. If you `softmax(logits)` first and pass *probabilities* to `CrossEntropyLoss`, the math is wrong (you are taking `log_softmax` of a softmax). *Symptom:* very slow learning or never converges. *Fix:* delete the `softmax` from the forward pass; let the loss function do it.
4. **Mixing `float32` and `float64` tensors.** PyTorch defaults to `float32`; NumPy defaults to `float64`. Multiplying them raises `RuntimeError: expected scalar type Float but found Double`. *Fix:* cast one with `tensor.float()` or `tensor.double()`.
5. **Calling `.backward()` on a non-scalar.** If `loss` is `(batch,)` instead of `()`, PyTorch demands a gradient argument the same shape. *Symptom:* `RuntimeError: grad can be implicitly created only for scalar outputs`. *Fix:* reduce the loss with `.mean()` (or `.sum()` if you want unaveraged gradients).
6. **Calling `.backward()` twice on the same graph.** The graph is freed after the first call. *Symptom:* `RuntimeError: Trying to backward through the graph a second time, but the saved intermediate results have already been freed`. *Fix:* unless you are doing something exotic (double backward, gradient penalty), you should not be calling backward twice. The fix is usually "re-run the forward pass first."
7. **`Tensor.grad` is `None` on a non-leaf.** As discussed in Section 6: only leaves accumulate `.grad`. *Fix:* call `.retain_grad()` on the intermediate before backward, or check that you are inspecting the right tensor.
8. **The CPU-GPU device mismatch.** A model on `cuda` and a batch on `cpu` errors with `Expected all tensors to be on the same device`. *Fix:* `X = X.to(device)` on every batch; covered in Lecture 3.

Take screenshots. The week's debugging will fit in one of these eight buckets, almost without exception.

---

## 11. What the framework actually does for you

The honest summary of what autograd buys you over Week 7's NumPy:

- **Mechanized chain rule.** You no longer derive `dW1 = X.T @ dZ1`; the framework does, correctly, for every op in its library.
- **Architecture flexibility.** Adding a layer in PyTorch is one line: `nn.Linear(128, 64)` and a `forward` update. In NumPy it is a new derivation, a new shape table, and a new gradient check.
- **GPU acceleration.** `model.to("cuda")` and `X.to("cuda")` is the entire ceremony. Lecture 3.
- **Compatibility with the rest of the ecosystem.** `torchvision` for datasets, `torchmetrics` for metrics, `wandb` for logging, Hugging Face Hub for pre-trained models. None of those work with a NumPy MLP.
- **Stability.** PyTorch has shipped numerically stable implementations of softmax, log-softmax, cross-entropy, layer-norm, etc. — implementations you would otherwise have to write yourself.

The honest summary of what autograd does *not* buy you:

- **Understanding.** A student who only learns PyTorch will pass this week. A student who only learns PyTorch will, in three months, hit a NaN loss and have no idea where to start because they have never written the backward pass. Week 7 is the inoculation.
- **Magical correctness.** Autograd computes the chain rule on the graph you built. If your *forward* pass is buggy — wrong shape, wrong activation, wrong loss — autograd will faithfully compute the gradient of the wrong thing.
- **Free numerical stability.** If you write `loss = -torch.log(softmax(logits)[range(m), y]).mean()` instead of using `CrossEntropyLoss`, you reintroduce the overflow bug Week 7 spent half a lecture on. The framework gives you the option to be stable; you must take it.

The trade is real, and the framework wins. Use it. Use it knowing what it is doing for you.

---

## 12. What's next

Lecture 2 introduces `nn.Module` (the way to package the forward pass into a class), `torch.optim` (the way to package the parameter update), and the **eight-line training loop** that is the architecture of every PyTorch training script you will ever read.

Lecture 3 introduces `Dataset` and `DataLoader` (the data plumbing), device placement (`.to(device)`), and the `state_dict` save / load pattern.

Exercise 1 walks through the autograd material in this lecture with a side-by-side NumPy / PyTorch gradient check on a two-layer MLP. If you do nothing else this week, do Exercise 1. The check at the end — Week 7's manual gradient agreeing with PyTorch's autograd to within `1e-6` — is the moment the framework stops being a black box and starts being a tool.

Reference reading for Lecture 1: <https://pytorch.org/docs/stable/notes/autograd.html> (the autograd mechanics note; thirty minutes; cover-to-cover).
