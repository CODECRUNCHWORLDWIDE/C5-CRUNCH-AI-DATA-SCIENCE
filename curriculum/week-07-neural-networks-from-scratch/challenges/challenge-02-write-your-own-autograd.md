# Challenge 2 — Write Your Own Autograd

**Time estimate:** 3 hours.

## Problem statement

Build a tiny **reverse-mode automatic differentiation** engine over scalar values, modeled on Andrej Karpathy's `micrograd` (~100 lines, MIT license, free on GitHub at <https://github.com/karpathy/micrograd>). The engine should support `+`, `-`, `*`, `/`, `**`, `relu`, `tanh`, and `exp` — enough to differentiate any expression you would write in a hand-derived backward pass — and should compute exact gradients of arbitrary scalar functions via the chain rule applied to a computation graph.

Then use your autograd engine to train a tiny MLP (2 neurons in the hidden layer, 1 output) on the XOR dataset. Verify it converges. This is the simplest non-trivial neural network learnable by autograd.

The point of this challenge is to **understand what `loss.backward()` does** in PyTorch and JAX. The framework version is a 10,000-line implementation of the same idea; yours will be 100 lines, but the algorithm is identical.

## What you will produce

A single Python module `tinygrad.py` and a notebook or script `xor_demo.py` that:

1. Defines a `Value` class that wraps a single Python float, tracks its parents in a computation graph, and implements `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__`, and the activations `relu`, `tanh`, `exp`.
2. Each operation stores a `_backward` closure that knows how to propagate the gradient from the output of that operation to its inputs.
3. Implements a `Value.backward()` method that performs **topological sort** of the computation graph and then traverses it in reverse, calling each `_backward` to accumulate gradients.
4. Trains a 2-2-1 MLP on the XOR dataset (4 points; 2 features each; binary labels) using your autograd engine. The MLP should reach training loss `< 0.05` in 200 SGD steps with `lr = 0.1`.
5. Includes a `pytest` test that compares `tinygrad.Value` gradients against NumPy finite differences for a non-trivial expression like `f(x, y) = (x * y + relu(x - y)) ** 2`.

## Acceptance criteria

- [ ] `tinygrad.py` defines a `Value` class with `data: float` and `grad: float` attributes; `_children: set[Value]`; and the operations listed above.
- [ ] `Value.backward()` correctly accumulates gradients via reverse-mode autograd. The topological-sort-then-reverse-traversal is the standard implementation.
- [ ] `python -m py_compile tinygrad.py xor_demo.py` succeeds.
- [ ] The XOR MLP reaches training loss < 0.05 in 200 SGD steps.
- [ ] The gradient-check test against NumPy finite differences passes with relative error < `1e-5`.
- [ ] The code is under 200 lines total. (Karpathy's reference is 100 lines; ours has type hints, so allow some slack.)

## Hints

<details>
<summary>The Value class skeleton</summary>

```python
from __future__ import annotations

from typing import Callable, Union


Number = Union[int, float]


class Value:
    """A scalar value with autograd."""

    def __init__(
        self,
        data: float,
        _children: tuple["Value", ...] = (),
        _op: str = "",
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: set["Value"] = set(_children)
        self._op: str = _op

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    # __mul__, __pow__, __sub__, __truediv__ follow the same pattern.

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list["Value"] = []
        visited: set["Value"] = set()

        def build(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

The pattern repeats for every operation: define the forward, define the gradient propagation as a closure that uses `out.grad`, attach the closure to the output Value.
</details>

<details>
<summary>The chain rule, in autograd form</summary>

For `out = self + other`, the gradients are `dout/dself = 1` and `dout/dother = 1`. So `self.grad += 1 * out.grad` and `other.grad += 1 * out.grad`.

For `out = self * other`, the gradients are `dout/dself = other.data` and `dout/dother = self.data`. So `self.grad += other.data * out.grad` and `other.grad += self.data * out.grad`.

For `out = self ** k` (where `k` is a Python number, not a Value), `dout/dself = k * self.data ** (k - 1)`. So `self.grad += k * self.data ** (k - 1) * out.grad`.

For `out = relu(self)`, `dout/dself = 1 if self.data > 0 else 0`. So `self.grad += (out.data > 0) * out.grad`.

The pattern is always: the local gradient (output with respect to input) times the upstream gradient (`out.grad`).
</details>

<details>
<summary>Why the topological sort</summary>

If a value `x` is used multiple times in a computation — say, `out = x * x + relu(x)` — its gradient is the sum of the gradients from each use. Reverse-traversing in topological order guarantees that by the time we call `x._backward()`, every consumer of `x` has already contributed its `x.grad +=` term. Without topo sort, you might compute a partial gradient first and use it before the rest arrives.

The topo sort is implemented as a post-order DFS: visit all parents of a node before adding the node itself. The standard recursive implementation is in the skeleton above.
</details>

<details>
<summary>The XOR MLP</summary>

A 2-neuron hidden layer plus a 1-neuron output, with `tanh` activations:

```python
import random


class Neuron:
    def __init__(self, n_in: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x: list[Value]) -> Value:
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.tanh()


class MLP:
    def __init__(self, n_in: int, n_hidden: int, n_out: int) -> None:
        self.layer1 = [Neuron(n_in) for _ in range(n_hidden)]
        self.layer2 = [Neuron(n_hidden) for _ in range(n_out)]

    def __call__(self, x: list[Value]) -> list[Value]:
        h = [n(x) for n in self.layer1]
        out = [n(h) for n in self.layer2]
        return out

    def parameters(self) -> list[Value]:
        return [
            p for layer in (self.layer1, self.layer2)
            for n in layer for p in (n.w + [n.b])
        ]
```

The XOR data:
```python
xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]
ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]
```

The training loop:
```python
random.seed(0)
model = MLP(n_in=2, n_hidden=2, n_out=1)
for step in range(200):
    preds = [model(x)[0] for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    for p in model.parameters():
        p.data -= 0.1 * p.grad
    if step % 20 == 0:
        print(f"step {step}  loss = {loss.data:.4f}")
```

With random seed 0, the loss should drop from ~0.8 to under 0.05 in 200 steps.
</details>

<details>
<summary>Test for gradient correctness</summary>

```python
def test_gradient_check() -> None:
    """Compare autograd gradients to NumPy finite differences."""
    import numpy as np

    def f_numpy(x, y):
        return (x * y + np.maximum(0.0, x - y)) ** 2

    x_val, y_val = 1.7, 2.3
    eps = 1e-6
    dfx_num = (f_numpy(x_val + eps, y_val) - f_numpy(x_val - eps, y_val)) / (2 * eps)
    dfy_num = (f_numpy(x_val, y_val + eps) - f_numpy(x_val, y_val - eps)) / (2 * eps)

    x = Value(x_val)
    y = Value(y_val)
    z = (x * y + (x - y).relu()) ** 2
    z.backward()

    assert abs(x.grad - dfx_num) < 1e-5, f"x.grad = {x.grad}, num = {dfx_num}"
    assert abs(y.grad - dfy_num) < 1e-5, f"y.grad = {y.grad}, num = {dfy_num}"
```

This is the single most important test in `tinygrad.py`. It verifies that your engine is computing gradients correctly on a non-trivial expression with branching, multiplication, subtraction, ReLU, and powers.
</details>

## Stretch goals

- **Add more operations.** `sigmoid`, `log`, `sin`, `cos`. Each is a few lines.
- **Replace the Python `Value` with a NumPy `Array` version** that holds a NumPy array instead of a single float. This is the step from `micrograd` to `tinygrad` (the *real* tinygrad, by George Hotz; see <https://github.com/tinygrad/tinygrad>). You will need to be careful about broadcasting and shapes; the topology and the chain rule are unchanged.
- **Compute a Jacobian via repeated `backward()` calls.** For a vector-valued function, call `.backward()` once per output dimension and stack the gradients into the Jacobian matrix.
- **Implement a simple `torch.nn.Module`-style API.** A `Linear` class that holds a weight matrix; a `Sequential` class that composes layers. The XOR MLP above already implements this; refactor into a generalizable abstraction.

## What this challenge teaches

Three things:

1. **Autograd is not magic.** It is reverse-mode automatic differentiation on a computation graph, with one closure per operation. PyTorch's `autograd` does exactly this, with optimizations for vectorized operations and GPU dispatch. JAX's `grad` does this with tracing instead of operator overloading, but the underlying algorithm is the same.
2. **The forward pass builds the graph.** Every operation creates a new `Value` with a reference to its parents. By the time you call `.backward()`, the graph is fully constructed; the backward pass is graph traversal.
3. **The hard part is the topological sort.** A naive depth-first traversal can over-count gradients on shared subexpressions. Topological sort guarantees you traverse each node exactly once, after all its consumers.

Karpathy's micrograd is the cleanest implementation of this idea. Read it before you write yours, and again after. Reading and writing both are how you internalize the algorithm; reading without writing is how you forget it.

## Submission

Push `tinygrad.py`, `xor_demo.py`, and the test (`test_tinygrad.py`) to your `crunch-ai-portfolio-<yourhandle>` repo under `week-07/challenges/`. Include the XOR loss-after-200-steps in the commit message.

## Reading

- **Karpathy's `micrograd`** on GitHub: <https://github.com/karpathy/micrograd>. The 100-line reference. Read the file from top to bottom; the implementation is dense but every line is doing exactly one thing.
- **Karpathy's "The spelled-out intro to neural networks and backpropagation" video** (2.5 hours; free on YouTube). The video builds `micrograd` from nothing. Strongly recommended.
- **JAX documentation on autograd** (the principles section): <https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>. The same algorithm, with a more explicit derivation.
