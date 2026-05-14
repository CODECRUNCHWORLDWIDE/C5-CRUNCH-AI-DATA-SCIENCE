# Lecture 2 — Backpropagation from the Chain Rule

> **Outcome:** You can derive `∂L/∂W1`, `∂L/∂b1`, `∂L/∂W2`, `∂L/∂b2` for a two-layer ReLU + softmax MLP from the chain rule on one sheet of paper. You can translate the derivation into eight lines of NumPy. You can verify the analytical gradient against a finite-difference numerical gradient and accept the result only when the relative error is below `1e-6`. After this lecture, the backward pass is a sequence of matrix multiplications you can write from memory; the chain rule is the engine, and matrix shapes are the bookkeeping.

Yesterday we wrote the forward pass: five lines that turn an input batch `X` into class probabilities `A2` and a scalar loss `L`. The forward pass is the "easy" half of training a neural network. The hard half is the **backward pass**: computing the gradient of the scalar loss with respect to every parameter, so that gradient descent can update them.

This lecture derives the backward pass from first principles. There is no fancy math. The chain rule is the same single-variable chain rule you saw in your first calculus course — `(f ∘ g)'(x) = f'(g(x)) · g'(x)` — applied repeatedly. The multivariable version uses Jacobian matrices instead of scalar derivatives; in practice, every Jacobian in our two-layer MLP either *is* a matrix product or *reduces* to one, and the backward pass is six NumPy operations.

We target **numpy 2.x** and Python 3.11+. Open Lecture 1 in another tab — the forward-pass cache `{Z1, A1, Z2, A2}` is the input to the backward pass.

---

## 1. The chain rule, refreshed

Given two functions composed `h(x) = f(g(x))`, the chain rule says:

```text
h'(x)  =  f'(g(x))  ·  g'(x)
```

In multiple dimensions: if `g: R^p → R^n` and `f: R^n → R^m`, the Jacobian of the composition is the matrix product of the Jacobians:

```text
J_h(x)  =  J_f(g(x))  ·  J_g(x)         shape (m, p) = (m, n) · (n, p)
```

The Jacobian `J_f` has rows indexed by the output dimension and columns indexed by the input dimension; the `(i, j)` entry is `∂f_i / ∂g_j`. The matrix product captures the multivariable chain rule exactly.

For scalar-valued `f`, the Jacobian degenerates to the gradient `(∇f)^T`, a row vector. For backprop, this is the case we care about: the loss `L` is a scalar; everything before it is vector-valued; the chain rule says the gradient of the loss with respect to any parameter is a *row vector* obtained by chaining Jacobians back from the loss.

The convention in deep-learning derivations is to compute the gradient as a **column vector of the same shape as the parameter**. For a weight matrix `W ∈ R^{n_in × n_out}`, the gradient `dW = ∂L/∂W` has shape `(n_in, n_out)`, the same as `W`. The reason is practical: the SGD update `W ← W - η · dW` requires `dW` to have the same shape as `W`. The chain rule produces the gradient as a Jacobian; we *reshape* it to match the parameter. The reshaping is automatic in matrix form, as we will see.

---

## 2. The shape convention

For the rest of this lecture, **everything is a matrix** and the shapes are:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `X`  | `(m, n_in)`   | Input batch (`m` examples, `n_in` features) |
| `W1` | `(n_in, h)`   | First-layer weights |
| `b1` | `(h,)`        | First-layer biases |
| `Z1` | `(m, h)`      | First-layer pre-activations: `X W1 + b1` |
| `A1` | `(m, h)`      | First-layer activations: `ReLU(Z1)` |
| `W2` | `(h, k)`      | Second-layer weights |
| `b2` | `(k,)`        | Second-layer biases |
| `Z2` | `(m, k)`      | Output logits: `A1 W2 + b2` |
| `A2` | `(m, k)`      | Output probabilities: `softmax(Z2)` |
| `Y`  | `(m, k)`      | One-hot true labels (`Y[i, y_i] = 1`, else 0) |
| `L`  | scalar        | Mean cross-entropy loss |

The gradients have the same shape as the variables they are with respect to. We write `dZ2 = ∂L / ∂Z2` and so on. Shape mismatches in the backward pass are the most common bug — every step below has the shape annotated and the rule is "the gradient has the same shape as the variable."

---

## 3. The loss in one expression

The full forward pass, written end-to-end:

```text
L(W1, b1, W2, b2)
  =  (1 / m)  ·  sum_{i=1..m}  - log [ softmax( ReLU( X_i W1 + b1 ) W2 + b2 )_{y_i} ]
```

That is one nested expression. The chain rule will unpack it from the outside in. We start with the output and work backwards.

We will use the convention that **lowercase `d`** denotes "the gradient of the loss with respect to," so `dZ2` means `∂L / ∂Z2`. This is the PyTorch convention and the CS231n convention; it is also the most readable.

---

## 4. Step one: the softmax + cross-entropy gradient

We computed in Lecture 1 Section 7 that:

```text
dZ2  =  (1 / m)  ·  (A2  -  Y)            shape (m, k)
```

Let us derive it. The cross-entropy loss for one example is `L_i = - log p_{i, y_i}` where `p_i = softmax(z_i)`. The gradient with respect to `z_{i, j}` is:

```text
∂L_i / ∂z_{i, j}  =  - ∂(log p_{i, y_i}) / ∂z_{i, j}
                  =  - (1 / p_{i, y_i})  ·  ∂p_{i, y_i} / ∂z_{i, j}
```

We need `∂p_{i, y_i} / ∂z_{i, j}`. From the definition of softmax:

```text
∂p_{i, c} / ∂z_{i, j}  =  p_{i, c}  ·  (δ_{cj}  -  p_{i, j})
```

where `δ_{cj}` is the Kronecker delta (1 if `c = j`, else 0). For `c = y_i` specifically:

- If `j = y_i`: `∂p_{i, y_i} / ∂z_{i, y_i} = p_{i, y_i}  (1 - p_{i, y_i})`.
- If `j ≠ y_i`: `∂p_{i, y_i} / ∂z_{i, j} = - p_{i, y_i}  p_{i, j}`.

Substituting back into the gradient of `L_i`:

- For `j = y_i`: `∂L_i / ∂z_{i, j} = - (1/p_{i, y_i})  ·  p_{i, y_i} (1 - p_{i, y_i})  =  -(1 - p_{i, y_i})  =  p_{i, y_i} - 1`.
- For `j ≠ y_i`: `∂L_i / ∂z_{i, j} = - (1/p_{i, y_i})  ·  (-p_{i, y_i}  p_{i, j})  =  p_{i, j}`.

In both cases, using the one-hot `Y_i` with `Y_{i, y_i} = 1` and `Y_{i, j} = 0` for `j ≠ y_i`:

```text
∂L_i / ∂z_{i, j}  =  p_{i, j}  -  Y_{i, j}
```

Stacking over the batch and dividing by `m` (because `L = (1/m) sum L_i`):

```text
dZ2  =  (1 / m)  ·  (A2  -  Y)
```

That is the identity. Three lines of partial derivatives; one line of code. Memorize it.

```python
# Shape check:  dZ2 has shape (m, k), same as Z2.
dZ2 = (A2 - Y) / m
```

---

## 5. Step two: the gradients of `W2` and `b2`

`Z2 = A1 W2 + b2`. The chain rule gives:

```text
dW2  =  A1^T  ·  dZ2                      shape (h, k) = (h, m) · (m, k)
db2  =  sum over batch (dZ2)              shape (k,)   = sum along axis 0 of (m, k)
```

Let us check the matrix calculus. `Z2 = A1 W2 + b2` means `z_{i, j} = sum_p a_{i, p} w_{p, j} + b_j`. The partial derivative of `z_{i, j}` with respect to `w_{p, q}` is `a_{i, p} · δ_{jq}` (the bias does not depend on `W2`). Therefore:

```text
∂L / ∂w_{p, q}  =  sum_{i, j}  (∂L / ∂z_{i, j})  ·  (∂z_{i, j} / ∂w_{p, q})
              =  sum_i  (dZ2)_{i, q}  ·  a_{i, p}
              =  sum_i  a_{i, p}  ·  (dZ2)_{i, q}
              =  (A1^T  dZ2)_{p, q}
```

So `dW2 = A1^T @ dZ2`. The shape is `(h, m) @ (m, k) = (h, k)`, matching `W2`.

The bias gradient is even simpler. `z_{i, j} = ... + b_j` means `∂z_{i, j} / ∂b_q = δ_{jq}`. Therefore:

```text
∂L / ∂b_q  =  sum_{i, j}  (dZ2)_{i, j}  ·  δ_{jq}
           =  sum_i  (dZ2)_{i, q}
```

In NumPy:

```python
db2 = dZ2.sum(axis=0)                     # shape (k,)
```

The sum along axis 0 collapses the batch dimension. The "sum-out-batch-dimension" rule is the same for every bias in every layer: bias gradients are the sum of the corresponding pre-activation gradient over the batch.

> **EXPERIMENT — verify shapes by hand.** Pick `m = 4`, `n_in = 5`, `h = 3`, `k = 2`. Write down the shapes of `X, W1, b1, Z1, A1, W2, b2, Z2, A2, Y` and the shapes of `dZ2, dW2, db2`. Verify that `dW2 = A1^T @ dZ2` is `(3, 4) @ (4, 2) = (3, 2)`. The matrix-product shape rules will catch every bug in the backward pass; if you cannot recite this shape sequence from memory by Wednesday, re-derive it.

---

## 6. Step three: backpropagate through `W2` to get `dA1`

We have the gradient of `L` with respect to `Z2`. To continue backwards, we need the gradient of `L` with respect to `A1`. Since `Z2 = A1 W2 + b2`, by the same matrix-calculus reasoning as Section 5:

```text
dA1  =  dZ2  ·  W2^T                      shape (m, h) = (m, k) · (k, h)
```

The derivation: `z_{i, j} = sum_p a_{i, p} w_{p, j} + b_j` means `∂z_{i, j} / ∂a_{r, p} = w_{p, j} · δ_{ir}`. Therefore:

```text
∂L / ∂a_{r, p}  =  sum_{i, j}  (dZ2)_{i, j}  ·  w_{p, j}  ·  δ_{ir}
              =  sum_j  (dZ2)_{r, j}  ·  w_{p, j}
              =  (dZ2  W2^T)_{r, p}
```

In NumPy:

```python
dA1 = dZ2 @ W2.T                          # shape (m, h)
```

The pattern "backprop through a matrix multiply is multiply by the transpose" is one of the two patterns that will repeat throughout the backward pass. The other pattern, coming next, is "backprop through a nonlinearity is element-wise multiplication by the derivative."

---

## 7. Step four: backprop through ReLU

`A1 = ReLU(Z1) = max(0, Z1)`. The derivative is:

```text
∂A1 / ∂Z1  =  1 if Z1 > 0 else 0          element-wise
```

This is *not* a Jacobian in the textbook sense; it is a diagonal matrix at each example. In matrix form, the backward pass through an element-wise nonlinearity is an **element-wise multiplication** by the derivative:

```text
dZ1  =  dA1  ⊙  1[Z1 > 0]                 shape (m, h)
```

where `⊙` is the Hadamard (element-wise) product and `1[Z1 > 0]` is the binary mask of where `Z1` is positive. In NumPy:

```python
dZ1 = dA1 * (Z1 > 0)                      # broadcasting the bool mask to float
```

The boolean mask `(Z1 > 0)` is implicitly cast to `0.0` and `1.0` when multiplied by a float array. The result has the same shape as `dA1`.

**The ReLU at zero is the canonical edge case.** Strictly, `ReLU'(0)` is undefined (the function is not differentiable at zero). The convention is to set the derivative to `0` at `z = 0` (some libraries use `0.5`; either works in practice because the case is measure-zero). The mask `(Z1 > 0)` uses strict inequality and assigns `0` at the boundary; the alternative `(Z1 >= 0)` uses `1`. Pick one and be consistent; the difference is undetectable on any real training run.

> **EXPERIMENT — the ReLU mask.** In a REPL: `Z1 = np.array([[-1.0, 0.0, 1.0, 2.0]]); mask = (Z1 > 0); print(mask)`. The mask is `[[False, False, True, True]]`. Multiply: `dA1 = np.ones_like(Z1); dZ1 = dA1 * mask; print(dZ1)`. The result is `[[0.0, 0.0, 1.0, 1.0]]`. The "dead" half of the ReLU is exactly what the mask zeroes out; this is the algebraic source of dead-neuron problems (Section 12).

---

## 8. Step five: the gradients of `W1` and `b1`

Now we have `dZ1`. By the same reasoning as Section 5 (this layer has the same form `Z1 = X W1 + b1`), the gradients are:

```text
dW1  =  X^T  ·  dZ1                       shape (n_in, h) = (n_in, m) · (m, h)
db1  =  sum over batch (dZ1)              shape (h,)
```

In NumPy:

```python
dW1 = X.T @ dZ1                           # shape (n_in, h)
db1 = dZ1.sum(axis=0)                     # shape (h,)
```

And we are done. The backward pass for a two-layer MLP is six lines:

```python
dZ2 = (A2 - Y) / m
dW2 = A1.T @ dZ2
db2 = dZ2.sum(axis=0)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0)
dW1 = X.T @ dZ1
db1 = dZ1.sum(axis=0)
```

Seven lines if you count both bias rows. The full forward + backward + loss is ~15 lines of NumPy. Once you have written it, you can stare at it for an hour and not be sure what every line is doing; this is normal. The reading comprehension comes after Exercise 2.

---

## 9. The full backward pass in a function

```python
"""Backward pass for a 2-layer ReLU + softmax MLP."""

import numpy as np
from numpy.typing import NDArray


def backward(
    cache: dict[str, NDArray[np.float64]],
    params: dict[str, NDArray[np.float64]],
    y: NDArray[np.int64],
) -> dict[str, NDArray[np.float64]]:
    """Return the gradients of the parameters as a dict keyed W1, b1, W2, b2.

    cache  -- the dict returned by forward(), containing X, Z1, A1, Z2, A2
    params -- the dict containing W1, b1, W2, b2
    y      -- the integer labels of shape (m,)
    """
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]
    W2 = params["W2"]
    m, k = A2.shape

    # One-hot encode the labels.
    Y = np.zeros((m, k), dtype=np.float64)
    Y[np.arange(m), y] = 1.0

    # Backward pass, six lines:
    dZ2 = (A2 - Y) / m                    # (m, k)
    dW2 = A1.T @ dZ2                      # (h, k)
    db2 = dZ2.sum(axis=0)                 # (k,)
    dA1 = dZ2 @ W2.T                      # (m, h)
    dZ1 = dA1 * (Z1 > 0)                  # (m, h)
    dW1 = X.T @ dZ1                       # (n_in, h)
    db1 = dZ1.sum(axis=0)                 # (h,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

The function is in Exercise 2's solution. Read it once before writing your own; you should be able to map every line to a step in Sections 4-8.

---

## 10. Gradient checking — the single most important debugging tool

The backward pass we derived has six lines of math and a dozen places to introduce a bug:

- The `/ m` in `dZ2`. (Forgotten: the loss is averaged but the gradient is the sum.)
- The transpose in `dW2 = A1.T @ dZ2`. (Wrong: `A1 @ dZ2` is a shape error; `A1 @ dZ2.T` is even more wrong but easier to type by accident.)
- The mask in `dZ1 = dA1 * (Z1 > 0)`. (Wrong: `dA1 * (A1 > 0)` is *almost* right because ReLU's output is `> 0` iff its input is `> 0`, but it can fail at the boundary; the correct mask uses the pre-activation `Z1`.)
- The order in `dA1 = dZ2 @ W2.T`. (Wrong: `dA1 = W2 @ dZ2.T` has the right shape but is the wrong product.)

The **gradient check** is the systematic way to catch all of these. The idea: compute the *numerical* gradient by finite differences, compare to your *analytical* gradient, and require the two to agree to six decimal places.

The numerical gradient of `L` with respect to a single scalar parameter `θ_i` is:

```text
g̃_i  =  ( L(θ + ε · e_i)  -  L(θ - ε · e_i) )  /  (2ε)
```

where `e_i` is the standard basis vector and `ε` is a small number (typically `1e-7`). This is the **centered finite difference**; it has `O(ε^2)` error, much better than the forward difference's `O(ε)`.

The check itself:

```text
relative_error  =  | g_i  -  g̃_i |  /  max(| g_i |, | g̃_i |, 1e-12)
```

Accept if `relative_error < 1e-5`. Suspicious if between `1e-5` and `1e-3`. Reject (debug the analytical gradient) if `relative_error > 1e-3`.

```python
def gradient_check_one_param(
    forward_and_loss,                     # (params, X, y) -> (loss, _)
    params: dict[str, NDArray[np.float64]],
    analytical_grads: dict[str, NDArray[np.float64]],
    param_name: str,
    X: NDArray[np.float64], y: NDArray[np.int64],
    eps: float = 1e-7,
    n_samples: int = 10,
    rng: np.random.Generator | None = None,
) -> float:
    """Return the maximum relative error over n_samples random elements
    of params[param_name].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    p = params[param_name]
    g_a = analytical_grads[param_name]
    flat_idx = rng.choice(p.size, size=n_samples, replace=False)

    max_rel_err = 0.0
    for idx_flat in flat_idx:
        idx = np.unravel_index(idx_flat, p.shape)

        # +eps
        p[idx] += eps
        loss_plus, _ = forward_and_loss(params, X, y)
        # -eps (relative to original)
        p[idx] -= 2 * eps
        loss_minus, _ = forward_and_loss(params, X, y)
        # restore
        p[idx] += eps

        g_num = (loss_plus - loss_minus) / (2 * eps)
        g_ana = g_a[idx]
        denom = max(abs(g_ana), abs(g_num), 1e-12)
        rel_err = abs(g_ana - g_num) / denom
        max_rel_err = max(max_rel_err, rel_err)

    return max_rel_err
```

Two important practical notes:

- **Sample, do not exhaustively check.** Computing the numerical gradient for every parameter would take `O(p)` forward passes; for our 50,000-parameter MLP that is impractical. Sampling 10-20 random elements per matrix is enough to catch a bug; the bug is in the *derivation*, not in a particular element.
- **Use small inputs, double precision, and small ε.** Gradient checking on a single tiny batch (`m = 5`, `n_in = 3`, `h = 4`) is what the literature calls a "smoke test." It is the standard practice: most bugs in backprop are *algebraic* and visible on tiny data. If you can pass a gradient check on a 5-example mini-MLP, your derivation is correct and you can scale up.

> **EXPERIMENT — gradient check pays for itself.** On Tuesday afternoon you will write a `backward` function. The first version will almost certainly have a bug; the most common one is a missing `/ m` somewhere. If you skip the gradient check and start training, the network will appear to learn (because the gradient direction is still roughly right) but converge to a worse accuracy or fail to reach 95%. The gradient check, run on a 5-example batch, takes 0.3 seconds and catches the bug in three lines of output. It is the single most cost-effective ten minutes of debugging in the deep-learning curriculum.

---

## 11. The four canonical bugs (and how the gradient check finds each)

After grading thousands of from-scratch MLPs, the deep-learning community has converged on a short list of bugs. The same four come up every year.

### Bug 1 — Shape mismatch in the backward pass

Symptom: a `ValueError: shapes ... not aligned` somewhere in `backward`. The error message tells you which line; the fix is to transpose one of the operands.

Catch: writing the shapes in a comment next to each line, as we did in Section 9. The shape annotation forces you to check the matrix-product rule for every line; the error appears before you run.

### Bug 2 — Off-by-one in batch averaging

Symptom: the gradient check fails with a relative error of *exactly* `m` (the batch size). The analytical gradient is `m` times too large because you forgot the `/ m` in `dZ2`; or `m` times too small because you applied `/ m` twice (once in the loss and again in the gradient).

Catch: by the gradient check, on a 5-example batch. The relative error of `5.0` is the unmistakable signature.

### Bug 3 — Wrong sign on the gradient update

Symptom: the loss *increases* instead of decreasing. The training curve thrashes upward.

Cause: `params[k] = params[k] + learning_rate * grads[k]` (a `+` where there should be a `-`). The gradient direction is steepest *ascent*; we want to step in the opposite direction.

Catch: the first few SGD steps; the loss going up is the unmistakable signature.

### Bug 4 — Unstable softmax

Symptom: `RuntimeWarning: overflow encountered in exp`, followed by `nan` everywhere.

Cause: the naive `np.exp(z) / np.exp(z).sum()` without the `z - z.max()` subtraction. Lecture 1 Section 5.

Catch: by running the forward pass on a moderately large random `X` (any logit value above ~700 will overflow). The `nan` propagation is the signature.

---

## 12. A note on dead ReLUs

The ReLU has a derivative of `0` on the negative side. If a neuron's pre-activation `z_{i, j}` is always negative — for every input in the training set — then:

- The neuron's output `a_{i, j} = ReLU(z_{i, j})` is always zero (no contribution to the next layer).
- The neuron's gradient `dz_{i, j} = da_{i, j} · 1[z_{i, j} > 0] = 0` is also always zero.
- The weights into the neuron, which depend on `dz_{i, j}`, receive zero gradient.
- The neuron is **dead**. It will never learn again.

This is the canonical pathology of ReLU networks. It is the price we pay for the no-saturation-on-the-positive-side property that made ReLU successful in the first place.

The fixes, in order of how often they are used in 2026:

1. **Lower the learning rate.** Dead ReLUs are usually caused by a large initial gradient step pushing the neuron into a negative-pre-activation regime; a smaller learning rate avoids this.
2. **Use He initialization.** Scaled to `sqrt(2 / n_in)` instead of `sqrt(1 / n_in)`; reduces the probability of large initial pre-activations.
3. **Use Leaky ReLU.** `LeakyReLU(z) = max(αz, z)` for small `α` (typically `0.01`). The negative side has slope `α` instead of 0, so gradients flow even when the neuron is on the "wrong" side.
4. **Use ELU or GELU.** Smoother alternatives with non-zero negative-side gradients; popular in modern transformer architectures.

For Week 7, ReLU + He initialization is fine. The mini-project's tabular data has enough signal in the gradients that dead neurons are rare. Lecture 3 includes a diagnostic — counting the fraction of dead neurons after training — and recommends Leaky ReLU only if more than 25% of neurons are dead.

---

## 13. Backprop as a *graph* operation

The derivation above is mechanical: chain rule, apply, chain rule, apply, until you have the gradient with respect to every parameter. A useful mental model is the **computation graph**: every node is a value (`X`, `Z1`, `A1`, ...) and every edge is an operation (`@`, `+`, `ReLU`, ...). The forward pass evaluates the graph from inputs to loss; the backward pass traverses it in reverse, multiplying gradients at each edge.

This is exactly what PyTorch's `autograd` does, and what JAX's `grad` transformation does, and what Karpathy's `micrograd` does in 100 lines. The graph is implicit in NumPy: every operation creates a new array, and you can in principle record every operation and replay it in reverse. The "in principle" is the difference between writing the backward pass by hand (Week 7) and having a library do it for you (Week 8).

The micrograd source (Challenge 2) is the cleanest implementation. We recommend reading it after Exercise 2 and before Lecture 3 — it makes the graph mental model concrete.

---

## 14. A clean training-step function

Putting forward, backward, and the SGD update together:

```python
"""One SGD step for a 2-layer MLP."""

import numpy as np
from numpy.typing import NDArray


def sgd_step(
    params: dict[str, NDArray[np.float64]],
    X_batch: NDArray[np.float64],
    y_batch: NDArray[np.int64],
    learning_rate: float,
) -> float:
    """Run forward, backward, and update params in place. Return the loss."""
    loss, cache = forward_and_loss(X_batch, y_batch, params)
    grads = backward(cache, params, y_batch)
    for k in params:
        params[k] -= learning_rate * grads[k]
    return loss
```

That is the heart of the training loop. Five lines. Wrap it in `for epoch in range(n_epochs): for batch in mini_batches(...): sgd_step(...)` and you have a trainable network. Lecture 3 covers the loop, batch generation, learning-rate selection, and what to plot.

---

## 15. What this lecture established

You can now:

- Derive the backward pass for a two-layer MLP from the chain rule, in one page.
- Translate the derivation into eight lines of NumPy.
- Verify the analytical gradient against a finite-difference numerical gradient.
- Recognize the four canonical bugs (shape mismatch, batch-averaging off-by-one, sign error, softmax overflow).
- Explain why dead ReLU neurons happen and what the four fixes are.

What you cannot yet do:

- Train the network. We have one SGD step; we need many.
- Pick a learning rate, a batch size, and a number of epochs.
- Read a training curve and decide whether it is converging.
- Reach 95% test accuracy on MNIST.

Tomorrow, the training loop.

---

## 16. Reading and exercises before Lecture 3

- **Read** Nielsen Chapter 2 (the backward pass in slightly different notation; the matrix equations of backprop in equations BP1-BP4).
- **Read** CS231n notes on "Backpropagation, Intuitions" (the Karpathy view of backprop as computation-graph-traversal; the "patterns in backprop" section is the part to memorize).
- **Watch** 3Blue1Brown episodes 3 and 4 (backprop visually; calculus of backprop).
- **Do** Exercise 2 (`relu`, `relu_grad`, `softmax`, `cross_entropy_loss` + the backward pass). 2 hours.
- **Read** Karpathy's `micrograd` source (100 lines; on GitHub). Time-box to 30 minutes; do not try to refactor.
- **Skim** the Rumelhart, Hinton, Williams (1986) paper. The matrix-form derivation in the 1986 paper is what we re-derived above. It is striking how stable this material has been across four decades.

Tomorrow, the training loop and MNIST.
