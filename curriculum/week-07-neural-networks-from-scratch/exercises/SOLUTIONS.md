# Week 7 — Exercise Solutions and Commentary

Read this *only after* attempting each exercise on your own. The solutions are short — fifteen lines on average — but the path to them is the point of the week, not the destination.

---

## Exercise 1 — NumPy MLP Skeleton

### Part A — `init_params`

```python
def init_params(n_in, n_hidden, n_out, rng):
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / n_in),     size=(n_in, n_hidden)),
        "b1": np.zeros((n_hidden,), dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out)),
        "b2": np.zeros((n_out,), dtype=np.float64),
    }
```

**Commentary.** Three things to notice. First, the standard deviation of `W1` scales with the *input* dimension (`n_in = 784` for MNIST), and the standard deviation of `W2` scales with the *hidden* dimension (`n_hidden = 128`). The scale always uses the fan-in (the number of incoming connections to each neuron in that layer); He et al. 2015 derives this from preserving the variance of activations through a ReLU. Second, biases start at zero. This is the standard convention; you may also see `b1 = 0.01` or some small constant, which is harmless on MNIST but does not matter empirically. Third, `np.sqrt(2.0 / n_in)` — note the `2`, not `1`. Xavier (Glorot) uses `1`; He uses `2`. The `2` is the right one for ReLU networks because ReLU zeros out half the input variance.

### Part B — `softmax`

```python
def softmax(z):
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)
```

**Commentary.** Three lines, two ideas. The first idea is the `axis=-1` argument throughout: this lets the same code work for any batch size, because the last axis is always the class axis. The second idea is the `z - z.max(...)` subtraction with `keepdims=True`: it shifts each row's logits independently so the largest value is `0` and `exp(0) = 1` is the largest value in the numerator. Without this, an input like `z = [[1000, 1001]]` overflows; with it, the result is `[[0.269, 0.731]]`. The Karpathy quote that this is "the most important line of code in deep learning" is hyperbole — but the typical bug in a from-scratch softmax is omitting it.

### Part C — `forward`

```python
def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = softmax(Z2)
    return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```

**Commentary.** Five lines, exactly the equation set from Lecture 1 Section 4. The cache includes `X` so the backward pass can compute `dW1 = X.T @ dZ1` without re-reading from a closure. Storing `Z1` and `A1` separately costs ~`m * h * 8` bytes each — for `m = 128, h = 128`, that is 16 KB each — which is the standard time-memory trade in backprop. The alternative (recomputing `Z1` from `A1` and `params`) doubles the forward-pass cost.

### Part D — `cross_entropy_loss`

```python
def cross_entropy_loss(probs, y):
    m = probs.shape[0]
    safe = np.clip(probs[np.arange(m), y], 1e-12, 1.0)
    return float(-np.log(safe).mean())
```

**Commentary.** The `probs[np.arange(m), y]` is the workhorse of multiclass classification in NumPy: it gathers, per row `i`, the entry at column `y[i]`. The result is a 1D array of shape `(m,)` containing the predicted probability of each example's true class. The `np.clip(safe, 1e-12, 1.0)` guards against `log(0) = -inf`; in practice, on a moderately-trained network, the smallest predicted probability is `> 1e-6` and the clip never fires. Cast the result to `float` so the return type is unambiguous.

### Part E — `predict`

```python
def predict(X, params):
    cache = forward(X, params)
    return cache["A2"].argmax(axis=-1).astype(np.int64)
```

**Commentary.** The argmax across the class axis gives the most-probable class. The `astype(np.int64)` is for stable dtype semantics; without it, the return dtype could be `int32` or `int64` depending on the platform.

### A check the tests do not cover

If `init_loss ≠ ~2.30` on MNIST-shape data, your initialization is too large or your softmax is broken. The check is one line:

```python
loss = cross_entropy_loss(forward(X, init_params(784, 128, 10, rng))["A2"], y)
assert abs(loss - np.log(10)) < 0.5
```

This catches a class of bugs that the shape tests miss.

---

## Exercise 2 — ReLU, Softmax, Cross-Entropy, and the Backward Pass

### Part A — `relu` and `relu_grad`

```python
def relu(z):
    return np.maximum(z, 0.0)


def relu_grad(z):
    return (z > 0).astype(np.float64)
```

**Commentary.** `np.maximum(z, 0.0)` is the element-wise version of `max`. The alternative `z * (z > 0)` is equivalent but slightly slower in benchmarks because it does a multiply where `np.maximum` does a comparison. For the gradient, `(z > 0)` returns a bool array; `astype(np.float64)` is explicit. The convention at the boundary point `z = 0` is to return `0` (`>` is strict); some libraries use `0.5`. The difference is undetectable on any training run.

**Common bug.** Computing the gradient as `(a > 0)` from the activation rather than `(z > 0)` from the pre-activation. The two are equivalent when `z != 0`, but `(a > 0)` produces `False` at the boundary (because `relu(0) = 0`), while `(z > 0)` produces `False` for any `z <= 0`. Either way the boundary is measure-zero; the bug is more theoretical than practical. *But* always use the pre-activation `Z1`: it is what Lecture 2 derives, and it is what every textbook uses.

### Part B — `softmax`

Same as Exercise 1.

### Part C — `cross_entropy_loss`

Same as Exercise 1.

### Part E — `backward`

```python
def backward(cache, params):
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]
    y = cache["y"]
    W2 = params["W2"]
    m, k = A2.shape

    Y = np.zeros((m, k), dtype=np.float64)
    Y[np.arange(m), y] = 1.0

    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
```

**Commentary.** This is the most important code in Week 7. Six lines of math, derived in Lecture 2 Sections 4-8. Notice:

- **`dZ2 = (A2 - Y) / m`.** The `/ m` is the easiest place to introduce a bug. The forward pass averages the loss over the batch (Lecture 1 Section 6); the gradient must therefore also be averaged. If you compute `dZ2 = A2 - Y` (without `/ m`), the gradient check will fail with a relative error of exactly `m`.
- **`dW2 = A1.T @ dZ2`.** The transpose puts `A1` in shape `(h, m)` so the matmul with `dZ2` shape `(m, k)` produces a `(h, k)` matrix — matching `W2`. If you write `A1 @ dZ2`, you get a shape error; if you write `A1 @ dZ2.T`, you get the right shape but the wrong matrix.
- **`dZ1 = dA1 * relu_grad(Z1)`.** The `*` is element-wise; the mask zeros out gradient components where the ReLU was on its inactive side. The mask comes from `Z1`, not `A1`, for the convention reasons in Part A.

**The four canonical bugs:**

1. **Missing `/ m`.** Symptom: gradient check fails with relative error exactly `m`. Fix: add `/ m` to the `dZ2` line.
2. **Transposed matmul.** Symptom: `ValueError: shapes ... not aligned`. Fix: read Lecture 2 Section 5 again; the rule is "to get a gradient with the same shape as the parameter, the input is multiplied by the output gradient with appropriate transposes."
3. **ReLU mask from `A1` instead of `Z1`.** Symptom: gradient check passes (because at the boundary, `Z1 == 0` is rare on float64 data), but the convention is inconsistent. Fix: use `Z1`.
4. **Missing `sum(axis=0)` on bias.** Symptom: shape error in the SGD step (bias is shape `(k,)`, gradient is shape `(m, k)`). Fix: `db = dZ.sum(axis=0)`.

### Part F — gradient checking

```python
def numerical_gradient(params, X, y, param_name, idx, eps=1e-7):
    p = params[param_name]
    original = float(p[idx])
    p[idx] = original + eps
    loss_plus, _ = forward_and_loss(X, y, params)
    p[idx] = original - eps
    loss_minus, _ = forward_and_loss(X, y, params)
    p[idx] = original
    return (loss_plus - loss_minus) / (2.0 * eps)


def gradient_check(params, grads, X, y, n_samples=12, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    max_errs = {}
    for k in params:
        p = params[k]
        g_a = grads[k]
        flat_idx = rng.choice(p.size, size=min(n_samples, p.size), replace=False)
        max_err = 0.0
        for fi in flat_idx:
            idx = np.unravel_index(int(fi), p.shape)
            g_num = numerical_gradient(params, X, y, k, idx)
            g_ana = float(g_a[idx])
            denom = max(abs(g_ana), abs(g_num), 1e-12)
            rel_err = abs(g_ana - g_num) / denom
            max_err = max(max_err, rel_err)
        max_errs[k] = max_err
    return max_errs
```

**Commentary.** The mutate-in-place pattern (set `p[idx] += eps`; compute loss; restore) is the standard way to avoid copying the parameter dict. Centered differences (`(f(+eps) - f(-eps)) / (2*eps)`) have `O(eps^2)` error vs. `O(eps)` for one-sided; the cost is one extra forward pass. The `n_samples=12` is enough to catch bugs in any of the four parameter matrices; checking every element would be `O(p)` forward passes — for our 50k-parameter MLP that is 50,000 forward passes (intractable).

**Relative error tolerance.**

- `< 1e-5` is accept.
- `1e-5 to 1e-3` is suspicious; the most common cause is `eps` too small (`1e-12` is too small — the numerator becomes noise) or too large (`1e-3` is too large — the centered difference has higher-order error).
- `> 1e-3` is reject; debug the derivation.

---

## Exercise 3 — Train on MNIST Subset

### Part B — `sgd_step`

```python
def sgd_step(params, X_batch, y_batch, learning_rate):
    cache = forward(X_batch, params)
    loss = cross_entropy_loss(cache["A2"], y_batch)
    grads = backward(cache, params, y_batch)
    for k in params:
        params[k] -= learning_rate * grads[k]
    return loss
```

**Commentary.** Five lines. The `params[k] -= ...` is in-place mutation; the SGD update is destructive on purpose (we do not want to copy ~100,000 parameters every step). Note `params[k] -= learning_rate * grads[k]` is *not* the same as `params[k] = params[k] - learning_rate * grads[k]` — the second form creates a new array and rebinds the dict entry; the first mutates the existing array. For correctness either works; for memory, the `-=` form is preferred.

**Common bug.** The sign. `params[k] += learning_rate * grads[k]` is the wrong sign and the loss will increase. The test `test_backward_step_decreases_loss` in Exercise 2 catches this.

### Part C — `train`

```python
def train(X_train, y_train, X_test, y_test, *, n_hidden=128, learning_rate=0.1,
          batch_size=128, n_epochs=10, random_state=42, verbose=True):
    rng = np.random.default_rng(random_state)
    n_train, n_in = X_train.shape
    n_out = int(y_train.max()) + 1
    params = init_params(n_in, n_hidden, n_out, rng)
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(n_epochs):
        perm = rng.permutation(n_train)
        epoch_losses = []

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            loss = sgd_step(params, X_batch, y_batch, learning_rate)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))
        test_acc = float((predict(X_test, params) == y_test).mean())
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        if verbose:
            print(f"epoch {epoch + 1:2d}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    return params, history
```

**Commentary.** Eighteen lines; the entire training loop. The structure follows Lecture 3 Section 1 exactly. Three details:

- **`rng.permutation(n_train)`** is the shuffle every epoch. The order changes; without this, the network can develop pathological gradient sequences. See the EXPERIMENT in Lecture 3 Section 2.
- **The last batch is smaller.** With `n_train = 10_000` and `batch_size = 128`, we have `9984 = 78 * 128` full batches and a final batch of `16`. The smaller batch has slightly noisier gradients; the effect is negligible.
- **`predict(X_test, params)`** runs the *full* test set through the forward pass. With `X_test` of shape `(2000, 784)` and `h = 128`, the activations require `2000 * 128 * 8 ≈ 2 MB` — comfortably in memory. For larger test sets, batch the prediction.

### Expected outputs on the MNIST subset

With the default hyperparameters (`h = 128`, `η = 0.1`, batch 128, 10 epochs, 10k train / 2k test):

```text
epoch  1  train_loss=0.51  test_acc=0.92
epoch  2  train_loss=0.25  test_acc=0.94
epoch  3  train_loss=0.17  test_acc=0.94
epoch  4  train_loss=0.13  test_acc=0.94
epoch  5  train_loss=0.10  test_acc=0.95
epoch  6  train_loss=0.08  test_acc=0.95
epoch  7  train_loss=0.06  test_acc=0.95
epoch  8  train_loss=0.05  test_acc=0.95
epoch  9  train_loss=0.04  test_acc=0.95
epoch 10  train_loss=0.04  test_acc=0.95
```

(Numbers vary by seed and floating-point version; expect `test_acc` in `[0.93, 0.96]` at epoch 10.)

The training loss decreases by roughly 10× over the run. The test accuracy plateaus at ~95%. The train-test gap at epoch 10 is small (`train_loss = 0.04` corresponds to ~99% training accuracy; test accuracy is 95%; the gap of ~4 percentage points is typical for unregularized MLPs on MNIST).

### What "great" looks like

A great Exercise 3 submission:

1. **The integration test passes.** `test_mnist_subset_reaches_90_percent` returns `>=0.90` on the 10k subset. (The mini-project pushes this to `>=0.95` on the full 60k.)
2. **The training loss is monotonically decreasing.** No thrashing, no plateau at the start, no `nan`s.
3. **The script runs in under three minutes** on a laptop CPU (the C5 reference machine is an M2 MacBook Air; YMMV).
4. **You can read the curve.** Print the history; the first-epoch loss is ~0.5, the last is ~0.05; the test accuracy starts at ~0.92, ends at ~0.95.

If your test accuracy is below 0.90 in 10 epochs, the debugging path is:

- Did you set `learning_rate = 0.1`? Lower learning rates need more epochs.
- Did you shuffle in the epoch loop? `perm = rng.permutation(n_train)` inside the outer loop, not outside.
- Is your gradient check (Exercise 2) passing? If not, the gradient is wrong and the loss decrease is from luck.
- Are you using ReLU and softmax? The lectures and the solution assume both.

---

## Common questions, answered

**Q: Why is my test accuracy *higher* than my training accuracy at epoch 1?**

A: Train accuracy at epoch 1 averages the per-batch accuracies *during training* — when the parameters are still being updated. The first batch sees random parameters; by the last batch of epoch 1, the parameters have been updated 78 times. The test accuracy is measured *after* the epoch, with the final parameters. So "epoch-1 train_acc" averages over a moving target while "epoch-1 test_acc" measures only the endpoint. The two will be closer by epoch 3.

**Q: My loss is `nan` after one epoch. What happened?**

A: One of:

1. The softmax overflowed (you skipped the `z - z.max(...)` trick). Re-check Exercise 1 Part B.
2. `log(0)` was hit (you skipped the `np.clip` in cross-entropy). Re-check Exercise 1 Part D.
3. The learning rate is too high and the weights have diverged. Lower `learning_rate` to `0.01` and retry.

**Q: My gradient check passes but training does not converge. What now?**

A: The gradient is correct; the bug is in the training loop. Re-check:

- The sign of the update (`-=` not `+=`).
- The shuffle (must be inside the epoch loop).
- The batch indexing (`batch_idx = perm[start : start + batch_size]` is right; `batch_idx = np.arange(start, start + batch_size)` skips the shuffle).

**Q: What if I want to use `tanh` instead of ReLU?**

A: Replace `relu(z)` with `np.tanh(z)` and `relu_grad(z)` with `1 - np.tanh(z) ** 2` (the derivative of `tanh`). You will also want to change He initialization to Xavier (`std = sqrt(1 / n_in)`) because Xavier is calibrated for symmetric activations. The MLP will train somewhat more slowly and reach ~93% (not 95%); the difference is the empirical superiority of ReLU on deep networks, evident even at two layers.

---

## What the next two weeks add

- **Week 8** (PyTorch / JAX): the entire forward + backward + loop you wrote this week compresses to ~30 lines using `loss.backward()` and `optim.SGD`. You will train the same architecture, then a CNN, on MNIST and CIFAR-10.
- **Week 9** (CNNs from scratch): the next from-scratch unit. Convolutional layers in NumPy; pooling; the LeNet architecture; training to 99% on MNIST. Then back to PyTorch in Week 10.

The from-scratch discipline of Week 7 is the *deep-learning equivalent* of the from-scratch linear regression in Week 1 and the from-scratch k-means in Week 6. Three from-scratch artifacts; three weeks where you wrote the algorithm before using the library. That is the C5 discipline.
