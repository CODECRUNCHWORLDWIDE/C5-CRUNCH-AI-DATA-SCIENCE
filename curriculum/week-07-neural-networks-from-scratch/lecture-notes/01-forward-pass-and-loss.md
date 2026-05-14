# Lecture 1 — The Forward Pass and the Loss

> **Outcome:** You can describe a two-layer MLP in five lines of NumPy. You can name every shape in the forward pass for a `(batch, 784)` input through a `(784, h)` hidden layer to a `(h, 10)` output, and explain why each shape is what it is. You can write the softmax function in a way that does not overflow, derive the cross-entropy loss from the negative log-likelihood of a categorical distribution, and put the two together to see why "softmax + cross-entropy" is the canonical pairing for classification. After this lecture, the forward pass is no longer a black box.

Six weeks of supervised learning have given you four model types — a from-scratch linear regressor, a `Ridge` from scikit-learn, a from-scratch decision tree, and a `HistGradientBoostingRegressor`. Every one of those models had a forward pass you could write in one line: `y_hat = X @ w + b` for the linear models, and a stack of `if x_j > threshold` branches for the trees. None of them needed a "backward pass" because the parameters were either solved analytically (linear regression's normal equations) or chosen greedily one split at a time (the trees).

A neural network is the first model where the forward pass is more than one operation and the parameters are fit by gradient descent. This lecture is the forward pass. Lecture 2 derives the backward pass. Lecture 3 puts them in a training loop on MNIST. By Friday you will have a working MLP that classifies handwritten digits with 95%+ test accuracy, written from scratch.

We target **numpy 2.x**, **scikit-learn 1.5+** (only for the MNIST loader), and **matplotlib 3.8+** for plotting. Python 3.11+.

---

## 1. The single neuron

A neuron is a linear function followed by a nonlinearity. With input `x ∈ R^p`, weight vector `w ∈ R^p`, and bias `b ∈ R`:

```text
z  =  w · x  +  b           pre-activation (a scalar)
a  =  σ(z)                  activation (also a scalar)
```

The function `σ` is the **activation function**. Three classical choices:

- **Step function** (Rosenblatt 1958): `σ(z) = 1 if z ≥ 0 else 0`. The original perceptron. Not differentiable, so it cannot be trained by gradient descent.
- **Sigmoid** (Werbos 1974; popular in 1980s–1990s): `σ(z) = 1 / (1 + e^{-z})`. Smooth, differentiable, outputs in `(0, 1)`. The historical default.
- **ReLU** (Nair and Hinton 2010; Glorot et al. 2011): `σ(z) = max(0, z)`. Piecewise linear, simple derivative, not bounded above. The modern default.

The choice of activation function is *the* most consequential architectural decision after "how many layers." Sigmoid networks have well-known training problems past two or three layers (vanishing gradients; see Section 4). ReLU networks train more reliably but have their own pathology (dead neurons; see Lecture 2 Section 7). We will use sigmoid in some early examples and ReLU in the MNIST classifier; you should be fluent in both.

> **EXPERIMENT — plot the three activations.** In a REPL: `import numpy as np; import matplotlib.pyplot as plt; z = np.linspace(-5, 5, 200); plt.plot(z, 1 / (1 + np.exp(-z)), label="sigmoid"); plt.plot(z, np.tanh(z), label="tanh"); plt.plot(z, np.maximum(0, z), label="ReLU"); plt.legend(); plt.show()`. Three lines, one figure. Notice that sigmoid is bounded in `(0, 1)`; tanh is bounded in `(-1, 1)`; ReLU is unbounded above and clips to zero below. The bounded ones saturate (their derivatives approach zero) when `|z|` is large; the unbounded one does not. The saturation matters for backprop.

---

## 2. Why one neuron is not enough

A single neuron computes a linear function of its inputs (plus an activation). With a sigmoid activation, the neuron computes a **logistic regression**: the same model as Week 4 with a different name and a different optimizer. It can only classify data that is linearly separable in the input space.

The classic example is the **XOR problem**. The four points `(0, 0), (0, 1), (1, 0), (1, 1)` with labels `0, 1, 1, 0` are not linearly separable: no straight line in the 2D plane puts the `1`s on one side and the `0`s on the other. A single logistic neuron *cannot* fit this dataset, no matter how long you train it. Minsky and Papert proved this in *Perceptrons* (1969), and the result almost killed the field for a decade.

The solution, which took until the 1970s to be properly understood, is to **stack neurons in layers**. Two logistic neurons in a hidden layer, with appropriate weights, can each draw one of the two diagonals of the XOR pattern; a third neuron in the output layer can combine the two. The XOR function is solvable by a two-layer MLP with two hidden neurons and two weights of `+1` and `-1`. The full construction is in Nielsen Chapter 1 — read it.

The takeaway: **a single layer of neurons is a single linear transformation followed by a single nonlinearity**. The composition of two such layers gives a strictly larger function class. The **universal approximation theorem** (Cybenko 1989, Hornik 1991) makes this rigorous: any continuous function on a compact set can be approximated to arbitrary precision by an MLP with one hidden layer and sufficient width. In theory, every function we will ever want to learn is learnable by a sufficiently large two-layer MLP.

The theorem says nothing about how *many* neurons that will require or whether SGD will find the right weights. Both questions are open.

---

## 3. The two-layer MLP — the architecture

A two-layer MLP on MNIST has:

- **Input layer**: 784 features (28×28 pixel images, flattened). One example is a vector `x ∈ R^784`; a mini-batch of `m` examples is a matrix `X ∈ R^{m × 784}`.
- **Hidden layer**: `h` neurons (we use `h = 64` in the lectures, `h = 128` in the exercises, both are fine). Weight matrix `W1 ∈ R^{784 × h}`; bias vector `b1 ∈ R^h`. Pre-activation `Z1 = X W1 + b1 ∈ R^{m × h}`. Activation `A1 = ReLU(Z1) ∈ R^{m × h}`.
- **Output layer**: 10 neurons (one per digit class). Weight matrix `W2 ∈ R^{h × 10}`; bias vector `b2 ∈ R^10`. Pre-activation `Z2 = A1 W2 + b2 ∈ R^{m × 10}`. Activation `A2 = softmax(Z2) ∈ R^{m × 10}` — the predicted probabilities over the ten classes.

The total parameter count is:

```text
W1:  784 · h         W2:  h · 10
b1:  h               b2:  10

For h = 64:   784·64 + 64 + 64·10 + 10 = 50,176 + 64 + 640 + 10 = 50,890 parameters.
For h = 128:  784·128 + 128 + 128·10 + 10 = 100,352 + 128 + 1280 + 10 = 101,770 parameters.
```

This is the size of a small spreadsheet. Modern deep networks have parameter counts in the billions; ours has 50,000. We are at the small end of the MLP scale, which is exactly the point — at this size, training fits on a CPU in minutes, gradient checking is feasible, and every shape is small enough to print.

The convention to **count weight matrices, not layers** trips up newcomers. Our "two-layer MLP" has *one* hidden layer; some authors call this a "one-hidden-layer MLP" or, confusingly, a "three-layer MLP" (input, hidden, output). The C5 convention follows the modern PyTorch convention: count the number of weight matrices. Two weight matrices = two-layer MLP.

---

## 4. The forward pass — line by line

The forward pass for a mini-batch of `m` examples is five lines of NumPy:

```python
import numpy as np
from numpy.typing import NDArray


def forward(
    X: NDArray[np.float64],
    W1: NDArray[np.float64], b1: NDArray[np.float64],
    W2: NDArray[np.float64], b2: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Forward pass for a 2-layer MLP with ReLU hidden + softmax output.

    X  shape (m, 784)
    W1 shape (784, h);  b1 shape (h,)
    W2 shape (h, 10);   b2 shape (10,)

    Returns a dict with Z1, A1, Z2, A2 -- the cache used by the backward pass.
    """
    Z1 = X @ W1 + b1                                # (m, h)
    A1 = np.maximum(Z1, 0.0)                        # (m, h)  ReLU
    Z2 = A1 @ W2 + b2                               # (m, 10)
    A2 = softmax(Z2)                                # (m, 10)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```

Five operations: a matmul, an elementwise max, another matmul, a softmax. Every shape is bounded above by `m × h` (the hidden-layer activation), which for `m = 128` and `h = 128` is a `(128, 128)` matrix — 64 KB at `float64`. The forward pass runs in milliseconds on a laptop CPU.

Three things to notice about the shapes:

1. **`X W1 + b1` broadcasts the bias.** `X @ W1` has shape `(m, h)` and `b1` has shape `(h,)`; NumPy broadcasts `b1` along the batch dimension, adding the same bias vector to every row. This is implicit in the `+` operator — no `np.tile`, no `np.repeat`, no `np.broadcast_to` needed.
2. **The hidden dimension `h` is the "width" of the network.** A wider network has more capacity (more parameters); a narrower network is faster and less likely to overfit. The trade-off is one of the few you can adjust without re-deriving anything; `h = 32, 64, 128, 256` all work for MNIST.
3. **The cache is mandatory for backprop.** The backward pass needs `Z1`, `A1`, `Z2`, and `A2` (or some subset, depending on the implementation) to compute the gradients. Storing them costs memory; the alternative — recomputing them — costs time. The standard library approach is to cache, so we cache.

> **EXPERIMENT — verify the forward shapes.** In a REPL: load `W1, b1, W2, b2` from random initialization with `h = 64`, generate a fake batch `X = np.random.randn(128, 784)`, and run `forward(X, W1, b1, W2, b2)`. Print the shape of each return value. They should be `(128, 64)`, `(128, 64)`, `(128, 10)`, `(128, 10)`. If any shape is wrong, the bug is one of: `W1` has the wrong shape (most common — students often transpose `W1` to `(h, 784)` and end up with `(784, 64) @ (128, 784) = error`), or you wrote `X @ W2` somewhere instead of `A1 @ W2`. Fix the shape error before continuing; never train a network whose forward pass shapes you do not understand.

---

## 5. The softmax function

The output layer of a classifier returns a vector of **logits** `z ∈ R^k`. These are unnormalized scores: bigger means "more confident in this class," but the scores are not probabilities. The **softmax** function maps logits to a probability distribution over the `k` classes:

```text
p_i  =  exp(z_i)  /  sum_{j=1..k} exp(z_j)
```

Three properties:

1. **Each `p_i ≥ 0`.** Because `exp(·) > 0`.
2. **`sum_i p_i = 1`.** By construction; the denominator is the sum of the numerators.
3. **`p` is a valid probability distribution over `{1, ..., k}`.** That is what makes softmax the natural output for a multi-class classifier.

The naive NumPy implementation is one line:

```python
def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """WRONG: this overflows on z values larger than ~700."""
    return np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
```

It is also broken. `np.exp(z)` overflows when `z > 709` (the upper limit of `float64`). On real-world logits — early in training, before the loss has shrunk anything — values of 800 or 1000 are not uncommon, and the naive softmax returns `inf / inf = nan`. Once a `nan` appears in the gradient, the whole training run is poisoned.

The fix is the **numerical-stability trick**: subtract the maximum logit before exponentiating. The result is identical mathematically — `exp(z - c) / sum exp(z - c) = exp(z) exp(-c) / (exp(-c) sum exp(z)) = exp(z) / sum exp(z)` — but `exp(z - max(z))` is guaranteed to be in `(0, 1]` and cannot overflow.

```python
def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax. Operates on the last axis."""
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)
```

Three notes on the implementation:

- **The `axis=-1` argument.** In our 2-layer MLP, `z` has shape `(m, k)`; the softmax should normalize *per example*, not across the batch. `axis=-1` is the last axis (the class axis); `keepdims=True` keeps the broadcast dimension so the division is element-wise.
- **`z.max(axis=-1, keepdims=True)` has shape `(m, 1)`.** Subtracting it from `z` broadcasts along the class axis, shifting each example's logits independently.
- **The maximum after the subtraction is exactly zero.** So `exp(0) = 1` is the largest value in `exp_z`, and the denominator is at least `1`. No overflow.

> **EXPERIMENT — softmax stability.** In a REPL: `z = np.array([[1000.0, 1000.1, 999.9]])`. The naive softmax returns `[nan, nan, nan]` (run it; you will see). The stable softmax returns `[0.300, 0.332, 0.298]` — a perfectly reasonable probability distribution. The difference is one line of NumPy. Karpathy's micrograd notes call this "the most important line of code in deep learning" and they are not wrong.

---

## 6. Cross-entropy loss

For a classifier with predictions `p ∈ Δ^{k-1}` (a probability distribution over `k` classes) and a true label `y ∈ {0, ..., k-1}` (an integer class index), the **cross-entropy loss** is:

```text
L(y, p)  =  - log p_y                      (the log probability of the correct class)
```

Equivalently, if we one-hot encode `y` as `Y ∈ R^k` with `Y[y] = 1` and zeros elsewhere:

```text
L(Y, p)  =  - sum_{i=1..k}  Y_i  log p_i   (a single nonzero term, by Y's structure)
```

The two forms are identical because `Y` has exactly one `1` and the rest are `0`. The one-hot form generalizes to *soft labels* (where multiple `Y_i` can be nonzero), which we will revisit in Week 9 when we cover label smoothing.

For a mini-batch of `m` examples, the loss is averaged:

```text
L_batch  =  (1 / m)  sum_{i=1..m}  - log p_{i, y_i}
```

The average (not the sum) is the conventional definition; it makes the learning rate independent of the batch size.

The NumPy implementation:

```python
def cross_entropy_loss(
    probs: NDArray[np.float64], y: NDArray[np.int64]
) -> float:
    """Cross-entropy loss for a batch.

    probs: (m, k) array of predicted probabilities (rows sum to 1)
    y:     (m,) array of integer class labels in [0, k)

    Returns the scalar mean negative-log-likelihood.
    """
    m = probs.shape[0]
    # Clip away exact zeros to avoid log(0) = -inf. The clip value is
    # below floating-point single precision; the gradient is unaffected.
    safe = np.clip(probs[np.arange(m), y], 1e-12, 1.0)
    return float(-np.log(safe).mean())
```

Two notes:

- **`probs[np.arange(m), y]` is fancy indexing.** It selects, for each row `i`, the column `y[i]`. The result is the predicted probability of the correct class for each example — a 1D array of shape `(m,)`.
- **`np.clip(safe, 1e-12, 1.0)` guards against `log(0)`.** A correctly-trained softmax produces probabilities in `(0, 1)`, never exactly zero. But early in training, the predicted probability for the correct class can be tiny — `1e-30` — and `log(1e-30) = -69`, while `log(0) = -inf`. Clipping to `1e-12` makes the worst-case loss `log(1e12) = 27.6`, which is large but finite. The bound `1e-12` is below `float32` precision (~ `1.2e-7`) and well below `float64` precision (~ `2.2e-16`); the gradient is essentially unaffected, but the numerics are stable.

---

## 7. The famous identity: softmax + cross-entropy

When we put softmax and cross-entropy together — which we always do for multi-class classification — the gradient of the loss with respect to the logits simplifies to one line:

```text
∂L / ∂z_i  =  p_i  -  Y_i
```

where `p = softmax(z)` and `Y` is the one-hot label. The proof is one page of partial derivatives (we work it through in Lecture 2 Section 4), but the result is so clean that it is worth memorizing right now: **the gradient of softmax + cross-entropy with respect to the logits is "predicted probability minus one-hot true label."**

Three reasons this identity matters:

1. **The gradient is implemented in three lines of NumPy.** Without the simplification, you would compute the `(k, k)` Jacobian of softmax separately and then chain it through cross-entropy — a `(k, k) @ (k, 1)` matrix-vector product for every example. With the simplification, you compute `A2 - Y` and you are done.
2. **It tells you what the network is doing.** If `p_i > Y_i`, the network is predicting class `i` too confidently; the gradient pushes the logit `z_i` *down* (because the loss decreases as the predicted probability decreases). If `p_i < Y_i`, the network is predicting class `i` too weakly; the gradient pushes `z_i` *up*. The gradient direction has a clean interpretation in every case.
3. **It is numerically well-behaved.** `A2 - Y` cannot overflow or underflow regardless of the input; the values are always in `[-1, 1]`. Compare this to the alternative (Jacobian of softmax times derivative of log), which involves divisions and can be unstable.

The identity is the reason we always say "softmax + cross-entropy" as a single phrase. The two functions are paired because their composition has the cleanest possible gradient. Sigmoid + log-loss has an analogous property in the binary case (`p - y`); MSE on softmax does not have this property and is one of the standard wrong choices for classification.

---

## 8. The full forward + loss in twelve lines

Combining sections 4, 5, and 6 gives the full forward + loss in twelve lines of NumPy:

```python
"""A 2-layer MLP forward pass + cross-entropy loss in twelve lines."""

import numpy as np
from numpy.typing import NDArray


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z_shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def forward_and_loss(
    X: NDArray[np.float64], y: NDArray[np.int64],
    W1: NDArray[np.float64], b1: NDArray[np.float64],
    W2: NDArray[np.float64], b2: NDArray[np.float64],
) -> tuple[float, dict[str, NDArray[np.float64]]]:
    """Forward pass + cross-entropy loss for a 2-layer ReLU MLP."""
    Z1 = X @ W1 + b1
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    m = X.shape[0]
    safe = np.clip(A2[np.arange(m), y], 1e-12, 1.0)
    loss = float(-np.log(safe).mean())
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "y": y}
    return loss, cache
```

That is the whole forward pass. The cache holds everything we will need for the backward pass — `X` for the gradient of `W1`, `Z1` for the ReLU mask, `A1` for the gradient of `W2`, `A2` for the softmax + CE gradient, and `y` for the one-hot subtraction.

Run this on a `(128, 784)` batch of random inputs with `h = 64`. The loss should be approximately `log(10) = 2.30`, because before training the softmax outputs are roughly uniform `(0.1, 0.1, ..., 0.1)` and `-log(0.1) = 2.30`. If your initial loss is dramatically different from `2.30`, your initialization is wrong (see Section 9).

> **EXPERIMENT — initial loss is `log(k)`.** This is a non-negotiable sanity check for every classification network you ever build. After random initialization, the network has no information about the inputs; the softmax outputs should be approximately uniform across the `k` classes; the loss should be approximately `log(k)`. For MNIST with `k = 10`, the initial loss is `log(10) ≈ 2.302`. If your initial loss is `0.5`, you have a bug (probably the softmax is degenerate); if it is `15`, your initialization is too large and `exp` is producing extreme values. The check takes ten seconds; it has saved every working ML engineer at least one wasted day.

---

## 9. Initialization

How do we choose the initial values of `W1, b1, W2, b2`? Three failed choices, then the right one:

1. **All zeros.** `W1 = W2 = 0; b1 = b2 = 0`. Every neuron computes the same pre-activation; every neuron has the same gradient; the network is *one* neuron with `h × k` copies. No learning. This is called the **symmetry-breaking failure**.
2. **All ones.** Same problem (every neuron still computes the same thing, just at a different scalar).
3. **Random with large variance.** `W ~ N(0, 1)`. Forward-pass activations have variance proportional to the layer width; for `h = 128`, the pre-activations are `O(sqrt(128)) ≈ 11` in magnitude; the softmax saturates almost immediately, and the gradients are tiny.

The right move is **random with small, carefully-chosen variance**. Three common choices:

- **Naive small random**: `W ~ N(0, 0.01)`. Works for shallow networks (≤ 3 layers) and small enough hidden widths. The 1990s default. We use this in Exercise 1.
- **Xavier (Glorot) initialization** (Glorot and Bengio 2010): `W ~ N(0, 1/n_in)`. Scaled to preserve the variance of activations in a sigmoid or tanh network. Has been the textbook standard since 2010.
- **He initialization** (He et al. 2015): `W ~ N(0, 2/n_in)`. Scaled to preserve the variance of activations in a *ReLU* network. ReLU halves the variance of its input (because it zeros out the negative half), so He compensates by doubling the variance of `W`. The modern default for ReLU networks; we use it in Exercise 3 and the mini-project.

The He initialization in NumPy is two lines:

```python
def he_init(
    n_in: int, n_out: int, rng: np.random.Generator,
) -> NDArray[np.float64]:
    """He initialization for a layer mapping n_in -> n_out features."""
    std = np.sqrt(2.0 / n_in)
    return rng.normal(0.0, std, size=(n_in, n_out))
```

Biases are typically initialized to zero. Zero biases are not subject to the symmetry-breaking failure because the *weights* are random; the bias of the `j`-th hidden neuron is shared across the batch but the gradient is different for every neuron because the input-times-weight is different. Zero bias initialization is the standard.

> **EXPERIMENT — He vs naive on initial loss.** Initialize a 2-layer MLP for MNIST with `h = 128`. First with `W1 ~ N(0, 0.01)`, then with He. Compute the initial loss with each. Both should be close to `log(10) = 2.30`; the He version should be slightly closer. Now change `h` to `1024`. The naive init gives a loss of ~3.5 (too small per-row weights for the wide layer); He still gives ~2.30. He scales with the width; naive does not. This is why He is the default for any non-trivial width.

---

## 10. Putting it all together — a `forward` and `predict` function

For the rest of the week, we will assume a **parameter dict** of the form `params = {"W1": ..., "b1": ..., "W2": ..., "b2": ...}`. The forward pass takes `X` and `params` and returns the cache; the predict function takes `X` and `params` and returns the predicted class for each row:

```python
"""A 2-layer MLP forward + predict in fifteen lines."""

import numpy as np
from numpy.typing import NDArray


def init_params(
    n_in: int, n_hidden: int, n_out: int,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    """He initialization for the weights, zero initialization for biases."""
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / n_in),     size=(n_in, n_hidden)),
        "b1": np.zeros((n_hidden,), dtype=np.float64),
        "W2": rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_out)),
        "b2": np.zeros((n_out,), dtype=np.float64),
    }


def forward(
    X: NDArray[np.float64],
    params: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = softmax(Z2)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def predict(
    X: NDArray[np.float64],
    params: dict[str, NDArray[np.float64]],
) -> NDArray[np.int64]:
    """Return the argmax class for each row."""
    cache = forward(X, params)
    return cache["A2"].argmax(axis=-1)
```

Fifteen lines of code. We can now load MNIST, initialize a network, predict, and measure test accuracy — even though the network is untrained. The untrained accuracy will be approximately `1/10 = 10%` (chance, because the softmax is uniform); the trained accuracy after Lecture 3 will be 95%.

---

## 11. Why no autograd this week

PyTorch's `torch.autograd` and JAX's `jax.grad` would let us skip Lecture 2 entirely. Define the forward pass, call `.backward()`, and the framework computes every gradient by tracing the computation graph and applying the chain rule symbolically. In production, you will use one of these. In Week 8, we will.

This week we are deliberately doing without. Three reasons:

1. **You will know what `backward()` is doing.** When a framework's gradient does something surprising (and they all do; PyTorch's `retain_graph`, `detach`, and in-place-operation rules are non-trivial), the only way to debug is to have a mental model of what the graph is computing. The mental model is built by writing the backward pass by hand once.
2. **You will recognize the four canonical bugs.** Shape mismatches, off-by-one batch-averaging, sign errors, and softmax overflow account for >90% of failed from-scratch MLPs. Lecture 2 walks through each, and the exercise tests check for all four.
3. **You will appreciate the framework.** After you have written `d_W1 = X.T @ d_Z1` and verified that the shape works out, calling `loss.backward()` and getting `W1.grad` automatically feels less like magic and more like an obvious abstraction. That appreciation is the point.

Lecture 2 derives the backward pass. Lecture 3 puts it in a training loop. Friday you have a working MLP at 95% test accuracy on MNIST. The whole week is about 800 lines of code; PyTorch will compress it to fifty in Week 8, but the 800 lines are the ones that teach.

---

## 12. Where the data comes from

We use MNIST throughout this week. The dataset is famous enough that it has its own conventions:

- **60,000 training images and 10,000 test images.** This is the standard train/test split since LeCun et al. 1998. Do not re-split it; the canonical split is what every paper compares against.
- **28×28 pixels, grayscale.** Each image is 784 pixels when flattened. Pixel values are in `[0, 255]` as bytes; we normalize to `[0, 1]` by dividing by 255 in the loader.
- **Ten classes, 0 through 9.** Roughly balanced (each class has ~6,000 training and ~1,000 test images).
- **Easy by 2026 standards.** A two-layer MLP reaches ~95%; a CNN reaches ~99.5%; the state-of-the-art is ~99.8% and not improving. The Week 7 target is *the MLP number*, not the CNN number.

The canonical MNIST loader for scikit-learn is `fetch_openml`:

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
X, y = mnist.data.astype(np.float64) / 255.0, mnist.target.astype(np.int64)
# Convention: first 60k are train, last 10k are test (LeCun 1998).
X_train, X_test = X[:60_000], X[60_000:]
y_train, y_test = y[:60_000], y[60_000:]
```

The download is ~17 MB and runs once; subsequent calls hit the cache. If `openml.org` is unreachable, Exercise 3 has a fallback path that loads from the canonical IDX files at `yann.lecun.com/exdb/mnist`.

> **EXPERIMENT — visualize MNIST.** Load MNIST and plot a 5×5 grid of training images with their labels. `fig, axes = plt.subplots(5, 5, figsize=(8, 8)); for ax, img, label in zip(axes.flat, X_train[:25], y_train[:25]): ax.imshow(img.reshape(28, 28), cmap="gray_r"); ax.set_title(str(label)); ax.axis("off"); plt.show()`. The images are immediately recognizable as digits. Spend two minutes looking at the variation — some `7`s have a horizontal slash, some do not; some `2`s have a loop at the bottom, some are open. The variation is what makes MNIST learnable but not trivial.

---

## 13. What this lecture set up

You can now write:

- A two-layer MLP forward pass in five lines of NumPy.
- A numerically stable softmax with the `max` subtraction trick.
- A cross-entropy loss with `log(0)` clipping.
- He initialization for the weights.
- A predict function that returns argmax class indices.

What you cannot yet do:

- Compute the gradient of the loss with respect to the parameters.
- Update the parameters by gradient descent.
- Train the network.

Those are Lecture 2 and Lecture 3. By the end of Lecture 2 you will have an analytical gradient that agrees with a numerical gradient to six decimal places; by the end of Lecture 3 you will have a training loop that hits 95% test accuracy on MNIST in 10 epochs.

---

## 14. Reading and exercises before Lecture 2

- **Read** Nielsen Chapter 1 (the forward pass and the loss in slightly different notation; the MNIST examples in particular).
- **Read** CS231n notes on "Neural Networks Part 1: Setting up the Architecture" (the activation-function survey).
- **Watch** 3Blue1Brown episode 1 (the visual introduction to the MLP).
- **Do** Exercise 1 (the runnable scaffold) and Exercise 2 (`relu`, `softmax`, `cross_entropy_loss`). Both can be done in 1-2 hours each.
- **Skim** the original perceptron paper (Rosenblatt 1958) and the backprop paper (Rumelhart, Hinton, Williams 1986). Both are short; both are historically important.

Tomorrow, the backward pass.
