# Lecture 3 — The Training Loop and MNIST

> **Outcome:** You can wrap your forward and backward passes in a mini-batch SGD training loop and reach ≥95% test accuracy on MNIST. You can plot the training loss and test accuracy per epoch, read both curves honestly, and recognize the four standard failure modes (loss thrashes, loss plateaus high, loss converges to log(k), test accuracy lags far behind training accuracy). You can defend your learning rate, your batch size, and your number of epochs with one sentence each. By the end of this lecture, you have trained a neural network from scratch and you have a curve to put in your portfolio.

Lectures 1 and 2 built the *primitives*: the forward pass, the loss, the backward pass, the gradient check. None of it has been trained. A randomly-initialized MLP has accuracy ≈ 10% on MNIST (chance for ten classes). Today we close the loop.

The training algorithm is **mini-batch stochastic gradient descent**: sample a batch of `m` examples from the training set, compute the forward pass, compute the backward pass, update every parameter by `θ ← θ - η · g`, and repeat for many batches until the network has seen the full training set several times (an *epoch*). The whole loop is ten lines of Python. The hyperparameters — learning rate `η`, batch size `m`, number of epochs — are five numbers, and the right values for MNIST have been known since 1998.

We target **numpy 2.x**, **scikit-learn 1.5+** (for `fetch_openml`), and **matplotlib 3.8+**. Python 3.11+. The MNIST download takes ~30 seconds the first time and is cached afterwards.

---

## 1. The training loop in full

The full training loop is one function:

```python
"""Mini-batch SGD training loop for a 2-layer MLP."""

import numpy as np
from numpy.typing import NDArray


def train(
    X_train: NDArray[np.float64], y_train: NDArray[np.int64],
    X_test: NDArray[np.float64], y_test: NDArray[np.int64],
    *,
    n_hidden: int = 128,
    learning_rate: float = 0.1,
    batch_size: int = 128,
    n_epochs: int = 15,
    random_state: int = 42,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, list[float]]]:
    """Train a 2-layer MLP with mini-batch SGD.

    Returns:
        params: the trained parameter dict
        history: dict with keys "train_loss" and "test_acc", lists of length n_epochs
    """
    rng = np.random.default_rng(random_state)
    n_train, n_in = X_train.shape
    n_out = int(y_train.max()) + 1

    params = init_params(n_in, n_hidden, n_out, rng)
    history: dict[str, list[float]] = {"train_loss": [], "test_acc": []}

    for epoch in range(n_epochs):
        # Shuffle the training indices.
        perm = rng.permutation(n_train)
        epoch_losses: list[float] = []

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start : start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            loss = sgd_step(params, X_batch, y_batch, learning_rate)
            epoch_losses.append(loss)

        # End-of-epoch diagnostics.
        train_loss = float(np.mean(epoch_losses))
        test_acc = float((predict(X_test, params) == y_test).mean())
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch + 1:2d}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    return params, history
```

Eighteen lines. The whole MLP-on-MNIST pipeline. Let us walk through it.

---

## 2. The mini-batch

Three lines of the loop deserve a closer look:

```python
perm = rng.permutation(n_train)
# ...
batch_idx = perm[start : start + batch_size]
```

`rng.permutation(n_train)` returns a random permutation of `[0, 1, ..., n_train-1]`. We then slice this permutation into batches of `batch_size` consecutive elements. The result: every example is used exactly once per epoch, and the order changes every epoch.

Three reasons we shuffle at every epoch:

1. **Avoid degenerate batches.** If the training data is sorted by class (some datasets are), an unshuffled batch might contain only examples of class 0, and the gradient would be biased.
2. **Reduce overfitting to mini-batch order.** Stochastic gradients estimated from different random batches are different gradients; the noise helps escape shallow local minima.
3. **Match the convention of every deep-learning framework.** PyTorch's `DataLoader(shuffle=True)`, TensorFlow's `Dataset.shuffle(...)`, JAX's `jax.random.permutation` — all do this. By Week 8 you will not write this line; you will pass `shuffle=True`. The convention is to do it.

The **last batch** is often smaller than `batch_size` (if `n_train` is not divisible by `batch_size`, the final batch holds the remainder). Our code handles this implicitly because `X_batch = X_train[batch_idx]` returns whatever the index slice contains. The smaller last batch has a slightly noisier gradient; the effect is negligible.

> **EXPERIMENT — degenerate ordering.** Train two models, one with shuffling and one without (`perm = np.arange(n_train)`). For MNIST, the difference is small because MNIST's openml ordering is already mostly random. Now try with `X_train` and `y_train` sorted by class (`order = np.argsort(y_train); X_train_sorted = X_train[order]; y_train_sorted = y_train[order]`). Train without shuffling. The loss curve will be dominated by class-imbalance gradients; test accuracy will plateau at ~30%. Shuffling is the difference between "trains" and "does not train."

---

## 3. The SGD update rule, in detail

The parameter update for each batch:

```python
for k in params:
    params[k] -= learning_rate * grads[k]
```

This is the **vanilla SGD** update: `θ ← θ - η · g`. The `-=` operator updates `params[k]` in place, which is what we want (the `params` dict is mutable; the training loop modifies it).

Three observations:

1. **No momentum.** Modern training loops use SGD with momentum (`v ← β v + g; θ ← θ - η v`) or Adam (which adapts the per-parameter learning rate). For MNIST + two-layer MLP, vanilla SGD is sufficient; we will introduce momentum and Adam in Week 8.
2. **No L2 regularization.** A common modification is `θ ← (1 - λη) θ - η g`, which adds a weight-decay term. We skip this for the lectures but include it as an optional argument in Exercise 3. For MNIST, weight decay improves test accuracy by ~0.3 percentage points.
3. **No gradient clipping.** For shallow MLPs on MNIST, gradients are well-behaved. For recurrent networks or transformers, gradient clipping prevents explosions; we will revisit it in Week 9.

The simplicity of this update is the *point* of SGD: at each step, the parameter moves slightly in the direction of steepest descent on the loss, weighted by the learning rate. Over thousands of steps, the parameter converges to a region of low loss. That is the whole story.

---

## 4. The learning rate

The learning rate `η` is the most important hyperparameter in deep learning. Choosing it well is mostly empirical; the theoretical bounds (Bottou 2012, Lecture 4) tell you the order of magnitude, and a learning-rate sweep tells you the rest.

**The four regimes of the learning rate:**

| `η` | Symptom | What to do |
|-----|---------|------------|
| Way too high (`> 1` for MNIST MLP) | Loss diverges; `nan` within a few epochs | Reduce by 10× |
| Too high (`0.5–1`) | Loss thrashes, oscillates | Reduce by 2-3× |
| Just right (`0.01–0.5` for MNIST MLP) | Loss decreases smoothly, plateaus at ~`0.1` after 10-15 epochs | Keep |
| Too low (`< 0.001`) | Loss decreases very slowly; would need 100+ epochs | Increase by 10× |

The C5 default for the MNIST 2-layer MLP is **`η = 0.1`**. It works for `h ∈ {32, 64, 128, 256}` and batch sizes `m ∈ {32, 64, 128, 256}`. If your MLP is wider or your batch is larger, you can typically increase `η` slightly; the rule of thumb (Bottou) is `η ∝ batch_size`, but the relationship is rough.

The **learning rate finder** (Smith 2017) is a useful empirical tool: start with `η = 1e-7`, multiply by `1.1` every batch, and plot the loss. The point where the loss starts decreasing is "too low"; the point where it bottoms out is "just right"; the point where it starts increasing is "too high." We do not use the finder this week, but it is a standard technique in Weeks 8+.

> **EXPERIMENT — learning-rate sweep.** Train your MLP for 5 epochs at each of `η ∈ {0.001, 0.01, 0.1, 1.0}`. Plot all four loss curves on the same axes. You should see: `0.001` slow but smooth descent; `0.01` smooth descent to a moderate floor; `0.1` rapid descent to a low floor; `1.0` thrashing or divergence. Save the figure and look at it whenever you forget what "wrong learning rate" looks like.

---

## 5. The batch size

The batch size controls the trade-off between gradient quality and computation cost:

- **Smaller batch** (`m = 16`): noisier gradient, more steps per epoch, slower wall-clock per epoch (more Python overhead), better generalization (the noise is a regularizer).
- **Larger batch** (`m = 1024`): cleaner gradient, fewer steps per epoch, faster wall-clock per epoch (more matrix-multiply throughput), worse generalization (less noise).

For the MNIST 2-layer MLP, **`m = 128`** is the C5 default. It balances wall-clock cost (fast on CPU) with gradient noise (enough to escape shallow minima). With `n_train = 60_000`, this gives `60_000 / 128 ≈ 469` steps per epoch.

The relationship `η_eff ≈ η · m^(0.5)` is a useful approximation (the "linear scaling rule" with a square-root correction): if you double the batch size, you should increase the learning rate by ~`sqrt(2)`. For MNIST this is well within the "just right" regime; for ImageNet at batch sizes in the thousands, it becomes a careful tuning exercise. We will revisit this in Week 8.

---

## 6. The number of epochs

How long do we train? Three signals tell you to stop:

1. **The training loss has plateaued.** If `train_loss` at epoch `t` is within 1% of `train_loss` at epoch `t-3`, the model is no longer learning from the gradient. More epochs will only fit noise.
2. **The test accuracy has plateaued.** Same as above but on `test_acc`. This is the more important signal — training loss can keep decreasing while test accuracy stagnates (overfitting).
3. **The test accuracy has *started to decrease*.** This is the overfitting signal: the model is now fitting training noise that does not generalize. Stop *before* this point. (This is the basis of **early stopping**, which we do not implement this week but is standard practice.)

For MNIST 2-layer MLP at `η = 0.1, m = 128, h = 128, He init`, the curve looks like:

| Epoch | train_loss | test_acc |
|------:|-----------:|---------:|
| 1 | 0.45 | 0.92 |
| 2 | 0.21 | 0.94 |
| 3 | 0.15 | 0.95 |
| 5 | 0.09 | 0.96 |
| 10 | 0.05 | 0.97 |
| 15 | 0.03 | 0.974 |
| 20 | 0.02 | 0.975 |

Epoch 10 is the sweet spot for our target. Beyond epoch 15, the gains are tiny; beyond epoch 20, the network starts to overfit. The C5 default is **15 epochs**; pushing to 20-25 is fine but does not help much.

> **EXPERIMENT — when does test accuracy plateau?** Train for 30 epochs. Plot `test_acc` vs. epoch. The curve is sigmoidal: steep climb in the first 3-5 epochs, gentle climb to ~97% by epoch 15, flat or slightly decreasing afterward. The "elbow" of the climb is epoch 5-7; this is where you should think about stopping if compute is limited.

---

## 7. Reading the training curves

A trained MLP produces two curves: the training loss per epoch and the test accuracy per epoch. Plot both on a 2-panel figure:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax1.plot(history["train_loss"], "o-", color="#7C3AED")
ax1.set_xlabel("epoch")
ax1.set_ylabel("train loss (mean cross-entropy)")
ax1.set_yscale("log")
ax1.grid(alpha=0.3)

ax2.plot(history["test_acc"], "o-", color="#7C3AED")
ax2.set_xlabel("epoch")
ax2.set_ylabel("test accuracy")
ax2.set_ylim(0.85, 1.0)
ax2.grid(alpha=0.3)

fig.suptitle("2-layer MLP on MNIST: training curves")
plt.savefig("mnist_curves.png", dpi=150)
```

Three things to look at:

1. **The loss curve.** Should be monotonically decreasing on a log scale, with the slope flattening over epochs. If it thrashes (zigzags), the learning rate is too high. If it is flat from epoch 1, the gradient is too small (learning rate too low, or initialization too small, or there is a bug).
2. **The test accuracy curve.** Should rise from ~10% at epoch 0 (untrained) to ~95-97% by epoch 10-15. If it is flat at 10%, the network is not learning at all (probable bug in the gradient or in the update). If it rises and then *falls*, the network is overfitting (rare for MNIST 2-layer MLP, common in deep networks).
3. **The train-test gap.** Training accuracy at epoch 15 will be ~99%; test accuracy will be ~97%. The gap of ~2 percentage points is normal for an unregularized MLP on MNIST. A gap of 10+ percentage points means severe overfitting; add dropout or weight decay.

---

## 8. Sanity checks before you commit

Before pushing your trained MLP to the portfolio repo, verify:

- [ ] **Initial loss ≈ `log(k) = 2.30`.** From Lecture 1 Section 8. If it is dramatically different, the initialization is wrong.
- [ ] **Gradient check passes.** From Lecture 2 Section 10. Relative error < `1e-5` on a 5-example batch. Run this once after every backward-pass change.
- [ ] **Training loss decreases monotonically.** From this lecture, Section 7. If it thrashes, lower the learning rate.
- [ ] **Test accuracy ≥ 95%.** The Week 7 minimum. The C5 reference run gets 97% at epoch 10.
- [ ] **Train-test gap < 5 percentage points.** If larger, regularization is needed.
- [ ] **Confusion matrix is roughly diagonal.** The off-diagonal entries should be small; the only common confusions are `4 ↔ 9` and `7 ↔ 1` (which look similar in handwriting). If `0` is being confused with `8`, something is wrong.

The confusion matrix is one line:

```python
from collections import Counter

preds = predict(X_test, params)
cm = np.zeros((10, 10), dtype=int)
for true_label, pred in zip(y_test, preds):
    cm[true_label, pred] += 1
```

Plot it as a heatmap; the diagonal should glow.

---

## 9. The "what about overfitting?" question

A two-layer MLP on MNIST with `h = 128` has ~100,000 parameters and is trained on 60,000 examples. By the classical counting argument, the model has more parameters than data points; it *should* overfit catastrophically; the test accuracy *should* be much lower than the training accuracy.

The fact that it does not — that we comfortably reach 97% test accuracy — is one of the most surprising empirical facts of deep learning, sometimes called the **"benign overfitting" phenomenon** (Belkin et al. 2018, Bartlett et al. 2020). The current best explanation is some combination of:

- **Implicit regularization by SGD.** The noise of stochastic gradients steers the optimizer toward flat minima, which generalize better than sharp ones.
- **The bias of small initial weights toward simple solutions.** The network starts close to the linear regime and learns nonlinearities only as needed.
- **The structure of MNIST itself.** The dataset has many "easy" examples; the model can fit the easy examples without overfitting the hard ones.

For Week 7, the takeaway is empirical: an MLP this size on MNIST does *not* overfit, and you can train to 97% test accuracy without dropout, weight decay, or any other regularization. We will revisit regularization in Week 9 when we move to larger models on more complex data.

---

## 10. A note on randomness

The MLP training is **not** deterministic by default. Three sources of randomness:

1. **The parameter initialization.** Different `random_state` → different starting weights → different final accuracy (typically within 0.5 percentage points).
2. **The mini-batch order.** The shuffle is different every epoch; different orderings give slightly different gradient sequences.
3. **The optional dropout** (Challenge 1). The dropout mask is sampled fresh every forward pass.

For reproducibility, set `random_state=42` everywhere (which we do throughout the lectures and exercises). Different `random_state` values will give test accuracies in `[0.965, 0.975]` for the C5 default hyperparameters — close enough that the differences are noise. If two seeds give wildly different accuracies (say, 0.85 vs. 0.97), there is a bug.

A useful sanity check is to run the same training with seeds `0, 1, 2` and report the mean and standard deviation. For our MLP this is ~`0.971 ± 0.003`. Reporting a single seed's number is fine for the C5 mini-project; mentioning that you verified across seeds is what separates a "great" submission from a "good" one.

---

## 11. Predictions and the sample-prediction grid

After training, visualize the model's predictions on a grid of test images. The standard plot is a 5×5 grid:

```python
import matplotlib.pyplot as plt
import numpy as np

# Pick 25 random test indices.
rng = np.random.default_rng(0)
idx = rng.choice(len(X_test), size=25, replace=False)
preds = predict(X_test[idx], params)
probs = forward(X_test[idx], params)["A2"]
top_prob = probs.max(axis=1)

fig, axes = plt.subplots(5, 5, figsize=(10, 10), layout="constrained")
for ax, i in zip(axes.flat, idx):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray_r")
    correct = preds[i == idx][0] == y_test[i]
    color = "#7C3AED" if correct else "#DC2626"
    pred = preds[i == idx][0]
    p = top_prob[i == idx][0]
    ax.set_title(f"true={y_test[i]} pred={pred} ({p:.2f})", color=color, fontsize=8)
    ax.axis("off")
fig.suptitle("MLP predictions: green = correct, red = wrong")
plt.savefig("mnist_predictions.png", dpi=150)
```

The figure makes the model's strengths and weaknesses concrete. A `9` predicted as `4` with probability `0.49` is a model that is appropriately uncertain on a hard example; a `0` predicted as `8` with probability `0.99` is a model that is wrong with high confidence — which is the failure mode you want to look for in the confusion matrix.

---

## 12. Where this lecture (and the week) lands

The full curriculum-to-code mapping for Week 7:

- **Lecture 1** → `softmax`, `cross_entropy_loss`, `forward`, `init_params`, `predict`. 30 lines.
- **Lecture 2** → `backward`, `gradient_check_one_param`. 40 lines.
- **Lecture 3** → `sgd_step`, `train`, the plotting helpers. 40 lines.
- **Exercise 3** → the runnable script that ties it all together on an MNIST subset. ~150 lines.
- **Mini-project** → the same machinery on a non-MNIST tabular dataset, plus the report. ~250 lines plus the write-up.

Total: ~500 lines of Python that you wrote, derived, and verified. By Sunday you will have a portfolio entry that demonstrates: I can derive backprop from the chain rule; I can implement it in NumPy; I can train an MLP; I can read a training curve; I can write the honest paragraph about what I built. That is the artifact recruiters care about.

---

## 13. What you *will* do in Week 8

Week 8 is the framework week. You will:

- Reimplement the same two-layer MLP in PyTorch in 25 lines. The `loss.backward()` call replaces the entire `backward` function from Lecture 2.
- Stack four layers and add a Conv2d to make a small CNN. The architecture changes; the training loop does not.
- Switch from SGD to Adam. The optimizer change is one line: `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`.
- Train on CIFAR-10 (a harder dataset: 32×32 color images, 10 classes). The MLP from Week 7 gets ~55% on CIFAR-10; a small CNN gets 75-80%.
- Use the GPU (if you have one). PyTorch's `model.cuda()` moves everything to the GPU; the training loop is unchanged.

Every step in Week 8 builds on Week 7. The forward-and-backward picture, the SGD update, the learning-rate intuition, the training-curve reading — all carry over. The thing that *changes* is the level of abstraction: you stop writing `dW1 = X.T @ dZ1` and start writing `loss.backward()`, and the gain is roughly a 30× reduction in code with no loss in mental model.

---

## 14. Reading and exercises before the mini-project

- **Read** Nielsen Chapter 3 (Improving the way neural networks learn). Covers the cross-entropy loss in detail and introduces L2 regularization and weight initialization. Sections 3.1-3.3.
- **Read** the *Stanford CS231n notes on training*: <https://cs231n.github.io/neural-networks-3/>. The "babysitting the learning process" section is the part to read; the recommendations on learning rate, batch size, and weight decay are the same as ours.
- **Watch** 3Blue1Brown episode 2 (gradient descent visualized). Not as cleanly as episode 3, but the visualization of the loss surface is worth ten minutes.
- **Do** Exercise 3 (the full training loop on MNIST subset). 2 hours. Target: ≥90% test accuracy on a 10,000-example subset in 10 epochs.
- **Skim** LeCun et al. 1998. The MLP baseline in Table II — "MLP, 300 hidden, 10 output: 4.7% error" — is the result you are comparing against. (4.7% error means 95.3% accuracy; the C5 target of 95% comes from this number.)

Tomorrow, the mini-project: train an MLP on a non-MNIST tabular dataset and write the report.
