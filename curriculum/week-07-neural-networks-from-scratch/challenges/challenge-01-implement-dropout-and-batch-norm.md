# Challenge 1 — Implement Dropout and Batch Normalization

**Time estimate:** 2 hours.

## Problem statement

Extend the two-layer MLP from the exercises to support **dropout** (Srivastava et al. 2014) and **batch normalization** (Ioffe and Szegedy 2015), in pure NumPy, with no PyTorch and no `sklearn.neural_network`. Train the augmented MLP on MNIST and compare its test accuracy and train-test gap against the vanilla MLP from Exercise 3. The two regularizers should each improve the test accuracy slightly (~0.3-0.5 percentage points) and shrink the train-test gap noticeably.

The point is *not* to chase a specific number. The point is to understand the algorithms by writing them: dropout is a randomly-applied mask in the forward pass and a slightly subtle correction in the backward pass; batch normalization is two new layer types (one for training, one for inference) with five new pieces of state to maintain.

## What you will produce

A single script `dropout_batchnorm.py` and a one-page markdown writeup `notes.md` that:

1. Loads MNIST via `sklearn.datasets.fetch_openml` (see Exercise 3 Part D).
2. Trains three models with **identical** hyperparameters (`h = 128`, `lr = 0.1`, `batch_size = 128`, `n_epochs = 15`, `random_state = 42`):
   a. **Baseline**: the vanilla 2-layer MLP from Exercise 3.
   b. **Dropout**: the same architecture with dropout `p = 0.2` applied to the hidden activations during training.
   c. **Batch normalization**: the same architecture with a `BatchNorm1D` layer between the hidden weights and the ReLU.
3. Records the final test accuracy and the train-test gap for each.
4. Saves a 2-panel figure `dropout_batchnorm.png` showing the test-accuracy curves and the train-loss curves for all three models.
5. Writes `notes.md` (~250 words) defending the comparison.

## Acceptance criteria

- [ ] `dropout_batchnorm.py` runs end-to-end with `python dropout_batchnorm.py`.
- [ ] `python -m py_compile dropout_batchnorm.py` succeeds.
- [ ] `dropout_batchnorm.png` is saved; the test-accuracy panel shows three lines clearly labeled, all reaching at least 95% on full MNIST.
- [ ] The dropout implementation includes the **inverted-dropout** form (scale the kept activations by `1 / (1 - p)` during training; do nothing at inference). The naive form (scale at inference) is also correct but is not the modern default; document the choice.
- [ ] The batchnorm implementation includes **per-feature** running mean and variance (the standard EMA over batches during training; used directly at inference).
- [ ] The backward pass for batchnorm uses the four-step derivation from Ioffe and Szegedy 2015 Appendix A. Showing the derivation in `notes.md` is bonus but not required.
- [ ] The script uses `random_state=42` everywhere randomness is involved.
- [ ] The script verifies, via a small gradient check on a 5-example batch, that the augmented backward passes agree with finite differences to within `1e-5` relative error.

## Hints

<details>
<summary>Dropout forward pass</summary>

```python
def dropout_forward(a, p, rng, training):
    """Inverted dropout: zero out a fraction p of activations during
    training and rescale survivors by 1/(1-p). At inference, return a
    unchanged.
    """
    if not training:
        return a, None
    mask = (rng.random(a.shape) > p).astype(np.float64) / (1.0 - p)
    return a * mask, mask
```

The backward pass is the same mask: `da = da_out * mask`. The mask is *stochastic* — fresh draws every forward pass.
</details>

<details>
<summary>Dropout placement</summary>

Apply dropout to the hidden activation `A1`, after the ReLU. Do *not* apply it to the output `A2` (the softmax is already a normalization layer) and do *not* apply it to the input `X` (the input is not redundant in the same way).

The pattern is: `Z1 -> ReLU -> A1 -> Dropout -> Z2`.
</details>

<details>
<summary>Batchnorm forward pass</summary>

For a batch `Z1` of shape `(m, h)`:

```python
# Training:
mean = Z1.mean(axis=0)              # shape (h,)
var = Z1.var(axis=0)                # shape (h,)
Z1_norm = (Z1 - mean) / np.sqrt(var + 1e-8)
Z1_out = gamma * Z1_norm + beta     # gamma, beta are learnable, shape (h,)
# Inference: use the running mean and var maintained as EMA over batches.
```

The two new parameters `gamma` (init 1) and `beta` (init 0) are learned by SGD just like `W1, W2`. The running mean and variance are *not* learned by SGD; they are updated via exponential moving average over the training batches.
</details>

<details>
<summary>Batchnorm backward pass</summary>

The backward pass is the most algebraically tedious part of this challenge. From Ioffe and Szegedy 2015 Appendix A:

```python
def batchnorm_backward(dZ1_out, Z1, mean, var, gamma, eps=1e-8):
    m = Z1.shape[0]
    Z1_norm = (Z1 - mean) / np.sqrt(var + eps)
    dgamma = (dZ1_out * Z1_norm).sum(axis=0)
    dbeta = dZ1_out.sum(axis=0)
    dZ1_norm = dZ1_out * gamma
    dvar = (dZ1_norm * (Z1 - mean) * -0.5 * (var + eps) ** -1.5).sum(axis=0)
    dmean = (dZ1_norm * -1.0 / np.sqrt(var + eps)).sum(axis=0) + dvar * (-2.0 / m) * (Z1 - mean).sum(axis=0)
    dZ1 = dZ1_norm / np.sqrt(var + eps) + dvar * 2.0 * (Z1 - mean) / m + dmean / m
    return dZ1, dgamma, dbeta
```

Verify this against the finite-difference gradient. The relative error should be well under `1e-5`.
</details>

<details>
<summary>The running-mean update</summary>

After each training batch:

```python
running_mean = momentum * running_mean + (1 - momentum) * mean
running_var  = momentum * running_var  + (1 - momentum) * var
```

`momentum = 0.9` is the standard default. At inference time, use `running_mean` and `running_var` in place of the batch statistics.
</details>

## The notes prompt

In `notes.md` (~250 words):

1. **What were the three final test accuracies?** Report to three decimal places.
2. **What were the three train-test gaps?** (Train accuracy at the final epoch minus test accuracy at the final epoch.) Did dropout shrink the gap relative to the baseline? Did batchnorm?
3. **Which regularizer worked better on MNIST?** Justify with one sentence. (Empirically: batchnorm usually wins by a hair on MNIST because the dataset is large enough that the running statistics are well-estimated. On smaller datasets, dropout often wins.)
4. **What was the wall-clock cost of each regularizer?** Roughly. (Dropout: ~5% overhead from the mask multiplication. Batchnorm: ~15% overhead from the per-batch mean/var computations.)
5. **The C5 honest paragraph.** Did the regularizers help by an amount that would be visible to a stakeholder, or is the improvement in the noise? On full MNIST with `h = 128`, the answer is "barely visible." Write this honestly.

## Stretch goals

- **Implement BOTH dropout AND batchnorm** in the same model. Compare the four configurations (none / dropout-only / bn-only / both). On MNIST, "both" is usually no better than "bn-only"; the lesson is that regularizers can interfere with each other.
- **Vary the dropout probability** `p ∈ {0.1, 0.2, 0.3, 0.5}`. Plot test accuracy vs. `p`. The curve is typically U-shaped on MNIST: too little dropout = no regularization; too much = under-fitting.
- **Implement layer normalization** (the variant that normalizes per-row instead of per-feature; used in transformers). Compare against batchnorm. On the 2-layer MLP, layer norm typically performs about the same as batchnorm.
- **Visualize the running statistics.** After training, plot the per-feature `running_mean` and `running_var`. The mean should be ~0 for pre-ReLU activations; the variance should be `O(1)`. Deviations from this are diagnostic.

## What this challenge teaches

Three things:

1. **Regularizers are not magic.** Each one is a specific modification of the forward and backward passes, with specific code, specific hyperparameters, and a specific failure mode. Reading "dropout improves generalization" in a paper is not the same as having written the dropout backward pass.
2. **Training-mode vs. inference-mode matters.** Dropout zeros activations during training, not inference; batchnorm uses batch statistics during training, running statistics during inference. The PyTorch convention is the `model.train()` / `model.eval()` toggle; in our NumPy implementation it is a `training: bool` argument to every layer.
3. **Improvements are often small.** A modern deep-learning paper that reports "+ 0.3% accuracy" from a new regularizer is reporting a real improvement, but you have to *measure honestly* to see it. The MNIST gain from dropout on a 2-layer MLP is in the noise. The same regularizer on a deeper network with more parameters can be transformative; the C5 takeaway is that "it helps on MNIST" is not the right standard.

## Submission

Push `dropout_batchnorm.py`, `dropout_batchnorm.png`, and `notes.md` to your `crunch-ai-portfolio-<yourhandle>` repo under `week-07/challenges/`. Include the final test-accuracy table in the commit message.
