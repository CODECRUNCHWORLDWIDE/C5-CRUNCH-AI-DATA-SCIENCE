# Week 7 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** The cross-entropy loss for a multi-class classifier with predicted probabilities `p` and one-hot true labels `Y` is:

- A) `L = (1/m) sum_i (p_i - Y_i)^2`.
- B) `L = -(1/m) sum_i Y_i log(p_i)` summed over both batch and class axes.
- C) `L = (1/m) sum_i |p_i - Y_i|`.
- D) `L = max_i p_i - max_i Y_i`.

---

**Q2.** The "softmax + cross-entropy" simplification states that:

- A) `∂L/∂Z2 = (A2 - Y) / m` where `A2 = softmax(Z2)` and `Y` is the one-hot label. The derivation cancels the softmax Jacobian against the inverse from the log in cross-entropy, leaving a one-line gradient.
- B) `∂L/∂Z2 = A2 * (1 - A2)` (the softmax Jacobian alone).
- C) Softmax is equivalent to a single linear layer with bias, so they can be merged into one.
- D) The two functions cancel and the loss is `0` when computed with the right normalization.

---

**Q3.** The "subtract the max" trick in the softmax — `softmax(z) = exp(z - max(z)) / sum exp(z - max(z))` — exists because:

- A) It makes the softmax differentiable at zero.
- B) It prevents numerical overflow. `exp(z)` overflows when `z > 709` in float64; subtracting the max guarantees the largest exponent is `exp(0) = 1`, which cannot overflow.
- C) It normalizes the softmax to range `[0, 1]`, which it would not otherwise be.
- D) It is required for the cross-entropy gradient simplification to hold.

---

**Q4.** For a two-layer MLP with `X` shape `(m, 784)`, `W1` shape `(784, h)`, and `Z1 = X W1 + b1`, the gradient of the loss with respect to `W1` is:

- A) `dW1 = dZ1 @ X.T` with shape `(h, 784)`. (Wrong transpose; this has the wrong shape.)
- B) `dW1 = X @ dZ1` with shape `(m, h)`. (This is a shape error.)
- C) `dW1 = X.T @ dZ1` with shape `(784, h)`. The shape matches `W1`. The derivation is in Lecture 2 Section 8.
- D) `dW1 = X.T @ Z1` with shape `(784, h)`. (Wrong: this is the forward-pass gradient, not the backward.)

---

**Q5.** "Vanishing gradients" refers to:

- A) The gradient of the loss with respect to the *predictions* becoming small as accuracy improves. This is a feature; it is why training slows near convergence.
- B) The gradient of the loss with respect to *early-layer parameters* becoming exponentially small as the network deepens, because the chain rule multiplies many activation derivatives that are each less than 1. For sigmoid networks, the derivative is bounded above by `0.25`, so a 4-layer network has gradient magnitudes scaled by `0.25^4 ≈ 0.004`. ReLU fixed this (mostly) because its derivative is exactly `1` on the positive side. Lecture 2 Section 12.
- C) The gradient vanishing because the learning rate is too small.
- D) The gradient becoming zero after a finite number of SGD steps due to floating-point underflow.

---

**Q6.** The "dead ReLU" phenomenon happens when:

- A) A neuron's weights become exactly zero. (Happens but is rare; "dead" refers to a different mechanism.)
- B) A neuron's pre-activation `z` is always negative across the entire training set, so the ReLU output is always zero and the gradient flowing into the neuron is always zero. The neuron never learns. Caused by too-large a learning rate or too-large an initial weight; mitigated by He initialization, a smaller learning rate, or by switching to Leaky ReLU.
- C) The neuron has saturated on the positive side and is now flat.
- D) The neuron's gradient overflows and becomes `nan`.

---

**Q7.** You initialize a two-layer MLP for MNIST with `n_hidden = 128` and observe an initial loss of `0.5` on a randomly-sampled batch. This is a problem because:

- A) The initial loss for a randomly-initialized softmax classifier on `k` classes should be approximately `log(k) = log(10) = 2.30`. A loss of `0.5` means either the softmax is broken, the cross-entropy is broken, or the network is somehow already predicting the correct class with high probability — none of which is consistent with random initialization. Lecture 1 Section 8.
- B) The loss should be `0.0` after random initialization.
- C) The loss should equal the test accuracy.
- D) The loss should equal the learning rate.

---

**Q8.** In an epoch of mini-batch SGD on MNIST with `n_train = 60_000` and `batch_size = 128`:

- A) The full training set is processed exactly once per epoch, in 469 mini-batch updates (with a final partial batch of 16 examples). The training data should be shuffled at the start of every epoch to avoid degenerate gradient sequences. Lecture 3 Section 2.
- B) Each example is processed `batch_size = 128` times per epoch.
- C) The number of mini-batches per epoch equals the batch size.
- D) An epoch is exactly one parameter update.

---

**Q9.** Your two-layer MLP for MNIST has analytical gradient `g_a = 0.024` and finite-difference numerical gradient `g_num = 0.120` for one element of `W1`. The relative error is approximately `0.8`. The honest interpretation is:

- A) The two gradients are close enough; accept and start training.
- B) Lower the learning rate.
- C) The analytical gradient is wrong. Relative error above `1e-3` means there is a bug in the backward-pass derivation — most likely a missing `1/m` factor, a wrong transpose, or a ReLU mask error. Do not train; debug the gradient first. Lecture 2 Sections 10 and 11.
- D) The finite-difference gradient is wrong because `eps` is too small.

---

**Q10.** You train an MLP for 30 epochs and observe that the training loss continues to decrease monotonically while the test accuracy plateaus at epoch 12 and then *slowly decreases* by epoch 28. The honest interpretation is:

- A) The learning rate is too high.
- B) The model is overfitting: it is learning training-set features that do not generalize. The right move is to stop at or near epoch 12 (early stopping), or to add regularization (dropout, weight decay, batch normalization — Challenge 1). Continuing to train will only hurt the test number further. Lecture 3 Section 6.
- C) The optimizer should be switched from SGD to Adam.
- D) MNIST is too small; add more data.

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — Cross-entropy is the negative log-likelihood of the correct class, summed over the batch and (with one-hot labels) collapsing to a single nonzero term per example. The squared-error form (A) is what you would use for regression; the absolute-error form (C) is robust regression; the max difference (D) is nothing standard. Lecture 1, Section 6.

2. **A** — The full derivation in Lecture 2 Section 4: the Jacobian of softmax has the form `p_i (δ_{ij} - p_j)`; the gradient of `-log p_{y_i}` with respect to `z` cancels the softmax Jacobian against the inverse from the `log`; the result is `p - y` for each example, divided by `m` for the batch average. The simplification is the reason "softmax + cross-entropy" is the canonical pairing for multi-class classification.

3. **B** — `np.exp(z)` overflows when `z > 709` (the limit of `float64`); on real logits early in training, values of 800 or 1000 are not uncommon; the naive `exp(z) / sum(exp(z))` returns `inf / inf = nan`. The fix is to subtract the max so the largest exponent is `exp(0) = 1`. The math is unchanged; the numerics are stable. Lecture 1 Section 5.

4. **C** — The matrix-calculus derivation: `z_{i,j} = sum_p x_{i,p} w_{p,j} + b_j`; partial derivative with respect to `w_{p,q}` is `x_{i,p} δ_{jq}`; summing over the batch gives the `(p, q)` entry of `X.T @ dZ1`. The shape `(n_in, h)` matches `W1`. Lecture 2 Sections 5 and 8.

5. **B** — The chain rule says the gradient of the loss with respect to an early-layer parameter is the product of the local gradients along the backward path. For sigmoid activations with derivative bounded above by `0.25`, deep networks produce vanishingly small early-layer gradients. ReLU's derivative is `1` on the positive side, so the products do not vanish (in the active direction). The fix was a major reason ReLU replaced sigmoid as the default. Lecture 2 Section 12.

6. **B** — A ReLU neuron whose pre-activation is always negative outputs zero forever and has zero gradient forever. The neuron is "dead." Fixes: smaller learning rate, He initialization (which keeps initial activations small), Leaky ReLU or ELU (non-zero negative-side gradient). Lecture 2 Section 12.

7. **A** — Random init + softmax over `k = 10` classes gives roughly uniform predictions, which produce cross-entropy `-log(1/10) = log(10) ≈ 2.30`. The C5 sanity check is the first thing to run after writing your forward pass; if the initial loss is far from `log(k)`, there is a bug. Lecture 1 Section 8.

8. **A** — `60_000 / 128 = 468.75`, so 468 full batches plus one partial batch of 16. Total: 469 SGD updates per epoch. Shuffling is what prevents degenerate gradient sequences when the data is sorted by class. Lecture 3 Section 2.

9. **C** — Relative error of `0.8` is two orders of magnitude above the `1e-5` accept threshold. The analytical gradient is wrong. The most common causes are listed in Lecture 2 Section 11 — missing `/m`, wrong transpose, ReLU-mask error from the wrong array. Do not start training; debug the backward pass on a 5-example batch until the gradient check passes. Lecture 2 Section 10.

10. **B** — Slow test-accuracy decay alongside continued training-loss decrease is the textbook overfitting signature. Stop training at the test-accuracy maximum (epoch 12), or add regularization. The other options miss the point: lowering the learning rate would slow training but not fix the overfit; Adam would not help because the issue is generalization, not optimization; more data is the *underlying* fix but is rarely available on a fixed benchmark. Lecture 3 Section 6; the dropout/batchnorm challenge (Challenge 1) is the practical follow-up.

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
