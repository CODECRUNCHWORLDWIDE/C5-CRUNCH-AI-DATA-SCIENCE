# Week 7 — Homework

Six problems, about six hours total. Commit each in your Week 7 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — Learning-rate sweep (1 hour)

Train your two-layer MLP on the 10,000-example MNIST subset (from Exercise 3) for 5 epochs at each of `learning_rate ∈ [0.001, 0.01, 0.1, 0.5, 1.0]`, keeping everything else fixed (`h = 128`, `batch_size = 128`, `random_state = 42`). For each:

1. Record the training loss at each epoch.
2. Record the final test accuracy.

Save the figure `homework/01-lr-sweep.png` showing all five training-loss curves on the same axes (use a log scale on the y-axis). Add a legend with the learning rate and the final test accuracy in parentheses.

In `homework/01-lr-sweep.md` (~150 words):

1. Which learning rate produced the best final test accuracy?
2. Which learning rate showed the cleanest convergence curve (smooth, monotonically decreasing)?
3. The honest reading: "as the learning rate increases past the right value, the loss curve becomes noisier and eventually diverges." Confirm in your data.
4. What is the worst-case behavior at `lr = 1.0`? Does it converge at all?

---

## Problem 2 — Hidden-layer width sweep (1 hour)

Train your MLP on the same 10k MNIST subset for 10 epochs at each of `n_hidden ∈ [16, 32, 64, 128, 256, 512]`, keeping `learning_rate = 0.1`, `batch_size = 128`, `random_state = 42`. For each:

1. Record the final test accuracy.
2. Record the parameter count: `784 * h + h + h * 10 + 10`.
3. Record the wall-clock training time.

Save the figure `homework/02-width-sweep.png` as a 2-panel figure: left, test accuracy vs. `n_hidden` (log-x axis); right, parameter count vs. wall-clock time (both on log-log).

In `homework/02-width-sweep.md` (~150 words):

1. At what width does the test accuracy plateau? Why is there no benefit to going wider beyond that point on MNIST?
2. The parameter-count-vs-time scaling: is it linear? Sublinear? Superlinear? Why?
3. The C5 reading: "MNIST is a small task; even a 16-hidden-neuron MLP reaches 90% test accuracy. The marginal benefit of additional capacity is small, but the wall-clock cost is linear." Confirm in your data.

---

## Problem 3 — Implement Leaky ReLU (1 hour)

Replace the ReLU in your MLP with Leaky ReLU: `f(z) = max(0.01 * z, z)`. The derivative is `1` for `z > 0` and `0.01` for `z <= 0`.

In `homework/03-leaky-relu.py`:

1. Implement `leaky_relu(z, alpha)` and `leaky_relu_grad(z, alpha)`.
2. Train the MLP on the 10k MNIST subset for 10 epochs with `alpha = 0.01`. Record final test accuracy.
3. Train the same MLP with vanilla ReLU. Record final test accuracy.
4. Run gradient-checking on a 5-example tiny MLP with Leaky ReLU. The relative error should be `< 1e-5`.

In `homework/03-leaky-relu.md` (~150 words):

1. Did Leaky ReLU outperform vanilla ReLU on this task? By how much?
2. Inspect the hidden-layer activations after training: how many neurons have output `> 0` for at least one training example? (This is the count of "alive" neurons.) Compare for both activations.
3. The honest C5 reading: "Leaky ReLU rarely helps on small networks where dead neurons are not a problem. It is a fix for a problem that mostly manifests in deep networks."

---

## Problem 4 — He vs. Xavier vs. naive initialization (45 min)

Three initialization schemes:

- **Naive**: `W ~ N(0, 0.01)`.
- **Xavier (Glorot)**: `W ~ N(0, sqrt(1 / n_in))`.
- **He**: `W ~ N(0, sqrt(2 / n_in))`.

Train the MLP on the 10k MNIST subset for 10 epochs with each initialization, all other hyperparameters fixed. Record the initial loss (before any training) and the final test accuracy.

In `homework/04-init.md` (~150 words):

1. Initial loss with each? (Expected: all three should be around `log(10) = 2.30`.)
2. Final test accuracy with each?
3. Convergence speed: which one reaches `test_acc > 0.95` first (in terms of epochs)?
4. The C5 reading: "He initialization is calibrated for ReLU networks; the factor of 2 compensates for ReLU zeroing out half the activations. Xavier is calibrated for sigmoid/tanh. Naive `N(0, 0.01)` works on shallow networks but does not scale to deep ones."

---

## Problem 5 — Confusion matrix and per-class accuracy (45 min)

Take your trained MLP from Exercise 3 (or retrain it). On the test set:

1. Compute the 10x10 confusion matrix: `cm[i, j]` is the number of test examples with true label `i` predicted as `j`.
2. Compute per-class accuracy: `acc[c] = cm[c, c] / cm[c, :].sum()`.
3. Identify the worst-performing class. Identify the most-common confusion (which off-diagonal entry of `cm` is largest).

Save the confusion-matrix heatmap as `homework/05-confusion.png` with class indices on both axes and the value annotated in each cell.

In `homework/05-confusion.md` (~150 words):

1. What is the per-class accuracy distribution? (Best, worst, mean.)
2. Which two classes are most confused? Is the confusion symmetric? (For MNIST: `4 ↔ 9` and `7 ↔ 1` are common; `0 ↔ 8` is rare.)
3. Look at five misclassified examples. Are they reasonable confusions (the image is genuinely ambiguous) or is the model wrong on clear examples? Describe one of each.

---

## Problem 6 — Reflection (30 min)

Write `homework/06-reflection.md` (250-400 words) answering:

1. The whole forward + backward + training loop for a 2-layer MLP is under 100 lines of NumPy. Did the brevity surprise you? Where did the implementation get harder than you expected — the softmax overflow, the backward-pass derivation, the gradient check, or somewhere else?
2. The gradient check is non-negotiable: every from-scratch backward pass has a bug at first. Did yours? What was it, and how did the gradient check expose it?
3. Lecture 2 Section 11 lists the four canonical bugs: shape mismatch, batch-averaging off-by-one, sign error, softmax overflow. Which one (if any) bit you this week? Which one are you now most paranoid about?
4. PyTorch's `loss.backward()` would let you skip Lecture 2 entirely. After spending two days deriving the backward pass by hand, do you regret the framework-free week, or are you glad? Be specific.
5. After Week 7, what is the one habit you want to carry into Week 8 (PyTorch)? "Always run the initial-loss sanity check" carries over; "always gradient-check at the start of a new architecture" arguably does too, because PyTorch's autograd has had its own bugs in the past.

Honest is more valuable than polished.

---

## Time budget

| Problem | Time |
|--------:|----:|
| 1 | 1 h |
| 2 | 1 h |
| 3 | 1 h |
| 4 | 45 min |
| 5 | 45 min |
| 6 | 30 min |
| **Total** | **~5 h** |

(The schedule allocates 6h for homework; the remaining ~1h is buffer for the first MNIST download and for any debugging on the learning-rate sweep where higher rates may diverge.)

When done, push your Week 7 repo and start (or finish) the [challenges](./challenges/) and the [mini-project](./mini-project/README.md).
