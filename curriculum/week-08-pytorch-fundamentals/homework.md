# Week 8 — Homework

Six problems, about six hours total. Commit each in your Week 8 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — Optimizer comparison (1 hour)

Train your `FashionMLP` (from Exercise 3) for 10 epochs at each of the following optimizer configurations, keeping everything else fixed (`n_hidden=128`, `batch_size=128`, `random_state=42`):

1. `torch.optim.SGD(lr=0.01)` (plain SGD)
2. `torch.optim.SGD(lr=0.01, momentum=0.9)` (SGD + Nesterov-like momentum)
3. `torch.optim.Adam(lr=1e-3)` (Adam, default Week 8)
4. `torch.optim.AdamW(lr=1e-3, weight_decay=1e-4)` (Adam with decoupled weight decay; Loshchilov-Hutter 2019)

For each:

1. Record the test accuracy per epoch.
2. Record the wall-clock time per epoch.

Save the figure `homework/01-optimizers.png` showing all four test-accuracy curves on the same axes. Add a legend with the optimizer name and the final test accuracy in parentheses.

In `homework/01-optimizers.md` (~150 words):

1. Which optimizer reaches `test_acc > 0.85` fastest (in epochs)? In wall-clock seconds?
2. Which optimizer has the highest final accuracy after 10 epochs?
3. The honest C5 reading: "Adam-family optimizers reach the same eventual accuracy as well-tuned SGD on this task, but they require less hyperparameter search. Adam is the right default; SGD with momentum is the right choice once you have invested time in tuning the schedule."
4. Confirm or refute in your data.

References:
- `torch.optim.SGD`: <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>
- `torch.optim.Adam`: <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>
- `torch.optim.AdamW`: <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>

---

## Problem 2 — Hidden-layer width sweep on FashionMNIST (1 hour)

Train the `FashionMLP` for 8 epochs at each of `n_hidden ∈ [16, 32, 64, 128, 256, 512, 1024]`, keeping `optimizer = Adam(lr=1e-3)`, `batch_size = 128`, `random_state = 42`. For each:

1. Record the final test accuracy.
2. Record the parameter count: `784*h + h + h*10 + 10`.
3. Record the wall-clock training time for 8 epochs.

Save the figure `homework/02-width-sweep.png` as a 2-panel figure: left, test accuracy vs. `n_hidden` (log-x axis); right, parameter count vs. wall-clock time (both on log-log).

In `homework/02-width-sweep.md` (~150 words):

1. At what width does the test accuracy plateau on FashionMNIST?
2. Compare to Week 7 Problem 2 (the same sweep on MNIST). MNIST plateaus at `h=128`; what does FashionMNIST do?
3. The parameter-count-vs-time scaling: is it linear in `h`? Why is the training time *not* a perfect linear function of parameter count, even at this scale?
4. The honest reading: "FashionMNIST is harder than MNIST. The same MLP architecture that hits 98% on MNIST hits 88% on FashionMNIST. The capacity gap is real, but the *architecture* is the bigger issue — a CNN closes most of the gap, as Challenge 1 shows."

---

## Problem 3 — Manual gradient check on a custom layer (1 hour)

Implement a custom activation function — Swish, defined as `swish(x) = x * sigmoid(x)` (Ramachandran, Zoph, Le 2017; <https://arxiv.org/abs/1710.05941>) — and verify its autograd-computed gradient against a finite-difference numerical gradient.

In `homework/03-swish.py`:

1. Implement `swish(x: torch.Tensor) -> torch.Tensor` as one line using `torch.sigmoid`.
2. Compute `dL/dx` via autograd on a sample input.
3. Compute `dL/dx` via finite differences using the central-difference formula from Week 7 Lecture 2 Section 10: `(L(x + eps) - L(x - eps)) / (2 * eps)` with `eps = 1e-5`.
4. Compute the relative error and assert it is `< 1e-5`.
5. Do the same for a Module wrapping Swish (`class SwishModule(nn.Module)`).

In `homework/03-swish.md` (~150 words):

1. Why does the autograd-computed gradient agree with finite differences? (Hint: PyTorch's autograd is computing the chain rule, and the chain rule is exactly what finite differences approximate; the only difference is the eps-precision of the numerical approach.)
2. The relative error you observed.
3. The PyTorch convention for custom activations: should you write a Module, a function, or both? (Hint: <https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html> shows the official approach for Swish, which PyTorch calls `nn.SiLU`. You are reimplementing it from scratch.)

---

## Problem 4 — Checkpoint resume from interruption (1 hour)

Simulate an interrupted training run and verify your checkpoint mechanism works.

In `homework/04-resume.py`:

1. Train the `FashionMLP` for 5 epochs and save a checkpoint every epoch:

   ```python
   torch.save({
       "epoch": epoch,
       "model": model.state_dict(),
       "optimizer": optimizer.state_dict(),
       "rng_state": torch.get_rng_state(),
   }, f"homework/04-ckpt-epoch-{epoch}.pt")
   ```

2. Stop after epoch 2 (simulate a crash with `sys.exit` or just truncating the loop).
3. In a separate function `resume_training(ckpt_path)`, load the checkpoint and continue training to epoch 5. Restore the model parameters, the optimizer state, *and* the RNG state.
4. Train a *second* model end-to-end for 5 epochs without interruption.
5. Assert that the two final test accuracies agree to within 0.5 percentage points. The match should be near-exact because everything is seeded; the small allowable difference is for floating-point summation order in the optimizer.

In `homework/04-resume.md` (~150 words):

1. The reproducibility check. Did your interrupted-then-resumed run produce the same final test accuracy as the uninterrupted run?
2. What is the most-easily-forgotten piece of state to save? (Hint: the RNG state. Without it, the DataLoader shuffle in epoch 3 is different from the uninterrupted run.)
3. What about the *dataset* state? (Hint: for FashionMNIST it does not matter because the dataset is in memory, but for streaming datasets it does.)

Reference: <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>

---

## Problem 5 — Confusion matrix and per-class accuracy on FashionMNIST (45 min)

Take your trained `FashionMLP` from Exercise 3. On the test set:

1. Compute the 10x10 confusion matrix: `cm[i, j]` is the number of test examples with true label `i` predicted as `j`.
2. Compute per-class accuracy: `acc[c] = cm[c, c] / cm[c, :].sum()`.
3. Identify the worst-performing class. Identify the most-common confusion (which off-diagonal entry of `cm` is largest).

Save the confusion-matrix heatmap as `homework/05-confusion.png` with class names (T-shirt/top, Trouser, ..., Ankle boot) on both axes and the value annotated in each cell.

In `homework/05-confusion.md` (~150 words):

1. What is the per-class accuracy distribution? (Best, worst, mean.)
2. Which two classes are most confused? (For FashionMNIST, "Shirt" is famously the worst class — it is visually similar to T-shirt/top, Pullover, and Coat.)
3. Is the confusion symmetric? (If `cm[shirt, t_shirt] = 80` and `cm[t_shirt, shirt] = 30`, the answer is no, and the asymmetry is informative.)
4. Look at five misclassified examples (use `matplotlib.imshow` on the raw image, before normalization). Are they ambiguous (the image really does look like both classes) or wrong on clear images? Describe one of each.

Hints:
- Convert the confusion matrix to NumPy first: `cm = sklearn.metrics.confusion_matrix(y_true, y_pred)`.
- The class names come from `train_loader.dataset.classes` (a list of strings, in order).

---

## Problem 6 — Reflection (30 min)

Write `homework/06-reflection.md` (250-400 words) answering:

1. The Week 7 NumPy MLP took ~200 lines and was your first deep-learning artifact. The Week 8 PyTorch MLP is ~50 lines and produces the same numbers. Was the trade worth it? Where do you suspect you will reach for NumPy again versus where you will always reach for PyTorch?
2. The eight-line training loop from Lecture 2 Section 7 — `for X, y in train_loader: zero, forward, loss, backward, step` — how many times did you have to look it up before you could write it from memory? Did the experience of writing the manual SGD step in Exercise 1 help, or was it busywork?
3. Lecture 1 Section 10 lists eight autograd traps; Lecture 2 Section 9 lists three training failures; Lecture 3 Section 8 lists three more. Which (if any) bit you this week? Which one are you now most paranoid about?
4. The Week 7 mini-project was a NumPy MLP on a tabular dataset. The Week 8 mini-project is a PyTorch image classifier on FashionMNIST. The deliverables are both ~600-word reports. What did you learn this week that you would *not* have learned by reading the PyTorch tutorial cold, without the Week 7 grounding?
5. After Week 8, what is the one habit you want to carry into Week 9 (CNNs)? "Train on CPU first, then move to GPU" carries over; "save the state_dict, not the model" carries over; "the eight-line loop is invariant" carries over.

Honest is more valuable than polished.

---

## Time budget

| Problem | Time |
|--------:|----:|
| 1 | 1 h |
| 2 | 1 h |
| 3 | 1 h |
| 4 | 1 h |
| 5 | 45 min |
| 6 | 30 min |
| **Total** | **~5 h 15 min** |

(The schedule allocates 6h for homework; the remaining ~45 minutes is buffer for the FashionMNIST download and for any debugging on the optimizer sweep.)

When done, push your Week 8 repo and start (or finish) the [challenges](./challenges/) and the [mini-project](./mini-project/README.md).

## Grading rubric

| Score | Threshold |
|------:|-----------|
| **A** | All six problems complete; the eight-line training loop is correct in every script; the reflection (Problem 6) shows specific learning rather than general appreciation; figures are labeled and legible. |
| **B** | Five problems complete or all six but with one figure missing labels or one writeup under 100 words. The reflection still names specifics. |
| **C** | Four problems complete. The training loop works; the reflection is generic. |
| **D / F** | Three or fewer problems. Code does not run. |
