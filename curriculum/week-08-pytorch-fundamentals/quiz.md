# Week 8 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** You write `t = torch.from_numpy(arr)` where `arr` is a NumPy array. You then run `t[0] = 99.0`. Which of the following is true?

- A) `arr[0]` is unchanged; `torch.from_numpy` always copies.
- B) `arr[0]` is now `99.0`; `torch.from_numpy` produces a zero-copy view that shares memory.
- C) The assignment errors because tensors are immutable.
- D) `arr[0]` is now `99.0` only if `arr` had `dtype=float32`.

---

**Q2.** Inside a training loop, what is the correct order of the four critical operations?

- A) `loss.backward(); optimizer.step(); optimizer.zero_grad(); model(X)`
- B) `optimizer.zero_grad(); loss.backward(); optimizer.step(); model(X)`
- C) `model(X) -> loss; optimizer.zero_grad(); loss.backward(); optimizer.step()`. The "zero before backward" order is mandatory; backward must precede step.
- D) `model(X) -> loss; loss.backward(); optimizer.zero_grad(); optimizer.step()`

---

**Q3.** The reason `nn.CrossEntropyLoss` expects raw logits (not softmax-applied probabilities) is:

- A) It is a backwards-compatibility quirk inherited from an older version of PyTorch.
- B) `CrossEntropyLoss` applies `log_softmax` internally for numerical stability. Applying softmax in your forward pass and then passing probabilities to `CrossEntropyLoss` causes a `log_softmax(softmax(...))` chain, which is wrong math and reintroduces the overflow / underflow bugs that the "subtract the max" trick from Week 7 exists to prevent.
- C) `CrossEntropyLoss` does not accept negative numbers; logits can be negative but softmax outputs cannot.
- D) `CrossEntropyLoss` requires the input to be in `[0, 1]`.

---

**Q4.** Inspecting a trained model: `model.fc1.weight.shape` returns `(128, 784)`. The model was built with `self.fc1 = nn.Linear(784, 128)`. Why is the weight `(128, 784)` and not `(784, 128)`?

- A) PyTorch has a bug; the weight should be transposed.
- B) `nn.Linear(in_features, out_features)` stores `weight` with shape `(out_features, in_features)` and computes `output = input @ weight.T + bias`. This is the standard math convention `y = Wx + b`, not the Week 7 NumPy convention `y = xW + b`. See Lecture 2 Section 2.
- C) The weight is stored transposed because PyTorch tensors are column-major.
- D) The weight shape depends on the device; CUDA tensors store the transpose.

---

**Q5.** You are training on a GPU. Your loop reads:

```python
for X, y in train_loader:
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
```

You see `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu`. The fix is:

- A) Add `model.to(device)` before the loop.
- B) Add `X = X.to(device); y = y.to(device)` inside the loop.
- C) Both A and B; the model must be on GPU and every batch must be moved to GPU on each iteration. The model move is once; the batch moves are per-iteration. Lecture 3 Section 5.
- D) Change `optim.SGD` to `optim.Adam`.

---

**Q6.** You omit `optimizer.zero_grad()` from your training loop. The most likely symptom after a few iterations is:

- A) The loss is constant.
- B) The loss explodes upward. Gradients accumulate across iterations, so by step 10 the effective gradient is roughly 10x what it should be; the parameter updates are oversized; the loss diverges. Lecture 2 Section 9 Failure 2 and Lecture 1 Section 10 Trap 1.
- C) Validation accuracy plateaus at 10%.
- D) PyTorch raises a `RuntimeError`.

---

**Q7.** The recommended way to save a trained model is:

- A) `torch.save(model, "model.pt")`. Pickles the entire object including class definitions; the simplest API.
- B) `torch.save(model.state_dict(), "model.pt")`. Saves only the parameter and buffer tensors as a dict; the load side reconstructs the class first and then loads the state. Portable across PyTorch versions and immune to the pickle attack surface. Lecture 3 Section 7.
- C) `pickle.dump(model, open("model.pkl", "wb"))`.
- D) `model.save("model.pt")` (no such API).

---

**Q8.** Inside an evaluation pass, you omit the `with torch.no_grad():` wrapper. The most likely consequences are:

- A) The loop runs but produces incorrect predictions.
- B) The loop runs but uses roughly twice the memory it should and is roughly half the speed; the model also retains autograd graphs for every batch, which can OOM on a large validation set. Predictions themselves are correct; the issue is resource usage. Lecture 1 Section 7 and Lecture 2 Section 8.
- C) The loop errors with `RuntimeError: cannot evaluate while training`.
- D) The model silently switches into training mode.

---

**Q9.** You construct a `DataLoader` with `num_workers=4, pin_memory=True`. On a tiny in-memory dataset (FashionMNIST loaded into RAM), the speedup from these flags over `num_workers=0, pin_memory=False` is:

- A) Roughly 4x; one worker per CPU core.
- B) None; the in-memory `__getitem__` is so fast that worker overhead exceeds the gain. The flags are designed for the case where `__getitem__` is slow (image-from-disk, augmentation, etc.); for an in-memory dataset, the default `num_workers=0` is the right choice. Lecture 3 Section 2.
- C) Roughly 2x; the prefetch helps even on tiny data.
- D) Negative; the script hangs.

---

**Q10.** On macOS, you set `num_workers=4` on the DataLoader and run your training script. The script hangs at the start of the first epoch with high CPU usage but no output. The fix is:

- A) Switch to `num_workers=0`.
- B) Wrap the entry point in `if __name__ == "__main__":`. On macOS (and Windows), multi-worker DataLoaders require the script to be guardable for the multiprocessing module to safely re-import it in subprocesses. Without the guard, the workers re-execute the training loop, spawn their own workers, and the system fork-bombs itself. Lecture 3 Section 8 Failure 5.
- C) Set `pin_memory=True`.
- D) Switch from `optim.SGD` to `optim.Adam`.

---

## Answer key

<details>
<summary>Click to reveal</summary>

1. **B** — `torch.from_numpy(arr)` is a zero-copy view. `torch.tensor(arr)` is the copy alternative. The aliasing is documented at <https://pytorch.org/docs/stable/generated/torch.from_numpy.html>; the test in Exercise 1 (`test_tensor_from_numpy_shares_memory`) is the experiment. Lecture 1 Section 1.

2. **C** — The eight-line loop from Lecture 2 Section 7. The order is "forward → loss → zero → backward → step." The zero must precede the backward (otherwise you clear the gradients you just computed); the backward must precede the step (otherwise the step uses stale gradients). Memorize this.

3. **B** — `CrossEntropyLoss` does `log_softmax + nll_loss` in one numerically-stable op. Doing softmax yourself first breaks the stability *and* the math. The PyTorch reference is at <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>; the same trick was the headline of Week 7 Lecture 1 Section 5.

4. **B** — `nn.Linear(in, out)` stores `weight` with shape `(out, in)` and computes `y = x @ W.T + b`. The convention matches the `y = Wx + b` math notation, where `W` is `out × in`. PyTorch is consistent on this across every layer (Conv2d weights are also output-channels-first). Lecture 2 Section 2.

5. **C** — Both. The model move happens once before training; the batch moves happen on every iteration. Missing either produces the device-mismatch error. Lecture 3 Section 5 has the full pattern.

6. **B** — The classic missing-`zero_grad` symptom. Gradients accumulate; effective step size grows linearly; loss diverges. The fix is one line. If you ever see "my loss went to infinity in epoch 1," this is the first thing to check. Lecture 2 Section 9.

7. **B** — The `state_dict` pattern is the production standard. Pickle-the-whole-object works narrowly but is brittle across PyTorch versions and is a security risk. The official tutorial is at <https://pytorch.org/tutorials/beginner/saving_loading_models.html>. Lecture 3 Section 7.

8. **B** — Without `no_grad`, the eval pass builds the autograd graph for every batch. Predictions are still correct (the forward pass is unchanged), but memory doubles and speed halves. On large eval sets, OOM is also possible. The fix is one line.

9. **B** — `num_workers` pays off when `__getitem__` is slow (disk reads, image augmentation, tokenization). For in-memory datasets, the worker overhead exceeds the parallelism gain. The C5 rule: `num_workers=0` for in-memory data, `num_workers=2-4` for on-disk image data, more for heavily augmented production pipelines. Lecture 3 Section 2.

10. **B** — The `if __name__ == "__main__":` guard is required on macOS and Windows for multi-worker DataLoaders because both platforms use a multiprocessing start method that re-imports the script. Linux's `fork` start method does not need the guard. The fix is the C5 default in every entry-point script. Lecture 3 Section 8 Failure 5.

</details>

If you got 7 or fewer right, re-read the lectures for the topics you missed. If 9+, you are ready for the [homework](./homework.md).
