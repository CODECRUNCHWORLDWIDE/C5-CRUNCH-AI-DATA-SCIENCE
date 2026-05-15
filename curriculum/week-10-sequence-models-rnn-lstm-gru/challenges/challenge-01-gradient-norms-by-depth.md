# Challenge 1 — Measure the Vanishing Gradient on Your Own Machine

> Reproduce the gradient-norm-by-timestep curve from Pascanu, Mikolov, Bengio 2013 (<https://arxiv.org/abs/1211.5063>, Figure 2) on a 200-step vanilla RNN. Quantify how rapidly the gradient at the final step vanishes as you backpropagate through the unroll. Then repeat with an LSTM and observe that the cell-state pathway delays the decay by an order of magnitude.

**Estimated time:** 90-120 minutes.
**Deliverable:** A Python script `challenge-01-solution.py` plus a 200-word write-up `challenge-01-writeup.md` explaining what you found. Both committed to your portfolio repo under `week-10/challenges/`.

---

## The setup

Build a single-layer vanilla RNN with `hidden_size=64` and `input_size=4`. Use PyTorch's default initialization (uniform in `[-1/sqrt(64), +1/sqrt(64)]`). Pass it a single batch element (`batch_size=1`) consisting of `T=200` random standard-normal input vectors of dimension 4. Initialize `h_0` to a random unit-norm vector.

Run the RNN forward; collect the hidden states `h_1, h_2, ..., h_T`. Define a scalar loss as `L = sum(h_T)` (the sum of the final hidden-state's coordinates). Call `loss.backward()`. PyTorch will compute the gradient `dL/dh_t` for every `t` along the way; we will read these gradients off via forward hooks.

**The measurement.** For each `t ∈ {1, ..., T}`, record `||dL/dh_t||_2`. Plot the result on a log scale. The Pascanu paper's Figure 2 shows this curve for a similar setup; it decays exponentially from `t = T` (where the gradient is 1 per coordinate, totaling `sqrt(64) ≈ 8`) down to `t = 1` (where the gradient should be 1e-30 or smaller — limited by floating-point underflow).

---

## Requirements

1. **Hook every hidden state.** Use `tensor.register_hook(callback)` on each `h_t` to capture its gradient when backward runs. The callback should append the gradient's L2 norm to a list keyed by timestep. Reference: <https://pytorch.org/docs/stable/generated/torch.Tensor.html#torch.Tensor.register_hook>.
2. **Produce two plots, both log-scale on y.** Plot 1: gradient norm vs. timestep for the vanilla RNN. Plot 2: same setup but with `nn.LSTM(input_size=4, hidden_size=64, num_layers=1)`. For the LSTM, hook the *cell state* `c_t` rather than the hidden state, because the cell state is the pathway that the LSTM saves from vanishing.
3. **Verify the geometric-decay rate.** Fit a line to `log(||grad||)` vs. `t` for the RNN. The slope should be approximately `log(||W_hh||_2)` where `||W_hh||_2` is the spectral norm of the recurrent weight matrix. Report both numbers.
4. **Use `torch.manual_seed(42)`** at the top of the script. The C5 convention.

---

## What you should find

For the vanilla RNN with PyTorch's default initialization, `||W_hh||_2` is roughly 1.0 (the initialization is engineered to put it near the boundary). The gradient norm should decay slowly at first and then exponentially, with a slope corresponding to `||W_hh||_2 * <average tanh-derivative>` — which is below 1 because `1 - tanh^2` is bounded by 1 and is typically much smaller after the first few steps. In practice, the gradient at `t = 1` is roughly `1e-8` to `1e-12`, vs. about `8` at `t = T`. That is a 10-12 order-of-magnitude collapse.

For the LSTM cell state, the gradient norm should decay much more slowly. The exact rate depends on the forget-gate activations along the trajectory, which in turn depend on the input statistics. With random-Gaussian inputs and the default forget-bias initialization (`b_f = 0`, so `sigmoid(b_f) ≈ 0.5`), expect the cell-state gradient at `t = 1` to be on the order of `1e-2` to `1e-4` — still substantially less than at `t = T`, but five to ten orders of magnitude better than the vanilla RNN.

If you apply the Jozefowicz fix (set the forget-gate bias to 1, so `sigmoid(b_f) ≈ 0.73`), the LSTM cell-state gradient at `t = 1` should be on the order of `0.1` to `0.5` — barely any decay at all over 200 steps. This is the empirical evidence for why the forget-bias-1 initialization is so common in production LSTMs.

---

## Write-up requirements

In `challenge-01-writeup.md` (about 200 words), cover:

1. **The two plots, embedded.** Embed both as `![](./grad-norms.png)` and `![](./lstm-grad-norms.png)`.
2. **The numerical claim.** "Vanilla RNN: gradient at step 1 is `X` (vs. `Y` at step 200), decay rate `Z` per step. LSTM cell-state: gradient at step 1 is `X` (vs. `Y` at step 200), decay rate `Z` per step."
3. **The interpretation.** One sentence explaining why the LSTM's additive cell-state update produces this difference. Refer to Lecture 2 Section 4.
4. **Honest caveats.** State at least one limitation of your measurement (e.g., "Only one random seed", "Random Gaussian inputs are unrealistic", "The decay rate depends on the forget-gate trajectory which itself depends on the input").

---

## Hints

- The forward hook pattern: `h_t.register_hook(lambda g, t=t: grads[t].append(g.norm().item()))`. The `t=t` default-argument capture is the standard Python idiom for closing over the loop variable correctly.
- For the LSTM, you need to expose the cell state at every timestep. The easiest way: unroll with `nn.LSTMCell` rather than `nn.LSTM`, and explicitly hook each `c_t`. The same pattern from Exercise 2.
- The `nn.LSTMCell` forward returns a fresh tensor for `c_t` at each step; you can hook each one individually. Make sure you do not free the tensors before backward runs by storing the list of cell states in a Python variable.
- The plot should be on `matplotlib`'s default log scale (`plt.yscale("log")`). The vanilla RNN curve will look almost perfectly linear on the log scale (because the decay is geometric); the LSTM curve will be much flatter.

---

## Stretch (optional)

Repeat the measurement for a GRU. The GRU's hidden-state-to-hidden-state Jacobian is more complicated than the LSTM's cell-state Jacobian (it involves both the update gate and the reset gate), so the gradient flow is somewhere between the vanilla RNN's and the LSTM's. Empirically, the GRU tends to be closer to the LSTM than the RNN. Verify or refute on your data.

---

## Acceptance criteria

- [ ] `challenge-01-solution.py` runs end-to-end and produces both PNG plots.
- [ ] `python -m py_compile challenge-01-solution.py` succeeds.
- [ ] `challenge-01-writeup.md` is 150-250 words; embeds both plots; reports the four numerical claims (RNN start/end gradients, LSTM start/end gradients).
- [ ] Both plots use a log-scale y-axis.
- [ ] `torch.manual_seed(42)` is set at the top of the script.

The C5 conviction here: you have *seen* the vanishing-gradient problem on your own machine. It is no longer a slogan, no longer a textbook claim, no longer "something in the LSTM paper." It is a number on your screen. Once you have done this measurement once, you will never again forget why the LSTM exists.

---

## References

- **Pascanu, R.; Mikolov, T.; Bengio, Y. (2013).** "On the difficulty of training Recurrent Neural Networks." *ICML 2013.* arXiv:1211.5063. <https://arxiv.org/abs/1211.5063>. Figure 2 is what you are reproducing.
- **Hochreiter, S. (1991).** German-language diploma thesis; the original vanishing-gradient analysis. Untranslated and not widely available; the C5 lectures cite Pascanu et al. as the modern reframing.
- **PyTorch forward hooks:** <https://pytorch.org/docs/stable/generated/torch.Tensor.html#torch.Tensor.register_hook>.
- **`torch.linalg.svdvals`:** <https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html>. Useful for measuring `||W_hh||_2`.
