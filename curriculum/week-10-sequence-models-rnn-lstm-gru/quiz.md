# Week 10 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** The vanilla RNN recurrence is `h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)`. If `input_size = 20` and `hidden_size = 64`, what are the shapes of `W_xh`, `W_hh`, and `b_h` respectively?

- A) `(64, 20)`, `(64, 64)`, `(64,)`
- B) `(20, 64)`, `(64, 64)`, `(64,)`
- C) `(64, 20)`, `(20, 64)`, `(20,)`
- D) `(20, 64)`, `(64, 64)`, `(20,)`

---

**Q2.** In Lecture 1 Section 5, we argued that the gradient of the loss at time `t` with respect to a parameter "near time `0`" decays geometrically when `||W_hh||_2 < 1`. The geometric factor is approximately:

- A) `||W_hh||_2 / t`
- B) `(||W_hh||_2)^t`
- C) `t * ||W_hh||_2`
- D) `||W_hh||_2 + t`

---

**Q3.** Gradient clipping (`torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)`) is the standard fix for which RNN training failure mode?

- A) The vanishing-gradient problem. Clipping rescales the small gradients up to a workable magnitude.
- B) The exploding-gradient problem. Clipping caps the gradient norm so the optimizer cannot overshoot. It does *not* fix vanishing gradients — a vanished gradient is information-poor and rescaling it up does not recover the lost signal.
- C) Overfitting. Clipping reduces the effective learning rate, providing regularization.
- D) The covariate-shift problem; it is the recurrent analog of BatchNorm.

---

**Q4.** The LSTM cell-state update is `c_t = f_t * c_{t-1} + i_t * g_t`. The Jacobian `dc_t / dc_{t-1}` is:

- A) `W_hh` (the recurrent weight matrix; same as vanilla RNN).
- B) `diag(f_t)` — a diagonal matrix whose entries are the forget-gate activations. The product of these diagonal Jacobians across timesteps is `diag(prod_t f_t)`, which does not vanish if the forget gates are near 1.
- C) `tanh'(c_{t-1})` — the derivative of the tanh activation on the previous cell state.
- D) The identity matrix; the cell state does not have a Jacobian because it is detached.

---

**Q5.** PyTorch's `nn.LSTMCell` packs the four affine transformations of the LSTM into a single weight tensor. In what order are the four gates concatenated along the output dimension?

- A) `[f, i, g, o]` — the order in which they appear in the Lecture 2 equations.
- B) `[i, f, g, o]` — input, forget, candidate, output. This is also the cuDNN convention. Mismatching this order is a famous source of bugs when implementing the cell from scratch.
- C) `[f, o, i, g]` — alphabetical.
- D) The order is undocumented and varies between PyTorch versions.

---

**Q6.** The GRU equations are `r_t = sigmoid(...)`, `z_t = sigmoid(...)`, `g_t = tanh(...)`, `h_t = (1 - z_t) * h_{t-1} + z_t * g_t`. Compared to an LSTM with the same hidden size, the GRU has approximately:

- A) The same number of parameters.
- B) 25-30% fewer parameters, because it has three affine transforms instead of the LSTM's four, and one state tensor instead of two.
- C) Twice as many parameters, because the reset gate adds an extra matmul.
- D) Half as many parameters; the GRU drops the cell state entirely and uses no candidate.

---

**Q7.** You call `pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)` on a CUDA tensor `emb` and a CUDA tensor `lengths`. PyTorch raises an error. What is most likely wrong?

- A) `emb` must be on CPU; only `lengths` can be on CUDA.
- B) `lengths` must be on CPU. The packing routine itself runs CPU-side and cannot read GPU memory directly. The standard idiom is `pack_padded_sequence(emb, lengths.cpu(), ...)`.
- C) `enforce_sorted=False` is incompatible with CUDA tensors.
- D) `pack_padded_sequence` requires `batch_first=False` and you must transpose the tensor first.

---

**Q8.** "Teacher forcing" during sequence-model training means:

- A) Feeding the model's own previous prediction `x_hat_{t-1}` as input at step `t`. This is the production / inference pattern.
- B) Feeding the *true* previous token `x_{t-1}` from the training data as input at step `t`, regardless of what the model would have predicted. This decouples per-step losses and speeds up training, at the cost of "exposure bias": the model has never seen its own mistakes at training time and may cascade into nonsense at inference.
- C) Using a much larger learning rate at the start of training and annealing it down.
- D) Adding a teacher network alongside the student RNN; the student is trained to match the teacher's predictions.

---

**Q9.** You are training a char-LM on a 700,000-character corpus. Running BPTT on the full corpus is infeasible. The standard fix is *truncated BPTT*: slice the corpus into chunks of length `T_chunk` and run BPTT within each chunk. Between chunks, the hidden state should be:

- A) Re-initialized to zero. This is the "reset every chunk" pattern; it loses cross-chunk context but is simple.
- B) Passed forward, with gradient. This causes the BPTT graph to grow without bound — exactly the problem we were trying to avoid.
- C) Passed forward with `.detach()`. The model gets the *values* of the previous state (so the recurrence is preserved across chunks for the forward pass) but the *gradient* does not flow back into earlier chunks. This is the standard truncated-BPTT recipe.
- D) Passed forward only on every fourth chunk; otherwise re-initialized.

---

**Q10.** Lecture 3 closed with three properties that the transformer has and the LSTM does not. Which of the following is *not* one of those three properties?

- A) Constant-time lookback: an attention layer reads any past position in one operation, vs. the LSTM's `O(T)` hidden-state propagation.
- B) Parallel training: the attention layer's forward pass parallelizes across the entire sequence length, vs. the LSTM's sequential unroll.
- C) `O(1)` gradient path from output to input: the transformer's attention is a single softmax-weighted lookup, vs. the LSTM's `O(T)` path through the cell state.
- D) Built-in translation equivariance: the transformer naturally handles inputs in any order. (This is *false* — attention is permutation-equivariant by default, which is why transformers need positional encodings; it is not an advantage over the LSTM, which has a natural sequence order built in.)

---

## Answer key

(Read after attempting.)

**Q1: A.** `W_xh` is `(hidden, input) = (64, 20)`; `W_hh` is `(hidden, hidden) = (64, 64)`; `b_h` is `(hidden,) = (64,)`. This matches PyTorch's `nn.RNNCell` convention exactly. The mnemonic: every weight matrix is `(output_dim, input_dim)`; the bias has shape `(output_dim,)`.

**Q2: B.** The product of `t` Jacobians, each with spectral norm at most `||W_hh||_2`, has spectral norm at most `(||W_hh||_2)^t`. For `||W_hh||_2 < 1` this decays geometrically; for `||W_hh||_2 > 1` it grows geometrically. The analysis is in Pascanu et al. 2013 section 2.2.

**Q3: B.** Clipping caps the gradient norm and is the standard fix for exploding gradients. It does nothing for vanishing gradients; rescaling a 1e-12 gradient up to magnitude 1.0 does not recover any information — the *direction* of a vanished gradient is dominated by floating-point noise, not by any meaningful learning signal. The vanishing case requires an architectural fix (LSTM, GRU, or attention), not an optimization fix.

**Q4: B.** The diagonal Jacobian `diag(f_t)` is the entire point of the additive cell-state update — it bypasses the matrix-product geometric-decay that kills the vanilla RNN. The product across timesteps `prod_t diag(f_t)` is still diagonal, with entries `prod_t f_t[i]`. If the forget gates are near 1 at every step, this product stays near 1; the gradient flows back losslessly. If the forget gates are near 0 at some step, the gradient is killed at that step — but in an "information-bearing" way (the network has decided to forget), not the "uninformative geometric decay" of the vanilla RNN.

**Q5: B.** PyTorch and cuDNN both use `[i, f, g, o]`. The Lecture 2 equations list them as `[f, i, g, o]` for pedagogical clarity (the forget gate is conceptually the most important), but PyTorch's packed-tensor layout is `[i, f, g, o]`. Exercise 2 has you debug this exact ordering; the test will fail with the wrong order. This convention is unfortunately not documented prominently in the PyTorch docs — the docs only say "the gates are packed in a specific order" and you have to read the cuDNN documentation or the PyTorch source to find out which.

**Q6: B.** GRU has three affine transforms (`r`, `z`, `g`) vs. LSTM's four (`f`, `i`, `g`, `o`). At matched hidden size, the GRU has about 75% of the LSTM's recurrent-body parameter count. The GRU also drops the cell state, so the state passed forward is a single tensor vs. the LSTM's two. The Chung et al. 2014 paper reports that the parameter savings does not cost quality on most sequence tasks.

**Q7: B.** `lengths` must be on CPU because the packing routine itself runs CPU-side. This is documented in the `pack_padded_sequence` docs but is one of the most common first-time errors. The standard idiom: `pack_padded_sequence(emb, lengths.cpu(), ...)`. Note that `emb` can (and should) stay on the GPU; only `lengths` needs to be moved.

**Q8: B.** Teacher forcing feeds the true context at training time. It is fast and clean; the downside is exposure bias (the model never trains on its own mistakes). The standard mitigation is scheduled sampling (Bengio et al. 2015), which gradually anneals from teacher forcing to free-running. For most char-LM tasks teacher forcing throughout is fine; for sequence-to-sequence translation it can matter.

**Q9: C.** Pass forward with `.detach()`. This is the truncated-BPTT recipe and is the only one of the four options that does the right thing. Option A (reset every chunk) discards cross-chunk context entirely. Option B (pass forward with gradient) grows the BPTT graph without bound. Option D is invented.

**Q10: D.** The transformer is *permutation-equivariant by default*, which is the *opposite* of "handles inputs in any order naturally." Attention has no built-in notion of position; you have to add positional encodings to recover sequence order. This is not an advantage; it is a quirk that the transformer pays for. The three actual advantages (constant-time lookback, parallel training, `O(1)` gradient path) are stated correctly in A, B, and C.
