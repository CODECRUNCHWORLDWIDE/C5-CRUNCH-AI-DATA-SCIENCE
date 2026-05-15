# Lecture 1 — The Vanilla RNN and the Vanishing-Gradient Problem

> **Outcome:** You can write the vanilla RNN recurrence on paper from memory. You can unroll it across `T` timesteps and identify the computational graph that backpropagation through time (BPTT) traverses. You can state the vanishing- and exploding-gradient analyses in terms of the singular values of the recurrent weight matrix, and you can argue why those failure modes are *not* artifacts of `tanh` specifically — they appear for any saturating nonlinearity. You can implement gradient clipping in three lines of PyTorch and explain why it cures the exploding case but does nothing for the vanishing case. By the end of this lecture, "RNNs are hard to train on long sequences" stops being a slogan and becomes a quantitative claim you can derive on paper.

Week 9 left you with a ResNet-18 reaching 92% on CIFAR-10. The model took an image — a fixed-size grid of pixels — and produced a single class label. The architectural assumptions baked into the convolutional layer (locality, weight sharing across spatial positions, translation equivariance) matched the structure of the data perfectly.

This week the data has a different structure. The input is a sequence: a stream of tokens of variable length, where the meaning of position `t` depends on positions `1..t-1`. The architectural property we need is no longer spatial locality but *temporal carryover* — a way for information from early in the sequence to influence the prediction late in the sequence. The architectural unit that does this is the **recurrent neural network**.

This lecture builds the simplest such unit, the vanilla RNN, mechanically from one equation. Then it does the math that explains why this simplest version does not work past sequence lengths of about twenty timesteps, and why an architectural refinement — the LSTM, which Lecture 2 introduces — was needed before recurrent networks could be put in production.

We target **PyTorch 2.x**; the primary references are `torch.nn.RNN` (<https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>) and `torch.nn.RNNCell` (<https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>). The conceptual companion is Goodfellow, Bengio, Courville chapter 10 (<https://www.deeplearningbook.org/contents/rnn.html>). The historical reference for the vanishing-gradient analysis is Pascanu, Mikolov, Bengio 2013 (<https://arxiv.org/abs/1211.5063>).

---

## 1. The sequence-modeling problem

A sequence is an ordered list `x_1, x_2, ..., x_T` of elements, each of which lives in some feature space `R^d`. The length `T` is *not* fixed across examples; a batch of sequences typically has different lengths and needs padding to be stored in a single rectangular tensor. The C5 working convention this week:

- `x` is the input sequence; element `x_t` is in `R^d` (a `d`-dimensional vector, or a single integer index that an `nn.Embedding` will map to `R^d`).
- `y` is the target; it can be a single label (sequence classification), a sequence of labels of the same length (sequence labeling), or a sequence of labels of a different length (sequence-to-sequence).
- `h` is the model's hidden state at each timestep; `h_t` is in `R^k` for some hidden-size hyperparameter `k`. The hidden state is what the model carries forward in time; it is the architectural answer to "how does information from `x_1` reach the prediction at step `T`?"

Three canonical instances of the problem:

1. **Sequence classification.** Input: a sequence of characters. Output: a single label (English / Spanish / French; "spam" / "not spam"; positive / negative sentiment). The model emits one output at the final timestep.
2. **Sequence labeling.** Input: a sequence of word tokens. Output: a sequence of POS tags or named-entity tags of the same length. The model emits one output per input timestep.
3. **Language modeling.** Input: a sequence of tokens. Output: at each timestep, a probability distribution over the next token. This is the canonical task this week and is what the mini-project trains.

What distinguishes these from the image-classification problems of Week 9 is that **the model has to produce its output as a function of variable-length context**. An MLP cannot do this directly — it needs a fixed-size input vector. A CNN can handle variable spatial sizes via `AdaptiveAvgPool2d`, but its receptive field grows only linearly with depth, so a deep CNN is needed to see the entire history of a long sequence. The RNN solves the problem by carrying a hidden state forward across timesteps, so the output at step `T` depends *in principle* on every earlier input.

That "in principle" is the entire subject of this lecture's second half. The vanishing-gradient problem is the gap between principle and practice.

---

## 2. The vanilla RNN cell, mechanically

The simplest recurrent unit, often called the **Elman RNN** after Jeffrey Elman's 1990 paper, is two affine transforms summed and squashed:

```text
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
```

That is the recurrence. The model has three parameter tensors:

- `W_xh` of shape `(hidden_size, input_size)` — mixes the new input into the hidden state.
- `W_hh` of shape `(hidden_size, hidden_size)` — mixes the previous hidden state into the new one.
- `b_h` of shape `(hidden_size,)` — the additive bias.

The hidden state at time `0` is conventionally initialized to zero (`h_0 = 0`) or learned as an additional parameter. PyTorch's `nn.RNN` defaults to zero initialization.

When the task requires an output at each timestep (sequence labeling, language modeling), a second affine layer projects the hidden state to the output space:

```text
y_t = W_hy @ h_t + b_y
```

This output layer is *separate* from the recurrent cell; it has its own weights and is not part of the `nn.RNNCell` itself. In PyTorch this projection is typically a `nn.Linear(hidden_size, output_size)` applied after the RNN's output. See `nn.RNN`'s docs (<https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>) for the exact convention.

> **EXPERIMENT — print the parameter shapes of nn.RNN.** In a Python REPL: `import torch; from torch import nn; rnn = nn.RNN(input_size=20, hidden_size=64, batch_first=True); print([(n, p.shape) for n, p in rnn.named_parameters()])`. You will see `weight_ih_l0` of shape `(64, 20)`, `weight_hh_l0` of shape `(64, 64)`, `bias_ih_l0` and `bias_hh_l0` each of shape `(64,)`. PyTorch separates the input-to-hidden and hidden-to-hidden biases for historical reasons; in the math the two biases are typically summed into a single `b_h`. The total parameter count for a single-layer RNN is `hidden * input + hidden * hidden + 2 * hidden = 64*20 + 64*64 + 2*64 = 5440`.

---

## 3. Unrolling the recurrence

Given a sequence `x_1, ..., x_T` and initial hidden state `h_0 = 0`, the RNN computes `T` hidden states by repeated application of the cell:

```text
h_1 = tanh(W_xh @ x_1 + W_hh @ h_0 + b_h)
h_2 = tanh(W_xh @ x_2 + W_hh @ h_1 + b_h)
h_3 = tanh(W_xh @ x_3 + W_hh @ h_2 + b_h)
 ...
h_T = tanh(W_xh @ x_T + W_hh @ h_{T-1} + b_h)
```

This is a feedforward computational graph with `T` repetitions of the same parameters. The graph is *deep* in time (depth equals sequence length) but *narrow* in width (a single hidden-state vector at each step). Standard backpropagation runs over this graph; the procedure is called **backpropagation through time (BPTT)** to emphasize that the graph dimension being traversed is time, but mechanically it is the same chain rule from Week 7.

The PyTorch `nn.RNN` layer does this unrolling for you. Given an input of shape `(seq_len, batch, input_size)` (or `(batch, seq_len, input_size)` with `batch_first=True`), it produces an output of shape `(seq_len, batch, hidden_size)` containing all `T` hidden states. The final hidden state is also returned separately. From the docs:

```python
output, h_n = rnn(x, h_0)
# output: (seq_len, batch, hidden_size) -- every h_t
# h_n:    (1, batch, hidden_size)       -- only h_T (the final)
```

The `output` tensor stacks `[h_1; h_2; ...; h_T]` along the time dimension; `h_n` is just `h_T` reshaped to `(num_layers, batch, hidden_size)`.

---

## 4. The BPTT gradient, formally

Suppose the loss at time `t` is `L_t = loss(y_t, target_t)`, and the total loss is `L = sum_t L_t`. We want `dL / d(parameters)` so we can update.

The gradient with respect to `W_hh` decomposes into a sum over timesteps:

```text
dL/dW_hh = sum_t dL_t/dW_hh
```

The chain rule for each `dL_t/dW_hh` requires backpropagating through all `t` previous applications of the cell, because `W_hh` is reused at every step. The full expression is messy but the key observation is the recursion:

```text
dL_t/dh_k = dL_t/dh_t * product_{i=k+1}^{t} (dh_i / dh_{i-1})
```

That product of Jacobians — the "transfer function" of the recurrence — is the single most important object in this lecture. Each factor `dh_i / dh_{i-1}` is a `(hidden, hidden)` matrix. For the vanilla RNN recurrence above:

```text
dh_i / dh_{i-1} = diag(1 - tanh^2(...)) @ W_hh
```

The `diag(1 - tanh^2(...))` factor comes from the derivative of `tanh`; each diagonal entry is in `[0, 1]` and equals 1 only when the pre-activation is exactly zero. The `W_hh` factor is the recurrent weight matrix. The product of `t - k` such Jacobians governs how strongly the gradient at step `t` reaches back to step `k`.

This is the recursion that backpropagation evaluates. It is also the recursion that, for any sequence longer than ~20 timesteps, *blows up or shrinks to zero* — the topic of section 5.

---

## 5. The vanishing- and exploding-gradient analyses

Consider the spectral norm (largest singular value) `||W_hh||_2` of the recurrent weight matrix. Three regimes:

- **`||W_hh||_2 < 1` (the contractive case).** The product of `t - k` Jacobians has spectral norm at most `(||W_hh||_2)^(t-k)`, which decays geometrically with the gap `t - k`. Concretely: if `||W_hh||_2 = 0.9` and `t - k = 100`, the product's spectral norm is at most `0.9^100 ≈ 2.65e-5`. The gradient at step `t` cannot reach step `k`; the early timesteps receive a vanishing learning signal. **This is the vanishing-gradient problem.**
- **`||W_hh||_2 > 1` (the expansive case).** The product grows geometrically. If `||W_hh||_2 = 1.1` and `t - k = 100`, the product's spectral norm is at least `1.1^100 ≈ 13780`. Gradients become enormous; weight updates overshoot; training diverges. **This is the exploding-gradient problem.**
- **`||W_hh||_2 = 1` (the balanced case).** The product's norm is bounded but does not decay. This is the regime LSTMs try to engineer into the cell state's gradient flow, and it is also the regime that some weight-orthogonal initializations target.

The full analysis is in Pascanu, Mikolov, Bengio 2013 (<https://arxiv.org/abs/1211.5063>). The original arguments are in Hochreiter's 1991 diploma thesis (German) and Bengio, Simard, Frasconi 1994 (IEEE Trans. on Neural Networks 5(2), 157-166). The conclusion has been re-derived in approximately every dialect of mathematics; the result is robust.

Two important honest qualifications:

1. **The analysis is a worst-case bound, not an inevitability.** The exact gradient depends on the alignment of singular vectors and the input data, not just the singular values. In practice, well-initialized vanilla RNNs on tasks with short dependencies (under 20 timesteps) train fine. The vanishing-gradient problem becomes acute around `t - k = 50` and catastrophic past `t - k = 100`.
2. **The `tanh` is not the cause; it is the messenger.** Replacing `tanh` with `ReLU` does not fix the vanishing-gradient problem; ReLU has its own pathology (exploding gradients are easier and the unbounded activations can drive the cell state to infinity). The fundamental issue is the *repeated matrix multiplication* in the recurrence; the nonlinearity choice modulates the constants but does not change the geometric-decay or geometric-growth structure.

> **EXPERIMENT — measure the hidden-state norm decay.** Build a vanilla RNN with `hidden_size=64`, random Gaussian initialization. Set `h_0` to a random unit-norm vector. Set every `x_t = 0` (so the recurrence becomes `h_t = tanh(W_hh @ h_{t-1} + b_h)`). Unroll for 200 timesteps and plot `||h_t||` against `t`. With the default PyTorch initialization (Xavier uniform, `||W_hh||_2 ≈ 1`), the norm typically settles to a fixed point near zero within 50 steps. Repeat with `W_hh` scaled to have spectral norm 0.5 (the contractive case) — the norm decays much faster. Repeat with spectral norm 1.5 (the expansive case) — the norm saturates at the `tanh` ceiling. This is the entire vanishing-gradient story on one plot.

---

## 6. Gradient clipping

The exploding-gradient case has a simple practical fix that does not address the underlying issue but does prevent the optimizer from blowing up: clip the global gradient norm before each update step. PyTorch ships this as `torch.nn.utils.clip_grad_norm_`:

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

The argument `max_norm=1.0` is a common default for RNN training; values up to 5 are also reasonable. The clip preserves direction (it scales the gradient down to the target norm without changing its angle) so it is a "safer" version of the SGD update than e.g. gradient sign clipping.

Gradient clipping does *not* fix the vanishing-gradient problem. A gradient that has vanished to magnitude 1e-10 is not improved by being multiplied by 1.0 / 1e-10 — the *direction* of a vanishing gradient is just noise from the optimizer's floating-point arithmetic, not a meaningful learning signal. The vanishing case requires an architectural fix (the LSTM and GRU of Lecture 2), not an optimization fix.

Reference: `clip_grad_norm_` docs at <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>.

---

## 7. Bidirectional RNNs

An aside before we move on. For sequence labeling tasks where the entire input is available at training time (POS tagging, named-entity recognition), it is useful to run two RNNs over the sequence — one forward, one backward — and concatenate their hidden states at each step. This gives every output position access to both its left and right context.

PyTorch supports this directly: `nn.RNN(input_size, hidden_size, bidirectional=True)`. The output has shape `(seq_len, batch, 2 * hidden_size)` because the forward and backward hidden states are concatenated along the feature dimension.

Bidirectional RNNs are *not* appropriate for online inference (real-time speech-to-text) or for autoregressive generation (language modeling) — both require causal models that only see the past. The C5 char-LM mini-project is autoregressive, so we use unidirectional LSTMs throughout. Bidirectional models become relevant in Week 12 (encoder-decoder for translation).

Reference: Schuster and Paliwal 1997, "Bidirectional Recurrent Neural Networks," IEEE Trans. Signal Processing 45(11) — paywalled, but the idea is straightforward and the PyTorch docs are sufficient.

---

## 8. The `nn.RNNCell` vs. `nn.RNN` distinction

PyTorch ships two related APIs and you will see both in production code:

- **`nn.RNNCell(input_size, hidden_size)`** is the single-step cell. You call it `h_t = cell(x_t, h_prev)` inside your own Python `for` loop over the time dimension. This is what you use when you want a custom unroll — for example, to inject control flow (teacher-forcing toggles, scheduled sampling) at each timestep.
- **`nn.RNN(input_size, hidden_size, num_layers=L)`** is the multilayer wrapper. You pass it the entire sequence at once and it does the unroll internally, in optimized C++ code (and cuDNN on GPU). This is what you use for the common case of "run the RNN forward over the whole sequence."

The mini-project uses `nn.LSTM` (the multilayer LSTM), but Exercise 1 has you implement a vanilla RNN cell from scratch and verify against `nn.RNNCell`. Both APIs are documented at <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html> and <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>.

> **EXPERIMENT — verify the cell matches the layer.** Build a `nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)` and a `nn.RNNCell(input_size=10, hidden_size=20)`. Copy the layer's weights into the cell: `cell.weight_ih.data = rnn.weight_ih_l0.data; cell.weight_hh.data = rnn.weight_hh_l0.data; cell.bias_ih.data = rnn.bias_ih_l0.data; cell.bias_hh.data = rnn.bias_hh_l0.data`. Now unroll the cell by hand on the same input and check that the per-step hidden states match the layer's output. They should agree to within 1e-6 absolute error. This is the verification Exercise 1 asks you to write.

---

## 9. What the vanilla RNN cannot do

Three failure modes that motivate the LSTM and GRU of Lecture 2:

1. **Long-range dependencies.** Section 5 quantified the vanishing-gradient issue. In practice, a vanilla RNN on a language-modeling task forgets context after about 20-30 characters. A 200-character sentence is well beyond its useful range.
2. **Selective forgetting.** Even when the gradient flows, a vanilla RNN cannot easily *choose* to forget information. The recurrence overwrites `h_t` with a function of `x_t` and `h_{t-1}` at every step; there is no architectural mechanism to say "keep this part of the state, discard that part." LSTMs add gates that perform exactly this selective overwrite.
3. **Stable training at deep timesteps.** Even with gradient clipping and careful initialization, vanilla RNNs are finicky to train past sequence length ~50. The 1990s spent a decade trying to fix this with optimization tricks (RTRL, BPTT(h, h'), teacher forcing) before the architectural answer arrived in 1997.

The 1997 paper is Hochreiter and Schmidhuber's "Long Short-Term Memory" (<https://www.bioinf.jku.at/publications/older/2604.pdf>). The paper sat near-uncited for fifteen years before deep learning picked it up around 2013 and turned it into one of the most-used neural network architectures ever shipped. Lecture 2 introduces the cell.

---

## 9b. A worked BPTT example, three timesteps

To make the chain-rule mechanics concrete, work through a three-step unroll on paper. Let `T = 3`, `hidden_size = 2`, `input_size = 1`. The recurrence is:

```text
h_1 = tanh(W_xh @ x_1 + W_hh @ h_0 + b)
h_2 = tanh(W_xh @ x_2 + W_hh @ h_1 + b)
h_3 = tanh(W_xh @ x_3 + W_hh @ h_2 + b)
```

Suppose the loss is `L = ||h_3||^2 / 2` — a scalar function of `h_3` only. The gradient with respect to `W_hh` is:

```text
dL/dW_hh = dL/dh_3 * (dh_3/dW_hh + dh_3/dh_2 * dh_2/dW_hh + dh_3/dh_2 * dh_2/dh_1 * dh_1/dW_hh)
```

Three terms, one for each timestep at which `W_hh` was used. Each term has a "direct" contribution `dh_t/dW_hh` and a "through-time" contribution `dh_t/dh_{t-1} * dh_{t-1}/dW_hh * ...`. The first term reaches back zero steps, the second reaches back one step, the third reaches back two steps.

The factor `dh_t/dh_{t-1}` is the Jacobian from Section 4: `diag(1 - tanh^2(...)) @ W_hh`. The product `dh_3/dh_2 * dh_2/dh_1` is a product of two such Jacobians. For `T = 200`, the analogous product has 199 factors, and the geometric-decay argument applies.

Two practical lessons from this worked example. **First**, the gradient with respect to `W_hh` is a *sum* over timesteps, and each summand has a different magnitude — the gradient is dominated by the recent steps if the long-range terms have vanished, which is exactly the failure mode. **Second**, autograd does this computation for you with one `loss.backward()` call. You do not have to write out the three-term sum by hand; PyTorch's reverse-mode autodiff figures out which parameters were used at which timesteps and accumulates the gradients correctly. This is one of the underrated convenience features of modern deep-learning frameworks; in 1997 you had to derive BPTT by hand.

---

## 9c. Sequence length as the new "depth"

In Week 9 we measured a CNN's expressiveness in terms of its receptive field — the size of the input neighborhood that influences a single output activation. Stacking more conv layers, or adding strided convs, increased the receptive field, which increased the network's ability to integrate information from larger image regions.

The RNN analog of receptive field is *temporal context*: how far back in the sequence can a hidden-state activation influence the output? In principle, the answer is "infinitely far" — the recurrence is non-Markovian, every previous input can affect every future output through the hidden state. In practice, the vanishing-gradient analysis from Section 5 puts a hard ceiling on this: for a vanilla RNN, the effective temporal context is roughly `1 / (1 - ||W_hh||_2)` steps when `||W_hh||_2 < 1`, which for typical initializations is 10-30 steps. Past that, the gradient signal is below floating-point precision and the model cannot learn the dependency.

This is why CNNs and RNNs are dual architectures in a precise sense. CNNs have an exact, deterministic receptive field that grows with depth-in-layers; RNNs have an in-principle-unbounded but in-practice-truncated receptive field that grows with depth-in-time. Both run into the same problem (information attenuation over depth) and both have the same architectural fixes (skip connections in CNNs; gated recurrence in RNNs). The transformer of Week 11 abandons both depth-in-layers (for context) and depth-in-time (for context) and uses a single attention layer to look at every input position simultaneously — `O(1)` "depth" for the context lookup, at the cost of `O(T^2)` compute and memory.

This receptive-field-vs-temporal-context analogy is one of those bridges between Week 9 and Week 10 that is easy to miss if you read the weeks in isolation. The C5 conviction is that the *same* architectural pressure — "I need to integrate information across a large input region" — is what drove both the CNN architectural arc (LeNet → VGG → ResNet) and the RNN architectural arc (vanilla RNN → LSTM → attention).

---

## 10. Recap and what is next

You can now:

- Write the vanilla RNN recurrence on paper from memory.
- Identify the three parameter tensors (`W_xh`, `W_hh`, `b`) and their shapes given the input and hidden sizes.
- Unroll the recurrence across `T` timesteps and identify the BPTT computational graph.
- State the vanishing- and exploding-gradient analyses in terms of the spectral norm of `W_hh`.
- Explain why gradient clipping cures the exploding case but does nothing for the vanishing case.
- Distinguish `nn.RNNCell` (single-step, manual unroll) from `nn.RNN` (whole-sequence, optimized C++ unroll).

In Lecture 2 we replace the single-affine-plus-tanh recurrence with the **LSTM cell**: four affine transforms, three gates (forget / input / output), a separate cell state with an additive update, and a gradient-flow story that lets information propagate across hundreds of timesteps. The 1997 paper, the 2000 forget-gate amendment, and the modern PyTorch `nn.LSTM` implementation. Bring your patience for the four-equation block.
