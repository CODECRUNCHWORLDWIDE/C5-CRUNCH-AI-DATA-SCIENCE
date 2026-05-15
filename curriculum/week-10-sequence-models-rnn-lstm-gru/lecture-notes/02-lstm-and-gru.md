# Lecture 2 — LSTM and GRU: Gated Recurrence That Actually Works

> **Outcome:** You can state the four LSTM equations (forget gate, input gate, candidate, output gate) and the cell-state update on paper from memory. You can explain — quantitatively, not in slogans — why the additive cell-state update lets gradient flow back through hundreds of timesteps where the vanilla RNN's product-of-Jacobians does not. You can state the three GRU equations (reset gate, update gate, candidate) and identify the GRU as a parameter-economical variant of the LSTM with one fewer state tensor. You can pick between LSTM and GRU for a given task and defend the choice with the Chung et al. 2014 empirical result. You can build a 2-layer LSTM in PyTorch in three lines and inspect its parameter shapes. By the end of this lecture, the gates are not magic; they are four affine layers and three element-wise multiplies, and you have written them out long-hand.

Lecture 1 ended with the vanilla RNN's failure mode: the product of Jacobians along the BPTT path either vanishes or explodes, geometrically with the gap between source and sink timesteps. This lecture introduces the architectural fix that the field converged on after fifteen years of trying optimization tricks: a recurrent cell whose state-to-state Jacobian has an *additive* component that escapes the product-of-saturating-nonlinearities trap. The cell is the **Long Short-Term Memory (LSTM)**, introduced by Hochreiter and Schmidhuber in 1997 (<https://www.bioinf.jku.at/publications/older/2604.pdf>). The modern three-gate version, with the forget gate added by Gers, Schmidhuber, and Cummins in 2000, is what `torch.nn.LSTM` implements.

A simpler variant — the **Gated Recurrent Unit (GRU)**, introduced by Cho et al. in 2014 (<https://arxiv.org/abs/1406.1078>) — collapses the cell state into the hidden state and uses only two gates. It has 25-30% fewer parameters than the LSTM for the same hidden size and matches the LSTM's quality on most tasks. The Chung et al. 2014 paper (<https://arxiv.org/abs/1412.3555>) is the empirical reference.

The pace of this lecture is slower than usual; the algebra is the densest in C5 so far. Plan for fifty minutes of reading plus thirty minutes of working through the equations with pencil and paper.

We target **PyTorch 2.x**; the primary references are `torch.nn.LSTM` (<https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>) and `torch.nn.GRU` (<https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>). Christopher Olah's 2015 blog post "Understanding LSTM Networks" (<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>) is the visual companion; the diagrams of the gates in section 4 are the same diagrams reused in every deep-learning textbook published since 2016.

---

## 1. The architectural problem the LSTM solves

Lecture 1 Section 5 showed that the product of state-to-state Jacobians in the vanilla RNN has the form:

```text
dh_i / dh_{i-1} = diag(1 - tanh^2(...)) @ W_hh
```

Each factor has spectral norm bounded by `||W_hh||_2`; the product of `t - k` factors has spectral norm bounded by `||W_hh||_2^(t-k)`, which decays or explodes geometrically.

The LSTM rewires the recurrence so that there is a path from `c_{t-1}` to `c_t` that is *additive* rather than multiplicative. Specifically, the cell-state update is:

```text
c_t = f_t * c_{t-1} + i_t * g_t
```

where `f_t`, `i_t` are gates (vectors in `[0, 1]^k`) and `g_t` is a candidate update. The crucial observation: the Jacobian `dc_t / dc_{t-1}` is `diag(f_t)`. Not the matrix product `diag(1 - tanh^2) @ W_hh` of the vanilla RNN — just `diag(f_t)`. If the forget gate `f_t` is near 1, the Jacobian is near the identity, and the gradient flows through it almost losslessly. *That* is the trick. Hochreiter and Schmidhuber called it the **constant error carousel** in the 1997 paper. The modern formulation calls it the additive update; both names refer to the same architectural choice.

The three gates (`f`, `i`, `o`) and the candidate `g` are each their own affine layer over `[x_t, h_{t-1}]`. The LSTM cell has four affine layers, three element-wise sigmoid gates, one element-wise tanh, and two element-wise multiplications. We work through it now, slowly.

---

## 2. The LSTM equations

The PyTorch convention follows the 2000 amendment (Gers, Schmidhuber, Cummins) with three gates. Let `x_t` be the input at time `t`, `h_{t-1}` the previous hidden state, `c_{t-1}` the previous cell state. The LSTM computes:

```text
f_t = sigmoid(W_f @ [x_t, h_{t-1}] + b_f)     # forget gate
i_t = sigmoid(W_i @ [x_t, h_{t-1}] + b_i)     # input gate
g_t = tanh   (W_g @ [x_t, h_{t-1}] + b_g)     # candidate
o_t = sigmoid(W_o @ [x_t, h_{t-1}] + b_o)     # output gate
c_t = f_t * c_{t-1} + i_t * g_t               # cell state update (the trick)
h_t = o_t * tanh(c_t)                         # hidden state output
```

Six equations. Four affine transforms (`f`, `i`, `g`, `o`), each parameterized by a `(hidden, input + hidden)` weight matrix and a `(hidden,)` bias. Three element-wise multiplies (`f*c`, `i*g`, `o*tanh(c)`). One sum (`f*c + i*g`). The cell has two state tensors that travel forward in time: the hidden state `h` (the "output channel," visible to downstream layers) and the cell state `c` (the "memory channel," internal to the LSTM, not directly observable).

The parameter count for a single-layer LSTM with `input_size = d`, `hidden_size = k` is:

```text
4 * (k * (d + k) + k) = 4 * k * (d + k + 1)
```

For `d = 50, k = 256` (typical char-LM sizing), that is `4 * 256 * 307 = 314368` ≈ 314k parameters per layer. A 2-layer LSTM of this size is about 628k parameters in the recurrent body, plus the embedding and projection layers on either end.

> **EXPERIMENT — print the parameter shapes of nn.LSTM.** In a Python REPL: `import torch; from torch import nn; lstm = nn.LSTM(input_size=50, hidden_size=256, num_layers=2, batch_first=True); print([(n, p.shape) for n, p in lstm.named_parameters()])`. You will see eight tensors per layer: `weight_ih_l0` of shape `(1024, 50)`, `weight_hh_l0` of shape `(1024, 256)`, `bias_ih_l0` and `bias_hh_l0` each of shape `(1024,)`, and the corresponding `_l1` tensors for the second layer. The `1024 = 4 * 256` is PyTorch packing the four gate matrices into a single concatenated tensor for cuDNN efficiency. The packing order is `i`, `f`, `g`, `o` — **note that PyTorch's convention is `[i, f, g, o]`, not `[f, i, g, o]` as in the equations above**. The cuDNN convention is the same. This is a tedious source of bugs and we will revisit it in Exercise 2.

---

## 3. What the gates are for, intuitively

Each gate is a vector in `[0, 1]^k` (one value per hidden unit) controlled by the previous hidden state and the current input.

- **Forget gate `f_t`**. At each cell-state position, the forget gate decides how much of the previous cell state to retain. `f_t = 1` means "keep everything"; `f_t = 0` means "wipe this position." A well-trained LSTM on a language-modeling task often shows forget-gate activations near 1 across long stretches of text and near 0 at sentence or paragraph boundaries — the cell is using the forget gate to compartmentalize memory.
- **Input gate `i_t`**. The input gate decides how much of the candidate `g_t` to write into the cell state. `i_t = 1` is "write everything"; `i_t = 0` is "write nothing, keep the cell state unchanged."
- **Output gate `o_t`**. The output gate decides how much of the cell state (passed through a `tanh` for boundedness) to expose as the hidden state. `o_t = 1` makes `h_t = tanh(c_t)`; `o_t = 0` makes `h_t = 0`. The cell state continues to evolve internally regardless; the output gate controls visibility.

The product `i_t * g_t` is the actual update — `i_t` decides *how much* to write, `g_t` decides *what* to write. Distinguishing magnitude (gate) from direction (candidate) is the architectural pattern that recurs throughout deep learning; attention's softmax-times-values is the same pattern.

---

## 4. The gradient flow story, carefully

This is the section where the magic happens. The LSTM Jacobian `dc_t / dc_{t-1}` is:

```text
dc_t / dc_{t-1} = diag(f_t)
```

That is *not* a matrix product. It is a diagonal matrix whose entries are the forget-gate activations, each in `[0, 1]`. The gradient of `c_T` with respect to `c_0` is:

```text
dc_T / dc_0 = product_{t=1}^{T} diag(f_t)
            = diag(product_{t=1}^{T} f_t)
```

In other words, the gradient *per cell-state position* is the product of the forget-gate activations at that position across all timesteps. If the forget gates at a given position are all near 1, the gradient flows back almost losslessly. If they are near 0, the gradient at that position is killed — but this is *information-bearing* killing (the network has decided to forget that position) rather than the *uninformative* killing of the vanilla RNN.

Three honest qualifications:

1. **The Jacobian above is only the cell-state-to-cell-state Jacobian.** The full Jacobian `dh_t / dh_{t-1}` includes a path through the hidden state and another through the cell state; the cell-state path is the one that does not vanish. The hidden-state path *does* still suffer from the vanilla-RNN failure mode, but the cell-state path bypasses it.
2. **A learned forget bias near 1 helps.** The default PyTorch initialization gives the forget gate bias `b_f` a value of 0, which means `f_t` starts near `sigmoid(0) = 0.5`. Many practitioners initialize `b_f = 1.0` (sometimes called the "Jozefowicz fix" after Jozefowicz, Zaremba, Sutskever 2015 — <https://proceedings.mlr.press/v37/jozefowicz15.pdf>) to bias the network toward retention at the start of training. PyTorch's `nn.LSTM` does *not* do this by default; you can do it manually after construction.
3. **The cell state can still explode.** Nothing constrains the magnitude of `c_t`; in principle the candidate `g_t` (which is `tanh`-bounded to `[-1, 1]`) summed across timesteps can drive the cell state to large values, and the subsequent `tanh(c_t)` saturates. In practice this is rare with sensible initialization but can happen in pathological cases.

---

## 5. The 1997 paper, briefly

The original Hochreiter and Schmidhuber 1997 paper is dense but worth skimming. The paper's structure:

- **Sections 1-3**: motivation, prior work, the constant-error-carousel idea. The mathematical analysis of why standard RNNs cannot learn long-term dependencies is the conceptual setup; if you read only one part of the paper, read these eight pages.
- **Section 4**: the LSTM cell definition. The 1997 version had only two gates — input gate (called "input gate") and output gate (called "output gate"). The cell state in the 1997 version did *not* have a forget gate; the original LSTM relied on the input gate being zero to prevent the cell state from growing. This worked for the experiments in the paper but failed on tasks where the model needed to *reset* its memory, which is why Gers, Schmidhuber, and Cummins added the forget gate in 2000 (<https://www.bioinf.jku.at/publications/2000/Gers_Schmidhuber_Cummins_NC2000.pdf>).
- **Section 5**: experiments. Synthetic tasks (the "noise vs. signal" sequences) designed to demonstrate long-range memory. The experiments are small by 2026 standards (sequence lengths up to a few hundred, hidden sizes in the tens) but the result — LSTM beats vanilla RNN — is unambiguous.
- **Section 6 and appendix**: the truncated-gradient derivation. Optional; the modern PyTorch implementation does not need it because autograd handles the gradient computation automatically.

The paper is at <https://www.bioinf.jku.at/publications/older/2604.pdf>. It is in the C5 stretch goals; read sections 1-4 and skim the rest.

---

## 6. The GRU equations

In 2014, Cho et al. introduced a simpler gated recurrent unit while building an encoder-decoder for machine translation. The cell merges the cell state into the hidden state and uses two gates instead of three:

```text
r_t = sigmoid(W_r @ [x_t, h_{t-1}] + b_r)              # reset gate
z_t = sigmoid(W_z @ [x_t, h_{t-1}] + b_z)              # update gate
g_t = tanh(W_g @ [x_t, r_t * h_{t-1}] + b_g)           # candidate
h_t = (1 - z_t) * h_{t-1} + z_t * g_t                  # convex combination
```

Three affine transforms (vs. four in the LSTM), two gates (vs. three), one state tensor (vs. two). The parameter count is:

```text
3 * (k * (d + k) + k) = 3 * k * (d + k + 1)
```

For `d = 50, k = 256`, that is `3 * 256 * 307 = 235776` ≈ 236k parameters per layer, vs. 314k for an LSTM of the same size. The savings: about 25%.

The reset gate `r_t` controls how much of `h_{t-1}` is mixed into the candidate computation. `r_t = 1` is "use all of the previous state"; `r_t = 0` makes the candidate a pure function of `x_t` (resetting the recurrent context). The update gate `z_t` controls the convex combination between the previous state and the candidate: `z_t = 0` is "keep the previous state unchanged"; `z_t = 1` is "replace it entirely."

The GRU's gradient flow story is similar to the LSTM's. The state-to-state Jacobian `dh_t / dh_{t-1}` has an additive component `diag(1 - z_t)` that flows back losslessly when `z_t` is near 0, and a multiplicative component that involves the candidate and the reset gate. The analysis is in section 4 of the Chung et al. 2014 paper (<https://arxiv.org/abs/1412.3555>).

> **EXPERIMENT — count parameters of LSTM vs. GRU.** Build `nn.LSTM(input_size=50, hidden_size=256, num_layers=2)` and `nn.GRU(input_size=50, hidden_size=256, num_layers=2)`. Print `sum(p.numel() for p in lstm.parameters())` and the same for `gru`. The LSTM should be ~728k parameters; the GRU should be ~547k. The ratio is ~0.75, consistent with the 25% savings. If your project has tight memory or speed constraints, this is the architecture-level lever to pull.

---

## 7. The LSTM-vs-GRU question

When does LSTM beat GRU and vice versa? The honest answer from the literature:

- **They are statistically indistinguishable on most sequence tasks.** Chung et al. 2014 compared them on polyphonic music modeling (four MIDI datasets) and speech-signal modeling (two speech datasets). On every dataset, LSTM and GRU produced negative log-likelihoods within 1-3% of each other; the differences were not statistically significant. Both decisively beat the vanilla RNN.
- **GRU trains slightly faster.** Fewer parameters, fewer matrix multiplies per step, less memory. On the C5 char-LM mini-project, swapping the LSTM for a GRU of matched hidden size shaves about 15% off the wall-clock training time on CPU.
- **LSTM has more inductive flexibility.** The separate cell state lets the LSTM maintain a memory channel that is decoupled from the output channel; in principle this is useful for tasks where the network needs to *hide* information from the readout. In practice this advantage is small on most tasks and the GRU's simpler architecture is often preferable.
- **For large-scale language modeling (millions of examples, billions of parameters), the question is moot.** Transformers superseded both LSTMs and GRUs in 2017 and the field never went back. The C5 conviction in 2026: pick GRU for small-data sequence tasks where the parameter count matters; pick LSTM if you are reproducing a paper that uses LSTM; pick transformer for anything bigger than a hobby project. Lecture 3 sets up the third option.

The C5 char-LM mini-project uses LSTM by default (because the Karpathy 2015 reference implementation used LSTM, and reproducing the reference is pedagogically clean). Challenge 2 of this week has you train both and report a head-to-head comparison.

---

## 8. Stacking and dropout

Both `nn.LSTM` and `nn.GRU` support multi-layer stacking via the `num_layers` argument. The second layer takes the first layer's output (the sequence of hidden states) as its input; the third layer takes the second layer's output; and so on. PyTorch hand-rolls this internally and you can read the entire output of any layer if you want intermediate features.

A second `dropout` argument applies dropout *between* layers (not within a layer's recurrence). The argument is the dropout probability and only takes effect if `num_layers > 1`; on a single-layer LSTM the `dropout` argument is silently ignored with a warning. This is documented at <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html> in the "Inputs" section.

For the C5 mini-project, a 2-layer LSTM with hidden size 256 and `dropout=0.2` is the default recipe. Larger models (4 layers, hidden size 512) train fine on a GPU but overfit `Pride and Prejudice` within a few epochs because the corpus is only ~700 KB.

Reference: the dropout paper (Srivastava et al. 2014, <https://jmlr.org/papers/v15/srivastava14a.html>) describes feedforward dropout; the analog for recurrent connections within an LSTM is Gal and Ghahramani 2016 (<https://arxiv.org/abs/1512.05287>), called "variational dropout." PyTorch's `nn.LSTM` does *not* implement variational dropout; the `dropout` argument only applies between layers. If you want recurrent dropout, you have to write the unroll yourself with `nn.LSTMCell`.

---

## 9. A 2-layer LSTM in PyTorch, end to end

The full skeleton of a character-level language model — which the mini-project expands into a full training pipeline — fits in twenty lines:

```python
from __future__ import annotations
import torch
from torch import nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        state: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]":
        emb = self.embed(x)
        out, state = self.lstm(emb, state)
        logits = self.head(out)
        return logits, state
```

The `forward` returns the logits over the vocabulary at every timestep (shape `(batch, seq_len, vocab_size)`) and the final LSTM state, so the caller can pass the state forward across truncated-BPTT chunks. The loss is `nn.CrossEntropyLoss()` applied to the flattened logits and the target token sequence (shifted by one position from the input — the standard language-modeling target).

That is the entire model. Lecture 3 covers the data pipeline (`pack_padded_sequence` for variable-length batches, the truncated-BPTT chunking for long sequences), teacher forcing, and the temperature-sampling loop for generation.

---

## 9b. Inspecting gate activations on a real sentence

A useful exercise to internalize what the gates are actually doing: train a small LSTM-LM (e.g. the Week 10 mini-project's CharLSTM, even at epoch 1) and then on a forward pass through a held-out sentence, record the values of `f_t`, `i_t`, and `o_t` at every timestep and at every hidden unit. Plot them as a heatmap of shape `(hidden_size, seq_len)` per gate.

What you should see:

- **The forget gate is mostly near 1 inside words and mostly near 0 at sentence boundaries.** This is the cell using the forget gate to compartmentalize memory — keep the recent context within a sentence, wipe at the period or paragraph break.
- **The input gate is mostly near 0 between characters of the same word and spikes near 1 at word boundaries (after a space).** The cell is choosing to write to memory at the points where new "tokens" begin.
- **The output gate is most strongly influenced by the character predicted next, not by what is in the cell state.** At a position where the next character is highly predictable (e.g., the `e` in `the`), the output gate is sharp; at less predictable positions it is more diffuse.

These observations are reproducible on a 1-epoch-trained CharLSTM on Pride and Prejudice; the C5 mini-project's `exploration.ipynb` has a cell that produces this heatmap. The visual proof that "the gates do meaningful things" is one of the more satisfying pedagogical artifacts of Week 10.

The original investigations of "what are the gates learning" come from the 2014-2015 era; Karpathy, Johnson, Fei-Fei 2015, "Visualizing and Understanding Recurrent Networks" (<https://arxiv.org/abs/1506.02078>) is the canonical paper, and it identified hidden units in a char-LSTM that fire on specific features of the input (quote marks, indentation levels, comment delimiters in source code). The paper is short and free; it is in the resources as optional reading.

---

## 9c. Why LSTMs don't need the "Jozefowicz fix" forever

The Jozefowicz, Zaremba, Sutskever 2015 paper (<https://proceedings.mlr.press/v37/jozefowicz15.pdf>) recommended initializing the forget-gate bias to 1.0 so that `f_t` starts near `sigmoid(1) ≈ 0.73` rather than `sigmoid(0) = 0.5`. The argument: at the start of training, the cell has no learned reason to forget or remember, and the default `b_f = 0` initialization gives it a 50/50 retain-or-forget prior — which is the maximally entropic prior and which destroys cell-state information geometrically (`0.5^T → 0` for `T = 200`).

After training, the forget gate learns task-appropriate retention behavior and the initial bias matters less. Specifically, the network learns to make `f_t` near 1 in contexts where retention helps (most of the time) and near 0 at clear "reset" boundaries (sentence ends, paragraph breaks). The Jozefowicz fix is a *warm start* into this regime; it does not change the asymptotic equilibrium the network converges to.

Empirically, the Jozefowicz fix helps most when:

1. The corpus has long-range dependencies (>100 timesteps) that the LSTM must learn to track.
2. The training budget is short (under ~10 epochs) and there is not enough training time for the network to discover the right forget-bias on its own.
3. The base LSTM is small (under ~50k parameters) and lacks the capacity to compensate for a bad initialization.

For the C5 char-LM mini-project — a 1.5M-parameter LSTM trained for 20 epochs — the Jozefowicz fix typically buys you 0.02-0.05 nats/char of final validation loss. Not nothing, but not transformative. PyTorch's `nn.LSTM` does not apply it by default; if you want it, you have to do it manually after construction. Exercise 2 Part C asks you to implement it from scratch and verify the empirical effect on `||c_t||` decay.

---

## 10. Recap and what is next

You can now:

- State the four LSTM equations and the cell-state update from memory.
- State the three GRU equations and the convex-combination update.
- Explain the additive cell-state update as the architectural fix to the vanishing-gradient problem.
- Identify the parameter shapes of `nn.LSTM` and `nn.GRU` and compute their parameter counts.
- Defend the choice between LSTM and GRU with the Chung et al. 2014 empirical result.
- Build a 2-layer LSTM in PyTorch in ten lines and trace the data flow from `nn.Embedding` to `nn.Linear`.

In Lecture 3 we move from the cell to the system. The data pipeline: how to batch variable-length sequences with `pack_padded_sequence`. The training loop: teacher forcing and truncated BPTT. The end-to-end char-LM: read a Project Gutenberg text, train the LSTM from Section 9, sample at three temperatures, observe English-shaped text. And we end with a deliberate, pointed tease — three things the LSTM cannot do that the architecture introduced in Week 11 can. Bring your skepticism for the tease; it will be paid off next week.
