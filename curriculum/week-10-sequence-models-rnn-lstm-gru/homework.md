# Week 10 — Homework

Six problems, about six hours total. Commit each in your Week 10 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — Parameter counts on paper, then verified (30 minutes)

For each of the following layer configurations, compute the parameter count on paper using the formulas from Lecture 1 Section 2 (vanilla RNN) and Lecture 2 Section 2 (LSTM, GRU), then verify in PyTorch by counting `sum(p.numel() for p in layer.parameters())`.

Configurations:

1. `nn.RNN(input_size=10, hidden_size=64, num_layers=1)`. Formula: `64 * 10 + 64 * 64 + 2 * 64` (two biases). Verify.
2. `nn.LSTM(input_size=10, hidden_size=64, num_layers=1)`. Formula: `4 * (64 * 10 + 64 * 64 + 2 * 64)`. Verify.
3. `nn.GRU(input_size=10, hidden_size=64, num_layers=1)`. Formula: `3 * (64 * 10 + 64 * 64 + 2 * 64)`. Verify.
4. `nn.LSTM(input_size=10, hidden_size=64, num_layers=2)`. The second layer takes the first layer's `hidden_size = 64` as its input. Compute both layers separately and sum. Verify.
5. `nn.LSTM(input_size=10, hidden_size=64, num_layers=2, bidirectional=True)`. The forward and backward halves double the count. Verify.

For each, write the predicted count and the formula computation (one line each). Save the work as `homework/01-param-counts.md` (table) and `homework/01-verify.py` (the verification script).

Acceptance: every predicted count matches the PyTorch count exactly. `python -m py_compile homework/01-verify.py` succeeds. Every function has a type hint.

References:
- `nn.RNN`: <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>
- `nn.LSTM`: <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>
- `nn.GRU`: <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>

---

## Problem 2 — Reproduce the EXPERIMENT from Lecture 1 Section 5 (1 hour)

Build a vanilla RNN with `hidden_size=64`. Initialize `h_0` to a random unit-norm vector. Set the input at every timestep to zero. Unroll for 200 timesteps and plot `||h_t||` against `t`.

Repeat with three settings of the recurrent weight matrix `W_hh`:
1. The default PyTorch initialization (spectral norm approximately 1).
2. `W_hh` rescaled to spectral norm 0.5 (contractive case).
3. `W_hh` rescaled to spectral norm 1.5 (expansive case).

Produce `homework/02-norm-trajectory.png`: a single figure with three curves (legend) on a log-scale y-axis. Write `homework/02-norm-trajectory.md` (~150 words) describing what you see in each regime. Reference Lecture 1 Section 5; report the slope of `log(||h_t||)` vs. `t` for the contractive case.

Acceptance: `homework/02-norm-trajectory.png` and `homework/02-norm-trajectory.md` are committed. `python -m py_compile homework/02-trajectory.py` succeeds. The contractive case shows roughly geometric decay; the expansive case saturates near `sqrt(64) ≈ 8`.

References:
- `torch.linalg.svdvals`: <https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html>
- Pascanu et al. 2013, Figure 2: <https://arxiv.org/abs/1211.5063>

---

## Problem 3 — Read and write up the LSTM paper Section 4 (1 hour)

Read **Section 4 ("Long Short-Term Memory")** of Hochreiter and Schmidhuber 1997 (<https://www.bioinf.jku.at/publications/older/2604.pdf>). Pages 10-17 of the PDF (numbered as pages 1744-1751 in the journal).

Write `homework/03-lstm-paper.md` (~400 words) covering:

1. **The constant-error-carousel motivation.** What is the problem the authors identify in Section 3, and how does the LSTM's architecture (Section 4) solve it? State the gradient-flow argument in your own words. Reference the equations they use.
2. **The 1997 vs. 2000 gate counts.** The original 1997 paper has only two gates (input and output); the forget gate was added by Gers, Schmidhuber, Cummins in 2000. PyTorch's `nn.LSTM` implements the 2000 version. Why was the forget gate added? Reference the 2000 paper (<https://www.bioinf.jku.at/publications/2000/Gers_Schmidhuber_Cummins_NC2000.pdf>) for the answer.
3. **The "memory blocks" terminology.** Section 4.2 introduces *memory blocks* (multiple cells sharing the same gates). The modern interpretation is that each "block" of hidden units shares one gate vector. Is this still how PyTorch's `nn.LSTM` is structured? (Answer: yes — the gate vector is `hidden_size`-dimensional, one entry per hidden unit.)
4. **The experimental sequence lengths.** Section 5 reports experiments on sequences of lengths 100, 1000, and longer. What was the result on length-1000 sequences vs. a vanilla RNN baseline? Reference the table.

Write in your own words; do not paraphrase. Include section and equation numbers you used. The C5 conviction: you should be able to summarize Section 4 of this paper to a colleague in five minutes.

References:
- LSTM paper: <https://www.bioinf.jku.at/publications/older/2604.pdf>
- Forget-gate paper: <https://www.bioinf.jku.at/publications/2000/Gers_Schmidhuber_Cummins_NC2000.pdf>

---

## Problem 4 — Implement a `nn.RNN`-equivalent layer with `nn.RNNCell` (1.5 hours)

Write `homework/04-rnn-layer.py` containing a class `ManualRNNLayer` that wraps an `nn.RNNCell` and exposes the same interface as `nn.RNN`. Specifically:

```python
class ManualRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True) -> None: ...
    def forward(
        self,
        inputs: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

The forward should:

1. Accept input of shape `(batch, seq_len, input_size)` if `batch_first=True`, else `(seq_len, batch, input_size)`.
2. If `h_0 is None`, initialize it to zeros.
3. Unroll the cell across the time dimension.
4. Return `(outputs, h_T)` where `outputs` is `(batch, seq_len, hidden_size)` (or `(seq_len, batch, hidden_size)`) and `h_T` is `(1, batch, hidden_size)`.

Verify against `nn.RNN(input_size, hidden_size, num_layers=1, batch_first=batch_first)` by copying the cell's weights into the layer and checking that the per-step outputs agree to within 1e-5 absolute error. Write a `pytest` function to verify.

Acceptance: `python -m py_compile homework/04-rnn-layer.py` succeeds. The verification test passes. Every function has type hints. The class docstring explains the relationship between `nn.RNNCell` and `nn.RNN`.

References:
- `nn.RNNCell`: <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>
- `nn.RNN`: <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>

---

## Problem 5 — Char-LM hidden-size sweep (1.5 hours)

Train the mini-project's `CharLSTM` on `Pride and Prejudice` for **5 epochs each** at three hidden sizes: `64`, `128`, `256`. All other hyperparameters held fixed (2 layers, `dropout=0.2`, `Adam(lr=1e-3)`, batch size 32, chunk length 100, gradient clipping at 1.0).

Record:
- Validation loss after each epoch (nats per character).
- Wall-clock seconds per epoch (median across the 5 epochs).
- Total parameter count.

Produce `homework/05-hidden-size.png` with three validation-loss curves (one per hidden size) on the same axes. Produce `homework/05-hidden-size.md` (~200 words) covering:

1. Final validation losses, ordered.
2. Wall-clock per epoch, ordered.
3. Parameter counts, ordered.
4. The hyperparameter-scaling argument: validation loss should drop as hidden size increases, with diminishing returns. Verify or refute in your data.
5. The C5 conviction: at the small-corpus scale, hidden size 256 is the sweet spot — bigger models start to overfit the 700-KB corpus within 5 epochs, and the validation loss can actually *worsen* at hidden size 512+. If you have a CUDA GPU, run a 512-size model and report whether you see the overfit.

Acceptance: `homework/05-hidden-size.png` and `homework/05-hidden-size.md` are committed. `python -m py_compile homework/05-sweep.py` succeeds. The script uses `torch.manual_seed(42)` at the top.

References:
- The mini-project starter: `mini-project/starter.py`.

---

## Problem 6 — Sample at five temperatures (45 minutes)

Take the trained model from the mini-project (or, if you have not started it yet, train a 5-epoch checkpoint with the starter's defaults). Sample 200 characters at temperatures `0.3, 0.5, 0.8, 1.0, 1.5`. Use the prompt `"It is a truth universally acknowledged, "` (the opening of Pride and Prejudice).

Produce `homework/06-temperatures.md` containing:

1. **The five sampled passages**, each in a fenced code block, each labeled with its temperature.
2. **A 200-word commentary** on the qualitative differences. Specifically:
   - At `T = 0.3`, the sampling is deterministic enough that the model often falls into loops (repeating the same word or phrase). Observe and report.
   - At `T = 0.5-0.8`, the samples are in the "interesting" zone — English-shaped, mostly grammatical, occasionally creative.
   - At `T = 1.0-1.5`, the samples become more random. At `T = 1.5` you should see invented pseudo-words and broken sentence structure.
3. **The C5 reading**: which temperature looks best in your samples? There is no "right" answer; the C5 default for portfolio sampling is `T = 0.8`, but explain why your favorite (which may differ) is the better choice for showing off the model.

Acceptance: `homework/06-temperatures.md` is committed. The samples are visibly different across temperatures. `python -m py_compile homework/06-sample.py` succeeds.

References:
- `torch.multinomial`: <https://pytorch.org/docs/stable/generated/torch.multinomial.html>
- Karpathy 2015: <https://karpathy.github.io/2015/05/21/rnn-effectiveness/> — the section on sampling explores the temperature trade-off at length.

---

## Submission checklist

- [ ] All six problems' artifacts are in `homework/` of your Week 10 repo.
- [ ] Every `.py` file passes `python -m py_compile`.
- [ ] Every figure is reproducible by re-running the corresponding script.
- [ ] All Markdown write-ups stay within the word counts noted in each problem.
- [ ] You set `torch.manual_seed(42)` at the top of every training script.
- [ ] No emojis in any file; type hints on every Python function; the writing voice is plain and declarative.
