# Week 10 Exercise Solutions

Reference solutions for the three Week 10 exercises. Read **only after attempting** each exercise. If you get stuck for more than thirty minutes on a part, peek at the solution for that part only; do not read the entire solution before you have tried.

---

## Exercise 1 — Vanilla RNN cell from scratch

### Part A — `VanillaRNNCell.__init__` and `forward`

```python
def __init__(self) -> None:
    super().__init__()
    self.input_size: int = input_size
    self.hidden_size: int = hidden_size
    bound: float = 1.0 / math.sqrt(hidden_size)
    self.W_xh = nn.Parameter(torch.empty(hidden_size, input_size).uniform_(-bound, bound))
    self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size).uniform_(-bound, bound))
    self.b = nn.Parameter(torch.zeros(hidden_size))

def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
    # x: (batch, input_size); h_prev: (batch, hidden_size)
    # W_xh: (hidden, input); we want W_xh @ x_t for each batch element,
    # which is x @ W_xh.T in batched form.
    return torch.tanh(x @ self.W_xh.T + h_prev @ self.W_hh.T + self.b)
```

The bound `1.0 / sqrt(hidden_size)` is what PyTorch's `nn.RNNCell` uses by default; the choice is documented in the `nn.RNNCell` docs ("Variables" section, "uniform distribution"). Why this bound? It is the Kaiming/Xavier-style "preserve activation variance" heuristic for `tanh` activations: drawing weights from `U(-bound, bound)` gives the matrix a roughly unit spectral norm in expectation, which is the boundary between vanishing and exploding regimes. For details see the He et al. 2015 paper (<https://arxiv.org/abs/1502.01852>) or Goodfellow chapter 8.4.

The `@` operator is `torch.matmul`; `x @ W.T` is the batched matrix-vector product `W @ x_i` for each batch element `i`. The transpose is necessary because PyTorch stores `W` as `(output_dim, input_dim)` for convention, and we want `(batch, input_dim) @ (input_dim, output_dim) -> (batch, output_dim)`.

### Part B — `unroll_rnn_cell`

```python
def unroll_rnn_cell(cell, inputs, h_0):
    n_batch, seq_len, _ = inputs.shape
    hidden_size = h_0.shape[-1]
    hiddens = torch.zeros(n_batch, seq_len, hidden_size, dtype=inputs.dtype, device=inputs.device)
    h_prev = h_0
    for t in range(seq_len):
        h_next = cell(inputs[:, t, :], h_prev)
        hiddens[:, t, :] = h_next
        h_prev = h_next
    return hiddens
```

Simple Python `for` loop. Note that we write into `hiddens[:, t, :]` rather than calling `torch.stack`; both work, but the in-place write avoids a list-of-tensors allocation. For a 100-step unroll this is the only sensible pattern.

The reason `nn.RNN` is faster than this for the same model: PyTorch's implementation runs the loop in C++ (and cuDNN on GPU) without the Python interpreter overhead, and fuses the four element-wise operations (the two matmuls, the add, the tanh) into a single CUDA kernel. The mathematical result is identical; the wall-clock can be 5-10x faster.

### Part C — `rescale_spectral_norm` and `hidden_norm_trajectory`

```python
def rescale_spectral_norm(weight, target_norm):
    s = torch.linalg.svdvals(weight)[0]
    return weight * (target_norm / s)


def hidden_norm_trajectory(cell, h_0, n_steps, input_size):
    n_batch = h_0.shape[0]
    zero_input = torch.zeros(n_batch, input_size, dtype=h_0.dtype, device=h_0.device)
    norms = []
    h_prev = h_0
    for _ in range(n_steps):
        h_next = cell(zero_input, h_prev)
        norms.append(torch.linalg.norm(h_next, dim=-1).mean().item())
        h_prev = h_next
    return norms
```

`torch.linalg.svdvals` returns singular values in descending order; `[0]` is the spectral norm. The rescale is a single multiplication. Both operations are differentiable through; if you wrap the entire trajectory in a `torch.no_grad()` block (which the tests do not require but is a small efficiency), you save the autograd bookkeeping.

The three regimes from Lecture 1 Section 5 are reproduced exactly by setting `target_norm` to 0.5, 1.0, and 1.5 respectively. The contractive case (`0.5`) collapses to near zero by step 30; the balanced case (`1.0`) stays near a fixed-point norm; the expansive case (`1.5`) saturates at the `tanh` ceiling (where each hidden-unit value is near `±1`, so `||h||` is near `sqrt(hidden_size)`).

### Why the gradient-clipping test is structured the way it is

`torch.nn.utils.clip_grad_norm_` returns the *pre-clip* norm (so you can log it) and modifies the gradients in place. The test verifies both behaviors: the returned value equals the pre-clip norm, and the post-clip norm is bounded by `max_norm`. The post-clip can be slightly above `max_norm` (within floating-point error) but should never exceed it by more than 1e-4 in practice.

---

## Exercise 2 — LSTM cell from scratch

### Part A — `CustomLSTMCell.forward`

```python
def forward(self, x, state):
    h_prev, c_prev = state
    # Packed pre-activations: shape (batch, 4*hidden_size)
    gates_in = x @ self.weight_ih.T + h_prev @ self.weight_hh.T + self.bias_ih
    # Split into four chunks. PyTorch's convention: [i, f, g, o].
    i_gate, f_gate, g_gate, o_gate = gates_in.chunk(4, dim=-1)
    i = torch.sigmoid(i_gate)
    f = torch.sigmoid(f_gate)
    g = torch.tanh(g_gate)
    o = torch.sigmoid(o_gate)
    c_next = f * c_prev + i * g
    h_next = o * torch.tanh(c_next)
    return h_next, c_next
```

The single biggest source of bugs in this exercise: **the gate order**. PyTorch's `nn.LSTMCell` packs the four gates in the order `[i, f, g, o]`, which is *not* the order they appear in the Lecture 2 equations (which lists them as `f, i, g, o`). The cuDNN backend follows the same `[i, f, g, o]` convention. If you split in the wrong order, the cell still computes a valid LSTM-shaped function, but it will not match `nn.LSTMCell` and the verification test will fail.

The packed-tensor layout (one weight tensor for all four gates, one bias tensor for all four) is a cuDNN optimization. The four-gate matmul becomes a single `(4*hidden, input + hidden) @ (input + hidden, 1)` matmul, which is more efficient than four separate `(hidden, input + hidden)` matmuls. We use it here because PyTorch uses it.

### Part B — `unroll_lstm_cell`

```python
def unroll_lstm_cell(cell, inputs, h_0, c_0):
    n_batch, seq_len, _ = inputs.shape
    hidden_size = h_0.shape[-1]
    hiddens = torch.zeros(n_batch, seq_len, hidden_size, dtype=inputs.dtype, device=inputs.device)
    h_prev, c_prev = h_0, c_0
    for t in range(seq_len):
        h_next, c_next = cell(inputs[:, t, :], (h_prev, c_prev))
        hiddens[:, t, :] = h_next
        h_prev, c_prev = h_next, c_next
    return hiddens, h_prev, c_prev
```

Same structure as the RNN unroll but two state tensors to track. After the loop, `h_prev` and `c_prev` hold the final hidden and cell states (which we return as `h_T` and `c_T`).

### Part C — `jozefowicz_init_forget_bias` and `cell_state_norm_trajectory`

```python
def jozefowicz_init_forget_bias(cell, value=1.0):
    hidden_size = cell.hidden_size
    with torch.no_grad():
        cell.bias_ih.data[hidden_size:2*hidden_size].fill_(value)


def cell_state_norm_trajectory(cell, h_0, c_0, n_steps, input_size):
    n_batch = h_0.shape[0]
    zero_input = torch.zeros(n_batch, input_size, dtype=h_0.dtype, device=h_0.device)
    norms = []
    h_prev, c_prev = h_0, c_0
    for _ in range(n_steps):
        h_next, c_next = cell(zero_input, (h_prev, c_prev))
        norms.append(torch.linalg.norm(c_next, dim=-1).mean().item())
        h_prev, c_prev = h_next, c_next
    return norms
```

The Jozefowicz fix modifies only one slice of the bias tensor. The `[hidden_size:2*hidden_size]` slice corresponds to the forget gate, because the packed order is `[i, f, g, o]`. The `with torch.no_grad():` block is needed because `bias_ih` is a leaf tensor with `requires_grad=True`; modifying it outside `no_grad` would trip autograd's in-place-modification check.

The cell-state norm trajectory should slow markedly when the forget bias is 1.0: `sigmoid(1.0) ≈ 0.73`, so each step retains 73% of the previous cell-state magnitude (per position) instead of `sigmoid(0) = 0.5` (50%). The test allows the Jozefowicz trajectory's endpoint to be merely *greater* than the default trajectory's endpoint, not by a specific factor, because the actual decay depends on the random initialization and the input/output gates.

---

## Exercise 3 — `pack_padded_sequence`

### Part A — `collate_pad`

```python
def collate_pad(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.long)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=PAD_INPUT)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_TARGET)
    return padded_inputs, padded_targets, lengths
```

`pad_sequence` defaults to `batch_first=False` (sequence-first), which produces `(max_len, batch)` output. We pass `batch_first=True` to get the more PyTorch-conventional `(batch, max_len)`. The `padding_value` controls the fill value; `0` for the inputs is consistent with how `nn.Embedding(padding_idx=0)` works (the embedding for token 0 is held at zero throughout training, so padding does not corrupt the gradient through the embedding lookup), and `-100` for the targets is the magic constant that `nn.CrossEntropyLoss` interprets as "ignore this position" by default.

### Part B — `run_packed_lstm`

```python
def run_packed_lstm(lstm, embed, padded_inputs, lengths):
    emb = embed(padded_inputs)
    packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_output, (h_T, c_T) = lstm(packed)
    padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)
    final_hidden = h_T[-1]  # last layer's final hidden state
    return padded_output, final_hidden
```

Four lines, fiddly. The `lengths.cpu()` is required because the packing routine itself runs on CPU; if you pass a CUDA tensor as `lengths` you get a clear error message but it is a frequent first-time bug. The `enforce_sorted=False` lets you pass batches in arbitrary order; PyTorch reorders internally and the output is in the original order (this is what makes the per-sequence verification test work).

`h_T` has shape `(num_layers, batch, hidden_size)`. `h_T[-1]` is the last layer's final hidden state, shape `(batch, hidden_size)` — the value you typically project to a classification head.

### Part C — `run_lstm_per_sequence`

```python
def run_lstm_per_sequence(lstm, embed, inputs):
    per_seq_outputs = []
    final_hiddens = []
    for seq in inputs:
        emb = embed(seq.unsqueeze(0))   # (1, seq_len, embed_dim)
        out, (h_T, _) = lstm(emb)
        per_seq_outputs.append(out.squeeze(0))
        final_hiddens.append(h_T[-1, 0])
    final_hidden = torch.stack(final_hiddens, dim=0)
    return per_seq_outputs, final_hidden
```

The baseline runs the LSTM on each sequence as a batch of size 1, with no padding involved. The outputs at the *real* positions of the packed-path version must match these outputs to within floating-point tolerance.

### Part D — `SimpleClassifier.forward`

```python
def forward(self, padded_inputs, lengths):
    _, final_hidden = run_packed_lstm(self.lstm, self.embed, padded_inputs, lengths)
    return self.head(final_hidden)
```

Three lines. The classifier discards the per-step outputs and uses only the final hidden state — the standard sequence-classification pattern.

### The bug the packing is preventing

If you skip the `pack_padded_sequence` call and run the LSTM directly on `emb` (which has shape `(batch, max_len, embed_dim)`), the LSTM processes every position, including the padded ones. Three consequences:

1. **The "final hidden state" in `h_T` is the hidden state at `max_len`, not at each sequence's true final position.** For short sequences this is the wrong value entirely. `pack_padded_sequence` returns the hidden state at each sequence's *true* final position, which is what you want.
2. **The padded positions consume compute.** A batch with sequences of lengths 14 and 412 would process 412 timesteps for the short sequence's slot — almost 30x wasted work. On a CPU this is barely noticeable; on a GPU with high-throughput LSTM kernels, packing routinely doubles throughput.
3. **The gradient flows through the padded positions.** The LSTM treats the pad-token embedding as input and updates its parameters to predict the next token from the padding. If your task involves per-step targets (language modeling, sequence labeling), and you mark those targets with `-100`, the loss correctly ignores the padded positions and the gradient through the padding is small; but the hidden state at non-padded positions can still be influenced by what happens at the padded positions, because the LSTM is reading the padding as if it were real.

The pack/unpack dance is the canonical fix for all three issues. Use it.

### Why the classifier training-loss test is structured the way it is

The test verifies that after 20 Adam steps at `lr=1e-2`, the loss has dropped to at least 80% of the initial value. The threshold is generous because the synthetic data (random labels) does not have a strong learnable signal; the loss should drop because the classifier overfits four labeled examples quickly, not because it has learned anything generalizable. A more rigorous test would split into train and test and verify generalization, but for a 4-example synthetic batch the overfit-fast pattern is what you measure.

---

## A note on tests requiring `torch`

All three exercises follow the C5 convention from Week 8/9 of having a `main()` entry that performs shape-only checks (which run with only NumPy and Python's standard library — though Exercise 1 needs `math`, all need `torch` for the unroll itself) and a set of `test_*` functions that perform the deeper verification against PyTorch references. The `_import_torch` helper makes the torch dependency explicit.

The `python3 -m py_compile` check passes without `torch` installed because the imports inside class bodies and function definitions are lazy: they execute only when the function is called, not when the file is compiled. If you copy this pattern to your own work, remember that `mypy` and other static analyzers may complain about the unguarded `torch.Tensor` annotations; the C5 convention is to leave them as string-quoted annotations (`"torch.Tensor"`) so they are treated as forward references and never evaluated.
