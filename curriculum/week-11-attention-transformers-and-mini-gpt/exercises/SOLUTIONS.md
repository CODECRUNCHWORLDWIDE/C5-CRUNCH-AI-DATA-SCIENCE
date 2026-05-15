# Week 11 — Exercise Solutions

Reference solutions for the three Week 11 exercises. Read **only after attempting** each exercise. The C5 convention: if you have not spent at least sixty minutes on an exercise on your own, do not open this file. Looking at the solution before trying loses you most of the learning.

Each solution shows the filled-in code with brief commentary. The full lecture context is in `lecture-notes/`; the brief one-paragraph reminders here are not a substitute.

---

## Exercise 1 — Scaled dot-product attention

### Part A — `attention(q, k, v)`

```python
def attention(q, k, v):
    import torch
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    weights = scores.softmax(dim=-1)
    return weights @ v
```

Four lines. The transpose-and-matmul `q @ k.transpose(-2, -1)` is the `Q K^T` from the equation. The division by `math.sqrt(d_k)` is the variance-keeping scaling from Lecture 1 Section 4. The softmax along the last dimension turns each row into a probability distribution. The final matmul `weights @ v` is the convex combination of value vectors.

The function handles arbitrary leading dimensions because `@` and `transpose(-2, -1)` and `softmax(dim=-1)` are all leading-dimension-broadcast. A 3-D input `(B, T, d_k)` produces a 3-D output `(B, T, d_v)`; a 4-D input `(B, n_heads, T, d_k)` produces a 4-D output `(B, n_heads, T, d_v)`.

### Part B — `causal_attention(q, k, v)`

```python
def causal_attention(q, k, v):
    import torch
    T = q.size(-2)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v
```

Five additional lines beyond Part A. `torch.triu(..., diagonal=1)` produces a `(T, T)` boolean tensor that is `True` strictly above the diagonal. `scores.masked_fill(mask, float("-inf"))` replaces the masked positions; after softmax, those positions have weight zero.

The mask is built fresh on every call. In production code (e.g., the mini-project), the mask is registered as a buffer at module init to avoid re-allocation. For the exercise, building on every call is clearer.

### Part C — `softmax_saturation(d_k)`

```python
def softmax_saturation(d_k, seed=RANDOM_STATE, use_scale=True):
    import torch
    torch.manual_seed(seed)
    q = torch.randn(8, d_k)
    k = torch.randn(8, d_k)
    scores = q @ k.T
    if use_scale:
        scores = scores / math.sqrt(d_k)
    weights = scores.softmax(dim=-1)
    return weights.max(dim=-1).values.mean().item()
```

Without the scaling, the max attention weight grows with `d_k` (the softmax saturates onto the argmax). With the scaling, the max weight stays near `0.2-0.4` (the softmax remains spread out over the 8 positions). This is the empirical version of the variance argument from Lecture 1 Section 4: the dot product `q · k` has variance `d_k`, so without scaling the logits' standard deviation grows with `sqrt(d_k)` and the softmax becomes ever more concentrated.

---

## Exercise 2 — Multi-head attention with explicit Q/K/V

### Part A — `build_multihead_attention(d_model, n_heads, causal)`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)
```

The shape progression:

- After `W_q(x)`: `(B, T, d_model)`.
- After `.view(B, T, n_heads, d_k)`: `(B, T, n_heads, d_k)`.
- After `.transpose(1, 2)`: `(B, n_heads, T, d_k)`.
- After `F.scaled_dot_product_attention(...)`: `(B, n_heads, T, d_k)`.
- After `.transpose(1, 2).contiguous().view(B, T, d_model)`: `(B, T, d_model)`.
- After `W_o(out)`: `(B, T, d_model)`.

The `.contiguous()` call after the transpose is necessary because `.view` requires a contiguous tensor; without it, you get a runtime error.

The `is_causal=self.causal` argument to `F.scaled_dot_product_attention` applies the lower-triangular mask internally and dispatches to a kernel that does not materialize the masked positions. On CPU, this is a small constant-factor speedup; on flash-attention-supported GPUs, it is a large speedup.

### Part B — `copy_weights_into_pytorch_mha(ours, theirs)`

```python
def copy_weights_into_pytorch_mha(ours, theirs):
    import torch
    with torch.no_grad():
        in_proj = torch.cat([ours.W_q.weight, ours.W_k.weight, ours.W_v.weight], dim=0)
        theirs.in_proj_weight.copy_(in_proj)
        theirs.out_proj.weight.copy_(ours.W_o.weight)
```

PyTorch's `nn.MultiheadAttention` packs Q, K, V into a single `(3 * embed_dim, embed_dim)` tensor with row layout `[Q-rows; K-rows; V-rows]`. We concatenate our three projection weights along dimension 0 (the output dimension) and copy them in. The output projection weight has the same shape in both modules and is copied directly.

Two gotchas:

1. **`with torch.no_grad()`** prevents the copy from being added to any autograd graph. This is the standard idiom for weight initialization.
2. **`bias=False` in `nn.MultiheadAttention`** is the matching setting. If `bias=True`, PyTorch also has `in_proj_bias` and `out_proj.bias` tensors to copy.

### Part C — `count_attention_parameters(d_model)`

```python
def count_attention_parameters(d_model):
    return 4 * d_model * d_model
```

One line. The formula comes from Lecture 1 Section 5: four `(d_model, d_model)` projection matrices, each with `d_model * d_model = d_model^2` parameters, gives `4 * d_model^2` total. The count is independent of `n_heads` because the per-head parameter count is `d_model * d_k = d_model * (d_model / n_heads)`, and there are `n_heads` heads, so the total is `n_heads * d_model * (d_model / n_heads) = d_model^2` per projection.

---

## Exercise 3 — Causal masks and positional encoding

### Part A — `build_causal_mask` and `apply_causal_mask_to_logits`

```python
def build_causal_mask(T, device=None):
    import torch
    if device is None:
        device = torch.device("cpu")
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def apply_causal_mask_to_logits(scores, mask):
    return scores.masked_fill(mask, float("-inf"))
```

Two lines each. The `diagonal=1` argument to `torch.triu` says "include only entries strictly above the diagonal" — exactly the positions we want to mask out for causal attention. The diagonal itself (i.e., position `i` attending to position `i`) is *not* masked; a token can always attend to itself.

`masked_fill` is the PyTorch idiom for "replace entries where the mask is True with the given fill value." `float("-inf")` is the standard sentinel: `exp(-inf) = 0`, so `softmax(-inf) = 0` for those positions.

### Part B — `sinusoidal_positional_encoding`

```python
def sinusoidal_positional_encoding(seq_len, d_model):
    import torch
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32).unsqueeze(0)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe
```

The structure: `pos` is a column vector of position indices; `div_term` is a row vector of frequency factors. Broadcasting `pos * div_term` produces a `(seq_len, d_model // 2)` matrix where entry `(p, i)` is `p / 10000^(2i / d_model)`. The sine and cosine of this matrix populate the even and odd columns of `pe`.

The `pos * div_term` broadcast is the key trick. Without broadcasting we would write a double loop; with it, the whole encoding is computed in three lines of vectorized PyTorch.

The `math.log(10000.0)` trick converts the formula `1 / 10000^x` to `exp(-x * log(10000))`, which is numerically more stable (no risk of underflow for large `2i`).

### Part C — `build_learned_positional_encoding`

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len):
        assert seq_len <= self.max_seq_len
        positions = torch.arange(seq_len, device=self.pos_embed.weight.device)
        return self.pos_embed(positions)
```

Three lines of substance. The `nn.Embedding(max_seq_len, d_model)` is a `(max_seq_len, d_model)` learnable matrix; indexing it with an integer `i` returns its `i`-th row. The forward builds a `(seq_len,)` integer tensor `[0, 1, ..., seq_len - 1]` and uses it to look up the first `seq_len` positions.

Note that the positional embedding is *trainable*; it starts as the default `nn.Embedding` initialization (small random values) and is updated by the optimizer along with all other parameters. This is the key difference from the sinusoidal version, which is fixed.

The `device=self.pos_embed.weight.device` argument ensures the position indices are on the same device as the embedding table. Without it, if the module is moved to a GPU, the indices would be created on the CPU and the lookup would fail.

---

## Common debugging notes

- **Shape mismatches**: the most common error in this week is forgetting `.contiguous()` after a `.transpose()` before a `.view()`. If you get `RuntimeError: view size is not compatible with input tensor's size and stride`, add `.contiguous()`.
- **Causal mask not strict enough**: if your causality test fails by a small amount (e.g., `1e-3` rather than `1e-5`), check that you applied the mask *before* the softmax, not after. After softmax, the masked positions still have nonzero weight; before, they become exactly zero.
- **`torch.triu` vs `torch.tril`**: `triu` is *upper* triangular (the positions we want to mask out); `tril` is *lower* triangular (the positions we want to keep). The exercises use `triu` for the mask and `masked_fill` to apply it; an equivalent approach would use `tril` and `masked_fill` with the negated mask. Both work; the C5 default is `triu`.
- **`d_model` not divisible by `n_heads`**: the `assert d_model % n_heads == 0` in Exercise 2 is essential. If you forget the assert, the `view` reshape silently produces wrong shapes and the test failures are harder to debug.
- **Float32 vs float64 mismatches**: PyTorch's default dtype is float32; if you accidentally create a positional encoding in float64 (because `math.log` returns a Python float), the addition `token_emb + pe` will upcast the whole tensor to float64 and downstream computations slow down. The exercises pin `dtype=torch.float32` in the relevant `arange` and `zeros` calls.

---

## What to do after these exercises

If all three exercises pass and you have time, the natural next step is the mini-project: assemble the pieces into a working `GPTModel` and train it on `Pride and Prejudice`. The starter code in `mini-project/starter.py` shows the architecture in one place; the mini-project README walks through the training procedure.

If you want more challenge, the two challenges this week (`challenges/challenge-01-attention-head-visualization.md` and `challenges/challenge-02-lstm-vs-mini-gpt-bake-off.md`) extend the exercises with measurement-driven investigations. Both build on the same code you just wrote.
