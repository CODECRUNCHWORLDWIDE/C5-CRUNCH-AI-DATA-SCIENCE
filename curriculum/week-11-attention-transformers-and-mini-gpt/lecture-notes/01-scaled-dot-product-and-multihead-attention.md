# Lecture 1 — Scaled Dot-Product Attention and Multi-Head Attention

> **Outcome:** You can derive scaled dot-product attention on paper from the equation `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`. You can identify the `(T, T)` attention-probability matrix as the object that "looks up" pairwise relationships between every position in the sequence. You can argue why the `1 / sqrt(d_k)` scaling factor is necessary (and what goes wrong without it). You can implement multi-head attention as `h` parallel scaled-dot-product attentions on `d_k = d_model / h`-dimensional subspaces. You can apply causal masking to make attention autoregressive. By the end of this lecture, the QKV story stops being a slogan and becomes an algorithm you can write in twenty lines of PyTorch.

Week 10 closed with an LSTM and an explicit list of three things it could not do. It could not look back across 500 tokens in one operation. It could not parallelize its forward pass across the sequence dimension. And its gradient path from output to input scaled linearly with sequence length, so the vanishing-gradient analysis applied. The 2017 paper, "Attention Is All You Need" by Vaswani et al. (<https://arxiv.org/abs/1706.03762>), proposed an architecture that fixes all three.

This lecture builds the single computational unit that does the fixing: **scaled dot-product attention**. The next lecture wraps that unit in a transformer block. The lecture after that stacks the blocks into a decoder-only model and ships a mini-GPT. The recipe is short; the conceptual leap from recurrence to attention is large; this lecture is where the leap happens.

We target **PyTorch 2.x**. The primary reference is `torch.nn.functional.scaled_dot_product_attention` (<https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>); the secondary references are `torch.nn.MultiheadAttention` (<https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>) and the Vaswani 2017 paper. The pictorial companion is Alammar's "Illustrated Transformer" (<https://jalammar.github.io/illustrated-transformer/>); read it before this lecture if you have not already.

---

## 1. The shape of the problem

Recall the sequence-modeling setup from Week 10. An input is a sequence of `T` token embeddings, each of dimension `d_model`. Stacked, the input is a tensor of shape `(T, d_model)` (or `(batch, T, d_model)` in batched form). The model's job is to produce an output of the same shape (one vector per position), where the output at position `t` integrates information from the rest of the sequence.

The Week 10 LSTM did this by carrying a hidden state forward through `T` recurrence steps. The transformer does it differently: at each output position `t`, the model emits a *query vector* and uses that query to look up information from all other positions, in one parallel operation.

The lookup is implemented as a soft, content-based dictionary. Each position contributes a *key* (an index into the dictionary) and a *value* (the content at that index). The query is compared against every key to produce a probability distribution; the output at the query's position is a weighted average of the values under that distribution. This is exactly the operation a Python `dict.get()` would perform, with two differences: the lookup is probabilistic (soft, not hard) and the comparison is via a learned inner product rather than equality.

Three matrices, all of shape `(T, d)` for some dimensions:

- **`Q`**: the query matrix. Row `t` is the query emitted at position `t`. Shape `(T, d_k)`.
- **`K`**: the key matrix. Row `t` is the key at position `t`. Shape `(T, d_k)`. Same dimension as `Q` because the comparison is `Q @ K^T`.
- **`V`**: the value matrix. Row `t` is the value at position `t`. Shape `(T, d_v)`. The dimension of the output is determined by `d_v`.

In **self-attention** — the case that the entire transformer uses — `Q`, `K`, and `V` are all linear projections of the same input `X`. That is:

```text
Q = X @ W_Q      # W_Q has shape (d_model, d_k)
K = X @ W_K      # W_K has shape (d_model, d_k)
V = X @ W_V      # W_V has shape (d_model, d_v)
```

Three learned weight matrices, applied to the same input, producing three different views. The intuition: `W_Q` projects the input into the "what am I looking for" space; `W_K` projects it into the "what content do I have" space; `W_V` projects it into the "if you look me up, here is what you get" space. The three projections need not be related; they are three independently-learned filters.

> **EXPERIMENT — print the three projections of a random input.** In a Python REPL: `import torch; from torch import nn; x = torch.randn(2, 5, 16); q = nn.Linear(16, 8)(x); k = nn.Linear(16, 8)(x); v = nn.Linear(16, 8)(x); print(q.shape, k.shape, v.shape)`. The output is three `(2, 5, 8)` tensors. Three independent linear projections of the same input. The transformer's entire information flow is built out of this pattern.

---

## 2. Scaled dot-product attention, mechanically

Given `Q`, `K`, `V`, attention is computed in three steps:

**Step 1: compute the attention logits.** Take the dot product of every query with every key:

```text
S = Q @ K^T              # shape (T, T)
```

`S[i, j]` is the dot product of query `i` with key `j` — a scalar measuring how relevant key `j` is to query `i`.

**Step 2: scale and softmax.** Divide by `sqrt(d_k)` and apply softmax across the last dimension:

```text
A = softmax(S / sqrt(d_k), dim=-1)    # shape (T, T)
```

`A[i, :]` is now a probability distribution over keys, summing to 1.0. The reason for the `sqrt(d_k)` division is Section 4 below; assume it for now.

**Step 3: weighted sum of values.** The output at position `i` is the convex combination of value vectors under `A[i, :]`:

```text
Output = A @ V           # shape (T, d_v)
```

Output row `i` is `sum_j A[i, j] * V[j, :]`. This is the soft-dictionary lookup.

Putting it together:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

That is the formula from Vaswani et al. 2017 equation 1. Three matrix multiplications and one softmax. The compute is `O(T^2 * d_k)` for the `Q K^T` and `O(T^2 * d_v)` for the `A V`, so the total scales as `O(T^2 * d)`. The memory is dominated by the `(T, T)` attention matrix, also `O(T^2)`. Both numbers are why scaling transformers to long contexts (T > 100K) is hard and why every "efficient attention" paper (FlashAttention, Longformer, RWKV, Mamba) attacks the `O(T^2)` term.

In PyTorch:

```python
import torch
from torch.nn import functional as F

# Q, K, V are (batch, num_heads, T, d_k). For single-head it is (batch, T, d_k);
# we will collapse the head dimension in section 5.
def attention(q, k, v):
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
    weights = scores.softmax(dim=-1)
    return weights @ v
```

Four lines. That is the entire content of the 2017 paper's central equation.

> **EXPERIMENT — compute attention on a `T = 3` toy by hand.** Take `d_k = 2`, `T = 3`. Let `Q = [[1, 0], [0, 1], [1, 1]]` and `K = V = Q`. Compute `S = Q K^T` on paper: `S = [[1, 0, 1], [0, 1, 1], [1, 1, 2]]`. Divide by `sqrt(2) ≈ 1.41`: `S / sqrt(2) ≈ [[0.71, 0, 0.71], [0, 0.71, 0.71], [0.71, 0.71, 1.41]]`. Softmax each row: row 0 is `softmax([0.71, 0, 0.71]) ≈ [0.40, 0.20, 0.40]`. Verify in PyTorch: `q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]); F.scaled_dot_product_attention(q, q, q)`. The output should match your hand-computed `A @ V`. This is the "I have done it" moment.

---

## 3. What attention is doing, in plain English

Three intuitions that the algebra in Section 2 implements:

1. **Soft dictionary lookup.** The query at position `i` is compared to every key in the sequence. Keys that are similar (high dot product) to the query get high attention weight; keys that are dissimilar get low weight. The output at position `i` is then the weighted average of the corresponding values. If the queries and keys were one-hot vectors, attention would degenerate to an exact dictionary lookup: `Output[i] = V[j*]` where `j* = argmax_j q_i · k_j`. The softmax makes the lookup differentiable and the dot product makes it learned.

2. **Content-based addressing.** Unlike a CNN or an RNN, attention has no built-in notion of position. The output at position `i` depends only on the *content* of the keys and values, not on their indices. If we shuffled the input sequence, every output position would change correspondingly (`Attention(P @ Q, P @ K, P @ V) = P @ Attention(Q, K, V)` for any permutation `P`). This is what we mean when we say attention is *permutation-equivariant*. Sequence order has to be added back in via positional encoding; that is the topic of Lecture 2.

3. **Pairwise interaction.** The `(T, T)` attention matrix `A` encodes one number per pair of positions — `A[i, j]` is "how much does position `i` attend to position `j`." The transformer can be read as a model that computes an explicit weighted graph over the input sequence, then propagates information along that graph. The graph is *learned*; it is also *layered* (each transformer block computes a fresh attention matrix), so the model can build complex multi-hop dependencies over depth. This is the lens that Anthropic's mechanistic-interpretability work (Elhage et al. 2021, <https://transformer-circuits.pub/2021/framework/index.html>) uses to dissect what specific attention heads in GPT-2 are doing.

---

## 4. The `sqrt(d_k)` scaling, derived

Why divide by `sqrt(d_k)` in the softmax argument? The honest answer is in footnote 4 of the Vaswani 2017 paper, and it is a one-paragraph variance argument. Here is the derivation.

Suppose `q` and `k` are independent random vectors in `R^d_k`, with each coordinate drawn iid from a distribution of mean 0 and variance 1 (a reasonable model for the initialized projections at the start of training). Then `q · k = sum_{i=1}^{d_k} q_i * k_i` is a sum of `d_k` independent products of mean-0 unit-variance random variables.

The mean of each product `q_i * k_i` is `E[q_i] * E[k_i] = 0`. The variance is `Var(q_i * k_i) = E[q_i^2 * k_i^2] - E[q_i * k_i]^2 = E[q_i^2] * E[k_i^2] - 0 = 1 * 1 = 1`. (The first equality uses independence.)

The variance of the sum of `d_k` independent terms each with variance 1 is `d_k`. So `Var(q · k) = d_k`, and the standard deviation is `sqrt(d_k)`.

Now consider what happens to the softmax. If the inputs to the softmax have standard deviation `sigma`, the *largest* logit in a `T`-way softmax is roughly `sigma * sqrt(2 * log T)` larger than the mean. For typical `T` and `sigma >> 1`, this gap is many standard deviations; the softmax saturates onto the argmax, and the gradient becomes near-zero outside the argmax position.

Concretely: if `d_k = 64` (typical for a transformer head), the dot products have standard deviation `8`. Softmax of logits with `sigma = 8` and `T = 100` puts essentially all mass on one position — the gradient flowing back through softmax is zero almost everywhere, and learning stalls.

The `1 / sqrt(d_k)` scaling normalizes the dot product back to unit variance, regardless of `d_k`. Softmax of unit-variance logits is well-distributed across positions, the gradient is informative, and the model can learn.

This is the entire reason for the scaling factor. It is not a magic constant; it is the unique number that keeps the softmax in its useful regime as `d_k` varies. The "additive attention" of Bahdanau et al. 2015 (<https://arxiv.org/abs/1409.0473>) uses a small MLP instead of a dot product and does not have this issue, but dot-product attention is much faster on GPU (it is just two matmuls), so the 2017 paper used dot-product attention with the scaling fix and the resulting kernel dominated.

> **EXPERIMENT — show the softmax saturating.** In a REPL, compute the attention weights with and without the scale: `d_k = 64; q = torch.randn(8, d_k); k = torch.randn(8, d_k); unscaled = (q @ k.T).softmax(-1); scaled = ((q @ k.T) / d_k**0.5).softmax(-1)`. Print `unscaled.max(-1).values` and `scaled.max(-1).values`. The unscaled version typically has max attention weight 0.9-0.99 (the softmax has collapsed onto one position); the scaled version typically has max 0.2-0.4 (the softmax is spread across positions). Repeat with `d_k = 256`; the unscaled version's max approaches 1.0 as `d_k` grows.

---

## 5. Multi-head attention

A single attention operation produces one `(T, d_v)` output from one set of Q/K/V projections. The 2017 paper's improvement is to run `h` such operations in parallel, on `d_k = d_v = d_model / h`-dimensional subspaces, and concatenate the outputs. The architecture is:

```text
MultiHead(X) = concat(head_1, ..., head_h) @ W_O
where head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)
```

Each `W_Q_i`, `W_K_i`, `W_V_i` is a `(d_model, d_k)` matrix; there are `h` of each. The output of each head is `(T, d_k)`; concatenating `h` heads along the feature dimension gives `(T, h * d_k) = (T, d_model)`. The final output projection `W_O` is `(d_model, d_model)`.

**Why `d_k = d_model / h` and not `d_k = d_model`?** The choice keeps the parameter count of multi-head attention roughly equal to that of a single-head attention with `d_k = d_model`. Single-head with `d_k = d_model` has `3 * d_model^2` parameters in the Q/K/V projections; multi-head with `d_k = d_model / h` has `3 * h * (d_model * d_k) = 3 * d_model^2` parameters total. The multi-head version is parameter-equivalent but computationally distinct: each head operates on a smaller subspace and is forced to specialize, which the empirical literature (Voita et al. 2019, "Analyzing Multi-Head Self-Attention") and the Anthropic interpretability work (Olsson et al. 2022) both confirm.

**Implementation detail: the four projections fuse into one.** Production transformer code never stores `h` separate `W_Q_i` matrices. Instead, it packs them into a single `(d_model, d_model)` matrix `W_Q` whose output is reshaped to `(T, h, d_k)` and transposed to `(h, T, d_k)`. The same for `W_K` and `W_V`. After scaled dot-product attention (now operating on `(h, T, d_k)` tensors with the head dimension treated as a batch dimension), the output is reshaped back to `(T, h * d_k) = (T, d_model)`, and `W_O` projects it. This is one of the standard patterns of transformer code; the C5 mini-project implements exactly this pattern.

In PyTorch:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.W_qkv(x).view(B, T, 3, self.n_heads, self.d_k)
        # Permute to (3, B, n_heads, T, d_k); split into q, k, v.
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # each (B, n_heads, T, d_k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out)
```

About twenty lines of PyTorch. Exercise 2 has you write a slightly more verbose version (with explicit Q/K/V projections rather than the packed three-in-one) and verify against `nn.MultiheadAttention`.

> **EXPERIMENT — verify multi-head attention against the PyTorch built-in.** Build a custom multi-head module with `d_model = 16` and `n_heads = 4`. Build `nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)`. Copy your weights into the PyTorch module's `in_proj_weight` and `out_proj.weight`. Run both on a random `(2, 5, 16)` input. The outputs should agree to within 1e-5. This is the verification Exercise 2 asks you to write.

---

## 6. Causal masking

The attention computed in Sections 2-5 is *bidirectional*: at position `i`, the output depends on every position `j` in the sequence, including future positions `j > i`. For tasks where the entire input is available at training time (sentence classification, masked-language-modeling à la BERT, machine translation's encoder), this is fine.

For *autoregressive* language modeling — what GPT and the C5 mini-project do — it is wrong. We want the model at position `i` to predict position `i + 1` given only positions `1..i`. If the model were allowed to attend to future positions during training, it would learn to "cheat" by reading the answer.

The fix is **causal masking**: before the softmax in step 2, we set the upper-triangular entries of `S` to `-inf`. After softmax, those positions have weight zero, and the output at position `i` depends only on values at positions `1..i`.

```python
T = q.size(-2)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
weights = scores.softmax(dim=-1)
```

The mask is fixed; it has no parameters. It is a function of the sequence length alone. PyTorch's `F.scaled_dot_product_attention` accepts an `is_causal=True` flag that applies the mask internally and dispatches to a kernel that does not materialize the masked positions in memory, so the asymptotic compute drops from `O(T^2)` to `O(T^2 / 2)` (still quadratic in `T` but with a constant-factor speedup). On flash-attention-supported GPUs the speedup is larger.

The mathematical content of causal masking: the attention matrix `A` becomes *lower-triangular*. Row `i` has nonzero entries only at columns `1..i`. This is the architectural property that makes the transformer autoregressive; without it, a transformer cannot be used for next-token prediction.

> **EXPERIMENT — print the causal-masked attention matrix.** With `T = 5`, build a random `Q` and `K`, compute the unscaled scores `S = Q K^T`, apply the upper-triangular mask, softmax: the result is a lower-triangular row-stochastic matrix. Row 0 has nonzero weight only at column 0 (probability 1.0). Row 1 has weight at columns 0 and 1 summing to 1.0. Row 4 has weight at all 5 positions. This is the matrix that causal self-attention computes. Drawing it on a 5x5 grid is the cleanest way to see what "causal" means.

---

## 7. Why this works: the constant-time lookback claim

Recall the Week 10 Lecture 3 list of three things transformers do that LSTMs cannot. We can now name all three precisely.

1. **Constant-time lookback.** In an LSTM, information from position `j` reaches position `i` by propagating through `(i - j)` hidden-state updates, each of which can lose information (Week 10 vanishing-gradient analysis). In a transformer, information from position `j` reaches position `i` in *one* attention operation: `i` queries `j`, gets the value, done. The number of compute steps between any pair of positions is constant (`O(1)`), regardless of how far apart they are.

2. **Parallel training.** In an LSTM, the hidden state at position `i` depends on the hidden state at position `i - 1`, which depends on `i - 2`, and so on. The forward pass is sequential across `i`; you cannot compute `h_{i+1}` until you have `h_i`. In a transformer, all output positions are computed by the same matmul (`Q K^T`) and the same softmax; they parallelize across the time dimension. On a GPU, a transformer's forward pass uses the full FLOP budget; an LSTM's forward pass leaves most of it idle.

3. **`O(1)` gradient path.** In an LSTM, the gradient from the loss at position `T` to a parameter at position `0` traverses `T` cell updates, accumulating through Jacobians that can vanish (Week 10 Section 5). In a transformer, the gradient from the loss at position `T` to a parameter at position `0` traverses one attention layer per block, and there are typically only 6-96 blocks in the entire model. The gradient path depth is `O(n_layers)`, not `O(T)`. Vanishing gradients across the time dimension are no longer possible.

These three properties are the architectural advance the 2017 paper made. The mini-project at the end of this week trains a small transformer and reproduces them quantitatively against the Week 10 LSTM baseline.

---

## 8. Cross-attention: a brief preview

So far we have discussed *self-attention*, where `Q`, `K`, `V` all come from the same input. The 2017 paper also introduced *cross-attention*: the queries come from one sequence (the decoder side) and the keys and values come from another (the encoder side). Cross-attention is what an encoder-decoder transformer uses to let the decoder "look at" the encoder's output during translation.

The C5 mini-project uses only self-attention; cross-attention is a Week 12 topic. We mention it here for completeness because the algebra is identical:

```text
CrossAttention(X_decoder, X_encoder) = Attention(
    X_decoder @ W_Q,  # queries from the decoder
    X_encoder @ W_K,  # keys from the encoder
    X_encoder @ W_V,  # values from the encoder
)
```

The only difference is the source of each projection's input. The scaled dot-product machinery, the multi-head splitting, the `sqrt(d_k)` scaling — all identical. A useful exercise for the curious: skim Vaswani 2017 section 3.2.3 ("Applications of Attention in our Model") to see where in the encoder-decoder architecture each of the three attention uses occurs.

---

## 9. The shape table (pin this)

For a typical decoder-only transformer with:

- batch size `B`
- sequence length `T`
- model dimension `d_model = 384`
- number of heads `h = 6`
- per-head dimension `d_k = d_model / h = 64`

the shapes inside one multi-head self-attention layer are:

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `X` (input) | `(B, T, d_model)` | the residual stream entering the block |
| `Q`, `K`, `V` after `W_QKV` | `(B, T, 3 * d_model)` | packed projection, before splitting |
| `Q`, `K`, `V` after head reshape | `(B, h, T, d_k)` | one head per slice; `Q`, `K`, `V` separated |
| `S = Q @ K^T` | `(B, h, T, T)` | attention logits, one head, one batch element each |
| `A = softmax(S / sqrt(d_k))` | `(B, h, T, T)` | attention probability matrix |
| `out = A @ V` | `(B, h, T, d_k)` | per-head output |
| `out` after concat | `(B, T, d_model)` | heads concatenated along feature dim |
| `W_O(out)` | `(B, T, d_model)` | the block's output, same shape as `X` |

The output shape equals the input shape; the residual connection (Lecture 2) adds them. This is the "residual stream stays the same shape" property that lets us stack arbitrarily many blocks.

---

## 10. A worked numerical example, end to end

Set `B = 1`, `T = 4`, `d_model = 4`, `h = 2`, so `d_k = 2`. Let the input be:

```text
X = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]
```

(One-hot tokens for clarity.) Suppose the QKV projection weights are the identity (so `Q = K = V = X`). The attention logits before scaling are:

```text
S = Q K^T = X X^T = I_4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
```

Each token attends only to itself. Divide by `sqrt(d_k) = sqrt(2) ≈ 1.41`:

```text
S / sqrt(2) ≈ [[0.71, 0, 0, 0], [0, 0.71, 0, 0], [0, 0, 0.71, 0], [0, 0, 0, 0.71]]
```

Apply softmax row-wise. Each row has one logit of 0.71 and three of 0; softmax gives the row `[exp(0.71), 1, 1, 1] / (exp(0.71) + 3) ≈ [0.40, 0.20, 0.20, 0.20]`:

```text
A ≈ [[0.40, 0.20, 0.20, 0.20],
     [0.20, 0.40, 0.20, 0.20],
     [0.20, 0.20, 0.40, 0.20],
     [0.20, 0.20, 0.20, 0.40]]
```

Each row sums to 1.0. With one-hot values, the output is:

```text
Output = A V = A X = A
```

Output row 0 is `0.40 * X_0 + 0.20 * (X_1 + X_2 + X_3) = (0.40, 0.20, 0.20, 0.20)`. Token 0 has output that puts 40% weight on itself and 20% on each of the three other tokens.

Now apply causal masking. The upper-triangular entries of `S` become `-inf`:

```text
S_masked = [[0.71, -inf, -inf, -inf],
            [0,    0.71, -inf, -inf],
            [0,    0,    0.71, -inf],
            [0,    0,    0,    0.71]]
```

Softmax row 0 is `[exp(0.71), 0, 0, 0] / exp(0.71) = [1.0, 0, 0, 0]`. Token 0 attends only to itself (because there is no past). Row 1 is `softmax([0, 0.71, -inf, -inf]) = [0.33, 0.67, 0, 0]`; token 1 attends 33% to token 0 and 67% to itself. Row 3 is `softmax([0, 0, 0, 0.71]) = [0.20, 0.20, 0.20, 0.40]`; token 3 distributes attention over all four positions, with most weight on itself.

This is the entire mechanism. Twelve scalar arithmetic operations per token, repeated for every layer in the stack. The full transformer is this primitive scaled up: more tokens (`T` up to 100,000 in modern models), more dimensions (`d_model` up to 12,288 in GPT-3 175B), more heads (`h` up to 128), more layers (`n_layer` up to 96+).

---

## 11. Recap and what is next

You can now:

- Write the scaled dot-product attention equation from memory.
- Identify the `(T, T)` attention probability matrix and explain what each entry means.
- Argue why the `1 / sqrt(d_k)` scaling is necessary (the variance argument).
- Implement single-head and multi-head attention in PyTorch.
- Apply causal masking and reason about the resulting lower-triangular attention matrix.
- State the three properties (constant-time lookback, parallel training, `O(1)` gradient path) that attention has and recurrence does not.

In Lecture 2 we surround the attention operation with the rest of the transformer block. Specifically: the position-wise feedforward MLP, the LayerNorm wrappers, the residual connections, and the positional encoding (sinusoidal vs. learned) that gives the otherwise-permutation-equivariant attention a notion of order. The lecture ends with the "residual stream" mental model that Anthropic's interpretability work has made the canonical way to read what a transformer is doing internally. Bring your patience for the pre-norm-vs.-post-norm digression; it is short but consequential.

The references for further reading, all free: Vaswani 2017 (<https://arxiv.org/abs/1706.03762>), Alammar's "Illustrated Transformer" (<https://jalammar.github.io/illustrated-transformer/>), Karpathy's "Let's build GPT" video (<https://www.youtube.com/watch?v=kCc8FmEb1nY>), nanoGPT (<https://github.com/karpathy/nanoGPT>).
