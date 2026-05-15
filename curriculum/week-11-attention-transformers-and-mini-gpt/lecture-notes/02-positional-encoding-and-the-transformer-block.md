# Lecture 2 — Positional Encoding, the Transformer Block, and the Residual Stream

> **Outcome:** You can explain why attention by itself has no notion of sequence order (the permutation-equivariance argument from Lecture 1). You can state the two canonical positional-encoding schemes — sinusoidal and learned — and argue when each is appropriate. You can write the full transformer block on paper: two sublayers (multi-head self-attention and a position-wise feedforward MLP), each wrapped by LayerNorm and a residual connection. You can argue for the pre-norm wrapper variant over the original 2017 post-norm. You can read the transformer through the *residual stream* lens that Anthropic's mechanistic-interpretability work uses, which is the modern conceptual frame for understanding what each block contributes to the final prediction. By the end of this lecture, you have every piece needed to assemble a working transformer in Lecture 3.

Lecture 1 built the central computational unit of the transformer: scaled multi-head causal self-attention. The unit takes an input of shape `(B, T, d_model)` and produces an output of the same shape, where information has flowed between positions in a learned, content-dependent way. Lecture 2 wraps that unit into a *block*, and adds the one piece attention does not give us: sequence order.

The block is what stacks. The C5 mini-project at the end of this week stacks six of them; nanoGPT for OpenWebText stacks twelve; GPT-2 small stacks twelve; GPT-3 175B stacks ninety-six. The architecture of one block is fixed; the depth and width are the only knobs you turn as you scale.

We target **PyTorch 2.x**. The primary references are `nn.TransformerEncoderLayer` (<https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>) and `nn.LayerNorm` (<https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>). The conceptual companions are Vaswani 2017 section 3.1 (the "Model Architecture" overview), Xiong et al. 2020 (<https://arxiv.org/abs/2002.04745>) for the pre-norm argument, and Elhage et al. 2021 (<https://transformer-circuits.pub/2021/framework/index.html>) for the residual-stream view.

---

## 1. The permutation-equivariance problem

Recall from Lecture 1 Section 3 that self-attention is permutation-equivariant: if `P` is a permutation matrix and `X` is the input, then `Attention(P @ X) = P @ Attention(X)`. Shuffling the rows of the input shuffles the rows of the output correspondingly, but the *content* of the output at each (now-shuffled) position is unchanged.

This is a problem for language modeling. The sentences "the dog bit the man" and "the man bit the dog" have the same multiset of tokens. Under a permutation-equivariant model with no positional information, they would produce the same output (modulo permutation of the rows). They obviously have different meanings; we need the model to depend on token order.

The fix is **positional encoding**: we add a position-dependent vector to each token's embedding at the input layer. After the addition, the model sees not just "this is the token `dog`" but "this is the token `dog` at position 4." The downstream attention layers then have access to both content and position, and the permutation symmetry is broken.

Two design choices:

1. What kind of vector to add (sinusoidal or learned)?
2. Should the vector be added to the input only (the 2017 paper) or to every block's input (some more modern variants)?

The C5 default for both: learned, applied at the input only. The next two sections motivate the choice.

---

## 2. Sinusoidal positional encoding

The 2017 paper's choice. For each position `pos` and each dimension `i` of the `d_model`-dimensional positional vector, define:

```text
PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
```

The even-indexed dimensions are sines; the odd-indexed dimensions are cosines. The frequency varies geometrically from `1` (at dimension 0) to `1 / 10000` (at dimension `d_model`). The lowest frequency oscillates once per `2 * pi * 10000 ≈ 62832` positions; the highest frequency oscillates once per `2 * pi` positions.

Three reasons the 2017 paper chose this scheme:

1. **No parameters.** The encoding is a deterministic function of position; it adds zero learnable parameters to the model. For a `d_model = 512` model with sequence length `T = 5000`, a learned position embedding would add `5000 * 512 = 2.56M` parameters; the sinusoidal version adds none.

2. **Generalization to unseen positions.** Because the encoding is a fixed function, you can evaluate it at any position, including positions longer than the model was trained on. A learned position embedding has only `max_seq_len` rows; positions beyond that are undefined. (In practice this advantage is theoretical — extrapolating a transformer past its training context length does not work well even with sinusoidal encodings, because the *attention pattern* changes at unfamiliar position scales. Modern positional schemes like RoPE and ALiBi were designed to fix this.)

3. **Linear relationships between positions.** The sinusoidal encoding has the algebraic property that `PE(pos + k)` is a linear function of `PE(pos)`, for any fixed offset `k`. Specifically: there is a rotation matrix `R_k` (depending on `k` but not on `pos`) such that `PE(pos + k) = R_k @ PE(pos)`. The 2017 paper conjectured (without proof) that this property makes it "easy" for the model to learn to attend by relative position. The empirical evidence for this conjecture is mixed; subsequent papers have not consistently found that sinusoidal beats learned on this axis.

**The linear-relationship identity, derived.** Recall the sine and cosine angle-addition formulas:

```text
sin(a + b) = sin(a) cos(b) + cos(a) sin(b)
cos(a + b) = cos(a) cos(b) - sin(a) sin(b)
```

For the `i`-th frequency band (`omega_i = 1 / 10000^(2i / d_model)`), let `theta = omega_i * pos`. Then `PE(pos, 2i) = sin(theta)` and `PE(pos, 2i + 1) = cos(theta)`. The encoding at position `pos + k` for that band is `[sin(theta + omega_i * k), cos(theta + omega_i * k)]`. Substituting the angle-addition formulas:

```text
PE(pos + k, 2i)     = sin(theta) cos(omega_i * k) + cos(theta) sin(omega_i * k)
                    = PE(pos, 2i) * cos(omega_i * k) + PE(pos, 2i + 1) * sin(omega_i * k)
PE(pos + k, 2i + 1) = cos(theta) cos(omega_i * k) - sin(theta) sin(omega_i * k)
                    = PE(pos, 2i + 1) * cos(omega_i * k) - PE(pos, 2i) * sin(omega_i * k)
```

These two equations say that the `(2i, 2i+1)` pair of dimensions transforms under a translation of position by a 2D rotation matrix whose angle is `omega_i * k`. Different frequency bands rotate by different amounts; the full `d_model`-dimensional positional encoding is a stack of `d_model / 2` independent 2D rotations, one per frequency band.

This is the algebraic property the paper claimed. It is also the inductive bias that *Rotary Position Embedding* (RoPE, Su et al. 2021, <https://arxiv.org/abs/2104.09864>) makes explicit and applies inside the attention operation rather than at the input. RoPE is what Llama and most 2024+ open-weight LLMs use; the C5 lecture mentions it for context but does not implement it.

> **EXPERIMENT — visualize the sinusoidal positional encoding.** In a REPL: `import torch; import math; d_model = 64; T = 100; pos = torch.arange(T).float().unsqueeze(1); i = torch.arange(0, d_model, 2).float().unsqueeze(0); div = torch.exp(i * -(math.log(10000.0) / d_model)); pe = torch.zeros(T, d_model); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)`. Plot `pe` as a heatmap (`matplotlib.pyplot.imshow(pe, aspect="auto")`). You will see the band structure: low-frequency bands (right edge of the heatmap) change slowly with position; high-frequency bands (left edge) oscillate rapidly. The plot is in Vaswani 2017 Figure 1 and is the cleanest visual proof that the encoding has the "many different periodicities" property the paper claimed.

---

## 3. Learned positional encoding

The simpler alternative. Define an `nn.Embedding(max_seq_len, d_model)` lookup table; index it by position; add the result to the token embedding. Each row of the table is a learned `d_model`-dimensional vector; the position embedding has `max_seq_len * d_model` trainable parameters.

Three reasons the GPT line of work (Radford et al. 2018, 2019, 2020) preferred this:

1. **Empirically slightly better in-distribution.** GPT-2's ablations reported that learned position embeddings outperformed sinusoidal on the WebText pretraining task. The margin was small (under 0.1 nats per character) but consistent across model sizes.

2. **Simpler implementation.** Two lines of PyTorch instead of the sinusoidal formula. The C5 mini-project uses learned for this reason; the visualization in Section 2 above is run as an EXPERIMENT but is not implemented in the mini-project.

3. **No assumption about position relationships.** The sinusoidal encoding bakes in the linear-relationship-by-rotation property from Section 2. The learned encoding makes no such assumption; the model can learn whatever position representation works best. The downside is that the model cannot evaluate positions outside the training range (no extrapolation), but for fixed-context-length models this does not matter.

The C5 mini-project uses learned positional encoding with `max_seq_len = 256` (the chunk length). In code:

```python
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, ...):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x_emb = self.tok_embed(x) + self.pos_embed(positions)
        for block in self.blocks:
            x_emb = block(x_emb)
        x_emb = self.ln_f(x_emb)
        return self.head(x_emb)
```

This is structurally identical to `nanoGPT/model.py` (<https://github.com/karpathy/nanoGPT/blob/master/model.py>), which Karpathy walks through in his "Let's build GPT" video.

---

## 4. The position-wise feedforward MLP

Each transformer block has two sublayers. The first is multi-head causal self-attention (Lecture 1). The second is a position-wise feedforward MLP that operates on each token's vector independently:

```text
FFN(x) = W_2 @ GELU(W_1 @ x + b_1) + b_2
```

Where:

- `W_1` has shape `(d_ff, d_model)` and `W_2` has shape `(d_model, d_ff)`.
- The intermediate dimension `d_ff` is conventionally `4 * d_model` (this ratio comes from the 2017 paper and has held up empirically; some modern variants use `d_ff = 8/3 * d_model` for GLU-style activations).
- `GELU` is the activation function (Hendrycks and Gimpel 2016, <https://arxiv.org/abs/1606.08415>); the 2017 paper used ReLU, but GPT-2 onward use GELU. Both work; GELU is slightly smoother and slightly improves training stability.

The MLP is applied identically at every position. There is no cross-position interaction in the FFN; that role is reserved for self-attention. The MLP's job is to *compute* on each position's vector independently — to transform the information that attention has just gathered.

A useful frame from Anthropic's interpretability work (Section 7): the FFN is where the model stores *factual knowledge*. The MLP's `(d_ff, d_model)` first weight matrix can be read as a set of `d_ff` "keys" (the rows of `W_1`); for each key, the corresponding column of `W_2` is a learned "value" that gets added to the residual stream when that key fires. This view is supported by experiments showing that you can identify, in a real GPT-2, the specific MLP neurons that fire on tokens like "the Eiffel Tower is in" and that point to the value "Paris" in the residual stream. The view is approximate but useful.

**Parameter accounting.** The MLP has `d_model * d_ff + d_ff * d_model = 2 * d_model * d_ff = 8 * d_model^2` weights (ignoring biases). The attention sublayer has `4 * d_model^2` weights (`W_Q + W_K + W_V + W_O`). So the MLP is roughly *twice* as parameter-heavy as the attention, despite getting less attention in the popular literature.

---

## 5. The residual connection

After each sublayer (attention or MLP), the block adds the sublayer's output back to its input:

```text
x = x + Attention(x)   # not LayerNormed yet; see Section 6
x = x + FFN(x)
```

Two reasons this is essential:

1. **Identity bypass at initialization.** When the sublayer's weights are near zero (as they are at initialization), `x + sublayer(x) ≈ x`. The block computes approximately the identity, which means the gradient flows through the residual path unchanged. Without the residual, the gradient at depth would shrink toward zero (cf. the Week 9 ResNet motivation). Transformer depths of 96+ would be impossible to train without residual connections.

2. **The residual stream view.** Stacking residual blocks gives `x_final = x_input + sum_k sublayer_k(x_k)`. The output is the input *plus* a sum of deltas, one per sublayer. Each sublayer reads from the current state of the residual stream and adds a delta back. This is exactly the picture that the Anthropic 2021 paper popularized; we will use it for the rest of the lecture.

The 2017 paper called the wrapper `LayerNorm(x + sublayer(x))` ("post-norm"). The modern variant is `x + sublayer(LayerNorm(x))` ("pre-norm"), and Section 7 argues for pre-norm.

---

## 6. Layer normalization

The 2017 paper places a LayerNorm operation around the residual connection. `LayerNorm` standardizes each token's `d_model`-dimensional vector to mean 0, variance 1, then applies a learned per-feature scale and bias:

```text
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

Where `mean(x)` and `var(x)` are computed across the `d_model` dimension (not across batch, not across time). `gamma` and `beta` are learned `d_model`-dimensional vectors; `eps` is a small constant (typically 1e-5) for numerical stability.

Three reasons LayerNorm and not BatchNorm:

1. **No batch dependency.** BatchNorm's statistics are computed across the batch dimension, so the value at one example depends on the other examples in the batch. This is awkward at inference time (you need to maintain running statistics) and breaks for variable batch sizes. LayerNorm has no batch dependency; the statistics for example `i` depend only on example `i`.

2. **Works with sequences of variable length.** BatchNorm would also need statistics across the sequence dimension; LayerNorm computes per-token statistics and is independent of `T`.

3. **Stabilizes residual-stream magnitudes.** Without normalization, the residual stream's variance grows linearly with the number of layers (each block adds a delta of comparable magnitude). LayerNorm rescales the stream back to unit variance at every block, keeping the numerical range consistent.

Reference: Ba, Kiros, Hinton 2016 ("Layer Normalization", <https://arxiv.org/abs/1607.06450>). The original BatchNorm paper is Ioffe and Szegedy 2015 (<https://arxiv.org/abs/1502.03167>).

`nn.LayerNorm(normalized_shape=d_model)` is two lines of PyTorch and is what the mini-project uses.

> **EXPERIMENT — verify LayerNorm preserves per-token statistics.** In a REPL: `ln = nn.LayerNorm(64); x = torch.randn(2, 5, 64) * 3 + 7; y = ln(x); print(y.mean(-1), y.std(-1))`. The mean should be approximately 0 and the std approximately 1 along the last dimension, regardless of the input distribution. The batch and time dimensions are untouched.

---

## 7. Pre-norm vs. post-norm

The 2017 paper used the **post-norm** wrapper:

```text
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

The modern convention, popularized by GPT-2 and codified in Xiong et al. 2020 (<https://arxiv.org/abs/2002.04745>), is the **pre-norm** wrapper:

```text
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

The difference is whether LayerNorm is applied *before* the sublayer (pre-norm) or *after* the residual sum (post-norm). The Xiong et al. paper has two arguments for pre-norm:

1. **The gradient through the residual path is smaller in pre-norm.** In post-norm, the gradient through a deep stack accumulates multiplicative factors that grow exponentially in the depth `L`; this forces learning-rate warmup to prevent divergence at the start of training. In pre-norm, the gradient through the residual path is `O(1)` regardless of depth, so no warmup is needed. The original 2017 paper used learning-rate warmup; modern pre-norm transformers do not.

2. **Empirically more stable to train at large scale.** GPT-2, GPT-3, and every modern open-weights LLM (Llama, Mistral, Falcon, OPT) use pre-norm. Empirically, post-norm transformers at 24+ layers require careful learning-rate schedules and are prone to divergence; pre-norm transformers at 96+ layers train with constant or cosine-decay learning rates and rarely diverge.

There is one downside to pre-norm: the residual stream is never re-normalized; over a deep stack the residual's magnitude can grow without bound. The fix is a final `LayerNorm` at the very end of the stack, just before the output projection. The mini-project includes this final LayerNorm (`self.ln_f` in the code block in Section 3).

The C5 mini-project uses pre-norm. PyTorch's `nn.TransformerEncoderLayer(..., norm_first=True)` selects pre-norm; `norm_first=False` (the PyTorch default, matching the 2017 paper) selects post-norm. nanoGPT also uses pre-norm.

---

## 8. The full transformer block

Putting Sections 4-7 together, one pre-norm transformer block is:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

About fifteen lines. That is one block. The full nanoGPT model stacks `n_layer` of these blocks (with `n_layer = 6` in the C5 mini-project, `n_layer = 12` in GPT-2 small, `n_layer = 96` in GPT-3 175B).

The block has the property that the input and output shapes are both `(B, T, d_model)`. The residual stream is preserved. You can stack arbitrarily many blocks and the shape never changes.

---

## 9. The residual stream view, in depth

The Anthropic 2021 paper "A Mathematical Framework for Transformer Circuits" (<https://transformer-circuits.pub/2021/framework/index.html>) introduced (and popularized) a way of reading the transformer that has since become canonical in interpretability research and is increasingly common in pedagogy. The frame:

1. **The residual stream is the central object.** A `(B, T, d_model)` tensor that flows through the entire stack, top to bottom. Each token has its own slice of the residual stream (one `d_model`-dimensional vector); these slices interact only inside attention layers.

2. **Each sublayer reads, computes, and writes back.** Attention reads from the residual stream, gathers information across positions, and writes back to each position. The MLP reads from the residual stream at each position, computes on that position's vector independently, and writes back.

3. **All writes are additive.** Every sublayer's output is *added* to the residual stream. Sublayers do not overwrite; they contribute. This is exactly what the residual connection (Section 5) implements.

4. **The unembed reads the final residual stream.** The output projection `head = nn.Linear(d_model, vocab_size)` reads the final state of the residual stream at each position and produces logits. There is no other readout mechanism in a transformer.

Under this view, the transformer is best read as a set of *learned operations* on the residual stream:

- **Token and position embeddings** initialize the residual stream with content (which token) and position (where in the sequence).
- **Attention layers** move information *between* positions in the residual stream.
- **MLP layers** transform information *within* each position's slice of the residual stream.
- **Unembedding** reads the final residual stream and produces a prediction.

Three implications of this frame:

1. **Different "channels" of the residual stream can carry different information.** Because the residual stream is `d_model`-dimensional and is written into additively, different subsets of the `d_model` dimensions can be allocated by training to different functions. One subset may carry "what token is this"; another may carry "what was the subject of the previous clause"; another may carry "this looks like the second half of an enumerated list." The model learns this allocation. The Anthropic paper calls these "features" or "directions in the residual stream."

2. **Interventions on the residual stream are well-defined operations.** If you zero out a specific direction in the residual stream after a specific layer, you have removed exactly that information from the model's downstream computation. This is the basis of *causal probing* and *activation patching*, two staples of mechanistic-interpretability methodology.

3. **The residual stream is what gets stuck in the bottleneck.** When a transformer "fails to remember" some piece of context, the failure mechanism is almost always that the relevant information was not written into a usable direction of the residual stream. Either no attention head moved it, or it was overwritten by a subsequent MLP, or the unembed projection does not have a strong readout in that direction.

This frame is not necessary for *implementing* a transformer (Section 8 gave you fifteen lines of PyTorch with no mention of residual streams), but it is increasingly necessary for *reasoning* about what a transformer is doing. Challenge 1 of this week reproduces a small piece of the Anthropic methodology by visualizing what individual attention heads do.

---

## 10. Weight tying

A minor implementation detail that is universal in production transformers: the token-embedding matrix `tok_embed.weight` (shape `(vocab_size, d_model)`) is tied to the output projection `head.weight` (shape `(vocab_size, d_model)`). The same `(vocab_size, d_model)` matrix serves both the input-side lookup (which `d_model`-dimensional vector represents token `i`) and the output-side projection (the unembedding).

In PyTorch this is one line at the end of `__init__`:

```python
self.head.weight = self.tok_embed.weight
```

Two reasons:

1. **Parameter savings.** For a `vocab_size = 50257` (GPT-2's BPE vocab) and `d_model = 768` model, the tied matrix has ~38M parameters; un-tying would double-count those, costing 38M extra parameters with no quality benefit.

2. **Empirically slightly better.** Inan et al. 2017 (<https://arxiv.org/abs/1611.01462>) and Press and Wolf 2017 (<https://arxiv.org/abs/1608.05859>) both showed that tied weights produce marginally lower perplexity than un-tied. The effect is small (under 1 perplexity point) but consistent.

The C5 mini-project uses weight tying. nanoGPT uses weight tying. GPT-2 uses weight tying. So does the C5 mini-GPT.

---

## 11. Dropout

The transformer uses dropout in three places:

1. **After the input embedding.** Drops out random dimensions of the input residual stream. Acts as input regularization.
2. **After the attention output projection.** Inside the attention sublayer, after `W_O`. Encourages each head to be useful on its own.
3. **In the MLP, between `GELU` and `W_2`.** Inside the FFN sublayer. Encourages the MLP to be robust.

The 2017 paper used `dropout = 0.1`. nanoGPT uses `dropout = 0.0` for the large OpenWebText models (which never overfit) and `dropout = 0.2` for the small TinyShakespeare model (which does). The C5 mini-project uses `dropout = 0.1`.

Dropout is *disabled* at inference time via `model.eval()`. Forgetting to call `eval()` before sampling is a classic source of confusion: the samples become noisier than they should be because dropout is still randomizing the residual stream.

---

## 12. The shape table (extended)

For a typical decoder-only transformer with `B = 32`, `T = 256`, `d_model = 384`, `h = 6`, `n_layer = 6`, `vocab_size = 80` (the typical char-LM vocab for English):

| Tensor | Shape | Where |
|--------|-------|-------|
| `x` (input token IDs) | `(B, T)` | int64 |
| `tok_embed(x)` | `(B, T, d_model)` | after token embedding |
| `pos_embed(positions)` | `(1, T, d_model)` | broadcast across batch |
| `x_emb = tok_embed + pos_embed` | `(B, T, d_model)` | residual stream entry |
| ...after `n_layer` blocks... | `(B, T, d_model)` | residual stream exit |
| `ln_f(x)` | `(B, T, d_model)` | final layer norm |
| `head(x)` | `(B, T, vocab_size)` | logits over the vocabulary |

The model has roughly `12 * d_model^2 * n_layer + vocab_size * d_model + max_seq_len * d_model` parameters in total. For the C5 mini-project: `12 * 384^2 * 6 + 80 * 384 + 256 * 384 ≈ 10.6M + 30K + 98K ≈ 10.7M` parameters. About 7x the Week 10 LSTM (1.5M parameters), at comparable compute.

---

## 13. Recap and what is next

You can now:

- Explain why attention is permutation-equivariant and why positional encoding is necessary.
- State the sinusoidal positional encoding formula and derive the linear-relationship-by-rotation identity.
- Compare sinusoidal and learned positional encodings; defend the learned choice for the C5 mini-project.
- Write the position-wise feedforward MLP and explain why `d_ff = 4 * d_model` is the standard ratio.
- Wrap each sublayer in `LayerNorm + residual`; defend the pre-norm wrapper over post-norm.
- Read the transformer through the residual-stream lens; describe what each sublayer reads from and writes to.
- Identify weight tying and dropout as the two implementation details that distinguish modern from 2017 transformers.

In Lecture 3 we assemble these pieces into a working decoder-only model, train it on the same Project Gutenberg corpus as Week 10, and compare its convergence, wall-clock per epoch, and final sample quality against the Week 10 LSTM. The model class changes; the training loop does not. That is the entire point of the C5 architectural pivot week.

References for further reading, all free: Vaswani 2017 (<https://arxiv.org/abs/1706.03762>), Xiong et al. 2020 (<https://arxiv.org/abs/2002.04745>), Elhage et al. 2021 (<https://transformer-circuits.pub/2021/framework/index.html>), nanoGPT (<https://github.com/karpathy/nanoGPT>), Alammar's "Illustrated GPT-2" (<https://jalammar.github.io/illustrated-gpt2/>).
