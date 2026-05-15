# Lecture 3 — The Decoder-Only Stack and the Mini-GPT

> **Outcome:** You can assemble the pieces from Lectures 1 and 2 into a working decoder-only transformer. You can read every line of the C5 mini-project's `GPTModel` class and explain what it is doing in terms of attention, MLP, residual stream, and unembedding. You can train the model on the same Project Gutenberg corpus that Week 10 used, with the same training loop, and run side-by-side comparisons of validation loss, wall-clock per epoch, and sample quality against the Week 10 LSTM. You can sample from the trained model at three temperatures and describe the qualitative differences. By the end of this lecture, "transformer" stops being a black box and becomes a stack of fifteen-line blocks you can read line by line.

Lectures 1 and 2 built the components. Lecture 3 assembles them.

We target **PyTorch 2.x**. The reference implementations are `nanoGPT/model.py` (<https://github.com/karpathy/nanoGPT/blob/master/model.py>) for the model and `nanoGPT/train.py` for the training loop. The companion video is Karpathy's "Let's build GPT" (<https://www.youtube.com/watch?v=kCc8FmEb1nY>), which walks through the same recipe at the level of every line of code.

---

## 1. The decoder-only architecture, end to end

The 2017 paper introduced two architectures: encoder-decoder (for translation) and the decoder side alone (which was repurposed by the GPT line). The decoder-only stack is what every modern LLM uses: GPT-2, GPT-3, GPT-4, Claude, Llama, Mistral, Falcon, and so on. We build the decoder-only stack this week; the full encoder-decoder is a Week 12 topic.

The decoder-only model has six components:

1. **Token embedding** `nn.Embedding(vocab_size, d_model)`. Maps each integer token ID to a `d_model`-dimensional vector.
2. **Positional embedding** `nn.Embedding(max_seq_len, d_model)`. Adds position-dependent vectors to the token embeddings.
3. **A stack of `n_layer` transformer blocks** (Lecture 2, Section 8). Each block is two sublayers (multi-head causal self-attention and a position-wise MLP), each wrapped by pre-norm LayerNorm and a residual connection.
4. **A final LayerNorm** `nn.LayerNorm(d_model)`. Normalizes the residual stream one last time before the unembed projection (Lecture 2 Section 7 explained why pre-norm transformers need this).
5. **An unembed projection** `nn.Linear(d_model, vocab_size, bias=False)`. Reads the residual stream and produces logits over the vocabulary. The weight is *tied* to the token-embedding weight (Lecture 2 Section 10).
6. **Cross-entropy loss** `nn.CrossEntropyLoss()`. Compares logits to the target (the input shifted by one position). Same as Week 10.

Components 1, 2, 5, and 6 are unchanged from Week 10's char-LM. Components 3 and 4 are the new pieces. The training loop, the data pipeline, the sampling code — all unchanged.

That is the cleanest demonstration of architectural progress in the entire C5 curriculum. Same task, same data, same training loop, two different model classes. The mini-project's `compare-with-lstm.md` write-up makes the comparison quantitative; this lecture makes it precise.

---

## 2. The mini-GPT, in code

The C5 mini-project's `GPTModel` class is structurally identical to `nanoGPT/model.py` with cosmetic renamings:

```python
class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie the token-embedding and output-projection weights.
        self.head.weight = self.tok_embed.weight

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_seq_len, f"sequence length {T} exceeds {self.max_seq_len}"
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x_emb = self.tok_embed(x) + self.pos_embed(positions)      # (B, T, d_model)
        x_emb = self.drop(x_emb)
        for block in self.blocks:
            x_emb = block(x_emb)
        x_emb = self.ln_f(x_emb)
        return self.head(x_emb)                                     # (B, T, vocab_size)
```

About thirty lines including the docstring-equivalent comments. Every line maps to a lecture concept:

- `self.tok_embed`: Week 10 introduced the embedding lookup; same here.
- `self.pos_embed`: Lecture 2 Section 3 (learned positional encoding).
- `self.drop`: Lecture 2 Section 11 (dropout on the input residual stream).
- `self.blocks`: a list of `n_layers` transformer blocks (Lecture 2 Section 8).
- `self.ln_f`: Lecture 2 Section 7 (the final LayerNorm pre-norm transformers need).
- `self.head`: the unembedding (Lecture 2 Section 9).
- `self.head.weight = self.tok_embed.weight`: Lecture 2 Section 10 (weight tying).

The `forward` is even shorter: embed (token + position), drop, run through each block, final norm, unembed. Eight lines of computation.

The `TransformerBlock` is the fifteen-line class from Lecture 2 Section 8. The `MultiHeadAttention` is the twenty-line class from Lecture 1 Section 5. Total: about sixty lines of model code, plus another fifteen for the multi-head attention helper. The mini-project starter has the full thing.

---

## 3. The training loop, identical to Week 10

The training loop is verbatim Week 10:

```python
for epoch in range(N_EPOCHS):
    for chunk_idx in range(n_chunks):
        x = inputs[:, chunk_idx, :].to(device)        # (B, T) token IDs
        y = targets[:, chunk_idx, :].to(device)       # (B, T) shifted targets

        optimizer.zero_grad()
        logits = model(x)                              # (B, T, vocab_size)
        loss = loss_fn(logits.flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
```

Eight lines (plus a `for` over chunks; same as Week 10). The only differences from Week 10's loop:

1. **The model call returns logits, not `(logits, state)`.** A transformer has no recurrent state to thread through chunks; each chunk's forward pass is independent of the previous one. This is the *parallel-training* property from Lecture 1 Section 7.

2. **No `state = (state[0].detach(), state[1].detach())` line.** No state, no detachment. Each chunk is its own forward pass; the gradient does not flow between chunks. This is the same truncated-BPTT trade-off as Week 10, but it falls out automatically because there is no state to pass forward.

Everything else is identical: `Adam(lr=3e-4)` (lower than Week 10's `1e-3` because transformers prefer smaller learning rates), `CrossEntropyLoss`, gradient clipping at `max_norm=1.0`. The C5 conviction stands: write the same eight lines every week.

---

## 4. Sampling with temperature, identical to Week 10

The autoregressive sampling code is also unchanged:

```python
def sample(model, prompt, n_chars, temperature, char_to_idx, idx_to_char, device):
    model.eval()
    out_chars = list(prompt)
    with torch.no_grad():
        input_ids = torch.tensor(
            [[char_to_idx[c] for c in prompt]], device=device, dtype=torch.long
        )
        for _ in range(n_chars):
            # Crop to last max_seq_len tokens (the model has no positional embeddings beyond that).
            input_cropped = input_ids[:, -model.max_seq_len:]
            logits = model(input_cropped)
            last_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out_chars.append(idx_to_char[next_id.item()])
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)
    return "".join(out_chars)
```

Three differences from the Week 10 sampler, all minor:

1. **No state threading.** The transformer is stateless; each generation step re-runs the full prefix through the model. This is `O(T)` extra compute per token compared to the LSTM's `O(1)`, and it is one reason transformer inference is more expensive than RNN inference (in spite of being cheaper *per training step* because of parallelism). The standard production fix is KV-caching, which we do not implement in the mini-project but mention in the homework.

2. **Crop to `max_seq_len`.** The model's positional embeddings are only defined for positions `0..max_seq_len - 1`; generating beyond that would crash. The standard idiom is to keep a sliding window of the last `max_seq_len` tokens. The mini-project's `max_seq_len = 256` and we sample 200 characters, so the cropping never triggers; it is there for correctness in case you bump up the sample length.

3. **`model.eval()` at the start.** Same as Week 10; without it, dropout would still be randomizing the residual stream.

The same three temperatures (0.5, 0.8, 1.2) and the same prompt ("It is a truth universally acknowledged, ") as Week 10. The mini-project rubric asks you to put the LSTM and mini-GPT samples side-by-side for each temperature; the qualitative difference is most visible at `T = 0.8`.

---

## 5. What you should see, quantitatively

When you train the mini-GPT (6 layers, 6 heads, `d_model = 384`, ~10.7M parameters) for 20 epochs on `Pride and Prejudice`, with `lr = 3e-4`, batch size 32, chunk length 256, on a CPU you should see:

- **Wall-clock per epoch:** approximately 100-120 seconds on a 2.5 GHz CPU; 5-10 seconds on a Colab T4; 2-4 seconds on an A100. Compare to the Week 10 LSTM at ~90 seconds CPU / 8 seconds T4 / 2 seconds A100 — the transformer is comparable on CPU, slightly faster on GPU.
- **Final training loss:** approximately 1.1-1.3 nats per character.
- **Final validation loss:** approximately 1.3-1.5 nats per character. The Week 10 LSTM achieves approximately 1.5-1.7 nats per character at the same epoch count on the same data; the mini-GPT wins by 0.2-0.4 nats per character.
- **Sample quality:** at `T = 0.8`, the mini-GPT samples are noticeably more coherent across sentence boundaries than the LSTM. The model "remembers" the topic of a paragraph and uses character names consistently within a 200-character generation. The LSTM tends to drift between topics within the same generation.

These are typical numbers; your seed will affect everything by ~5%, and a GPU may train to slightly different final losses because of nondeterministic kernel choices. The rubric does not grade absolute numbers; it grades the relative comparison.

> **EXPERIMENT — time the mini-GPT and the LSTM on the same batch.** With `T = 256` and `B = 32`, the mini-GPT forward+backward should take 60-100 ms on CPU; the Week 10 LSTM should take 120-180 ms on the same input. The transformer is ~2x faster *per batch* on CPU because the inner loop is matmul-bound rather than sequential-cell-bound. On GPU the gap widens to 4-10x in favor of the transformer. This is the *parallel training* property from Lecture 1 Section 7, expressed in wall-clock terms.

---

## 6. Where the speed advantage comes from

A useful decomposition of why the transformer is faster per training step:

- The LSTM has a sequential dependency along the `T` dimension: `h_t` depends on `h_{t-1}`. The forward pass is `T` matrix-vector multiplies, each waiting for the previous. On GPU, this leaves most of the FLOP budget idle because there is nothing to parallelize.
- The transformer's forward pass is two batched matmuls per block (the `Q K^T` and `A V`), plus an MLP that is matmul-bound. The matmuls operate on the full `(B, T, d_model)` tensor at once; the GPU runs them at peak FLOPs.

This is the difference between an algorithm that the GPU loves (transformer) and one that the GPU tolerates (LSTM). The transformer's quadratic-in-`T` attention is more expensive *asymptotically* than the LSTM's linear-in-`T` recurrence, but for the `T = 256` chunks in the C5 mini-project the constant factors strongly favor the transformer.

The crossover point is around `T = 8000` for matched-parameter models on an A100 — below that, the transformer is faster; above that, the linear-in-`T` recurrence starts to win on raw FLOPs but the transformer's per-step quality remains higher. This is why the long-context modeling literature (Mamba, RWKV, S4) is currently exploring linear-attention variants. Out of scope for Week 11.

---

## 7. The convergence story

A second useful comparison: how quickly the loss drops as a function of training step.

The Week 10 LSTM typically reaches 2.0 nats per character on `Pride and Prejudice` after 5 epochs and plateaus around 1.6 by epoch 20. The mini-GPT typically reaches 1.7 nats per character after 5 epochs and continues to drop, reaching 1.3 by epoch 20.

Two factors:

1. **The transformer has more parameters.** The C5 mini-GPT has 10.7M parameters; the Week 10 LSTM has 1.5M. More parameters means more capacity to fit the training distribution. Challenge 2 of this week strips out the parameter-count difference by training a matched-parameter LSTM (with hidden size 512) and comparing; the transformer still wins, but by a smaller margin.

2. **The inductive bias of attention matches language better than recurrence does.** Even at matched parameters, attention's ability to look across the entire context in one operation is a better fit for language than the LSTM's "forget some of the past at every step." This is the architectural-prior advantage that gave rise to the entire LLM era; it is not just a parameter-count story.

The mini-project's `compare-with-lstm.md` write-up asks you to make this comparison precise on your machine. The rubric awards full marks for a measurement that finds the mini-GPT 0.2-0.4 nats per character ahead of the LSTM after 20 epochs; if your numbers are different, the rubric awards full marks if you can explain why.

---

## 8. Hyperparameter notes (so you know what to vary)

The C5 mini-project defaults:

- `d_model = 384`. Matches the smallest nanoGPT variant.
- `n_heads = 6`. So `d_k = 64` per head — a common choice.
- `n_layers = 6`. Smaller than GPT-2 small (12 layers) but sufficient for a 700 KB corpus.
- `max_seq_len = 256`. Twice the Week 10 LSTM chunk length, exploiting the transformer's parallelism.
- `dropout = 0.1`. The 2017 paper's default; effective for a small corpus where overfitting is real.
- `batch_size = 32`. Same as Week 10.
- `lr = 3e-4`. Lower than Week 10's `1e-3`; transformers prefer smaller learning rates because their gradient through the residual stream is `O(1)` rather than `O(T)` (Lecture 2 Section 7).
- `n_epochs = 20`. Same as Week 10. The transformer reaches lower loss in fewer epochs but we hold the count fixed for the comparison.
- `grad_clip = 1.0`. Same as Week 10. Transformers are less prone to gradient explosion than LSTMs but clipping is cheap insurance.

Things you might profitably vary in the homework or on your own:

- Replace learned positional encoding with sinusoidal. Two-line change; should produce comparable loss.
- Replace pre-norm with post-norm. Set `norm_first=False` in `nn.TransformerEncoderLayer` (if using that) or restructure the `TransformerBlock`. Add learning-rate warmup. Verify that post-norm at depth 6 still trains, then try depth 12 (post-norm often diverges around depth 16+ without warmup).
- Vary `n_layers` over `{2, 4, 6, 8, 12}` at fixed `d_model`. Plot the validation-loss curve. The C5 conviction: validation loss drops monotonically with depth on this corpus up to about 12 layers, then plateaus.
- Vary `d_model` over `{128, 256, 384, 512}` at fixed `n_layers = 6`. Same plot. The C5 conviction: bigger models overfit the 700 KB corpus past `d_model = 512`.

These are the homework problems for the week. The mini-project ships the defaults and expects you to reproduce them; the homework explores the sensitivity.

---

## 9. The KV cache (a preview, no implementation)

A small but important note about inference efficiency.

The mini-project's sampler (Section 4) re-runs the entire prefix through the model at every generation step. For a 200-character generation, that is 200 forward passes of increasing length: the first sees 50 characters, the last sees 250. Total work: `O(T^2)` rather than the `O(T)` you would expect.

The production fix is the **KV cache**. At each generation step, we save the `K` and `V` tensors from each attention layer. On the next step, we only need to compute `K` and `V` for the *new* token; the previous tokens' keys and values are already cached. The attention operation then becomes one row of the `Q K^T` matrix per layer per step — `O(1)` work, not `O(T)`.

Implementing the KV cache adds about thirty lines to the model's `forward` (it needs an optional `cache` argument and must return updated cache tensors) and changes the sampler from a `O(T^2)` loop to an `O(T)` loop. The C5 mini-project does not implement it; the homework's optional problem covers it; nanoGPT has a `generate()` method that does. The standard reference is the discussion in `nanoGPT/model.py` lines 200-260.

The KV cache is the reason real LLM inference is fast despite the model being huge. Without it, ChatGPT would be ten to a hundred times slower than it is. It is a small optimization with an outsized effect.

---

## 10. Numeric stability traps to avoid

Three places where the transformer can produce `NaN` or numerically nonsense outputs if you are not careful:

1. **Softmax with all-`-inf` rows.** If the causal mask is applied incorrectly (e.g., the entire row is `-inf` instead of just the upper-triangular entries), `softmax` produces `NaN`. The standard fix in `F.scaled_dot_product_attention` is to subtract `max(logits)` before exponentiating, but this still fails when *all* logits are `-inf`. The mini-project's `is_causal=True` flag protects against this by construction.

2. **LayerNorm with zero variance.** If the input to LayerNorm is constant across the `d_model` dimension (zero variance), the `1 / sqrt(var + eps)` factor is finite but the gradient through it is ill-conditioned. In practice this never happens with a real residual stream; the small `eps = 1e-5` keeps things stable.

3. **Cross-entropy with logits of huge magnitude.** If the model's output logits become very large (say, `> 100`), the `log_softmax` inside `nn.CrossEntropyLoss` can lose precision in float32. The standard fix is the `log_softmax` trick (subtract the max), which PyTorch does internally. Still, if the gradient clipping is too loose and the unembed layer's output magnitudes drift up, training can become unstable around epoch 10-15. The mini-project's `grad_clip = 1.0` is conservative.

These are the three traps. The mini-project's default code avoids all three; the homework's "build the attention from scratch" path can fall into them.

---

## 11. A worked example: generating 200 characters

Suppose you have a trained mini-GPT and you call:

```python
sample(model, prompt="It is a truth universally acknowledged, ", n_chars=200, temperature=0.8, ...)
```

Here is what happens, step by step:

1. The prompt is encoded to a `(1, 41)` int64 tensor (41 = length of the prompt). Call this `input_ids`.
2. `model(input_ids)` runs the forward pass. The residual stream starts as `tok_embed + pos_embed` (shape `(1, 41, 384)`), flows through 6 transformer blocks (still `(1, 41, 384)`), gets a final LayerNorm, gets unembedded to `(1, 41, 80)` logits.
3. We read `logits[0, -1, :]` — the logits at the last position. This is an 80-vector.
4. Divide by `temperature = 0.8`. Apply softmax. Sample one token from the resulting distribution. Append to `out_chars` and to `input_ids`.
5. Loop back to step 2 with the extended `input_ids` (now length 42). Continue for 200 iterations.

Each iteration costs one forward pass at an increasing prefix length. The total cost is approximately `T_initial + (T_initial + 1) + ... + (T_initial + 200) ≈ 200 * (T_initial + 100) ≈ 28,000` token-positions of attention compute. For a 6-layer 384-dim model on CPU, this is approximately 4-8 seconds. On GPU, under a second.

The KV cache (Section 9) would replace this with `200` token-position evaluations and reduce the wall-clock to approximately 200 ms on CPU. The mini-project ships without the cache for pedagogical clarity; the homework problem optionally implements it.

---

## 12. The Week 10 vs. Week 11 takeaway

Same task. Same data. Same training loop. Two model classes. The transformer wins on every axis we can measure:

- **Lower validation loss in the same training time.** 1.3-1.5 nats per character vs. 1.5-1.7 for the LSTM.
- **Faster wall-clock per epoch on GPU.** 2-3 seconds vs. 5-8 seconds for the LSTM (at matched parameter counts; at the actual parameter counts of 10.7M vs 1.5M, the LSTM is still faster on CPU but slower on GPU).
- **Better long-range coherence.** Samples at `T = 0.8` are visibly more coherent across sentence boundaries.
- **Easier to scale.** The transformer's compute pattern is matmul-friendly; the LSTM's is not. This is why every modern LLM is a transformer, not an LSTM.

The architectural advance the 2017 paper made was not "attention is a clever idea." Attention had been in the NMT literature since 2015 (Bahdanau et al.). The 2017 advance was "we can do *all* the sequence-modeling work with attention, drop the recurrence entirely, and the result trains faster and reaches lower loss." Lecture 3 is the empirical demonstration of that claim on the C5 char-LM task.

The C5 mini-project at the end of this week makes the comparison part of the submission. Train both models for the same number of epochs on the same data. Report the wall-clock, final loss, and a side-by-side sample at `T = 0.8`. The rubric awards full marks for a careful comparison that supports or refutes the textbook claim above; it does not require the transformer to win by any specific margin.

---

## 13. Recap and what is next

You can now:

- Read every line of `GPTModel` and identify its lecture-1 or lecture-2 origin.
- Train the mini-GPT on the same data and training loop as Week 10's LSTM.
- Compare the two models on wall-clock, validation loss, and sample quality.
- Identify the parallel-training property as the reason the transformer is faster per step.
- Identify the constant-time lookback property as the reason the transformer has better long-range coherence.
- Identify the `O(1)` gradient path as the reason the transformer scales to depths far beyond what LSTMs reach.
- Name the KV cache as the standard production optimization for fast inference.

Week 12 introduces the *encoder* half of the 2017 paper, subword tokenization (BPE, SentencePiece), and a small English-to-French translation model. The architecture changes; the training loop does not. The C5 conviction — write the same eight lines every week — continues to hold.

The mini-project is the rest of this week's deliverable. Start with the starter, train it to convergence, sample at three temperatures, write the comparison with Week 10's LSTM, and push to your portfolio repo. The C5 conviction here: you have *built* a small GPT. You can now read any paper on transformer architecture and understand exactly what is being claimed.

References for further reading, all free: Vaswani 2017 (<https://arxiv.org/abs/1706.03762>), Karpathy "Let's build GPT" video (<https://www.youtube.com/watch?v=kCc8FmEb1nY>), nanoGPT (<https://github.com/karpathy/nanoGPT>), Elhage et al. 2021 (<https://transformer-circuits.pub/2021/framework/index.html>), Alammar "Illustrated GPT-2" (<https://jalammar.github.io/illustrated-gpt2/>).
