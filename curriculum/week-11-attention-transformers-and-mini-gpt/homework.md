# Week 11 — Homework

Six problems, about six hours total. Commit each in your Week 11 repo under a `homework/` directory. Each problem expects either a small `.py` file or a short Markdown write-up; specifics below.

---

## Problem 1 — Parameter counts on paper, then verified (45 minutes)

For each of the following transformer configurations, compute the parameter count on paper using the formulas from Lecture 1 Section 5 (attention) and Lecture 2 Section 4 (MLP) and Lecture 2 Section 12 (full model), then verify in PyTorch by counting `sum(p.numel() for p in model.parameters())`.

Configurations:

1. **One transformer block.** `d_model = 64`, `n_heads = 4`, `d_ff = 256`. Formula: `4 * d_model^2 + 2 * d_model * d_ff + 4 * d_model` (the last term is the LayerNorm scale and bias for both norms; 2 norms * 2 * d_model). Verify.
2. **A 6-block stack with embeddings.** `d_model = 384`, `n_heads = 6`, `d_ff = 1536`, `n_layers = 6`, `vocab_size = 80`, `max_seq_len = 256`, weight tying enabled. Formula: `n_layers * (4 * d_model^2 + 2 * d_model * d_ff + 4 * d_model) + vocab_size * d_model + max_seq_len * d_model + 2 * d_model` (the last term is the final LayerNorm). Verify.
3. **GPT-2 small** (the public spec): `d_model = 768`, `n_heads = 12`, `d_ff = 3072`, `n_layers = 12`, `vocab_size = 50257`, `max_seq_len = 1024`. The published parameter count is 124M; compute yours and verify within 1%.
4. **The C5 mini-project defaults**: `d_model = 384`, `n_heads = 6`, `d_ff = 1536`, `n_layers = 6`, `vocab_size = 80`, `max_seq_len = 256`. About 10.7M parameters expected.

For each, write the predicted count and the formula computation (one line each). Save the work as `homework/01-param-counts.md` (table) and `homework/01-verify.py` (verification script that builds the model and counts).

Acceptance: every predicted count matches the PyTorch count within 1%. `python -m py_compile homework/01-verify.py` succeeds. Every function has type hints.

References:
- Lecture 1 Section 5 (attention parameter accounting).
- Lecture 2 Section 4 (MLP parameter accounting).
- Lecture 2 Section 12 (full model parameter accounting).
- nanoGPT model.py: <https://github.com/karpathy/nanoGPT/blob/master/model.py>.

---

## Problem 2 — Reproduce the softmax-saturation experiment (45 minutes)

Lecture 1 Section 4 argued that the `1 / sqrt(d_k)` scaling factor is necessary to keep the softmax in its useful regime as `d_k` grows. Reproduce this empirically.

Build random Q and K matrices of shape `(8, d_k)` with iid standard-normal entries, for `d_k ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512}`. For each `d_k`, compute:

1. The mean of the maximum attention weight (max over keys, mean over queries) *without* the `sqrt(d_k)` scaling.
2. The same quantity *with* the scaling.
3. The mean of the entropy of the attention distribution (in nats), with and without scaling.

Produce `homework/02-saturation.png`: a single figure with four curves (max-weight unscaled, max-weight scaled, entropy unscaled, entropy scaled) on a single log-x axis. Write `homework/02-saturation.md` (~200 words) describing what you see. The expected pattern: unscaled max-weight grows toward 1.0 as `d_k` increases; scaled stays near 0.25; unscaled entropy approaches 0; scaled stays near `log(8) ≈ 2.08` nats.

Acceptance: `homework/02-saturation.png` and `homework/02-saturation.md` are committed. `python -m py_compile homework/02-saturation.py` succeeds. `torch.manual_seed(42)` at the top.

References:
- Lecture 1 Section 4.
- Vaswani 2017 footnote 4: <https://arxiv.org/abs/1706.03762>.

---

## Problem 3 — Read and write up Vaswani 2017 Section 3 (1 hour)

Read **Section 3 ("Model Architecture")** of Vaswani et al. 2017 (<https://arxiv.org/abs/1706.03762>). Pages 2-6 of the PDF.

Write `homework/03-vaswani-section-3.md` (~500 words) covering:

1. **The encoder-decoder structure (section 3.1).** The 2017 paper introduces an encoder *and* a decoder. The C5 Week 11 mini-project uses only the decoder. In your own words, describe what the encoder does and where its output is consumed by the decoder. Reference Figure 1 of the paper.
2. **Scaled dot-product attention (section 3.2.1).** Re-derive the equation `softmax(Q K^T / sqrt(d_k)) V` in your own words. State the variance argument for the `sqrt(d_k)` scaling. Reference Lecture 1 Section 4.
3. **Multi-head attention (section 3.2.2).** Explain why the paper uses `h` heads of dimension `d_k = d_model / h` rather than one head of dimension `d_model`. Reference the parameter-count argument from Lecture 1 Section 5.
4. **Three uses of attention (section 3.2.3).** The paper identifies three places in the architecture where attention appears: encoder self-attention, decoder self-attention, and encoder-decoder cross-attention. For each, state where the queries, keys, and values come from. The C5 mini-project uses only decoder self-attention.
5. **The position-wise feedforward (section 3.3).** Two sentences on what it does and the `d_ff = 4 * d_model` convention.
6. **Positional encoding (section 3.5).** Why is it needed? Why did the paper use the sinusoidal version? Why does the GPT line use learned instead? Reference Lecture 2 Sections 1-3.

Write in your own words; do not paraphrase. Include section and equation numbers you used. The C5 conviction: you should be able to summarize Section 3 of this paper to a colleague in ten minutes.

References:
- Vaswani et al. 2017: <https://arxiv.org/abs/1706.03762>.
- Alammar 2018 (for the diagrams): <https://jalammar.github.io/illustrated-transformer/>.

---

## Problem 4 — Implement a transformer block with `nn.TransformerEncoderLayer` (1 hour)

Write `homework/04-encoder-layer.py` containing a function `compare_blocks(d_model, n_heads, d_ff)` that:

1. Builds a custom `TransformerBlock` (the one from Lecture 2 Section 8, with pre-norm).
2. Builds `nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)`.
3. Copies the custom block's weights into the PyTorch layer (the weights that exist in both modules: the MLP, the two LayerNorms; the attention weights are in different packing formats so you can leave them and use the PyTorch attention).
4. Runs both on a random input of shape `(2, 10, d_model)`.
5. Reports the max absolute difference of the outputs.

The two blocks will *not* agree exactly because of attention-weight packing differences (your `W_q, W_k, W_v` are separate; PyTorch packs them). But the LayerNorm and MLP paths will be identical if you copy those weights. The point of the exercise is to (a) match the structure of the PyTorch built-in and (b) understand which parts of the architecture you need to share vs. let differ.

For full credit, also include a sub-problem 4b: instead of copying weights, set both modules' weights from the same seed (`torch.manual_seed(42)` at the top, build both modules in the same order, then check that the parameter dictionaries have matching keys via `set(ours.state_dict().keys()) ^ set(theirs.state_dict().keys())`). Document the differences.

Acceptance: `python -m py_compile homework/04-encoder-layer.py` succeeds. The script runs to completion with a non-zero (because of attention packing) but small (under 5.0) max diff. Every function has type hints. `torch.manual_seed(42)` at the top.

References:
- `nn.TransformerEncoderLayer`: <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>.
- Lecture 2 Section 8.
- nanoGPT model.py for the comparison structure.

---

## Problem 5 — Layer-count sweep on the mini-project (1.5 hours)

Train the mini-project's `GPTModel` on `Pride and Prejudice` for **5 epochs each** at three depths: `n_layers ∈ {2, 4, 6}`. All other hyperparameters held fixed (`d_model = 384`, `n_heads = 6`, `max_seq_len = 256`, dropout 0.1, lr 3e-4, batch size 32, chunk length 256, gradient clipping at 1.0).

Record:

- Validation loss after each epoch (nats per character).
- Wall-clock seconds per epoch (median across the 5 epochs).
- Total parameter count.

Produce `homework/05-depth-sweep.png` with three validation-loss curves (one per depth) on the same axes. Produce `homework/05-depth-sweep.md` (~250 words) covering:

1. Final validation losses, ordered.
2. Wall-clock per epoch, ordered.
3. Parameter counts, ordered.
4. The depth-vs.-loss argument: validation loss should drop as depth increases, with diminishing returns. Verify or refute on your data.
5. The wall-clock scaling: time-per-epoch should scale roughly linearly with depth. Verify or refute.
6. The C5 conviction: on the 700-KB `Pride and Prejudice` corpus, the depth-quality curve plateaus around `n_layers = 6`. Past that, you start to overfit. If you have a CUDA GPU, run a `n_layers = 12` model and report whether you see the overfit (validation loss going *up* after a few epochs).

Acceptance: `homework/05-depth-sweep.png` and `homework/05-depth-sweep.md` are committed. `python -m py_compile homework/05-depth-sweep.py` succeeds. `torch.manual_seed(42)` at the top.

References:
- The mini-project starter: `mini-project/starter.py`.

---

## Problem 6 — KV-cache implementation (optional, 1 hour)

Lecture 3 Section 9 described the KV cache as the standard production optimization for fast inference. This problem optionally implements it.

Modify the mini-project's `GPTModel.forward` to accept an optional `kv_cache` argument (a list of `n_layers` tuples of `(K, V)` tensors). When `kv_cache` is provided, the model:

1. Only computes Q, K, V for the *new* tokens in the input (typically just the last token).
2. Concatenates the new K, V with the cached K, V for each layer.
3. Returns both the logits and the updated `kv_cache` for the next step.

When `kv_cache` is `None`, the model behaves as before.

Then modify the `sample` function to use the KV cache: at each step, pass only the most-recent token (not the full prefix) and the current cache. Verify that the sampled text is identical to the no-cache version (within reasonable bounds — the kernels are nondeterministic, so a small Hamming distance is acceptable but the text should be clearly the same English).

Measure the wall-clock improvement: time the sample of 200 characters with and without the KV cache. The expected speedup is 5-10x on CPU and 20-50x on GPU for a 200-character generation.

Acceptance: `homework/06-kv-cache.py` is committed; `python -m py_compile` succeeds; the script prints both timings; the sampled texts match to within a Hamming distance of 5 characters out of 200; `torch.manual_seed(42)` at the top.

References:
- Lecture 3 Section 9.
- nanoGPT `model.py` `forward()` method: <https://github.com/karpathy/nanoGPT/blob/master/model.py>. The reference implementation uses a slightly different cache convention; both work.

If you do not have time for this problem, skip it. The mini-project does not depend on the KV cache.

---

## Submission checklist

- [ ] All six problems' artifacts are in `homework/` of your Week 11 repo.
- [ ] Every `.py` file passes `python -m py_compile`.
- [ ] Every figure is reproducible by re-running the corresponding script.
- [ ] All Markdown write-ups stay within the word counts noted in each problem.
- [ ] You set `torch.manual_seed(42)` at the top of every training script.
- [ ] No emojis in any file; type hints on every Python function; the writing voice is plain and declarative.
