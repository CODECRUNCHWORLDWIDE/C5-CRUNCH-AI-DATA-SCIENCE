# Week 11 Mini-Project — A nanoGPT-Style Char-Level Transformer

Build a small decoder-only transformer in PyTorch from scratch and train it on the same `Pride and Prejudice` corpus that Week 10's LSTM trained on. Compare the two models on validation loss, wall-clock per epoch, and qualitative sample quality. Sample passages at three temperatures. Write a portfolio-quality report.

**Estimated time:** about 8 hours total, spread across Friday-Sunday. Of that: ~3 hours scaffolding and training, ~2 hours running the Week 10 LSTM comparison, ~3 hours writing the report and polishing the samples.

**Deliverables (commit to your portfolio repo):**

- `model.py` — the `GPTModel`, `TransformerBlock`, and `MultiHeadAttention` classes.
- `data.py` — the data pipeline (text download, vocab build, chunked streams). May be copied verbatim from the Week 10 mini-project.
- `train.py` — the training loop and validation. May be near-verbatim from the Week 10 mini-project with two small changes (no state threading, lower learning rate).
- `sample.py` — the temperature-scaled generation function.
- `compare-with-lstm.md` — the side-by-side report comparing your Week 10 LSTM and Week 11 mini-GPT. 800-1200 words.
- `samples-at-temperatures.md` — the three final samples at `T = 0.5`, `T = 0.8`, `T = 1.2`, with a 200-word commentary.
- `model.pt` — the trained model checkpoint (optional; large files can be omitted with a note in the report).

---

## Architecture (the C5 defaults)

The model is a pre-norm decoder-only transformer with:

| Hyperparameter | Default | Why |
|---|---|---|
| `vocab_size` | ~80 | Determined by the corpus; the same as Week 10. |
| `d_model` | 384 | Matches the smallest nanoGPT variant. About 7x the Week 10 LSTM. |
| `n_heads` | 6 | So `d_k = 64` per head, the universal choice. |
| `n_layers` | 6 | Deep enough to learn structure; shallow enough to train on CPU. |
| `max_seq_len` | 256 | Longer chunks than Week 10's LSTM; the transformer can exploit them. |
| `dropout` | 0.1 | The 2017 paper's default; effective for our small corpus. |
| `batch_size` | 32 | Same as Week 10. |
| `lr` | 3e-4 | Lower than Week 10's `1e-3`; transformers prefer smaller learning rates. |
| `n_epochs` | 20 | Same as Week 10; the transformer reaches lower loss in fewer epochs but we hold the count fixed for the comparison. |
| `grad_clip` | 1.0 | Same as Week 10. |

Total parameters: about 10.7M. Training time on CPU: ~30-50 minutes for 20 epochs. On a Colab T4: ~3-5 minutes.

---

## Procedure

### Step 1: scaffold the code (1-2 hours)

Start from `starter.py`. It is a single-file runnable scaffold that defines all the model classes and the training loop. To produce the deliverable layout (`model.py`, `data.py`, `train.py`, `sample.py`), split the starter along the section headers — each header marks a logical file boundary.

**Two changes from Week 10**:

1. The model `forward` returns just `logits` (no `state`). Each chunk's forward pass is independent of the previous one.
2. The training loop has no `state.detach()` line between chunks. Transformers have no recurrent state to thread through.

Everything else (data pipeline, optimizer, cross-entropy loss, gradient clipping) is unchanged.

### Step 2: train (30-50 minutes on CPU, 5 minutes on Colab T4)

Run `python train.py` (or `python starter.py` if you have not split yet). The script:

1. Downloads `Pride and Prejudice` (or reuses the Week 10 cache).
2. Builds the char vocab.
3. Chunks the text into `(batch_size, n_chunks, chunk_len)` streams.
4. Builds the model. Prints the parameter count.
5. Trains for 20 epochs. Prints per-epoch training loss, validation loss, and wall-clock.
6. Every 5 epochs, prints a mid-training sample at `T = 0.8`.
7. Saves the final model to `model.pt`.

**Expected results:**

- Training loss should drop from ~3.8 nats (random initial value, `log(80) ≈ 4.4`) to ~1.2 nats over 20 epochs.
- Validation loss should track training closely until epoch ~12; after that, dropout starts to matter and validation may plateau or fluctuate.
- Wall-clock per epoch: 100-150 seconds on a 2.5 GHz CPU; 5-10 seconds on a T4.

**Checkpoint advice:** save after every epoch (`torch.save(model.state_dict(), f"checkpoint-{epoch}.pt")`). If a training run crashes near the end, you can resume rather than restart.

### Step 3: sample (15 minutes)

Run `python sample.py` (or whatever sampling script you split out). The script:

1. Loads `model.pt`.
2. Re-loads the vocab.
3. Samples 200 characters at each of `T ∈ {0.5, 0.8, 1.2}` using the prompt `"It is a truth universally acknowledged, "`.
4. Saves the three samples to `samples-at-temperatures.md`.

**Expected qualitative differences:**

- **`T = 0.5`** is the most deterministic. The model often falls into loops, repeating phrases. The text is most "English-shaped" but least creative.
- **`T = 0.8`** is the "interesting" zone. Most coherent samples; English-shaped, mostly grammatical, occasionally creative.
- **`T = 1.2`** is more random. You see invented words and broken sentence structure. The text is more diverse but less coherent.

The mini-GPT samples at `T = 0.8` should noticeably outperform the Week 10 LSTM samples at the same temperature in terms of long-range coherence (mentioning the same character across the 200-character window, maintaining a topic).

### Step 4: compare with Week 10's LSTM (1 hour)

Open your Week 10 mini-project. Run the LSTM training script for 20 epochs on the same data (or use the previously-trained checkpoint). Sample at the same three temperatures with the same prompt.

Build a side-by-side comparison in `compare-with-lstm.md`:

1. **Architecture summary** (two paragraphs). State both models' hyperparameters and parameter counts. The transformer has ~10.7M parameters; the LSTM has ~1.5M.
2. **Convergence plot** (`loss-vs-epoch.png`). Training and validation loss vs. epoch for both models, on the same axes.
3. **Wall-clock comparison** (one paragraph). Median seconds per epoch for both models on the same hardware.
4. **Sample comparison** (three side-by-side blocks, one per temperature). For each temperature: LSTM sample as a fenced code block, mini-GPT sample as a fenced code block, one-sentence commentary.
5. **Discussion** (one to two paragraphs). Which model wins on each axis? By how much? Cite Lecture 3 Section 12 for the architectural argument; cite your own numbers for the empirical evidence.
6. **Honest caveats** (one paragraph). The transformer is much larger; the comparison is not parameter-matched. Note this. The C5 Challenge 2 of this week offers the parameter-matched version.

### Step 5: write the portfolio report (1.5 hours)

The `compare-with-lstm.md` from Step 4 is the report. Polish it. Run a spelling check. Make sure all numbers are precise (don't say "about 5 seconds" if you measured 4.7 seconds; just say 4.7). Make sure the embedded code blocks render correctly when previewed in Markdown.

**Word counts:**

- `compare-with-lstm.md`: 800-1200 words. The rubric does not penalize concise; it does penalize padding.
- `samples-at-temperatures.md`: 200 words of commentary plus the three samples.

### Step 6: push to portfolio (15 minutes)

Push the deliverables to your portfolio repo under `week-11/`. The rubric grades the files in your portfolio, not the working directory of your dev environment.

---

## What "done" looks like

- [ ] `model.py`, `data.py`, `train.py`, `sample.py` are split out (or, alternatively, the whole thing is in a single `starter.py` and that file is committed). All `.py` files pass `python -m py_compile`.
- [ ] The model trains for 20 epochs and reaches a validation loss below 1.5 nats per character.
- [ ] Three samples at `T ∈ {0.5, 0.8, 1.2}` are in `samples-at-temperatures.md`. Each is at least 200 characters. Each is qualitatively distinct from the others.
- [ ] `compare-with-lstm.md` is 800-1200 words. It embeds the `loss-vs-epoch.png` figure, the LSTM and mini-GPT samples at three temperatures, and the wall-clock comparison.
- [ ] The grading rubric (next file) is internally satisfied.
- [ ] Everything is pushed to your portfolio repo under `week-11/`.

---

## Tips for getting unstuck

- **Training diverges (loss goes to NaN):** lower the learning rate to `1e-4`. The 2017 paper used learning-rate warmup precisely to avoid this; pre-norm transformers mostly do not need warmup but on a small corpus the gradient at step 0 can still spike. If the divergence happens past epoch 1, increase gradient clipping (`max_norm = 0.5` instead of `1.0`).
- **Validation loss starts going up after a few epochs:** the model is overfitting the 700-KB corpus. Either increase dropout (`0.2` or `0.3`), reduce `n_layers` to 4, or stop training earlier. The C5 conviction: small corpora overfit transformers fast.
- **Samples are gibberish:** check the temperature. `T = 0.0` will crash; `T = 100` will produce uniform noise. Make sure you are in the `0.3-1.5` range. Also confirm that `model.eval()` is called before sampling — without it, dropout is still randomizing the residual stream.
- **`F.scaled_dot_product_attention` is unavailable:** you are on PyTorch < 2.0. Either upgrade (`pip install "torch>=2.4"`) or fall back to the explicit `softmax` implementation from Exercise 1. The latter is slower but works.
- **The KV cache is too clever:** skip it. The mini-project does not require KV caching; it is a homework problem (Problem 6) and an Anthropic-paper-tier optimization. Plain re-evaluation at every step is fine for a 200-character generation.

---

## References

- **Vaswani et al. 2017**: <https://arxiv.org/abs/1706.03762>. The paper.
- **Karpathy's `nanoGPT`**: <https://github.com/karpathy/nanoGPT>. The reference implementation. The C5 starter is structurally identical to `nanoGPT/model.py`.
- **Karpathy 2023 "Let's build GPT"**: <https://www.youtube.com/watch?v=kCc8FmEb1nY>. The companion video; 100 minutes, every line of code explained.
- **Alammar 2018 "The Illustrated Transformer"**: <https://jalammar.github.io/illustrated-transformer/>. The pictorial companion.
- **PyTorch `F.scaled_dot_product_attention`**: <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>.
- **PyTorch `nn.LayerNorm`**: <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>.
