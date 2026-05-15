# Week 11 Mini-Project — Grading Rubric

Total: 100 points. Pass threshold: 70.

The C5 conviction: the rubric grades *the engineering and the comparison*, not the absolute loss. A careful mini-project that produces a validation loss of 1.7 (worse than the C5 prior of 1.3-1.5) is graded higher than a sloppy one that produces a validation loss of 1.2 with no comparison report.

---

## Code quality (35 points)

- **Model implementation correctness (15).** The `GPTModel`, `TransformerBlock`, and `MultiHeadAttention` classes implement the pre-norm decoder-only architecture from Lecture 2. Specifically:
  - Pre-norm wrapper (`x = x + sublayer(LayerNorm(x))`).
  - Causal masking in the self-attention (use `F.scaled_dot_product_attention(..., is_causal=True)` or an equivalent explicit mask).
  - Weight tying between `tok_embed.weight` and `head.weight`.
  - Final `LayerNorm` before the output projection.
  - Position-wise MLP with `GELU` activation and `d_ff = 4 * d_model`.
  - Type hints on every function and method.
- **Training-loop correctness (10).** The script:
  - Uses `Adam(lr=3e-4)`.
  - Uses `CrossEntropyLoss`.
  - Clips gradients with `clip_grad_norm_(parameters, max_norm=1.0)`.
  - Computes validation loss on a held-out 10% of the corpus.
  - Sets `torch.manual_seed(42)` at the start.
  - Saves the final model to `model.pt`.
- **Sampling correctness (5).** The `sample` function:
  - Calls `model.eval()` before generating.
  - Wraps the loop in `torch.no_grad()`.
  - Crops the input to `max_seq_len` tokens (so position-embedding bounds are respected).
  - Uses `torch.multinomial` (or equivalent) for temperature-scaled sampling.
- **No emojis, no exclamation marks, no excess docstrings (5).** The C5 voice convention. Code comments explain *why*, not *what*. Function docstrings are short and informative.

---

## Training and convergence (20 points)

- **Final validation loss under 1.7 nats per character (10).** With the C5 defaults, the model typically reaches 1.3-1.5. The threshold is 1.7. If your loss is above 1.7, document what you tried (lower learning rate, different seed, longer training) and why you think it failed.
- **No NaN or Inf during training (5).** The script runs to completion without numerical disaster. If you see a NaN, fix it (lower learning rate, tighter gradient clipping, smaller initialization scale).
- **The `loss-vs-epoch.png` figure exists and is reproducible (5).** Re-running `train.py` produces the same figure (modulo random fluctuations). The figure shows both training and validation loss on the same axes.

---

## Samples (15 points)

- **Three samples at three temperatures (10).** `T ∈ {0.5, 0.8, 1.2}`, prompt `"It is a truth universally acknowledged, "`, 200 characters each. All three are committed to `samples-at-temperatures.md` as fenced code blocks.
- **Qualitative differentiation between temperatures (3).** The `T = 0.5` sample is visibly more deterministic (loops, repetition) than `T = 1.2` (more diversity, more invented words). The commentary in `samples-at-temperatures.md` describes the differences.
- **English-shapedness (2).** All three samples are recognizable as English-shaped text. Most words are real English words; sentences end with sensible punctuation. The mini-GPT at 20 epochs typically produces text in this regime even at `T = 1.2`.

---

## Comparison with the Week 10 LSTM (25 points)

- **The `compare-with-lstm.md` report exists, is 800-1200 words, and is internally well-structured (10).**
- **The convergence comparison is supported by your own measured numbers (5).** Both models' final validation losses are reported and embedded in the figure. The transformer should typically reach lower loss; if it does not on your machine, document why.
- **The wall-clock comparison is reported (3).** Median seconds per epoch for both models on the same hardware. If you only have CPU, say so; if you have both, report both.
- **The sample comparison is side-by-side (5).** Three temperatures, each showing LSTM and mini-GPT outputs. At least one paragraph of commentary explaining the qualitative differences.
- **The report has at least one honest caveat (2).** Examples: "The transformer is much larger; the comparison is not parameter-matched"; "Only one random seed"; "I only ran 10 epochs because of CPU time."

---

## Code organization and portability (5 points)

- **Files are split or the single-file `starter.py` is committed (2).** Splitting is preferred (matches the production layout) but not required.
- **`python -m py_compile` succeeds on every `.py` file (2).** Verified before submission.
- **The README explains how to reproduce your results (1).** A one-paragraph "to reproduce, run X then Y then Z" at the top of `compare-with-lstm.md` or in a separate `README.md`.

---

## Bonus (up to 10 points; do not count toward base 100)

- **KV cache implementation (5).** Lecture 3 Section 9 described the optimization; Homework Problem 6 asks for it explicitly. If you implement it for the mini-project's sampler, document the 5-50x speedup measurement in `compare-with-lstm.md`.
- **A parameter-matched comparison with the LSTM (3).** Challenge 2 of this week is exactly this; if you have done Challenge 2, embed the result in `compare-with-lstm.md` for bonus.
- **Attention-head visualization (2).** Challenge 1 of this week renders the attention probability matrix of one or more heads. If you have done Challenge 1, embed at least one heatmap in `compare-with-lstm.md`.

---

## Submission requirements

Commit the following files to your portfolio repo under `week-11/mini-project/`:

- [ ] `model.py` (or include the model code in `starter.py`).
- [ ] `data.py` (or include in `starter.py`).
- [ ] `train.py` (or include in `starter.py`).
- [ ] `sample.py` (or include in `starter.py`).
- [ ] `compare-with-lstm.md` (800-1200 words; embeds figures and samples).
- [ ] `samples-at-temperatures.md` (three samples + 200 words of commentary).
- [ ] `loss-vs-epoch.png`.
- [ ] `model.pt` (the trained checkpoint; optional, can be omitted with a note explaining why).

Push to GitHub. Verify the file structure renders correctly on the GitHub web UI before requesting grading.

---

## A note on partial credit

The rubric does not require any specific number; it requires *honest measurement and clear writing*. A mini-project that ships a validation loss of 1.65 (above the typical 1.3-1.5 but below the threshold of 1.7) with a careful write-up of why is graded above a mini-project that ships 1.25 with no comparison report.

If something does not work — the training diverges, the sampling produces gibberish, the comparison numbers come out wrong — write about it. The portfolio repo is a record of *what you did and what you learned*, not a leaderboard submission. A clean, honest report of a partial result is worth more than a polished report of a magical success that we cannot verify reproduced.
