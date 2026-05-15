# Challenge 2 — LSTM vs. GRU Head-to-Head on the Char-LM

> Train an LSTM and a GRU of *matched parameter count* on the same Project Gutenberg corpus, with identical training budgets, and report a head-to-head comparison: wall-clock per epoch, validation loss, sample quality at three temperatures. Either result is acceptable as a finding; what we are after is the experimental discipline.

**Estimated time:** 90-120 minutes (most of it is training).
**Deliverable:** A Python script `challenge-02-solution.py`, a CSV `challenge-02-results.csv`, a 300-word write-up `challenge-02-writeup.md`. Committed to your portfolio repo under `week-10/challenges/`.

---

## The setup

You will train two character-level language models on `Pride and Prejudice` (the mini-project default), with everything held constant except the recurrent cell. The protocol mirrors the mini-project recipe from Lecture 3 Section 7.

### Matching parameter counts

An LSTM with hidden size `k_LSTM` and a GRU with hidden size `k_GRU` have different parameter counts even at the same hidden size: an LSTM has roughly `4 * k * (d + k + 1)` recurrent parameters per layer; a GRU has roughly `3 * k * (d + k + 1)`. To compare them fairly, you should pick `k_GRU` so the GRU's parameter count is within 5% of the LSTM's.

The standard approach: pick `k_LSTM = 256` first (the C5 default). Then solve for `k_GRU` such that `3 * k_GRU * (d + k_GRU + 1) ≈ 4 * k_LSTM * (d + k_LSTM + 1)`. For `d = 64` (the C5 embedding dimension), `k_LSTM = 256`, this gives `k_GRU ≈ 296`. Round to a multiple of 8 (CUDA prefers it) so `k_GRU = 296` or `k_GRU = 304`; verify the parameter count is within 5%.

Both models use:
- The same `nn.Embedding(vocab_size, 64)`.
- A 2-layer recurrent stack with `dropout=0.2`.
- The same `nn.Linear(hidden_size, vocab_size)` head.
- `Adam(lr=1e-3)`.
- Gradient clipping at `max_norm=1.0`.
- 20 epochs, batch size 32, chunk length 100.
- `torch.manual_seed(42)`.

The only architectural difference is `nn.LSTM` vs. `nn.GRU`. The total parameter count of both models should be reported in the write-up.

---

## Requirements

1. **Same train/val split.** Use the first 90% of the corpus for training and the last 10% for validation (a sentence-level split is harder; the deterministic prefix-suffix split is fine for char-LM).
2. **Record per-epoch validation loss and wall-clock for both models.** Save to `challenge-02-results.csv` with columns `epoch, lstm_train_loss, lstm_val_loss, lstm_seconds, gru_train_loss, gru_val_loss, gru_seconds`.
3. **Sample 200 characters from each model at temperatures `0.5, 0.8, 1.2`** at the end of training. Save the six samples to `challenge-02-samples.txt`.
4. **Plot the validation-loss curves on the same axes.** `challenge-02-curves.png`. Two lines, labeled. The y-axis is in nats per character.
5. **Use `torch.manual_seed(42)`** at the top of the script. Use the same seed for both models so the data shuffling is consistent.

---

## What you might find

The Chung et al. 2014 paper (<https://arxiv.org/abs/1412.3555>) reported LSTM and GRU were statistically indistinguishable on polyphonic music and speech-signal modeling. The C5 conviction is that the same is true on small-corpus char-LM: the validation loss after 20 epochs should be within 0.05 nats per character between the two models. Wall-clock per epoch: GRU is typically 10-15% faster than LSTM at matched parameter count, because the GRU has three matmuls per step vs. the LSTM's four.

Sample quality at `T = 0.8` should be visually similar — both models produce English-shaped sentences with mostly-real words and mostly-correct grammar. At `T = 0.5` both are repetitive; at `T = 1.2` both produce more nonsense but with longer creative pseudo-words.

**Honest about what is *not* an interesting finding.** If the two models' final validation losses are within 0.05 nats and the samples are visually indistinguishable, the right write-up is *not* "the GRU lost / won" but "the cell choice does not matter much for this task and corpus size, consistent with the literature." That is a respectable, well-supported scientific conclusion.

---

## Write-up requirements

In `challenge-02-writeup.md` (about 300 words), cover:

1. **The protocol.** Two sentences describing what you held constant and what you varied.
2. **Parameter counts.** Report both. Verify they are within 5%.
3. **The headline numbers.** Final validation loss for both. Per-epoch wall-clock for both. Total training time for both.
4. **The plot, embedded.** `![](./challenge-02-curves.png)`.
5. **Three sample passages.** One from each cell at `T = 0.8`, plus one more from your favorite (your choice; flag which one).
6. **Interpretation.** One paragraph. If the numbers are within noise, say so honestly. If one model clearly won, propose a specific reason (e.g., "the LSTM has a separate cell state that may help on this corpus because..." — but be honest if you cannot identify a mechanism).
7. **What you would do next.** One sentence. If the result was within noise, the next experiment is probably "scale both up" or "try a different corpus."

---

## Hints

- The `Adam` optimizer keeps internal state. Build a *fresh* optimizer for each model, not a shared one.
- The wall-clock measurement should exclude data loading. Time the inner training loop only, not the file-read at the top.
- The first epoch is often slower than subsequent epochs because of CUDA kernel compilation. Report the *median* per-epoch wall-clock, not the mean.
- The `Pride and Prejudice` text has some boilerplate at the top (Project Gutenberg license preamble) and bottom (license afterword). Strip these before training; the mini-project starter has a helper.
- Sampling from both models at the same temperature with the same prompt is the cleanest visual comparison. Use the prompt `"It is a truth universally acknowledged, "` (the opening of Pride and Prejudice).

---

## Acceptance criteria

- [ ] `challenge-02-solution.py` runs end-to-end. Total wall-clock: ~60 minutes on CPU, ~10 minutes on a Colab T4.
- [ ] `python -m py_compile challenge-02-solution.py` succeeds.
- [ ] `challenge-02-results.csv` has 20 rows plus the header.
- [ ] `challenge-02-samples.txt` has six labeled passages.
- [ ] `challenge-02-curves.png` shows two lines.
- [ ] `challenge-02-writeup.md` is 250-350 words and contains all seven elements.

---

## A note on what this challenge is teaching you beyond LSTM vs. GRU

The deliverable is the *experimental protocol*, not the headline number. In a portfolio review, the question is not "which cell won?" — that is a 2014 question, and the literature has answered it (they are basically tied). The question is "can you set up a controlled comparison, hold the right variables constant, report honest numbers, and acknowledge when the result is null?" This challenge is a mini-version of the experimental discipline that real ML research demands. Treat it accordingly.

---

## References

- **Chung, J.; Gulcehre, C.; Cho, K.; Bengio, Y. (2014).** "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling." *NIPS 2014 Workshop.* arXiv:1412.3555. <https://arxiv.org/abs/1412.3555>. The original LSTM-vs-GRU comparison; the C5 conviction traces to this paper.
- **`nn.LSTM`:** <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>.
- **`nn.GRU`:** <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>.
- **Karpathy 2015:** <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>. The conceptual reference for the char-LM setup.
