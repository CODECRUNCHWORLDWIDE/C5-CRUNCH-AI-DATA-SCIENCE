# Challenge 2 — Matched-Parameter Bake-Off: LSTM vs. Mini-GPT

> Train the Week 10 LSTM and the Week 11 mini-GPT on the same `Pride and Prejudice` corpus at **matched parameter counts** (about 10M parameters each). Hold every other hyperparameter as close as possible (same chunk length, same batch size, same number of epochs, same gradient clipping). Compare wall-clock per epoch, convergence speed (loss vs. epoch), final validation loss, and sample quality at `T = 0.8`. The C5 conviction is that the transformer wins on every axis at matched parameter count, but the margin is smaller than the unmatched comparison from Lecture 3 Section 12 suggests. This challenge measures the margin.

**Estimated time:** 90-150 minutes (depends heavily on whether you have a GPU).
**Deliverable:** A Python script `challenge-02-solution.py` plus a 400-word write-up `challenge-02-writeup.md` with two plots (training-loss curves, wall-clock comparison) and a side-by-side sample. Both committed to your portfolio repo under `week-11/challenges/`.

---

## The setup

The Week 10 mini-project's LSTM has about 1.5M parameters with the defaults (`d_model = 64`, `hidden_size = 256`, `n_layers = 2`). To reach 10M parameters, scale up:

- `hidden_size = 512`
- `n_layers = 3`
- `d_model = 128` (the input embedding dimension; nanoGPT calls this `embed_dim`)
- All other Week 10 defaults (dropout 0.2, batch size 32, chunk length 100, Adam lr=1e-3)

The Week 11 mini-project's transformer has about 10.7M parameters with the defaults (`d_model = 384`, `n_heads = 6`, `n_layers = 6`, `max_seq_len = 256`). Keep these.

Quick parameter accounting for the matched LSTM:
- Embedding: `vocab_size * d_model ≈ 80 * 128 = 10,240`
- LSTM (3 layers, hidden 512, input 128 for layer 1 / hidden 512 for layers 2-3):
  - Layer 1: `4 * (512 * 128 + 512 * 512 + 2 * 512) = 4 * (65,536 + 262,144 + 1,024) = 1,315,840`
  - Layers 2, 3: each `4 * (512 * 512 + 512 * 512 + 2 * 512) = 4 * (262,144 + 262,144 + 1,024) = 2,101,248`
  - Total: `1,315,840 + 2 * 2,101,248 = 5,518,336`
- Output: `512 * vocab_size + vocab_size ≈ 512 * 80 + 80 = 41,040`
- Grand total: ~5.6M parameters. Close but not quite matched.

To push to 10M, also widen the embedding to `d_model = 256` or add a 4th LSTM layer. The exact count is not critical; document what you used.

---

## Requirements

1. **Train both models for the same number of epochs** (suggest 15 epochs; the LSTM at hidden 512 trains a little slower than at hidden 256, so 15 is enough). Record per-epoch training and validation losses.
2. **Record wall-clock per epoch.** Use `time.time()` around the training loop. Report the median across the 15 epochs (the first epoch is sometimes slower due to JIT compilation; the median ignores it).
3. **Final samples at `T = 0.8`.** Use the same prompt for both models: "It is a truth universally acknowledged, ". Sample 200 characters. Save both samples to the write-up.
4. **Two plots.**
   - **Plot 1 (`loss-vs-epoch.png`)**: training and validation loss vs. epoch for both models on the same axes. The transformer should reach lower loss in fewer epochs.
   - **Plot 2 (`wall-clock.png`)**: a bar chart with two bars — median seconds per epoch for the LSTM and for the mini-GPT. Annotate with the actual numbers.
5. **Use `torch.manual_seed(42)` at the top of the script.**
6. **Report parameter counts.** Print both at the start: `LSTM params: X,XXX,XXX; mini-GPT params: X,XXX,XXX`. They should be within 50% of each other.

---

## What you should find

The C5 priors for this challenge:

- **The transformer reaches lower validation loss in fewer epochs.** Typical: LSTM reaches 1.45 nats per char by epoch 15; mini-GPT reaches 1.25 by epoch 15. The transformer plateaus earlier (around epoch 10) and stops improving on this small corpus; the LSTM continues to improve slowly.
- **The transformer is faster per epoch on GPU; comparable or slower on CPU.** On a T4: LSTM ~10 s/epoch, mini-GPT ~5 s/epoch. On CPU: LSTM ~80 s/epoch, mini-GPT ~100 s/epoch (the matched-parameter LSTM has fewer FLOPs because the recurrent kernel reuses the hidden state aggressively). The transformer's GPU advantage is *paralleliam*, not raw FLOPs.
- **Sample quality is visibly better for the transformer.** The mini-GPT samples track topic and character names across the 200-character generation. The LSTM tends to drift. Both produce English-shaped text; both have many typos.

If your numbers differ significantly from these, that is fine — the rubric does not require specific outcomes. It does require an honest report of what you measured and a discussion of why the numbers came out the way they did.

---

## Write-up requirements

In `challenge-02-writeup.md` (about 400 words), cover:

1. **The setup.** State the parameter counts (both should be 5-12M). State the hyperparameters that differed between the two models. State your hardware.
2. **The plots, embedded.** `![](./loss-vs-epoch.png)` and `![](./wall-clock.png)`.
3. **The convergence story.** Which model reached lower loss first? Was there a crossover point? Did either model plateau? Cite Lecture 3 Section 7.
4. **The wall-clock story.** Which model trained faster per epoch? Was the answer the same on CPU and GPU? If you only have one, say so explicitly. Cite Lecture 3 Section 6.
5. **The sample quality.** Embed both 200-character samples as fenced code blocks. Comment in one paragraph on the qualitative differences. Specifically: does the mini-GPT sample show better long-range coherence (mentioning the same character or topic across the whole sample)?
6. **The C5 reading.** Did the transformer win at matched parameters? By how much? Was the margin smaller than the unmatched-parameter comparison from the mini-project would suggest? Comment.
7. **Honest caveats.** State at least three limitations. Examples: "Only one random seed; the LSTM's loss could improve with more epochs"; "The matched-parameter LSTM is unusual — production LSTMs are usually smaller"; "The transformer's `max_seq_len = 256` vs the LSTM's `chunk_len = 100` is a confound."

---

## Hints

- The two models will run with different optimal learning rates. The Week 10 LSTM works with `lr = 1e-3`; the Week 11 mini-GPT prefers `lr = 3e-4`. Use each model's standard LR.
- Use the same data pipeline (`make_chunked_streams` from the Week 10 mini-project). The transformer can use a longer chunk length than the LSTM, but for a clean comparison, use the same chunk length (100) for both. Note this in the write-up.
- If your machine is slow, reduce both models to 10 epochs. The convergence story is usually clear by epoch 10.
- Save intermediate checkpoints. If the script crashes near the end, you do not want to lose 90 minutes of training. Save after every epoch with `torch.save(model.state_dict(), f"checkpoint-epoch-{epoch}.pt")`.
- The KV cache trick (Lecture 3 Section 9) is optional but speeds up sampling 5-10x. If you implement it, note that fact in the write-up; the rubric counts it as bonus credit.

---

## Stretch (optional)

Add a *third* model: a GRU at the same parameter count. The Week 10 mini-project covered GRU in passing (Chung et al. 2014, <https://arxiv.org/abs/1412.3555>); the comparison "GRU vs LSTM vs mini-GPT" gives the full picture of where the architectural advantage lies. Empirically the GRU at matched parameters comes within 5-10% of the LSTM on this corpus; both are 15-20% worse than the transformer. Add a third curve to plot 1 and a third bar to plot 2.

---

## Acceptance criteria

- [ ] `challenge-02-solution.py` runs end-to-end (trains both models, generates the plots and samples).
- [ ] `python -m py_compile challenge-02-solution.py` succeeds.
- [ ] `challenge-02-writeup.md` is 350-500 words; embeds two plots and two samples.
- [ ] Parameter counts are reported; they are within 50% of each other.
- [ ] `torch.manual_seed(42)` is set at the top of the script.
- [ ] All claims about wall-clock are backed by your measured numbers.

The C5 conviction here: the transformer's win over the LSTM is *not* purely a parameter-count story. Even at matched parameters, attention is a better inductive bias for language than recurrence. This challenge makes that claim quantitative on your own machine.

---

## References

- **Vaswani et al. 2017** ("Attention Is All You Need"): <https://arxiv.org/abs/1706.03762>. The transformer paper.
- **Hochreiter and Schmidhuber 1997** (the LSTM paper): <https://www.bioinf.jku.at/publications/older/2604.pdf>.
- **Karpathy 2015** ("The Unreasonable Effectiveness of Recurrent Neural Networks"): <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>. The original char-RNN essay; the inspiration for the Week 10 mini-project.
- **Karpathy 2023** ("Let's build GPT"): <https://www.youtube.com/watch?v=kCc8FmEb1nY>. The companion video for the Week 11 mini-project.
- **nanoGPT**: <https://github.com/karpathy/nanoGPT>. The reference implementation.
- **Chung et al. 2014** (the LSTM-vs-GRU comparison): <https://arxiv.org/abs/1412.3555>. The methodological template for the optional stretch.
