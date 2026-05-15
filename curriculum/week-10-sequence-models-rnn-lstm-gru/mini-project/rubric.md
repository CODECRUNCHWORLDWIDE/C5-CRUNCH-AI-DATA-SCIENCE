# Week 10 Mini-Project — Grading Rubric

The mini-project is graded out of **100 points** across six dimensions. Self-grade against this rubric before submitting; the C5 grader will use the same rubric.

---

## 1. Reproducibility (15 points)

| Points | Criterion |
|-------:|-----------|
|     5 | `pip install -r requirements.txt && python train.py` runs end-to-end from a fresh clone, without manual intervention. `requirements.txt` pins `torch>=2.4,<3`. |
|     5 | `torch.manual_seed(42)` is at the top of `train.py` and `sample.py`. Running `train.py` twice with the default settings produces final validation losses that agree to the third decimal. |
|     5 | The corpus download is cached in `./data/`; subsequent runs do not re-fetch. The script handles the case where `./data/` does not exist (creates it). |

**Common deductions:** missing `requirements.txt`; missing seed; using `np.random.seed` instead of `torch.manual_seed`; hardcoded paths to your home directory.

---

## 2. Model correctness (20 points)

| Points | Criterion |
|-------:|-----------|
|     5 | `CharLSTM` is an `nn.Module` subclass with the architecture from Lecture 2 Section 9 (Embedding → LSTM → Linear). |
|     5 | The LSTM has `num_layers=2`, `dropout=0.2`, `batch_first=True`. |
|     5 | The forward returns `(logits, state)` where `state` is a 2-tuple of `(h, c)` ready to feed back on the next chunk. |
|     5 | Parameter count is in the 1.0M-2.0M range. Smaller models (<500k) typically underfit; much larger (>3M) overfit the 700-KB corpus. |

**Common deductions:** missing or incorrect dropout setting; using `nn.RNN` instead of `nn.LSTM`; forgetting `batch_first=True`; returning only logits without the state (breaks truncated BPTT).

---

## 3. Training loop (20 points)

| Points | Criterion |
|-------:|-----------|
|     5 | The training loop uses truncated BPTT: hidden state is detached at chunk boundaries, passed forward within an epoch, and reset at epoch boundaries. |
|     5 | Gradient clipping at `max_norm=1.0` is in the loop. (Specifically, `torch.nn.utils.clip_grad_norm_` called after `loss.backward()` and before `optimizer.step()`.) |
|     5 | Per-epoch validation loss is computed on a held-out split of the corpus (the last 10% is the C5 default). Validation runs under `torch.no_grad()` and `model.eval()`. |
|     5 | `Adam(lr=1e-3)` (or a learning-rate schedule, justified in the report). |

**Common deductions:** running BPTT across chunks without detaching (out of memory on long training); no gradient clipping (training diverges around epoch 5-10 with default initialization); validation in training mode (dropout corrupts the loss); SGD without momentum (trains an order of magnitude slower).

---

## 4. Final validation loss (15 points)

| Points | Criterion |
|-------:|-----------|
|    15 | Final validation loss ≤ **1.40 nats/char** on the held-out 10% of `Pride and Prejudice`. |
|    12 | Final validation loss ≤ 1.45 nats/char. |
|     9 | Final validation loss ≤ 1.50 nats/char. |
|     6 | Final validation loss ≤ 1.55 nats/char. |
|     3 | Final validation loss in `(1.55, 2.0]` nats/char. |
|     0 | Validation loss > 2.0 nats/char, or no validation loss reported. |

For reference: random guessing on an 80-char vocab is `log(80) ≈ 4.4` nats/char. A well-trained 2-layer 256-hidden LSTM on `Pride and Prejudice` typically reaches 1.35-1.50. Reaching below 1.35 usually requires either a larger model (which starts to overfit) or a larger corpus. The rubric does not reward going lower than 1.35.

**Honesty note.** Report the *validation* loss, not the training loss. The training loss after 20 epochs is typically 0.5-1.0 nats/char lower than the validation loss; reporting the lower number is overfitting your portfolio reviewer, not the data.

---

## 5. Sample quality (15 points)

| Points | Criterion |
|-------:|-----------|
|     5 | Three sampled passages at `T ∈ {0.5, 0.8, 1.2}`, each 200 characters, included in the report and labeled. |
|     5 | At `T = 0.8`, the sample is recognizably English (no random-character regions), most words are real English words, sentences have plausible structure. |
|     5 | The three samples show the expected temperature progression: T=0.5 is repetitive/conservative; T=0.8 is balanced; T=1.2 has more invented pseudo-words and broken structure. |

**Common deductions:** all three samples look identical (temperature is not actually being applied); samples are < 100 characters (rubric requires 200); samples are *from the training corpus* (the model is memorizing rather than generating — likely a bug in the train/val split or the truncated-BPTT logic); samples are full of Project Gutenberg license boilerplate (the preamble strip is broken).

---

## 6. Report quality (15 points)

| Points | Criterion |
|-------:|-----------|
|     3 | Report is 600-900 words. Strictly enforced; outside this range loses the full 3 points. |
|     3 | Training and validation loss curves embedded (`training_curves.png`). The y-axis is labeled "loss (nats/char)". |
|     3 | Three labeled sample passages embedded in fenced code blocks, with their temperatures noted. |
|     3 | The "discussion" section makes at least two specific qualitative observations about what the model does well or badly. The observations are anchored in the data ("at T=0.5 the sample contains the word 'Darcy' 12 times" rather than "the model is repetitive"). |
|     3 | The "what we would do next" section mentions training a transformer on the same corpus for comparison, and references Vaswani et al. 2017 (Week 11's reading). The tease at the end of Lecture 3 should be reflected here. |

**Common deductions:** report is too short (<600) or too long (>900); no figures embedded; samples are in a single paragraph rather than fenced; the "what we would do next" section is generic ("try more data, bigger model") rather than specific.

---

## Bonus achievements (+5 each, up to +15)

- **Reach validation loss ≤ 1.30 nats/char.** Documented in the report; requires either careful hyperparameter tuning, a learning-rate schedule, or a longer training run on a larger corpus.
- **LSTM vs. GRU bake-off.** Train both at matched parameter counts; report a head-to-head table. (Doing Challenge 2 satisfies this bonus — flag it in the report.)
- **Top-k filtered sampling.** Implement and demonstrate `top_k` sampling in `sample.py`; show that at `T = 1.2` with `top_k = 20`, samples look noticeably more coherent than full-softmax sampling at the same temperature. Cite Fan et al. 2018 (<https://arxiv.org/abs/1805.04833>).

---

## Auto-failures (do not pass go)

- **No `model.pt` file** in the deliverable.
- **`python train.py` does not run** from a fresh clone (missing dependency, missing data path, import error).
- **The validation loss is computed on the training data.** This is the "cheat by looking at the test set" failure; auto-zero on the validation-loss section.
- **Samples are clearly directly quoted from the training corpus.** The model is supposed to *generate* text, not retrieve it. If `sample.py` is doing something like printing random spans of the training corpus, that is dishonest.

---

## A note on the writing voice

The C5 conviction on writing: declarative sentences, present tense, concrete numbers. The bad version: "Our model performed well." The good version: "The 2-layer LSTM reached 1.42 nats/char on the held-out 10% of Pride and Prejudice after 20 epochs of training, 1.5 minutes per epoch on a Colab T4." Specific, verifiable, useful.

Avoid: hype, emojis, "amazing", "state of the art", "cutting edge". The model is a 1.5M-parameter char-level RNN; it is not state-of-the-art and was not state-of-the-art in 2017 when transformers replaced it. Honesty about scope is part of what makes the report credible.

---

## Self-grade template

Copy this into a comment in your `report.md` before submitting:

```
Self-grade (out of 100):
- Reproducibility:    /15
- Model correctness:  /20
- Training loop:      /20
- Final val loss:     /15  (my val loss: X.XX nats/char)
- Sample quality:     /15
- Report quality:     /15
- Bonus:              /15
Total:                /100 + bonus

Honest assessment of where I lost points:
[1-2 sentences]
```

The self-grade is not graded itself, but submissions that include an honest self-grade tend to come back from the C5 grader with very few surprises. Reviewers reward calibration.
