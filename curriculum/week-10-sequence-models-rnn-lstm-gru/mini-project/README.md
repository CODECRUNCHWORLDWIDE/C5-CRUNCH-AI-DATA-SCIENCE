# Mini-Project — A Char-Level Language Model in PyTorch

> Build a character-level language model on a Project Gutenberg public-domain text (the C5 default is `Pride and Prejudice`). Train a 2-layer LSTM with truncated BPTT for 20 epochs, reaching a held-out cross-entropy loss of roughly 1.4 nats per character. Sample 200 characters at three temperatures (`0.5, 0.8, 1.2`) and ship the three passages in a 600-900 word report that an engineering manager could read in five minutes and conclude "this person has internalized how a language model works."

This is the tenth artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 8 was the bare PyTorch training loop. Week 9 was vision: a CNN, then a transfer-learning ResNet. Week 10 is sequences: an LSTM-based character-level language model that learns the statistics of a public-domain novel and generates new English-shaped text. The visceral payoff is the same as the original Karpathy 2015 char-RNN demo (<https://karpathy.github.io/2015/05/21/rnn-effectiveness/>): you watch the model's samples progress from uniform random characters to English-frequency characters to word-shaped clumps to grammatical sentences over twenty epochs of training. The progression is the headline artifact of the project.

**Estimated time:** 8 hours, spread across Thursday-Sunday.

---

## What you will build

Two scripts, one notebook, one report:

1. **`train.py`** — downloads the corpus on first run, builds the char vocabulary, instantiates `CharLSTM`, trains for 20 epochs with truncated BPTT, prints per-epoch validation loss and a 200-char sample at `T = 0.8`. Saves `model.pt`. Expected final validation loss: 1.35-1.50 nats/char.
2. **`sample.py`** — loads `model.pt`, takes a prompt and a temperature as command-line arguments, samples 200 characters, prints them. Used to produce the three temperature passages for the report.
3. **`exploration.ipynb`** — loads the trained model, plots the training and validation loss curves, shows three sampled passages at `T ∈ {0.5, 0.8, 1.2}`, and inspects the LSTM's forget-gate activations on the first sentence of the corpus (a small "what is the model paying attention to" plot). The notebook is for the human reader; the scripts are for the machine reader.
4. **`report.md`** — a 600-900 word executive summary. The headline is the progression of sample quality across epochs (with quotes at epochs 1, 5, 10, 20); the narrative is "we measured the LSTM's char-LM ability on a real corpus; here is what worked, what did not, and what we would change."

---

## The dataset

The C5 default is **`Pride and Prejudice` by Jane Austen** (first published 1813, public domain everywhere we care about). Plain-text URL: <https://www.gutenberg.org/files/1342/1342-0.txt>. Size: about 700 KB after stripping the Project Gutenberg license preamble and afterword.

Approximately 700,000 characters, vocabulary of around 80 unique characters (uppercase + lowercase letters + digits + punctuation + whitespace). The starter script downloads, strips, and caches the text on first run; subsequent runs read from the local cache.

Alternates that work equally well for char-LM, all on Project Gutenberg:
- **`Frankenstein`** (Mary Shelley, 1818). <https://www.gutenberg.org/files/84/84-0.txt>. About 450 KB.
- **`Alice's Adventures in Wonderland`** (Lewis Carroll, 1865). <https://www.gutenberg.org/files/11/11-0.txt>. About 170 KB. Small corpus, fast iteration; the model overfits quickly.
- **`Moby-Dick; or, The Whale`** (Herman Melville, 1851). <https://www.gutenberg.org/files/2701/2701-0.txt>. About 1.2 MB. Larger corpus, more diverse vocabulary.

Pick `Pride and Prejudice` unless you have a reason to deviate. The rubric does not penalize using one of the alternates; it does ask that the report state which text you used and why.

---

## Acceptance criteria

- [ ] A new directory `week-10/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `torch>=2.4,<3`, `numpy>=2,<3`, `matplotlib>=3.8,<4`, `jupyter`.
- [ ] **`python train.py`** runs end-to-end, downloads the corpus on first run (Pride and Prejudice from Project Gutenberg's HTTPS URL), trains for ≥20 epochs, prints per-epoch validation loss, and produces `model.pt` and `training_curves.png`.
- [ ] **`python sample.py --prompt "It is a truth universally acknowledged, " --temperature 0.8 --n-chars 200`** loads the model, samples, and prints 200 characters of English-shaped text. Run with `--temperature 0.5`, `--temperature 1.2` to produce the other two samples for the report.
- [ ] Both scripts use **`torch.manual_seed(42)`** at the top. The same run twice produces the same final validation loss.
- [ ] The `CharLSTM` is defined as an `nn.Module` subclass with `nn.Embedding`, `nn.LSTM`, and `nn.Linear` head, exactly as in Lecture 2 Section 9. The class lives in `model.py`; both `train.py` and `sample.py` import from there.
- [ ] The training loop uses **truncated BPTT** with chunk length 100 and parallel-stream batching (the recipe from Lecture 3 Section 4).
- [ ] Gradient clipping at `max_norm=1.0` is in the training loop.
- [ ] Final validation loss is **≤ 1.55 nats per character** on the held-out 10% of the corpus. (Strong models reach 1.35; the upper bound here is generous to allow for hyperparameter variation.)
- [ ] `python -m py_compile *.py` succeeds for every Python file in `week-10/`.
- [ ] `report.md` is 600-900 words. No more, no less.

---

## File layout

```text
week-10/
├── requirements.txt
├── model.py             # CharLSTM class
├── data.py              # download_corpus, build_vocab, encode, chunked_streams
├── train.py             # the 20-epoch training script
├── sample.py            # the CLI sampling script
├── exploration.ipynb    # the notebook with curves and samples
├── report.md            # the 600-900 word executive summary
├── model.pt             # the trained state_dict (~6 MB)
├── training_curves.png  # train + val loss curves
└── README.md            # this file (a slight rewrite of the C5 mini-project README)
```

The starter file at `mini-project/starter.py` in the C5 curriculum is a single-file scaffold; you will refactor it into `model.py`, `data.py`, `train.py`, and `sample.py` as you build out the project. The single-file starter is the working version of the recipe; the modular layout is the deliverable.

---

## The recipe at a glance

The full pipeline, condensed:

1. **Download corpus.** Fetch `Pride and Prejudice` from Project Gutenberg; strip the license preamble and afterword. Cache locally.
2. **Build vocab.** `vocab = sorted(set(text))`. Build `char_to_idx` and `idx_to_char` dictionaries.
3. **Encode.** Map the entire corpus to a single 1-D int64 tensor of length ~700k.
4. **Split.** First 90% as train, last 10% as validation.
5. **Reshape into parallel streams.** Reshape the train tensor to `(batch_size, -1)` so each batch slot is a contiguous slice through the corpus. The mini-project default: `batch_size = 32`.
6. **Iterate over chunks.** For each chunk of length `T_chunk = 100`, slice `(batch_size, T_chunk)` from the reshaped tensor as the input, and the chunk shifted by one position as the target.
7. **Forward, loss, backward, clip, step.** The standard eight-line loop. The model's hidden state is passed forward across chunks within an epoch, detached at the chunk boundary (truncated BPTT).
8. **Reset hidden state at epoch boundaries.** At the start of each epoch.
9. **Per-epoch validation.** After each epoch, run the model on the validation tensor (no gradient, no chunking — just one pass) and report the mean cross-entropy loss.
10. **Sample.** After training, sample 200 chars at each of `T ∈ {0.5, 0.8, 1.2}`.

The expected per-epoch wall-clock on a 2.5 GHz CPU is about 90 seconds; on a Colab T4 about 15 seconds. Total training time: 30 minutes on CPU, 5 minutes on T4.

---

## The model

```python
class CharLSTM(nn.Module):
    """A 2-layer LSTM char-level language model.

    Architecture: Embedding -> LSTM -> Linear over vocab.
    The Linear head is applied at every timestep (sequence labeling at the
    character level), and the loss is cross-entropy summed across positions.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        state: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]": ...
```

This is the same shape as the `CharLSTM` from Lecture 2 Section 9. Total parameter count: roughly 1.5M for the default settings.

---

## Reproducibility

- `torch.manual_seed(42)` at the top of `train.py`.
- `torch.cuda.manual_seed(42)` and `torch.backends.cudnn.deterministic = True` if you are training on GPU and want bit-exact reproducibility (this costs a small wall-clock penalty; the rubric does not require it).
- Pin the corpus URL and the byte offsets of the preamble/afterword strip — the Project Gutenberg files do not have a stable format and a sloppy strip can leak licensing text into the training data (which the model will dutifully memorize and emit at low temperatures, which is a legitimate but distracting source of "bad samples").
- The starter script does this strip via a fixed text marker ("*** START OF THE PROJECT GUTENBERG EBOOK ..." and "*** END OF THE PROJECT GUTENBERG EBOOK ...") that Project Gutenberg has used consistently since around 2010.

---

## The report

`report.md` is a 600-900 word executive summary. Structure:

1. **Headline (1 short paragraph).** "We trained a 2-layer LSTM character-level language model on `Pride and Prejudice`, reaching a held-out cross-entropy of `X` nats per character. Samples at `T = 0.8` are below."
2. **Method (2-3 paragraphs).** The architecture (briefly). The truncated-BPTT recipe. The hyperparameters. The training and validation split. Cite Karpathy 2015 and the LSTM paper.
3. **Results (2-3 paragraphs).** The training and validation loss curves (embed `training_curves.png`). The progression of sample quality across epochs (quote a passage at epoch 1, 5, 10, 20). The three temperature passages at the final checkpoint, each labeled.
4. **Discussion (2 paragraphs).** What worked. What surprised you. Honest about what the model cannot do (long-range coherence, plot, character consistency). The "tease toward attention" — name the three failure modes that motivate the Week 11 transformer.
5. **What we would do next (1 short paragraph).** Concrete next step: train a transformer on the same corpus and compare.

The rubric weights the discussion and "what we would do next" sections heavily. The numbers are a hygiene check; the writing is the deliverable.

---

## Stretch achievements (optional)

- **Train on a 1+ MB corpus.** `Moby-Dick`, or concatenate two Gutenberg texts. Report the new validation loss; expect 1.2-1.4 nats/char on a larger corpus.
- **Compare LSTM vs. GRU.** Train a parameter-matched GRU and report the head-to-head. (This is Challenge 2; if you have done that challenge, fold its results into the mini-project report and cite the challenge.)
- **Implement temperature sampling with top-k filtering.** Instead of sampling from the full softmax, restrict to the top `k` most likely characters. Reference: Fan, Lewis, Dauphin 2018, "Hierarchical Neural Story Generation" (<https://arxiv.org/abs/1805.04833>). At `k = 10` the samples are noticeably more coherent at higher temperatures.
- **Try a different prompt.** The default prompt is the opening of Pride and Prejudice. Try `"Mr. Darcy walked into the room and "`, `"In a hole in the ground there lived "`, or your own. Report whether the model is recognizably "in the corpus distribution" given a prompt that is not literally in the training text.

---

## Common gotchas

1. **The Gutenberg preamble.** If you don't strip it, the model will memorize and emit "PROJECT GUTENBERG" at low temperatures. This is a "bad sample" that is technically the model doing exactly what you asked of it; it just exposes a data-cleaning gap. The starter strips on the standard text markers; verify your stripped corpus starts at the novel's title page or chapter 1.
2. **The chunk-batch reshape.** This is the only non-trivial piece of tensor manipulation in the project. Walk through it on paper if you are confused: `train_tensor` has shape `(N,)` where `N` is the total training-corpus length; we want to reshape it to `(batch_size, n_chunks_per_stream * T_chunk)` so each row is an independent stream through a contiguous prefix of the corpus, then iterate `n_chunks_per_stream` times to consume the entire corpus once per epoch.
3. **Forgetting to detach the hidden state across chunks.** The classic "out of memory" failure mode on long training. The fix is one line: `hidden = (hidden[0].detach(), hidden[1].detach())` at the top of each chunk iteration.
4. **Confusing `(seq_len, batch)` with `(batch, seq_len)`.** Pass `batch_first=True` to `nn.LSTM` and stay in `(batch, seq_len, features)` throughout. The Karpathy 2015 reference implementation uses the opposite convention; do not be confused.
5. **The loss target.** The target for the language-modeling task is the input shifted by one position. If your input is `text[i:i+T]`, your target is `text[i+1:i+T+1]`. Off-by-one bugs here are common and produce a loss that mysteriously plateaus around `log(vocab_size) ≈ 4.4` — the random-guess baseline.

---

## References (the four big ones; everything else in `resources.md`)

- **Karpathy 2015**, "The Unreasonable Effectiveness of Recurrent Neural Networks": <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>. The spiritual reference.
- **`nn.LSTM` docs**: <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>.
- **Hochreiter & Schmidhuber 1997**, the LSTM paper: <https://www.bioinf.jku.at/publications/older/2604.pdf>.
- **`torch.multinomial`**: <https://pytorch.org/docs/stable/generated/torch.multinomial.html>. The sampler.
