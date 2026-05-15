# Week 10 — Sequence Models: RNN, LSTM, GRU

> *Week 9 ended with a ResNet-18 reaching 92% on CIFAR-10 in fifteen minutes of fine-tuning. The model assumed an image — a fixed-size grid of pixels with a meaningful spatial neighborhood. This week the input is a sequence: a stream of tokens of unknown length, where the meaning of position 50 depends on positions 1 through 49. We build the three workhorse recurrent cells from scratch. The vanilla RNN (and the vanishing-gradient story that killed it for long sequences). The LSTM with its three gates (Hochreiter and Schmidhuber 1997 — still in production in 2026). The GRU (Cho et al. 2014 — a 2x lighter variant with similar quality on most tasks). Then we wire it all together into a char-level language model trained on a public-domain Project Gutenberg novel and watch it sample English-shaped text after twenty minutes of training. Lecture 3 ends with the tease that Karpathy's 2015 "Unreasonable Effectiveness" essay set up — and that Vaswani et al. 2017 closed with attention. Next week.*

Welcome to week ten of **C5 · Crunch AI / Data Science**. Week 9 closed the vision portion of the course: CNNs, transfer learning, ResNet, the 90%+ recipe on CIFAR-10. Week 10 swings hard the other way, to the second of the two great pre-transformer architectural families: **recurrent neural networks**. Where CNNs assume spatial locality, RNNs assume temporal order. Where CNNs share weights across positions in space, RNNs share weights across positions in time. The two architectures are mirror images of one another, and the inductive biases they bake in — locality, weight sharing — are the same two ideas in different clothes.

Three commitments before we start:

1. **The Week 8 training loop carries over verbatim, again.** The `for epoch: for batch: zero_grad / forward / loss / backward / step` pattern is the same. What changes is the model class (now contains `nn.LSTM` or `nn.GRU` rather than `nn.Conv2d`) and the data pipeline (now produces `(seq_len, batch)` or `(batch, seq_len)` integer tensors rather than `(batch, channels, height, width)` floats). The C5 conviction stands: write the same eight lines every week.
2. **We build the cells by hand before we touch `nn.LSTM`.** Exercise 1 has you implement a vanilla RNN cell in pure PyTorch from the equation `h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)` and verify against `nn.RNNCell`. Exercise 2 does the same for the LSTM cell with its four gates. Only Exercise 3 (and the mini-project) use the fast, cuDNN-backed `nn.LSTM` / `nn.GRU`. The order matters: students who skip the hand-rolled cell never internalize *why* the gates exist, and they cannot answer "what is the forget gate doing?" in interview.
3. **We train a char-level language model on a real public-domain text.** Project Gutenberg's `Pride and Prejudice` (Jane Austen, public domain, 700 KB) is the C5 default. The model is a 2-layer LSTM with 256 hidden units, trained for 20 epochs on truncated BPTT chunks of length 100. After training, we sample from the model at three temperatures (0.5, 0.8, 1.2) and watch English-shaped text fall out of a 1.5M-parameter network. Karpathy's 2015 "The Unreasonable Effectiveness of Recurrent Neural Networks" (<https://karpathy.github.io/2015/05/21/rnn-effectiveness/>) is the spiritual reference; we update the implementation to PyTorch 2.x but the architecture is unchanged.

We target **PyTorch 2.x** (we test on 2.4 and 2.5) and **Python 3.11+**. No torchvision this week (no images). No torchtext (the project is deprecated since 2024; we tokenize by hand at the character level). The Apple Silicon `mps` backend is supported by `nn.LSTM` and `nn.GRU` in PyTorch 2.4+; CUDA is what the C5 grading rig runs on. CPU-only is enough for every exercise, and the mini-project trains the language model in ~30 minutes on CPU, ~5 minutes on a free Google Colab T4.

PyTorch reference docs: <https://pytorch.org/docs/stable/index.html>. The three primary references for this week:

- `nn.RNN` — <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>
- `nn.LSTM` — <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>
- `nn.GRU` — <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>

Pin all three.

---

## Learning objectives

By the end of this week, you will be able to:

- **Describe the vanilla RNN equation.** Given the recurrence `h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)` and `y_t = W_hy @ h_t + b_y`, walk through one step on paper. Identify the three parameter tensors and their shapes. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>.
- **Explain the vanishing-gradient problem.** Argue why, in the recurrence above, the gradient of the loss at time `t` with respect to the hidden state at time `0` is a product of `t` Jacobians, and why for `tanh` activations and `||W_hh|| < 1` that product shrinks geometrically with `t`. Cite Hochreiter (1991) and Bengio, Simard, Frasconi (1994) as the foundational analyses, and Pascanu, Mikolov, Bengio (2013) — <https://arxiv.org/abs/1211.5063> — as the modern reframing.
- **State the LSTM equations from memory.** The forget gate `f_t = sigmoid(W_f @ [x_t, h_{t-1}] + b_f)`, the input gate `i_t`, the candidate `g_t = tanh(W_g @ ...)`, the output gate `o_t`, the cell state update `c_t = f_t * c_{t-1} + i_t * g_t`, the hidden state `h_t = o_t * tanh(c_t)`. Reference: Hochreiter and Schmidhuber 1997 — <https://www.bioinf.jku.at/publications/older/2604.pdf> — and the PyTorch `nn.LSTM` docs.
- **State the GRU equations from memory.** The reset gate, the update gate, the candidate, the convex combination `h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate`. Reference: Cho et al. 2014 — <https://arxiv.org/abs/1406.1078>.
- **Choose between RNN, LSTM, and GRU for a given task.** Vanilla RNN: short sequences only (under 20 timesteps), now mostly a teaching example. LSTM: the default for long sequences with strong precedent and large models. GRU: 25-30% fewer parameters than LSTM, trains faster, matches LSTM on most tasks under a few million examples. Reference: Chung et al. 2014 — <https://arxiv.org/abs/1412.3555> — the empirical comparison.
- **Batch variable-length sequences correctly.** Build a `collate_fn` that pads a list of variable-length tensors to a common length and produces a `lengths` tensor. Use `torch.nn.utils.rnn.pack_padded_sequence` to skip padding inside the LSTM, and `pad_packed_sequence` to recover a padded output tensor. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html>.
- **Apply teacher forcing during seq-to-seq training.** Feed the *true* previous token as input at each step during training; feed the model's own previous *prediction* at inference. Argue why this trick speeds up convergence and what its downside is (exposure bias). Reference: Williams and Zipser 1989 (the original "Profitable" paper, summarized in Goodfellow, Bengio, Courville, *Deep Learning*, chapter 10, free at <https://www.deeplearningbook.org/contents/rnn.html>).
- **Apply truncated BPTT.** When a sequence is longer than fits in memory, slice it into chunks of length `T`, run backward on each chunk separately, and pass the final hidden state forward (detached from the graph). Reference: Karpathy 2015 blog — <https://karpathy.github.io/2015/05/21/rnn-effectiveness/> — for the practitioner's framing.
- **Build a char-level language model on Project Gutenberg text.** Read a public-domain `.txt` file, build a char-to-index map, materialize integer-tensor chunks of length 100, train a 2-layer LSTM with `nn.Embedding` input and a `nn.Linear` projection to the vocab, optimize with `Adam(lr=1e-3)` and `nn.CrossEntropyLoss`. Sample from the trained model with temperature scaling.
- **Tease toward attention and transformers.** Lecture 3 closes by naming the failure mode that pure recurrence cannot escape — sequential compute, no constant-time lookback, no parallel training — and pointing at the result that fixed all three: Vaswani et al. 2017, "Attention Is All You Need" (<https://arxiv.org/abs/1706.03762>). Next week.
- **Ship** a mini-project: a char-level language-model training script, a sampling script, three sampled passages at three temperatures, and a 600-900 word report. Pushed to your portfolio repo.
- **Pass** every `pytest` case on the Week 10 exercises.

---

## Prerequisites

- **Week 9 complete.** The CNN exercises, the transfer-learning mini-project, the bare PyTorch loop. If you have not pushed the Week 9 CIFAR-10 classifier to your portfolio repo, finish that first; Week 10 builds on the same `nn.Module` reflexes.
- **Python 3.11+** with PyTorch 2.x installed (`pip install "torch>=2.4,<3"`). No torchvision this week unless you want to run the optional homework problem that revisits ResNet-18.
- **About 10 MB of disk** for the Project Gutenberg text cache. The mini-project downloads a single `.txt` file from Project Gutenberg's HTTPS mirror on first run; no special access required, the file is in the public domain in the United States.
- **Optional but useful:** a CUDA GPU, an Apple Silicon M1+, or a free Google Colab T4. The char-LM trains on CPU in ~30 minutes but is more fun to iterate on with a GPU.

You should already be comfortable with:

- **PyTorch tensor shapes for sequences.** `(seq_len, batch, features)` is the PyTorch default for `nn.RNN`/`nn.LSTM`/`nn.GRU`; `(batch, seq_len, features)` is enabled by `batch_first=True`. Both conventions appear in the wild; we use `batch_first=True` throughout the C5 lectures because it matches the rest of PyTorch.
- **`nn.Module` subclassing.** Week 8 Lecture 2.
- **The DataLoader pattern** including `collate_fn`. Week 8 Lecture 3 introduced `DataLoader`; this week we customize `collate_fn` to pad sequences.

---

## Topics covered

- **The sequence-modeling problem.** Given a sequence `x_1, x_2, ..., x_T` of variable length `T`, learn a mapping to either (a) a single output `y` (sequence classification: sentiment analysis), (b) a sequence of outputs of the same length (sequence labeling: part-of-speech tagging, named-entity recognition), or (c) a sequence of outputs of a different length (sequence-to-sequence: translation, summarization). Language modeling is the canonical instance of (b) shifted by one position — predict `x_{t+1}` given `x_1..x_t`. Reference: Goodfellow, Bengio, Courville chapter 10 (<https://www.deeplearningbook.org/contents/rnn.html>).
- **Why convolutions are not enough.** A `Conv1d` with kernel size `K` can see a `K`-position window; stacking `L` such layers extends the receptive field to roughly `L*K`. To see the entire history of a 500-token sequence with `K=3` you need ~167 layers, which is impractical. Recurrence solves this by carrying a hidden state forward in time — every output activation depends, in principle, on every earlier input. This is the architectural property that motivates RNNs and the one that attention reproduces (more efficiently) with a single softmax-weighted lookup.
- **The vanilla RNN cell.** `h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)`. Three parameter tensors: `W_xh` of shape `(hidden, input)`, `W_hh` of shape `(hidden, hidden)`, `b` of shape `(hidden,)`. The cell is a function from `(x_t, h_{t-1})` to `h_t`; unrolling it across `T` timesteps gives a feedforward computation graph that backprop handles natively. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>.
- **The vanishing-gradient problem.** During backpropagation through time, the gradient of the loss at step `t` with respect to a parameter at step `0` is a product of `t` Jacobian matrices. For `tanh` cells with weight matrices whose largest singular value is below 1, that product shrinks geometrically with `t`; for matrices whose largest singular value exceeds 1, the product *explodes*. Both failure modes were named in Hochreiter's 1991 thesis and rediscovered repeatedly throughout the 1990s. They are the reason the LSTM exists. Reference: Pascanu, Mikolov, Bengio 2013 — <https://arxiv.org/abs/1211.5063>.
- **The LSTM cell.** Four affine layers and three element-wise gates. The cell state `c_t` is a separate channel that *adds* the candidate `i_t * g_t` rather than multiplying — the additive update is the architectural trick that lets gradient flow back through hundreds of timesteps without vanishing. The 1997 paper is one of the most-cited papers in machine learning. Reference: Hochreiter and Schmidhuber 1997 — <https://www.bioinf.jku.at/publications/older/2604.pdf>.
- **The GRU cell.** Two gates (reset, update), a candidate, and a single hidden-state output (no separate cell state). The update gate is a convex combination: `h_t = (1 - z_t) * h_{t-1} + z_t * h_candidate`. Roughly 25-30% fewer parameters than an LSTM of the same hidden size, and on most tasks matches LSTM quality. Reference: Cho et al. 2014 — <https://arxiv.org/abs/1406.1078>.
- **The empirical LSTM-vs-GRU question.** Chung et al. 2014 (<https://arxiv.org/abs/1412.3555>) compared all three cells on polyphonic music modeling and speech-signal modeling; LSTM and GRU were statistically indistinguishable, both well above vanilla RNN. The C5 conviction in 2026: pick GRU for speed-constrained projects and LSTM for everything else. The architecture rarely matters as much as the data, the optimizer, and the regularization.
- **Batching variable-length sequences.** Pad with a sentinel value (typically 0) to the longest sequence in the batch. Use `pack_padded_sequence` to tell the LSTM where the real content ends and the padding starts, so it does not waste compute on padded positions and does not propagate gradient through them. Reference: <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html>.
- **Teacher forcing.** During training, at step `t` the model is fed the *true* token `x_t` rather than its own previous prediction `x_hat_{t-1}`. This decouples the per-step losses and lets gradient flow cleanly through the entire sequence. The downside is *exposure bias*: at inference the model has never seen its own mistakes as inputs, so a single bad prediction can cascade. The C5 lectures cover the standard mitigation, scheduled sampling (Bengio et al. 2015, <https://arxiv.org/abs/1506.03099>), in a single paragraph; the production answer is "switch to a transformer."
- **Truncated BPTT.** Split a long sequence into chunks of length `T_chunk` (the C5 default is 100 for char-LM, 50 for word-LM). Run forward + backward on each chunk independently. Pass the final hidden state from chunk `k` as the initial hidden state of chunk `k+1`, but `.detach()` it so the gradient does not flow back into the previous chunk. The net effect: training is `O(T_chunk)` in memory rather than `O(full sequence length)`, at the cost of capping the receptive field of the gradient to `T_chunk`. The trade-off is usually worth it.
- **Char-level language modeling.** Tokens are individual UTF-8 characters; the vocabulary is ~80-150 unique characters for English text. The model is small (1-2M parameters), trains in minutes, and produces qualitatively interesting samples even on a single CPU. The reason char-LM is the C5 default rather than word-LM: no tokenizer to install, no vocabulary file to keep in sync, no out-of-vocabulary handling — the entire pipeline is in a single Python file. Reference: Karpathy 2015 — <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>.
- **Sampling with temperature.** At generation time, the model outputs a logit vector over the vocabulary; we apply `softmax(logits / T)` for some temperature `T > 0` and sample from the resulting distribution. Lower `T` makes the sampling more deterministic (T → 0 is argmax); higher `T` flattens the distribution (T → ∞ is uniform). The C5 default sweep is T ∈ {0.5, 0.8, 1.2}.
- **The tease toward attention.** Three things RNNs cannot do that attention can. (a) **Constant-time lookback:** an LSTM accessing information from 500 steps ago must propagate that information through 500 hidden-state updates; an attention layer can read it in one operation. (b) **Parallel training:** an LSTM unrolls sequentially across the sequence length; an attention layer's forward pass parallelizes over the entire sequence. (c) **No gradient-vanishing through depth-in-time:** attention's gradient path from output to input is `O(1)`, not `O(T)`. Lecture 3 names these three properties and ends with the citation: Vaswani et al. 2017 (<https://arxiv.org/abs/1706.03762>). Next week.
- **What we do not do.** No bidirectional sequence-to-sequence models (the encoder-decoder architecture is a Week 12 topic). No attention mechanisms (next week). No transformers (next week). No BERT, no GPT, no T5 — those are language-model-architecture concerns, not sequence-modeling fundamentals. No torchtext (deprecated since 2024). No `torch.compile()` on the recurrent cells (the compile speedup on `nn.LSTM` was small as of PyTorch 2.5; revisit in Week 13).

---

## Weekly schedule

Target about **38 hours**. Monday and Tuesday cover the vanilla RNN and the vanishing-gradient story. Wednesday is LSTM and GRU. Thursday is sequence batching, teacher forcing, and truncated BPTT. Friday-Sunday is the char-LM mini-project.

| Day       | Focus                                                            | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Vanilla RNN; the recurrence; vanishing-gradient analysis         |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | LSTM gates; GRU as lighter variant; the 1997 and 2014 papers     |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Wednesday | Sequence batching, packing, teacher forcing, truncated BPTT      |   2h     |   2h      |     2h     |   0.5h    |   1h     |     0h       |   0h       |    7.5h     |
| Thursday  | Char-LM architecture; embedding, projection, sampling            |   2h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0h       |    7.5h     |
| Friday    | Mini-project: train the LSTM on Pride and Prejudice              |   0h     |   0.5h    |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    5.5h     |
| Saturday  | Mini-project: sample at three temperatures, write report         |   0h     |   0h      |     0h     |   0h      |   1h     |     2h       |   0h       |    3h       |
| Sunday    | Quiz, push to portfolio repo                                     |   0h     |   0h      |     0h     |   0.5h    |   0h     |     0h       |   0h       |   0.5h      |
| **Total** |                                                                  | **10h**  | **7.5h**  | **2h**     | **3h**    | **6h**   | **8h**       | **1h**     |  **38h**    |

The schedule is generous on Monday because the vanilla-RNN equation is short and the vanishing-gradient analysis is the hardest piece of math in the week (one paragraph of singular-value reasoning). Wednesday's `pack_padded_sequence` exercise is the most-fiddly piece of PyTorch code in the C5 curriculum so far; budget the full two hours and read the docs twice.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | PyTorch `nn.RNN`/`nn.LSTM`/`nn.GRU` docs, the 1997 LSTM paper, the 2014 GRU paper, the Karpathy 2015 essay, Goodfellow chapter 10, the Pascanu et al. exploding-gradient paper |
| [lecture-notes/01-rnn-and-vanishing-gradients.md](./lecture-notes/01-rnn-and-vanishing-gradients.md) | The sequence-modeling problem, the vanilla RNN cell, the BPTT derivation, the vanishing- and exploding-gradient analyses, gradient clipping |
| [lecture-notes/02-lstm-and-gru.md](./lecture-notes/02-lstm-and-gru.md) | The LSTM gates (forget, input, output) and the cell state, the GRU (reset, update) and its convex combination, the LSTM-vs-GRU empirical comparison, why these cells solve the vanishing-gradient problem |
| [lecture-notes/03-sequence-batching-and-char-lm.md](./lecture-notes/03-sequence-batching-and-char-lm.md) | `pack_padded_sequence`, teacher forcing, truncated BPTT, building a char-level language model end-to-end, sampling with temperature, the tease toward attention |
| [exercises/exercise-01-rnn-cell-from-scratch.py](./exercises/exercise-01-rnn-cell-from-scratch.py) | Implement a vanilla RNN cell from the recurrence equation in pure PyTorch; verify against `nn.RNNCell` |
| [exercises/exercise-02-lstm-cell-from-scratch.py](./exercises/exercise-02-lstm-cell-from-scratch.py) | Implement an LSTM cell from the four-gate equations; verify against `nn.LSTMCell` |
| [exercises/exercise-03-pack-padded-sequence.py](./exercises/exercise-03-pack-padded-sequence.py) | Build a `collate_fn` and use `pack_padded_sequence` correctly; verify the LSTM's output matches a manually-loop-implementation on padded data |
| [exercises/SOLUTIONS.md](./exercises/SOLUTIONS.md) | Reference solutions with commentary; read only after attempting |
| [challenges/challenge-01-gradient-norms-by-depth.md](./challenges/challenge-01-gradient-norms-by-depth.md) | Measure the gradient norm at every timestep of a 200-step RNN unroll; reproduce the vanishing-gradient curve from Pascanu et al. 2013 |
| [challenges/challenge-02-lstm-vs-gru-bake-off.md](./challenges/challenge-02-lstm-vs-gru-bake-off.md) | Train an LSTM and a GRU of matched parameter counts on the same char-LM task; compare wall-clock, validation loss, and sample quality |
| [quiz.md](./quiz.md) | 10 multiple-choice questions on recurrence, vanishing gradients, the LSTM gates, GRU equations, and sequence batching |
| [homework.md](./homework.md) | Six problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Build a char-level language model in PyTorch and train it on a Project Gutenberg public-domain text |
| [mini-project/starter.py](./mini-project/starter.py) | Runnable starter: text download, char vocab, chunked dataset, 2-layer LSTM, training loop, temperature sampler |
| [mini-project/rubric.md](./mini-project/rubric.md) | Grading rubric for the mini-project |

---

## Stretch goals

- Read **Karpathy 2015 — "The Unreasonable Effectiveness of Recurrent Neural Networks"** (<https://karpathy.github.io/2015/05/21/rnn-effectiveness/>). The single best practitioner's introduction to char-level language modeling ever written. The figures of Shakespeare-shaped text and Linux-kernel-shaped C code samples are the artifacts of the original `char-rnn` Torch (not PyTorch) implementation; the C5 mini-project reproduces the recipe in modern PyTorch.
- Read the **LSTM paper** (Hochreiter and Schmidhuber 1997; <https://www.bioinf.jku.at/publications/older/2604.pdf>). Thirty-two pages; the prose is denser than a modern arXiv paper but the introduction (sections 1-3) and the LSTM definition (section 4) are the conceptual core. The paper sat near-uncited for fifteen years before deep learning picked it up around 2013-2014; it is now in the top-100 most-cited ML papers ever written.
- Read the **GRU paper** (Cho et al. 2014; <https://arxiv.org/abs/1406.1078>). Nine pages, primarily about machine translation, with the GRU defined in section 2.3 almost as an afterthought. Read for the architectural definition; skip the encoder-decoder material (it returns in Week 12).
- Read **Goodfellow, Bengio, Courville chapter 10** (<https://www.deeplearningbook.org/contents/rnn.html>). The chapter is free, online, and exhaustive; sections 10.1-10.4 cover everything in this week's lectures, plus deep RNNs, recursive nets, and echo-state networks (the latter two are out of C5 scope but historically important).
- Read **Pascanu, Mikolov, Bengio 2013 — "On the difficulty of training Recurrent Neural Networks"** (<https://arxiv.org/abs/1211.5063>). Twelve pages. The modern reframing of the vanishing/exploding-gradient analysis, plus the gradient-clipping prescription that every RNN training script now uses by default.
- Read **Chung et al. 2014 — "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"** (<https://arxiv.org/abs/1412.3555>). Nine pages. The paper that established "LSTM and GRU are roughly equivalent, both beat vanilla RNN." Read for the experimental protocol; the result is widely cited but the methodology is the durable contribution.
- Read **Olah 2015 — "Understanding LSTM Networks"** (<https://colah.github.io/posts/2015-08-Understanding-LSTMs/>). The diagrams of the LSTM gates that show up in every textbook published since 2016 originated here. The blog post is free and remains the cleanest pictorial explanation on the internet.

---

## What you will *not* do this week

You will not:

- **Implement an encoder-decoder for translation.** Week 12. This week the model is a single recurrent stack with a per-step classification head.
- **Use attention.** Next week. The whole point of Week 11 is to motivate attention from the failure modes of pure recurrence; introducing it here would skip the motivation.
- **Use a pretrained language model.** No GPT-2, no BERT, no T5. Those are Week 13 and beyond.
- **Use torchtext, spacy, or HuggingFace tokenizers.** Char-level tokenization is `set(text)` followed by `{c: i for i, c in enumerate(sorted(chars))}`. Four lines. No installs.
- **Tune hyperparameters extensively.** The mini-project ships a single working recipe; the homework explores one-axis sensitivity (hidden size, dropout) but does not run a full grid. Hyperparameter search is a Week 14 topic.
- **Build a word-level LM.** Char-level is the C5 default because it sidesteps the tokenizer question entirely. Word-level LMs are mentioned in the resources but are not on the exercise track.

That list is deliberate. The point of Week 10 is to *understand the inductive biases that make recurrent networks work* and *to feel the friction of sequential compute* — the friction that motivates the architectural pivot to attention next week. SOTA is not the goal; the conceptual scaffolding is.

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The Lecture 1 experiment that unrolls a randomly-initialized vanilla RNN for 200 steps and plots the norm of `||h_t||` is the experiment that makes "vanishing-gradient" stop being a slogan — you watch the norm decay geometrically toward zero on your own screen. The Lecture 2 experiment that prints the LSTM forget-gate activations on the first few characters of a sentence is the experiment that makes "the gates are real" stop feeling like a textbook claim. Both take less than ten minutes; both are the difference between "I read about it" and "I have done it."

---

## Up next

[Week 11 — Attention and the Transformer architecture](../week-11/) — once you have pushed the Week 10 char-LM, the LSTM-vs-GRU comparison, and the sampled-text passages to your portfolio repo. Week 11 is the architectural pivot the entire field made in 2017: Vaswani et al., "Attention Is All You Need" (<https://arxiv.org/abs/1706.03762>). The recurrence is replaced by self-attention; the sequential compute is replaced by parallel matmuls; the vanishing-gradient analysis stops mattering because the gradient path from output to input is constant. The model you ship in Week 11 will reuse the char-LM data pipeline from this week unchanged; only the model class changes. That is the cleanest demonstration of architectural progress in the entire C5 curriculum, and Lecture 3 of this week ends pointing straight at it.
