# Week 10 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs, no signup-required courses. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **Karpathy, Andrej — "The Unreasonable Effectiveness of Recurrent Neural Networks"** (the 2015 essay that made char-level language modeling famous; the inspiration for this week's mini-project):
  <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>
  Read **Sunday evening** before Monday's lecture. Twenty minutes. The Shakespeare and Linux-kernel samples are the cultural reference; the C5 mini-project reproduces the recipe in modern PyTorch on Project Gutenberg's `Pride and Prejudice`.
- **Olah, Christopher — "Understanding LSTM Networks"** (the 2015 blog post that became the canonical pictorial explanation of LSTM gates):
  <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
  Read **Tuesday morning** before Lecture 2. Fifteen minutes. The hand-drawn gate diagrams have been reused in every deep-learning textbook published since 2016; the original is still the cleanest version.
- **PyTorch docs — `nn.RNN`** (the multilayer wrapper around `nn.RNNCell`):
  <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>
  Pin the tab. The "Inputs" / "Outputs" / "Variables" tables are the reference for every shape question this week.
- **PyTorch docs — `nn.LSTM`** (the multilayer LSTM; the most-used recurrent layer in production PyTorch):
  <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>
  Pin it. The "Inputs" section documents the `(h_0, c_0)` initial-state tuple convention — note that LSTMs have two state tensors per layer where RNN and GRU have one. This is the single most common source of confusion this week.
- **PyTorch docs — `nn.GRU`** (the multilayer GRU; lighter and often as good as LSTM):
  <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>
  Pin it. Note that GRU has only a hidden state (no cell state); the API mirrors `nn.RNN`, not `nn.LSTM`.
- **PyTorch docs — `torch.nn.utils.rnn.pack_padded_sequence`** (the function that makes variable-length sequence batching efficient):
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html>
  Read after Lecture 3 Section 4. The companion is `pad_packed_sequence` (<https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html>); you will use both in Exercise 3.
- **Goodfellow, Bengio, Courville — *Deep Learning*, chapter 10 ("Sequence Modeling: Recurrent and Recursive Nets")** (the free online textbook):
  <https://www.deeplearningbook.org/contents/rnn.html>
  Sections 10.1-10.4 cover the vanilla RNN, BPTT, the vanishing-gradient problem, and the LSTM. Section 10.10 covers the GRU and other gated variants. The chapter is forty-five pages of moderately-formal prose; you do not have to read all of it, but sections 10.1, 10.2, 10.7, and 10.10 are the conceptual core of this week.

## The papers (free PDFs)

- **Hochreiter, Sepp; Schmidhuber, Jürgen — "Long Short-Term Memory"** (Neural Computation 1997; the LSTM paper):
  <https://www.bioinf.jku.at/publications/older/2604.pdf>
  Thirty-two pages. The single most-cited paper in sequence modeling. The introduction (sections 1-3) is the conceptual setup; section 4 defines the LSTM cell with its three gates (the 1997 version had only an input gate and an output gate — the forget gate was added by Gers, Schmidhuber, Cummins in 2000, and the C5 lectures use the modern three-gate definition). Read sections 1, 2, 3, and 4 — about twelve pages total. The full appendix on the truncated-error gradient is optional but is the rigorous version of the "constant error carousel" intuition.
- **Cho, Kyunghyun; van Merriënboer, Bart; Gulcehre, Caglar; Bahdanau, Dzmitry; Bougares, Fethi; Schwenk, Holger; Bengio, Yoshua — "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"** (EMNLP 2014; the GRU paper):
  <https://arxiv.org/abs/1406.1078>
  Nine pages. The paper is primarily about a phrase-based machine-translation pipeline; the GRU is defined in section 2.3 in three equations as a "novel hidden unit." The C5 lectures cite the GRU definition; the encoder-decoder material returns in Week 12.
- **Pascanu, Razvan; Mikolov, Tomas; Bengio, Yoshua — "On the difficulty of training Recurrent Neural Networks"** (ICML 2013):
  <https://arxiv.org/abs/1211.5063>
  Twelve pages. The modern reframing of the vanishing- and exploding-gradient analyses (the originals are Hochreiter's 1991 thesis and Bengio, Simard, Frasconi 1994, both German-language or paywalled). Section 2 has the singular-value derivation that motivates gradient clipping; section 3 introduces the clipping algorithm itself. Every modern RNN training script clips gradients to a norm of 1-5; this paper is why.
- **Chung, Junyoung; Gulcehre, Caglar; Cho, KyungHyun; Bengio, Yoshua — "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"** (NIPS 2014 deep-learning workshop):
  <https://arxiv.org/abs/1412.3555>
  Nine pages. The first head-to-head comparison of vanilla RNN, LSTM, and GRU on polyphonic music modeling and speech-signal modeling. Result: LSTM and GRU are statistically indistinguishable; both clobber vanilla RNN. The C5 conviction "pick GRU for speed-constrained projects, LSTM for everything else" traces to this paper.
- **Bengio, Yoshua; Vinyals, Oriol; Jaitly, Navdeep; Shazeer, Noam — "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"** (NIPS 2015; the exposure-bias mitigation):
  <https://arxiv.org/abs/1506.03099>
  Six pages. The C5 lectures mention scheduled sampling in a single paragraph as the standard mitigation for teacher-forcing's exposure-bias problem; this paper is the formal statement. Read it if your mini-project samples cascade into nonsense.
- **Vaswani, Ashish; et al. — "Attention Is All You Need"** (NeurIPS 2017; the transformer paper):
  <https://arxiv.org/abs/1706.03762>
  Fifteen pages. **Do not read this paper yet — it is the Week 11 reading.** Listed here so the citation is in one place; Lecture 3 of Week 10 ends pointing at it as the answer to the failure modes of pure recurrence.

## The official PyTorch 2.x docs you will live in this week

- **`torch.nn.RNN`** (the multilayer wrapper around the vanilla RNN cell):
  <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>
- **`torch.nn.RNNCell`** (the single-step vanilla RNN cell; useful for custom unrolls):
  <https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html>
- **`torch.nn.LSTM`** (the multilayer LSTM):
  <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>
- **`torch.nn.LSTMCell`** (the single-step LSTM cell):
  <https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html>
- **`torch.nn.GRU`** (the multilayer GRU):
  <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>
- **`torch.nn.GRUCell`** (the single-step GRU cell):
  <https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html>
- **`torch.nn.Embedding`** (the lookup table that maps integer token IDs to dense vectors; the input to every NLP model):
  <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>
- **`torch.nn.utils.rnn.pack_padded_sequence`** (packs a padded batch so RNNs skip the padding):
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html>
- **`torch.nn.utils.rnn.pad_packed_sequence`** (unpacks back to a padded tensor):
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html>
- **`torch.nn.utils.rnn.pad_sequence`** (pads a list of variable-length tensors to a common length):
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html>
- **`torch.nn.utils.clip_grad_norm_`** (clips the global gradient norm; the standard RNN training trick):
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>
- **`torch.nn.CrossEntropyLoss`** (the loss for language modeling; applies log-softmax and NLL together):
  <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
- **`torch.multinomial`** (the sampler used in temperature-scaled generation):
  <https://pytorch.org/docs/stable/generated/torch.multinomial.html>

## Companion blog posts and tutorials (free)

- **PyTorch official tutorial — "NLP From Scratch: Classifying Names with a Character-Level RNN"** (the canonical PyTorch RNN tutorial; uses `nn.RNN` to classify last names by language):
  <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html>
  Read after Exercise 1. The tutorial covers the same vanilla-RNN cell the exercise asks you to implement, applied to a different task.
- **PyTorch official tutorial — "NLP From Scratch: Generating Names with a Character-Level RNN"** (the inverse of the previous tutorial; generates names from a starting letter using an unrolled RNN):
  <https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html>
  Read after Lecture 3. The mini-project is a scaled-up version of this tutorial (more data, LSTM instead of RNN, temperature sampling).
- **PyTorch official tutorial — "Sequence Models and Long Short-Term Memory Networks"** (the canonical `nn.LSTM` tutorial; uses an LSTM for part-of-speech tagging on a toy dataset):
  <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html>
  Read after Lecture 2. The exercise hands you an analogous but slightly larger task; this tutorial is the warm-up.
- **Olah, Christopher — "Attention and Augmented Recurrent Neural Networks"** (Distill 2016; the post that previewed the architectural pivot to attention):
  <https://distill.pub/2016/augmented-rnns/>
  Optional; read it Sunday after the quiz as a bridge to Week 11. Free, interactive, beautiful.
- **Goldberg, Yoav — "A Primer on Neural Network Models for Natural Language Processing"** (a 75-page free survey from 2015; the standard pre-transformer NLP reference):
  <https://arxiv.org/abs/1510.00726>
  Sections 7 and 10 cover recurrent networks specifically. Optional but thorough. The survey predates transformers by two years, so the architectures it covers are exactly the Week 10 architectures.

## Project Gutenberg

The mini-project trains on a public-domain `.txt` file from Project Gutenberg, the volunteer-run free-ebook library founded in 1971 (the first internet text archive, in fact). Three notes:

- **Public domain in the United States.** Project Gutenberg texts are public-domain in the US; some are still under copyright in other jurisdictions. The mini-project's default text — `Pride and Prejudice` by Jane Austen, first published 1813 — is public-domain everywhere we care about.
- **Catalog and download:** <https://www.gutenberg.org/>. The `Pride and Prejudice` plain-text URL is <https://www.gutenberg.org/files/1342/1342-0.txt>. Alternates that work well for char-LM: `Frankenstein` (Mary Shelley, <https://www.gutenberg.org/files/84/84-0.txt>), `Alice's Adventures in Wonderland` (Lewis Carroll, <https://www.gutenberg.org/files/11/11-0.txt>), `Moby-Dick` (Herman Melville, <https://www.gutenberg.org/files/2701/2701-0.txt> — about 1.2 MB, larger and richer).
- **Robots note.** Project Gutenberg's `robots.txt` does not block automated downloads of individual text files, but you should cache the download locally on first run (the mini-project starter does this) and not refetch on every script invocation. Be a good citizen.

## Datasets you might use beyond the mini-project

- **WikiText-2** (a 2 million-word text corpus, character-level or word-level, free):
  <https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/>
  The 2017 Salesforce dataset. Used by every word-LM paper from 2017 to ~2020 as a benchmark. About 12 MB.
- **Penn Treebank (PTB)** (the classic word-LM benchmark; preprocessed text-only version is 5 MB):
  <https://github.com/tensorflow/models/tree/master/research/lm_1b>
  The PTB was the standard LM benchmark from ~2000 to ~2015; results on it appear in every RNN paper from that era. The full PTB has license restrictions but the preprocessed text-only version distributed with research codebases is widely circulated and treated as effectively public for academic use.
- **enwik8** (the first 100 million bytes of Wikipedia; the canonical char-LM benchmark):
  <http://prize.hutter1.net/>
  ~100 MB. Used by the "Mixer of Experts" and "Sparse Transformer" papers; out of scope for a single-laptop Week 10 mini-project, but the right next dataset if you want to push char-LM further.

## Voices worth following on this material (free; no signups)

- **Andrej Karpathy** — his 2015 char-RNN blog post is the inspiration for this week's mini-project. His more recent 2023 video series "Neural Networks: Zero to Hero" includes a from-scratch RNN, LSTM, and GRU implementation in pure Python that is the cleanest pedagogical version on the internet. Free on YouTube: <https://karpathy.ai/zero-to-hero.html>.
- **Christopher Olah** — the 2015 LSTM blog and the 2016 attention-and-augmented-RNN essay (both linked above). His blog at <https://colah.github.io/> is the gold standard for visual explanations of neural network internals.
- **Sebastian Raschka** — his Substack <https://magazine.sebastianraschka.com/> regularly covers recurrent-network internals and the transition to transformers; the historical pieces on "how we got here" are particularly good for situating Week 10 in the broader timeline.
- **Yann LeCun's NYU Deep Learning course (free)** — <https://atcold.github.io/NYU-DLSP21/>. Lectures 6 and 7 cover RNNs and LSTMs; the slides are downloadable PDFs, the video lectures are on YouTube.
- **The Annotated Transformer / Annotated GRU pedagogical line.** Sasha Rush's "The Annotated Transformer" (<http://nlp.seas.harvard.edu/annotated-transformer/>) is Week 11 reading, but the same pedagogical genre — paper + working code in a single notebook — has produced "Annotated GRU" and "Annotated LSTM" tutorials over the years. Search and skim if the lecture-note algebra is not enough.

## What to skip (or save for later)

- **TensorFlow / Keras LSTM tutorials.** PyTorch's `nn.LSTM` API differs from `tf.keras.layers.LSTM` in subtle ways (most notably: PyTorch defaults to `(seq, batch, feat)` while Keras defaults to `(batch, seq, feat)`). Mixing the two reference styles is a common source of confusion. Stay in PyTorch this week.
- **HuggingFace's `transformers` library.** Empty for sequence-modeling fundamentals; entirely about pretrained transformer models. Week 13 territory.
- **`torchtext`.** Officially deprecated since June 2024. The C5 char-LM mini-project does its own tokenization (one line: `vocab = sorted(set(text))`); word-level tokenization in 2026 is usually done with `tiktoken`, `sentencepiece`, or the tokenizer that ships with whatever pretrained model you are fine-tuning.
- **WaveNet and other dilated-CNN sequence models.** Conceptually adjacent but out of C5 scope. Dilated CNNs reach long sequences with deep stacks rather than recurrence; the transformer that supersedes both is the Week 11 topic. See van den Oord et al. 2016 (<https://arxiv.org/abs/1609.03499>) if curious.
- **Echo State Networks and Liquid State Machines.** Goodfellow chapter 10.7 covers them; they were a 2000s-era alternative to BPTT that bypassed gradient flow by freezing the recurrent weights and learning only the readout. Historically interesting; not in production anywhere we know of.

## Citation cheatsheet (for the homework report)

If you cite any of the following in your mini-project report, use this format:

- **LSTM:** Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- **GRU:** Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP 2014*. arXiv:1406.1078.
- **Vanishing gradients:** Pascanu, R., Mikolov, T., and Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML 2013*. arXiv:1211.5063.
- **GRU vs. LSTM:** Chung, J., Gulcehre, C., Cho, K., and Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *NIPS 2014 Workshop*. arXiv:1412.3555.
- **Char-RNN:** Karpathy, A. (2015). The unreasonable effectiveness of recurrent neural networks. Blog post, 21 May 2015. <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>.

That is enough to anchor a portfolio write-up. The mini-project rubric does not require formal citations; it does require that any quantitative claim ("LSTM beats GRU on this task") be backed by your own measured numbers or a cited paper.
