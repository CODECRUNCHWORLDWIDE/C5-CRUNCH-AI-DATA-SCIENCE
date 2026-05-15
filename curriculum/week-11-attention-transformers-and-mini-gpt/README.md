# Week 11 — Attention, Transformers, and a Mini-GPT

> *Week 10 closed with a 2-layer LSTM on Pride and Prejudice. The model worked — it produced English-shaped samples after thirty minutes of CPU training — but Lecture 3 of that week named three things the LSTM could not do. It could not look back across a 500-token context in one operation. It could not parallelize its forward pass across the sequence length. And its gradient path from output to input scaled linearly with sequence length, so the analysis we did for vanishing gradients applied. Vaswani et al. 2017, "Attention Is All You Need" (<https://arxiv.org/abs/1706.03762>), proposed an architecture that fixes all three. The recurrence is replaced by a softmax-weighted lookup over the entire sequence; the unroll is replaced by a single matmul that parallelizes over the time dimension; the gradient path from output to input is `O(1)`. This week we build that architecture from scratch in PyTorch 2.x and train a small nanoGPT-style char-level model on the same Project Gutenberg corpus that Week 10 used. The training pipeline is identical to Week 10's. Only the model changes. That is the entire point of the lecture sequence.*

Welcome to week eleven of **C5 · Crunch AI / Data Science**. Week 10 took us through the recurrent architectures that dominated sequence modeling from 1997 to 2017 — the vanilla RNN, the LSTM, the GRU. Week 11 makes the architectural pivot that the field made in mid-2017. We replace recurrence with **self-attention**, we replace the sequential unroll with a single matmul over the sequence dimension, and we end the week with a working char-level transformer trained on the same `Pride and Prejudice` text that the Week 10 LSTM trained on. The before-and-after comparison — same data, same training loop, same eight lines, two different model classes — is the cleanest demonstration of architectural progress in the entire C5 curriculum.

Four commitments before we start:

1. **The Week 8 training loop carries over verbatim, again.** The `for epoch: for batch: zero_grad / forward / loss / backward / step` pattern has not changed since Week 8. The model class changes (a `nn.Module` containing self-attention blocks rather than `nn.LSTM`); the data pipeline does not change (the same `make_chunked_streams` from Week 10 produces the integer-tensor chunks the transformer eats). The C5 conviction stands: write the same eight lines every week.
2. **We build attention by hand before we touch `nn.MultiheadAttention`.** Exercise 1 has you implement scaled dot-product attention from the equation `softmax(Q K^T / sqrt(d_k)) V` in pure PyTorch and verify against `torch.nn.functional.scaled_dot_product_attention`. Exercise 2 implements multi-head attention with explicit Q/K/V projections and head splitting. Only Exercise 3 (and the mini-project) use the fused `F.scaled_dot_product_attention` kernel. Students who skip the hand-rolled attention cannot answer "where does the `sqrt(d_k)` come from?" in interview.
3. **We build a decoder-only (GPT-style) transformer, not an encoder-decoder.** The 2017 paper introduced both halves of the encoder-decoder architecture for machine translation. The GPT line of work (Radford et al. 2018, 2019, 2020) showed that a decoder-only stack with causal masking is sufficient for language modeling and is the architectural ancestor of every modern large language model. We build the decoder-only stack this week; the full encoder-decoder is a Week 12 topic.
4. **We train the same char-LM task as Week 10.** Same Project Gutenberg text. Same `make_chunked_streams` data pipeline. Same `nn.CrossEntropyLoss`. Same `Adam(lr=1e-3)`. Same sampling code with temperature. The mini-project rubric asks you to train both the Week 10 LSTM and the Week 11 mini-GPT for the same number of epochs and compare convergence, wall-clock, and sample quality. The transformer should win on all three axes, but only barely on a 700 KB corpus — the architectural advantage of attention shows up most clearly on the much larger corpora that are out of scope for a single-laptop week.

We target **PyTorch 2.x** (we test on 2.4 and 2.5) and **Python 3.11+**. No torchvision, no torchtext, no HuggingFace libraries this week — the entire mini-project is one Python file plus the cached corpus. The Apple Silicon `mps` backend works for self-attention and is the fastest option on a MacBook; CUDA is what the C5 grading rig runs on. CPU-only is enough for every exercise, and the mini-project trains the 6-layer 384-dim transformer in ~40 minutes on CPU, ~3 minutes on a free Google Colab T4.

PyTorch reference docs: <https://pytorch.org/docs/stable/index.html>. The four primary references for this week:

- `nn.MultiheadAttention` — <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>
- `torch.nn.functional.scaled_dot_product_attention` — <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
- `nn.TransformerEncoderLayer` — <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>
- `nn.LayerNorm` — <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>

Pin all four.

---

## Learning objectives

By the end of this week, you will be able to:

- **Derive scaled dot-product attention.** Given three matrices `Q`, `K`, `V` of shapes `(T, d_k)`, `(T, d_k)`, `(T, d_v)`, compute `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V` on paper for a `T = 3` example, and identify the `(T, T)` attention-weight matrix as the object that "looks up" relationships between every pair of positions. Reference: Vaswani et al. 2017, equation 1 (<https://arxiv.org/abs/1706.03762>).
- **Explain the `sqrt(d_k)` scaling.** Argue why, without the `1 / sqrt(d_k)` factor, the softmax saturates as `d_k` grows. The dot product of two random unit-variance `d_k`-dimensional vectors has variance `d_k`; dividing by `sqrt(d_k)` keeps the variance at unity and the softmax in its useful range. Reference: Vaswani et al. 2017, footnote 4.
- **Implement multi-head attention from Q/K/V projections.** Project the input three times into a `(num_heads, d_k)` space, run scaled dot-product attention per head, concatenate the heads' outputs, and project back. The multi-head version uses `num_heads * d_k = d_model` total dimension, so the parameter cost matches a single-head attention with the full model dimension. Reference: Vaswani et al. 2017, section 3.2.2.
- **Implement causal masking.** Build the lower-triangular mask `tril(ones(T, T))` and apply it by setting the upper-triangular entries of the attention logits to `-inf` before the softmax. Causal masking is what makes the transformer autoregressive: at position `t` it can only see positions `1..t`, not `t+1..T`. Reference: Vaswani et al. 2017, section 3.2.3 (the "masked self-attention" in the decoder).
- **Explain sinusoidal vs. learned positional encoding.** Sinusoidal: `PE(pos, 2i) = sin(pos / 10000^(2i / d_model))`, fixed, generalizes to longer sequences than seen at training time. Learned: a `(max_seq_len, d_model)` `nn.Embedding` of position indices, trainable, slightly better in-distribution on the original paper's tasks. GPT-style models use learned; the C5 default for this week is learned for pedagogical simplicity. Reference: Vaswani et al. 2017, section 3.5, and Radford et al. 2018 (GPT-1).
- **Describe the transformer block.** A residual stream that flows top to bottom; at each block, two sublayers — multi-head self-attention and a position-wise feedforward MLP — each surrounded by a `LayerNorm + residual` wrapper. The C5 default is the modern "pre-norm" variant (`LN -> sublayer -> residual`), not the original 2017 "post-norm" variant; pre-norm is more stable to train and is what nanoGPT, GPT-2, GPT-3, and every modern transformer uses. Reference: Vaswani et al. 2017 for post-norm, Xiong et al. 2020 (<https://arxiv.org/abs/2002.04745>) for the pre-norm argument.
- **Explain the residual stream view.** The transformer is best understood as a **residual stream** of `d_model`-dimensional vectors (one per token position), where each block reads from the stream and writes back additively. Self-attention writes a "move information across positions" delta; the MLP writes a "compute on each position" delta. The residual stream is the unifying mental model that the Anthropic 2022 mechanistic-interpretability work brought into the mainstream. Reference: Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits" (<https://transformer-circuits.pub/2021/framework/index.html>).
- **Implement a decoder-only (GPT-style) transformer.** Stack `n_layer` transformer blocks with causal masking; prepend a token-embedding plus positional-encoding sum; append a `LayerNorm + Linear(d_model, vocab_size)` head; tie the input embedding to the output projection. Reference: nanoGPT (<https://github.com/karpathy/nanoGPT>) — the cleanest pedagogical implementation in PyTorch.
- **Train the mini-GPT on the same corpus as Week 10.** Same `Pride and Prejudice` text. Same `make_chunked_streams`. Same training loop. Compare the final validation loss, wall-clock per epoch, and sampled passage at `T = 0.8` against the Week 10 LSTM. The mini-GPT should reach lower validation loss in fewer epochs and produce more coherent samples; report by how much.
- **Cite the four foundational references.** Vaswani et al. 2017 ("Attention Is All You Need"; <https://arxiv.org/abs/1706.03762>), Karpathy's nanoGPT (<https://github.com/karpathy/nanoGPT>), Karpathy's "Let's build GPT" tutorial video and transcript (<https://www.youtube.com/watch?v=kCc8FmEb1nY> and the companion repo), Alammar's "The Illustrated Transformer" (<https://jalammar.github.io/illustrated-transformer/>). All four are free.
- **Ship** a mini-project: a nanoGPT-style char-level transformer training script, a sampling script, a side-by-side comparison with Week 10's LSTM, three sampled passages at three temperatures, and an 800-1200 word report. Pushed to your portfolio repo.
- **Pass** every `pytest` case on the Week 11 exercises.

---

## Prerequisites

- **Week 10 complete.** The LSTM exercises, the char-LM mini-project, the sampled passages at three temperatures. The Week 11 mini-project loads the same data pipeline; if your Week 10 code is broken, your Week 11 code will not run.
- **Python 3.11+** with PyTorch 2.x installed (`pip install "torch>=2.4,<3"`). No torchvision, no torchtext, no transformers. The fused-attention kernel `F.scaled_dot_product_attention` requires PyTorch 2.0+; we use 2.4+ as the C5 floor.
- **About 10 MB of disk** for the same `Pride and Prejudice` text cache that Week 10 used. The mini-project will reuse Week 10's `data/pride_and_prejudice.txt` if it exists.
- **Optional but useful:** a CUDA GPU, an Apple Silicon M1+, or a free Google Colab T4. The mini-GPT trains in ~40 minutes on a CPU but is significantly more fun to iterate on with a GPU; the attention kernels are heavily optimized on GPU.

You should already be comfortable with:

- **PyTorch tensor shapes for sequences.** `(batch, seq_len, d_model)` is the convention we use throughout this week (and that every modern transformer codebase uses). The Week 10 `batch_first=True` convention carries forward.
- **`nn.Module` subclassing, `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, `nn.CrossEntropyLoss`.** All introduced in Weeks 7-10.
- **Truncated BPTT and temperature sampling.** Week 10 Lecture 3. The transformer training loop reuses both ideas with minor adjustments.

---

## Topics covered

- **What attention is.** A learned, content-based lookup over a sequence. Given a query vector `q`, a set of keys `K = [k_1, ..., k_T]`, and a set of values `V = [v_1, ..., v_T]`, attention computes `softmax(q @ K^T / sqrt(d_k))` and uses that probability distribution to take a convex combination of the values. The output is a single vector of dimension `d_v`. Generalized to a set of queries `Q = [q_1, ..., q_T]`, attention computes one such combination per query, producing an output matrix of shape `(T, d_v)`. Reference: Vaswani et al. 2017, equation 1.
- **The `sqrt(d_k)` scaling.** Without it, the variance of `q · k` grows linearly with `d_k`, and the softmax saturates for `d_k > ~32`. With it, the variance stays at 1.0 regardless of `d_k`, and the softmax operates in the regime where the gradient is informative. The scaling is the difference between the "additive" attention of Bahdanau et al. 2014 and the "scaled dot-product" attention of Vaswani et al. 2017; the latter is faster on GPU because it is just two matmuls and a softmax.
- **Self-attention.** The case where `Q`, `K`, `V` all come from the same input sequence. Each token position becomes a query that looks up information from all other positions. This is the architectural mechanism by which a token "sees" the rest of the sequence — directly, in one operation, without any recurrence.
- **Multi-head attention.** Run `h` separate attention operations in parallel, each on a `d_k = d_model / h`-dimensional subspace. Concatenate the `h` outputs and project back to `d_model`. The intuition: different heads learn to attend to different relationships (one head may attend to the previous token, another to the subject of the sentence, another to long-range context). The Anthropic interpretability work demonstrated this empirically on real GPT-2 heads; see "induction heads" in Olsson et al. 2022 (<https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>). Reference: Vaswani et al. 2017, section 3.2.2.
- **Causal masking.** For autoregressive language modeling, the model at position `t` must only see positions `1..t`. We enforce this by adding a lower-triangular mask of `-inf` to the attention logits (`softmax(-inf) = 0`). The mask is fixed; it has no parameters. PyTorch's `F.scaled_dot_product_attention` accepts an `is_causal=True` flag that applies the mask internally on GPU at no extra cost. Reference: Vaswani et al. 2017, section 3.2.3.
- **Positional encoding.** Self-attention is *permutation-equivariant* — `Attention(P @ X) = P @ Attention(X)` for any permutation matrix `P`. To recover sequence order we add a position-dependent vector to each token's embedding at the input. Two flavors: sinusoidal (fixed, generalizes to longer sequences) and learned (an `nn.Embedding` of position indices, trained from scratch). GPT-2 uses learned; the original Transformer used sinusoidal; modern large models often use RoPE or ALiBi (out of C5 scope this week). The C5 default for the mini-project: learned, because it is two lines of code and is what nanoGPT uses.
- **The transformer block.** Two sublayers — multi-head self-attention and a position-wise feedforward MLP (typically `Linear(d_model, 4 * d_model) -> GELU -> Linear(4 * d_model, d_model)`) — each wrapped by `LayerNorm + residual`. The 2017 paper used post-norm (`x = LayerNorm(x + sublayer(x))`); modern practice and nanoGPT use pre-norm (`x = x + sublayer(LayerNorm(x))`) because it is empirically more stable to train. Reference: Vaswani et al. 2017 (post-norm) and Xiong et al. 2020 (pre-norm).
- **Layer normalization.** `LayerNorm` standardizes each token's `d_model`-dimensional vector to mean 0, variance 1, then applies a learned scale and bias. Unlike `BatchNorm`, it operates per-token-per-example, so it does not need a batch dimension and behaves identically at training and inference. Reference: Ba, Kiros, Hinton 2016 (<https://arxiv.org/abs/1607.06450>) and `nn.LayerNorm` docs.
- **The residual stream.** The transformer's `(batch, seq_len, d_model)` tensor flows top to bottom of the stack, with each block adding a delta. This view, popularized by Anthropic's mechanistic-interpretability research, is the cleanest mental model for understanding what each block contributes. Self-attention writes a "move information across positions" delta; the MLP writes a "compute on each position" delta. The unembed (output) projection reads the final residual stream and produces logits.
- **Decoder-only vs. encoder-decoder.** The 2017 paper introduced both. The decoder-only stack (used by GPT-1, GPT-2, GPT-3, GPT-4, Claude, Llama, every modern LLM) takes the input as the "decoder side" of the original architecture, with causal masking. The encoder-decoder stack (used by T5, BART, the original 2017 machine-translation model) keeps both halves; it is a Week 12 topic. The C5 mini-project is decoder-only.
- **nanoGPT.** Karpathy's 300-line PyTorch reference implementation (<https://github.com/karpathy/nanoGPT>). It is the most-readable transformer implementation on the internet and is the structural model for the Week 11 mini-project. The companion video "Let's build GPT: from scratch, in code, spelled out" (<https://www.youtube.com/watch?v=kCc8FmEb1nY>) walks through every line; the mini-project is a slightly-condensed reproduction of the same recipe.
- **The Illustrated Transformer.** Jay Alammar's 2018 blog post (<https://jalammar.github.io/illustrated-transformer/>). The diagrams of Q/K/V multiplication, of multi-head splitting, of the attention probability matrix, are the canonical pictorial explanation of the 2017 architecture. Required reading for this week.
- **What we do not do.** No pretrained models (Week 13). No fine-tuning (Week 13). No HuggingFace `transformers` library (Week 13). No mixed precision, no `torch.compile`, no flash-attention beyond `F.scaled_dot_product_attention`'s built-in selection of the best available kernel (Week 14 covers performance). No RoPE, ALiBi, or other modern positional schemes (mentioned in lecture, not on the exercise track). No encoder-decoder architectures (Week 12). No BERT-style masked-language-modeling (Week 12).

---

## Weekly schedule

Target about **40 hours**. Monday and Tuesday cover scaled dot-product attention, multi-head attention, and positional encoding. Wednesday is the transformer block, the residual stream, and the pre-norm vs. post-norm question. Thursday is the decoder-only stack and causal masking. Friday-Sunday is the nanoGPT-style mini-project.

| Day       | Focus                                                            | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Scaled dot-product attention; the QKV story; sqrt(d_k) scaling   |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Tuesday   | Multi-head attention; positional encoding; sinusoidal vs learned |   3h     |   2h      |     0h     |   0.5h    |   1h     |     0h       |   0.5h     |    7h       |
| Wednesday | The transformer block; residual stream; pre-norm; LayerNorm      |   2h     |   2h      |     2h     |   0.5h    |   1h     |     0h       |   0h       |    7.5h     |
| Thursday  | Decoder-only stack; causal masking; the full nanoGPT model       |   2h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0h       |    7.5h     |
| Friday    | Mini-project: train the mini-GPT on Pride and Prejudice          |   0h     |   0.5h    |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    5.5h     |
| Saturday  | Mini-project: compare to Week 10 LSTM; sample at 3 temperatures  |   0h     |   0h      |     0h     |   0h      |   1h     |     2h       |   0h       |    3h       |
| Sunday    | Quiz, push to portfolio repo                                     |   0h     |   0h      |     2h     |   0.5h    |   0h     |     0h       |   0h       |    2.5h     |
| **Total** |                                                                  | **10h**  | **7.5h**  | **4h**     | **3h**    | **6h**   | **8h**       | **1h**     |  **40h**    |

Monday's lecture takes the longest because the QKV mechanics need to be drawn on paper for a `T = 3` toy example before the algebra makes sense. Tuesday's positional-encoding lecture is short but the sinusoidal-encoding identity (the one that says `PE(pos + k)` is a linear function of `PE(pos)`) deserves the full thirty minutes the lecture devotes to it. Wednesday's `LayerNorm + residual` material is the architectural reason transformers train at depth 96+; do not rush it.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | Vaswani 2017, nanoGPT, the Karpathy video, the Alammar blog, the Anthropic interpretability essays, PyTorch attention/LayerNorm docs |
| [lecture-notes/01-scaled-dot-product-and-multihead-attention.md](./lecture-notes/01-scaled-dot-product-and-multihead-attention.md) | Scaled dot-product attention from first principles; the QKV story; the sqrt(d_k) scaling; multi-head attention; causal masking |
| [lecture-notes/02-positional-encoding-and-the-transformer-block.md](./lecture-notes/02-positional-encoding-and-the-transformer-block.md) | Sinusoidal vs. learned positional encoding; the transformer block; pre-norm vs. post-norm; LayerNorm; the residual stream |
| [lecture-notes/03-decoder-only-and-mini-gpt.md](./lecture-notes/03-decoder-only-and-mini-gpt.md) | The decoder-only architecture; causal masking end-to-end; weight tying; training the mini-GPT on Project Gutenberg text; sampling with temperature; the comparison to the Week 10 LSTM |
| [exercises/exercise-01-scaled-dot-product-attention.py](./exercises/exercise-01-scaled-dot-product-attention.py) | Implement scaled dot-product attention from the equation in pure PyTorch; verify against `F.scaled_dot_product_attention` |
| [exercises/exercise-02-multihead-attention.py](./exercises/exercise-02-multihead-attention.py) | Implement multi-head attention with explicit Q/K/V projections and head splitting; verify against `nn.MultiheadAttention` |
| [exercises/exercise-03-causal-mask-and-positional-encoding.py](./exercises/exercise-03-causal-mask-and-positional-encoding.py) | Build the lower-triangular causal mask and the sinusoidal positional encoding; verify both with worked numerical examples |
| [exercises/SOLUTIONS.md](./exercises/SOLUTIONS.md) | Reference solutions with commentary; read only after attempting |
| [challenges/challenge-01-attention-head-visualization.md](./challenges/challenge-01-attention-head-visualization.md) | Train a small transformer and visualize what each attention head is doing as a heatmap; reproduce the "previous-token head" and "induction head" findings from Anthropic 2022 |
| [challenges/challenge-02-lstm-vs-mini-gpt-bake-off.md](./challenges/challenge-02-lstm-vs-mini-gpt-bake-off.md) | Train the Week 10 LSTM and the Week 11 mini-GPT on matched-parameter budgets; compare convergence, wall-clock per epoch, and final sample quality |
| [quiz.md](./quiz.md) | 10 multiple-choice questions on attention, multi-head, positional encoding, the transformer block, and causal masking |
| [homework.md](./homework.md) | Six problems for the week |
| [mini-project/README.md](./mini-project/README.md) | Build a nanoGPT-style decoder-only transformer in PyTorch and train it on the same Project Gutenberg text that Week 10 used |
| [mini-project/starter.py](./mini-project/starter.py) | Runnable starter: same text download as Week 10, same chunked dataset, a 6-layer pre-norm transformer with multi-head causal self-attention, training loop, temperature sampler |
| [mini-project/rubric.md](./mini-project/rubric.md) | Grading rubric for the mini-project, including the side-by-side comparison to Week 10's LSTM |

---

## Stretch goals

- Read **Vaswani et al. 2017 — "Attention Is All You Need"** (<https://arxiv.org/abs/1706.03762>). Fifteen pages. The single most-cited paper in deep learning. The math is straightforward; the prose is dense but rewards a careful read. Section 3 is the architecture (attention, multi-head, positional encoding, block); section 5 is the experiments on English-to-German and English-to-French translation; section 6 is the ablations. Read in this order: section 3.2.1 (scaled dot-product attention), section 3.2.2 (multi-head), section 3.3 (the feedforward), section 3.5 (positional encoding), then section 3.1 (the overall architecture) for the integration. The 2017 paper is one of the cleanest research papers ever published — there is a reason it has 130,000+ citations.
- Watch **Karpathy 2023 — "Let's build GPT: from scratch, in code, spelled out"** (<https://www.youtube.com/watch?v=kCc8FmEb1nY>). About 100 minutes. Karpathy implements a decoder-only transformer from a blank Python file, line by line, with running commentary on every design choice. The video is the single best practitioner's tutorial on the architecture; the C5 mini-project follows the same recipe.
- Read **Karpathy's nanoGPT repository** (<https://github.com/karpathy/nanoGPT>). About 300 lines of PyTorch. The cleanest production-quality transformer implementation on the internet; the C5 mini-project starter is a condensed version of `nanoGPT/model.py`. Read `model.py` line by line; the structure mirrors the lecture notes exactly.
- Read **Alammar 2018 — "The Illustrated Transformer"** (<https://jalammar.github.io/illustrated-transformer/>). Twenty minutes. The diagrams of QKV multiplication, multi-head splitting, and the attention probability matrix are the canonical pictorial explanations. Pair with Lecture 1 for maximum effect.
- Read **Elhage et al. 2021 — "A Mathematical Framework for Transformer Circuits"** (<https://transformer-circuits.pub/2021/framework/index.html>). Long; about 60 pages. Anthropic's foundational mechanistic-interpretability paper. The "residual stream" mental model that Lecture 2 uses originates here. Read the introduction and section "Decomposing the residual stream" — about thirty minutes. The rest is a deep dive that pays off in Week 12 and beyond.
- Read **Olsson et al. 2022 — "In-context Learning and Induction Heads"** (<https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>). Anthropic's empirical demonstration that real GPT-2 attention heads implement specific computational primitives. The "induction head" finding — heads that copy `[B]` after `[A B] [A]` — is the canonical example of a learned, interpretable attention pattern. Challenge 1 reproduces a simplified version.

---

## What you will *not* do this week

You will not:

- **Use a pretrained model.** No GPT-2, no GPT-3, no Llama, no Claude. Those are Week 13 topics (transfer learning) and Week 14 (fine-tuning). Week 11 is about understanding the architecture; pretrained weights would short-circuit the lesson.
- **Use HuggingFace transformers.** The library is excellent and is the right tool for production work, but it abstracts away every interesting detail of the architecture. Week 13 introduces it; this week the entire model is in one file.
- **Build an encoder-decoder.** Week 12. The 2017 paper introduced both, but the decoder-only architecture is the one every modern LLM uses; we cover it first.
- **Use mixed-precision or `torch.compile`.** The default float32 forward and backward is fast enough for a 1M-parameter model on `Pride and Prejudice`. Performance engineering is a Week 14 topic.
- **Use flash-attention as a separately-installed library.** PyTorch 2.x's `F.scaled_dot_product_attention` automatically dispatches to flash-attention on supported GPUs; we use that one API everywhere. The standalone `flash-attn` package is out of scope.
- **Tune hyperparameters extensively.** The mini-project ships a single working recipe (6 layers, 6 heads, `d_model = 384`, `lr = 3e-4`). The homework explores one-axis sensitivity (number of layers, number of heads); a full sweep is a Week 14 topic.
- **Use a tokenizer beyond character-level.** Subword tokenization (BPE, WordPiece, SentencePiece) is a Week 12 topic. The char-level recipe is sufficient for the architectural lessons and keeps the entire pipeline in one Python file.

That list is deliberate. The point of Week 11 is to *build the transformer block from scratch and feel why attention is the inductive bias that replaced recurrence*. SOTA is not the goal; the conceptual scaffolding is.

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The Lecture 1 experiment that prints the attention probability matrix on a random `Q`, `K`, `V` is the experiment that makes "attention is a softmax over positions" stop being a slogan. The Lecture 2 experiment that plots the sinusoidal positional encoding as a heatmap reveals the band structure of the encoding and makes the "linear function of position" identity visible to the eye. The Lecture 3 experiment that times the mini-GPT and the Week 10 LSTM on a fixed batch shows the parallel-vs-sequential difference in wall-clock terms. All three are under fifteen minutes; all three are the difference between "I read about it" and "I have done it."

---

## Up next

[Week 12 — Encoder-decoder, BPE tokenization, and seq2seq translation](../week-12/) — once you have pushed the Week 11 mini-GPT, the LSTM-vs-mini-GPT comparison, and the sampled-text passages to your portfolio repo. Week 12 introduces the *encoder* half of the 2017 paper (the half we skipped this week), subword tokenization (BPE and SentencePiece), and a small English-to-French translation model. The architecture changes; the training loop does not. The C5 conviction — write the same eight lines every week — continues to hold.
