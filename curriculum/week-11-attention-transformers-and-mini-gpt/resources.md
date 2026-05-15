# Week 11 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs, no signup-required courses. If a link breaks, please open an issue.

## Required reading (work it into your week)

- **Vaswani, Ashish; Shazeer, Noam; Parmar, Niki; Uszkoreit, Jakob; Jones, Llion; Gomez, Aidan N.; Kaiser, Łukasz; Polosukhin, Illia — "Attention Is All You Need"** (NeurIPS 2017; the transformer paper):
  <https://arxiv.org/abs/1706.03762>
  Read **Sunday evening** before Monday's lecture. Fifteen pages. The single most-cited deep-learning paper of the 2010s. Read in this order: section 3.2.1 (scaled dot-product attention), section 3.2.2 (multi-head), section 3.3 (the position-wise feedforward), section 3.5 (positional encoding), then section 3.1 (the overall architecture). Section 5 is the original English-to-German translation experiments and is optional for this week. Section 6 (ablations) is interesting but also optional.
- **Alammar, Jay — "The Illustrated Transformer"** (the 2018 blog post that became the canonical pictorial explanation of the 2017 architecture):
  <https://jalammar.github.io/illustrated-transformer/>
  Read **Monday morning** before Lecture 1. Twenty minutes. The diagrams of Q/K/V multiplication, of head splitting, of the attention probability matrix, appear in every textbook and every YouTube transformer explanation. The original is still the cleanest version. Pair with Vaswani 2017 for maximum effect.
- **Karpathy, Andrej — "Let's build GPT: from scratch, in code, spelled out"** (the 2023 YouTube tutorial; about 100 minutes):
  <https://www.youtube.com/watch?v=kCc8FmEb1nY>
  Watch **Tuesday evening** after Lecture 2. The most-watched practitioner tutorial on transformer implementation. Karpathy builds a decoder-only model from a blank Python file with running commentary on every design choice. The companion repository (<https://github.com/karpathy/ng-video-lecture>) has the final notebook. The C5 mini-project follows the same recipe with minor renamings.
- **Karpathy, Andrej — `nanoGPT`** (the reference implementation; about 300 lines of PyTorch):
  <https://github.com/karpathy/nanoGPT>
  Read `model.py` **Wednesday morning** before Lecture 3. About thirty minutes. The cleanest production-quality decoder-only transformer implementation on the internet. The C5 mini-project starter is structurally identical to `nanoGPT/model.py`. Pin the tab; refer to it whenever the mini-project asks for an implementation detail that the lecture notes do not specify.
- **PyTorch docs — `torch.nn.functional.scaled_dot_product_attention`** (the fused attention kernel introduced in PyTorch 2.0):
  <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
  Pin the tab. The function dispatches to flash-attention on supported CUDA GPUs, to a memory-efficient kernel on others, and to a math fallback on CPU. The `is_causal=True` argument applies the lower-triangular mask for you. This is the function the mini-project uses.
- **PyTorch docs — `nn.MultiheadAttention`** (the higher-level multi-head wrapper):
  <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>
  Pin it. The module wraps the Q/K/V projections, the head splitting, and the scaled dot-product call into one. The mini-project uses a custom implementation for pedagogical clarity, but the verification tests in Exercise 2 compare against `nn.MultiheadAttention`.
- **PyTorch docs — `nn.TransformerEncoderLayer`** (the canonical transformer block):
  <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>
  Pin it. The module implements one transformer block (attention + feedforward, each with residual + layernorm). PyTorch's default is post-norm; the `norm_first=True` flag selects the pre-norm variant that the C5 mini-project uses.

## The papers (free PDFs)

- **Vaswani et al. 2017 — "Attention Is All You Need"** (NeurIPS 2017):
  <https://arxiv.org/abs/1706.03762>
  The paper. Required reading.
- **Bahdanau, Dzmitry; Cho, Kyunghyun; Bengio, Yoshua — "Neural Machine Translation by Jointly Learning to Align and Translate"** (ICLR 2015; the attention mechanism's first appearance in NMT):
  <https://arxiv.org/abs/1409.0473>
  Twelve pages. The 2015 paper that introduced "attention" as a learned alignment between encoder and decoder hidden states in a sequence-to-sequence translation model. The architecture is encoder-decoder RNN with additive attention; the 2017 Vaswani paper drops the RNN and uses self-attention everywhere. Worth reading section 3 for the historical context — attention did not arrive in 2017; it arrived in 2015 and reached its modern form in 2017.
- **Radford, Alec; et al. — "Improving Language Understanding by Generative Pre-Training"** (OpenAI 2018; GPT-1):
  <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>
  Twelve pages. The first paper to demonstrate that a decoder-only transformer trained on a large unlabeled text corpus generalizes to many downstream tasks via fine-tuning. The architectural diagram in Figure 1 is what the C5 mini-project implements (at 1/1000th the scale). Optional but recommended.
- **Radford, Alec; et al. — "Language Models are Unsupervised Multitask Learners"** (OpenAI 2019; GPT-2):
  <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
  Twenty-four pages. The GPT-2 paper. Architecturally near-identical to GPT-1; scaled up from 117M to 1.5B parameters. The C5 mini-project is structurally identical to a tiny GPT-2.
- **Ba, Lei Jimmy; Kiros, Jamie Ryan; Hinton, Geoffrey E. — "Layer Normalization"** (2016):
  <https://arxiv.org/abs/1607.06450>
  Fourteen pages. The paper that introduced `LayerNorm` as a replacement for `BatchNorm` in recurrent and attention architectures. Sections 1-3 are the conceptual core. The C5 lectures cover the *what* and *why* in five paragraphs; this paper has the rigorous version.
- **Xiong, Ruibin; et al. — "On Layer Normalization in the Transformer Architecture"** (ICML 2020):
  <https://arxiv.org/abs/2002.04745>
  Twelve pages. The paper that demonstrated pre-norm transformers are more stable to train than the original post-norm ones. The argument is theoretical (the gradient through a post-norm residual stream is exponentially large at initialization) and empirical (pre-norm transformers do not need learning-rate warmup). Modern practice followed this paper; nanoGPT and the C5 mini-project use pre-norm.
- **Elhage et al. 2021 — "A Mathematical Framework for Transformer Circuits"** (Anthropic):
  <https://transformer-circuits.pub/2021/framework/index.html>
  Long; about 60 web pages. The paper that introduced the *residual stream* mental model that Lecture 2 leans on. Read the introduction and the "Decomposing the residual stream" section; everything else is bonus. The full paper is a deep dive that pays off in Week 12 and beyond.
- **Olsson et al. 2022 — "In-context Learning and Induction Heads"** (Anthropic):
  <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>
  The empirical demonstration that real GPT-2 attention heads implement specific, interpretable computational primitives. Challenge 1 reproduces a simplified version of the "previous-token head" and "induction head" findings.

## The official PyTorch 2.x docs you will live in this week

- **`torch.nn.functional.scaled_dot_product_attention`** (the fused attention kernel; flash-attention on supported GPUs):
  <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
- **`torch.nn.MultiheadAttention`** (the higher-level multi-head wrapper):
  <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>
- **`torch.nn.TransformerEncoderLayer`** (one transformer block; non-causal):
  <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>
- **`torch.nn.TransformerDecoderLayer`** (one transformer-decoder block; causal):
  <https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html>
- **`torch.nn.Transformer`** (the full encoder-decoder stack from the 2017 paper):
  <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>
- **`torch.nn.LayerNorm`** (the layer normalization used in every transformer block):
  <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>
- **`torch.nn.Embedding`** (the token-embedding and position-embedding lookup table):
  <https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html>
- **`torch.nn.GELU`** (the activation function used in the position-wise feedforward block; GPT-2 default):
  <https://pytorch.org/docs/stable/generated/torch.nn.GELU.html>
- **`torch.nn.Dropout`** (the dropout regularization applied after attention and MLP sublayers):
  <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html>
- **`torch.tril`** (the lower-triangular helper for building causal masks):
  <https://pytorch.org/docs/stable/generated/torch.tril.html>
- **`torch.nn.utils.clip_grad_norm_`** (still used; training transformers without gradient clipping is risky):
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>
- **`torch.nn.CrossEntropyLoss`** (the loss for language modeling, same as Week 10):
  <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>

## Companion blog posts and tutorials (free)

- **Alammar, Jay — "The Illustrated Transformer"** (the 2018 canonical pictorial explanation):
  <https://jalammar.github.io/illustrated-transformer/>
  Required reading; listed again here so the URL is in one place.
- **Alammar, Jay — "The Illustrated GPT-2"** (the 2019 follow-up; covers the decoder-only architecture specifically):
  <https://jalammar.github.io/illustrated-gpt2/>
  Twenty minutes. The diagrams of the GPT-2 stack are nearly identical to what the C5 mini-project implements. Read after Lecture 3.
- **Karpathy, Andrej — "Let's build GPT: from scratch, in code, spelled out"** (the 2023 YouTube tutorial):
  <https://www.youtube.com/watch?v=kCc8FmEb1nY>
  Required watching; listed again here for one-stop reference.
- **Karpathy — nanoGPT repository** (the reference PyTorch implementation):
  <https://github.com/karpathy/nanoGPT>
  Read `model.py` line by line. The most-readable transformer implementation on the internet.
- **Rush, Sasha — "The Annotated Transformer"** (Harvard NLP 2018; updated 2022):
  <http://nlp.seas.harvard.edu/annotated-transformer/>
  Notebook-style walkthrough of the 2017 paper with working PyTorch code interleaved with the paper's prose. Longer and more thorough than nanoGPT; covers the encoder-decoder architecture too (which we skip this week and revisit in Week 12).
- **PyTorch official tutorial — "Language Modeling with `nn.Transformer` and `torchtext`"**:
  <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>
  The PyTorch team's beginner walkthrough of `nn.TransformerEncoderLayer` for language modeling. Useful as a cross-check that the C5 mini-project's per-block API matches the official one. Note that the tutorial uses `torchtext` (deprecated since 2024) for tokenization; ignore that part.
- **Olah, Christopher — "Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases"** (Distill 2022):
  <https://distill.pub/2020/circuits/zoom-in/>
  The introductory post in Anthropic's mechanistic-interpretability series. Optional but lovely; situates the residual-stream view from Lecture 2 in the broader research program.
- **Phuong, Mary; Hutter, Marcus — "Formal Algorithms for Transformers"** (DeepMind 2022):
  <https://arxiv.org/abs/2207.09238>
  Twenty pages. A formal pseudocode specification of the entire transformer architecture (encoder, decoder, attention, feedforward, layernorm, training, sampling) in numbered algorithms. The reference if you want the most-rigorous version of the architecture in one place. Optional.
- **Smith, Christian — "Transformer Inference Arithmetic"** (Kipp's blog 2022):
  <https://kipp.ly/blog/transformer-inference-arithmetic/>
  Twenty-five minutes. The standard reference for back-of-envelope FLOPs and memory calculations on transformer inference. Out of scope for this week's mini-project but useful for Week 14 (performance).

## Datasets you might use beyond the mini-project

- **Project Gutenberg** (the same public-domain text repository Week 10 used):
  <https://www.gutenberg.org/>
  The mini-project uses `Pride and Prejudice` (<https://www.gutenberg.org/files/1342/1342-0.txt>) by default; alternates include `Frankenstein`, `Alice's Adventures in Wonderland`, and `Moby-Dick`. All public-domain in the US.
- **TinyShakespeare** (the 1.1 MB Shakespeare concatenation that nanoGPT uses by default):
  <https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt>
  The exact corpus from Karpathy 2015. About 1.1 MB. A drop-in alternative to `Pride and Prejudice` if you want to reproduce the nanoGPT defaults exactly.
- **WikiText-2 / WikiText-103** (Salesforce, 2017; the medium-scale language-modeling benchmark):
  <https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/>
  WikiText-2 is about 12 MB; WikiText-103 is about 500 MB. The C5 mini-project trains in 40 minutes on `Pride and Prejudice`; on WikiText-2 it would take 3-4 hours. Use only if you have a GPU.
- **enwik8** (the first 100 MB of Wikipedia; the canonical char-LM benchmark):
  <http://prize.hutter1.net/>
  100 MB. The dataset every char-LM transformer paper from 2018-2020 reported numbers on. Out of scope for a single-laptop week, but the right next dataset if you want to push the architecture.
- **OpenWebText** (an open reproduction of the GPT-2 training corpus; about 40 GB):
  <https://github.com/jcpeterson/openwebtext>
  Used by `nanoGPT` for the larger experiments in Karpathy's repository. Far out of scope for Week 11; included for reference.

## Voices worth following on this material (free; no signups)

- **Andrej Karpathy** — his 2023 video series "Neural Networks: Zero to Hero" (<https://karpathy.ai/zero-to-hero.html>) culminates in "Let's build GPT," the single most-influential transformer-implementation tutorial of the past five years. The nanoGPT repository is its production-quality companion. Follow him on YouTube and on the X-formerly-Twitter for occasional architectural commentary.
- **Anthropic's Transformer Circuits team** — their publications at <https://transformer-circuits.pub/> are the gold standard for *mechanistic interpretability* of transformer internals. The "residual stream" mental model in Lecture 2 originates there; the "induction heads" reading in Challenge 1 is from their 2022 paper. Updated quarterly.
- **Sasha Rush** — his "The Annotated Transformer" is the most-cited pedagogical walkthrough of the 2017 paper. His more recent work at Cornell on efficient transformer variants is also worth following.
- **Jay Alammar** — his blog at <https://jalammar.github.io/> is the canonical pictorial explanations site for sequence models. Beyond the Illustrated Transformer, see the Illustrated BERT, GPT-2, and Stable Diffusion.
- **Yannic Kilcher's YouTube** — <https://www.youtube.com/c/YannicKilcher>. Paper walk-through videos at the level of "I read this paper today and here is what it says." His 2017 video on the original Vaswani paper is a useful supplement to the C5 lecture notes.
- **Lilian Weng's blog at OpenAI** — <https://lilianweng.github.io/>. Her 2018 post "The Transformer Family" surveyed every architectural variant up to that point; her 2023 post on "LLM Powered Autonomous Agents" surveyed the post-2022 application landscape. Comprehensive and reliable.

## What to skip (or save for later)

- **The HuggingFace `transformers` library.** Excellent for production work, but it abstracts away every interesting detail of the architecture. Week 13 introduces it; this week the entire model is one Python file.
- **JAX / Flax implementations.** Conceptually identical to the PyTorch ones; the API differences are not architectural. If you are a JAX user, see <https://github.com/google/flax/tree/main/examples/lm1b> for the canonical Flax transformer; you can transliterate the C5 mini-project to JAX in an afternoon. We stay in PyTorch for the C5 curriculum.
- **The encoder-decoder architecture (the 2017 paper's full model).** Week 12. This week's mini-project is decoder-only.
- **RoPE and ALiBi positional encodings.** These are the modern (2021+) replacements for sinusoidal and learned positional embeddings. RoPE is what Llama and most 2024+ models use. Mentioned in the lecture notes; not on the exercise track this week. See Su et al. 2021 (<https://arxiv.org/abs/2104.09864>) for RoPE and Press et al. 2022 (<https://arxiv.org/abs/2108.12409>) for ALiBi.
- **Mixed precision (`torch.cuda.amp`) and `torch.compile`.** Both speed up transformer training by 1.5-3x on modern GPUs, but they obscure the forward and backward semantics. Week 14 covers performance.
- **Flash-attention as a separately-installed package** (`pip install flash-attn`). PyTorch 2.x's `F.scaled_dot_product_attention` dispatches to flash-attention internally on supported GPUs; we use only the PyTorch API.
- **BERT, T5, BART, and other masked / encoder-only / encoder-decoder variants.** Week 12 and Week 13. This week is decoder-only.
- **Reinforcement-learning-from-human-feedback (RLHF) and instruction tuning.** Week 15 and beyond.

## Citation cheatsheet (for the homework report)

If you cite any of the following in your mini-project report, use this format:

- **Transformer:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*. arXiv:1706.03762.
- **Layer normalization:** Ba, J.L., Kiros, J.R., and Hinton, G.E. (2016). Layer normalization. arXiv:1607.06450.
- **Pre-norm transformers:** Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T.-Y. (2020). On layer normalization in the transformer architecture. *ICML 2020*. arXiv:2002.04745.
- **Bahdanau attention:** Bahdanau, D., Cho, K., and Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*. arXiv:1409.0473.
- **GPT-1:** Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. (2018). Improving language understanding by generative pre-training. OpenAI technical report.
- **GPT-2:** Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI technical report.
- **nanoGPT:** Karpathy, A. (2023). nanoGPT [Computer software]. <https://github.com/karpathy/nanoGPT>.
- **Illustrated Transformer:** Alammar, J. (2018). The illustrated transformer [Blog post]. <https://jalammar.github.io/illustrated-transformer/>.
- **Residual stream framework:** Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., et al. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*. <https://transformer-circuits.pub/2021/framework/index.html>.

That is enough to anchor a portfolio write-up. The mini-project rubric does not require formal citations; it does require that any quantitative claim ("mini-GPT beats Week 10 LSTM by X nats per character") be backed by your own measured numbers.
