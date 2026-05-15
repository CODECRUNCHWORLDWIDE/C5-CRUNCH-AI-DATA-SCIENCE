# Week 11 — Quiz

Ten questions. Lectures closed. Aim for 9/10. The answer key is at the bottom; do not peek.

---

**Q1.** Scaled dot-product attention is defined as `Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V`. If `Q` has shape `(B, T, d_k)`, `K` has shape `(B, T, d_k)`, and `V` has shape `(B, T, d_v)`, what is the shape of the output?

- A) `(B, T, d_k)`
- B) `(B, T, d_v)`
- C) `(B, T, T)`
- D) `(B, T, d_k + d_v)`

---

**Q2.** Why is the scaling factor `1 / sqrt(d_k)` in scaled dot-product attention?

- A) To make the gradient flow through softmax better; without it the gradient vanishes.
- B) To keep the variance of the attention logits at 1.0 regardless of `d_k`. The dot product of two unit-variance `d_k`-dimensional vectors has variance `d_k`; dividing by `sqrt(d_k)` rescales it to 1. Without the scaling, the softmax saturates onto a single position for `d_k > ~32` and the gradient signal collapses.
- C) For numerical stability — it prevents the exponentials in softmax from overflowing.
- D) It is a hyperparameter chosen empirically; the value `sqrt(d_k)` happened to work best in the original ablations.

---

**Q3.** In multi-head attention with `d_model = 384` and `n_heads = 6`, what is the per-head dimension `d_k`?

- A) `384`
- B) `64`
- C) `6`
- D) `2304` (i.e., `384 * 6`)

---

**Q4.** Self-attention (without positional encoding) has the property `Attention(P @ X) = P @ Attention(X)` for any permutation matrix `P`. This property is called:

- A) Translation invariance — like a CNN.
- B) Permutation equivariance — shuffling the input shuffles the output the same way, so attention has no notion of input order. This is the architectural reason positional encoding must be added separately.
- C) Causality — the model cannot see future inputs.
- D) Idempotence — applying attention twice gives the same result as once.

---

**Q5.** Causal masking in self-attention sets which entries of the attention-logit matrix `S` to `-inf` before the softmax?

- A) The diagonal (`S[i, i] = -inf`).
- B) The lower-triangular entries (`S[i, j] = -inf` for `j < i`).
- C) The upper-triangular entries excluding the diagonal (`S[i, j] = -inf` for `j > i`).
- D) Random entries determined by a fixed seed.

---

**Q6.** The sinusoidal positional encoding from Vaswani 2017 has the algebraic property that `PE(pos + k)` is a fixed linear function of `PE(pos)`. The linear function is:

- A) An identity (`PE(pos + k) = PE(pos)` for all `k`).
- B) A scalar multiplication by `k`.
- C) A 2D rotation per frequency band — each `(2i, 2i+1)` pair of dimensions rotates by `omega_i * k` for the band's frequency `omega_i`. The full encoding is a block-diagonal rotation in `d_model / 2` separate 2D subspaces.
- D) A translation (additive shift) by a fixed vector that depends on `k`.

---

**Q7.** Modern transformers (GPT-2, GPT-3, Llama, the C5 mini-project) use *pre-norm* layer normalization: `x = x + sublayer(LayerNorm(x))`. The 2017 paper used *post-norm*: `x = LayerNorm(x + sublayer(x))`. Why did the field switch to pre-norm?

- A) Pre-norm uses less compute (LayerNorm before the sublayer is cheaper than after).
- B) Pre-norm produces lower validation loss at all model sizes.
- C) Pre-norm has a better-behaved gradient through deep stacks; the gradient through the residual path is `O(1)` rather than growing exponentially with depth. As a result, pre-norm transformers do not need learning-rate warmup and are more stable to train at large depths. Reference: Xiong et al. 2020.
- D) Post-norm was patented and pre-norm is an open-source alternative.

---

**Q8.** The position-wise feedforward MLP in a transformer block has the structure `Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)`. The conventional choice for `d_ff` is:

- A) `d_model` (same dimension).
- B) `4 * d_model` (the 2017 paper's choice; held up empirically in nearly every modern transformer).
- C) `d_model / 4` (a bottleneck).
- D) `d_model ** 2` (quadratic in model dimension).

---

**Q9.** The C5 mini-GPT ties the token-embedding weight `tok_embed.weight` to the output-projection weight `head.weight`. The two reasons are:

- A) Required by PyTorch — without weight tying, `nn.Embedding` and `nn.Linear` cannot share gradients.
- B) (i) Parameter savings — the tied matrix is shared instead of duplicated; (ii) marginally lower perplexity, as shown by Inan et al. 2017 and Press and Wolf 2017. Empirically about 1 perplexity point of improvement.
- C) Required for the model to be permutation-equivariant.
- D) It is a numerical-stability trick; the tied weight has better-conditioned gradients.

---

**Q10.** Lecture 3 named three architectural properties that the transformer has and the LSTM does not. Which of the following is NOT one of those three properties?

- A) Constant-time lookback: an attention layer can read any past position in one operation, vs. the LSTM's `O(T)` hidden-state propagation.
- B) Parallel training: the transformer's forward pass parallelizes across the entire sequence length, vs. the LSTM's sequential unroll.
- C) `O(1)` gradient path from output to input through the residual stream, vs. the LSTM's `O(T)` path through the cell-state recurrence.
- D) Inherent translation invariance: the transformer is robust to shifts in the input position, whereas the LSTM is not.

---

## Answer key

(Read after attempting.)

**Q1: B.** The output of `Attention(Q, K, V)` has the row dimension of `Q` (`T`) and the column dimension of `V` (`d_v`). The attention weights `softmax(Q K^T / sqrt(d_k))` have shape `(T, T)` but they are not the output — they are the intermediate probability matrix that gets multiplied by `V`. In self-attention, where `Q`, `K`, `V` all come from the same input, typically `d_v = d_k = d_model / n_heads` and the output shape is `(B, T, d_model / n_heads)` per head.

**Q2: B.** The variance argument from Lecture 1 Section 4. Without the scaling, the softmax saturates as `d_k` grows; the gradient through the saturated softmax is near-zero almost everywhere, and learning stalls. The scaling is the unique factor that keeps the softmax in its useful regime for any `d_k`. Option A is wrong: gradient flow is the *consequence*, not the cause. Option C is partially right but not the main reason. Option D is wrong — the scaling has a principled derivation, not an empirical one.

**Q3: B.** `d_k = d_model / n_heads = 384 / 6 = 64`. This is a common choice; `d_k = 64` per head matches GPT-2 small (`d_model = 768`, `n_heads = 12`, `d_k = 64`) and GPT-2 medium (`d_model = 1024`, `n_heads = 16`, `d_k = 64`). The convention `d_k = 64` per head is robust and seems to have been arrived at empirically. Option A is wrong (`d_k != d_model`); options C and D are obviously wrong.

**Q4: B.** Permutation equivariance. Shuffling the rows of the input shuffles the rows of the output correspondingly, with the content at each (now-shuffled) position unchanged. This is why positional encoding is necessary — without it, the transformer cannot distinguish "the dog bit the man" from "the man bit the dog." Option A is wrong (translation invariance is a CNN property and the wrong word for the wrong concept); option C is unrelated; option D is unrelated.

**Q5: C.** The upper-triangular entries strictly above the diagonal. Setting them to `-inf` and applying softmax produces a *lower-triangular* attention probability matrix: position `i` can attend to positions `0..i` (including itself), not to positions `i + 1..T - 1`. The diagonal stays finite — a token can always attend to itself. Option A is wrong (would prevent self-attention). Option B is the *opposite* of the correct mask. Option D is nonsense.

**Q6: C.** The 2D-rotation-per-frequency-band identity from Lecture 2 Section 2. Each `(2i, 2i+1)` pair of dimensions of the sinusoidal encoding transforms under a position shift of `k` as a 2D rotation by angle `omega_i * k`. Different frequency bands rotate by different amounts; the full `d_model`-dimensional encoding is a block-diagonal rotation. This algebraic property is the basis for Rotary Position Embedding (RoPE), which makes the rotation explicit and applies it inside attention instead of at the input.

**Q7: C.** The Xiong et al. 2020 gradient argument. Pre-norm transformers have `O(1)` gradient through the residual path; post-norm transformers have a gradient that grows multiplicatively with depth, which is why the 2017 paper used learning-rate warmup. Modern pre-norm transformers do not need warmup and train stably at 96+ layers. Option A is technically false (the compute is the same). Option B is too strong (the validation loss difference is small in-distribution). Option D is a joke.

**Q8: B.** `d_ff = 4 * d_model` is the 2017 paper's choice and is the empirical default in nanoGPT, GPT-2, GPT-3, BERT, T5, Llama, and basically every transformer published. The factor of 4 is suspiciously round but has not been displaced by any other ratio in seven years of architecture search. Some recent variants (e.g., GLU-based MLPs) use `d_ff ≈ 8/3 * d_model` to keep total parameters comparable, but the dense-MLP convention is firmly `4 * d_model`.

**Q9: B.** The two reasons are parameter savings and marginally lower perplexity. The references are Inan et al. 2017 (<https://arxiv.org/abs/1611.01462>) and Press and Wolf 2017 (<https://arxiv.org/abs/1608.05859>); both showed the improvement experimentally on word-LM tasks. nanoGPT, GPT-2, and the C5 mini-project all tie weights. Option A is false (the tying is *optional*; the framework does not require it). Options C and D are inventions.

**Q10: D.** "Translation invariance" is a CNN property and is *not* a transformer property. The transformer is permutation-*equivariant* (Q4), which is closer to the *opposite* of translation invariance. The three actual advantages (constant-time lookback, parallel training, `O(1)` gradient path) are stated correctly in A, B, and C. Lecture 1 Section 7 covered all three; Lecture 3 Section 12 reiterated them.
