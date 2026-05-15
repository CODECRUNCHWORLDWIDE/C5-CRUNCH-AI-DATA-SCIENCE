# Challenge 1 — Visualize What an Attention Head is Doing

> Train a small transformer on the same `Pride and Prejudice` corpus as Week 10 and the Week 11 mini-project. After training, extract the attention probability matrix from each head in each layer and render it as a heatmap. Identify at least one head that does something interpretable — a "previous-token head," a "punctuation-attending head," a "character-name-attending head" — and write 200 words describing what you found. This is the simplified version of the methodology Anthropic used in Olsson et al. 2022 (<https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>) to find induction heads in GPT-2.

**Estimated time:** 120-180 minutes (including the small training run).
**Deliverable:** A Python script `challenge-01-solution.py` plus a 200-word write-up `challenge-01-writeup.md` with at least three embedded heatmap images. Both committed to your portfolio repo under `week-11/challenges/`.

---

## The setup

The model you train for this challenge is small enough that the entire experiment fits in a 200-line script. Use these hyperparameters:

- `d_model = 128`
- `n_heads = 4` (so `d_k = 32` per head)
- `n_layers = 3`
- `max_seq_len = 128`
- `dropout = 0.0`

Total parameter count: about 600K. Train for 5 epochs on `Pride and Prejudice` (you can reuse the mini-project's data pipeline). Final validation loss should be around 1.8-2.0 nats per character — well above what the full-size mini-project achieves, but enough that the attention heads have learned something.

A smaller model is easier to interpret: fewer heads (4 per layer instead of 6), shallower stack (3 layers instead of 6), so only `4 * 3 = 12` heads total to inspect. The interpretability is the goal, not the perplexity.

---

## The extraction

To get the attention weights out of the model, you need to modify the `MultiHeadAttention` class to optionally return the weights. The standard trick:

```python
def forward(self, x, return_weights=False):
    # ... compute q, k, v ...
    if return_weights:
        # Compute attention manually to expose the weights.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply causal mask.
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        out = weights @ v
    else:
        # Use the fused kernel.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        weights = None
    # ... reshape and project ...
    return out, weights
```

Note that `F.scaled_dot_product_attention` does not expose the attention weights (the fused kernel computes them internally and discards them). So for this challenge you need the explicit `softmax` path.

After training, run the model on a short prompt (50-100 characters of text from the corpus) and collect the weights from each layer's attention. You will have `n_layers * n_heads = 12` matrices, each of shape `(T, T)`.

---

## Requirements

1. **Train the small transformer to validation loss under 2.1 nats per character.** This typically takes 5 epochs with the hyperparameters above. Print the per-epoch losses.
2. **Render at least three heatmaps.** Each heatmap is a `(T, T)` attention probability matrix from one head. Use `matplotlib.pyplot.imshow(weights, aspect="auto", cmap="viridis")` with the prompt characters as both x-axis and y-axis labels (so you can read which token attends to which).
3. **Identify and label at least one "interpretable" head.** Some examples of patterns to look for:
   - **Previous-token head**: high weight on the immediately-preceding token (`weights[i, i-1]` near 1.0). The heatmap looks like a sub-diagonal stripe.
   - **Self-attention head**: high weight on the current token (`weights[i, i]` near 1.0). The heatmap looks like a diagonal stripe.
   - **Punctuation-attending head**: high weight on tokens at the positions of `.`, `?`, `!`. The heatmap has vertical stripes at those columns.
   - **Beginning-of-sequence head**: high weight on position 0 regardless of the query. The heatmap has a strong leftmost column.
4. **Write the 200-word write-up.** Embed the three heatmaps. For each, name the head (e.g., "layer 1 head 2") and describe in one sentence what pattern you see. For at least one head, propose a hypothesis about what computational role the head is playing in language modeling.
5. **Use `torch.manual_seed(42)` at the top of the script.** The C5 convention.

---

## What you should find

In a 3-layer, 4-head transformer trained on `Pride and Prejudice` for 5 epochs, you typically find:

- **Layer 0**: heads tend to be low-rank and uninterpretable. Layer 0 is doing low-level character-statistics work — "are we in the middle of a word? at a vowel? after a capital letter?" — and the attention patterns reflect this with broad, diffuse weights.
- **Layer 1**: at least one head develops a clear "previous-token" pattern (sub-diagonal stripe). At least one head develops a "space-attending" pattern (vertical stripes at the positions of space characters). These are the simplest interpretable patterns and they appear reliably across training runs.
- **Layer 2**: heads become more specialized. You may see a head that strongly attends to the most-recent capital letter (useful for character names) or one that attends to the most-recent quotation mark (useful for tracking who is speaking in dialogue). Layer-2 patterns are noisier and depend on the random seed.

If you do not see at least one clearly-interpretable head, retrain with a different seed; the patterns are robust but the specific heads they appear in vary.

---

## Write-up requirements

In `challenge-01-writeup.md` (about 200 words), cover:

1. **The three embedded heatmaps**, each labeled with `[Layer X, Head Y]`.
2. **A one-sentence description** of each heatmap's pattern.
3. **A hypothesis** for at least one head: what role does this head play in the model's language-modeling task? Reference Olsson et al. 2022 (<https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>) if you find a head that looks like an induction head (this is rare in a 3-layer model but not impossible).
4. **Honest caveats.** State at least one limitation of your finding. Suggestions: "Only one random seed", "The pattern I named could equally well be doing something else", "Small models have noisier heads than large ones", "I only inspected one prompt; the pattern might not generalize."

---

## Hints

- Use a *short* prompt (50-80 characters). Heatmaps with `T > 100` become hard to read.
- Print the prompt as a string along the axes; matplotlib lets you set tick labels: `plt.xticks(range(T), list(prompt), fontsize=8); plt.yticks(range(T), list(prompt), fontsize=8)`.
- A good prompt to use: the opening line of the novel, "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife." (97 characters; a famous and varied piece of English text).
- After the softmax, the upper-triangular entries should all be zero. If you see nonzero entries above the diagonal, your causal mask is broken.
- The "previous-token head" pattern is the easiest to spot: a bright sub-diagonal stripe. If you see it in layer 0, that is unusual; most often it appears in layer 1.
- For the hypothesis: think about what computation a head with the observed pattern could be doing. A previous-token head, for example, is a useful primitive for any task where the model needs to "copy" or "shift" information forward by one position; combined with a head that copies the token at the position the previous-token head attended to, you get an induction head.

---

## Stretch (optional)

Train a slightly larger model (`n_layers = 6`, `n_heads = 6`, the mini-project's defaults) for 20 epochs. Look for an *induction head*: a layer-2-or-deeper head that, on a prompt of the form `[A] [B] ... [A] ?`, attends from the second `[A]` to the position of `[B]` (the token that followed the first `[A]`). Induction heads are the primary mechanism by which transformers implement in-context learning; Olsson et al. 2022 argues they are the most-important interpretable head type. If you find one, document it carefully — this is publication-quality interpretability work in a single afternoon.

---

## Acceptance criteria

- [ ] `challenge-01-solution.py` runs end-to-end (trains the small transformer, extracts attention weights, saves three heatmap PNGs).
- [ ] `python -m py_compile challenge-01-solution.py` succeeds.
- [ ] `challenge-01-writeup.md` is 150-250 words; embeds three heatmaps; describes each; proposes a hypothesis for at least one.
- [ ] Three heatmap PNGs are in the same directory.
- [ ] `torch.manual_seed(42)` is set at the top of the script.

The C5 conviction here: you have *seen* what an attention head is doing on your own machine. The "attention head" abstraction is no longer a slogan; it is a `(T, T)` matrix on your screen that you can read pixel by pixel. Once you have done this measurement once, you will read transformer-interpretability papers very differently.

---

## References

- **Olsson, C.; Elhage, N.; Nanda, N.; et al. (2022).** "In-context Learning and Induction Heads." *Transformer Circuits Thread.* <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>. The empirical demonstration of induction heads in GPT-2 small. The simplified version of this paper's methodology is what you are reproducing.
- **Elhage, N.; Nanda, N.; Olsson, C.; et al. (2021).** "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread.* <https://transformer-circuits.pub/2021/framework/index.html>. The introduction of the residual-stream view that Lecture 2 used.
- **Anthropic's TransformerLens library**: <https://github.com/TransformerLensOrg/TransformerLens>. The production-quality tool for attention-head interpretability. Out of scope for the challenge (this exercise asks you to roll your own) but worth bookmarking.
- **PyTorch's `imshow`**: <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>. Standard heatmap rendering.
- **`F.scaled_dot_product_attention`**: <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>. Note that this kernel does not return the attention weights; you must use the explicit `softmax` path for this challenge.
