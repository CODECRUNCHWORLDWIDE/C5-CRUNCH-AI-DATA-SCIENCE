# Lecture 3 — Sequence Batching, Teacher Forcing, Truncated BPTT, and a Char-LM End to End

> **Outcome:** You can write a `collate_fn` that pads a list of variable-length integer sequences and produces a `lengths` tensor. You can apply `pack_padded_sequence` correctly around an `nn.LSTM` call so it skips the padding. You can explain teacher forcing in one sentence and name its weakness (exposure bias) in another. You can implement truncated BPTT by detaching the hidden state across chunks, and you can argue what is lost by doing so. You can stitch all of these pieces into a char-level language model that reads a Project Gutenberg `.txt`, trains for twenty minutes on CPU, and produces English-shaped samples at three temperatures. By the end of this lecture, the entire mini-project recipe is in your head; the remaining work is engineering, not understanding. The lecture closes with a tease — three things this whole apparatus cannot do — that points directly at Week 11.

Lecture 2 ended with a 20-line `CharLSTM` class. This lecture wires it into a training pipeline. We will work through four pieces, in order: variable-length sequence batching with `pack_padded_sequence`, teacher forcing, truncated BPTT, and temperature sampling. The pieces are independently useful; together they constitute the entire C5 char-LM recipe and most of the structure of the mini-project.

We target **PyTorch 2.x**; the primary references are `torch.nn.utils.rnn.pack_padded_sequence` (<https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html>), `torch.nn.utils.rnn.pad_packed_sequence` (<https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html>), and `torch.nn.CrossEntropyLoss` (<https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>). The pedagogical companion is Karpathy 2015 (<https://karpathy.github.io/2015/05/21/rnn-effectiveness/>), which is the spiritual ancestor of the mini-project.

---

## 1. Why sequences are awkward to batch

The image-batching problem of Week 9 was trivial: every CIFAR-10 image is 32×32; stack 32 of them into a `(32, 3, 32, 32)` tensor and call it a batch. Sequences are not so cooperative. The lengths of the sequences in a batch are different — three customer reviews in a sentiment-analysis batch might have lengths 14, 89, and 412 tokens.

You have two structural choices:

1. **Pad every sequence to the longest length in the batch.** Stack the padded sequences into a `(batch, max_len)` integer tensor. Wasteful: the short sequences spend most of their compute on the pad-token channel, which carries no information. PyTorch's `nn.LSTM` *will* process the padding correctly if you give it a `lengths` tensor and use `pack_padded_sequence`; without that tensor it will silently process the padding as if it were real input, producing garbage.
2. **Pad globally to the longest sequence in the *dataset*.** Even more wasteful. Don't.

The C5 convention this week: pad per-batch with a custom `collate_fn`, and use `pack_padded_sequence` to skip the padding inside the LSTM. Both steps are required; the collate without the pack is silently wrong.

A `collate_fn` is a function passed to `DataLoader` that takes a list of dataset items and returns a batched tensor. The default `collate_fn` stacks tensors of the same shape; for variable-length sequences you have to write your own. The standard pattern:

```python
def collate_pad(batch: "list[tuple[torch.Tensor, torch.Tensor]]") -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Pad a list of (input, target) sequence pairs to the max length in the batch.

    Returns:
        inputs:  (batch, max_len) integer tensor, padded with 0.
        targets: (batch, max_len) integer tensor, padded with -100 so CrossEntropyLoss ignores it.
        lengths: (batch,) integer tensor with the true length of each sequence.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.long)

    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=-100)
    return padded_inputs, padded_targets, lengths
```

The `pad_sequence` helper is at <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html>. The padding value `-100` for the targets is the magic constant that `nn.CrossEntropyLoss` interprets as "ignore this position"; the loss is computed only on the real (non-padded) positions. This is documented at <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html> under the `ignore_index` argument (default `-100`).

For the char-LM mini-project we use *fixed-length* chunks (every input is a length-100 slice of the text) so this padding machinery is not needed inside the project itself. But Exercise 3 builds it from scratch because every other sequence task needs it.

---

## 2. `pack_padded_sequence`, mechanically

Given a `(batch, max_len)` padded input and the per-sequence `lengths` tensor, `pack_padded_sequence` produces a `PackedSequence` object — a special data structure that PyTorch's recurrent layers understand and process efficiently by skipping padded positions.

The basic usage:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# inputs: (batch, max_len, embed_dim)   -- after nn.Embedding
# lengths: (batch,) long tensor          -- true sequence lengths
packed = pack_padded_sequence(
    inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
)
packed_output, hidden = lstm(packed)
padded_output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
# padded_output: (batch, max_len, hidden_size)
# output_lengths: (batch,) -- equals the input lengths
```

Three details worth knowing:

1. **`enforce_sorted=False`** is the modern default. Earlier versions of PyTorch required the batch to be sorted in decreasing order of length; `enforce_sorted=False` lifts that requirement and reorders internally. Use `enforce_sorted=False` unless you have a specific reason not to.
2. **`lengths.cpu()`** is required even on GPU. The packing routine itself is CPU-side; passing a CUDA tensor as `lengths` raises an error. The trailing `.cpu()` is the standard idiom.
3. **The final hidden state `hidden` is the same as the unpacked LSTM's final hidden state.** Specifically, `hidden[0]` is the final `h_T` at each batch element's *true* final position (not at the padded `max_len` position). This is the value you want to use as the sequence representation in a sequence-classification head.

The full docs are at <https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html>. Exercise 3 has you exercise all three details with assertions against a manually-implemented baseline.

> **EXPERIMENT — verify that packing is equivalent to padding.** Build a small LSTM (input 8, hidden 16, batch_first=True). Construct a batch of three sequences of lengths 5, 3, and 7, padded to length 7. Run the LSTM both ways: (a) packed via `pack_padded_sequence` then unpacked; (b) unpacked, on the padded input directly. Compare the per-timestep outputs at the *real* positions (ignoring the padded tail). They should agree to within 1e-6 absolute error. Repeat with the unpacked path passing through the padded positions — the hidden states at the padded positions are *not* zero; they are the LSTM's response to the pad token (which is typically the embedding of token 0). This is the bug that silently corrupts your loss if you skip the packing step.

---

## 3. Teacher forcing

For autoregressive sequence models — language models, machine translation decoders, music generation — the model emits one token at a time, and each emission is conditioned on the previous tokens. Two ways to wire the training:

- **Teacher forcing.** At each step `t`, feed the *true* previous token `x_t` from the training data as input. The model's previous prediction is ignored at training time. This decouples the per-step losses: each step's loss depends only on the model's prediction at that step, not on the entire history of predictions. Gradient flow is clean, training is fast, convergence is good.
- **Free-running (no teacher forcing).** At each step `t`, feed the model's *own previous prediction* `x_hat_{t-1}` as input. The model trains on its own outputs, which means a single bad prediction propagates forward and corrupts the rest of the sequence's loss. Hard to train; rarely used in practice; only relevant when you can't get away with teacher forcing.

For language modeling specifically, the "true previous token" is the token at position `t-1` in the source text. The model's input sequence is `text[0:T-1]` and the target sequence is `text[1:T]` — the targets are the inputs shifted by one position. PyTorch's `nn.CrossEntropyLoss` handles this naturally; the entire batch is processed in parallel because every step uses the true context.

The downside is **exposure bias**. At training time the model only ever sees ground-truth inputs; at inference time it sees its own (possibly erroneous) outputs. A single mistake at step 10 puts the model into a region of input space it has never been trained on, and the model can spiral into nonsense by step 100. The standard mitigation is **scheduled sampling** (Bengio et al. 2015, <https://arxiv.org/abs/1506.03099>): with probability `p`, feed the model's own prediction; with probability `1 - p`, feed the true token. `p` is annealed from 0 (full teacher forcing) toward 1 (full free-running) across training. The C5 mini-project uses full teacher forcing throughout because the corpus is small and the model is mostly used for next-step prediction (where the input is always the true context anyway); scheduled sampling is mentioned for completeness.

> **EXPERIMENT — what happens at inference.** Train the mini-project's char-LM for 5 epochs (enough to start producing English-shaped output). Then take a random 200-character passage from the training text. (a) Compute the per-step loss when feeding the true text. (b) Compute the per-step loss when feeding the model's own predictions at each step. The (b) loss is dramatically higher because the model's predictions drift into low-probability regions almost immediately. This *is* the exposure-bias problem, on your own machine, in five minutes.

---

## 4. Truncated BPTT

The recurrent gradient flows from the last step back to the first. For a language-model corpus of length 700,000 characters (the size of `Pride and Prejudice`), running BPTT across the entire corpus is impossible — the computation graph would not fit in any reasonable amount of memory.

The standard fix is to slice the corpus into chunks of length `T_chunk` (the C5 char-LM default is 100 characters), and run BPTT chunk-by-chunk. Between chunks, the hidden state is *passed forward* (so the model has access to context from earlier chunks) but the gradient is *detached* (so the optimizer only sees gradients within the current chunk).

The pattern, in PyTorch:

```python
hidden = None  # let nn.LSTM initialize to zero on the first chunk
for chunk_input, chunk_target in chunked_corpus:
    if hidden is not None:
        # Detach so backward only sees this chunk's graph.
        hidden = (hidden[0].detach(), hidden[1].detach())
    optimizer.zero_grad()
    logits, hidden = model(chunk_input, hidden)
    loss = loss_fn(logits.flatten(0, 1), chunk_target.flatten())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

The `.detach()` is the entire trick. It tells autograd "treat this tensor as a constant; do not trace gradients into the chunk that produced it." The model still gets the *values* of the previous hidden state (the recurrence is preserved across chunks for the forward pass), but the *gradient* path terminates at the chunk boundary.

The trade-off: the gradient can only flow back `T_chunk` steps. The model can learn dependencies up to `T_chunk` characters long, but no longer. For `T_chunk = 100`, that is roughly the length of a sentence — enough to learn intra-sentence grammar and word-level coherence, but not enough to learn paragraph-level coherence or chapter-level plot structure. Karpathy's 2015 essay discusses this trade-off at length; the C5 conviction is that for hobby-scale char-LM, `T_chunk = 100` is the right setting and the lack of long-range coherence is a feature ("the model produces English-looking gibberish, which is the cultural reference we are after"), not a bug.

Two production considerations not covered in detail:

1. **The chunk *batch* dimension.** For efficient training on GPU, you want the batch to contain *multiple parallel streams* through the corpus. The standard trick: reshape the corpus into a `(batch_size, n_chunks_per_stream, T_chunk)` tensor and iterate `n_chunks_per_stream` times, each iteration processing a `(batch_size, T_chunk)` slice. The hidden state is `(num_layers, batch_size, hidden_size)`; each batch slot has its own hidden state, persisted across chunks within its stream.
2. **Resetting hidden state at epoch boundaries.** At the start of each epoch, reset the hidden state to zero. The corpus has wrapped around; the previous epoch's final-chunk context is no longer relevant.

The mini-project starter implements both. The chunk batch dimension is the only piece of PyTorch this week that does not fit on a single screen; budget thirty minutes to read it carefully.

---

## 5. The char-LM forward pass and loss

Putting the previous sections together, the inner loop of the training script is six lines:

```python
# x:        (batch, seq_len)            -- int64 token IDs
# targets:  (batch, seq_len)            -- int64 token IDs (x shifted by one)
# hidden:   (h_0, c_0) or None
logits, hidden = model(x, hidden)        # logits: (batch, seq_len, vocab_size)
loss = loss_fn(logits.flatten(0, 1), targets.flatten())   # cross-entropy over vocab
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

The `logits.flatten(0, 1)` collapses the batch and time dimensions into one (so the loss treats every `(batch, time)` position as an independent classification problem); the `targets.flatten()` does the same for the targets. The loss is the average cross-entropy across all `batch * seq_len` positions.

The cross-entropy loss is at <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>. It applies `log_softmax` internally and then computes the negative log-likelihood; do not apply `softmax` to the model's outputs before the loss (a common bug).

The gradient clip is the standard `max_norm=1.0` from Lecture 1 Section 6. The Adam optimizer with `lr=1e-3` is the C5 default. Each step processes one `T_chunk * batch_size` slice of the corpus; one epoch processes `total_corpus_length / (T_chunk * batch_size)` steps.

---

## 6. Sampling with temperature

After training, the model can generate text autoregressively: feed it a starting character (a "prompt"), get the logits over the next character, sample, append, repeat. The standard generation loop:

```python
def sample(model: nn.Module, prompt: str, n_chars: int, temperature: float,
           char_to_idx: dict, idx_to_char: dict, device: torch.device) -> str:
    """Generate `n_chars` characters of text starting from `prompt`."""
    import torch
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([[char_to_idx[c] for c in prompt]], device=device, dtype=torch.long)
        logits, hidden = model(input_ids)
        out = list(prompt)
        for _ in range(n_chars):
            last_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            out.append(idx_to_char[next_id])
            next_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
            logits, hidden = model(next_input, hidden)
    return "".join(out)
```

Three points:

1. **`model.eval()`** disables dropout. The training-time dropout would corrupt the samples if left on.
2. **`with torch.no_grad():`** disables gradient tracking. Sampling does not need gradients; turning them off saves memory and a bit of time.
3. **`temperature`** scales the logits before softmax. `temperature=1.0` is plain sampling from the model's distribution. `temperature → 0` sharpens the distribution toward argmax (deterministic, often repetitive). `temperature → ∞` flattens toward uniform (random characters). The C5 default sweep is `{0.5, 0.8, 1.2}`:
   - **0.5**: conservative, repetitive, often gets stuck in loops of common words.
   - **0.8**: balanced; the C5 default for "show a portfolio reviewer."
   - **1.2**: creative, more nonsense, often produces inventive but malformed pseudo-words.

The mini-project rubric requires you to ship samples at all three temperatures so the reader can see the trade-off.

`torch.multinomial` is at <https://pytorch.org/docs/stable/generated/torch.multinomial.html>. The "trick" of dividing logits by temperature before softmax is the same trick that shows up in every transformer's `generate` method; we will reuse it next week.

> **EXPERIMENT — sample before training.** The mini-project starter has a `sample(model, prompt, n_chars=200, temperature=0.8)` call. Run it once before any training. The output will be uniform random characters because the embedding and the head are initialized from a unit-variance normal and the model's distribution is essentially uniform over the vocabulary. After one epoch of training, the output begins to look like character frequencies of English (lots of vowels and spaces). After ten epochs, you see word-shaped clumps separated by spaces. After twenty epochs, the words are mostly real English words and the sentences begin to be grammatical. Watching this transition is the single most satisfying experience in the C5 curriculum so far.

---

## 7. End-to-end mini-project recipe

The mini-project (`mini-project/starter.py`) is the full pipeline from Karpathy 2015 in PyTorch 2.x. The script:

1. Downloads `Pride and Prejudice` from Project Gutenberg if not cached locally.
2. Builds a char-to-index map: `vocab = sorted(set(text))`, then `char_to_idx = {c: i for i, c in enumerate(vocab)}`.
3. Encodes the entire text as a single `(N,)` integer tensor.
4. Reshapes the tensor into `(batch_size, n_chunks, T_chunk)` for parallel streaming.
5. Builds a 2-layer `CharLSTM` with `hidden_size=256`, `dropout=0.2`.
6. Trains for 20 epochs with `Adam(lr=1e-3)` and gradient clipping at `max_norm=1.0`.
7. After each epoch, samples 200 characters at temperature 0.8 and prints them. (The progression of these samples across epochs is the most pedagogically valuable artifact of the project.)
8. After training, saves the model and produces the three required temperature samples for the report.

Total training time: ~30 minutes on CPU, ~5 minutes on a free Google Colab T4 (the wall-clock dominates over hyperparameter tuning, so iterate on Colab if you can).

Expected results:

- **Validation loss** (held-out 10% of the corpus): around 1.4 nats per character after 20 epochs. (For reference, English text has a Shannon entropy of approximately 1.0-1.5 bits per character with an optimal model; the LSTM is in the right zone.)
- **Sample quality at T=0.8** after 20 epochs: 80% of the words are real English words, 60% of the sentences are grammatically plausible, 0% of the sentences make any sense as fiction. This is the expected level for a 1.5M-parameter character-level model on a 700 KB corpus.

---

## 8. The honest tease toward attention

Everything we have built this week — the LSTM, the truncated BPTT, the teacher forcing, the temperature sampling — was the state of the art in sequence modeling from approximately 2014 to 2017. Three years.

In 2017, Vaswani et al. published "Attention Is All You Need" (<https://arxiv.org/abs/1706.03762>) and changed the architecture entirely. The new architecture, the **transformer**, replaces recurrence with **self-attention**: at each output position, the model computes a weighted average over *all* input positions, with the weights determined by a softmax of dot products between learned query and key vectors. There is no hidden state passed forward in time. There is no truncated BPTT. There is no sequential dependency between timesteps.

Three things the transformer does that the LSTM does not:

1. **Constant-time lookback.** An LSTM accessing information from 500 steps ago must propagate that information through 500 hidden-state updates. A transformer's attention layer can read it in *one* operation — a single softmax over all 500 positions, with the relevant position weighted high.
2. **Parallel training.** An LSTM unrolls sequentially over the sequence length; the GPU is mostly idle during a single forward pass because each timestep depends on the previous. A transformer's forward pass parallelizes over the *entire* sequence; on a GPU, a 1024-step transformer forward is dramatically faster than a 1024-step LSTM forward.
3. **No gradient vanishing through depth in time.** The transformer's gradient path from output to input is `O(1)` — there is a single softmax-weighted lookup. The LSTM's gradient path is `O(T)` through the cell state (which, as Lecture 2 Section 4 argued, is only saved from vanishing by the forget gate being near 1). The transformer simply does not have this problem.

These three properties — and a fourth, the ability to "look at" any position from any other position without architectural bias toward locality — are why the transformer ate everything in sequence modeling between 2017 and 2020 and shows no signs of being displaced in 2026. Every large language model you have heard of (GPT-3, GPT-4, Claude, Gemini, LLaMA, etc.) is a transformer. The LSTM is a respected pre-2017 architecture; the transformer is the post-2017 architecture.

We will build a transformer from scratch in Week 11. We will reuse the data pipeline you built this week — the Project Gutenberg loader, the chunked dataset, the temperature sampler — and replace only the model class. The result will be a character-level transformer that trains in the same time as the LSTM and produces visibly more coherent samples on the same corpus. The headline number of Week 11 is exactly that: held-out loss on Pride and Prejudice, LSTM vs. transformer, same compute budget.

Next week. Save your trained LSTM; you will compare against it.

---

## 8b. A short detour: why we don't use a transformer this week

A reasonable question: if the transformer is going to win next week, why bother building an LSTM at all? Three reasons.

**Pedagogical.** The transformer's attention layer is the architectural answer to a specific problem — the failure modes of recurrence. To appreciate the answer, you need to have felt the problem first. The whole point of this week is to put you in front of those failure modes (vanishing gradients, sequential compute, truncated BPTT) so that next week's introduction of attention lands as a *fix*, not as a parachute drop from a different universe. Students who skip the recurrent week and start with attention often misunderstand why attention is structured the way it is.

**Historical.** Between 2014 and 2017, every major NLP system was LSTM-based: Google's neural machine translation, Apple's Siri, Amazon's Alexa, Facebook's early language models, every academic paper that beat a previous benchmark. The LSTM was the workhorse architecture for three years, which is short by some measures and an eternity by others. Knowing why the LSTM was the workhorse — and what was difficult about it — is part of literacy in the field.

**Practical.** LSTMs still ship in production. Edge devices that cannot afford the `O(T^2)` memory cost of attention use recurrent models. Time-series forecasting (energy prices, weather, stock movements) routinely uses LSTM-based architectures because the data is genuinely sequential and the sequences are short enough that the recurrent inductive bias is useful. Speech recognition pipelines that need streaming inference (latency-sensitive applications) use LSTMs or their variants because attention requires the full sequence to be available; an LSTM can emit predictions as the audio streams in. The transformer ate language modeling but did not eat every sequence-modeling problem.

That said: in 2026, if you have the choice between an LSTM and a transformer for a new project with enough data and compute, default to the transformer. The LSTM is a respected predecessor; the transformer is the modern default. This week is about the predecessor; next week is about the default.

---

## 9. Recap

You can now:

- Write a `collate_fn` that pads variable-length sequences and returns a `lengths` tensor.
- Use `pack_padded_sequence` and `pad_packed_sequence` correctly around an `nn.LSTM` call.
- Explain teacher forcing and exposure bias in two sentences each.
- Implement truncated BPTT by detaching the hidden state at chunk boundaries.
- Sample from a trained LSTM with temperature scaling and explain the three-point sweep `T ∈ {0.5, 0.8, 1.2}`.
- Recite the three properties that the transformer has and the LSTM does not, in preparation for Week 11.

The exercises and the mini-project are the test of all of this. Open the starter file, run it once with the defaults, watch a 2-layer LSTM learn English on your laptop in twenty minutes. That is the deliverable.
