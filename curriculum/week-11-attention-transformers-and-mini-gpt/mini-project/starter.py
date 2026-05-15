"""
Mini-project starter: a nanoGPT-style char-level transformer.

Single-file working scaffold. Reads `Pride and Prejudice` from the same
local cache that the Week 10 LSTM mini-project used (downloading on first
run), trains a 6-layer pre-norm decoder-only transformer with multi-head
causal self-attention, prints per-epoch validation loss and a 200-character
sample at temperature 0.8 every 5 epochs.

Run with:    python starter.py
Train time:  ~30-50 min on a 2.5 GHz CPU, ~3-5 min on a Colab T4.

When refactoring into the deliverable layout (model.py / data.py / train.py /
sample.py), split this file along the section headers below.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
    https://pytorch.org/docs/stable/generated/torch.multinomial.html

Free reading:
    Vaswani et al. 2017: https://arxiv.org/abs/1706.03762
    Karpathy nanoGPT: https://github.com/karpathy/nanoGPT
    Karpathy "Let's build GPT" video: https://www.youtube.com/watch?v=kCc8FmEb1nY
    Alammar "The Illustrated Transformer": https://jalammar.github.io/illustrated-transformer/
"""

from __future__ import annotations

import math
import os
import time
import urllib.request
from typing import Dict, List, Tuple

# torch and friends are imported lazily inside functions so this file
# `python -m py_compile`s cleanly without them installed.


RANDOM_STATE: int = 42

# Project Gutenberg's Pride and Prejudice plain-text URL. The file is public-
# domain in the US (1813 publication date). Cached locally on first run; the
# cache path matches the Week 10 mini-project so the corpus is downloaded
# only once.
CORPUS_URL: str = "https://www.gutenberg.org/files/1342/1342-0.txt"
CORPUS_CACHE: str = "./data/pride_and_prejudice.txt"

# Standard Project Gutenberg markers that surround the novel text.
GUTENBERG_START_MARKER: str = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END_MARKER: str = "*** END OF THE PROJECT GUTENBERG EBOOK"

# Hyperparameters. These are the C5 defaults; the rubric allows reasonable
# variation. The values match the smallest nanoGPT variant.
D_MODEL: int = 384
N_HEADS: int = 6
N_LAYERS: int = 6
MAX_SEQ_LEN: int = 256
DROPOUT: float = 0.1
BATCH_SIZE: int = 32
CHUNK_LEN: int = 256
N_EPOCHS: int = 20
LR: float = 3e-4
GRAD_CLIP: float = 1.0
VAL_FRAC: float = 0.10


# -----------------------------------------------------------------------------
# data.py -- corpus download, vocab build, encoding, chunked streaming
# (verbatim from the Week 10 mini-project with a longer CHUNK_LEN default)
# -----------------------------------------------------------------------------


def download_corpus(url: str = CORPUS_URL, cache_path: str = CORPUS_CACHE) -> str:
    """Download the corpus to a local cache and return the raw text.

    On the first call, the corpus is fetched from `url`; subsequent calls
    read from `cache_path`. The function strips the Project Gutenberg
    license preamble and afterword before returning.
    """
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Downloading {url} to {cache_path} ...")
        with urllib.request.urlopen(url) as response:
            raw = response.read().decode("utf-8", errors="replace")
        with open(cache_path, "w", encoding="utf-8") as fp:
            fp.write(raw)
    else:
        with open(cache_path, "r", encoding="utf-8") as fp:
            raw = fp.read()
    return _strip_gutenberg_preamble(raw)


def _strip_gutenberg_preamble(raw: str) -> str:
    """Cut off the Project Gutenberg preamble and afterword around the novel."""
    start_idx: int = raw.find(GUTENBERG_START_MARKER)
    end_idx: int = raw.find(GUTENBERG_END_MARKER)
    if start_idx == -1 or end_idx == -1:
        return raw
    start_of_text: int = raw.find("\n", start_idx) + 1
    return raw[start_of_text:end_idx].strip()


def build_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Build the char vocabulary and the lookup dictionaries.

    Returns (vocab, char_to_idx, idx_to_char). The vocab is sorted for
    reproducibility. For Pride and Prejudice, the vocab has approximately
    80 unique characters.
    """
    vocab: List[str] = sorted(set(text))
    char_to_idx: Dict[str, int] = {c: i for i, c in enumerate(vocab)}
    idx_to_char: Dict[int, str] = {i: c for c, i in char_to_idx.items()}
    return vocab, char_to_idx, idx_to_char


def encode_text(text: str, char_to_idx: Dict[str, int]) -> "torch.Tensor":
    """Map the entire text to a 1-D int64 tensor of token IDs."""
    import torch

    return torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)


def make_chunked_streams(
    encoded: "torch.Tensor",
    batch_size: int = BATCH_SIZE,
    chunk_len: int = CHUNK_LEN,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Reshape the encoded corpus into parallel streams for transformer training.

    The returned `inputs` and `targets` have shape (batch_size, n_chunks,
    chunk_len) where each batch slot is a contiguous slice through the
    corpus. Targets are inputs shifted by one position. The training loop
    iterates over n_chunks and reads one chunk at a time.

    Unlike the Week 10 LSTM, the transformer has no hidden state to pass
    forward between chunks, so chunk boundaries are simpler -- each chunk
    is independent.
    """
    import torch

    n_tokens = encoded.numel()
    stream_len = (n_tokens - 1) // batch_size
    n_chunks = stream_len // chunk_len
    usable_per_stream = n_chunks * chunk_len
    truncated = encoded[: batch_size * (usable_per_stream + 1)].view(
        batch_size, usable_per_stream + 1
    )
    inputs = truncated[:, :-1].contiguous().view(batch_size, n_chunks, chunk_len)
    targets = truncated[:, 1:].contiguous().view(batch_size, n_chunks, chunk_len)
    return inputs, targets


# -----------------------------------------------------------------------------
# model.py -- the MultiHeadAttention, TransformerBlock, and GPTModel
# -----------------------------------------------------------------------------


def build_multihead_attention(
    d_model: int,
    n_heads: int,
    dropout: float = 0.0,
) -> "object":  # type: ignore[type-arg]
    """Construct a causal multi-head self-attention module.

    Uses the packed Q/K/V projection (one nn.Linear of output 3 * d_model)
    rather than three separate projections, matching nanoGPT's
    CausalSelfAttention. The fused F.scaled_dot_product_attention kernel is
    used with is_causal=True; on CUDA flash-attention will be selected
    automatically on supported GPUs.

    Returns an nn.Module with forward(x: torch.Tensor) -> torch.Tensor
    that takes (B, T, d_model) and returns (B, T, d_model).

    Reference:
        Lecture 1 Section 5.
        nanoGPT model.py: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    assert d_model % n_heads == 0, (
        f"d_model={d_model} must be divisible by n_heads={n_heads}"
    )

    class CausalSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d_model: int = d_model
            self.n_heads: int = n_heads
            self.d_k: int = d_model // n_heads
            self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)
            self.attn_dropout = nn.Dropout(dropout)
            self.resid_dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, T, _ = x.shape
            qkv = self.W_qkv(x)
            # Reshape to (B, T, 3, n_heads, d_k); permute to (3, B, n_heads, T, d_k).
            qkv = qkv.view(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            # F.scaled_dot_product_attention dispatches to flash-attention on
            # supported GPUs; on CPU it uses a math fallback.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout if self.training else 0.0, is_causal=True
            )
            out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
            return self.resid_dropout(self.W_o(out))

    return CausalSelfAttention()


def build_transformer_block(
    d_model: int,
    n_heads: int,
    dropout: float = 0.0,
) -> "object":  # type: ignore[type-arg]
    """Construct one pre-norm transformer block.

    The block has the architecture from Lecture 2 Section 8:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    The MLP is Linear(d_model, 4 * d_model) -> GELU -> Linear(4 * d_model, d_model)
    -> Dropout. This is the position-wise feedforward; it applies independently
    to each token's residual-stream vector.

    Returns an nn.Module with forward(x) -> output, same shape.
    """
    import torch
    from torch import nn

    class TransformerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln_1 = nn.LayerNorm(d_model)
            self.attn = build_multihead_attention(d_model, n_heads, dropout)
            self.ln_2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    return TransformerBlock()


def build_gpt_model(
    vocab_size: int,
    d_model: int = D_MODEL,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    max_seq_len: int = MAX_SEQ_LEN,
    dropout: float = DROPOUT,
) -> "object":  # type: ignore[type-arg]
    """Construct the full decoder-only transformer.

    Architecture (Lecture 3 Section 2):
        token_embed + pos_embed -> dropout
        -> n_layers transformer blocks
        -> final LayerNorm
        -> head (linear projection to vocab_size), weight-tied to token_embed.

    Forward(x: torch.Tensor) -> torch.Tensor:
        x has shape (batch, seq_len) int64.
        Returns (batch, seq_len, vocab_size) float logits.

    Reference:
        nanoGPT model.py GPT class.
    """
    import torch
    from torch import nn

    class GPTModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab_size: int = vocab_size
            self.max_seq_len: int = max_seq_len
            self.d_model: int = d_model

            self.tok_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
            self.drop = nn.Dropout(dropout)
            self.blocks = nn.ModuleList(
                [build_transformer_block(d_model, n_heads, dropout) for _ in range(n_layers)]
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            # Tie the token-embedding and output-projection weights (Lecture 2
            # Section 10). After this assignment, both modules share the same
            # underlying tensor; updates to one are reflected in the other.
            self.head.weight = self.tok_embed.weight

            # Initialize with the nanoGPT scheme: linear and embedding weights
            # are sampled from N(0, 0.02^2); biases (where present) are zero.
            self.apply(_init_weights)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, T = x.shape
            assert T <= self.max_seq_len, (
                f"sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
            positions = torch.arange(T, device=x.device).unsqueeze(0)
            x_emb = self.tok_embed(x) + self.pos_embed(positions)
            x_emb = self.drop(x_emb)
            for block in self.blocks:
                x_emb = block(x_emb)
            x_emb = self.ln_f(x_emb)
            return self.head(x_emb)

    return GPTModel()


def _init_weights(module: "object") -> None:  # type: ignore[type-arg]
    """Apply the nanoGPT initialization scheme to an nn.Module.

    Linear weights ~ N(0, 0.02^2); biases zero. Embedding weights ~ N(0, 0.02^2).
    This is the GPT-2 initialization and is what nanoGPT uses by default.
    """
    import torch
    from torch import nn

    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def best_device() -> "object":  # type: ignore[type-arg]
    """Return cuda > mps > cpu as a torch.device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: "object") -> int:
    """Return the total parameter count of the model.

    With weight tying, the tok_embed and head share weights, so we count
    once. PyTorch's `.parameters()` iterator does not double-count tied
    weights (they are the same Parameter object), so a plain sum is correct.
    """
    return sum(p.numel() for p in model.parameters())  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# train.py -- the training loop
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: "object",
    inputs: "torch.Tensor",
    targets: "torch.Tensor",
    optimizer: "object",
    loss_fn: "object",
    device: "object",
    grad_clip: float = GRAD_CLIP,
) -> float:
    """Run one epoch of transformer training.

    Unlike the Week 10 LSTM loop, there is no state to thread through
    chunks. Each chunk's forward pass is independent.

    Arguments:
        model: a GPTModel nn.Module.
        inputs: (batch_size, n_chunks, chunk_len) int64 tensor.
        targets: same shape; the input shifted by one position.
        optimizer: a torch.optim optimizer.
        loss_fn: nn.CrossEntropyLoss().
        device: torch.device.
        grad_clip: max norm for gradient clipping.

    Returns the mean per-character loss across the epoch.
    """
    import torch

    model.train()  # type: ignore[attr-defined]
    n_chunks = inputs.shape[1]
    total_loss = 0.0

    for chunk_idx in range(n_chunks):
        x = inputs[:, chunk_idx, :].to(device)
        y = targets[:, chunk_idx, :].to(device)

        optimizer.zero_grad()  # type: ignore[attr-defined]
        logits = model(x)  # type: ignore[operator]
        # logits: (batch, chunk_len, vocab_size); flatten for cross-entropy.
        loss = loss_fn(logits.flatten(0, 1), y.flatten())  # type: ignore[operator]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)  # type: ignore[attr-defined]
        optimizer.step()  # type: ignore[attr-defined]
        total_loss += loss.item()

    return total_loss / n_chunks


def validate(
    model: "object",
    encoded_val: "torch.Tensor",
    loss_fn: "object",
    device: "object",
    chunk_len: int = CHUNK_LEN,
) -> float:
    """Compute the mean per-character cross-entropy on the validation tensor.

    The validation tensor is a 1-D int64 tensor; we chunk it into pieces of
    length chunk_len.

    Returns the mean per-character loss in nats.
    """
    import torch

    model.eval()  # type: ignore[attr-defined]
    total_loss = 0.0
    n_chunks = 0
    with torch.no_grad():
        for start in range(0, encoded_val.numel() - chunk_len - 1, chunk_len):
            x = encoded_val[start : start + chunk_len].unsqueeze(0).to(device)
            y = encoded_val[start + 1 : start + chunk_len + 1].unsqueeze(0).to(device)
            logits = model(x)  # type: ignore[operator]
            loss = loss_fn(logits.flatten(0, 1), y.flatten())  # type: ignore[operator]
            total_loss += loss.item()
            n_chunks += 1
    return total_loss / max(n_chunks, 1)


# -----------------------------------------------------------------------------
# sample.py -- temperature-scaled generation
# -----------------------------------------------------------------------------


def sample(
    model: "object",
    prompt: str,
    n_chars: int,
    temperature: float,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    device: "object",
) -> str:
    """Sample n_chars characters from the model, conditioned on prompt.

    Uses temperature-scaled softmax and torch.multinomial sampling. Each
    generation step re-runs the (cropped) prefix through the model; this
    is O(T^2) total work for a T-character generation. The KV-cache
    optimization (Lecture 3 Section 9) replaces this with O(T) work but
    is not implemented here.

    Arguments:
        model: a trained GPTModel nn.Module.
        prompt: a string of seed characters; all must be in the vocabulary.
        n_chars: positive integer, the number of characters to generate.
        temperature: positive float. Lower is more deterministic.
        char_to_idx, idx_to_char: the vocabulary lookup dicts.
        device: torch.device.

    Returns the prompt followed by the generated characters.
    """
    import torch

    assert temperature > 0.0, f"temperature must be positive, got {temperature}"
    model.eval()  # type: ignore[attr-defined]
    out_chars: List[str] = list(prompt)

    with torch.no_grad():
        input_ids = torch.tensor(
            [[char_to_idx[c] for c in prompt]], device=device, dtype=torch.long
        )
        for _ in range(n_chars):
            # Crop to last max_seq_len tokens to respect the position-embedding bounds.
            max_seq_len = getattr(model, "max_seq_len", MAX_SEQ_LEN)
            input_cropped = input_ids[:, -max_seq_len:]
            logits = model(input_cropped)  # type: ignore[operator]
            last_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            out_chars.append(idx_to_char[next_id.item()])
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=-1)

    return "".join(out_chars)


# -----------------------------------------------------------------------------
# main entry
# -----------------------------------------------------------------------------


def main() -> None:
    """Train the mini-GPT end to end and print samples."""
    import torch
    from torch import nn

    torch.manual_seed(RANDOM_STATE)

    # Data
    print("== Data ==")
    text = download_corpus()
    print(f"corpus length: {len(text)} chars")
    vocab, char_to_idx, idx_to_char = build_vocab(text)
    print(f"vocab size: {len(vocab)}")

    encoded = encode_text(text, char_to_idx)
    n_val = int(VAL_FRAC * encoded.numel())
    encoded_train = encoded[: encoded.numel() - n_val]
    encoded_val = encoded[encoded.numel() - n_val :]
    print(f"train: {encoded_train.numel()} chars, val: {encoded_val.numel()} chars")

    inputs, targets = make_chunked_streams(encoded_train, BATCH_SIZE, CHUNK_LEN)
    print(f"chunked streams: {tuple(inputs.shape)} (batch, n_chunks, chunk_len)")

    # Model
    print("\n== Model ==")
    device = best_device()
    print(f"device: {device}")
    model = build_gpt_model(vocab_size=len(vocab))
    model.to(device)  # type: ignore[attr-defined]
    print(f"parameter count: {count_parameters(model):,}")

    # Note that nanoGPT uses AdamW with weight decay; for the C5 mini-project
    # plain Adam with no weight decay is sufficient and matches the Week 10
    # LSTM optimizer for the comparison.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # type: ignore[attr-defined]
    loss_fn = nn.CrossEntropyLoss()

    # Training
    print("\n== Training ==")
    train_losses: List[float] = []
    val_losses: List[float] = []
    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, inputs, targets, optimizer, loss_fn, device, GRAD_CLIP)
        val_loss = validate(model, encoded_val, loss_fn, device, CHUNK_LEN)
        elapsed = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"epoch {epoch:2d}/{N_EPOCHS} | "
            f"train {train_loss:.3f} nats/char | "
            f"val {val_loss:.3f} nats/char | "
            f"{elapsed:.1f}s"
        )
        # Mid-training sample at T=0.8.
        if epoch in (1, 5, 10, 15, 20):
            sample_text = sample(
                model,
                prompt="It is a truth universally acknowledged, ",
                n_chars=120,
                temperature=0.8,
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                device=device,
            )
            print(f"  [sample @ T=0.8] {sample_text!r}")

    # Save
    print("\n== Save ==")
    torch.save(model.state_dict(), "model.pt")  # type: ignore[attr-defined]
    print("saved model.pt")

    # Final samples at three temperatures.
    print("\n== Final samples ==")
    for temperature in (0.5, 0.8, 1.2):
        text_out = sample(
            model,
            prompt="It is a truth universally acknowledged, ",
            n_chars=200,
            temperature=temperature,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            device=device,
        )
        print(f"\n--- T = {temperature} ---\n{text_out}\n")

    print("done.")


if __name__ == "__main__":
    main()
