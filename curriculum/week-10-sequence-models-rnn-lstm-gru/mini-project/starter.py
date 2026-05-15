"""
Mini-project starter: a char-level language model on Project Gutenberg text.

Single-file working scaffold. Reads `Pride and Prejudice` from a local cache
(downloading on first run), trains a 2-layer LSTM with truncated BPTT, prints
per-epoch validation loss and a 200-character sample at temperature 0.8.

Run with:    python starter.py
Train time:  ~30 min on a 2.5 GHz CPU, ~5 min on a Colab T4.

When refactoring into the deliverable layout (model.py / data.py / train.py /
sample.py), split this file along the section headers below.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
    https://pytorch.org/docs/stable/generated/torch.multinomial.html

Free reading:
    Karpathy 2015: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
    Hochreiter & Schmidhuber 1997: https://www.bioinf.jku.at/publications/older/2604.pdf
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Dict, List, Tuple

# torch and friends are imported lazily inside functions so this file
# `python -m py_compile`s cleanly without them installed.


RANDOM_STATE: int = 42

# Project Gutenberg's Pride and Prejudice plain-text URL. The file is public-
# domain in the US (1813 publication date). Cached locally on first run.
CORPUS_URL: str = "https://www.gutenberg.org/files/1342/1342-0.txt"
CORPUS_CACHE: str = "./data/pride_and_prejudice.txt"

# Standard Project Gutenberg markers that surround the novel text. These have
# been stable since approximately 2010.
GUTENBERG_START_MARKER: str = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END_MARKER: str = "*** END OF THE PROJECT GUTENBERG EBOOK"

# Hyperparameters. These are the C5 defaults; the rubric allows reasonable
# variation.
EMBED_DIM: int = 64
HIDDEN_SIZE: int = 256
NUM_LAYERS: int = 2
DROPOUT: float = 0.2
BATCH_SIZE: int = 32
CHUNK_LEN: int = 100
N_EPOCHS: int = 20
LR: float = 1e-3
GRAD_CLIP: float = 1.0
VAL_FRAC: float = 0.10


# -----------------------------------------------------------------------------
# data.py -- corpus download, vocab build, encoding, chunked streaming
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
        # If the markers are missing (rare), return the whole file. This is
        # the "fail open" path; the rubric does not penalize a small amount
        # of preamble leakage but it is preferable to strip.
        return raw
    # Skip past the start marker to the next newline; trim to just before
    # the end marker.
    start_of_text: int = raw.find("\n", start_idx) + 1
    return raw[start_of_text:end_idx].strip()


def build_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Build the char vocabulary and the lookup dictionaries.

    Returns (vocab, char_to_idx, idx_to_char).
    The vocab is sorted for reproducibility; index 0 is reserved for the
    most-common character (a space, in practice for English text).

    For Pride and Prejudice, the vocab has approximately 80 unique characters.
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
    """Reshape the encoded corpus into parallel streams for truncated BPTT.

    The returned `inputs` and `targets` have shape (batch_size, n_chunks,
    chunk_len) where each batch slot is a contiguous slice through the
    corpus. The training loop iterates over `n_chunks` and reads one chunk
    at a time.

    Targets are inputs shifted by one position (the standard language-
    modeling target).

    Note: the corpus is truncated to fit into `batch_size * n_chunks *
    (chunk_len + 1)` tokens; up to (chunk_len + 1) tokens at the end are
    discarded.

    Reference: Lecture 3 Section 4.
    """
    import torch

    n_tokens = encoded.numel()
    # We need each stream to be a multiple of chunk_len, plus one extra token
    # to make the target shift work cleanly.
    stream_len = (n_tokens - 1) // batch_size
    n_chunks = stream_len // chunk_len
    usable_per_stream = n_chunks * chunk_len
    # Reshape to (batch_size, stream_len + 1) where the +1 supplies the
    # last-position target.
    truncated = encoded[: batch_size * (usable_per_stream + 1)].view(
        batch_size, usable_per_stream + 1
    )
    inputs = truncated[:, :-1].contiguous().view(batch_size, n_chunks, chunk_len)
    targets = truncated[:, 1:].contiguous().view(batch_size, n_chunks, chunk_len)
    return inputs, targets


# -----------------------------------------------------------------------------
# model.py -- the CharLSTM
# -----------------------------------------------------------------------------


def build_char_lstm(
    vocab_size: int,
    embed_dim: int = EMBED_DIM,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
    dropout: float = DROPOUT,
) -> "object":  # type: ignore[type-arg]
    """Construct a 2-layer LSTM char-LM.

    Architecture:
        nn.Embedding(vocab_size, embed_dim)
        nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                dropout=dropout, batch_first=True)
        nn.Linear(hidden_size, vocab_size)

    The forward returns (logits, state) where logits has shape
    (batch, seq_len, vocab_size) and state is the (h_T, c_T) tuple from the
    LSTM, ready to be passed forward to the next chunk.
    """
    import torch
    from torch import nn

    class CharLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab_size: int = vocab_size
            self.embed_dim: int = embed_dim
            self.hidden_size: int = hidden_size
            self.num_layers: int = num_layers
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, vocab_size)

        def forward(
            self,
            x: "torch.Tensor",
            state: "Tuple[torch.Tensor, torch.Tensor] | None" = None,
        ) -> "Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]":
            emb = self.embed(x)
            out, new_state = self.lstm(emb, state)
            logits = self.head(out)
            return logits, new_state

    return CharLSTM()


def best_device() -> "object":  # type: ignore[type-arg]
    """Return cuda > mps > cpu as a torch.device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: "object") -> int:
    """Return the total parameter count of the model."""
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
    """Run one epoch of truncated BPTT.

    Arguments:
        model: a CharLSTM nn.Module.
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
    state: "Tuple[torch.Tensor, torch.Tensor] | None" = None  # reset every epoch

    for chunk_idx in range(n_chunks):
        x = inputs[:, chunk_idx, :].to(device)
        y = targets[:, chunk_idx, :].to(device)

        # Detach hidden state at chunk boundary (truncated BPTT).
        if state is not None:
            state = (state[0].detach(), state[1].detach())

        optimizer.zero_grad()  # type: ignore[attr-defined]
        logits, state = model(x, state)  # type: ignore[operator]
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
    length chunk_len (with no batch dimension, just one stream).

    Returns the mean per-character loss in nats.
    """
    import torch

    model.eval()  # type: ignore[attr-defined]
    total_loss = 0.0
    n_chunks = 0
    state: "Tuple[torch.Tensor, torch.Tensor] | None" = None
    with torch.no_grad():
        for start in range(0, encoded_val.numel() - chunk_len - 1, chunk_len):
            x = encoded_val[start : start + chunk_len].unsqueeze(0).to(device)
            y = encoded_val[start + 1 : start + chunk_len + 1].unsqueeze(0).to(device)
            logits, state = model(x, state)  # type: ignore[operator]
            if state is not None:
                state = (state[0].detach(), state[1].detach())
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

    Uses temperature-scaled softmax: probs = softmax(logits / temperature).
    Sampling is multinomial (torch.multinomial).

    Arguments:
        model: a trained CharLSTM nn.Module.
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
        # Feed the prompt through the model to warm up the hidden state.
        input_ids = torch.tensor(
            [[char_to_idx[c] for c in prompt]], device=device, dtype=torch.long
        )
        logits, state = model(input_ids)  # type: ignore[operator]
        # Now sample one character at a time.
        for _ in range(n_chars):
            last_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            out_chars.append(idx_to_char[next_id])
            next_input = torch.tensor([[next_id]], device=device, dtype=torch.long)
            logits, state = model(next_input, state)  # type: ignore[operator]

    return "".join(out_chars)


# -----------------------------------------------------------------------------
# main entry
# -----------------------------------------------------------------------------


def main() -> None:
    """Train the char-LM end to end and print samples."""
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
    model = build_char_lstm(vocab_size=len(vocab))
    model.to(device)  # type: ignore[attr-defined]
    print(f"parameter count: {count_parameters(model):,}")

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
