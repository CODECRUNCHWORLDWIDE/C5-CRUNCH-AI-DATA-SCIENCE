"""
Exercise 3 -- Variable-length sequence batching with pack_padded_sequence.

Goal: build a custom collate_fn that pads a batch of variable-length integer
sequences and produces a lengths tensor. Then use pack_padded_sequence and
pad_packed_sequence correctly around an nn.LSTM call so the padding is not
processed as input. Verify the packed-path outputs match a hand-rolled
per-sequence baseline.

By the end of this exercise you will have:

    (1) Implemented a collate_pad function that takes a list of variable-length
        (input, target) integer-tensor pairs and returns (padded_inputs,
        padded_targets, lengths). Inputs are padded with 0; targets are
        padded with -100 so nn.CrossEntropyLoss ignores them.
    (2) Implemented run_packed_lstm that uses pack_padded_sequence around an
        nn.LSTM and returns the unpacked (batch, max_len, hidden_size) tensor.
    (3) Verified the packed-path outputs at the *real* (non-padded) positions
        match the outputs produced by running the LSTM on each sequence
        separately (no padding involved at all). They must agree to within
        1e-5 absolute error.
    (4) Demonstrated the bug that skipping the pack-step would cause: running
        the LSTM on the padded input directly, the hidden states at the
        padded positions are NOT zero -- they are the LSTM's response to the
        pad-token embedding, which silently corrupts the loss if the targets
        at padded positions are not also ignored.
    (5) Implemented a SimpleClassifier that takes the final (true) hidden
        state from the LSTM and projects it to class logits, for the
        sequence-classification use case.

Estimated time: 90-120 minutes (the pack/unpack API is famously fiddly).

Run with:    python exercise-03-pack-padded-sequence.py
Or test:     pytest exercise-03-pack-padded-sequence.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-03-pack-padded-sequence.py succeeds.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
    https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
"""

from __future__ import annotations

from typing import List, Tuple

# torch is imported lazily inside functions.


RANDOM_STATE: int = 42

PAD_INPUT: int = 0
PAD_TARGET: int = -100   # nn.CrossEntropyLoss's default ignore_index


# -----------------------------------------------------------------------------
# Part A -- collate_pad
# -----------------------------------------------------------------------------


def collate_pad(
    batch: "List[Tuple[torch.Tensor, torch.Tensor]]",
) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":  # type: ignore[type-arg]
    """Pad a list of (input, target) integer-tensor pairs.

    Each item in `batch` is a 2-tuple (input, target) where input and target
    are 1-D int64 tensors of the same length (the per-step token IDs and the
    per-step target IDs). Different items in the batch have different
    lengths.

    Returns:
        padded_inputs:  (batch, max_len) int64 tensor, padded with PAD_INPUT (0).
        padded_targets: (batch, max_len) int64 tensor, padded with PAD_TARGET (-100).
        lengths:        (batch,) int64 tensor with the true length of each
                        sequence (before padding).

    Use torch.nn.utils.rnn.pad_sequence; do not hand-pad with torch.cat.

    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = torch.tensor([t.shape[0] for t in inputs], dtype=torch.long)

    # TODO: call pad_sequence on `inputs` with batch_first=True and
    # padding_value=PAD_INPUT to produce padded_inputs. Do the same on
    # `targets` with padding_value=PAD_TARGET. Return the three tensors.
    raise NotImplementedError("Part A -- collate_pad")


# -----------------------------------------------------------------------------
# Part B -- packed-LSTM forward
# -----------------------------------------------------------------------------


def run_packed_lstm(
    lstm: "object",
    embed: "object",
    padded_inputs: "torch.Tensor",
    lengths: "torch.Tensor",
) -> "Tuple[torch.Tensor, torch.Tensor]":  # type: ignore[type-arg]
    """Run an LSTM over a padded batch using pack_padded_sequence.

    Arguments:
        lstm:  nn.LSTM with batch_first=True.
        embed: nn.Embedding from token IDs to embedding vectors.
        padded_inputs: (batch, max_len) int64 tensor of token IDs.
        lengths:       (batch,) int64 tensor of true sequence lengths.

    Returns:
        padded_output: (batch, max_len, hidden_size) tensor. At positions
                       beyond each sequence's true length, the values are 0.
        final_hidden: (batch, hidden_size) tensor. The hidden state h_T at
                      each sequence's true final position (NOT at the padded
                      max-len position).

    The packing requires `lengths` on CPU, even if the tensors are on GPU.
    Use enforce_sorted=False so the function reorders internally; this is
    the modern default.

    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
    """
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    # Step 1: embed the tokens. shape (batch, max_len, embed_dim).
    emb = embed(padded_inputs)

    # Step 2: pack. Note lengths must be on CPU.
    # TODO: packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
    # Step 3: run the LSTM. The LSTM returns (packed_output, (h_T, c_T)).
    # h_T has shape (num_layers, batch, hidden_size).
    # TODO: packed_output, (h_T, c_T) = lstm(packed)
    # Step 4: unpack to get the padded output back.
    # TODO: padded_output, _ = pad_packed_sequence(packed_output, batch_first=True)
    # Step 5: take the last layer's final hidden state. h_T[-1] is shape
    # (batch, hidden_size).
    # TODO: final_hidden = h_T[-1]
    # Step 6: return.
    raise NotImplementedError("Part B -- run_packed_lstm")


# -----------------------------------------------------------------------------
# Part C -- per-sequence reference (no padding involved)
# -----------------------------------------------------------------------------


def run_lstm_per_sequence(
    lstm: "object",
    embed: "object",
    inputs: "List[torch.Tensor]",
) -> "Tuple[List[torch.Tensor], torch.Tensor]":  # type: ignore[type-arg]
    """Run the LSTM on each sequence individually and stack the results.

    This is the slow, transparent baseline that the packed-path must
    reproduce at the real (non-padded) positions.

    Arguments:
        lstm:  nn.LSTM with batch_first=True.
        embed: nn.Embedding.
        inputs: List of 1-D int64 tensors of varying length.

    Returns:
        per_seq_outputs: List of (seq_len, hidden_size) tensors, one per
                         input sequence. No padding.
        final_hidden: (batch, hidden_size) tensor stacking each sequence's
                      final h_T.
    """
    import torch

    per_seq_outputs: List["torch.Tensor"] = []
    final_hiddens: List["torch.Tensor"] = []
    for seq in inputs:
        # Add a leading batch dimension; nn.LSTM with batch_first=True wants (1, seq_len, embed_dim).
        emb = embed(seq.unsqueeze(0))
        # TODO: call out, (h_T, c_T) = lstm(emb).
        # out: (1, seq_len, hidden_size). Squeeze the batch dim and append.
        # h_T: (num_layers, 1, hidden_size). Take h_T[-1, 0] and append.
        raise NotImplementedError("Part C -- run_lstm_per_sequence")

    final_hidden = torch.stack(final_hiddens, dim=0)
    return per_seq_outputs, final_hidden


# -----------------------------------------------------------------------------
# Part D -- a SimpleClassifier built on the packed LSTM
# -----------------------------------------------------------------------------


def build_simple_classifier(
    vocab_size: int,
    embed_dim: int,
    hidden_size: int,
    n_classes: int,
    num_layers: int = 1,
) -> "object":  # type: ignore[type-arg]
    """Build a sequence classifier: Embedding -> LSTM -> Linear over final hidden.

    This is the standard pattern for sequence classification (sentiment,
    language identification, etc.). The forward takes (padded_inputs, lengths)
    and returns (batch, n_classes) logits.

    Architecture:
        nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_INPUT)
        nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        nn.Linear(hidden_size, n_classes)

    Returns:
        A torch.nn.Module instance.
    """
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence

    class SimpleClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_INPUT)
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(
            self,
            padded_inputs: "torch.Tensor",
            lengths: "torch.Tensor",
        ) -> "torch.Tensor":
            # TODO: use run_packed_lstm to get (padded_output, final_hidden).
            # Apply self.head to final_hidden to produce logits of shape
            # (batch, n_classes). Return logits.
            raise NotImplementedError("Part D -- SimpleClassifier.forward")

    return SimpleClassifier()


# -----------------------------------------------------------------------------
# Part E -- main entry; sanity checks
# -----------------------------------------------------------------------------


def main() -> None:
    """Run sanity checks (no nontrivial verification against PyTorch references)."""
    import torch
    from torch import nn

    torch.manual_seed(RANDOM_STATE)

    # A: collate_pad
    seqs = [torch.tensor([1, 2, 3, 4], dtype=torch.long),
            torch.tensor([5, 6], dtype=torch.long),
            torch.tensor([7, 8, 9, 10, 11], dtype=torch.long)]
    targets = [torch.tensor([2, 3, 4, 5], dtype=torch.long),
               torch.tensor([6, 7], dtype=torch.long),
               torch.tensor([8, 9, 10, 11, 12], dtype=torch.long)]
    batch = list(zip(seqs, targets))
    padded_inputs, padded_targets, lengths = collate_pad(batch)
    assert padded_inputs.shape == (3, 5), f"padded_inputs shape: {tuple(padded_inputs.shape)}"
    assert padded_targets.shape == (3, 5), f"padded_targets shape: {tuple(padded_targets.shape)}"
    assert torch.equal(lengths, torch.tensor([4, 2, 5], dtype=torch.long)), f"lengths: {lengths}"
    # The padding values should be present.
    assert padded_inputs[1, 2].item() == PAD_INPUT, "input padding wrong"
    assert padded_targets[1, 2].item() == PAD_TARGET, "target padding wrong"

    # D: classifier forward, shape only.
    classifier = build_simple_classifier(
        vocab_size=20, embed_dim=8, hidden_size=16, n_classes=3, num_layers=1
    )
    logits = classifier(padded_inputs, lengths)
    assert logits.shape == (3, 3), f"classifier output shape: {tuple(logits.shape)}"

    print("OK -- exercise 3 (shape checks)")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def _import_torch() -> "Tuple[object, object]":  # type: ignore[type-arg]
    import torch
    from torch import nn

    return torch, nn


def test_collate_pad_basic() -> None:
    """collate_pad pads inputs with 0 and targets with -100."""
    torch, _ = _import_torch()
    seqs = [torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4], dtype=torch.long)]
    targets = [torch.tensor([2, 3, 4], dtype=torch.long),
               torch.tensor([5], dtype=torch.long)]
    batch = list(zip(seqs, targets))
    padded_inputs, padded_targets, lengths = collate_pad(batch)
    assert padded_inputs.shape == (2, 3)
    assert padded_targets.shape == (2, 3)
    assert lengths.tolist() == [3, 1]
    # Input padding at row 1, cols 1, 2 should be 0.
    assert padded_inputs[1, 1].item() == 0 and padded_inputs[1, 2].item() == 0
    # Target padding at row 1, cols 1, 2 should be -100.
    assert padded_targets[1, 1].item() == -100 and padded_targets[1, 2].item() == -100


def test_packed_lstm_matches_per_sequence() -> None:
    """Packed-path outputs at non-padded positions match per-sequence baseline."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    vocab_size, embed_dim, hidden_size = 20, 8, 16
    embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_INPUT)
    lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
    lstm.eval()
    embed.eval()

    # Three sequences of different lengths.
    seqs = [
        torch.tensor([3, 7, 11, 2], dtype=torch.long),
        torch.tensor([5, 9], dtype=torch.long),
        torch.tensor([1, 13, 4, 8, 6], dtype=torch.long),
    ]
    # Build dummy targets (not used here).
    targets = [s.clone() for s in seqs]
    batch = list(zip(seqs, targets))
    padded_inputs, _, lengths = collate_pad(batch)

    padded_output, final_hidden = run_packed_lstm(lstm, embed, padded_inputs, lengths)
    per_seq_outputs, per_seq_final = run_lstm_per_sequence(lstm, embed, seqs)

    # Verify every real position matches.
    for i, (length, ref_out) in enumerate(zip(lengths.tolist(), per_seq_outputs)):
        slice_packed = padded_output[i, :length, :]
        assert torch.allclose(slice_packed, ref_out, atol=1e-5), (
            f"sequence {i} (length {length}): packed-path output disagrees by "
            f"{(slice_packed - ref_out).abs().max().item():.3e}"
        )

    # And the final hidden states match.
    assert torch.allclose(final_hidden, per_seq_final, atol=1e-5), (
        f"final hidden states disagree by {(final_hidden - per_seq_final).abs().max().item():.3e}"
    )


def test_packed_output_zeros_at_padding() -> None:
    """pad_packed_sequence fills positions beyond each true length with zero."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)
    vocab_size, embed_dim, hidden_size = 20, 8, 16
    embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_INPUT)
    lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
    lstm.eval()
    embed.eval()

    seqs = [
        torch.tensor([3, 7, 11], dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
    ]
    targets = [s.clone() for s in seqs]
    batch = list(zip(seqs, targets))
    padded_inputs, _, lengths = collate_pad(batch)

    padded_output, _ = run_packed_lstm(lstm, embed, padded_inputs, lengths)
    # Row 1 (length 1) should have zeros at columns 1, 2.
    assert torch.allclose(padded_output[1, 1, :], torch.zeros(hidden_size), atol=1e-7), (
        "padded position should be zero from pad_packed_sequence"
    )
    assert torch.allclose(padded_output[1, 2, :], torch.zeros(hidden_size), atol=1e-7), (
        "padded position should be zero from pad_packed_sequence"
    )


def test_classifier_training_loss_decreases() -> None:
    """One Adam step on a tiny classification task decreases the loss."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    vocab_size, embed_dim, hidden_size, n_classes = 30, 8, 16, 3
    classifier = build_simple_classifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        n_classes=n_classes,
        num_layers=1,
    )
    # Synthetic data: random integer sequences with random labels.
    seqs = [torch.randint(1, vocab_size, (length,), dtype=torch.long)
            for length in [5, 3, 7, 4]]
    targets = [torch.randint(0, n_classes, (), dtype=torch.long) for _ in seqs]
    # Build a batch of (input, target=input.clone()) just to reuse collate_pad,
    # then override the class label below.
    dummy_targets = [s.clone() for s in seqs]
    batch = list(zip(seqs, dummy_targets))
    padded_inputs, _, lengths = collate_pad(batch)
    class_labels = torch.stack(targets, dim=0)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    classifier.train()
    initial_loss = loss_fn(classifier(padded_inputs, lengths), class_labels).item()
    for _ in range(20):
        optimizer.zero_grad()
        logits = classifier(padded_inputs, lengths)
        loss = loss_fn(logits, class_labels)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(classifier(padded_inputs, lengths), class_labels).item()
    assert final_loss < initial_loss * 0.8, (
        f"loss did not decrease enough: {initial_loss:.3f} -> {final_loss:.3f}"
    )


if __name__ == "__main__":
    main()
