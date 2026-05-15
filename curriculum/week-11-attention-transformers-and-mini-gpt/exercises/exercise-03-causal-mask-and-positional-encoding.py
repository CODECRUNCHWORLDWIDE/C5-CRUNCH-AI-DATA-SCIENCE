"""
Exercise 3 -- Causal masks and positional encoding.

Goal: build the two pieces that the multi-head attention from Exercise 2
hides inside of itself or relies on as inputs. Specifically: implement the
lower-triangular causal mask explicitly, and implement both the sinusoidal
positional encoding (the 2017 paper's version) and the learned positional
encoding (the GPT line's version). Verify each with worked numerical
examples.

By the end of this exercise you will have:

    (1) Implemented `build_causal_mask(T, device)` that returns a (T, T)
        boolean mask whose upper-triangular entries are True and lower-
        triangular entries (including the diagonal) are False.
    (2) Implemented `apply_causal_mask_to_logits(scores, mask)` that
        replaces masked positions with -inf.
    (3) Implemented `sinusoidal_positional_encoding(seq_len, d_model)` from
        the Vaswani 2017 formula in Lecture 2 Section 2.
    (4) Verified the sinusoidal encoding has the "linear-relationship"
        property: PE(pos + k) is a fixed linear function of PE(pos).
    (5) Implemented `LearnedPositionalEncoding(max_seq_len, d_model)` as an
        nn.Module wrapping an nn.Embedding.
    (6) Verified that both encodings, when added to a random token
        embedding, produce position-distinguishable inputs (no two
        positions yield the same embedding).

Estimated time: 60 minutes.

Run with:    python exercise-03-causal-mask-and-positional-encoding.py
Or test:     pytest exercise-03-causal-mask-and-positional-encoding.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-03-causal-mask-and-positional-encoding.py
  succeeds.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.tril.html
    https://pytorch.org/docs/stable/generated/torch.triu.html
    https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

Free reading:
    Vaswani et al. 2017, section 3.5 (positional encoding):
        https://arxiv.org/abs/1706.03762
    Karpathy nanoGPT model.py, the GPT class for the learned version:
        https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from __future__ import annotations

import math
from typing import Tuple

# torch is imported lazily inside functions so this file compiles cleanly
# without it installed.


RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Part A -- causal mask construction
# -----------------------------------------------------------------------------


def build_causal_mask(
    T: int,
    device: "object" = None,
) -> "torch.Tensor":  # type: ignore[type-arg]
    """Return a (T, T) boolean mask for causal self-attention.

    The mask is True at positions (i, j) where j > i (the upper triangle,
    excluding the diagonal) and False elsewhere. When applied to an
    attention-logit matrix via masked_fill(mask, -inf), the result is the
    causal-attention pattern from Lecture 1 Section 6.

    Arguments:
        T: positive integer, the sequence length.
        device: optional torch.device for the output tensor. Defaults to
                CPU.

    Returns:
        A (T, T) bool tensor.

    Example:
        For T=4, the mask is:
            [[False, True,  True,  True],
             [False, False, True,  True],
             [False, False, False, True],
             [False, False, False, False]]
    """
    import torch

    assert T > 0, f"T must be positive, got {T}"
    if device is None:
        device = torch.device("cpu")
    # TODO: build the mask as torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1).
    # torch.triu with diagonal=1 selects strictly above the diagonal.
    # Return the mask.
    raise NotImplementedError("Part A -- build_causal_mask")


def apply_causal_mask_to_logits(
    scores: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":  # type: ignore[type-arg]
    """Apply the causal mask to attention-logit scores.

    Arguments:
        scores: shape (..., T, T) -- the attention logits Q K^T / sqrt(d_k).
        mask: shape (T, T) -- a boolean mask from build_causal_mask.

    Returns:
        A tensor with the same shape as scores, where every entry at a
        masked position has been replaced by -inf.

    Reference:
        Lecture 1 Section 6.
    """
    import torch

    assert scores.size(-1) == scores.size(-2), (
        f"expected square logits, got shape {tuple(scores.shape)}"
    )
    T = scores.size(-1)
    assert mask.shape == (T, T), (
        f"mask shape mismatch: expected ({T}, {T}), got {tuple(mask.shape)}"
    )
    # TODO: return scores.masked_fill(mask, float("-inf")).
    raise NotImplementedError("Part A -- apply_causal_mask_to_logits")


# -----------------------------------------------------------------------------
# Part B -- sinusoidal positional encoding (Vaswani 2017)
# -----------------------------------------------------------------------------


def sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
) -> "torch.Tensor":  # type: ignore[type-arg]
    """Compute the sinusoidal positional encoding from Vaswani 2017.

    The formula:
        PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))

    where pos ranges over 0..seq_len-1 and 2i ranges over 0..d_model-1.

    Arguments:
        seq_len: number of positions.
        d_model: encoding dimension. Must be even.

    Returns:
        A (seq_len, d_model) float32 tensor. Row pos is the encoding at
        position pos.

    Reference:
        Vaswani et al. 2017, section 3.5.
        Lecture 2 Section 2.
    """
    import torch

    assert seq_len > 0 and d_model > 0, (
        f"seq_len and d_model must be positive, got {seq_len}, {d_model}"
    )
    assert d_model % 2 == 0, f"d_model must be even, got {d_model}"

    # Build position indices (shape (seq_len, 1)) and frequency factors
    # (shape (1, d_model // 2)).
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    # The 2i values are 0, 2, 4, ..., d_model - 2. The divisor for each is
    # 10000^(2i / d_model), and we compute it as exp(-2i * log(10000) / d_model).
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32).unsqueeze(0)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    # TODO: pe[:, 0::2] = torch.sin(pos * div_term)
    # TODO: pe[:, 1::2] = torch.cos(pos * div_term)
    # Return pe.
    raise NotImplementedError("Part B -- sinusoidal_positional_encoding")


# -----------------------------------------------------------------------------
# Part C -- learned positional encoding (GPT-style)
# -----------------------------------------------------------------------------


def build_learned_positional_encoding(
    max_seq_len: int,
    d_model: int,
) -> "object":  # type: ignore[type-arg]
    """Construct a learned positional encoding nn.Module.

    The module wraps an nn.Embedding(max_seq_len, d_model). The forward
    takes a sequence length T (must be <= max_seq_len) and returns a
    (T, d_model) tensor of the first T position embeddings.

    Arguments:
        max_seq_len: maximum supported sequence length.
        d_model: encoding dimension.

    Returns:
        A torch.nn.Module with forward(seq_len: int) -> torch.Tensor of
        shape (seq_len, d_model).

    Reference:
        Lecture 2 Section 3.
        nanoGPT model.py: the wpe (word position embedding) attribute.
    """
    import torch
    from torch import nn

    class LearnedPositionalEncoding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_seq_len: int = max_seq_len
            self.d_model: int = d_model
            # TODO: register self.pos_embed = nn.Embedding(max_seq_len, d_model).
            raise NotImplementedError("Part C -- LearnedPositionalEncoding.__init__")

        def forward(self, seq_len: int) -> "torch.Tensor":
            assert seq_len <= self.max_seq_len, (
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
            # TODO: build positions = torch.arange(seq_len, device=self.pos_embed.weight.device).
            # TODO: return self.pos_embed(positions).
            raise NotImplementedError("Part C -- LearnedPositionalEncoding.forward")

    return LearnedPositionalEncoding()


# -----------------------------------------------------------------------------
# Part D -- main entry; sanity checks
# -----------------------------------------------------------------------------


def main() -> None:
    """Run shape and sanity checks."""
    import torch

    torch.manual_seed(RANDOM_STATE)

    # A: causal mask
    mask = build_causal_mask(T=5)
    assert mask.shape == (5, 5), f"mask shape wrong: got {tuple(mask.shape)}"
    assert mask.dtype == torch.bool, f"mask dtype wrong: got {mask.dtype}"
    # Upper triangle (excluding diagonal) should be True.
    expected_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.equal(mask, expected_mask), (
        f"mask does not match triu(ones, diagonal=1); got {mask.int()}"
    )

    # A: apply mask to a random scores tensor.
    scores = torch.randn(2, 3, 5, 5)
    masked = apply_causal_mask_to_logits(scores, mask)
    assert masked.shape == scores.shape, (
        f"masked shape mismatch: got {tuple(masked.shape)} vs {tuple(scores.shape)}"
    )
    # Upper-triangular entries should be -inf.
    upper_entries = masked[..., 0, 1:]  # row 0, columns 1..end
    assert torch.isinf(upper_entries).all() and (upper_entries < 0).all(), (
        f"expected -inf in upper triangle of row 0; got {upper_entries}"
    )
    # Diagonal should be finite.
    diag_entries = torch.stack([masked[..., i, i] for i in range(5)], dim=-1)
    assert torch.isfinite(diag_entries).all(), (
        "diagonal entries should not be masked"
    )

    # B: sinusoidal positional encoding
    pe = sinusoidal_positional_encoding(seq_len=10, d_model=8)
    assert pe.shape == (10, 8), f"pe shape wrong: got {tuple(pe.shape)}"
    # All entries should be in [-1, 1] (sin and cos are bounded).
    assert pe.abs().max().item() <= 1.0 + 1e-6, (
        f"pe entries should be in [-1, 1]; got max abs {pe.abs().max().item()}"
    )
    # Position 0: sin(0) = 0, cos(0) = 1. So pe[0] = [0, 1, 0, 1, ...].
    expected_row_0 = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert torch.allclose(pe[0], expected_row_0, atol=1e-6), (
        f"pe[0] should be [0, 1, 0, 1, ...]; got {pe[0]}"
    )

    # C: learned positional encoding
    lpe = build_learned_positional_encoding(max_seq_len=10, d_model=8)
    pos_out = lpe(seq_len=5)
    assert pos_out.shape == (5, 8), (
        f"learned pe shape wrong: got {tuple(pos_out.shape)}"
    )

    print("OK -- exercise 3 (shape and sanity checks)")


# -----------------------------------------------------------------------------
# Tests (require torch; will be skipped if torch is not installed)
# -----------------------------------------------------------------------------


def _import_torch() -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Lazy import of torch and torch.nn."""
    import torch
    from torch import nn

    return torch, nn


def test_causal_mask_shape_and_values() -> None:
    """The mask is True strictly above the diagonal."""
    torch, _ = _import_torch()
    mask = build_causal_mask(T=4)
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected), f"mask wrong: got\n{mask.int()}\nexpected\n{expected.int()}"


def test_apply_mask_to_logits_produces_lower_triangular_softmax() -> None:
    """After masking and softmax, row i has nonzero weight only at columns 0..i."""
    torch, _ = _import_torch()

    scores = torch.randn(1, 5, 5)
    mask = build_causal_mask(T=5)
    masked = apply_causal_mask_to_logits(scores, mask)
    weights = masked.softmax(dim=-1)

    # Lower-triangular weights should sum to 1; upper-triangular should be 0.
    for i in range(5):
        for j in range(5):
            if j > i:
                assert weights[0, i, j].item() == 0.0, (
                    f"weights[{i}][{j}] should be 0; got {weights[0, i, j].item()}"
                )
        # Row should sum to 1.
        assert abs(weights[0, i, :].sum().item() - 1.0) < 1e-5, (
            f"row {i} sums to {weights[0, i, :].sum().item()}, not 1"
        )


def test_sinusoidal_position_0_is_alternating_0_1() -> None:
    """At position 0, sin(0) = 0, cos(0) = 1. The encoding should alternate."""
    torch, _ = _import_torch()
    pe = sinusoidal_positional_encoding(seq_len=4, d_model=16)
    expected_row_0 = torch.tensor([0.0, 1.0] * 8)
    assert torch.allclose(pe[0], expected_row_0, atol=1e-6), (
        f"pe[0] wrong: got {pe[0]}"
    )


def test_sinusoidal_distinguishes_positions() -> None:
    """All positions should produce distinct encodings."""
    torch, _ = _import_torch()
    pe = sinusoidal_positional_encoding(seq_len=64, d_model=64)
    # Compute pairwise L2 distances; minimum should be > 0.
    min_dist = float("inf")
    for i in range(pe.size(0)):
        for j in range(i + 1, pe.size(0)):
            d = (pe[i] - pe[j]).norm().item()
            if d < min_dist:
                min_dist = d
    assert min_dist > 0.0, "two distinct positions produced identical encodings"


def test_sinusoidal_linear_relationship_property() -> None:
    """PE(pos + k) is a linear function of PE(pos) (Lecture 2 Section 2 identity).

    Specifically: for each pair of consecutive dimensions (2i, 2i+1), the
    map PE(pos) -> PE(pos + k) is a 2D rotation. We verify by checking
    that ||PE(pos)|| == ||PE(pos + k)|| for that pair of dimensions.
    """
    torch, _ = _import_torch()

    pe = sinusoidal_positional_encoding(seq_len=20, d_model=32)
    k = 5
    # For each pair (2i, 2i+1), the norm sqrt(sin^2 + cos^2) = 1 at every
    # position. So the norm of pe[pos][2i:2i+2] should equal pe[pos+k][2i:2i+2].
    for pos in range(15):
        for two_i in range(0, 32, 2):
            n1 = pe[pos, two_i : two_i + 2].norm().item()
            n2 = pe[pos + k, two_i : two_i + 2].norm().item()
            assert abs(n1 - n2) < 1e-5, (
                f"norm changed under translation: pos={pos}, 2i={two_i}, "
                f"{n1:.4f} vs {n2:.4f}"
            )


def test_learned_pe_shape() -> None:
    """forward(seq_len) returns (seq_len, d_model)."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    lpe = build_learned_positional_encoding(max_seq_len=128, d_model=64)
    for seq_len in (1, 16, 64, 128):
        out = lpe(seq_len=seq_len)
        assert out.shape == (seq_len, 64), (
            f"seq_len={seq_len}: shape mismatch, got {tuple(out.shape)}"
        )


def test_learned_pe_is_trainable() -> None:
    """The learned positional encoding has trainable parameters."""
    torch, _ = _import_torch()
    lpe = build_learned_positional_encoding(max_seq_len=16, d_model=8)
    params = list(lpe.parameters())
    assert len(params) == 1, f"expected 1 parameter tensor, got {len(params)}"
    assert params[0].shape == (16, 8), (
        f"expected (16, 8) parameter; got {tuple(params[0].shape)}"
    )
    assert params[0].requires_grad, "parameter should require grad"


def test_position_addition_breaks_permutation_equivariance() -> None:
    """Adding positional encoding to identical tokens produces distinct embeddings.

    This is the conceptual point of positional encoding (Lecture 2 Section 1):
    without it, attention is permutation-equivariant and cannot distinguish
    position. With it, identical tokens at different positions produce
    different residual-stream vectors.
    """
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    # 4 identical token embeddings.
    token_emb = torch.ones(4, 8)
    pe = sinusoidal_positional_encoding(seq_len=4, d_model=8)
    combined = token_emb + pe

    # All four combined embeddings should be distinct.
    for i in range(4):
        for j in range(i + 1, 4):
            diff = (combined[i] - combined[j]).norm().item()
            assert diff > 1e-3, (
                f"positions {i} and {j} should be distinguishable; got diff {diff:.3e}"
            )


if __name__ == "__main__":
    main()
