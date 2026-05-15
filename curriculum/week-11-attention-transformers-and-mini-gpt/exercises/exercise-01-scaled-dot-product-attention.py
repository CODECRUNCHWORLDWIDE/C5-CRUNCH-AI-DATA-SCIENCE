"""
Exercise 1 -- Implement scaled dot-product attention from scratch.

Goal: implement Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V from the
equation in Lecture 1 Section 2 as a pure-PyTorch function, then verify it
agrees with torch.nn.functional.scaled_dot_product_attention to within 1e-5
absolute error. Implement and verify the causal-mask variant. Demonstrate the
sqrt(d_k) scaling argument from Lecture 1 Section 4 by measuring softmax
saturation as d_k grows.

By the end of this exercise you will have:

    (1) Implemented Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V as a
        function `attention(q, k, v)` that handles 3-D and 4-D inputs.
    (2) Verified that your function agrees with PyTorch's
        F.scaled_dot_product_attention to within 1e-5 absolute error.
    (3) Implemented the causal-masked variant `causal_attention(q, k, v)`
        that sets the upper-triangular entries of the logits to -inf.
    (4) Verified that the causal version agrees with
        F.scaled_dot_product_attention(..., is_causal=True).
    (5) Implemented `softmax_saturation(d_k)` that measures the max attention
        weight on random Q, K. With the sqrt(d_k) scaling, the max should
        stay near 1/T regardless of d_k; without it, the max should approach
        1.0 as d_k grows. This is the empirical version of Lecture 1
        Section 4.

Estimated time: 60-90 minutes.

Run with:    python exercise-01-scaled-dot-product-attention.py
Or test:     pytest exercise-01-scaled-dot-product-attention.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-scaled-dot-product-attention.py succeeds.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    https://pytorch.org/docs/stable/generated/torch.softmax.html
    https://pytorch.org/docs/stable/generated/torch.tril.html

Free reading:
    Vaswani et al. 2017, equation 1: https://arxiv.org/abs/1706.03762
    Alammar 2018, "The Illustrated Transformer": https://jalammar.github.io/illustrated-transformer/
"""

from __future__ import annotations

import math
from typing import Tuple

# torch is imported lazily inside functions so this file compiles cleanly
# without it installed. The pytest functions do require torch.


RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Part A -- single-head scaled dot-product attention
# -----------------------------------------------------------------------------


def attention(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
) -> "torch.Tensor":  # type: ignore[type-arg]
    """Compute Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Arguments:
        q: shape (..., T_q, d_k) -- query matrix.
        k: shape (..., T_k, d_k) -- key matrix. Must have the same d_k as q.
        v: shape (..., T_k, d_v) -- value matrix. Same T_k as k.

    The leading dimensions (...) are broadcast and are typically
    (batch,) or (batch, n_heads). For single-head attention the shape is
    (batch, T, d_k); for multi-head it is (batch, n_heads, T, d_k).

    Returns:
        Attention output of shape (..., T_q, d_v).

    Steps:
        1. Compute scores = q @ k.transpose(-2, -1). Shape (..., T_q, T_k).
        2. Scale by 1.0 / sqrt(d_k).
        3. Apply softmax along the last dimension. Each row of the result is
           a probability distribution over keys.
        4. Multiply by v: output = weights @ v. Shape (..., T_q, d_v).

    Do not use F.scaled_dot_product_attention or any other PyTorch
    convenience function. Implement the steps above explicitly.

    Reference:
        Vaswani et al. 2017, equation 1.
    """
    import torch

    d_k = q.size(-1)
    # TODO: compute scores = q @ k.transpose(-2, -1), shape (..., T_q, T_k).
    # TODO: scale by 1.0 / math.sqrt(d_k).
    # TODO: apply softmax along the last dimension.
    # TODO: multiply by v: output = weights @ v.
    # TODO: return the output.
    raise NotImplementedError("Part A -- attention")


# -----------------------------------------------------------------------------
# Part B -- causal-masked attention
# -----------------------------------------------------------------------------


def causal_attention(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
) -> "torch.Tensor":  # type: ignore[type-arg]
    """Compute causal-masked attention.

    Same as attention(q, k, v) except that the upper-triangular entries of
    the attention scores are set to -inf before the softmax. After softmax,
    those entries have weight zero; the output at position i depends only on
    keys at positions 0..i.

    For causal attention to make sense, we require T_q == T_k. The mask is a
    lower-triangular T x T matrix; entry (i, j) is 0 if j <= i and -inf if
    j > i.

    Arguments:
        q: shape (..., T, d_k).
        k: shape (..., T, d_k).
        v: shape (..., T, d_v).

    Returns:
        Causal-attention output of shape (..., T, d_v).

    Reference:
        Vaswani et al. 2017, section 3.2.3 (the "masked" self-attention in
        the decoder).
    """
    import torch

    assert q.size(-2) == k.size(-2) == v.size(-2), (
        "causal attention requires T_q == T_k == T_v"
    )
    T = q.size(-2)
    d_k = q.size(-1)
    # TODO: compute scores = q @ k.transpose(-2, -1).
    # TODO: scale by 1.0 / math.sqrt(d_k).
    # TODO: build the upper-triangular mask:
    #         mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    #       Entries where mask is True are the positions to mask out.
    # TODO: apply scores = scores.masked_fill(mask, float("-inf")).
    # TODO: apply softmax along the last dimension.
    # TODO: multiply by v. Return.
    raise NotImplementedError("Part B -- causal_attention")


# -----------------------------------------------------------------------------
# Part C -- the sqrt(d_k) scaling argument, empirically
# -----------------------------------------------------------------------------


def softmax_saturation(
    d_k: int,
    seed: int = RANDOM_STATE,
    use_scale: bool = True,
) -> float:
    """Measure how saturated the attention softmax is as a function of d_k.

    Build random Q, K of shape (8, d_k) with iid standard-normal entries.
    Compute the attention weights (with or without the 1 / sqrt(d_k) scale).
    Return the average over rows of the maximum attention weight.

    For a softmax over 8 positions:
        - At uniform distribution, max weight = 1/8 = 0.125.
        - At fully saturated (argmax), max weight = 1.0.

    Empirically, without the scale, the max weight should approach 1.0 as
    d_k grows. With the scale, it should stay near 0.2-0.4.

    Arguments:
        d_k: the per-head dimension.
        seed: for reproducibility.
        use_scale: if True, divide by sqrt(d_k) before softmax.

    Returns:
        The average max-attention-weight across the 8 rows.

    Reference:
        Lecture 1 Section 4 -- the sqrt(d_k) variance argument.
    """
    import torch

    assert d_k > 0, f"d_k must be positive, got {d_k}"
    torch.manual_seed(seed)
    q = torch.randn(8, d_k)
    k = torch.randn(8, d_k)
    # TODO: compute scores = q @ k.T.
    # TODO: if use_scale is True, divide by math.sqrt(d_k).
    # TODO: apply softmax along the last dimension; call this `weights`.
    # TODO: take weights.max(dim=-1).values; return its mean as a Python float.
    raise NotImplementedError("Part C -- softmax_saturation")


# -----------------------------------------------------------------------------
# Part D -- main entry; sanity checks
# -----------------------------------------------------------------------------


def main() -> None:
    """Run a handful of shape and sanity checks."""
    import torch

    torch.manual_seed(RANDOM_STATE)

    # A: single-head attention shape check.
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 5, 16)
    v = torch.randn(2, 5, 16)
    out = attention(q, k, v)
    assert out.shape == (2, 5, 16), f"attention shape wrong: got {tuple(out.shape)}"

    # B: causal attention shape and lower-triangular check.
    causal_out = causal_attention(q, k, v)
    assert causal_out.shape == (2, 5, 16), (
        f"causal_attention shape wrong: got {tuple(causal_out.shape)}"
    )

    # B (lower-triangular): at the first position, the output should depend
    # only on V[0]. Test by perturbing V[1] and verifying causal_attention's
    # output at position 0 is unchanged.
    v_perturbed = v.clone()
    v_perturbed[:, 1, :] += 100.0
    causal_perturbed = causal_attention(q, k, v_perturbed)
    diff_at_0 = (causal_out[:, 0, :] - causal_perturbed[:, 0, :]).abs().max().item()
    assert diff_at_0 < 1e-5, (
        f"causal attention at position 0 should not depend on V[1]; "
        f"got max diff {diff_at_0:.3e}"
    )

    # C: softmax saturation.
    sat_small = softmax_saturation(d_k=4, use_scale=False)
    sat_large = softmax_saturation(d_k=256, use_scale=False)
    assert sat_large > sat_small, (
        f"without scaling, max weight should grow with d_k; "
        f"got {sat_small:.3f} (d_k=4) vs {sat_large:.3f} (d_k=256)"
    )
    sat_scaled_small = softmax_saturation(d_k=4, use_scale=True)
    sat_scaled_large = softmax_saturation(d_k=256, use_scale=True)
    # With scaling, both should be roughly the same.
    assert abs(sat_scaled_small - sat_scaled_large) < 0.2, (
        f"with scaling, max weight should be roughly d_k-independent; "
        f"got {sat_scaled_small:.3f} (d_k=4) vs {sat_scaled_large:.3f} (d_k=256)"
    )

    print("OK -- exercise 1 (shape and sanity checks)")


# -----------------------------------------------------------------------------
# Tests (require torch; will be skipped if torch is not installed)
# -----------------------------------------------------------------------------


def _import_torch() -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Lazy import of torch and torch.nn.functional.

    Returns (torch, F). Raises ImportError if torch is not installed;
    pytest treats that as a skip.
    """
    import torch
    from torch.nn import functional as F

    return torch, F


def test_attention_matches_pytorch() -> None:
    """Our attention matches F.scaled_dot_product_attention to within 1e-5."""
    torch, F = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    # Test on 3-D inputs (no head dimension).
    q = torch.randn(2, 6, 16)
    k = torch.randn(2, 6, 16)
    v = torch.randn(2, 6, 16)
    ours = attention(q, k, v)
    theirs = F.scaled_dot_product_attention(q, k, v)
    assert ours.shape == theirs.shape, (
        f"shape mismatch: ours {tuple(ours.shape)} vs theirs {tuple(theirs.shape)}"
    )
    assert torch.allclose(ours, theirs, atol=1e-5), (
        f"per-element max diff {(ours - theirs).abs().max().item():.3e}"
    )


def test_attention_matches_pytorch_4d() -> None:
    """Our attention handles 4-D (multi-head) inputs correctly."""
    torch, F = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    # (batch, n_heads, T, d_k)
    q = torch.randn(2, 4, 8, 32)
    k = torch.randn(2, 4, 8, 32)
    v = torch.randn(2, 4, 8, 32)
    ours = attention(q, k, v)
    theirs = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, theirs, atol=1e-5), (
        f"4-D max diff {(ours - theirs).abs().max().item():.3e}"
    )


def test_causal_attention_matches_pytorch() -> None:
    """Our causal_attention matches F.scaled_dot_product_attention(is_causal=True)."""
    torch, F = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    q = torch.randn(2, 6, 16)
    k = torch.randn(2, 6, 16)
    v = torch.randn(2, 6, 16)
    ours = causal_attention(q, k, v)
    theirs = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, theirs, atol=1e-5), (
        f"causal max diff {(ours - theirs).abs().max().item():.3e}"
    )


def test_causal_attention_is_strictly_causal() -> None:
    """Position i's output must not depend on positions j > i."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    q = torch.randn(1, 8, 16)
    k = torch.randn(1, 8, 16)
    v = torch.randn(1, 8, 16)
    out = causal_attention(q, k, v)

    # Perturb position 5 of V; positions 0..4 of the output must be unchanged.
    v_perturbed = v.clone()
    v_perturbed[:, 5, :] += 50.0
    out_perturbed = causal_attention(q, k, v_perturbed)
    # Positions 0..4 should agree.
    max_diff_early = (out[:, :5, :] - out_perturbed[:, :5, :]).abs().max().item()
    assert max_diff_early < 1e-5, (
        f"causal violation: positions 0..4 changed by up to {max_diff_early:.3e} "
        f"when V[5] was perturbed"
    )
    # Position 5 itself should change (the perturbation affects V[5]).
    max_diff_at_5 = (out[:, 5, :] - out_perturbed[:, 5, :]).abs().max().item()
    assert max_diff_at_5 > 0.0, "expected position 5's output to change"


def test_scaling_keeps_softmax_unsaturated() -> None:
    """With the sqrt(d_k) scaling, softmax stays unsaturated as d_k grows."""
    torch, _ = _import_torch()

    # Without scaling: max attention weight grows with d_k.
    sat_unscaled = [softmax_saturation(d_k=d, use_scale=False) for d in (4, 16, 64, 256)]
    assert sat_unscaled[-1] > sat_unscaled[0] * 1.5, (
        f"without scale, expected growth in max weight; got {sat_unscaled}"
    )

    # With scaling: max weight stays roughly constant.
    sat_scaled = [softmax_saturation(d_k=d, use_scale=True) for d in (4, 16, 64, 256)]
    spread = max(sat_scaled) - min(sat_scaled)
    assert spread < 0.2, (
        f"with scale, max weight should stay roughly constant; got spread {spread:.3f} ({sat_scaled})"
    )


def test_attention_weights_sum_to_one() -> None:
    """A useful invariant: the attention weights are probabilities."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    # We will compute attention weights manually inside the function for this test.
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 5, 16)
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = scores.softmax(dim=-1)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"attention weights do not sum to 1 along the last dim; max deviation "
        f"{(row_sums - 1.0).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    main()
