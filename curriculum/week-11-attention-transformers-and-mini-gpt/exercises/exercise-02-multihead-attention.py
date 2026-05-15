"""
Exercise 2 -- Implement multi-head attention with explicit Q/K/V projections.

Goal: implement multi-head attention as an nn.Module with explicit, separate
Q, K, V projection layers (one nn.Linear each, not the packed three-in-one),
plus a final output projection. Then verify the module agrees with
torch.nn.MultiheadAttention (when given identical weights) to within 1e-5.

The pedagogical point: the production-style "packed Q, K, V into one
nn.Linear" trick is faster but obscures the math. This exercise has you
implement the verbose, three-projection version so the algebra in Lecture 1
Section 5 is concrete; the mini-project at the end of the week uses the
packed version.

By the end of this exercise you will have:

    (1) Implemented MultiHeadAttention with three separate nn.Linear
        projections for Q, K, V (each d_model -> d_model) and one output
        projection (d_model -> d_model).
    (2) Implemented head splitting: reshape (B, T, d_model) ->
        (B, T, n_heads, d_k) and transpose to (B, n_heads, T, d_k) before
        attention; reverse after.
    (3) Verified the module agrees with nn.MultiheadAttention (with copied
        weights) on random inputs to within 1e-5.
    (4) Verified the causal-masked variant agrees with
        nn.MultiheadAttention(... attn_mask=causal_mask).
    (5) Verified that the parameter count matches the formula
        4 * d_model^2 (no biases on the projections; the C5 default).

Estimated time: 90-120 minutes. This is the longest exercise of the week.

Run with:    python exercise-02-multihead-attention.py
Or test:     pytest exercise-02-multihead-attention.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-multihead-attention.py succeeds.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

Free reading:
    Vaswani et al. 2017, section 3.2.2 (multi-head attention):
        https://arxiv.org/abs/1706.03762
    Karpathy nanoGPT, the CausalSelfAttention class:
        https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from __future__ import annotations

import math
from typing import Tuple

# torch is imported lazily inside functions so this file compiles cleanly
# without it installed.


RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Part A -- the MultiHeadAttention nn.Module
# -----------------------------------------------------------------------------


def build_multihead_attention(
    d_model: int,
    n_heads: int,
    causal: bool = True,
) -> "object":  # type: ignore[type-arg]
    """Construct a MultiHeadAttention nn.Module with the verbose API.

    The module has:
        - self.W_q: nn.Linear(d_model, d_model, bias=False)
        - self.W_k: nn.Linear(d_model, d_model, bias=False)
        - self.W_v: nn.Linear(d_model, d_model, bias=False)
        - self.W_o: nn.Linear(d_model, d_model, bias=False)
        - self.d_k = d_model // n_heads
        - self.causal = causal

    Forward signature: forward(x: torch.Tensor) -> torch.Tensor
    where x has shape (batch, seq_len, d_model). Returns shape
    (batch, seq_len, d_model).

    The forward steps are:
        1. q = self.W_q(x); k = self.W_k(x); v = self.W_v(x).
           Each has shape (B, T, d_model).
        2. Reshape each to (B, T, n_heads, d_k) and transpose to
           (B, n_heads, T, d_k).
        3. Apply scaled dot-product attention (with causal mask if
           self.causal is True). The output is (B, n_heads, T, d_k).
        4. Transpose back to (B, T, n_heads, d_k) and reshape to
           (B, T, d_model).
        5. Apply self.W_o to get the final output (B, T, d_model).

    Use F.scaled_dot_product_attention for step 3 (it is faster and avoids
    re-implementing what Exercise 1 already covered). Pass is_causal=causal.

    Arguments:
        d_model: total embedding dimension.
        n_heads: number of attention heads. Must divide d_model.
        causal: if True, apply the lower-triangular causal mask.

    Returns:
        A torch.nn.Module instance.

    Reference:
        Vaswani et al. 2017, section 3.2.2.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    assert d_model % n_heads == 0, (
        f"d_model={d_model} must be divisible by n_heads={n_heads}"
    )

    class MultiHeadAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d_model: int = d_model
            self.n_heads: int = n_heads
            self.d_k: int = d_model // n_heads
            self.causal: bool = causal
            # TODO: register four nn.Linear(d_model, d_model, bias=False)
            # modules as self.W_q, self.W_k, self.W_v, self.W_o.
            raise NotImplementedError("Part A -- MultiHeadAttention.__init__")

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B, T, _ = x.shape
            # TODO: compute q = self.W_q(x), k = self.W_k(x), v = self.W_v(x).
            # Each has shape (B, T, d_model).
            #
            # TODO: reshape each to (B, T, n_heads, d_k):
            #     q = q.view(B, T, self.n_heads, self.d_k)
            # then transpose to (B, n_heads, T, d_k):
            #     q = q.transpose(1, 2)
            # Same for k and v.
            #
            # TODO: apply attention:
            #     out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
            # Out has shape (B, n_heads, T, d_k).
            #
            # TODO: transpose back to (B, T, n_heads, d_k) and reshape to
            # (B, T, d_model):
            #     out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
            #
            # TODO: apply self.W_o(out) and return.
            raise NotImplementedError("Part A -- MultiHeadAttention.forward")

    return MultiHeadAttention()


# -----------------------------------------------------------------------------
# Part B -- copy weights from our module into nn.MultiheadAttention
# -----------------------------------------------------------------------------


def copy_weights_into_pytorch_mha(
    ours: "object",
    theirs: "object",
) -> None:
    """Copy our four projections' weights into PyTorch's nn.MultiheadAttention.

    nn.MultiheadAttention packs Q, K, V into a single in_proj_weight tensor
    of shape (3 * embed_dim, embed_dim), with rows ordered as
    [Q-rows; K-rows; V-rows]. The output projection is theirs.out_proj.

    Arguments:
        ours: a MultiHeadAttention instance from build_multihead_attention.
        theirs: a torch.nn.MultiheadAttention instance with the same d_model
                and n_heads, batch_first=True, bias=False.
    """
    import torch

    # Concatenate our [W_q.weight; W_k.weight; W_v.weight] into the packed
    # in_proj_weight tensor.
    with torch.no_grad():
        # TODO: build in_proj_weight as torch.cat([ours.W_q.weight, ours.W_k.weight, ours.W_v.weight], dim=0).
        # The shape should be (3 * d_model, d_model).
        # TODO: theirs.in_proj_weight.copy_(in_proj_weight).
        # TODO: theirs.out_proj.weight.copy_(ours.W_o.weight).
        # nn.MultiheadAttention with bias=False has no bias tensors to copy.
        raise NotImplementedError("Part B -- copy_weights_into_pytorch_mha")


# -----------------------------------------------------------------------------
# Part C -- parameter counting
# -----------------------------------------------------------------------------


def count_attention_parameters(d_model: int) -> int:
    """Return the parameter count of MultiHeadAttention(d_model, ...).

    With no biases, the count is 4 * d_model^2:
        W_q: d_model * d_model
        W_k: d_model * d_model
        W_v: d_model * d_model
        W_o: d_model * d_model

    Note that this is independent of n_heads -- the per-head dimension
    is d_model / n_heads, and there are n_heads heads, so the total
    parameter count of the Q (or K or V) projection is always n_heads *
    (d_model * d_k) = n_heads * (d_model * d_model / n_heads) = d_model^2.

    Reference:
        Lecture 1 Section 5 (parameter accounting).
    """
    # TODO: return 4 * d_model * d_model.
    raise NotImplementedError("Part C -- count_attention_parameters")


# -----------------------------------------------------------------------------
# Part D -- main entry; sanity checks
# -----------------------------------------------------------------------------


def main() -> None:
    """Run a handful of shape and sanity checks."""
    import torch

    torch.manual_seed(RANDOM_STATE)

    # A: build the module, check forward shape.
    mha = build_multihead_attention(d_model=16, n_heads=4, causal=True)
    x = torch.randn(2, 5, 16)
    out = mha(x)
    assert out.shape == (2, 5, 16), f"forward shape wrong: got {tuple(out.shape)}"

    # A (parameter shapes): each projection has shape (d_model, d_model).
    params = dict(mha.named_parameters())
    for name in ("W_q.weight", "W_k.weight", "W_v.weight", "W_o.weight"):
        assert name in params, f"missing parameter {name}"
        assert params[name].shape == (16, 16), (
            f"{name} shape wrong: got {tuple(params[name].shape)}"
        )

    # C: parameter count.
    expected = count_attention_parameters(d_model=16)
    actual = sum(p.numel() for p in mha.parameters())
    assert actual == expected, f"param count mismatch: expected {expected}, got {actual}"
    assert expected == 4 * 16 * 16, f"formula wrong: expected {4 * 16 * 16}, got {expected}"

    # A (causality): position 0's output must not depend on later positions.
    x_perturbed = x.clone()
    x_perturbed[:, 1, :] += 100.0
    out_perturbed = mha(x_perturbed)
    diff_at_0 = (out[:, 0, :] - out_perturbed[:, 0, :]).abs().max().item()
    # The causality check passes only if mha was built with causal=True.
    assert diff_at_0 < 1e-4, (
        f"causal mha at position 0 should not depend on x[1]; "
        f"got max diff {diff_at_0:.3e}"
    )

    print("OK -- exercise 2 (shape and sanity checks)")


# -----------------------------------------------------------------------------
# Tests (require torch; will be skipped if torch is not installed)
# -----------------------------------------------------------------------------


def _import_torch() -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Lazy import of torch and torch.nn."""
    import torch
    from torch import nn

    return torch, nn


def test_param_count_formula() -> None:
    """4 * d_model^2 for several d_model values."""
    _, _ = _import_torch()
    for d_model in (16, 32, 64, 128, 384):
        assert count_attention_parameters(d_model) == 4 * d_model * d_model, (
            f"d_model={d_model}: count mismatch"
        )


def test_param_count_matches_module() -> None:
    """The module's actual parameter count matches the formula."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)
    for d_model, n_heads in ((16, 4), (32, 8), (64, 8), (384, 6)):
        mha = build_multihead_attention(d_model=d_model, n_heads=n_heads)
        actual = sum(p.numel() for p in mha.parameters())
        assert actual == count_attention_parameters(d_model), (
            f"d_model={d_model}, n_heads={n_heads}: param count mismatch "
            f"(actual {actual} vs formula {count_attention_parameters(d_model)})"
        )


def test_non_causal_matches_pytorch_mha() -> None:
    """The non-causal version agrees with nn.MultiheadAttention."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    d_model, n_heads = 16, 4
    ours = build_multihead_attention(d_model=d_model, n_heads=n_heads, causal=False)
    theirs = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=n_heads,
        batch_first=True,
        bias=False,
    )
    copy_weights_into_pytorch_mha(ours, theirs)

    x = torch.randn(2, 6, d_model)
    out_ours = ours(x)
    # nn.MultiheadAttention expects (query, key, value) all separately.
    out_theirs, _weights = theirs(x, x, x, need_weights=False)
    assert torch.allclose(out_ours, out_theirs, atol=1e-5), (
        f"non-causal max diff {(out_ours - out_theirs).abs().max().item():.3e}"
    )


def test_causal_version_is_strictly_causal() -> None:
    """Causal mha: position i's output must not depend on positions j > i."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    mha = build_multihead_attention(d_model=16, n_heads=4, causal=True)
    x = torch.randn(1, 8, 16)
    out = mha(x)

    # Perturb position 5; positions 0..4 of the output should be unchanged.
    x_perturbed = x.clone()
    x_perturbed[:, 5, :] += 50.0
    out_perturbed = mha(x_perturbed)
    max_diff_early = (out[:, :5, :] - out_perturbed[:, :5, :]).abs().max().item()
    assert max_diff_early < 1e-4, (
        f"causal violation in multi-head: positions 0..4 changed by "
        f"{max_diff_early:.3e} when x[5] was perturbed"
    )
    # Position 5 must change.
    max_diff_at_5 = (out[:, 5, :] - out_perturbed[:, 5, :]).abs().max().item()
    assert max_diff_at_5 > 0.0, "expected position 5's output to change"


def test_head_splitting_preserves_information() -> None:
    """Reshaping into heads and back should be a no-op when no attention runs.

    This is a property of the reshape -- if we just split into heads and
    immediately recombine, we should recover the input. The test confirms
    our head-splitting indexing convention matches PyTorch's.
    """
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    B, T, d_model, n_heads = 2, 5, 16, 4
    d_k = d_model // n_heads
    x = torch.randn(B, T, d_model)

    # Split into heads.
    x_heads = x.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
    # Recombine.
    x_recombined = x_heads.transpose(1, 2).contiguous().view(B, T, d_model)
    assert torch.allclose(x, x_recombined), (
        "head split-recombine should be a no-op"
    )


def test_output_norm_is_finite() -> None:
    """Forward should produce finite outputs (no NaN/Inf)."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    mha = build_multihead_attention(d_model=32, n_heads=8, causal=True)
    x = torch.randn(4, 20, 32)
    out = mha(x)
    assert torch.isfinite(out).all(), "output contains NaN or Inf"


if __name__ == "__main__":
    main()
