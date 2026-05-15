"""
Exercise 1 -- Implement a vanilla RNN cell from scratch.

Goal: implement the Elman RNN recurrence from Lecture 1 Section 2 as a custom
nn.Module, then verify it agrees with nn.RNNCell to within 1e-6 absolute error.
Then unroll it across 200 timesteps and reproduce the hidden-state-norm
plot from Lecture 1's EXPERIMENT in Section 5.

By the end of this exercise you will have:

    (1) Implemented the recurrence h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
        as a torch.nn.Module called VanillaRNNCell.
    (2) Verified the cell against nn.RNNCell by copying weights and comparing
        per-step hidden states across three input/state configurations.
    (3) Implemented an unroll helper that runs the cell across a sequence
        and returns the stack of hidden states.
    (4) Measured the L2 norm of the hidden state at each step of a 200-step
        unroll with zero input, with W_hh rescaled to spectral norms 0.5,
        1.0, and 1.5. The three regimes reproduce the vanishing-gradient,
        balanced, and exploding-gradient cases from Lecture 1 Section 5.
    (5) Verified that gradient clipping (clip_grad_norm_) caps the gradient
        magnitude at the requested value.

Estimated time: 60-90 minutes.

Run with:    python exercise-01-rnn-cell-from-scratch.py
Or test:     pytest exercise-01-rnn-cell-from-scratch.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-rnn-cell-from-scratch.py succeeds.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
    https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
"""

from __future__ import annotations

import math
from typing import List, Tuple

# torch is imported lazily inside functions so this file compiles cleanly
# without it installed. The pytest functions do require torch.


RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Part A -- the VanillaRNNCell module
# -----------------------------------------------------------------------------


def build_vanilla_rnn_cell(input_size: int, hidden_size: int) -> "object":  # type: ignore[type-arg]
    """Construct a VanillaRNNCell as an nn.Module.

    The cell implements the recurrence from Lecture 1 Section 2:

        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)

    Three parameter tensors:
        - W_xh: shape (hidden_size, input_size), initialized uniformly
          in [-1/sqrt(hidden_size), +1/sqrt(hidden_size)].
        - W_hh: shape (hidden_size, hidden_size), same initialization.
        - b:    shape (hidden_size,), initialized to zero.

    The forward signature is forward(x: torch.Tensor, h_prev: torch.Tensor)
    where x has shape (batch, input_size) and h_prev has shape
    (batch, hidden_size). The return is h_next of shape (batch, hidden_size).

    Returns:
        A torch.nn.Module instance.

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
    """
    # Lazy import so py_compile passes without torch.
    import torch
    from torch import nn

    class VanillaRNNCell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_size: int = input_size
            self.hidden_size: int = hidden_size
            # TODO: register three nn.Parameter tensors: W_xh, W_hh, b
            # following the shapes and initializations above.
            #
            # Hint: bound = 1.0 / math.sqrt(hidden_size); use nn.init.uniform_
            # or simply construct torch.empty(...).uniform_(-bound, +bound).
            raise NotImplementedError("Part A -- VanillaRNNCell.__init__")

        def forward(self, x: "torch.Tensor", h_prev: "torch.Tensor") -> "torch.Tensor":
            # TODO: implement h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
            # using torch.tanh and the @ operator (matrix multiply).
            # Note: x has shape (batch, input_size); we want W_xh @ x_t
            # for each batch element, which is x @ W_xh.T in batched form.
            raise NotImplementedError("Part A -- VanillaRNNCell.forward")

    return VanillaRNNCell()


# -----------------------------------------------------------------------------
# Part B -- the unroll helper
# -----------------------------------------------------------------------------


def unroll_rnn_cell(
    cell: "object",
    inputs: "torch.Tensor",
    h_0: "torch.Tensor",
) -> "torch.Tensor":  # type: ignore[type-arg]
    """Unroll an RNN cell across a sequence and return all hidden states.

    Arguments:
        cell: an nn.Module with signature forward(x, h_prev) -> h_next.
              Either our VanillaRNNCell or PyTorch's nn.RNNCell.
        inputs: shape (batch, seq_len, input_size).
        h_0:    shape (batch, hidden_size).

    Returns:
        hiddens: shape (batch, seq_len, hidden_size). hiddens[:, t, :] is
                 the hidden state h_t after consuming inputs[:, t, :].

    The function iterates Python-side across the seq_len dimension and calls
    cell forward at each step. This is the slow, transparent version; in
    production you would call nn.RNN to do the same unroll in C++.
    """
    import torch

    n_batch, seq_len, _ = inputs.shape
    hidden_size = h_0.shape[-1]
    hiddens = torch.zeros(n_batch, seq_len, hidden_size, dtype=inputs.dtype, device=inputs.device)
    h_prev = h_0
    # TODO: at each step t in range(seq_len), set h_next = cell(inputs[:, t, :], h_prev),
    # write h_next into hiddens[:, t, :], and update h_prev = h_next.
    raise NotImplementedError("Part B -- unroll_rnn_cell")


# -----------------------------------------------------------------------------
# Part C -- spectral norm rescaling and hidden-state-norm measurement
# -----------------------------------------------------------------------------


def rescale_spectral_norm(weight: "torch.Tensor", target_norm: float) -> "torch.Tensor":
    """Return a copy of `weight` rescaled so its largest singular value equals
    `target_norm`.

    The spectral norm of a matrix M is its largest singular value, sigma_1(M).
    To rescale M to have spectral norm tau, compute sigma_1(M) and multiply
    M by (tau / sigma_1(M)).

    PyTorch's torch.linalg.svdvals returns the singular values in descending
    order, so the largest is svdvals[0].

    Arguments:
        weight: a 2-D tensor (the recurrent weight matrix W_hh).
        target_norm: the desired spectral norm; must be positive.

    Returns:
        A new tensor with the same shape as weight, with spectral norm
        equal to target_norm to within 1e-5 relative error.

    Reference:
        https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html
    """
    import torch

    assert weight.ndim == 2, f"expected 2-D matrix, got shape {tuple(weight.shape)}"
    assert target_norm > 0.0, f"target_norm must be positive, got {target_norm}"
    # TODO: compute s = torch.linalg.svdvals(weight)[0]; return weight * (target_norm / s).
    raise NotImplementedError("Part C -- rescale_spectral_norm")


def hidden_norm_trajectory(
    cell: "object",
    h_0: "torch.Tensor",
    n_steps: int,
    input_size: int,
) -> List[float]:  # type: ignore[type-arg]
    """Unroll the cell for n_steps with zero input and record ||h_t|| at each step.

    The zero-input unroll isolates the recurrent dynamics from the input
    drive. The trajectory of ||h_t|| reveals whether the recurrence is
    contractive (norm decays), balanced (norm stable), or expansive (norm
    saturates at the tanh ceiling).

    Arguments:
        cell: a VanillaRNNCell or nn.RNNCell.
        h_0: shape (batch, hidden_size).
        n_steps: number of steps to unroll. Must be positive.
        input_size: the dimensionality of the zero-input vector at each step.

    Returns:
        A Python list of length n_steps with the L2 norm of h_t (averaged
        across the batch dimension) at each timestep.
    """
    import torch

    assert n_steps > 0, f"n_steps must be positive, got {n_steps}"
    n_batch = h_0.shape[0]
    zero_input = torch.zeros(n_batch, input_size, dtype=h_0.dtype, device=h_0.device)
    norms: List[float] = []
    h_prev = h_0
    # TODO: at each step, call h_next = cell(zero_input, h_prev), compute
    # the mean L2 norm across the batch (torch.linalg.norm(h_next, dim=-1).mean().item()),
    # append it to norms, and update h_prev = h_next.
    raise NotImplementedError("Part C -- hidden_norm_trajectory")


# -----------------------------------------------------------------------------
# Part D -- main entry; sanity checks
# -----------------------------------------------------------------------------


def main() -> None:
    """Run a handful of sanity checks.

    The verification against nn.RNNCell happens in the pytest functions; main
    runs the shape and basic-recurrence checks that do not require copying
    PyTorch's parameters.
    """
    import torch

    torch.manual_seed(RANDOM_STATE)

    # A: instantiate and check parameter shapes.
    cell = build_vanilla_rnn_cell(input_size=10, hidden_size=20)
    params = dict(cell.named_parameters())
    assert "W_xh" in params and params["W_xh"].shape == (20, 10), (
        f"W_xh shape wrong: got {params.get('W_xh', None) and params['W_xh'].shape}"
    )
    assert "W_hh" in params and params["W_hh"].shape == (20, 20), (
        f"W_hh shape wrong: got {params.get('W_hh', None) and params['W_hh'].shape}"
    )
    assert "b" in params and params["b"].shape == (20,), (
        f"b shape wrong: got {params.get('b', None) and params['b'].shape}"
    )

    # B: one forward step, output shape sanity.
    x = torch.randn(4, 10)
    h_prev = torch.zeros(4, 20)
    h_next = cell(x, h_prev)
    assert h_next.shape == (4, 20), f"forward shape wrong: got {tuple(h_next.shape)}"
    # tanh keeps activations in (-1, 1).
    assert h_next.abs().max().item() < 1.0 + 1e-6, "tanh output exceeded 1"

    # B (unroll): unroll for 5 steps.
    inputs = torch.randn(4, 5, 10)
    h_0 = torch.zeros(4, 20)
    hiddens = unroll_rnn_cell(cell, inputs, h_0)
    assert hiddens.shape == (4, 5, 20), f"unroll shape wrong: got {tuple(hiddens.shape)}"

    # C: spectral-norm rescaling round trip.
    w = torch.randn(20, 20)
    w_scaled = rescale_spectral_norm(w, target_norm=0.5)
    s = torch.linalg.svdvals(w_scaled)[0].item()
    assert abs(s - 0.5) < 1e-4, f"rescale failed; got spectral norm {s}"

    print("OK -- exercise 1 (shape checks)")


# -----------------------------------------------------------------------------
# Tests (require torch; will be skipped if torch is not installed)
# -----------------------------------------------------------------------------


def _import_torch() -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Lazy import of torch and torch.nn.

    Returns (torch, torch.nn). Raises ImportError if torch is not installed;
    pytest treats that as a skip.
    """
    import torch
    from torch import nn

    return torch, nn


def test_vanilla_cell_matches_nn_rnn_cell() -> None:
    """Copy weights into nn.RNNCell and verify per-step outputs match."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    ours = build_vanilla_rnn_cell(input_size=8, hidden_size=16)
    theirs = nn.RNNCell(input_size=8, hidden_size=16, nonlinearity="tanh")

    # Copy our parameters into the PyTorch cell. PyTorch's RNNCell uses
    # two biases (bias_ih and bias_hh) summed together; we put all of our
    # bias into bias_ih and zero out bias_hh.
    with torch.no_grad():
        theirs.weight_ih.copy_(dict(ours.named_parameters())["W_xh"])
        theirs.weight_hh.copy_(dict(ours.named_parameters())["W_hh"])
        theirs.bias_ih.copy_(dict(ours.named_parameters())["b"])
        theirs.bias_hh.zero_()

    x = torch.randn(3, 8)
    h_prev = torch.randn(3, 16)
    h_ours = ours(x, h_prev)
    h_theirs = theirs(x, h_prev)
    assert torch.allclose(h_ours, h_theirs, atol=1e-6), (
        f"per-step outputs disagree by {(h_ours - h_theirs).abs().max().item():.3e}"
    )


def test_unroll_matches_nn_rnn() -> None:
    """The Python unroll of our cell matches nn.RNN's optimized C++ unroll."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    ours = build_vanilla_rnn_cell(input_size=4, hidden_size=8)
    rnn = nn.RNN(input_size=4, hidden_size=8, num_layers=1, nonlinearity="tanh", batch_first=True)
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(dict(ours.named_parameters())["W_xh"])
        rnn.weight_hh_l0.copy_(dict(ours.named_parameters())["W_hh"])
        rnn.bias_ih_l0.copy_(dict(ours.named_parameters())["b"])
        rnn.bias_hh_l0.zero_()

    inputs = torch.randn(2, 10, 4)
    h_0 = torch.zeros(2, 8)
    hiddens_ours = unroll_rnn_cell(ours, inputs, h_0)
    out_theirs, _ = rnn(inputs, h_0.unsqueeze(0))
    # nn.RNN returns (batch, seq_len, hidden_size) when batch_first=True.
    assert torch.allclose(hiddens_ours, out_theirs, atol=1e-5), (
        f"unrolled outputs disagree by {(hiddens_ours - out_theirs).abs().max().item():.3e}"
    )


def test_spectral_rescale_endpoint() -> None:
    """Spectral-norm rescaling produces matrices with the requested norm."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)
    w = torch.randn(32, 32)
    for target in (0.1, 0.5, 1.0, 1.5, 3.0):
        w_scaled = rescale_spectral_norm(w, target_norm=target)
        s = torch.linalg.svdvals(w_scaled)[0].item()
        assert abs(s - target) / target < 1e-4, (
            f"target {target}, got {s}, rel-err {abs(s - target) / target:.3e}"
        )


def test_norm_trajectory_three_regimes() -> None:
    """Vanishing / balanced / exploding regimes reproduce the Lecture 1 plot."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    cell = build_vanilla_rnn_cell(input_size=4, hidden_size=32)
    h_0 = torch.randn(2, 32) * 0.1

    # Contractive: rescale W_hh to spectral norm 0.5. Norm should decay
    # rapidly toward zero.
    with torch.no_grad():
        params = dict(cell.named_parameters())
        params["W_hh"].copy_(rescale_spectral_norm(params["W_hh"].data, target_norm=0.5))
    contractive = hidden_norm_trajectory(cell, h_0, n_steps=50, input_size=4)
    assert contractive[-1] < contractive[0] * 0.01, (
        f"contractive case: expected sharp decay, got {contractive[0]:.3e} -> {contractive[-1]:.3e}"
    )

    # Expansive: rescale to spectral norm 1.5. Norm should grow until tanh saturates.
    with torch.no_grad():
        params["W_hh"].copy_(rescale_spectral_norm(params["W_hh"].data, target_norm=1.5))
    expansive = hidden_norm_trajectory(cell, h_0, n_steps=50, input_size=4)
    assert expansive[-1] > contractive[-1] * 10, (
        f"expansive case: expected growth, got {contractive[-1]:.3e} vs {expansive[-1]:.3e}"
    )


def test_gradient_clipping_caps_norm() -> None:
    """clip_grad_norm_ caps the global gradient norm at the requested value."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    cell = build_vanilla_rnn_cell(input_size=4, hidden_size=8)
    inputs = torch.randn(2, 30, 4)
    h_0 = torch.zeros(2, 8)
    hiddens = unroll_rnn_cell(cell, inputs, h_0)
    # Synthetic loss: sum of all hidden states (gradient will be reused
    # across the entire unroll, so without clipping the gradient norm grows
    # with the sequence length).
    loss = hiddens.sum()
    loss.backward()

    pre_clip_norm = torch.norm(
        torch.stack([torch.linalg.norm(p.grad) for p in cell.parameters() if p.grad is not None])
    ).item()
    assert pre_clip_norm > 0, "expected non-zero pre-clip gradient norm"

    clipped = torch.nn.utils.clip_grad_norm_(cell.parameters(), max_norm=1.0)
    # clipped is the *pre*-clip norm (same as our pre_clip_norm).
    assert abs(clipped.item() - pre_clip_norm) < 1e-4, (
        f"clip_grad_norm_ should report pre-clip norm; got {clipped.item():.3e} vs {pre_clip_norm:.3e}"
    )

    post_clip_norm = torch.norm(
        torch.stack([torch.linalg.norm(p.grad) for p in cell.parameters() if p.grad is not None])
    ).item()
    # The post-clip norm should be at most 1.0 + small epsilon.
    assert post_clip_norm <= 1.0 + 1e-4, f"post-clip norm exceeded 1: {post_clip_norm:.3e}"


if __name__ == "__main__":
    main()
