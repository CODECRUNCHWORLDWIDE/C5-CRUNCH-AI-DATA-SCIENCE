"""
Exercise 2 -- Implement an LSTM cell from scratch.

Goal: implement the four-gate LSTM equations from Lecture 2 Section 2 as a
custom nn.Module, then verify it agrees with nn.LSTMCell to within 1e-5
absolute error. Then unroll it across a small sequence and verify against
nn.LSTM. Finally, demonstrate the additive cell-state-update property by
measuring ||c_t|| under zero input.

By the end of this exercise you will have:

    (1) Implemented the LSTM cell as a torch.nn.Module called CustomLSTMCell.
        The cell packs the four gates into a single (4*hidden, input + hidden)
        weight matrix for efficiency, matching PyTorch's internal layout.
    (2) Verified the cell against nn.LSTMCell. This requires understanding
        the PyTorch gate-packing order, which is [i, f, g, o] (NOT [f, i, g, o]
        as in the lecture equations). The discrepancy is a famous source of
        bugs.
    (3) Implemented an unroll helper and verified against nn.LSTM.
    (4) Measured ||c_t|| under zero input. The cell state should NOT decay
        to zero (in contrast to the vanilla RNN's ||h_t||) because the
        forget gate, initialized near 0.5, retains roughly half of c_{t-1}
        at each step.
    (5) Demonstrated the "Jozefowicz fix" -- initializing the forget-gate
        bias to 1.0 sharply slows the decay of ||c_t||.

Estimated time: 90-120 minutes.

Run with:    python exercise-02-lstm-cell-from-scratch.py
Or test:     pytest exercise-02-lstm-cell-from-scratch.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-02-lstm-cell-from-scratch.py succeeds.

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
"""

from __future__ import annotations

import math
from typing import List, Tuple

# torch is imported lazily inside functions.


RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Part A -- the CustomLSTMCell module
# -----------------------------------------------------------------------------


def build_lstm_cell(input_size: int, hidden_size: int) -> "object":  # type: ignore[type-arg]
    """Construct a CustomLSTMCell as an nn.Module.

    The cell implements the LSTM equations from Lecture 2 Section 2:

        f_t = sigmoid(W_f @ [x_t, h_{t-1}] + b_f)     # forget gate
        i_t = sigmoid(W_i @ [x_t, h_{t-1}] + b_i)     # input gate
        g_t = tanh   (W_g @ [x_t, h_{t-1}] + b_g)     # candidate
        o_t = sigmoid(W_o @ [x_t, h_{t-1}] + b_o)     # output gate
        c_t = f_t * c_{t-1} + i_t * g_t               # cell-state update
        h_t = o_t * tanh(c_t)                         # hidden-state output

    Internally the four affine layers are packed into two tensors:
        - weight_ih: shape (4*hidden_size, input_size). Rows are the four gates'
          contributions from x_t.
        - weight_hh: shape (4*hidden_size, hidden_size). Rows are the four
          gates' contributions from h_{t-1}.
        - bias_ih:   shape (4*hidden_size,). Per-gate bias.

    The gate order in the packed tensors is [i, f, g, o] -- this matches
    PyTorch's nn.LSTMCell layout. This is critical for the verification
    test to pass.

    Initialization: weights uniform in [-1/sqrt(hidden_size), +1/sqrt(hidden_size)];
    bias to zero.

    The forward signature is forward(x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor])
    where x has shape (batch, input_size) and state is (h_prev, c_prev) each
    of shape (batch, hidden_size). The return is (h_next, c_next).

    Returns:
        A torch.nn.Module instance.

    References:
        https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    """
    import torch
    from torch import nn

    class CustomLSTMCell(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_size: int = input_size
            self.hidden_size: int = hidden_size
            bound: float = 1.0 / math.sqrt(hidden_size)
            self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size).uniform_(-bound, bound))
            self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size).uniform_(-bound, bound))
            self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))

        def forward(
            self,
            x: "torch.Tensor",
            state: "Tuple[torch.Tensor, torch.Tensor]",
        ) -> "Tuple[torch.Tensor, torch.Tensor]":
            h_prev, c_prev = state
            # Compute the packed pre-activations: shape (batch, 4*hidden).
            # gates_in = x @ weight_ih.T + h_prev @ weight_hh.T + bias_ih.
            # TODO: compute gates_in.
            #
            # Then split gates_in into four chunks of size hidden_size along
            # the last dimension. PyTorch's convention is [i, f, g, o] in
            # that order.
            # TODO: i_gate, f_gate, g_gate, o_gate = gates_in.chunk(4, dim=-1)
            # Apply the per-gate nonlinearities:
            #   i = sigmoid(i_gate); f = sigmoid(f_gate);
            #   g = tanh(g_gate);    o = sigmoid(o_gate)
            # Then compute c_next = f * c_prev + i * g, and
            # h_next = o * tanh(c_next).
            raise NotImplementedError("Part A -- CustomLSTMCell.forward")

    return CustomLSTMCell()


# -----------------------------------------------------------------------------
# Part B -- unroll across a sequence
# -----------------------------------------------------------------------------


def unroll_lstm_cell(
    cell: "object",
    inputs: "torch.Tensor",
    h_0: "torch.Tensor",
    c_0: "torch.Tensor",
) -> "Tuple[torch.Tensor, torch.Tensor, torch.Tensor]":  # type: ignore[type-arg]
    """Unroll an LSTM cell across a sequence.

    Arguments:
        cell: nn.Module with signature forward(x, (h, c)) -> (h, c).
        inputs: shape (batch, seq_len, input_size).
        h_0, c_0: each shape (batch, hidden_size).

    Returns:
        (hiddens, h_T, c_T) where
            hiddens: shape (batch, seq_len, hidden_size). hiddens[:, t, :]
                     is h_t after step t.
            h_T:     final hidden state, shape (batch, hidden_size).
            c_T:     final cell state, shape (batch, hidden_size).
    """
    import torch

    n_batch, seq_len, _ = inputs.shape
    hidden_size = h_0.shape[-1]
    hiddens = torch.zeros(n_batch, seq_len, hidden_size, dtype=inputs.dtype, device=inputs.device)
    h_prev, c_prev = h_0, c_0
    # TODO: at each step t, call (h_next, c_next) = cell(inputs[:, t, :], (h_prev, c_prev)),
    # write h_next into hiddens[:, t, :], and update (h_prev, c_prev).
    raise NotImplementedError("Part B -- unroll_lstm_cell")


# -----------------------------------------------------------------------------
# Part C -- the Jozefowicz fix
# -----------------------------------------------------------------------------


def jozefowicz_init_forget_bias(cell: "object", value: float = 1.0) -> None:
    """Set the forget-gate bias to `value`.

    The forget-gate slice of bias_ih lives at indices [hidden_size:2*hidden_size]
    because the packed-gate order is [i, f, g, o]. Setting this slice to a
    positive value biases the forget gate toward 1 (sigmoid(>0) > 0.5),
    which encourages the cell to retain information across timesteps at the
    start of training.

    Reference: Jozefowicz, Zaremba, Sutskever 2015,
    "An Empirical Exploration of Recurrent Network Architectures":
    https://proceedings.mlr.press/v37/jozefowicz15.pdf

    Modifies the cell in place; returns None.
    """
    import torch

    hidden_size: int = cell.hidden_size  # type: ignore[attr-defined]
    # TODO: with torch.no_grad(), set cell.bias_ih.data[hidden_size:2*hidden_size]
    # to `value` (broadcasted across the slice).
    raise NotImplementedError("Part C -- jozefowicz_init_forget_bias")


def cell_state_norm_trajectory(
    cell: "object",
    h_0: "torch.Tensor",
    c_0: "torch.Tensor",
    n_steps: int,
    input_size: int,
) -> List[float]:  # type: ignore[type-arg]
    """Unroll the cell for n_steps with zero input; record ||c_t|| at each step.

    Arguments:
        cell: a CustomLSTMCell or nn.LSTMCell.
        h_0, c_0: each shape (batch, hidden_size).
        n_steps: positive integer.
        input_size: dimensionality of the (zero) input vector.

    Returns:
        Python list of length n_steps with the mean L2 norm of c_t across
        the batch dimension.
    """
    import torch

    assert n_steps > 0
    n_batch = h_0.shape[0]
    zero_input = torch.zeros(n_batch, input_size, dtype=h_0.dtype, device=h_0.device)
    norms: List[float] = []
    h_prev, c_prev = h_0, c_0
    # TODO: unroll; at each step, append torch.linalg.norm(c_next, dim=-1).mean().item()
    # to norms.
    raise NotImplementedError("Part C -- cell_state_norm_trajectory")


# -----------------------------------------------------------------------------
# Part D -- main entry; sanity checks
# -----------------------------------------------------------------------------


def main() -> None:
    """Run a handful of sanity checks (do not require copying PyTorch weights)."""
    import torch

    torch.manual_seed(RANDOM_STATE)

    # A: instantiate and check parameter shapes.
    cell = build_lstm_cell(input_size=8, hidden_size=16)
    params = dict(cell.named_parameters())
    assert "weight_ih" in params and params["weight_ih"].shape == (64, 8), (
        f"weight_ih shape wrong: got {params.get('weight_ih', None) and params['weight_ih'].shape}"
    )
    assert "weight_hh" in params and params["weight_hh"].shape == (64, 16), (
        f"weight_hh shape wrong: got {params.get('weight_hh', None) and params['weight_hh'].shape}"
    )
    assert "bias_ih" in params and params["bias_ih"].shape == (64,), (
        f"bias_ih shape wrong: got {params.get('bias_ih', None) and params['bias_ih'].shape}"
    )

    # B: one forward step.
    x = torch.randn(4, 8)
    h_prev = torch.zeros(4, 16)
    c_prev = torch.zeros(4, 16)
    h_next, c_next = cell(x, (h_prev, c_prev))
    assert h_next.shape == (4, 16) and c_next.shape == (4, 16), (
        f"forward shapes wrong: h_next={tuple(h_next.shape)} c_next={tuple(c_next.shape)}"
    )
    # tanh-bounded hidden state.
    assert h_next.abs().max().item() < 1.0 + 1e-6, "h_next exceeded tanh ceiling"

    # B (unroll): unroll for 7 steps.
    inputs = torch.randn(4, 7, 8)
    hiddens, h_T, c_T = unroll_lstm_cell(cell, inputs, h_prev, c_prev)
    assert hiddens.shape == (4, 7, 16), f"unroll hiddens shape wrong: got {tuple(hiddens.shape)}"
    assert h_T.shape == (4, 16) and c_T.shape == (4, 16), "final-state shape wrong"

    # C: Jozefowicz fix changes only the forget-gate slice of bias.
    cell2 = build_lstm_cell(input_size=8, hidden_size=16)
    pre = cell2.bias_ih.data.clone()
    jozefowicz_init_forget_bias(cell2, value=1.0)
    post = cell2.bias_ih.data
    # The [hidden_size:2*hidden_size] slice should now be 1.0; other slices unchanged.
    assert torch.allclose(post[16:32], torch.ones(16)), "forget-gate bias slice not set to 1"
    assert torch.allclose(post[0:16], pre[0:16]), "input-gate slice unexpectedly changed"
    assert torch.allclose(post[32:48], pre[32:48]), "candidate slice unexpectedly changed"
    assert torch.allclose(post[48:64], pre[48:64]), "output-gate slice unexpectedly changed"

    print("OK -- exercise 2 (shape checks)")


# -----------------------------------------------------------------------------
# Tests (require torch)
# -----------------------------------------------------------------------------


def _import_torch() -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Lazy import of torch and torch.nn."""
    import torch
    from torch import nn

    return torch, nn


def test_lstm_cell_matches_nn_lstm_cell() -> None:
    """Copy parameters into nn.LSTMCell and verify per-step outputs match."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    ours = build_lstm_cell(input_size=8, hidden_size=16)
    theirs = nn.LSTMCell(input_size=8, hidden_size=16)

    # Both PyTorch and our cell use the [i, f, g, o] gate order in the packed
    # weight tensor. PyTorch's bias is split (bias_ih + bias_hh); we put
    # everything in bias_ih and zero out bias_hh on PyTorch's side.
    with torch.no_grad():
        theirs.weight_ih.copy_(ours.weight_ih)  # type: ignore[attr-defined]
        theirs.weight_hh.copy_(ours.weight_hh)  # type: ignore[attr-defined]
        theirs.bias_ih.copy_(ours.bias_ih)  # type: ignore[attr-defined]
        theirs.bias_hh.zero_()

    x = torch.randn(3, 8)
    h_prev = torch.randn(3, 16)
    c_prev = torch.randn(3, 16)

    h_ours, c_ours = ours(x, (h_prev, c_prev))
    h_theirs, c_theirs = theirs(x, (h_prev, c_prev))
    assert torch.allclose(h_ours, h_theirs, atol=1e-5), (
        f"h disagreement: {(h_ours - h_theirs).abs().max().item():.3e}"
    )
    assert torch.allclose(c_ours, c_theirs, atol=1e-5), (
        f"c disagreement: {(c_ours - c_theirs).abs().max().item():.3e}"
    )


def test_unroll_matches_nn_lstm() -> None:
    """Python unroll of our cell matches nn.LSTM's optimized C++ unroll."""
    torch, nn = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    ours = build_lstm_cell(input_size=4, hidden_size=8)
    lstm = nn.LSTM(input_size=4, hidden_size=8, num_layers=1, batch_first=True)
    with torch.no_grad():
        lstm.weight_ih_l0.copy_(ours.weight_ih)  # type: ignore[attr-defined]
        lstm.weight_hh_l0.copy_(ours.weight_hh)  # type: ignore[attr-defined]
        lstm.bias_ih_l0.copy_(ours.bias_ih)  # type: ignore[attr-defined]
        lstm.bias_hh_l0.zero_()

    inputs = torch.randn(2, 10, 4)
    h_0 = torch.zeros(2, 8)
    c_0 = torch.zeros(2, 8)
    hiddens_ours, h_T_ours, c_T_ours = unroll_lstm_cell(ours, inputs, h_0, c_0)
    out_theirs, (h_T_theirs, c_T_theirs) = lstm(inputs, (h_0.unsqueeze(0), c_0.unsqueeze(0)))

    assert torch.allclose(hiddens_ours, out_theirs, atol=1e-5), (
        f"unroll outputs disagree by {(hiddens_ours - out_theirs).abs().max().item():.3e}"
    )
    assert torch.allclose(h_T_ours, h_T_theirs.squeeze(0), atol=1e-5), "h_T disagreement"
    assert torch.allclose(c_T_ours, c_T_theirs.squeeze(0), atol=1e-5), "c_T disagreement"


def test_jozefowicz_slows_decay() -> None:
    """Setting forget-gate bias to 1.0 slows the decay of ||c_t|| with zero input."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    # Build two identical-but-for-the-forget-bias cells.
    cell_default = build_lstm_cell(input_size=4, hidden_size=32)
    cell_jozefowicz = build_lstm_cell(input_size=4, hidden_size=32)
    # Copy params so the two cells differ only in the forget-bias slice.
    with torch.no_grad():
        cell_jozefowicz.weight_ih.copy_(cell_default.weight_ih)  # type: ignore[attr-defined]
        cell_jozefowicz.weight_hh.copy_(cell_default.weight_hh)  # type: ignore[attr-defined]
        cell_jozefowicz.bias_ih.copy_(cell_default.bias_ih)  # type: ignore[attr-defined]
    jozefowicz_init_forget_bias(cell_jozefowicz, value=1.0)

    h_0 = torch.randn(4, 32) * 0.1
    c_0 = torch.randn(4, 32) * 0.5
    default_traj = cell_state_norm_trajectory(cell_default, h_0, c_0, n_steps=40, input_size=4)
    joze_traj = cell_state_norm_trajectory(cell_jozefowicz, h_0, c_0, n_steps=40, input_size=4)
    # The Jozefowicz cell should retain a larger fraction of its cell-state norm
    # after 40 steps.
    assert joze_traj[-1] > default_traj[-1], (
        f"Jozefowicz fix did not slow decay: default {default_traj[-1]:.3e} vs joze {joze_traj[-1]:.3e}"
    )


def test_loss_backward_through_unroll() -> None:
    """Gradient flows through the entire unroll without numerical failure."""
    torch, _ = _import_torch()
    torch.manual_seed(RANDOM_STATE)

    cell = build_lstm_cell(input_size=4, hidden_size=8)
    inputs = torch.randn(2, 20, 4, requires_grad=False)
    h_0 = torch.zeros(2, 8)
    c_0 = torch.zeros(2, 8)
    hiddens, _, _ = unroll_lstm_cell(cell, inputs, h_0, c_0)
    loss = hiddens.pow(2).sum()
    loss.backward()
    # Every parameter should have a non-None grad.
    for n, p in cell.named_parameters():
        assert p.grad is not None, f"parameter {n} did not receive gradient"
        assert torch.isfinite(p.grad).all(), f"parameter {n} got non-finite gradient"


if __name__ == "__main__":
    main()
