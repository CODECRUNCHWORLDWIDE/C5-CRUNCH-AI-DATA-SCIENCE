"""
Exercise 1 -- Convolutions by hand.

Goal: implement 2D convolution from scratch with NumPy and verify it agrees
with torch.nn.functional.conv2d to within 1e-5 absolute error. Then compute
the output shape of several Conv2d layers using the formula from Lecture 1
Section 4 and verify against the PyTorch shape.

By the end of this exercise you will have:

    (1) Implemented 2D cross-correlation (what PyTorch calls "convolution")
        for a single (in_channels, out_channels, H, W) tensor with NumPy.
    (2) Verified against torch.nn.functional.conv2d for three input/kernel
        configurations.
    (3) Implemented the output-shape formula and tested it against the
        actual output of nn.Conv2d.
    (4) Constructed a 3x3 Sobel-Y edge detector by hand and applied it to
        a synthetic step-edge image.
    (5) Verified the translation-equivariance property of nn.Conv2d
        empirically.

Estimated time: 60-90 minutes.

Run with:    python exercise-01-convolutions-by-hand.py
Or test:     pytest exercise-01-convolutions-by-hand.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-convolutions-by-hand.py succeeds without
  importing torch (only the test functions import torch).

PyTorch references:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


RANDOM_STATE: int = 42


# -----------------------------------------------------------------------------
# Part A -- the output-shape formula (Lecture 1 Section 4)
# -----------------------------------------------------------------------------


def conv_output_size(
    in_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """Compute the output spatial size of a Conv2d (or MaxPool2d) layer.

    The formula is:

        out = floor((in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Examples:
        >>> conv_output_size(32, 3, stride=1, padding=1)
        32
        >>> conv_output_size(32, 3, stride=2, padding=1)
        16
        >>> conv_output_size(224, 7, stride=2, padding=3)
        112
    """
    # TODO: implement the formula above.
    raise NotImplementedError("Part A -- conv_output_size")


# -----------------------------------------------------------------------------
# Part B -- 2D convolution (well, cross-correlation) with NumPy
# -----------------------------------------------------------------------------


def naive_conv2d(
    x: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32],
    stride: int = 1,
    padding: int = 0,
) -> NDArray[np.float32]:
    """Compute 2D cross-correlation (PyTorch's "convolution") with NumPy.

    Arguments:
        x:      shape (N, C_in, H_in, W_in), float32.
        weight: shape (C_out, C_in, K, K), float32.
        bias:   shape (C_out,),            float32.
        stride: positive int.
        padding: non-negative int; zeros are added around the spatial dims.

    Returns:
        out: shape (N, C_out, H_out, W_out), float32, where H_out and W_out
             are given by `conv_output_size` with dilation=1.

    The math (from Lecture 1 Section 2):

        out[n, c_out, i, j] = bias[c_out] + sum_{c_in, ki, kj}
            weight[c_out, c_in, ki, kj]
            * input_padded[n, c_in, i*stride + ki, j*stride + kj]

    Use four nested loops (n, c_out, i, j); use vectorized NumPy for the
    inner sum over (c_in, ki, kj). The result must match the corresponding
    call to torch.nn.functional.conv2d to within 1e-5 absolute error.
    """
    n_batch, c_in, h_in, w_in = x.shape
    c_out, c_in_w, k_h, k_w = weight.shape
    assert c_in == c_in_w, f"channel mismatch: x has {c_in}, weight has {c_in_w}"
    assert k_h == k_w, f"this exercise only supports square kernels; got {k_h}x{k_w}"

    h_out = conv_output_size(h_in, k_h, stride=stride, padding=padding)
    w_out = conv_output_size(w_in, k_w, stride=stride, padding=padding)

    # Pad the input with zeros around the spatial dimensions.
    if padding > 0:
        x_padded = np.pad(
            x,
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        x_padded = x

    out = np.zeros((n_batch, c_out, h_out, w_out), dtype=np.float32)

    # TODO: fill out[n, c_out, i, j] using the math above. The inner sum
    # over (c_in, ki, kj) should be a single np.sum of an elementwise product
    # between weight[c_out] (shape (C_in, K, K)) and the input patch
    # x_padded[n, :, i*stride:i*stride+K, j*stride:j*stride+K] (also shape
    # (C_in, K, K)). Then add bias[c_out].
    raise NotImplementedError("Part B -- naive_conv2d")


# -----------------------------------------------------------------------------
# Part C -- a 3x3 Sobel-Y edge detector
# -----------------------------------------------------------------------------


def sobel_y_kernel() -> NDArray[np.float32]:
    """Return the 3x3 Sobel-Y edge detector kernel.

    The Sobel-Y kernel detects horizontal edges by approximating the vertical
    gradient. The standard form is:

        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]

    Returns a (1, 1, 3, 3) float32 array suitable for use with naive_conv2d
    (one output channel, one input channel).
    """
    # TODO: return the (1, 1, 3, 3) array containing the Sobel-Y kernel.
    raise NotImplementedError("Part C -- sobel_y_kernel")


def horizontal_step_image(height: int = 8, width: int = 8) -> NDArray[np.float32]:
    """Return a (1, 1, H, W) image with a horizontal step edge in the middle.

    The top half is 0.0, the bottom half is 1.0. The Sobel-Y kernel applied
    to this image should produce a strong positive response on the row just
    below the step and zero everywhere else (modulo boundary effects).
    """
    # TODO: build a (1, 1, height, width) array of zeros, then set the bottom
    # half (rows >= height // 2) to 1.0.
    raise NotImplementedError("Part C -- horizontal_step_image")


# -----------------------------------------------------------------------------
# Part D -- main entry; sanity checks (no torch required)
# -----------------------------------------------------------------------------


def main() -> None:
    """Run a handful of pure-NumPy sanity checks.

    The deep verification (NumPy vs. torch) lives in the pytest functions
    below, which import torch lazily. The main() entry is callable on a
    Python with only numpy installed.
    """
    # A: output-size formula
    assert conv_output_size(32, 3, stride=1, padding=1) == 32
    assert conv_output_size(32, 3, stride=2, padding=1) == 16
    assert conv_output_size(224, 7, stride=2, padding=3) == 112
    assert conv_output_size(28, 5, stride=1, padding=0) == 24

    # B: NumPy conv on a tiny synthetic input
    rng = np.random.default_rng(RANDOM_STATE)
    x = rng.standard_normal((1, 1, 5, 5)).astype(np.float32)
    w = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
    b = np.zeros((1,), dtype=np.float32)
    y = naive_conv2d(x, w, b, stride=1, padding=1)
    assert y.shape == (1, 1, 5, 5), f"shape mismatch: got {y.shape}"

    # C: Sobel-Y on a step edge
    img = horizontal_step_image(8, 8)
    sobel = sobel_y_kernel()
    bias = np.zeros((1,), dtype=np.float32)
    response = naive_conv2d(img, sobel, bias, stride=1, padding=1)
    # The response on the row just above the edge should be strongly positive.
    edge_row = 8 // 2 - 1
    assert response[0, 0, edge_row, 1:-1].mean() > 2.0, (
        f"Sobel-Y did not detect the horizontal edge; "
        f"row {edge_row} mean = {response[0, 0, edge_row, 1:-1].mean()}"
    )

    print("OK -- exercise 1 (numpy-only checks)")


# -----------------------------------------------------------------------------
# Tests (require torch; will be skipped if torch is not installed)
# -----------------------------------------------------------------------------


def _import_torch() -> "Tuple[object, object]":  # type: ignore[type-arg]
    """Lazy import of torch and torch.nn.functional.

    Returns (torch, torch.nn.functional). Raises ImportError if torch is
    not installed; pytest will treat that as a skip via the conftest, but
    if you are running outside pytest the ImportError surfaces.
    """
    import torch  # noqa: F401
    import torch.nn.functional as F  # noqa: F401

    return torch, F


def test_conv_output_size_matches_pytorch() -> None:
    """For three Conv2d configs, the formula matches the actual output shape."""
    torch, F = _import_torch()
    from torch import nn

    configs = [
        (3, 64, 3, 1, 1, 32),
        (3, 64, 3, 2, 1, 32),
        (3, 64, 7, 2, 3, 224),
    ]
    for c_in, c_out, k, s, p, in_size in configs:
        conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p)
        x = torch.zeros(1, c_in, in_size, in_size)
        out_size = conv(x).shape[-1]
        formula = conv_output_size(in_size, k, stride=s, padding=p)
        assert formula == out_size, (
            f"formula said {formula}, PyTorch produced {out_size} "
            f"for in={in_size} k={k} s={s} p={p}"
        )


def test_naive_conv2d_matches_pytorch_no_padding() -> None:
    """NumPy conv == torch.nn.functional.conv2d for a 1x1x5x5 / 1x1x3x3 case."""
    torch, F = _import_torch()
    rng = np.random.default_rng(RANDOM_STATE)
    x_np = rng.standard_normal((1, 1, 5, 5)).astype(np.float32)
    w_np = rng.standard_normal((1, 1, 3, 3)).astype(np.float32)
    b_np = np.zeros((1,), dtype=np.float32)

    y_np = naive_conv2d(x_np, w_np, b_np, stride=1, padding=0)
    x_t = torch.from_numpy(x_np)
    w_t = torch.from_numpy(w_np)
    b_t = torch.from_numpy(b_np)
    y_t = F.conv2d(x_t, w_t, b_t, stride=1, padding=0).numpy()

    assert np.allclose(y_np, y_t, atol=1e-5), (
        f"NumPy and PyTorch convs disagree by {np.abs(y_np - y_t).max():.3e}"
    )


def test_naive_conv2d_matches_pytorch_with_padding() -> None:
    """Same as above but with padding=1 and stride=2."""
    torch, F = _import_torch()
    rng = np.random.default_rng(RANDOM_STATE + 1)
    x_np = rng.standard_normal((2, 3, 16, 16)).astype(np.float32)
    w_np = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)
    b_np = rng.standard_normal((8,)).astype(np.float32)

    y_np = naive_conv2d(x_np, w_np, b_np, stride=2, padding=1)
    x_t = torch.from_numpy(x_np)
    w_t = torch.from_numpy(w_np)
    b_t = torch.from_numpy(b_np)
    y_t = F.conv2d(x_t, w_t, b_t, stride=2, padding=1).numpy()

    assert y_np.shape == y_t.shape, f"shape mismatch: {y_np.shape} vs {y_t.shape}"
    assert np.allclose(y_np, y_t, atol=1e-5), (
        f"NumPy and PyTorch convs disagree by {np.abs(y_np - y_t).max():.3e}"
    )


def test_translation_equivariance() -> None:
    """Conv layers are translation-equivariant; Linear layers are not."""
    torch, F = _import_torch()
    from torch import nn

    torch.manual_seed(RANDOM_STATE)
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 8, 8)
    x_shifted = torch.roll(x, shifts=1, dims=-1)

    y = conv(x).detach()
    y_shifted = conv(x_shifted).detach()
    # The output should also be shifted by 1 along the last spatial axis,
    # modulo boundary effects. Compare in the interior (drop a 1-pixel border).
    expected = torch.roll(y, shifts=1, dims=-1)
    diff = (y_shifted[..., :, 1:-1] - expected[..., :, 1:-1]).abs().max().item()
    assert diff < 1e-5, f"conv is not translation-equivariant; max diff {diff:.3e}"


if __name__ == "__main__":
    main()
