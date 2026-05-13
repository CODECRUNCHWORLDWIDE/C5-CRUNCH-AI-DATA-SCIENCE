"""
Exercise 3 — An image is an ndarray.

Goal: load a real RGB image as a 3-D uint8 array and manipulate it
using nothing but NumPy. This is the on-ramp to the mini-project.

Estimated time: 35 minutes.

Run with:    python exercise-03-image-as-array.py
Or test:     pytest exercise-03-image-as-array.py

Acceptance criteria:
- Every TODO is filled in.
- All asserts pass; the script prints "OK — exercise 3".
- If imageio is installed and the network is up, the script also writes
  three small PNG files next to itself so you can eyeball the results.

Setup:
    pip install numpy imageio

If you cannot fetch a remote image (offline, corporate firewall), the
script falls back to a synthetic 32x32x3 gradient so the tests still
run. The point is the array work, not the picture.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np


# A small, public-domain test image hosted on GitHub. Tiny (~80 KB).
SAMPLE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
    "PNG_transparency_demonstration_1.png/240px-"
    "PNG_transparency_demonstration_1.png"
)


def load_sample_image() -> np.ndarray:
    """Return a 3-D uint8 array of shape (H, W, 3).

    Tries to download a small public-domain PNG. Falls back to a
    synthetic gradient if imageio is missing or the download fails.
    The fallback is deterministic so the tests do not flake.
    """
    try:
        import imageio.v3 as iio
        with urllib.request.urlopen(SAMPLE_URL, timeout=5) as resp:
            data = resp.read()
        img = iio.imread(io.BytesIO(data))
        # Some PNGs come back as RGBA; drop the alpha channel for this exercise.
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return np.ascontiguousarray(img.astype(np.uint8))
    except Exception:
        # Synthetic fallback: a simple H x W x 3 gradient.
        H, W = 32, 32
        ys = np.arange(H, dtype=np.uint8)[:, None]
        xs = np.arange(W, dtype=np.uint8)[None, :]
        r = (ys * 8 % 256).astype(np.uint8)
        g = (xs * 8 % 256).astype(np.uint8)
        b = ((ys + xs) * 4 % 256).astype(np.uint8)
        return np.stack(
            [np.broadcast_to(r, (H, W)),
             np.broadcast_to(g, (H, W)),
             np.broadcast_to(b, (H, W))],
            axis=2,
        ).copy()


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------

def red_channel(img: np.ndarray) -> np.ndarray:
    """Return the red channel as a 2-D uint8 array of shape (H, W)."""
    # TODO: slice axis 2 at index 0.
    raise NotImplementedError("red_channel")


def flip_vertical(img: np.ndarray) -> np.ndarray:
    """Return the image flipped top-to-bottom WITHOUT copying.

    The returned array must share memory with `img` (np.shares_memory).
    """
    # TODO: use basic slicing to reverse axis 0.
    raise NotImplementedError("flip_vertical")


def naive_grayscale(img: np.ndarray) -> np.ndarray:
    """A first-pass grayscale: average of the three channels.

    Output dtype must be uint8 and shape (H, W). Clip if needed.
    """
    # TODO: mean across axis=2, cast to uint8.
    raise NotImplementedError("naive_grayscale")


def luma_grayscale(img: np.ndarray) -> np.ndarray:
    """A proper grayscale using ITU-R BT.601 luma weights:
        Y = 0.299 R + 0.587 G + 0.114 B

    Output dtype must be uint8 and shape (H, W). Clip to [0, 255].
    """
    # TODO: cast img to float32, multiply by a length-3 weight vector
    #       (broadcasting), sum along axis=2, clip, cast back to uint8.
    raise NotImplementedError("luma_grayscale")


def threshold_to_binary(gray: np.ndarray, t: int) -> np.ndarray:
    """Given a 2-D uint8 grayscale image, return a uint8 image that is
    255 where gray > t and 0 elsewhere.
    """
    # TODO: one expression with np.where, .astype(np.uint8).
    raise NotImplementedError("threshold_to_binary")


def crop_center(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return the center (h, w, 3) crop of `img`. Assume h <= H and w <= W."""
    # TODO: compute start row/col, then slice. Returns a view.
    raise NotImplementedError("crop_center")


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_image_loads_as_uint8_rgb() -> None:
    img = load_sample_image()
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.dtype == np.uint8


def test_red_channel() -> None:
    img = load_sample_image()
    r = red_channel(img)
    assert r.shape == img.shape[:2]
    assert r.dtype == np.uint8
    assert np.array_equal(r, img[:, :, 0])


def test_flip_vertical_is_a_view() -> None:
    img = load_sample_image()
    flipped = flip_vertical(img)
    assert flipped.shape == img.shape
    assert np.array_equal(flipped[0], img[-1])
    assert np.shares_memory(img, flipped), \
        "flip_vertical should not copy; use basic slicing."


def test_grayscales_have_right_shape_and_dtype() -> None:
    img = load_sample_image()
    g1 = naive_grayscale(img)
    g2 = luma_grayscale(img)
    for g in (g1, g2):
        assert g.shape == img.shape[:2]
        assert g.dtype == np.uint8
        assert g.min() >= 0 and g.max() <= 255


def test_threshold_to_binary() -> None:
    img = load_sample_image()
    g = luma_grayscale(img)
    b = threshold_to_binary(g, 128)
    assert b.shape == g.shape
    assert b.dtype == np.uint8
    assert set(np.unique(b).tolist()).issubset({0, 255})


def test_crop_center() -> None:
    img = load_sample_image()
    H, W = img.shape[:2]
    h, w = min(16, H), min(16, W)
    c = crop_center(img, h, w)
    assert c.shape == (h, w, 3)


def _save_outputs_if_possible(img: np.ndarray) -> None:
    """Best-effort: write three small PNGs next to this script. Silent fail."""
    try:
        import imageio.v3 as iio
        here = Path(__file__).parent
        iio.imwrite(here / "out-gray-luma.png", luma_grayscale(img))
        iio.imwrite(here / "out-gray-naive.png", naive_grayscale(img))
        iio.imwrite(here / "out-flipped.png", flip_vertical(img))
    except Exception:
        pass


def _run_all_tests() -> None:
    test_image_loads_as_uint8_rgb()
    test_red_channel()
    test_flip_vertical_is_a_view()
    test_grayscales_have_right_shape_and_dtype()
    test_threshold_to_binary()
    test_crop_center()
    _save_outputs_if_possible(load_sample_image())
    print("OK — exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# red_channel:
#     return img[:, :, 0]
#
# flip_vertical:
#     return img[::-1, :, :]
#
# naive_grayscale:
#     return img.mean(axis=2).clip(0, 255).astype(np.uint8)
#
# luma_grayscale:
#     w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
#     y = (img.astype(np.float32) * w).sum(axis=2)
#     return y.clip(0, 255).astype(np.uint8)
#
# threshold_to_binary:
#     return np.where(gray > t, 255, 0).astype(np.uint8)
#
# crop_center:
#     H, W = img.shape[:2]
#     r0 = (H - h) // 2
#     c0 = (W - w) // 2
#     return img[r0:r0 + h, c0:c0 + w, :]
#
# -----------------------------------------------------------------------------
