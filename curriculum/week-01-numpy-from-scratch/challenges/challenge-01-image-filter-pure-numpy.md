# Challenge 1 — Image filter, pure NumPy

**Time estimate:** 2–3 hours.

## Problem statement

Take a real RGB image and produce three transformed versions using **only NumPy** (plus `imageio` to read and write the PNG). No PIL, no OpenCV, no `scipy.ndimage`. The point is that every operation you need is a slice, a broadcast, or a reduction.

You will produce:

1. **`grayscale.png`** — a grayscale version using the ITU-R BT.601 luma formula
   `Y = 0.299·R + 0.587·G + 0.114·B`.
2. **`blur.png`** — a 3×3 box blur applied to the grayscale image. Each output pixel is the mean of itself and its eight neighbours.
3. **`edges.png`** — a Sobel edge-magnitude image of the grayscale.

All three must be saved as `uint8` images you can open in any image viewer.

## Acceptance criteria

- [ ] A single script `image_filter.py` runs end-to-end with `python image_filter.py path/to/input.png` and writes the three output PNGs next to the input.
- [ ] No `for` loop over pixels. (A `for` loop over the three filters is fine; a `for` over `(i, j)` pixel coordinates is not.)
- [ ] The only third-party imports are `numpy` and `imageio`. `from __future__ import annotations` is fine.
- [ ] Output dtypes are `uint8` and values are in `[0, 255]` (clip where needed).
- [ ] `python -m py_compile image_filter.py` succeeds.
- [ ] You include a one-paragraph "what I learned" note at the top of the file in a docstring.

## Background — the three filters

### Luma grayscale

You already wrote this in Exercise 3. Multiply each channel by its luma weight, sum across channels, clip, cast to `uint8`. Three lines.

### 3×3 box blur

A box blur replaces every pixel with the average of itself and its eight neighbours:

```
output[i, j] = mean of input[i-1..i+1, j-1..j+1]
```

The naive implementation is a double `for` loop. You are forbidden from writing it.

The vectorized implementation uses **`np.pad` + nine shifted views + average**:

```
padded = np.pad(gray, 1, mode='edge')                 # shape (H+2, W+2)

# nine shifted views into the padded array
top_left     = padded[0:H,   0:W]
top          = padded[0:H,   1:W+1]
top_right    = padded[0:H,   2:W+2]
left         = padded[1:H+1, 0:W]
center       = padded[1:H+1, 1:W+1]
right        = padded[1:H+1, 2:W+2]
bottom_left  = padded[2:H+2, 0:W]
bottom       = padded[2:H+2, 1:W+1]
bottom_right = padded[2:H+2, 2:W+2]
```

Stack them along a new axis and take the mean. **Cast to `float32` before averaging** — averaging `uint8`s will overflow silently.

Alternative (more elegant): use `np.lib.stride_tricks.sliding_window_view(padded, (3, 3))` to get a `(H, W, 3, 3)` view in one call, then `mean(axis=(-2, -1))`. The `sliding_window_view` approach is what real code uses; the nine-slice version is what real code *was* before that helper landed.

### Sobel edge detection

Sobel applies two convolutions to the grayscale image:

```
            [[-1,  0,  1],            [[-1, -2, -1],
   Kx =      [-2,  0,  2],     Ky =    [ 0,  0,  0],
             [-1,  0,  1]]             [ 1,  2,  1]]
```

`Kx` responds to horizontal gradients (edges going up-and-down); `Ky` responds to vertical gradients. The edge magnitude is:

```
G = sqrt(Gx**2 + Gy**2)
```

You can implement convolution with the same shifted-views trick, or with `sliding_window_view` + `(window * kernel).sum(axis=(-2, -1))`. The latter is the canonical vectorized convolution and the pattern you will reuse in Week 9 when we get to CNNs.

After convolution, normalize the magnitude into `[0, 255]` and cast to `uint8`.

## Suggested layout

```python
"""Pure-NumPy image filters: grayscale, box blur, Sobel edges."""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def luma_grayscale(img: np.ndarray) -> np.ndarray:
    ...

def box_blur(gray: np.ndarray) -> np.ndarray:
    ...

def sobel_edges(gray: np.ndarray) -> np.ndarray:
    ...


def main(path: str) -> None:
    img = iio.imread(path)
    if img.shape[-1] == 4:
        img = img[..., :3]
    gray = luma_grayscale(img)
    blur = box_blur(gray)
    edges = sobel_edges(gray)
    out_dir = Path(path).parent
    iio.imwrite(out_dir / "grayscale.png", gray)
    iio.imwrite(out_dir / "blur.png", blur)
    iio.imwrite(out_dir / "edges.png", edges)


if __name__ == "__main__":
    main(sys.argv[1])
```

## Hints

<details>
<summary>How big is your image?</summary>

If your source image is 4000 × 3000, a `(H, W, 3, 3)` `sliding_window_view` is 9× the size in memory. NumPy handles it (it is a view, not a copy, of the underlying buffer), but the subsequent `mean` / `sum` allocates the result. For a quick sanity check, resize to ~512 px on the long side first, or just pick a small input image.

</details>

<details>
<summary>If your edges look "noisy"</summary>

That is correct behavior on a sharp natural image — Sobel responds to every gradient, including JPEG compression artifacts. A common follow-up is to blur *first*, then take Sobel of the blurred image. Try both and compare.

</details>

<details>
<summary>If your blur output looks dark</summary>

You forgot to cast to `float32` before averaging. `np.mean` of a `uint8` array still does the right thing internally, but a mistake like averaging then casting to `uint8` *before* the mean will overflow. Check dtypes at each step.

</details>

## Stretch goals

- Add a `--sigma=N.N` flag and replace the box blur with a Gaussian blur. The Gaussian kernel is `np.exp(-(x**2 + y**2) / (2 * sigma**2))`, normalized to sum to 1. The vectorized application is identical — same `sliding_window_view + multiply + sum` pattern.
- Add a `--kernel=sobel|prewitt|scharr` flag with three edge operators. The structure is identical; only the 3×3 kernel changes.
- Time your blur with `time.perf_counter()`. Try the nine-slice version, the `sliding_window_view` version, and a naïve double-`for`-loop version *if you can stand to wait*. Report the speedup.

## Why this matters

Every image-processing library in Python (PIL, OpenCV, `skimage`) is internally exactly what you are about to write. The vectorized convolution pattern — `sliding_window_view` + multiply + sum — is the pattern that becomes a CNN layer in Week 9 once we add learned weights and stack many of them.

If you finish this, you will never again think of a convolution as something mysterious that PyTorch does for you. It is a sum of products over a sliding window. You implemented it.

## Submission

Commit `image_filter.py` and the three output PNGs to your Week 1 repo. In the commit message, paste the wall-clock time of one full run.
