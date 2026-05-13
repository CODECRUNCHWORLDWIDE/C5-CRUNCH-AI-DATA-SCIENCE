# Mini-Project — Vectorized Image Processing in Pure NumPy

> Build a small image-processing pipeline — grayscale, blur, edge detection, thresholding, and one filter of your choice — implemented from first principles in NumPy. The deliverable is a Jupyter notebook a reviewer can step through end-to-end.

This is the first artifact of your `crunch-ai-portfolio-<yourhandle>` repository. Every later week adds one. By Week 12 you will have eleven. Recruiters will read this one because it is the simplest; do it well.

**Estimated time:** 7 hours, spread across Thursday–Saturday.

---

## What you will build

A Jupyter notebook `image_pipeline.ipynb` that, on a real RGB photo of your choice:

1. **Loads** the image as an `(H, W, 3)` `uint8` NumPy array using `imageio` and shows it.
2. **Inspects** the array: shape, dtype, strides, min, max, mean per channel — all printed.
3. **Converts to grayscale** using ITU-R BT.601 luma weights.
4. **Applies a 3×3 box blur** to the grayscale image (vectorized; no per-pixel loop).
5. **Applies Sobel edge detection** to the blurred image (vectorized).
6. **Thresholds** the edge map into a binary mask.
7. **Implements one filter of your choice**, from the list below or one you propose. Pick something that exercises a NumPy feature.
8. **Renders every stage** with `matplotlib` so a reviewer can scroll the notebook top-to-bottom and see the visual progression.
9. **Ends with a short reflection** (~200 words) on what surprised you, where you were tempted to write a loop, and what you would change.

The pipeline must run on any reasonable input image (≤ 4000 px on the long side) in under 30 seconds on a laptop.

---

## Acceptance criteria

- [ ] A new public GitHub repo `crunch-ai-portfolio-<yourhandle>` exists, with `week-01/` containing this notebook.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `numpy>=2.0,<3`, `imageio>=2.34`, `matplotlib>=3.8`, `jupyter`.
- [ ] `jupyter nbconvert --to notebook --execute week-01/image_pipeline.ipynb` runs end-to-end without errors on a fresh clone.
- [ ] **Zero** Python `for` loops over pixel coordinates in the notebook. (One `for` loop iterating over a *list of filters* is fine; the body of that loop must be vectorized.)
- [ ] **Only** `numpy`, `imageio`, and `matplotlib` are imported. No PIL, no OpenCV, no `scipy`.
- [ ] Every cell that does NumPy work prints either an array's shape, an array's dtype, or both — explicitly. This is the discipline: name your shapes.
- [ ] The notebook is committed with **outputs cleared** (run-from-scratch is the test), and a separate `image_pipeline.html` export is committed alongside so reviewers can preview without running.
- [ ] A `README.md` in `week-01/` explains: setup, the input image's source and license, and a one-line summary of each stage.

---

## Suggested layout

```
crunch-ai-portfolio-<yourhandle>/
├── README.md                    ← portfolio root: what this repo is
└── week-01/
    ├── README.md                ← week-1 explainer
    ├── requirements.txt
    ├── input.jpg                ← your input image, with a LICENSE note
    ├── image_pipeline.ipynb
    └── image_pipeline.html      ← exported preview
```

---

## Suggested order of operations

### Phase 1 — Setup (45 min)

1. Make the portfolio repo, public, on GitHub. Add the GPL-3.0 license to match C5.
2. Create `week-01/` with a `requirements.txt` and a placeholder `image_pipeline.ipynb`.
3. Find an input image **you actually have rights to use**: your own photo, a Creative Commons / public-domain image (Wikimedia Commons is the easy source), or NASA's open image library. Put the URL and license in `week-01/README.md`. Commit.
4. Confirm `jupyter notebook` opens the file and the first cell `import numpy as np; print(np.__version__)` runs and prints `2.x`.

### Phase 2 — Inspect (1 h)

A markdown cell titled "Loading and inspecting." Then a code cell that:

- Loads the image with `imageio.v3.imread`.
- If the image is RGBA, drops the alpha channel.
- Prints `shape`, `dtype`, `nbytes` (in MB), `mean(axis=(0, 1))` (per-channel mean), and the strides.
- Shows the image with `matplotlib.pyplot.imshow`. Set the figure size sensibly.

A short markdown cell explains, in your own words: "This image is N MB in memory. It is `uint8`, so each pixel-channel is one byte; if we had loaded it as `float32` the memory cost would be N×4 = M MB. The strides are (a, b, c), which tells me the layout is C-contiguous."

### Phase 3 — Grayscale, blur, edges (2 h)

One section per filter, each with:

- A markdown cell explaining the operation in two sentences. Cite the source for the BT.601 luma coefficients (Wikipedia or ITU). Cite Sobel's original 1968 paper *only* if you actually read the abstract; otherwise cite a working reference.
- A code cell implementing the filter in 5–20 lines.
- A code cell showing the before/after with `matplotlib`.

For the blur, the recommended vectorization is `np.lib.stride_tricks.sliding_window_view(padded, (3, 3))` to produce a `(H, W, 3, 3)` view, then `.mean(axis=(-2, -1))`. The alternative is the nine-slice pattern from the challenge brief. Implement whichever you understand; explain *why* it has no per-pixel Python loop.

For Sobel, the canonical implementation is: pad, `sliding_window_view`, multiply by the 3×3 kernel (broadcasts), `.sum(axis=(-2, -1))`. Compute Gx and Gy, then magnitude.

### Phase 4 — Thresholding and your choice (1.5 h)

The thresholding step is a `np.where`. Take the edge-magnitude image and binarize it; show both side by side.

Then implement **one** of the following (or propose your own with a justification cell):

- **Histogram and contrast stretch.** Compute the histogram of a channel with `np.bincount` (no, you may not use `np.histogram` for this one — write `bincount`-based). Apply a linear contrast stretch: rescale `[p_low, p_high]` (e.g., 2nd and 98th percentile via `np.percentile`) to `[0, 255]`.
- **Channel swap and tint.** Build a 3×3 color-mix matrix and apply it via `np.einsum('hwc,cd->hwd', img.astype(np.float32), M)` to produce a tinted image. Pick a tasteful mix; document what each row of `M` does.
- **Downsample by 2× via mean pooling.** Reshape an `(H, W)` image into `(H/2, 2, W/2, 2)`, mean across the size-2 axes, get back `(H/2, W/2)`. This is exactly the pooling layer in a CNN, just without learning. State that connection.
- **Sepia tone.** Apply the standard sepia matrix to the RGB image:
  ```
        [[0.393, 0.769, 0.189],
         [0.349, 0.686, 0.168],
         [0.272, 0.534, 0.131]]
  ```
  Same `einsum` pattern as channel-swap.

### Phase 5 — Reflection and export (1 h)

A final markdown cell, 150–250 words, answering:

- Where were you tempted to write a `for` loop?
- Which NumPy feature did the most work in this notebook?
- If you had to do this in pure Python (no NumPy), what would the slowest part be — and how slow?
- What part of this maps onto Week 9's CNN material? (Hint: the 3×3 convolution pattern.)

Export the notebook to HTML: `jupyter nbconvert --to html image_pipeline.ipynb`. Commit both files. Push.

---

## Stretch goals

- Run the same pipeline on five different images and produce a small contact sheet (one row per image, one column per stage) in a final figure.
- Add a `--profile` script next to the notebook that times each stage and prints a small table.
- Re-implement the Sobel stage using `np.einsum('hwij,ij->hw', windows, kernel)` instead of `(windows * kernel).sum(axis=(-2, -1))`. Confirm they agree. This is your first taste of `einsum` solving a real problem.
- Publish the notebook on GitHub Pages or nbviewer so non-Git users can read it.

---

## Rubric

| Criterion | Weight | "Great" looks like |
|-----------|------:|--------------------|
| It runs | 25% | Fresh clone → `pip install -r requirements.txt` → `jupyter nbconvert --execute` → no errors |
| Vectorization discipline | 25% | Zero pixel-coordinate `for` loops; every filter is shape-aligned and broadcast-clean |
| Notebook readability | 20% | Markdown explains each step; shapes and dtypes are printed at every transition |
| Visual results | 15% | The before/after pairs are clear and well-rendered; the input image is not a stock test pattern unless you say so |
| Reflection | 10% | Honest, specific, and at least one paragraph names a concrete habit you have changed |
| Stretch | 5% | One stretch goal delivered |

---

## What this exercises (and what comes next)

This mini-project exercises *every* NumPy concept from Week 1:

- `dtype` discipline (uint8 ↔ float32 conversions, clipping before cast-back).
- View vs copy (the sliding-window view is the textbook view example).
- Broadcasting (kernel × window, weights × channels).
- Reductions along a chosen axis (the `mean` and `sum` in your convolution).
- Vectorization (the *whole point*).

The Sobel + sliding-window pattern is structurally the same as a CNN convolution. In Week 9 you will add *learned* weights and stack many of these layers. Same machinery, different weights. If you build this cleanly now, that week feels like a refinement, not a new world.

---

## Submission

Push your `crunch-ai-portfolio-<yourhandle>` repo public. Paste the URL into your Week 1 review thread (or the C5 community channel, if you are doing this in a cohort). That is the artifact recruiters will see.
