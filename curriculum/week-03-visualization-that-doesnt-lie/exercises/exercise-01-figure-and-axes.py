"""
Exercise 1 -- Figure and Axes.

Goal: build a 2x2 subplot grid using the matplotlib object-oriented API
from Lecture 1. Plot a different artist type on each Axes (line, hist,
bar, scatter), label every axis, give the Figure a sentence-form
suptitle, and save the result to a PNG that passes the publication
checklist.

Estimated time: 40 minutes.

Run with:    python exercise-01-figure-and-axes.py
Or test:     pytest exercise-01-figure-and-axes.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- A file _grid.png appears next to this script.
- The pytest functions all pass.

The exercise builds its own deterministic data so the tests never depend
on the network. Do not look at the HINT block at the bottom until you
have tried for fifteen minutes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in CI and on headless boxes.
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent
PNG_PATH = HERE / "_grid.png"

# Brand palette (from BRAND.md).
PLOT_VIOLET = "#7C3AED"
PLOT_VIOLET_DEEP = "#5B21B6"
GRAY = "#9CA3AF"


# -----------------------------------------------------------------------------
# Deterministic data
# -----------------------------------------------------------------------------

def make_data() -> dict[str, np.ndarray]:
    """Return a dict of four small NumPy arrays for the four subplots.

    Keys:
        line_x, line_y       -- x and y for a sine line (200 points, 0..4*pi).
        hist_values          -- 1000 samples from a standard normal.
        bar_labels, bar_h    -- four categorical labels and their bar heights.
        scatter_x, scatter_y -- 200 uniformly random points in the unit square.
    """
    rng = np.random.default_rng(42)
    line_x = np.linspace(0.0, 4.0 * np.pi, 200)
    line_y = np.sin(line_x)
    hist_values = rng.normal(size=1000)
    bar_labels = np.array(["A", "B", "C", "D"])
    bar_h = np.array([3.0, 1.0, 4.0, 1.5])
    scatter_x = rng.uniform(size=200)
    scatter_y = rng.uniform(size=200)
    return {
        "line_x": line_x,
        "line_y": line_y,
        "hist_values": hist_values,
        "bar_labels": bar_labels,
        "bar_h": bar_h,
        "scatter_x": scatter_x,
        "scatter_y": scatter_y,
    }


# -----------------------------------------------------------------------------
# Part A -- Build the Figure and Axes grid
# -----------------------------------------------------------------------------

def build_grid() -> tuple[plt.Figure, np.ndarray]:
    """Create a Figure with a 2x2 grid of Axes and constrained layout.

    Requirements:
    - figsize is 10 inches wide and 6 inches tall.
    - layout is "constrained" (the modern replacement for tight_layout).
    - Return (fig, axes) where axes is the 2-D numpy array of Axes.
    """
    # TODO: implement using plt.subplots with nrows=2, ncols=2.
    raise NotImplementedError("Part A -- build_grid")


# -----------------------------------------------------------------------------
# Part B -- Populate each subplot
# -----------------------------------------------------------------------------

def populate_axes(axes: np.ndarray, data: dict[str, np.ndarray]) -> None:
    """Draw one artist family on each of the four Axes.

    Layout:
        axes[0, 0] -- a line of sin(x) in brand violet, linewidth=2.
        axes[0, 1] -- a histogram of `hist_values` with 30 bins, color=GRAY.
        axes[1, 0] -- a vertical bar chart of `bar_labels` vs `bar_h` in
                      brand violet.
        axes[1, 1] -- a scatter plot of (scatter_x, scatter_y), s=10,
                      color=PLOT_VIOLET_DEEP, alpha=0.7.

    For each Axes you MUST set:
    - set_title with a one-sentence-ish description (not the column name).
    - set_xlabel and set_ylabel in plain English with units where they make
      sense ("x (radians)", "count", "category", "y").

    Do NOT touch the Figure suptitle here; that is Part C.
    """
    # TODO: implement.
    raise NotImplementedError("Part B -- populate_axes")


# -----------------------------------------------------------------------------
# Part C -- Suptitle, caption, save
# -----------------------------------------------------------------------------

def finish_and_save(fig: plt.Figure, axes: np.ndarray, out_path: Path) -> None:
    """Add a Figure-level title, a caption in the lower-right corner, and
    save the Figure as PNG at 150 dpi.

    Requirements:
    - fig.suptitle is "Four chart families on one Figure".
    - Add a caption with fig.text at (0.99, 0.01), right-aligned, gray, 8pt:
        "Source: synthetic, seed=42 -- exercise 1, week 3"
    - Save with fig.savefig(out_path, dpi=150). bbox_inches="tight" is OK
      but constrained layout already handles padding.
    """
    # TODO: implement.
    raise NotImplementedError("Part C -- finish_and_save")


# -----------------------------------------------------------------------------
# Glue: one function that runs the whole pipeline.
# -----------------------------------------------------------------------------

def run_pipeline() -> tuple[plt.Figure, np.ndarray, Path]:
    """End-to-end: build, populate, save. Returns the artifacts for tests."""
    data = make_data()
    fig, axes = build_grid()
    populate_axes(axes, data)
    finish_and_save(fig, axes, PNG_PATH)
    return fig, axes, PNG_PATH


# -----------------------------------------------------------------------------
# Pytest-style checks (also run when the file is executed directly).
# -----------------------------------------------------------------------------

def test_grid_shape() -> None:
    data = make_data()
    fig, axes = build_grid()
    assert isinstance(fig, plt.Figure)
    assert axes.shape == (2, 2), f"axes shape is {axes.shape}, expected (2, 2)"
    plt.close(fig)


def test_each_axes_has_title_and_labels() -> None:
    data = make_data()
    fig, axes = build_grid()
    populate_axes(axes, data)
    for (i, j), ax in np.ndenumerate(axes):
        assert ax.get_title().strip(), f"axes[{i}, {j}] has no title"
        assert ax.get_xlabel().strip(), f"axes[{i}, {j}] has no xlabel"
        assert ax.get_ylabel().strip(), f"axes[{i}, {j}] has no ylabel"
    plt.close(fig)


def test_artists_present() -> None:
    """Each Axes should have *some* drawn artist of the right family."""
    data = make_data()
    fig, axes = build_grid()
    populate_axes(axes, data)
    # axes[0, 0] is a line: at least one Line2D from ax.plot.
    assert axes[0, 0].lines, "axes[0, 0] has no Line2D"
    # axes[0, 1] is a hist: hist patches are Rectangles in ax.patches.
    assert len(axes[0, 1].patches) > 0, "axes[0, 1] has no histogram patches"
    # axes[1, 0] is a bar: at least one Rectangle in patches too.
    assert len(axes[1, 0].patches) > 0, "axes[1, 0] has no bar patches"
    # axes[1, 1] is a scatter: collections, not lines.
    assert len(axes[1, 1].collections) > 0, "axes[1, 1] has no scatter"
    plt.close(fig)


def test_png_written_and_nontrivial() -> None:
    if PNG_PATH.exists():
        PNG_PATH.unlink()
    fig, _, path = run_pipeline()
    assert path.exists(), "_grid.png was not written"
    # A trivial blank PNG at this size is < 1 KB; ours should be a few KB.
    size_bytes = path.stat().st_size
    assert size_bytes > 2000, f"PNG is suspiciously small: {size_bytes} bytes"
    plt.close(fig)


def test_suptitle_present() -> None:
    fig, _, _ = run_pipeline()
    suptitle = fig._suptitle  # type: ignore[attr-defined]
    assert suptitle is not None and suptitle.get_text().strip(), (
        "Figure suptitle is missing"
    )
    plt.close(fig)


def _run_all_tests() -> None:
    test_grid_shape()
    test_each_axes_has_title_and_labels()
    test_artists_present()
    test_png_written_and_nontrivial()
    test_suptitle_present()
    print("OK -- exercise 1")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# build_grid:
#     fig, axes = plt.subplots(
#         nrows=2, ncols=2,
#         figsize=(10, 6),
#         layout="constrained",
#     )
#     return fig, axes
#
# populate_axes:
#     ax = axes[0, 0]
#     ax.plot(data["line_x"], data["line_y"],
#             color=PLOT_VIOLET, linewidth=2)
#     ax.set_title("sin(x) over four periods")
#     ax.set_xlabel("x (radians)")
#     ax.set_ylabel("sin(x)")
#
#     ax = axes[0, 1]
#     ax.hist(data["hist_values"], bins=30, color=GRAY)
#     ax.set_title("Standard normal samples, n=1000")
#     ax.set_xlabel("value")
#     ax.set_ylabel("count")
#
#     ax = axes[1, 0]
#     ax.bar(data["bar_labels"], data["bar_h"], color=PLOT_VIOLET)
#     ax.set_title("Four categories, varying heights")
#     ax.set_xlabel("category")
#     ax.set_ylabel("height")
#
#     ax = axes[1, 1]
#     ax.scatter(data["scatter_x"], data["scatter_y"],
#                s=10, color=PLOT_VIOLET_DEEP, alpha=0.7)
#     ax.set_title("200 uniform points in the unit square")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#
# finish_and_save:
#     fig.suptitle("Four chart families on one Figure")
#     fig.text(0.99, 0.01,
#              "Source: synthetic, seed=42 -- exercise 1, week 3",
#              ha="right", color="gray", fontsize=8)
#     fig.savefig(out_path, dpi=150)
#
# -----------------------------------------------------------------------------
