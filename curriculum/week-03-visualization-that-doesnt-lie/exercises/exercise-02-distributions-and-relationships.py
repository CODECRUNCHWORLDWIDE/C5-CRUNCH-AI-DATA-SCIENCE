"""
Exercise 2 -- Distributions and relationships.

Goal: take a small synthetic "penguins-like" DataFrame and produce four
charts: a histogram, a boxplot-per-group, a scatter, and a hexbin. Pick
the right chart for each of the four questions in the docstrings below.

Estimated time: 50 minutes.

Run with:    python exercise-02-distributions-and-relationships.py
Or test:     pytest exercise-02-distributions-and-relationships.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- Four PNGs appear next to this script: _hist.png, _box.png,
  _scatter.png, _hexbin.png.
- The pytest functions all pass.

The exercise builds its own deterministic synthetic data so the tests
never depend on seaborn's load_dataset network call. The shape is
inspired by the Palmer Penguins dataset:
    https://github.com/allisonhorst/palmerpenguins

Do not look at the HINT block at the bottom until you have tried for
fifteen minutes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive, safe for CI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent

PLOT_VIOLET = "#7C3AED"
PLOT_VIOLET_DEEP = "#5B21B6"
GRAY = "#9CA3AF"

SPECIES = ("Adelie", "Chinstrap", "Gentoo")


# -----------------------------------------------------------------------------
# Deterministic data
# -----------------------------------------------------------------------------

def make_penguin_like_df(n_per_species: int = 100, seed: int = 7) -> pd.DataFrame:
    """Build a small synthetic DataFrame that imitates Palmer Penguins.

    Columns: species (category), bill_length_mm, bill_depth_mm,
    flipper_length_mm, body_mass_g. Each species has different means so
    the relationship charts have visible structure.
    """
    rng = np.random.default_rng(seed)
    frames = []
    means = {
        # Roughly the real Palmer Penguins per-species means.
        "Adelie":    {"bill_length": 38.8, "bill_depth": 18.3,
                      "flipper": 190.0, "mass": 3700.0},
        "Chinstrap": {"bill_length": 48.8, "bill_depth": 18.4,
                      "flipper": 195.0, "mass": 3733.0},
        "Gentoo":    {"bill_length": 47.5, "bill_depth": 15.0,
                      "flipper": 217.0, "mass": 5076.0},
    }
    for species, mu in means.items():
        n = n_per_species
        frames.append(pd.DataFrame({
            "species":            [species] * n,
            "bill_length_mm":     rng.normal(mu["bill_length"], 2.0, size=n),
            "bill_depth_mm":      rng.normal(mu["bill_depth"], 1.0, size=n),
            "flipper_length_mm":  rng.normal(mu["flipper"], 5.0, size=n),
            "body_mass_g":        rng.normal(mu["mass"], 300.0, size=n),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["species"] = df["species"].astype("category")
    return df


# -----------------------------------------------------------------------------
# Part A -- Distribution of one variable
# -----------------------------------------------------------------------------

def plot_body_mass_histogram(df: pd.DataFrame, out_path: Path) -> plt.Figure:
    """QUESTION: "What is the distribution of body mass across all penguins?"

    Right answer: a histogram of a single numeric column.

    Requirements:
    - 30 bins.
    - Color: PLOT_VIOLET.
    - xlabel "Body mass (g)"; ylabel "Number of penguins".
    - Title (sentence form): a one-line statement of what the chart shows.
      Example acceptable: "Body mass is roughly bimodal across species".
    - Save to out_path at dpi=150.
    - Return the Figure (for tests).
    """
    # TODO: implement using fig, ax = plt.subplots(...) and ax.hist(...).
    raise NotImplementedError("Part A -- plot_body_mass_histogram")


# -----------------------------------------------------------------------------
# Part B -- Distribution per group
# -----------------------------------------------------------------------------

def plot_mass_by_species_box(df: pd.DataFrame, out_path: Path) -> plt.Figure:
    """QUESTION: "How does body mass differ across the three species?"

    Right answer: a box plot, one box per species.

    Requirements:
    - One box per species, in alphabetical order.
    - Use matplotlib's ax.boxplot (no seaborn required, though seaborn is
      welcome -- see the hint at the bottom).
    - xlabel "Species"; ylabel "Body mass (g)".
    - Title (sentence form): one sentence on which species is heaviest.
      Example: "Gentoo penguins are the heaviest of the three species".
    - Save to out_path at dpi=150.
    - Return the Figure.
    """
    # TODO: implement.
    raise NotImplementedError("Part B -- plot_mass_by_species_box")


# -----------------------------------------------------------------------------
# Part C -- Relationship of two variables
# -----------------------------------------------------------------------------

def plot_flipper_vs_mass_scatter(df: pd.DataFrame, out_path: Path) -> plt.Figure:
    """QUESTION: "How does flipper length co-vary with body mass?"

    Right answer: a scatter plot, with species as the third dimension on
    color.

    Requirements:
    - One scatter per species, plotted in a loop. Use three distinct
      colors: PLOT_VIOLET, PLOT_VIOLET_DEEP, GRAY.
    - Add a legend (one entry per species).
    - xlabel "Flipper length (mm)"; ylabel "Body mass (g)".
    - Title (sentence form): one sentence on the relationship.
      Example: "Flipper length and body mass are strongly positively related".
    - Marker size s=18, alpha=0.7.
    - Save to out_path at dpi=150.
    - Return the Figure.
    """
    # TODO: implement using ax.scatter once per species.
    raise NotImplementedError("Part C -- plot_flipper_vs_mass_scatter")


# -----------------------------------------------------------------------------
# Part D -- Dense relationship: hexbin
# -----------------------------------------------------------------------------

def plot_dense_hexbin(out_path: Path, n: int = 20_000, seed: int = 0) -> plt.Figure:
    """QUESTION: "When n is large, the scatter plot is a solid blob -- what is
    the density structure of the underlying joint distribution?"

    Right answer: a hexbin (2-D histogram on a hexagonal grid).

    Requirements:
    - Generate n samples from a 2-D normal with a small positive correlation
      using the seed provided. Acceptable shortcut:
            rng = np.random.default_rng(seed)
            x = rng.normal(size=n)
            y = 0.6 * x + rng.normal(size=n)
    - ax.hexbin(x, y, gridsize=40, cmap="viridis").
    - Add a colorbar (fig.colorbar(...)) with label "Count".
    - xlabel "x"; ylabel "y".
    - Title (sentence form): one sentence on the joint shape.
      Example: "The dense joint distribution shows a clear positive trend".
    - Save to out_path at dpi=150.
    - Return the Figure.
    """
    # TODO: implement using ax.hexbin and fig.colorbar.
    raise NotImplementedError("Part D -- plot_dense_hexbin")


# -----------------------------------------------------------------------------
# Glue
# -----------------------------------------------------------------------------

def run_pipeline() -> dict[str, Path]:
    """Run all four charts and return the dict of output paths."""
    df = make_penguin_like_df()
    out = {
        "hist":    HERE / "_hist.png",
        "box":     HERE / "_box.png",
        "scatter": HERE / "_scatter.png",
        "hexbin":  HERE / "_hexbin.png",
    }
    for p in out.values():
        if p.exists():
            p.unlink()
    f1 = plot_body_mass_histogram(df, out["hist"])
    f2 = plot_mass_by_species_box(df, out["box"])
    f3 = plot_flipper_vs_mass_scatter(df, out["scatter"])
    f4 = plot_dense_hexbin(out["hexbin"])
    for f in (f1, f2, f3, f4):
        plt.close(f)
    return out


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_histogram_outputs() -> None:
    df = make_penguin_like_df()
    out_path = HERE / "_hist.png"
    if out_path.exists():
        out_path.unlink()
    fig = plot_body_mass_histogram(df, out_path)
    assert out_path.exists() and out_path.stat().st_size > 2000
    ax = fig.axes[0]
    assert ax.get_xlabel().strip(), "histogram has no xlabel"
    assert ax.get_ylabel().strip(), "histogram has no ylabel"
    assert ax.get_title().strip(), "histogram has no title"
    # ax.hist creates Rectangle patches.
    assert len(ax.patches) >= 10, "expected >=10 histogram bins; got fewer"
    plt.close(fig)


def test_box_outputs() -> None:
    df = make_penguin_like_df()
    out_path = HERE / "_box.png"
    if out_path.exists():
        out_path.unlink()
    fig = plot_mass_by_species_box(df, out_path)
    assert out_path.exists() and out_path.stat().st_size > 2000
    ax = fig.axes[0]
    assert ax.get_xlabel().strip()
    assert ax.get_ylabel().strip()
    assert ax.get_title().strip()
    plt.close(fig)


def test_scatter_outputs() -> None:
    df = make_penguin_like_df()
    out_path = HERE / "_scatter.png"
    if out_path.exists():
        out_path.unlink()
    fig = plot_flipper_vs_mass_scatter(df, out_path)
    assert out_path.exists() and out_path.stat().st_size > 2000
    ax = fig.axes[0]
    assert ax.get_xlabel().strip()
    assert ax.get_ylabel().strip()
    assert ax.get_title().strip()
    # Three species -> at least three scatter PathCollections.
    assert len(ax.collections) >= 3, (
        f"expected one scatter call per species (>=3 collections); "
        f"got {len(ax.collections)}"
    )
    # Legend should be present.
    assert ax.get_legend() is not None, "scatter chart needs a legend"
    plt.close(fig)


def test_hexbin_outputs() -> None:
    out_path = HERE / "_hexbin.png"
    if out_path.exists():
        out_path.unlink()
    fig = plot_dense_hexbin(out_path)
    assert out_path.exists() and out_path.stat().st_size > 2000
    # The first Axes is the hexbin; there should be at least 2 Axes total
    # (one for hexbin, one for the colorbar).
    assert len(fig.axes) >= 2, "hexbin chart should include a colorbar"
    ax = fig.axes[0]
    assert ax.get_xlabel().strip()
    assert ax.get_ylabel().strip()
    assert ax.get_title().strip()
    plt.close(fig)


def _run_all_tests() -> None:
    test_histogram_outputs()
    test_box_outputs()
    test_scatter_outputs()
    test_hexbin_outputs()
    print("OK -- exercise 2")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# plot_body_mass_histogram:
#     fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
#     ax.hist(df["body_mass_g"], bins=30, color=PLOT_VIOLET)
#     ax.set_xlabel("Body mass (g)")
#     ax.set_ylabel("Number of penguins")
#     ax.set_title("Body mass spans about 2.7-6.3 kg across species")
#     fig.savefig(out_path, dpi=150)
#     return fig
#
# plot_mass_by_species_box:
#     fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
#     species = sorted(df["species"].unique())
#     groups = [df.loc[df["species"] == s, "body_mass_g"].to_numpy()
#               for s in species]
#     ax.boxplot(groups, tick_labels=list(species))
#     ax.set_xlabel("Species")
#     ax.set_ylabel("Body mass (g)")
#     ax.set_title("Gentoo penguins are noticeably heavier than the other two")
#     fig.savefig(out_path, dpi=150)
#     return fig
#
# plot_flipper_vs_mass_scatter:
#     fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
#     colors = {"Adelie": PLOT_VIOLET, "Chinstrap": PLOT_VIOLET_DEEP,
#               "Gentoo": GRAY}
#     for s, c in colors.items():
#         sub = df[df["species"] == s]
#         ax.scatter(sub["flipper_length_mm"], sub["body_mass_g"],
#                    s=18, alpha=0.7, color=c, label=s)
#     ax.set_xlabel("Flipper length (mm)")
#     ax.set_ylabel("Body mass (g)")
#     ax.set_title("Flipper length and body mass are strongly positively related")
#     ax.legend(frameon=False)
#     fig.savefig(out_path, dpi=150)
#     return fig
#
# plot_dense_hexbin:
#     rng = np.random.default_rng(seed)
#     x = rng.normal(size=n)
#     y = 0.6 * x + rng.normal(size=n)
#     fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
#     hb = ax.hexbin(x, y, gridsize=40, cmap="viridis")
#     cbar = fig.colorbar(hb, ax=ax)
#     cbar.set_label("Count")
#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_title("Dense joint distribution shows a clear positive trend")
#     fig.savefig(out_path, dpi=150)
#     return fig
#
# -----------------------------------------------------------------------------
