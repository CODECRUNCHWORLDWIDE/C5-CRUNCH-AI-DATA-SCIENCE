"""
Exercise 3 -- Time series.

Goal: produce one publication-quality line chart from a deterministic
monthly time series. The chart must have a real date axis (not raw
integers or timestamps), a single annotation pointing to the largest
year-on-year drop, and a caption with the data source.

Estimated time: 40 minutes.

Run with:    python exercise-03-time-series.py
Or test:     pytest exercise-03-time-series.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- A file _timeseries.png appears next to this script.
- The pytest functions all pass.

The exercise builds its own deterministic time series in the shape of
"global air passengers, monthly, 2015..2024" with a pandemic-like dip
in 2020. No network needed.

Do not look at the HINT block at the bottom until you have tried for
fifteen minutes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).parent
PNG_PATH = HERE / "_timeseries.png"

PLOT_VIOLET = "#7C3AED"
PLOT_VIOLET_DEEP = "#5B21B6"
GRAY = "#9CA3AF"


# -----------------------------------------------------------------------------
# Deterministic data
# -----------------------------------------------------------------------------

def make_monthly_series(seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame with columns 'date' (monthly, 2015-01..2024-12)
    and 'passengers' (in millions). 2020 has a sharp pandemic-like drop;
    2022 onward recovers.

    The shape is: linear growth + seasonal cycle + a 2020 dip.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
    n = len(dates)
    t = np.arange(n)

    # Linear growth from 250 to 400 over 10 years.
    trend = 250.0 + (400.0 - 250.0) * t / (n - 1)

    # Seasonal component (peak in July-August).
    month = dates.month.to_numpy()
    seasonal = 25.0 * np.sin(2.0 * np.pi * (month - 4) / 12.0)

    # Pandemic dip in 2020.
    year = dates.year.to_numpy()
    dip = np.zeros(n)
    is_2020 = year == 2020
    is_2021 = year == 2021
    dip[is_2020] = -180.0
    dip[is_2021] = -90.0

    # Small noise.
    noise = rng.normal(scale=4.0, size=n)

    passengers = trend + seasonal + dip + noise
    return pd.DataFrame({"date": dates, "passengers": passengers})


# -----------------------------------------------------------------------------
# Part A -- Compute the biggest year-on-year drop
# -----------------------------------------------------------------------------

def biggest_yoy_drop(df: pd.DataFrame) -> tuple[pd.Timestamp, float, float]:
    """Find the month with the largest year-on-year decline in passengers.

    Year-on-year change is: this month - same month one year earlier.

    Return a tuple of:
        (date_of_max_drop, this_value, year_ago_value)

    The drop is negative (this_value < year_ago_value). For our synthetic
    data the answer is somewhere in spring 2020.

    Requirements:
    - Use df.set_index("date") and a 12-month shift (passengers.shift(12))
      to compute year-ago.
    - Use argmin/idxmin on the difference, not a Python for-loop.
    """
    # TODO: implement.
    raise NotImplementedError("Part A -- biggest_yoy_drop")


# -----------------------------------------------------------------------------
# Part B -- Build the chart
# -----------------------------------------------------------------------------

def plot_time_series(df: pd.DataFrame, out_path: Path) -> plt.Figure:
    """Plot the monthly passengers series as a line chart with:

    - figsize=(9, 4), layout="constrained".
    - The single line in brand violet (PLOT_VIOLET), linewidth=1.8.
    - X-axis: years as major ticks, formatted as 4-digit years.
      Use matplotlib.dates.YearLocator() and DateFormatter("%Y").
    - X label "Year".
    - Y label "Passengers (millions)".
    - Title (sentence form): one sentence on what the chart shows. The
      pandemic drop is the obvious takeaway.
    - The top and right spines hidden (editorial look).
    - One annotation, with an arrow, pointing at the biggest year-on-year
      drop. The label text should be:
            f"{drop_date:%Y-%m}: {pct_drop:+.0f}%% YoY"
      where pct_drop = 100 * (this_value - year_ago) / year_ago.
      The arrow's xytext should sit above the dip, in data coordinates.
    - A caption at (0.99, -0.05) of ax.transAxes, right-aligned, gray, 8pt:
        "Source: synthetic monthly series -- exercise 3, week 3"

    Save the Figure with dpi=150 and return it.
    """
    # TODO: implement.
    raise NotImplementedError("Part B -- plot_time_series")


# -----------------------------------------------------------------------------
# Glue
# -----------------------------------------------------------------------------

def run_pipeline() -> Path:
    df = make_monthly_series()
    if PNG_PATH.exists():
        PNG_PATH.unlink()
    fig = plot_time_series(df, PNG_PATH)
    plt.close(fig)
    return PNG_PATH


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_data_shape() -> None:
    df = make_monthly_series()
    # 10 years, 12 months each = 120 rows.
    assert len(df) == 120, f"expected 120 rows, got {len(df)}"
    assert (df["date"].dt.day == 1).all(), "dates should be month-starts"
    # Range of passengers is plausible (somewhere in 50..500 millions).
    assert df["passengers"].min() > 50.0
    assert df["passengers"].max() < 500.0


def test_biggest_yoy_drop_is_in_2020() -> None:
    df = make_monthly_series()
    date, this_v, year_ago = biggest_yoy_drop(df)
    assert isinstance(date, pd.Timestamp), f"expected Timestamp; got {type(date)}"
    assert date.year == 2020, f"expected the YoY drop in 2020; got {date.year}"
    assert this_v < year_ago, (
        f"YoY drop should be negative: this={this_v}, year_ago={year_ago}"
    )


def test_png_written() -> None:
    path = run_pipeline()
    assert path.exists()
    assert path.stat().st_size > 4000, (
        f"PNG is suspiciously small: {path.stat().st_size} bytes"
    )


def test_chart_decorations() -> None:
    df = make_monthly_series()
    fig = plot_time_series(df, PNG_PATH)
    ax = fig.axes[0]
    assert ax.get_xlabel().strip(), "missing xlabel"
    assert ax.get_ylabel().strip(), "missing ylabel"
    assert ax.get_title().strip(), "missing title"
    # Top and right spines should be hidden.
    assert not ax.spines["top"].get_visible(), (
        "top spine should be hidden for editorial style"
    )
    assert not ax.spines["right"].get_visible(), (
        "right spine should be hidden for editorial style"
    )
    # At least one annotation (the arrow + label) should be on the Axes.
    # ax.texts holds the annotation labels; the arrow lives on the
    # annotation Artist itself.
    has_annotation = any(
        t.arrow_patch is not None for t in ax.texts
        if hasattr(t, "arrow_patch")
    )
    assert has_annotation or len(ax.texts) >= 1, (
        "expected at least one annotation pointing at the YoY drop"
    )
    plt.close(fig)


def _run_all_tests() -> None:
    test_data_shape()
    test_biggest_yoy_drop_is_in_2020()
    test_png_written()
    test_chart_decorations()
    print("OK -- exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# biggest_yoy_drop:
#     s = df.set_index("date")["passengers"]
#     yoy = s - s.shift(12)
#     drop_date = yoy.idxmin()
#     this_v = float(s.loc[drop_date])
#     year_ago = float(s.loc[drop_date - pd.DateOffset(years=1)])
#     return drop_date, this_v, year_ago
#
# plot_time_series:
#     drop_date, this_v, year_ago = biggest_yoy_drop(df)
#     pct = 100.0 * (this_v - year_ago) / year_ago
#
#     fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
#     ax.plot(df["date"], df["passengers"], color=PLOT_VIOLET, linewidth=1.8)
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Passengers (millions)")
#     ax.set_title(
#         "A pandemic-shaped dip in 2020 is the dominant feature of the decade"
#     )
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#
#     ax.xaxis.set_major_locator(mdates.YearLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
#
#     ax.annotate(
#         f"{drop_date:%Y-%m}: {pct:+.0f}% YoY",
#         xy=(drop_date, this_v),
#         xytext=(drop_date, this_v + 80),
#         arrowprops=dict(arrowstyle="->", color="gray"),
#         fontsize=10, color="black",
#     )
#     ax.text(0.99, -0.05,
#             "Source: synthetic monthly series -- exercise 3, week 3",
#             transform=ax.transAxes, ha="right", fontsize=8, color="gray")
#     fig.savefig(out_path, dpi=150)
#     return fig
#
# -----------------------------------------------------------------------------
