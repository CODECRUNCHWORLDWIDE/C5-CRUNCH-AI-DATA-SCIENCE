"""
Exercise 1 — Load and inspect.

Goal: do the "five-minute first look" from Lecture 1 with your fingers
on a small NYC-taxi-style CSV. You will load it, inspect dtypes and
missing data, choose the right dtype for every column, and verify the
result with the built-in checks.

Estimated time: 40 minutes.

Run with:    python exercise-01-load-and-inspect.py
Or test:     pytest exercise-01-load-and-inspect.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The five pytest functions all pass.

The exercise builds its own deterministic CSV next to this file so the
tests never depend on the network. If you want the real flavor, the NYC
green-taxi Parquet files are documented at:
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Do not look at the HINT block at the bottom until you have tried for
fifteen minutes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent
CSV_PATH = HERE / "_taxi_sample.csv"


# -----------------------------------------------------------------------------
# Test-data generator (deterministic; safe to run anywhere)
# -----------------------------------------------------------------------------

def _generate_sample_csv(path: Path, n_rows: int = 500) -> None:
    """Write a small CSV that imitates the NYC TLC green-taxi schema.

    The file is small (~30 KB) and deterministic; no network needed.
    """
    rng = np.random.default_rng(42)
    pickup_ts = pd.date_range("2025-08-01", periods=n_rows, freq="17min")
    dropoff_ts = pickup_ts + pd.to_timedelta(
        rng.integers(3, 45, size=n_rows), unit="min"
    )
    df = pd.DataFrame(
        {
            "trip_id":         np.arange(1, n_rows + 1),
            "pickup_datetime":  pickup_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "dropoff_datetime": dropoff_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "passenger_count": rng.integers(1, 5, size=n_rows),
            "trip_distance":   np.round(rng.gamma(2.0, 1.5, size=n_rows), 2),
            "fare_amount":     np.round(2.5 + rng.gamma(2.0, 4.0, size=n_rows), 2),
            "payment_type":    rng.choice(
                ["card", "cash", "no charge", "dispute"],
                size=n_rows,
                p=[0.75, 0.20, 0.03, 0.02],
            ),
            "pickup_borough":  rng.choice(
                ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
                size=n_rows,
                p=[0.55, 0.25, 0.12, 0.06, 0.02],
            ),
        }
    )
    path.write_text(df.to_csv(index=False))


def _ensure_csv() -> Path:
    if not CSV_PATH.exists():
        _generate_sample_csv(CSV_PATH)
    return CSV_PATH


# -----------------------------------------------------------------------------
# Part A -- Load
# -----------------------------------------------------------------------------

def load_taxi_csv(path: Path) -> pd.DataFrame:
    """Read the CSV at `path` into a DataFrame.

    Requirements:
    - parse `pickup_datetime` and `dropoff_datetime` as datetimes.
    - cast `pickup_borough` and `payment_type` to the `category` dtype.
    - cast `passenger_count` to a small integer dtype (int8 is fine here).

    Use `pd.read_csv` with `parse_dates=` and a `dtype=` mapping; or load
    then `.astype(...)`. Either is acceptable.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- load_taxi_csv")


# -----------------------------------------------------------------------------
# Part B -- Inspect
# -----------------------------------------------------------------------------

def column_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Return a dict mapping each column name to the string form of its
    dtype (i.e. `str(df[col].dtype)`).
    """
    # TODO: implement
    raise NotImplementedError("Part B -- column_dtypes")


def memory_usage_mb(df: pd.DataFrame) -> float:
    """Return the deep memory usage of `df` in megabytes (1 MB = 1024**2 B).

    Use df.memory_usage(deep=True).sum().
    """
    # TODO: implement
    raise NotImplementedError("Part B -- memory_usage_mb")


def per_column_missing(df: pd.DataFrame) -> pd.Series:
    """Return a Series of `column -> missing_count`, sorted descending.

    Only count true missing values (NaN / NaT / pd.NA).
    """
    # TODO: implement using df.isna().sum() and sort_values.
    raise NotImplementedError("Part B -- per_column_missing")


# -----------------------------------------------------------------------------
# Part C -- Derived columns and a simple summary
# -----------------------------------------------------------------------------

def add_trip_duration_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with a new float column `trip_duration_min`
    equal to (dropoff - pickup) in minutes.

    Do NOT use .apply. Use vectorized datetime arithmetic and the
    .dt.total_seconds() accessor.
    """
    # TODO: implement. Make sure the output is a *new* DataFrame, not a
    # mutated input; pandas copy_on_write would help here but be explicit:
    # work on `df.copy()`.
    raise NotImplementedError("Part C -- add_trip_duration_minutes")


def trips_per_borough(df: pd.DataFrame) -> pd.Series:
    """Return a Series of `pickup_borough -> row_count`, sorted descending.

    Use groupby + size (NOT count, because count excludes NaN).
    """
    # TODO: implement
    raise NotImplementedError("Part C -- trips_per_borough")


# -----------------------------------------------------------------------------
# Pytest-style checks (also run when the file is executed directly).
# -----------------------------------------------------------------------------

def test_load_dtypes_correct() -> None:
    df = load_taxi_csv(_ensure_csv())
    # Datetimes parsed:
    assert pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]), \
        "pickup_datetime should be a datetime dtype"
    assert pd.api.types.is_datetime64_any_dtype(df["dropoff_datetime"]), \
        "dropoff_datetime should be a datetime dtype"
    # Categoricals:
    assert isinstance(df["pickup_borough"].dtype, pd.CategoricalDtype), \
        "pickup_borough should be category"
    assert isinstance(df["payment_type"].dtype, pd.CategoricalDtype), \
        "payment_type should be category"
    # Small int for passenger_count:
    assert pd.api.types.is_integer_dtype(df["passenger_count"])
    assert df["passenger_count"].dtype.itemsize <= 2, \
        "passenger_count fits in int8/int16; do not waste 8 bytes per row"


def test_column_dtypes_keys() -> None:
    df = load_taxi_csv(_ensure_csv())
    d = column_dtypes(df)
    assert set(d.keys()) == set(df.columns)
    for k, v in d.items():
        assert isinstance(v, str), f"{k} -> {type(v)} (should be str)"


def test_memory_usage_is_small() -> None:
    df = load_taxi_csv(_ensure_csv())
    mb = memory_usage_mb(df)
    assert 0.0 < mb < 5.0, \
        f"500 rows of well-typed data should be << 5 MB; got {mb:.3f}"


def test_per_column_missing_sorted() -> None:
    df = load_taxi_csv(_ensure_csv())
    miss = per_column_missing(df)
    assert isinstance(miss, pd.Series)
    # In our synthetic data, nothing is missing. Sorted descending means
    # the first value is >= the last.
    assert miss.iloc[0] >= miss.iloc[-1]
    assert (miss >= 0).all()


def test_add_trip_duration_minutes() -> None:
    df = load_taxi_csv(_ensure_csv())
    out = add_trip_duration_minutes(df)
    assert "trip_duration_min" in out.columns
    assert "trip_duration_min" not in df.columns, \
        "do not mutate the caller's DataFrame; work on a copy"
    # All trips should be 3..45 minutes by construction.
    d = out["trip_duration_min"]
    assert d.between(2.99, 45.01).all(), \
        f"durations out of expected range: min={d.min()}, max={d.max()}"


def test_trips_per_borough() -> None:
    df = load_taxi_csv(_ensure_csv())
    s = trips_per_borough(df)
    assert isinstance(s, pd.Series)
    assert s.sum() == len(df)
    # Manhattan is the largest by construction (p = 0.55).
    assert s.idxmax() == "Manhattan"


def _run_all_tests() -> None:
    test_load_dtypes_correct()
    test_column_dtypes_keys()
    test_memory_usage_is_small()
    test_per_column_missing_sorted()
    test_add_trip_duration_minutes()
    test_trips_per_borough()
    print("OK -- exercise 1")


if __name__ == "__main__":
    _ensure_csv()
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min)
# -----------------------------------------------------------------------------
#
# load_taxi_csv:
#     return pd.read_csv(
#         path,
#         parse_dates=["pickup_datetime", "dropoff_datetime"],
#         dtype={
#             "passenger_count": "int8",
#             "payment_type":    "category",
#             "pickup_borough":  "category",
#         },
#     )
#
# column_dtypes:
#     return {c: str(df[c].dtype) for c in df.columns}
#
# memory_usage_mb:
#     return float(df.memory_usage(deep=True).sum()) / (1024 ** 2)
#
# per_column_missing:
#     return df.isna().sum().sort_values(ascending=False)
#
# add_trip_duration_minutes:
#     out = df.copy()
#     delta = out["dropoff_datetime"] - out["pickup_datetime"]
#     out["trip_duration_min"] = delta.dt.total_seconds() / 60.0
#     return out
#
# trips_per_borough:
#     return (
#         df.groupby("pickup_borough", observed=True)
#           .size()
#           .sort_values(ascending=False)
#     )
#
# -----------------------------------------------------------------------------
