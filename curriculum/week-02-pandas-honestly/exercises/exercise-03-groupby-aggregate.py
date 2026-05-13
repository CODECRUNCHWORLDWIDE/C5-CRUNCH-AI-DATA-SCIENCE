"""
Exercise 3 -- Group-by, aggregate, pivot.

Goal: drive Lecture 2 with your fingers. You will group a small sales
table by one key and by two keys, run multiple named aggregations,
compute a per-group transformed column, and pivot from long to wide.

Estimated time: 40 minutes.

Run with:    python exercise-03-groupby-aggregate.py
Or test:     pytest exercise-03-groupby-aggregate.py

Acceptance criteria:
- Every TODO is filled in with a vectorized expression (no `.apply`).
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The seven pytest functions all pass.

Do not look at the HINT block at the bottom until you have tried for
fifteen minutes per problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Deterministic sample data: 200 sales rows across 3 regions and 4 months.
# -----------------------------------------------------------------------------

def make_sales(n_rows: int = 200, seed: int = 7) -> pd.DataFrame:
    """Return a small synthetic sales DataFrame.

    Columns:
        order_id    int
        region      category in {Northeast, Southeast, West}
        salesrep    category, 6 unique values
        month       Period[M] in {2025-08, 2025-09, 2025-10, 2025-11}
        units       int
        unit_price  float
        revenue     float (= units * unit_price)
    """
    rng = np.random.default_rng(seed)
    regions = ["Northeast", "Southeast", "West"]
    reps_by_region = {
        "Northeast": ["Ada", "Bo"],
        "Southeast": ["Cy", "Dee"],
        "West":      ["Edu", "Fae"],
    }
    months = pd.period_range("2025-08", periods=4, freq="M")

    region = rng.choice(regions, size=n_rows, p=[0.4, 0.35, 0.25])
    salesrep = np.array(
        [rng.choice(reps_by_region[r]) for r in region],
        dtype=object,
    )
    month = rng.choice(months, size=n_rows)
    units = rng.integers(1, 20, size=n_rows)
    unit_price = np.round(20 + rng.gamma(2.0, 8.0, size=n_rows), 2)

    df = pd.DataFrame(
        {
            "order_id":   np.arange(1, n_rows + 1),
            "region":     pd.Categorical(region,   categories=regions),
            "salesrep":   pd.Categorical(salesrep),
            "month":      pd.PeriodIndex(month, freq="M"),
            "units":      units.astype("int32"),
            "unit_price": unit_price,
        }
    )
    df["revenue"] = (df["units"] * df["unit_price"]).round(2)
    return df


# -----------------------------------------------------------------------------
# Part A -- One-key group-by
# -----------------------------------------------------------------------------

def revenue_per_region(df: pd.DataFrame) -> pd.Series:
    """Return a Series of `region -> total_revenue`, sorted descending.

    Use groupby(...).sum() on the `revenue` column, then sort_values.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- revenue_per_region")


def named_summary_per_region(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by `region` with three columns:

        revenue       sum of revenue
        n_orders      count of order_id
        n_reps        number of distinct salesreps in that region

    Use the named-aggregation form of .agg.
    """
    # TODO: implement using df.groupby("region", observed=True).agg(
    #           revenue=("revenue", "sum"),
    #           n_orders=("order_id", "count"),
    #           n_reps=("salesrep", "nunique"),
    #       )
    raise NotImplementedError("Part A -- named_summary_per_region")


# -----------------------------------------------------------------------------
# Part B -- Two-key group-by
# -----------------------------------------------------------------------------

def revenue_per_region_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-format DataFrame with columns (region, month, revenue),
    one row per (region, month) pair, sorted by (region, month).

    Use groupby(["region", "month"], observed=True, as_index=False).agg.
    """
    # TODO: implement
    raise NotImplementedError("Part B -- revenue_per_region_month")


def revenue_share_within_region(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with a new column `share_of_region` equal to
    each row's revenue divided by the *total* revenue of its region.

    Use groupby(...).transform("sum"). NO .apply.
    """
    # TODO: implement on df.copy()
    raise NotImplementedError("Part B -- revenue_share_within_region")


# -----------------------------------------------------------------------------
# Part C -- Long to wide and back
# -----------------------------------------------------------------------------

def pivot_region_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide DataFrame with `region` as the index and one column
    per `month`, holding the sum of `revenue`.

    Use pivot_table(index="region", columns="month", values="revenue",
                    aggfunc="sum").
    """
    # TODO: implement
    raise NotImplementedError("Part C -- pivot_region_by_month")


def melt_back_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Inverse of pivot_region_by_month: take a wide DataFrame indexed by
    region with months as columns and return long-format (region, month,
    revenue) rows.

    Use reset_index then melt; pass var_name='month', value_name='revenue'.
    """
    # TODO: implement
    raise NotImplementedError("Part C -- melt_back_to_long")


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_revenue_per_region_sorted() -> None:
    df = make_sales()
    rev = revenue_per_region(df)
    assert isinstance(rev, pd.Series)
    assert set(rev.index) == {"Northeast", "Southeast", "West"}
    # Sorted descending: each value >= the next.
    vals = rev.tolist()
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_named_summary_per_region_columns() -> None:
    df = make_sales()
    out = named_summary_per_region(df)
    assert set(out.columns) == {"revenue", "n_orders", "n_reps"}
    # Each region has exactly 2 salesreps by construction.
    assert (out["n_reps"] == 2).all()
    # Total order count adds up to the input length.
    assert int(out["n_orders"].sum()) == len(df)


def test_revenue_per_region_month_shape() -> None:
    df = make_sales()
    out = revenue_per_region_month(df)
    assert set(out.columns) == {"region", "month", "revenue"}
    # 3 regions * 4 months = up to 12 rows; could be less if any pair is empty.
    assert 1 <= len(out) <= 12
    # The total revenue matches the input total.
    assert np.isclose(out["revenue"].sum(), df["revenue"].sum())


def test_revenue_share_within_region_sums_to_one() -> None:
    df = make_sales()
    out = revenue_share_within_region(df)
    # For each region, the share column sums to 1.0 (approx).
    sums = out.groupby("region", observed=True)["share_of_region"].sum()
    assert np.allclose(sums.values, 1.0, atol=1e-9), \
        f"per-region shares should sum to 1; got {sums.tolist()}"


def test_pivot_region_by_month_shape() -> None:
    df = make_sales()
    wide = pivot_region_by_month(df)
    assert wide.index.name == "region"
    # 3 regions, up to 4 months.
    assert len(wide) == 3
    assert 1 <= len(wide.columns) <= 4


def test_melt_round_trips() -> None:
    df = make_sales()
    wide = pivot_region_by_month(df)
    long = melt_back_to_long(wide)
    assert set(long.columns) >= {"region", "month", "revenue"}
    # Up to 12 rows; we may drop the empty pairs by .dropna() but do not
    # require that. Just check the total revenue matches up to NaNs.
    total_long = long["revenue"].dropna().sum()
    total_wide = wide.fillna(0).to_numpy().sum()
    assert np.isclose(total_long, total_wide)


def test_no_apply_used_in_user_functions() -> None:
    """Self-check: this test passes by construction if you followed the
    spec. It exists as a reminder, not a real introspection.
    """
    assert True


def _run_all_tests() -> None:
    test_revenue_per_region_sorted()
    test_named_summary_per_region_columns()
    test_revenue_per_region_month_shape()
    test_revenue_share_within_region_sums_to_one()
    test_pivot_region_by_month_shape()
    test_melt_round_trips()
    test_no_apply_used_in_user_functions()
    print("OK -- exercise 3")


if __name__ == "__main__":
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min on a given problem)
# -----------------------------------------------------------------------------
#
# revenue_per_region:
#     return (
#         df.groupby("region", observed=True)["revenue"]
#           .sum()
#           .sort_values(ascending=False)
#     )
#
# named_summary_per_region:
#     return df.groupby("region", observed=True).agg(
#         revenue=("revenue", "sum"),
#         n_orders=("order_id", "count"),
#         n_reps=("salesrep", "nunique"),
#     )
#
# revenue_per_region_month:
#     return (
#         df.groupby(["region", "month"], observed=True, as_index=False)
#           .agg(revenue=("revenue", "sum"))
#           .sort_values(["region", "month"])
#           .reset_index(drop=True)
#     )
#
# revenue_share_within_region:
#     out = df.copy()
#     region_total = out.groupby("region", observed=True)["revenue"].transform("sum")
#     out["share_of_region"] = out["revenue"] / region_total
#     return out
#
# pivot_region_by_month:
#     return df.pivot_table(
#         index="region",
#         columns="month",
#         values="revenue",
#         aggfunc="sum",
#         observed=True,
#     )
#
# melt_back_to_long:
#     return (
#         wide.reset_index()
#             .melt(id_vars="region", var_name="month", value_name="revenue")
#     )
#
# -----------------------------------------------------------------------------
