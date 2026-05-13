"""
Exercise 2 -- Clean the messy.

Goal: take a deliberately ugly synthetic CSV with the bugs you will see
in real public data -- a numeric column read as `object`, four ways of
encoding "missing", two date formats in one column, leading whitespace
on strings, a negative sentinel for "unknown age" -- and repair every
one. Output is a tidy DataFrame with honest dtypes.

Estimated time: 50 minutes.

Run with:    python exercise-02-clean-the-messy.py
Or test:     pytest exercise-02-clean-the-messy.py

Acceptance criteria:
- Every TODO is filled in. No `.apply` calls anywhere.
- The script prints "OK -- exercise 2" at the end with no AssertionError.
- The six pytest functions all pass.

Do not look at the HINT block at the bottom until you have tried for
fifteen minutes per problem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent
CSV_PATH = HERE / "_messy_sample.csv"


# -----------------------------------------------------------------------------
# Test-data generator -- deliberately ugly
# -----------------------------------------------------------------------------

_RAW_CSV = """user_id,name,signup_date,age,salary,country,active
1,Ada Lovelace,2024-01-15,34,"85,000",United Kingdom,yes
2,  Bo Liu, 02/14/2024 ,27,62000,United States ,YES
3,Cy Reeves,2024-03-01,N/A,"110,000",Canada,no
4,Dee Singh,2024-04-22,22,unknown,India,Yes
5,Edu Garcia,05/01/2024,-1,72000,Spain,No
6,Fae Tanaka,2024-06-30,29,"NA",Japan,yes
7, Gus Mehta,07/12/2024,41,98000,India,YES
8,Hana Watanabe,2024-08-08,,55000,Japan,no
9,Ivo Pavlov,09/15/2024,38,"125,500",Russia,yes
10,Joi Park,2024-10-30,N/A,67000,South Korea,YES
"""


def _ensure_csv() -> Path:
    if not CSV_PATH.exists():
        CSV_PATH.write_text(_RAW_CSV)
    return CSV_PATH


# -----------------------------------------------------------------------------
# Part A -- Read it raw, then clean each column.
# -----------------------------------------------------------------------------

# Encoded-missing tokens you will see in the wild. Add to this list when
# you find a new one in a real dataset.
MISSING_TOKENS = ["", "N/A", "NA", "unknown", "Unknown", "null", "NULL", "-1"]


def read_raw(path: Path) -> pd.DataFrame:
    """Read the CSV with the missing tokens recognized as NaN.

    Use the `na_values=` argument of `pd.read_csv`. Leave everything else
    as default (so dtypes will mostly be `object`; we will fix that next).
    """
    # TODO: implement
    raise NotImplementedError("Part A -- read_raw")


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every string column.

    Use the `.str.strip()` accessor; do NOT loop over columns with .apply.
    Operate on a copy. Return the new DataFrame.

    Hint: `df.select_dtypes(include="object").columns` gives you the
    columns that are still object-typed.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- clean_strings")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `signup_date` to a real datetime dtype.

    The column contains TWO formats:
      'YYYY-MM-DD'  (ISO 8601)
      'MM/DD/YYYY'  (US slash format)

    Use `pd.to_datetime(..., format='mixed', errors='coerce')`. Any value
    that still cannot be parsed becomes NaT.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- parse_dates")


def parse_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `salary` to float.

    The column contains strings like "85,000" with thousand-separators.
    Strip the commas with `.str.replace(",", "")`, then `pd.to_numeric(...,
    errors='coerce')`. Negative or zero salaries are invalid -> NaN.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- parse_salary")


def parse_age(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `age` to a nullable integer dtype (`Int64` with capital I).

    Any value < 0 or > 120 is invalid -> pd.NA. Anything that is already
    NaN stays NaN.

    Note: use `Int64`, not `int64`, so missing values are preserved.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- parse_age")


def parse_active(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `active` to a nullable boolean dtype (`boolean`).

    The raw values are an inconsistent mix of yes / YES / Yes / no / No.
    Map them to True / False / NA without using .apply.

    Hint: use `.str.lower().map({"yes": True, "no": False})`.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- parse_active")


def to_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline: read_raw -> clean_strings -> parse_dates -> parse_salary
    -> parse_age -> parse_active. Cast `country` to category.

    Use `.pipe()` so the code reads top-to-bottom.
    """
    # TODO: implement
    raise NotImplementedError("Part A -- to_clean")


# -----------------------------------------------------------------------------
# Pytest-style checks
# -----------------------------------------------------------------------------

def test_read_raw_loads_ten_rows() -> None:
    df = read_raw(_ensure_csv())
    assert len(df) == 10
    assert set(df.columns) == {
        "user_id", "name", "signup_date", "age",
        "salary", "country", "active",
    }


def test_clean_strings_strips_whitespace() -> None:
    df = read_raw(_ensure_csv())
    df = clean_strings(df)
    # The "  Bo Liu" row had leading whitespace; that should be gone.
    assert "  Bo Liu" not in df["name"].dropna().tolist()
    assert "Bo Liu" in df["name"].dropna().tolist()


def test_parse_dates_datetime_dtype() -> None:
    df = read_raw(_ensure_csv())
    df = clean_strings(df)
    df = parse_dates(df)
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    # The first row's date is 2024-01-15.
    assert df.loc[0, "signup_date"] == pd.Timestamp("2024-01-15")
    # The second row used '02/14/2024' format.
    assert df.loc[1, "signup_date"] == pd.Timestamp("2024-02-14")


def test_parse_salary_floats_with_nans() -> None:
    df = read_raw(_ensure_csv())
    df = clean_strings(df)
    df = parse_salary(df)
    assert pd.api.types.is_float_dtype(df["salary"])
    # "85,000" -> 85000.0
    assert df.loc[0, "salary"] == 85000.0
    # Row index 3 had 'unknown' (already NaN after read_raw, still NaN).
    assert pd.isna(df.loc[3, "salary"])


def test_parse_age_nullable_int() -> None:
    df = read_raw(_ensure_csv())
    df = clean_strings(df)
    df = parse_age(df)
    # Nullable Int64, not float
    assert str(df["age"].dtype) == "Int64"
    # The 'N/A' rows are NA; the -1 row is also NA after cleaning.
    n_missing = df["age"].isna().sum()
    assert n_missing >= 3, f"expected >=3 missing ages, got {n_missing}"
    # Valid ages are in [18, 120]
    valid = df["age"].dropna()
    assert ((valid >= 0) & (valid <= 120)).all()


def test_parse_active_nullable_bool() -> None:
    df = read_raw(_ensure_csv())
    df = clean_strings(df)
    df = parse_active(df)
    # Nullable boolean dtype
    assert str(df["active"].dtype) == "boolean"
    # All originally-yes rows are True.
    assert df.loc[0, "active"] == True   # noqa: E712 -- intentional


def test_to_clean_pipeline_runs() -> None:
    df = to_clean(_ensure_csv())
    # Schema sanity: every column has a non-object dtype EXCEPT 'name'
    # which we leave alone in this exercise.
    assert pd.api.types.is_datetime64_any_dtype(df["signup_date"])
    assert pd.api.types.is_float_dtype(df["salary"])
    assert str(df["age"].dtype) == "Int64"
    assert str(df["active"].dtype) == "boolean"
    assert isinstance(df["country"].dtype, pd.CategoricalDtype)


def _run_all_tests() -> None:
    test_read_raw_loads_ten_rows()
    test_clean_strings_strips_whitespace()
    test_parse_dates_datetime_dtype()
    test_parse_salary_floats_with_nans()
    test_parse_age_nullable_int()
    test_parse_active_nullable_bool()
    test_to_clean_pipeline_runs()
    print("OK -- exercise 2")


if __name__ == "__main__":
    _ensure_csv()
    _run_all_tests()


# -----------------------------------------------------------------------------
# HINT (read only if stuck for >15 min on a given problem)
# -----------------------------------------------------------------------------
#
# read_raw:
#     return pd.read_csv(path, na_values=MISSING_TOKENS, keep_default_na=True)
#
# clean_strings:
#     out = df.copy()
#     obj_cols = out.select_dtypes(include="object").columns
#     for c in obj_cols:
#         out[c] = out[c].str.strip()
#     return out
#     # (The `for c in obj_cols` loop iterates over a handful of columns, not
#     #  millions of rows. That is fine and is NOT what Lecture 3 warns against.)
#
# parse_dates:
#     out = df.copy()
#     out["signup_date"] = pd.to_datetime(
#         out["signup_date"], format="mixed", errors="coerce",
#     )
#     return out
#
# parse_salary:
#     out = df.copy()
#     # The column is `object` after read_raw -- str.replace handles NaN.
#     s = out["salary"].str.replace(",", "", regex=False)
#     out["salary"] = pd.to_numeric(s, errors="coerce")     # -> float64
#     out.loc[out["salary"] <= 0, "salary"] = np.nan
#     return out
#
# parse_age:
#     out = df.copy()
#     a = pd.to_numeric(out["age"], errors="coerce")
#     a = a.where((a >= 0) & (a <= 120))     # invalid -> NaN
#     out["age"] = a.astype("Int64")
#     return out
#
# parse_active:
#     out = df.copy()
#     m = out["active"].astype("string").str.lower().map(
#         {"yes": True, "no": False}
#     )
#     out["active"] = m.astype("boolean")
#     return out
#
# to_clean:
#     return (
#         read_raw(path)
#         .pipe(clean_strings)
#         .pipe(parse_dates)
#         .pipe(parse_salary)
#         .pipe(parse_age)
#         .pipe(parse_active)
#         .assign(country=lambda d: d["country"].astype("category"))
#     )
#
# -----------------------------------------------------------------------------
