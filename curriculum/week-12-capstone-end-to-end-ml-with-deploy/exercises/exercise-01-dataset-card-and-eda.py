"""
Exercise 1 -- Dataset card and exploratory data analysis on UCI Adult Income.

Goal: load the UCI Adult Income dataset, write a Gebru-template dataset card,
run the eight-step EDA from Lecture 1, and flag the missing-not-at-random
columns. The exercise is the "monday morning" of the capstone, run on a known
dataset so that the workflow is unambiguous.

By the end of this exercise you will have:

    (1) Loaded UCI Adult Income from the canonical URL (no network in CI;
        the test fixture ships a 200-row sample).
    (2) Run df.info(), df.shape, df.dtypes, and df.isna().sum() and printed
        the results.
    (3) Detected that missing values in this dataset are encoded as the
        literal string "?", not as NaN, and converted them.
    (4) Computed the per-column missingness fraction and flagged any column
        with > 1 percent missing.
    (5) Run df.describe(include="all") and printed the result.
    (6) Computed the class balance for the `income` target column.
    (7) Identified the "fnlwgt" column as a survey weight (not a feature) and
        flagged it.
    (8) Written a one-paragraph EDA summary as a docstring on a `summary()`
        function.
    (9) Filled in the seven-section dataset card as a multi-line string in
        a `dataset_card()` function and asserted that every section has
        non-trivial content.

Estimated time: 60-90 minutes.

Run with:    python exercise-01-dataset-card-and-eda.py
Or test:     pytest exercise-01-dataset-card-and-eda.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 1" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-01-dataset-card-and-eda.py succeeds.

References:
    Gebru et al. 2021, "Datasheets for Datasets": https://arxiv.org/abs/1803.09010
    UCI Adult Income dataset page: https://archive.ics.uci.edu/dataset/2/adult
    pandas docs: https://pandas.pydata.org/docs/
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# pandas is imported lazily inside functions so this file compiles cleanly
# without it installed. The pytest functions do require pandas.


RANDOM_STATE: int = 42

# The UCI Adult Income dataset has these column names, in this order.
ADULT_COLUMNS: List[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


# -----------------------------------------------------------------------------
# Part A -- load the dataset
# -----------------------------------------------------------------------------


def make_sample_dataframe() -> "pd.DataFrame":  # type: ignore[name-defined]
    """Return a 200-row synthetic Adult-Income-shaped DataFrame for tests.

    The real UCI dataset has 32,561 train rows; using a 200-row synthetic
    sample keeps tests fast and offline. The columns and dtypes match the
    real dataset, including the "?" encoding of missing values for
    workclass, occupation, native_country.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(RANDOM_STATE)
    n: int = 200

    age = rng.integers(17, 80, size=n)
    workclass = rng.choice(
        ["Private", "Self-emp-not-inc", "State-gov", "?"],
        size=n,
        p=[0.7, 0.15, 0.1, 0.05],
    )
    fnlwgt = rng.integers(10000, 1000000, size=n)
    education = rng.choice(
        ["Bachelors", "HS-grad", "Some-college", "Masters", "Doctorate"],
        size=n,
    )
    education_num = rng.integers(1, 17, size=n)
    marital_status = rng.choice(
        ["Married-civ-spouse", "Never-married", "Divorced"],
        size=n,
    )
    occupation = rng.choice(
        ["Exec-managerial", "Adm-clerical", "Sales", "Craft-repair", "?"],
        size=n,
        p=[0.2, 0.2, 0.2, 0.3, 0.1],
    )
    relationship = rng.choice(["Husband", "Wife", "Not-in-family"], size=n)
    race = rng.choice(["White", "Black", "Asian-Pac-Islander"], size=n, p=[0.85, 0.1, 0.05])
    sex = rng.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    capital_gain = rng.choice([0, 0, 0, 0, 5000, 15000], size=n)
    capital_loss = rng.choice([0, 0, 0, 0, 1500], size=n)
    hours_per_week = rng.integers(20, 60, size=n)
    native_country = rng.choice(["United-States", "Mexico", "?"], size=n, p=[0.9, 0.05, 0.05])

    # income is correlated with hours_per_week and education_num.
    income_logit = (hours_per_week - 40) * 0.1 + (education_num - 10) * 0.3
    income_prob = 1.0 / (1.0 + np.exp(-income_logit))
    income = np.where(rng.random(n) < income_prob, ">50K", "<=50K")

    df = pd.DataFrame({
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
        "income": income,
    })
    return df


def replace_question_marks_with_nan(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    """Replace the literal string "?" with NaN in every object column.

    The UCI Adult Income dataset encodes missing values as the string "?".
    pandas does not detect this by default; the column dtypes come out as
    `object` and `df.isna().sum()` reports zero missingness on those columns.
    The first cleaning step is to convert "?" to NaN.

    Args:
        df: the raw DataFrame.

    Returns:
        A new DataFrame with "?" replaced by NaN in every object column.

    Steps:
        1. Identify the object-dtype columns.
        2. For each object column, replace "?" with `float('nan')` (or use
           pandas's `df.replace({"?": pd.NA})`).
        3. Return the modified DataFrame.
    """
    import numpy as np

    out = df.copy()
    # TODO: replace "?" with NaN in every object column. Two ways:
    #   (a) out = out.replace("?", np.nan)
    #   (b) for col in out.select_dtypes(include="object").columns:
    #           out[col] = out[col].replace("?", np.nan)
    # Either works. (a) is more concise.
    out = out.replace("?", np.nan)
    return out


# -----------------------------------------------------------------------------
# Part B -- the eight-step EDA
# -----------------------------------------------------------------------------


def eda_shape(df: "pd.DataFrame") -> Tuple[int, int]:  # type: ignore[name-defined]
    """Return (n_rows, n_cols)."""
    # TODO: return df.shape.
    return df.shape  # type: ignore[no-any-return]


def eda_dtypes(df: "pd.DataFrame") -> Dict[str, str]:  # type: ignore[name-defined]
    """Return a dict mapping column name to dtype name."""
    # TODO: return {col: str(df[col].dtype) for col in df.columns}.
    return {col: str(df[col].dtype) for col in df.columns}


def eda_missingness(df: "pd.DataFrame") -> Dict[str, float]:  # type: ignore[name-defined]
    """Return a dict mapping column name to missingness fraction (0.0 to 1.0)."""
    # TODO: compute df.isna().sum() / len(df) and convert to a dict.
    n = len(df)
    if n == 0:
        return {col: 0.0 for col in df.columns}
    return {col: float(df[col].isna().sum()) / n for col in df.columns}


def eda_class_balance(df: "pd.DataFrame", target_col: str = "income") -> Dict[str, float]:  # type: ignore[name-defined]
    """Return the class-balance dict for the target column.

    For a binary target like Adult Income's `income` column, this returns
    something like {"<=50K": 0.76, ">50K": 0.24}.
    """
    # TODO: compute df[target_col].value_counts(normalize=True).to_dict().
    return {str(k): float(v) for k, v in df[target_col].value_counts(normalize=True).items()}


def eda_describe_numeric(df: "pd.DataFrame") -> "pd.DataFrame":  # type: ignore[name-defined]
    """Return df.describe() restricted to numeric columns."""
    # TODO: return df.describe(include="number").
    return df.describe(include="number")


def eda_flag_columns_with_missing(
    df: "pd.DataFrame",  # type: ignore[name-defined]
    threshold: float = 0.01,
) -> List[str]:
    """Return the list of columns whose missingness exceeds the threshold.

    The default threshold is 1 percent (0.01). The function returns a sorted
    list of column names.
    """
    miss = eda_missingness(df)
    return sorted([col for col, frac in miss.items() if frac > threshold])


def eda_flag_survey_weights(df: "pd.DataFrame") -> List[str]:  # type: ignore[name-defined]
    """Return the list of columns that look like survey weights.

    Heuristic: numeric columns named `fnlwgt`, `weight`, `sample_weight`,
    `survey_weight` are typical survey-weight column names. The Adult Income
    dataset has `fnlwgt`. Survey weights are *not* features; they should
    never be passed to the model.
    """
    candidates = {"fnlwgt", "weight", "sample_weight", "survey_weight"}
    return sorted([col for col in df.columns if col.lower() in candidates])


# -----------------------------------------------------------------------------
# Part C -- the dataset card
# -----------------------------------------------------------------------------


def dataset_card() -> Dict[str, str]:
    """Return the seven-section Gebru dataset card for UCI Adult Income.

    Each value is a one-to-three-sentence string. The grader checks that
    every section has more than 20 characters of content (i.e., is not the
    empty string and is not "TODO").

    References:
        Gebru et al. 2021, section 3 (the seven-section questionnaire).
        UCI Adult Income page: https://archive.ics.uci.edu/dataset/2/adult
    """
    return {
        "motivation": (
            "Predict whether an adult's annual income exceeds 50,000 US dollars "
            "from demographic and employment census features. Extracted by Barry "
            "Becker from the 1994 US Census Bureau database; donated to UCI in 1996."
        ),
        "composition": (
            "48,842 instances (32,561 train; 16,281 test). Each instance is one "
            "anonymized adult worker. 14 features (6 continuous, 8 categorical) "
            "and one binary target. Approximately 24 percent positive (>50K). "
            "Categorical columns encode missing values as the literal string '?'."
        ),
        "collection_process": (
            "Drawn from the 1994 US Census Current Population Survey. Sample "
            "restricted to adults aged 16+, with income > 100 USD, weeks-worked > 0, "
            "hours-per-week > 0. No additional consent process for ML benchmarking."
        ),
        "preprocessing": (
            "Continuous features preserved. Categorical features kept as strings. "
            "The fnlwgt column is a survey weight and must not be used as a feature. "
            "education and education_num are redundant (the integer is an encoding of "
            "the string); pick one."
        ),
        "uses": (
            "Suitable for: benchmarking tabular classification, teaching the ML "
            "workflow, demonstrating fairness audits (race and sex are protected). "
            "Not suitable for: any real income-classification application; the data "
            "is over 30 years old and US-specific."
        ),
        "distribution": (
            "Available at https://archive.ics.uci.edu/dataset/2/adult under CC BY 4.0. "
            "DOI: 10.24432/C5XW20."
        ),
        "maintenance": (
            "Donated in 1996; static since. Maintained by UCI as part of the archival "
            "ML repository. Contact: ml-repository@ics.uci.edu."
        ),
    }


# -----------------------------------------------------------------------------
# Part D -- the EDA summary
# -----------------------------------------------------------------------------


def summary() -> str:
    """The one-paragraph EDA summary, as a single string.

    The capstone rubric requires a one-paragraph "here is what I learned"
    summary at the top of the EDA notebook. This function returns that
    paragraph; the test checks it has more than 200 characters and mentions
    at least three concrete findings.
    """
    return (
        "The Adult Income dataset has 200 synthetic rows (in this exercise; the real "
        "UCI source has 32,561 train and 16,281 test). The income target is imbalanced "
        "with roughly a 76/24 negative/positive split. Three categorical columns -- "
        "workclass, occupation, and native_country -- encode missing values as the "
        "literal string '?' rather than as NaN; the first cleaning step is to convert "
        "these. The fnlwgt column is a survey weight and must be excluded from features "
        "entirely; the education and education_num columns are redundant and only one "
        "should be retained. Sex and race are protected attributes and are the natural "
        "axes for the slice-based fairness audit at evaluation time."
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_make_sample_dataframe_shape() -> None:
    df = make_sample_dataframe()
    assert df.shape == (200, 15)
    assert list(df.columns) == ADULT_COLUMNS


def test_replace_question_marks_with_nan() -> None:
    df = make_sample_dataframe()
    cleaned = replace_question_marks_with_nan(df)
    # After cleaning, no cell should equal the literal string "?".
    for col in cleaned.select_dtypes(include="object").columns:
        assert (cleaned[col] == "?").sum() == 0


def test_eda_missingness_post_cleaning() -> None:
    df = make_sample_dataframe()
    cleaned = replace_question_marks_with_nan(df)
    miss = eda_missingness(cleaned)
    # workclass, occupation, native_country had "?" tokens; they should now
    # show > 0 missingness.
    assert miss["workclass"] > 0.0
    assert miss["occupation"] > 0.0
    assert miss["native_country"] > 0.0
    # age (a clean integer column) should have zero missingness.
    assert miss["age"] == 0.0


def test_eda_class_balance() -> None:
    df = make_sample_dataframe()
    balance = eda_class_balance(df)
    # Both classes present.
    assert "<=50K" in balance
    assert ">50K" in balance
    # The two values should sum to 1.0 (up to floating-point noise).
    assert abs(sum(balance.values()) - 1.0) < 1e-9


def test_eda_flag_survey_weights() -> None:
    df = make_sample_dataframe()
    flagged = eda_flag_survey_weights(df)
    assert "fnlwgt" in flagged


def test_eda_flag_columns_with_missing() -> None:
    df = make_sample_dataframe()
    cleaned = replace_question_marks_with_nan(df)
    flagged = eda_flag_columns_with_missing(cleaned, threshold=0.01)
    assert "workclass" in flagged or "occupation" in flagged or "native_country" in flagged


def test_dataset_card_completeness() -> None:
    card = dataset_card()
    expected = {"motivation", "composition", "collection_process", "preprocessing",
                "uses", "distribution", "maintenance"}
    assert set(card.keys()) == expected
    for section, content in card.items():
        assert len(content) > 20, f"section {section!r} is too short"
        assert "TODO" not in content, f"section {section!r} still has a TODO"


def test_summary_completeness() -> None:
    s = summary()
    assert len(s) > 200
    # Mentions specific concrete findings:
    assert "imbalanced" in s.lower() or "76/24" in s or "24" in s
    assert "fnlwgt" in s.lower()
    assert "missing" in s.lower() or "'?'" in s or '"?"' in s


def test_eda_shape_and_dtypes() -> None:
    df = make_sample_dataframe()
    n_rows, n_cols = eda_shape(df)
    assert n_rows == 200
    assert n_cols == 15
    dtypes = eda_dtypes(df)
    assert "age" in dtypes
    assert "income" in dtypes


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    df = make_sample_dataframe()
    print(f"shape: {df.shape}")
    print()
    cleaned = replace_question_marks_with_nan(df)
    miss = eda_missingness(cleaned)
    print("Missingness after replacing '?' with NaN:")
    for col, frac in sorted(miss.items(), key=lambda kv: -kv[1]):
        if frac > 0.0:
            print(f"  {col:20s}  {frac:.3f}")
    print()
    bal = eda_class_balance(cleaned)
    print("Class balance:")
    for label, frac in bal.items():
        print(f"  {label:8s}  {frac:.3f}")
    print()
    flagged = eda_flag_columns_with_missing(cleaned, threshold=0.01)
    print(f"Columns with > 1% missingness: {flagged}")
    print()
    weights = eda_flag_survey_weights(cleaned)
    print(f"Survey-weight columns to exclude from features: {weights}")
    print()
    print("Dataset card:")
    for section, content in dataset_card().items():
        print(f"  ## {section}")
        print(f"    {content}")
        print()
    print("Summary:")
    print(f"  {summary()}")
    print()

    # Run the assertions inline so that `python exercise-01-...py` flags
    # any TODO that was left unfilled.
    test_make_sample_dataframe_shape()
    test_replace_question_marks_with_nan()
    test_eda_missingness_post_cleaning()
    test_eda_class_balance()
    test_eda_flag_survey_weights()
    test_eda_flag_columns_with_missing()
    test_dataset_card_completeness()
    test_summary_completeness()
    test_eda_shape_and_dtypes()

    print("OK -- exercise 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
