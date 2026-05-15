"""
Exercise 3 -- Package the model and wrap it in a Gradio interface.

Goal: save a trained LogisticRegression to disk, write a `predict_one(row)`
function that loads the model and predicts a single row, wrap `predict_one`
in a `gr.Interface`, and verify the interface launches locally without
errors. The Gradio interface is what gets deployed to Hugging Face Spaces.

By the end of this exercise you will have:

    (1) Trained a LogisticRegression on the Exercise-2 synthetic data and
        saved it to disk via `joblib.dump`.
    (2) Computed and printed the SHA256 hash of the saved model file.
    (3) Loaded the model in a separate function `load_model()` that uses a
        module-level cache so the file is only read once per process.
    (4) Implemented `validate_row(row)` that raises ValueError for bad inputs.
    (5) Implemented `predict_one(row)` that returns
        {"label": "...", "probability": float, "threshold": float}.
    (6) Built a `gr.Interface(fn=gradio_predict, inputs=..., outputs=...)`
        that wraps predict_one with Gradio components.
    (7) Verified that calling `gradio_predict(...)` with sample inputs
        returns a non-empty string (Gradio launching is checked manually).

Estimated time: 60-90 minutes.

Run with:    python exercise-03-package-and-gradio.py
Or test:     pytest exercise-03-package-and-gradio.py

Acceptance criteria:
- Every TODO is filled in.
- The script prints "OK -- exercise 3" at the end with no AssertionError.
- The pytest functions all pass.
- python -m py_compile exercise-03-package-and-gradio.py succeeds.

References:
    Gradio quickstart: https://www.gradio.app/guides/quickstart
    Gradio Interface: https://www.gradio.app/docs/interface
    joblib docs: https://joblib.readthedocs.io/en/latest/
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

RANDOM_STATE: int = 42
MODEL_PATH: Path = Path("/tmp/c5_capstone_exercise_3_model.joblib")
THRESHOLD: float = 0.5

# Module-level cache for the loaded model.
_model_cache: Any = None
_feature_columns_cache: List[str] = []


# -----------------------------------------------------------------------------
# Part A -- train and save the model
# -----------------------------------------------------------------------------


def make_training_data(n: int = 2000) -> Tuple["pd.DataFrame", "pd.Series"]:  # type: ignore[name-defined]
    """Generate the same synthetic Adult-Income-shaped DataFrame as Exercise 2."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(17, 80, size=n)
    education_num = rng.integers(1, 17, size=n)
    sex = rng.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    race = rng.choice(
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        size=n,
        p=[0.85, 0.09, 0.03, 0.02, 0.01],
    )
    hours_per_week = rng.integers(20, 60, size=n).astype(int)
    capital_gain = rng.choice([0, 0, 0, 0, 5000, 15000], size=n)

    logit = (
        (hours_per_week - 40) * 0.1
        + (education_num - 10) * 0.3
        + (capital_gain / 5000) * 0.5
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    income = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "sex": sex,
        "race": race,
        "hours_per_week": hours_per_week,
        "capital_gain": capital_gain,
    })
    return df, pd.Series(income, name="income")


def fit_pipeline(X: "pd.DataFrame", y: "pd.Series"):  # type: ignore[name-defined]
    """Fit a Pipeline(ColumnTransformer, LogisticRegression) on the data.

    Using a Pipeline keeps the preprocessing inside the model artifact, so
    inference time accepts a raw row (dict) and the pipeline handles the
    one-hot encoding internally.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = ["age", "education_num", "hours_per_week", "capital_gain"]
    categorical = ["sex", "race"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
    )
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X, y)
    return pipeline


def save_model(model, path: Path = MODEL_PATH) -> None:  # noqa: ANN001
    """Save a fitted sklearn Pipeline via joblib.dump."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def sha256_of_file(path: Path) -> str:
    """Return the hex-encoded SHA256 hash of the file at `path`."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Part B -- the inference layer
# -----------------------------------------------------------------------------


def load_model(path: Path = MODEL_PATH):  # noqa: ANN201
    """Load and cache the saved model. Subsequent calls are O(1)."""
    global _model_cache
    if _model_cache is None:
        import joblib

        _model_cache = joblib.load(path)
    return _model_cache


def validate_row(row: Dict[str, object]) -> Dict[str, object]:
    """Validate a single inference-time row. Raise ValueError on bad input.

    Required keys:
        age: int, 16-100
        education_num: int, 1-17
        sex: "Male" or "Female"
        race: one of the 5 census categories
        hours_per_week: int, 1-100
        capital_gain: int, >= 0

    Returns the row (unchanged) if valid; raises otherwise.
    """
    required = {"age", "education_num", "sex", "race", "hours_per_week", "capital_gain"}
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")

    age = row["age"]
    if not isinstance(age, (int, float)) or isinstance(age, bool):
        raise ValueError(f"age must be numeric, got {type(age).__name__}")
    if not (16 <= float(age) <= 100):
        raise ValueError(f"age must be in [16, 100], got {age}")

    edu = row["education_num"]
    if not isinstance(edu, (int, float)) or isinstance(edu, bool):
        raise ValueError(f"education_num must be numeric, got {type(edu).__name__}")
    if not (1 <= float(edu) <= 17):
        raise ValueError(f"education_num must be in [1, 17], got {edu}")

    if row["sex"] not in {"Male", "Female"}:
        raise ValueError(f"sex must be 'Male' or 'Female', got {row['sex']!r}")

    valid_races = {"White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"}
    if row["race"] not in valid_races:
        raise ValueError(f"race must be one of {sorted(valid_races)}, got {row['race']!r}")

    hours = row["hours_per_week"]
    if not isinstance(hours, (int, float)) or isinstance(hours, bool):
        raise ValueError(f"hours_per_week must be numeric, got {type(hours).__name__}")
    if not (1 <= float(hours) <= 100):
        raise ValueError(f"hours_per_week must be in [1, 100], got {hours}")

    gain = row["capital_gain"]
    if not isinstance(gain, (int, float)) or isinstance(gain, bool):
        raise ValueError(f"capital_gain must be numeric, got {type(gain).__name__}")
    if float(gain) < 0:
        raise ValueError(f"capital_gain must be >= 0, got {gain}")

    return row


def predict_one(row: Dict[str, object]) -> Dict[str, object]:
    """Predict a single row.

    Returns:
        {"label": "<=50K" or ">50K", "probability": float, "threshold": float}.
    """
    import pandas as pd

    validate_row(row)
    model = load_model()
    df = pd.DataFrame([row])
    prob: float = float(model.predict_proba(df)[0, 1])
    label: str = ">50K" if prob >= THRESHOLD else "<=50K"
    return {"label": label, "probability": prob, "threshold": THRESHOLD}


# -----------------------------------------------------------------------------
# Part C -- the Gradio wrapper
# -----------------------------------------------------------------------------


def gradio_predict(
    age: int,
    education_num: int,
    sex: str,
    race: str,
    hours_per_week: int,
    capital_gain: int,
) -> str:
    """The Gradio-facing function. Returns a single string.

    Gradio's gr.Interface unpacks the input components into positional
    arguments. We rebuild the row dict and call predict_one, catching any
    ValueError and returning a friendly message.
    """
    row: Dict[str, object] = {
        "age": int(age),
        "education_num": int(education_num),
        "sex": str(sex),
        "race": str(race),
        "hours_per_week": int(hours_per_week),
        "capital_gain": int(capital_gain),
    }
    try:
        result = predict_one(row)
    except ValueError as e:
        return f"Invalid input: {e}"
    return (
        f"Predicted label: {result['label']}\n"
        f"Probability:    {float(result['probability']):.3f}\n"
        f"Threshold:      {float(result['threshold']):.3f}"
    )


def build_gradio_interface():  # noqa: ANN201
    """Build (but do not launch) the gr.Interface.

    Importing gradio is deferred to this function so the file compiles
    cleanly without gradio installed.
    """
    import gradio as gr

    races = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    iface = gr.Interface(
        fn=gradio_predict,
        inputs=[
            gr.Number(label="Age", value=39, precision=0),
            gr.Number(label="Education-num (1-17)", value=13, precision=0),
            gr.Radio(["Male", "Female"], label="Sex", value="Male"),
            gr.Dropdown(races, label="Race", value="White"),
            gr.Number(label="Hours per week", value=40, precision=0),
            gr.Number(label="Capital gain", value=0, precision=0),
        ],
        outputs=gr.Text(label="Result"),
        title="C5 Capstone (Exercise 3) -- Adult Income predictor",
        description=(
            "Synthetic Adult-Income-shaped data. Logistic regression. "
            "Exercise 3 of Week 12. Not for any real income decision."
        ),
        flagging_mode="never",
    )
    return iface


# -----------------------------------------------------------------------------
# Helper -- reset the module-level cache for tests
# -----------------------------------------------------------------------------


def _reset_cache() -> None:
    global _model_cache
    _model_cache = None


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_fit_and_save_round_trip() -> None:
    X, y = make_training_data(n=500)
    model = fit_pipeline(X, y)
    save_model(model, MODEL_PATH)
    assert MODEL_PATH.exists()
    # Hash is a 64-character hex string.
    h = sha256_of_file(MODEL_PATH)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_load_model_caches() -> None:
    X, y = make_training_data(n=500)
    model = fit_pipeline(X, y)
    save_model(model, MODEL_PATH)
    _reset_cache()
    m1 = load_model(MODEL_PATH)
    m2 = load_model(MODEL_PATH)
    # Same object id means the cache is doing its job.
    assert m1 is m2


def test_validate_row_accepts_valid() -> None:
    row: Dict[str, object] = {
        "age": 39,
        "education_num": 13,
        "sex": "Male",
        "race": "White",
        "hours_per_week": 40,
        "capital_gain": 0,
    }
    # Should not raise.
    validate_row(row)


def test_validate_row_rejects_bad_age() -> None:
    import pytest

    row: Dict[str, object] = {
        "age": -10, "education_num": 13, "sex": "Male", "race": "White",
        "hours_per_week": 40, "capital_gain": 0,
    }
    with pytest.raises(ValueError):
        validate_row(row)


def test_validate_row_rejects_missing_key() -> None:
    import pytest

    row: Dict[str, object] = {"age": 39}  # almost everything missing.
    with pytest.raises(ValueError):
        validate_row(row)


def test_validate_row_rejects_bad_sex() -> None:
    import pytest

    row: Dict[str, object] = {
        "age": 39, "education_num": 13, "sex": "Other", "race": "White",
        "hours_per_week": 40, "capital_gain": 0,
    }
    with pytest.raises(ValueError):
        validate_row(row)


def test_predict_one_round_trip() -> None:
    X, y = make_training_data(n=500)
    model = fit_pipeline(X, y)
    save_model(model, MODEL_PATH)
    _reset_cache()

    row: Dict[str, object] = {
        "age": 39, "education_num": 13, "sex": "Male", "race": "White",
        "hours_per_week": 60, "capital_gain": 5000,
    }
    result = predict_one(row)
    assert "label" in result
    assert "probability" in result
    assert "threshold" in result
    assert result["label"] in {"<=50K", ">50K"}
    assert 0.0 <= float(result["probability"]) <= 1.0


def test_gradio_predict_returns_string() -> None:
    X, y = make_training_data(n=500)
    model = fit_pipeline(X, y)
    save_model(model, MODEL_PATH)
    _reset_cache()

    s = gradio_predict(39, 13, "Male", "White", 40, 0)
    assert isinstance(s, str)
    assert "Predicted label:" in s


def test_gradio_predict_handles_bad_input() -> None:
    X, y = make_training_data(n=500)
    model = fit_pipeline(X, y)
    save_model(model, MODEL_PATH)
    _reset_cache()

    s = gradio_predict(-5, 13, "Male", "White", 40, 0)
    assert "Invalid input" in s


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    X, y = make_training_data(n=2000)
    print(f"Training on {len(X)} rows, {X.shape[1]} columns.")
    model = fit_pipeline(X, y)
    save_model(model, MODEL_PATH)
    h = sha256_of_file(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}.")
    print(f"Model SHA256: {h}")
    print()

    # Sample predictions.
    _reset_cache()
    samples: List[Dict[str, object]] = [
        {"age": 39, "education_num": 13, "sex": "Male", "race": "White",
         "hours_per_week": 60, "capital_gain": 5000},
        {"age": 22, "education_num": 7, "sex": "Female", "race": "Black",
         "hours_per_week": 25, "capital_gain": 0},
        {"age": 58, "education_num": 16, "sex": "Female", "race": "Asian-Pac-Islander",
         "hours_per_week": 45, "capital_gain": 15000},
    ]
    print("Sample predictions:")
    for s in samples:
        result = predict_one(s)
        print(f"  {s} -> {result}")
    print()

    # The Gradio interface is built but not launched (launching opens a
    # local server and blocks).
    try:
        _ = build_gradio_interface()
        print("Gradio interface built successfully.")
        print("Launch manually with: build_gradio_interface().launch()")
    except ImportError:
        print("(gradio not installed; skipping interface build)")
    print()

    test_fit_and_save_round_trip()
    test_load_model_caches()
    test_validate_row_accepts_valid()
    try:
        import pytest as _pytest  # noqa: F401
        test_validate_row_rejects_bad_age()
        test_validate_row_rejects_missing_key()
        test_validate_row_rejects_bad_sex()
    except ImportError:
        print("(pytest not installed; skipping the raises-ValueError tests)")
    test_predict_one_round_trip()
    test_gradio_predict_returns_string()
    try:
        import pytest as _pytest  # noqa: F811, F401
        test_gradio_predict_handles_bad_input()
    except ImportError:
        pass

    print("OK -- exercise 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
