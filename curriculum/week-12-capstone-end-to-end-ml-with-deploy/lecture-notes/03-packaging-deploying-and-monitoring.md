# Lecture 3 — Packaging, deploying, and monitoring

> *Reading time: about 85 minutes. The deploy walkthroughs in section 5 take an additional 30 minutes once you actually run them. Required reading for this lecture: the Hugging Face Spaces overview (<https://huggingface.co/docs/hub/spaces-overview>), the Gradio quickstart (<https://www.gradio.app/guides/quickstart>), and at least the first three sections of the Breck et al. 2017 ML Test Score paper (<https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>).*

---

## 1. The gap this lecture closes

By Friday morning of capstone week, you have a trained model in `runs/latest/model.joblib`, a model card on disk, a fairness audit table, and a `predict()` function inside `train.py`. The model exists; it lives on your laptop; no one outside the room can use it. This lecture is the part of the capstone that the rest of the C5 curriculum left out: turning a `.joblib` file into a URL that anyone with a browser can visit.

The three commitments of this lecture:

1. **The inference layer is a separate module from the training layer.** `serve.py` and `predict.py` do not import from `train.py`. The training code disappears in production; the inference code is what runs.
2. **The deploy target is one of three: Hugging Face Spaces, Streamlit Community Cloud, or Fly.io.** All three have free tiers. All three are walked through end-to-end below. The capstone rubric requires a working public URL; it does not require any specific platform.
3. **Monitoring is the cheapest viable version, not the production-grade version.** SQLite, `scipy.stats.ks_2samp`, a daily cron-style script. Production monitoring (Prometheus, Grafana, Sentry, Arize, WhyLabs) is a C5+ topic and is *not* graded this week.

---

## 2. The inference layer

The inference layer's contract:

- It accepts one input (a row of features, a piece of text, an image).
- It returns one output (a class label, a regression value, a list of recommended items).
- It does not touch the training data. Ever.
- It does not depend on any artifact except `model.joblib` (or `model.pt`) and `requirements.txt`.
- It validates the input. Bad inputs return an error message, not a 500.

### 2.1 The shape of `predict.py`

A canonical `predict.py` for a tabular binary classifier:

```python
"""Inference entrypoint for the Adult Income capstone model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model.joblib"
THRESHOLD: float = 0.27

# Module-level so cold-start happens once per process.
_model = None


def load_model():  # noqa: ANN201
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(row: dict[str, object]) -> dict[str, object]:
    """Predict a single row.

    Args:
        row: A dictionary of feature_name -> value. Must include every column
            the training pipeline expects.

    Returns:
        A dictionary with keys `label` (str: "<=50K" or ">50K"), `probability`
        (float: 0.0 to 1.0), `threshold` (float).
    """
    model = load_model()
    df = pd.DataFrame([row])
    prob: float = float(model.predict_proba(df)[0, 1])
    label: str = ">50K" if prob >= THRESHOLD else "<=50K"
    return {"label": label, "probability": prob, "threshold": THRESHOLD}


def main() -> int:
    """Read JSON from stdin, write JSON to stdout."""
    raw = sys.stdin.read()
    row = json.loads(raw)
    result = predict_one(row)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The function is the unit of deployment. The CLI `main()` is a convenience for command-line testing; the production callers wrap `predict_one` in HTTP. The module-level `_model` cache prevents reloading the file on every call — this is the single most-common performance bug in deployed inference layers.

### 2.2 Input validation

The inference layer accepts inputs from the outside world. The outside world sends garbage. A canonical mistake is to forward the garbage straight into `model.predict()`, which produces either a 500 error (`KeyError: 'age'`), a silent wrong answer (the model receives a string where it expected an int and outputs nonsense), or a security issue (the model receives a 100 MB tensor as the "feature" and your serverless instance OOMs).

The C5 input-validation pattern is Pydantic for FastAPI, manual checks for Gradio:

```python
def validate_row(row: dict[str, object]) -> dict[str, object]:
    """Raise ValueError if the row is missing required keys or has wrong types."""
    required = {"age", "workclass", "education", "marital_status", "occupation",
                "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "native_country"}
    missing = required - set(row.keys())
    if missing:
        raise ValueError(f"missing keys: {missing}")
    if not isinstance(row["age"], (int, float)):
        raise ValueError(f"age must be numeric, got {type(row['age']).__name__}")
    if row["age"] < 0 or row["age"] > 120:
        raise ValueError(f"age must be in [0, 120], got {row['age']}")
    return row
```

The validation function lives in `predict.py` and runs before `predict_one`. The Gradio interface wraps the `ValueError` in a user-visible message; the FastAPI endpoint returns a 422 with the message.

---

## 3. The three deploy targets

Three platforms, all with free tiers, all suitable for the C5 capstone. Pick one. The rubric is platform-agnostic.

### 3.1 Hugging Face Spaces (the C5 default)

What it is: a git-backed application hosting service for ML demos. A Space is a git repo with an `app.py` (Gradio or Streamlit) and a `requirements.txt`. Push the repo; Hugging Face builds the container and serves it. The free tier runs on a 2-vCPU, 16 GB RAM CPU machine indefinitely. Reference: <https://huggingface.co/docs/hub/spaces-overview>.

When to pick it: 90 percent of C5 capstones. Spaces is the simplest path; the build pipeline is automatic; the URL is permanent; the platform is designed for ML demos.

Limitations: CPU-only on the free tier (the inference must run on CPU); the model file must be under ~50 GB total repo size (huge but practical); the SDK is fixed at app start (you cannot serve a FastAPI app on the same Space as the Gradio one without a Docker SDK).

### 3.2 Streamlit Community Cloud

What it is: a Streamlit-specific hosting service. Connect a GitHub repo, point Streamlit at the `app.py`, get a URL at `<username>-<repo>.streamlit.app`. Reference: <https://docs.streamlit.io/streamlit-community-cloud>.

When to pick it: dashboard-shaped capstones where the UI is a small set of interactive charts and controls. Streamlit's `st.dataframe`, `st.plotly_chart`, and `st.slider` are more ergonomic than Gradio's equivalents for that use case.

Limitations: the free tier runs one app at a time; apps sleep after inactivity and take 10-30 seconds to wake; the platform is Python-only and Streamlit-only.

### 3.3 Fly.io

What it is: a Docker-based platform-as-a-service. Write a `Dockerfile` for a FastAPI app on Uvicorn; `fly launch`; the platform reads the Dockerfile, asks four questions, and deploys. Reference: <https://fly.io/docs/launch/>.

When to pick it: capstones that need a real HTTP API (not a UI). If the deliverable is an endpoint that other code calls, Fly is the right target. Also: the only one of the three that lets you choose the runtime (Python, Node, Rust, anything that fits in a Docker container).

Limitations: the free allowance covers a `shared-cpu-1x` machine with 256 MB of RAM, which fits a sklearn model server but not a transformer. The setup is half an hour of Docker-and-config the other two platforms do not require. Worth it if you want a portfolio piece that shows production-grade serving; overkill if you just need a demo URL.

### 3.4 The decision matrix

| | Capstone type | Recommended target |
|---|---|---|
| Tabular classification / regression, simple form input | Adult Income, California Housing | Hugging Face Spaces + Gradio |
| Dashboard-style exploration | Time-series, recommendation | Streamlit Community Cloud |
| Production-style API | Sentiment-API, "POST text get label" | Fly.io + FastAPI |
| Image classification | CIFAR-10, Caltech-101 | Hugging Face Spaces + Gradio |
| NLP fine-tuned transformer | DistilBERT on IMDB | Hugging Face Spaces + Gradio |

---

## 4. Gradio: the UI library that ships with Spaces

Gradio (<https://www.gradio.app/docs/>) is the C5 default UI library. The API has one entry point: `gr.Interface(fn=..., inputs=..., outputs=...)`. The function `fn` is your Python function; `inputs` and `outputs` are Gradio components.

### 4.1 The minimal `app.py`

```python
"""Gradio app for the Adult Income capstone."""
from __future__ import annotations

import gradio as gr

from predict import predict_one, validate_row


def gradio_predict(
    age: int,
    workclass: str,
    education: str,
    marital_status: str,
    occupation: str,
    relationship: str,
    race: str,
    sex: str,
    capital_gain: int,
    capital_loss: int,
    hours_per_week: int,
    native_country: str,
) -> str:
    row: dict[str, object] = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
    }
    try:
        validate_row(row)
    except ValueError as e:
        return f"Invalid input: {e}"
    result = predict_one(row)
    return (
        f"Predicted label: {result['label']}\n"
        f"Probability: {result['probability']:.3f}\n"
        f"Threshold:   {result['threshold']:.3f}"
    )


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Age", value=39, precision=0),
        gr.Dropdown(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                     "Local-gov", "State-gov", "Without-pay", "Never-worked"],
                    label="Workclass", value="Private"),
        gr.Dropdown(["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                     "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                     "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
                    label="Education", value="Bachelors"),
        gr.Dropdown(["Married-civ-spouse", "Divorced", "Never-married",
                     "Separated", "Widowed", "Married-spouse-absent",
                     "Married-AF-spouse"],
                    label="Marital status", value="Never-married"),
        gr.Dropdown(["Tech-support", "Craft-repair", "Other-service", "Sales",
                     "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                     "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                     "Transport-moving", "Priv-house-serv", "Protective-serv",
                     "Armed-Forces"],
                    label="Occupation", value="Adm-clerical"),
        gr.Dropdown(["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
                     "Unmarried"],
                    label="Relationship", value="Not-in-family"),
        gr.Dropdown(["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other",
                     "Black"],
                    label="Race", value="White"),
        gr.Radio(["Male", "Female"], label="Sex", value="Male"),
        gr.Number(label="Capital gain", value=0, precision=0),
        gr.Number(label="Capital loss", value=0, precision=0),
        gr.Number(label="Hours per week", value=40, precision=0),
        gr.Dropdown(["United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
                     "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan",
                     "Greece", "South", "China", "Cuba", "Iran", "Honduras",
                     "Philippines", "Italy", "Poland", "Jamaica", "Vietnam",
                     "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                     "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary",
                     "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
                     "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
                     "Holand-Netherlands"],
                    label="Native country", value="United-States"),
    ],
    outputs=gr.Text(label="Result"),
    title="Adult Income — Will this individual earn > $50K?",
    description=(
        "Capstone model for C5 Week 12. LightGBM binary classifier on UCI Adult Income (1994 census). "
        "See MODEL_CARD.md for fairness audit and limitations."
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
```

That is it. About 100 lines for a working, deployable, public ML demo. The same `app.py` runs locally (visit `http://localhost:7860`) and on Hugging Face Spaces.

### 4.2 The Spaces `README.md` frontmatter

Every Hugging Face Space has a `README.md` at the repo root with YAML frontmatter that picks the SDK:

```yaml
---
title: C5 Capstone Adult Income
emoji: U+1F4CA
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---
```

The `sdk` field is `gradio`, `streamlit`, or `docker`. The `sdk_version` pins the Gradio version (always pin; otherwise the Space rebuilds on every Gradio release and breaks). The `app_file` points to `app.py`.

The body of the `README.md` is the public-facing description of the Space and should be a short summary of `MODEL_CARD.md` with a link to the full card.

### 4.3 Deploying to Hugging Face Spaces

Step by step:

1. Sign in at <https://huggingface.co/> and create a new Space at <https://huggingface.co/new-space>. Pick SDK = Gradio, hardware = CPU basic (free), visibility = public.
2. Clone the empty Space repo locally:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/c5-capstone-adult-income
   cd c5-capstone-adult-income
   ```

3. Copy `app.py`, `predict.py`, `model.joblib`, `requirements.txt`, `MODEL_CARD.md` into the directory.
4. Edit `README.md` to add the YAML frontmatter above.
5. Push:

   ```bash
   git lfs install
   git lfs track "*.joblib"
   git add .gitattributes
   git add .
   git commit -m "initial capstone deploy"
   git push
   ```

6. Wait three to ten minutes for the Space to build. Visit `https://<your-username>-c5-capstone-adult-income.hf.space` (or whatever the Space's URL is).
7. If the build fails, click the "Logs" tab on the Space page; the most common failures are `requirements.txt` typos and `model.joblib` over the 10 MB git-LFS-required threshold (use `git lfs track`).

The URL is permanent. Put it on your portfolio README.

> **EXPERIMENT 3.1 — deploy a `gr.Interface(fn=lambda x: x.upper(), inputs="text", outputs="text")` to Hugging Face Spaces.** Do not wait for your capstone model to be ready. Deploy a trivial Space first; learn the workflow on a five-line app; then come back to the capstone. The exercise takes fifteen minutes and saves an hour of confusion later.

---

## 5. Streamlit: the dashboard alternative

Streamlit (<https://docs.streamlit.io/>) is the C5 alternative for capstones with dashboard-shaped UIs. The API is "your Python script reruns top to bottom on every interaction"; the `st.session_state` dictionary holds in-memory state across reruns.

### 5.1 A minimal Streamlit `app.py`

```python
"""Streamlit app for the Adult Income capstone."""
from __future__ import annotations

import streamlit as st

from predict import predict_one


@st.cache_resource
def get_model():  # noqa: ANN201
    from predict import load_model
    return load_model()


st.title("Adult Income — Will this individual earn > $50K?")
st.markdown(
    "Capstone model for C5 Week 12. See `MODEL_CARD.md` for fairness audit and limitations."
)

age: int = st.slider("Age", 16, 100, 39)
sex: str = st.radio("Sex", ["Male", "Female"], index=0)
hours: int = st.slider("Hours per week", 1, 99, 40)
# ... (additional inputs omitted for brevity) ...

if st.button("Predict"):
    row = {"age": age, "sex": sex, "hours_per_week": hours, "...": "..."}
    result = predict_one(row)
    st.metric("Probability", f"{result['probability']:.3f}")
    st.write(f"Predicted label: **{result['label']}**")
```

The `@st.cache_resource` decorator memoizes the model load so the file is not reread on every interaction; this is the single most-common Streamlit performance bug.

### 5.2 Deploying to Streamlit Community Cloud

Step by step:

1. Sign in at <https://streamlit.io/cloud> with your GitHub account.
2. Click "New app". Point at your GitHub repo's `main` branch, file path `app.py`.
3. Click "Deploy". Wait two minutes. The app URL is `https://<username>-<repo>-app-<hash>.streamlit.app`.
4. The app sleeps after seven days of inactivity. Visit it to wake it up. Cold-start is 30 seconds.

---

## 6. FastAPI + Fly.io: the API alternative

FastAPI (<https://fastapi.tiangolo.com/>) is the C5 alternative when the deliverable is an HTTP API rather than a UI. Fly.io (<https://fly.io/docs/>) is the deploy target with a free allowance that fits a small FastAPI app.

### 6.1 A minimal FastAPI `serve.py`

```python
"""FastAPI server for the Adult Income capstone."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_one, validate_row


class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., ge=0)
    capital_loss: int = Field(..., ge=0)
    hours_per_week: int = Field(..., ge=0, le=168)
    native_country: str


class PredictResponse(BaseModel):
    label: str
    probability: float
    threshold: float


app = FastAPI(title="C5 Capstone Adult Income", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    try:
        validate_row(req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    result = predict_one(req.model_dump())
    return PredictResponse(**result)
```

The `BaseModel` subclass validates the request body automatically (FastAPI uses Pydantic). The `/health` endpoint is a liveness check that Fly.io polls every 30 seconds. The `/docs` endpoint (free with FastAPI) is the auto-generated Swagger UI; share it as part of the deliverable.

### 6.2 The Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py serve.py model.joblib ./

EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
```

The `python:3.11-slim` base image is 60 MB; the full container with sklearn and lightgbm is roughly 400 MB; well under any free-tier image-size cap.

### 6.3 Deploying to Fly.io

Step by step:

1. Install the `flyctl` CLI: <https://fly.io/docs/hands-on/install-flyctl/>.
2. `flyctl auth signup` and verify your email.
3. In the repo directory, `flyctl launch --no-deploy`. Answer the four questions: app name, region (the closest one), Postgres (no), Redis (no).
4. The CLI writes a `fly.toml` config. Open it, set `memory_mb = 256` to stay in the free allowance, set `[[services]] internal_port = 8080`.
5. `flyctl deploy`. Wait three minutes. The URL is `https://<app-name>.fly.dev`.
6. Test: `curl https://<app-name>.fly.dev/health` should return `{"status":"ok"}`.

The free allowance covers ~3 machines indefinitely. The catch: the machine sleeps after inactivity; the first request after a sleep takes 5-10 seconds. For a portfolio piece this is acceptable; for production it is not.

---

## 7. Basic monitoring

The capstone's monitoring is the cheapest viable version. Three signals, logged to SQLite, summarized once a day.

### 7.1 The signals

- **Latency.** How long each prediction took. The 50th and 99th percentile latency are the deploy-quality signals.
- **Input drift.** How different today's inputs are from the training-set inputs. A two-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) per feature, computed on a rolling 24-hour window.
- **Prediction drift.** How different today's predicted-probability distribution is from yesterday's. A KL-divergence on the histogrammed probabilities.

### 7.2 The SQLite logger

```python
"""SQLite logger for the inference layer."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "predictions.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ts REAL PRIMARY KEY,
            input_json TEXT NOT NULL,
            output_json TEXT NOT NULL,
            latency_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(
    input_dict: dict[str, object],
    output_dict: dict[str, object],
    latency_ms: float,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions(ts, input_json, output_json, latency_ms) VALUES (?, ?, ?, ?)",
        (time.time(), json.dumps(input_dict), json.dumps(output_dict), latency_ms),
    )
    conn.commit()
    conn.close()
```

The logger is called from inside `predict_one` (or from the FastAPI / Gradio wrapper). The SQLite file lives on the Space's persistent storage (Hugging Face supports persistent storage on the free tier with a one-line config; <https://huggingface.co/docs/hub/spaces-storage>).

### 7.3 The daily drift script

```python
"""Daily drift report."""
from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def load_recent(hours: int = 24) -> pd.DataFrame:
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query(
        "SELECT * FROM predictions WHERE ts > ?",
        conn,
        params=(time.time() - hours * 3600,),
    )
    conn.close()
    inputs_df = pd.json_normalize(df["input_json"].apply(json.loads))
    return inputs_df


def baseline_inputs() -> pd.DataFrame:
    # The training set, saved as a parquet at training time.
    return pd.read_parquet("data/baseline_inputs.parquet")


def drift_report() -> dict[str, float]:
    recent = load_recent()
    baseline = baseline_inputs()
    if len(recent) == 0:
        return {"status": "no_recent_predictions"}
    report: dict[str, float] = {}
    for col in baseline.select_dtypes(include="number").columns:
        if col not in recent.columns:
            continue
        stat, _p = ks_2samp(baseline[col], recent[col])
        report[col] = float(stat)
    return report


if __name__ == "__main__":
    print(drift_report())
```

The drift report is a column name to KS-statistic mapping. A KS statistic over 0.2 on any feature is a signal that the distribution has shifted; over 0.4 is a signal that the model probably needs retraining.

### 7.4 Where the daily run lives

For the capstone, the daily run lives in a `scripts/run_drift.py` that is intended to be run manually or via a Hugging Face Space "scheduled run" (Spaces support cron-style triggers on paid tiers; on the free tier the C5 capstone runs the script weekly by hand). The capstone is graded on the *existence* of the monitor, not on the production scheduling — the C5+ industry track covers the scheduling and the alerting.

> **EXPERIMENT 3.2 — log 100 predictions and run the drift report.** From a Python REPL, call `predict_one` 100 times with rows sampled from your training set and call it another 100 times with rows where you have added 10 to every numeric feature. Run the drift report. The KS statistic on the shifted features should be near 1.0; the unshifted features should be near 0.0. The exercise calibrates the threshold for your problem.

---

## 8. The handoff document

The last file in the capstone repo is `HANDOFF.md`. The handoff doc is the document that tells the next engineer (or your future self in three months) how to operate the system. The template:

```text
# HANDOFF — <capstone name>

## What this is
<one paragraph: what the model does, who it serves, what URL it is at>

## How to retrain
<numbered steps: clone, pip install, dvc pull, python train.py, what to check>

## How to redeploy
<numbered steps: push to Spaces / Fly / Streamlit, wait, verify URL responds>

## What to do if something breaks
<a triage tree: URL is down (check Spaces status), predictions look wrong (check drift report), latency spiked (check logs)>

## What I would do differently
<bullets: the three things you would change if you had another week>

## Open questions
<bullets: the things you do not know that you wish you did>
```

The handoff doc is graded. The grader reads it; the grader notes whether the "what to do differently" bullets show technical maturity. The single most-impressive HANDOFF.md feature is honesty about the limitations.

---

## 9. The deploy checklist

Before declaring the capstone done, work through this checklist:

- [ ] `app.py` runs locally. `python app.py` opens at `http://localhost:7860` and serves predictions.
- [ ] `requirements.txt` is pinned. Every package has a version number.
- [ ] `model.joblib` (or `model.pt`) is in the repo or in DVC.
- [ ] `predict.py` runs from the CLI. `echo '{"age": 39, ...}' | python predict.py` returns a JSON.
- [ ] The Space / Streamlit app / Fly app builds and responds to a request.
- [ ] The URL is in the README.
- [ ] `MODEL_CARD.md` is at the repo root and references the public URL.
- [ ] `DATASET_CARD.md` is at `data/DATASET_CARD.md`.
- [ ] `PROBLEM.md` is at the repo root.
- [ ] `HANDOFF.md` is at the repo root.
- [ ] The fairness audit table is in `figures/fairness_audit.csv` and rendered in `MODEL_CARD.md`.
- [ ] The drift report runs. The output is in `runs/latest/drift.json`.
- [ ] Every protected attribute the dataset carries has been audited; the result is in the model card.
- [ ] The Mitchell template's nine sections are all filled in the model card.
- [ ] The Gebru template's seven sections are all filled in the dataset card.

Twelve checks. If any are missing, the capstone is not ready.

---

## 10. Recap

- The inference layer is `predict.py` and `serve.py`, separate from `train.py`. The training code does not run in production.
- Validate inputs before calling the model. Bad inputs return error messages, not 500s.
- Pick one of three deploy targets — Hugging Face Spaces (the C5 default), Streamlit Community Cloud, or Fly.io — and follow the walkthrough.
- Gradio's `gr.Interface(fn=..., inputs=..., outputs=...)` is the simplest UI. Pin the version in the Spaces README frontmatter.
- Streamlit's `@st.cache_resource` is non-optional for model loads.
- FastAPI's Pydantic models handle input validation and auto-generate Swagger docs at `/docs`.
- Monitor by logging every prediction to SQLite and running a daily KS-test drift report. The cheapest viable version is enough for the capstone.
- Write `HANDOFF.md` last; it is the artifact a hiring manager reads most carefully.

The mini-project (`mini-project/README.md`) ties the three lectures together. You have one week. The deliverable is a public URL.
