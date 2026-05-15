# Week 12 — Capstone: End-to-End ML with Deployment

> *Week 11 closed with a working mini-GPT that produced English-shaped samples from `Pride and Prejudice` after forty minutes on CPU. The model existed; it lived in a `.pt` file on your laptop; no one else could touch it. That is the gap Week 12 closes. The capstone takes one model — yours, from a problem domain you pick — and walks it through the unglamorous half of the lifecycle: dataset acquisition with a written **dataset card** (Gebru et al. 2021, <https://arxiv.org/abs/1803.09010>), cleaning and exploratory data analysis with a written record of every decision, model selection driven by a baseline you must beat, training with versioned artifacts (`dvc` against a free object store, or LakeFS Cloud's free tier, <https://lakefs.io/pricing/>), evaluation with a slice-based fairness audit, a written **model card** (Mitchell et al. 2019, <https://arxiv.org/abs/1810.03993>), packaging into a thin inference layer, and deployment to a free public URL — Hugging Face Spaces free tier (<https://huggingface.co/docs/hub/spaces-overview>), Streamlit Community Cloud (<https://streamlit.io/cloud>), or Fly.io's free-allowance tier (<https://fly.io/docs/about/pricing/>). The deliverable at the end of Week 12 is a URL someone can paste into a browser and a repo a hiring manager can read. C5 ends with a portfolio piece, not a notebook.*

Welcome to the final week of **C5 · Crunch AI / Data Science**. Twelve weeks ago you computed a dot product with NumPy and called it deep learning. Eleven weeks ago you opened your first `pandas` DataFrame and discovered that 4 percent of your column was the string `"N/A"` and that pandas had read it as the literal text. Six weeks ago you trained your first gradient-boosted tree on tabular data, six days ago you trained your first decoder-only transformer on character text. The arc of the course was always heading here: take **one** problem, **your** problem, and ship it. The capstone is that ship.

Four commitments before we start:

1. **Pick one problem, not three.** The capstone rubric scores depth, not breadth. A regression model on Austin home prices that you have written a fairness audit for, deployed to a Streamlit Cloud URL with a written model card, and monitored for drift over a week is a portfolio piece. Three half-trained models on three datasets is not. The lecture notes spend roughly equal time on six problem archetypes (binary classification, regression, multi-class image classification, sequence-to-label NLP, time-series forecasting, and recommendation); pick one Monday morning and do not pick a second on Friday.
2. **The deploy target is non-negotiable.** A capstone that ends with `python predict.py` on your laptop does not pass. The rubric requires a publicly reachable URL on a free tier (Hugging Face Spaces, Streamlit Community Cloud, or Fly.io's free allowance). All three are documented in Lecture 3 with screen-by-screen setup walkthroughs. The deploy step is twenty minutes of work after the model is trained; it is the entire reason the previous five days of work exist.
3. **The model card and the dataset card are not optional.** The C5 curriculum has named Mitchell et al. 2019 ("Model Cards for Model Reporting", <https://arxiv.org/abs/1810.03993>) and Gebru et al. 2021 ("Datasheets for Datasets", <https://arxiv.org/abs/1803.09010>) as required reading since Week 4. Week 12 makes both load-bearing. The rubric gives 15 of 100 points to a model card that follows the Mitchell template and 10 to a dataset card that follows the Gebru questionnaire. A capstone with strong accuracy and a missing model card scores in the C range; a capstone with mediocre accuracy and a thorough model card scores in the B range. This is intentional; this is how the field is currently calibrating.
4. **Reproducibility is a graded artifact, not a hope.** The rubric requires a `requirements.txt` with pinned versions, a `Makefile` (or one shell script) that runs end-to-end on a fresh checkout, a versioned dataset artifact (DVC remote on a free S3-compatible object store, or LakeFS Cloud free tier, or — the C5 fallback — a pinned Hugging Face Datasets URL with a commit hash), a versioned model artifact (a `.pt`/`.joblib` file with a SHA256 in the model card), and a `seed = 42` somewhere in `train.py`. "It ran on my laptop last Tuesday" is not reproducibility.

We do not pin a single framework this week. The capstone is framework-agnostic by design — pick `scikit-learn`, PyTorch, TensorFlow/Keras, JAX, XGBoost, LightGBM, or HuggingFace `transformers` based on which is the right tool for the problem you picked. The lecture notes show working starters in `scikit-learn` (tabular), PyTorch (image, NLP), and LightGBM (tabular high-performance) because those three cover roughly 95 percent of the realistic problem space for a one-week capstone.

The official reference docs you will live in this week:

- **Hugging Face Spaces** (the free hosting platform for Gradio and Streamlit apps; the C5 default deploy target): <https://huggingface.co/docs/hub/spaces-overview>
- **Gradio** (the simplest "wrap a Python function in a web UI" library; ships with Hugging Face Spaces): <https://www.gradio.app/docs/>
- **Streamlit** (the Python-only dashboard framework; deploys to Streamlit Community Cloud or Hugging Face Spaces): <https://docs.streamlit.io/>
- **FastAPI** (the production-grade Python web framework for serving model APIs; deploys to Fly.io's free allowance): <https://fastapi.tiangolo.com/>
- **Fly.io** (the platform-as-a-service with a free allowance that fits a small FastAPI model server): <https://fly.io/docs/>
- **DVC** (Data Version Control; tracks datasets and model artifacts in git without storing the bytes in git): <https://dvc.org/doc>
- **LakeFS** (an alternative git-for-data layer; the LakeFS Cloud free tier covers small projects): <https://docs.lakefs.io/>

Pin all seven.

---

## Learning objectives

By the end of this week, you will be able to:

- **Frame an ML problem from a vague stakeholder request.** Given the brief "we want to know who is likely to churn", convert it into a written **problem statement** with a stated prediction target, prediction window, success metric (and the *unit* it is measured in — dollars saved, false positive rate at a fixed recall, BLEU at a fixed length budget), a baseline number to beat (the always-predict-majority-class baseline, the previous-month-rate baseline, the SQL `LIKE`-rule baseline that a non-ML team would use), and an acceptance threshold. The problem statement is the first artifact in the capstone repository and is graded.
- **Pick a dataset and write its dataset card.** Use one of the eight C5-approved sources (the Hugging Face Hub, OpenML, Kaggle Datasets, UCI ML Repository, Project Gutenberg, Wikipedia dumps, public US government data portals at data.gov, public city open-data portals), then write a dataset card that answers the Gebru et al. 2021 questionnaire (motivation, composition, collection process, preprocessing, uses, distribution, maintenance). The dataset card lives at `data/DATASET_CARD.md` in the capstone repo and is graded.
- **Perform exploratory data analysis with a written narrative.** Open the dataset in `pandas`, run `df.info()`, `df.describe()`, `df.isna().sum()`, plot a histogram of every numeric column and a value-count bar of every categorical, identify class imbalance, identify the columns that have the missing-data pattern Lecture 1 names "missing-not-at-random". Write the findings into `notebooks/01-eda.ipynb` or `notebooks/01-eda.md`. The EDA notebook is graded on clarity and on whether it flags the problems the dataset actually has.
- **Pick a baseline and beat it.** No model is shipped without a baseline. The baseline is the dumbest thing that could possibly work — `DummyClassifier(strategy="most_frequent")`, `DummyRegressor(strategy="mean")`, a one-rule decision (e.g., predict "spam" if the email contains "viagra"), or a logistic regression on a single feature. The candidate model must beat the baseline on the held-out test set by an effect size large enough to be worth deploying. "Beat the baseline by 0.3 percent" is not worth deploying; "beat the baseline by 8 percent" is. This is a Week-4 lesson; Week 12 makes it the rubric.
- **Train one model with versioned artifacts.** Use `dvc` to version the dataset (free remote on a DigitalOcean Spaces or Cloudflare R2 free-allowance bucket, or — the C5 fallback — a versioned Hugging Face Dataset URL with a commit hash), use a SHA256 of the trained model file as the version identifier in the model card, and log every hyperparameter to a `runs/run-<datetime>/config.yaml`. Reproducibility means someone else can clone the repo, run `make train`, and produce a model with the same SHA256 you produced (within numerical noise; we discuss the determinism caveats in Lecture 2).
- **Evaluate with metrics that fit the problem.** Use `precision_recall_curve` and `roc_auc_score` for binary classification with imbalanced classes; use `RMSE` and `MAE` for regression; use `BLEU`, `ROUGE-L`, and exact-match accuracy for sequence tasks. *Never* report only accuracy on an imbalanced classifier; this is a Week-4 sin that we revisit in Lecture 2. Report metrics on a held-out test set, not on the validation set, and not on any data that touched hyperparameter search.
- **Run a slice-based fairness audit.** Split the test set by every protected attribute the dataset carries (sex, race, age bracket, geography) and report the headline metric per slice. If the gap between the best-performing slice and the worst-performing slice exceeds the threshold named in the model card, flag it. The C5 default threshold is 5 percentage points on the headline metric; the model card explains why this threshold is right for *your* problem. Reference: Buolamwini and Gebru 2018 ("Gender Shades", <https://proceedings.mlr.press/v81/buolamwini18a.html>), Bird et al. 2020 ("Fairlearn", <https://arxiv.org/abs/2310.10696> for the toolkit paper and <https://fairlearn.org/> for the docs).
- **Write a model card following the Mitchell template.** The card answers: what the model does, what the intended use is, what it should *not* be used for, the metrics, the training data, the evaluation data, the fairness analysis, the ethical considerations, the caveats, and the contact. The Mitchell paper (<https://arxiv.org/abs/1810.03993>) is short and is required reading; the C5 template at `mini-project/MODEL_CARD_TEMPLATE.md` is a one-page adaptation. The card lives at `MODEL_CARD.md` in the capstone repo and is graded.
- **Package the model behind a thin inference layer.** Write a `predict.py` that takes either CLI arguments or stdin JSON and prints predictions; write a `serve.py` that exposes the same model over HTTP (FastAPI or Gradio). The inference layer is the only code that runs in production; it must not import from `train.py`, must not reach into the training-set pandas DataFrame, and must not depend on any artifact that is not in `requirements.txt` plus the one model file.
- **Deploy to a free public URL.** Pick one of three targets: Hugging Face Spaces with a Gradio app (the easiest; 90 percent of C5 capstones land here), Streamlit Community Cloud (slick but Python-only and rate-limited), or Fly.io with a FastAPI app in a Docker container (the most flexible; the only choice if you need a real HTTP API rather than a UI). Lecture 3 has screen-by-screen walkthroughs for all three. The deploy step is twenty minutes once the model is trained; the *prep* — pinning versions, writing the `Dockerfile` or the `app.py`, registering the secrets — is half an hour. Budget an hour total for "model in `train.py`" to "URL in the browser".
- **Set up basic monitoring.** Add a `logging` block to `serve.py` that records every prediction (input hash, output, latency) to a SQLite or DuckDB file mounted on the Space; once a day, compute the data-drift score (a `scipy.stats.ks_2samp` test on each numeric feature) and the prediction-distribution drift (`KL(p_today || p_baseline)`). Production monitoring is a full Week 13 topic at staff level; the C5 capstone implements the cheapest viable version. Reference: Klaise et al. 2020 ("Alibi Detect", <https://github.com/SeldonIO/alibi-detect>) for the open-source toolkit; the C5 implementation rolls its own from `scipy.stats` because the lesson is the *concept*, not the library.
- **Cite the foundational references.** Mitchell et al. 2019 ("Model Cards for Model Reporting", <https://arxiv.org/abs/1810.03993>), Gebru et al. 2021 ("Datasheets for Datasets", <https://arxiv.org/abs/1803.09010>), Hugging Face Spaces docs (<https://huggingface.co/docs/hub/spaces-overview>), Gradio docs (<https://www.gradio.app/docs/>), Streamlit docs (<https://docs.streamlit.io/>), FastAPI docs (<https://fastapi.tiangolo.com/>), Fly.io docs (<https://fly.io/docs/>), DVC docs (<https://dvc.org/doc>).
- **Ship** a capstone repository — problem statement, dataset card, EDA notebook, training script, evaluation report, fairness audit, model card, inference layer, deployed URL, monitoring sketch — pushed to your portfolio repo. The C5 grader reads the repo top to bottom; ordering matters.
- **Pass** the final exam (the Week 12 quiz, which doubles as the course-wide review).

---

## Prerequisites

- **All of Weeks 1-11.** This is the capstone; everything is on the table. You will use `pandas` (W2), `matplotlib` (W3), `scikit-learn` (W4, W5), `PyTorch` (W7-W11) or one of the high-performance tabular libraries (`xgboost`, `lightgbm`; introduced as references in W5). You will use cross-validation (W4), regularization (W4, W7), train/val/test splits (W4), and the bias-variance vocabulary (W4) on the daily. If any of those words feel cold, re-read the relevant week's lecture-1 file before Monday.
- **Python 3.11+** with the C5 standard stack installed. The starter `requirements.txt` lists the union of all libraries used in Weeks 1-11; your final `requirements.txt` will be a small subset.
- **A GitHub account and `gh` CLI logged in.** The portfolio repo lives on GitHub.
- **A free Hugging Face account.** You will need it for Spaces deployment. Sign up at <https://huggingface.co/join> — no payment information required.
- **About 200 MB of disk** for the capstone working directory. Datasets up to ~1 GB fit; anything larger needs DVC pointing at an off-disk remote.
- **Optional:** a Streamlit Cloud account (<https://streamlit.io/cloud>) and/or a Fly.io account (<https://fly.io/app/sign-up>) if you deploy to one of those instead of Hugging Face Spaces. All three platforms have free tiers that fit the capstone; you only need one.

You should already be comfortable with:

- **Git basics and feature-branch workflow.** The capstone repository has a `main` branch that is the deployable HEAD; experimental work lives on branches. We do not relitigate git this week.
- **`pip`, `venv`, and `requirements.txt`.** Reproducibility starts here.
- **The Week 4 ML workflow vocabulary.** Train/val/test, cross-validation, overfitting, regularization, the gap between training metric and test metric.
- **The Week 5 tree-based models.** A random forest or a gradient-boosted tree (LightGBM or XGBoost) is the right baseline-and-final model for most tabular problems and is the first thing the rubric checks if your problem is tabular.

---

## Topics covered

- **Problem framing.** Translating "we want to know X" into a target, a prediction window, a metric, and a baseline. The single most underrated skill in industrial ML; the difference between a six-week project that ships and a six-week project that gets cancelled.
- **Dataset selection from free sources.** The Hugging Face Hub (<https://huggingface.co/datasets>; the modern default), OpenML (<https://www.openml.org/>; tabular focus, great metadata), Kaggle Datasets (<https://www.kaggle.com/datasets>; large variety, variable quality), UCI ML Repository (<https://archive.ics.uci.edu/>; classic small tabular sets), data.gov (<https://www.data.gov/>; US government open data), city open-data portals (<https://opendata.dc.gov/>, <https://data.cityofchicago.org/>, etc.), Project Gutenberg (text; reused from W10/W11), Wikipedia dumps (text; <https://dumps.wikimedia.org/>). All free, all permissively licensed, all suitable for a graded capstone.
- **Dataset cards.** Following Gebru et al. 2021 ("Datasheets for Datasets", <https://arxiv.org/abs/1803.09010>): the seven-section template (motivation, composition, collection, preprocessing, uses, distribution, maintenance). The C5 dataset card at `data/DATASET_CARD.md` is graded against this template.
- **EDA as written evidence, not silent exploration.** The capstone EDA notebook is read by a grader; the grader cannot see what you tried and rejected. Every plot in the notebook should have a one-sentence caption stating what it shows. Every column in the dataset should have at least one mention. The "I plotted everything but did not write it up" capstone fails the EDA rubric.
- **Train/val/test discipline.** The test set is sacred; touch it once at the very end. The validation set is for hyperparameter search and early-stopping decisions. The training set is for fitting. The capstone rubric checks for any reference to the test set in the training script and fails the submission if it finds one outside of the final evaluation block.
- **The baseline.** `DummyClassifier`, `DummyRegressor`, or the simplest non-ML rule. Documented in the model card alongside the candidate model. The rubric requires the baseline to be a number, not a phrase.
- **Model selection.** For tabular: gradient-boosted trees (`lightgbm` or `xgboost`) are the right default; the C5 rubric accepts `RandomForestClassifier` from `scikit-learn` as a fall-back. For image classification: a pretrained `torchvision` ResNet-18 fine-tuned for ten epochs (the Week 9 recipe). For text classification: a pretrained `distilbert-base-uncased` fine-tuned for three epochs (the C5 Week 13 preview; if you have not seen transformers fine-tuning before, use a `LogisticRegression` on `TfidfVectorizer` features instead). For time series: `prophet` (<https://facebook.github.io/prophet/>) or `statsmodels` `ARIMA` for the baseline, a `LightGBM` with engineered lag features for the candidate. For sequence-to-something: a small transformer from Week 11.
- **Hyperparameter search.** Either grid search (`GridSearchCV` from `sklearn.model_selection`; appropriate when the search space is small and tabular) or random search (`RandomizedSearchCV`; appropriate when the search space is large). The C5 rubric does *not* require Bayesian optimization (Optuna, `scikit-optimize`); we mention them and move on. The search must be done *only* on training and validation data; the test set is reserved for the final number.
- **Metrics that fit the problem.** Binary classification with imbalance: precision-recall curve, ROC-AUC, F1, the precision-at-fixed-recall and recall-at-fixed-precision operating points. Regression: RMSE, MAE, R-squared, a residual plot. Multi-class: the macro and micro F1, the per-class confusion matrix. Sequence: BLEU, ROUGE-L, perplexity, exact-match. Recommendation: Recall@K, NDCG@K. Time series: MAPE, sMAPE, MASE.
- **Slice-based fairness audit.** Split the test set by every protected attribute and report the headline metric per slice. If the dataset does not carry protected attributes directly, infer them from indirect signals where ethically permissible and where the dataset card states the inference. If the dataset is fundamentally not amenable to a fairness audit (a regression on weather data has no protected attributes), the model card states *why* the audit does not apply. The audit is graded on rigor, not on the result.
- **Model card.** Mitchell et al. 2019 template, adapted. The card lives at the repo root. The grader reads it first.
- **Versioning datasets with DVC.** `dvc init`, `dvc add data/raw.csv`, push to a free S3-compatible remote. The `.dvc` file (a tiny YAML with a hash) is committed to git; the actual data is not. Reference: <https://dvc.org/doc/start>. The C5 alternative is to pin a Hugging Face Datasets URL with a commit hash, which is simpler if the dataset already lives on the Hub.
- **Versioning models.** A SHA256 of the saved model file is recorded in the model card. The artifact itself is either committed to git (small models, < 50 MB), pushed to a DVC remote (medium models, 50 MB to 5 GB), or uploaded to the Hugging Face Hub as a model repo (any size; the modern default for transformer-shaped artifacts).
- **Packaging.** A `predict.py` that does inference from CLI; a `serve.py` that exposes the model over HTTP via Gradio, Streamlit, or FastAPI. The inference layer is the production interface; treat it as such.
- **Deploying to Hugging Face Spaces.** A `Spaces` is a git repo with an `app.py` (Gradio or Streamlit), a `requirements.txt`, and optionally a `Dockerfile`. Push the repo, wait three minutes for the build, get a public URL. The free tier runs on a 2-vCPU, 16 GB RAM CPU machine — enough for inference on any model under 1 GB. Reference: <https://huggingface.co/docs/hub/spaces-overview>.
- **Deploying to Streamlit Community Cloud.** Connect a GitHub repo, point Streamlit at `app.py`, get a URL at `<username>-<repo>.streamlit.app`. Free tier runs one app at a time and sleeps after inactivity. Reference: <https://docs.streamlit.io/streamlit-community-cloud>.
- **Deploying to Fly.io.** Write a `Dockerfile` that runs a FastAPI app on Uvicorn; `fly launch`, `fly deploy`. The free allowance covers a `shared-cpu-1x` machine with 256 MB of RAM, which fits a `scikit-learn` model server but not a transformer. Reference: <https://fly.io/docs/launch/>.
- **Basic monitoring.** Log every prediction to a SQLite file; compute drift scores once a day. Production-grade monitoring (Prometheus, Grafana, Sentry, Arize, WhyLabs, Fiddler, Truera) is a Week-13 topic at staff level; the C5 capstone implements the cheapest viable version. Reference: Breck et al. 2017 ("The ML Test Score", <https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>) for the production-readiness checklist; we score against it in the rubric.
- **The handoff document.** A `HANDOFF.md` at the repo root that tells the next engineer (or your future self) how to retrain, redeploy, and respond to an incident. The handoff doc is graded; it is also the single artifact that hiring managers read most carefully.

---

## Weekly schedule

Target about **45 hours**. The capstone takes longer than a normal week because the deliverable is bigger. Monday is problem framing and dataset selection. Tuesday is EDA and the baseline. Wednesday is training and evaluation. Thursday is the fairness audit and the model card. Friday is packaging and deploy. Saturday is monitoring and the writeup. Sunday is the final exam (the Week 12 quiz, which doubles as a course review) and the portfolio push.

| Day       | Focus                                                                  | Lectures | Exercises | Challenges | Quiz/Read | Homework | Mini-Project | Self-Study | Daily Total |
|-----------|------------------------------------------------------------------------|---------:|----------:|-----------:|----------:|---------:|-------------:|-----------:|------------:|
| Monday    | Problem framing; dataset selection; dataset card; EDA setup            |   3h     |   1.5h    |     0h     |   0.5h    |   1h     |     2h       |   0.5h     |    8.5h     |
| Tuesday   | EDA; baseline; train/val/test discipline; first model                   |   2h     |   1.5h    |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    8.5h     |
| Wednesday | Model selection; hyperparameter search; evaluation metrics             |   2h     |   1.5h    |     2h     |   0.5h    |   1h     |     3h       |   0h       |    10h      |
| Thursday  | Fairness audit; model card; dataset card finalization; reproducibility |   2h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0h       |    7.5h     |
| Friday    | Packaging; Gradio / Streamlit / FastAPI; deploy to Spaces or Fly       |   2h     |   1h      |     0h     |   0.5h    |   1h     |     3h       |   0.5h     |    8h       |
| Saturday  | Monitoring; the handoff doc; portfolio polish                          |   0h     |   0h      |     2h     |   0.5h    |   1h     |     3h       |   0h       |    6.5h     |
| Sunday    | Final exam; portfolio push; reflection                                  |   0h     |   0h      |     0h     |   2h      |   0h     |     1h       |   0.5h     |    3.5h     |
| **Total** |                                                                        | **11h**  | **6.5h**  | **4h**     | **5h**    | **6h**   | **18h**      | **2h**     |  **52.5h**  |

The mini-project block is **18 hours** this week, not the usual 8. That is the capstone; that is where the bulk of the grade lives. The lecture blocks are correspondingly shorter — by Week 12 you have seen the workflow eleven times, and the lectures focus on the *new* material (deploy, model cards, fairness audits, monitoring) rather than re-covering the ML basics.

---

## How to navigate this week

| File | What is inside |
|------|----------------|
| [README.md](./README.md) | This overview |
| [resources.md](./resources.md) | The Mitchell and Gebru papers; the deploy platform docs; the Fairlearn and Aequitas toolkits; the production-readiness literature |
| [lecture-notes/01-problem-framing-data-cards-and-eda.md](./lecture-notes/01-problem-framing-data-cards-and-eda.md) | Translating a stakeholder request into a problem statement; picking and documenting a dataset; the Gebru et al. dataset card; EDA as written evidence |
| [lecture-notes/02-evaluation-fairness-and-model-cards.md](./lecture-notes/02-evaluation-fairness-and-model-cards.md) | Metrics that fit the problem; the slice-based fairness audit; the Mitchell et al. model card; versioning datasets and models |
| [lecture-notes/03-packaging-deploying-and-monitoring.md](./lecture-notes/03-packaging-deploying-and-monitoring.md) | Packaging the inference layer; Gradio, Streamlit, FastAPI; deploying to Hugging Face Spaces, Streamlit Cloud, Fly.io; basic drift monitoring |
| [exercises/exercise-01-dataset-card-and-eda.py](./exercises/exercise-01-dataset-card-and-eda.py) | Load the UCI Adult Income dataset, write a dataset card, run EDA, flag the missing-not-at-random columns |
| [exercises/exercise-02-baseline-and-fairness-audit.py](./exercises/exercise-02-baseline-and-fairness-audit.py) | Train `DummyClassifier` and `LogisticRegression` on Adult Income, evaluate per-slice, write a fairness-audit table |
| [exercises/exercise-03-package-and-gradio.py](./exercises/exercise-03-package-and-gradio.py) | Save a trained model, write a `predict()` function, wrap it in a Gradio interface, run locally |
| [exercises/SOLUTIONS.md](./exercises/SOLUTIONS.md) | Reference solutions with commentary; read only after attempting |
| [challenges/challenge-01-deploy-to-hugging-face-spaces.md](./challenges/challenge-01-deploy-to-hugging-face-spaces.md) | Take Exercise 3's Gradio app, push it to a Hugging Face Space, share the public URL |
| [challenges/challenge-02-drift-monitor-on-sqlite.md](./challenges/challenge-02-drift-monitor-on-sqlite.md) | Log predictions to SQLite from `serve.py`, run a daily KS-test drift score, plot the drift over a week |
| [quiz.md](./quiz.md) | The final exam: 20 questions, course-wide; counts as the Week 12 quiz and the C5 course review |
| [homework.md](./homework.md) | Six problems that exercise the full lifecycle on a small dataset |
| [mini-project/README.md](./mini-project/README.md) | **THE CAPSTONE** — full spec, problem-archetype menu, dataset suggestions, deploy options, rubric pointer |
| [mini-project/PROJECT_IDEAS.md](./mini-project/PROJECT_IDEAS.md) | Twenty pre-vetted capstone project ideas with datasets, baselines, and expected difficulty |
| [mini-project/MODEL_CARD_TEMPLATE.md](./mini-project/MODEL_CARD_TEMPLATE.md) | The Mitchell et al. 2019 model-card template adapted to a one-page C5 deliverable |
| [mini-project/DATASET_CARD_TEMPLATE.md](./mini-project/DATASET_CARD_TEMPLATE.md) | The Gebru et al. 2021 datasheet template adapted to a one-page C5 deliverable |
| [mini-project/rubric.md](./mini-project/rubric.md) | The 100-point capstone rubric, broken into eleven graded sections |
| [mini-project/starter.py](./mini-project/starter.py) | A runnable scaffold: dataset load, train/val/test split, a baseline, a candidate model, evaluation, save, predict |

---

## C5 track wrap-up — what you covered, week by week

The C5 syllabus from start to finish, with one line per week and the artifact each week produced. Use this as the table of contents for the portfolio repository and as the talking points for an interview.

1. **Week 1 — NumPy from scratch.** Vectors, matrices, broadcasting, the dot product, the gradient. Artifact: a hand-rolled `linear_regression_gd.py` that fits `y = mx + b` by gradient descent using only NumPy primitives.
2. **Week 2 — Pandas, honestly.** `read_csv`, dtype inference, missing values, `groupby`, the long-vs-wide pivot, the join. Artifact: a notebook that cleans a real-world messy CSV (a 100k-row sample of the NYC 311 service request feed) and reports the seventeen ways the data was lying to you.
3. **Week 3 — Visualization that doesn't lie.** `matplotlib`, the figure-axes API, the four-chart rule, the deceptive-axis anti-patterns. Artifact: a `figure-portfolio/` directory with eight charts that follow the C5 visual-honesty checklist.
4. **Week 4 — The ML workflow and linear models.** Train/val/test, cross-validation, regularization (L1, L2, elastic net), bias-variance, the confusion matrix, the ROC curve, the precision-recall curve. Artifact: a `sklearn`-based pipeline that classifies UCI Adult Income with logistic regression, with a written analysis of the fairness trade-offs at three operating points.
5. **Week 5 — Trees, forests, boosting.** Decision trees, random forests, gradient boosting (`xgboost` and `lightgbm`), feature importance. Artifact: a `lightgbm` model on a tabular dataset that beats the Week 4 logistic-regression baseline, with a SHAP analysis (<https://shap.readthedocs.io/>) of the top features.
6. **Week 6 — Clustering and dimensionality reduction.** K-means, hierarchical clustering, DBSCAN, PCA, t-SNE, UMAP. Artifact: a notebook that clusters the MNIST validation set without labels and recovers the digit structure visible on a 2D t-SNE projection.
7. **Week 7 — Neural networks from scratch.** Forward pass, backprop, the chain rule, ReLU, SGD with momentum, weight initialization (Xavier, He). Artifact: a pure-NumPy 2-layer MLP that classifies MNIST to >97 percent test accuracy without any deep-learning framework.
8. **Week 8 — PyTorch fundamentals.** `Tensor`, `autograd`, `nn.Module`, `nn.Linear`, `DataLoader`, the canonical training loop, `cuda` and `mps`. Artifact: the same 2-layer MLP from Week 7, rewritten in PyTorch in 60 lines including the training loop.
9. **Week 9 — CNNs and transfer learning.** Convolution, pooling, the LeNet/AlexNet/VGG/ResNet lineage, transfer learning with a pretrained `torchvision` ResNet-18. Artifact: a fine-tuned ResNet-18 on a custom 8-class image dataset (the Caltech-101 ten-class subset), with >90 percent validation accuracy on ten training epochs.
10. **Week 10 — Sequence models: RNN, LSTM, GRU.** Recurrence, truncated backpropagation through time, vanishing gradients, the LSTM and GRU gates. Artifact: a char-level LSTM trained on `Pride and Prejudice` that produces English-shaped samples at temperature 0.8.
11. **Week 11 — Attention, transformers, and a mini-GPT.** Scaled dot-product attention, multi-head attention, positional encoding, the transformer block, the decoder-only stack, causal masking. Artifact: a nanoGPT-style decoder-only transformer trained on the same `Pride and Prejudice` text as Week 10, with a side-by-side comparison of LSTM vs transformer convergence.
12. **Week 12 — Capstone.** Problem framing, dataset card, EDA, baseline, model selection, training, evaluation, slice-based fairness audit, model card, packaging, deployment to a free public URL, monitoring. Artifact: a portfolio repo with a deployed URL.

The arc has a shape. Weeks 1-3 are the data-stack foundation. Weeks 4-6 are the classical ML algorithms (linear, trees, unsupervised). Weeks 7-9 are deep learning on tabular and image data. Weeks 10-11 are deep learning on sequence data. Week 12 takes whatever you trained in Weeks 4-11 and ships it to the world. The story you tell at interview is "I built twelve artifacts, eleven of which are technical exercises and one of which I deployed; here is the URL."

---

## Stretch goals

- Read **Mitchell et al. 2019 — "Model Cards for Model Reporting"** (FAT* 2019; <https://arxiv.org/abs/1810.03993>). Eight pages. The paper that established the model card as the standard documentation artifact for an ML model. Section 4 is the template. Read it once before Wednesday; read it again after writing your first draft of `MODEL_CARD.md`.
- Read **Gebru et al. 2021 — "Datasheets for Datasets"** (CACM 2021; <https://arxiv.org/abs/1803.09010>; the PMC version is also free). Twenty-four pages. The dataset analogue of the Mitchell paper, with a longer and more rigorous questionnaire. Section 3 is the seven-section template. Read it once before Monday.
- Read **Buolamwini and Gebru 2018 — "Gender Shades"** (PMLR 2018; <https://proceedings.mlr.press/v81/buolamwini18a.html>). Fifteen pages. The paper that demonstrated commercial face-classification systems had error rates of 0.8 percent on light-skinned men and 34.7 percent on dark-skinned women — a 43x gap. The empirical case for slice-based fairness audits in three pages of tables.
- Read **Breck et al. 2017 — "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"** (Google; <https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>). Eight pages. The 28-question rubric that Google's ML teams use as a production-readiness checklist. The C5 capstone rubric is a 25-point adaptation; this paper is the source.
- Read the **Hugging Face Spaces documentation, end to end** (<https://huggingface.co/docs/hub/spaces-overview>). Thirty minutes. The platform you will deploy to. Pin the "Configuring Spaces" page; read the "Spaces persistent storage" subsection if you plan to log predictions to a file.
- Read the **Streamlit documentation, "Get started" through "Deploying"** (<https://docs.streamlit.io/get-started>). Twenty minutes. The platform you will use if you prefer a dashboard-shaped capstone UI over Gradio's input-output-form UI.
- Read **Sculley et al. 2015 — "Hidden Technical Debt in Machine Learning Systems"** (NeurIPS 2015; <https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>). Nine pages. The canonical critique of "the small fraction of real-world ML systems is composed of the ML code"; the surrounding scaffolding (data pipelines, configuration, monitoring) is everything. Read after deploying your capstone — it will land harder.
- Read **Polyzotis et al. 2018 — "Data Lifecycle Challenges in Production Machine Learning"** (SIGMOD; <https://dl.acm.org/doi/10.1145/3299887.3299891>). Eight pages. The production data-lifecycle paper. Worth reading once you have shipped one model and want to see what the next ten will look like.

---

## What you will *not* do this week

You will not:

- **Pick a research-novel model architecture.** The capstone is about the *lifecycle*, not the architecture. Use the simplest model that beats the baseline. A `LightGBM` on tabular data is a stronger capstone submission than a custom transformer ablation that the grader cannot reproduce. The architectural depth you gained in Weeks 7-11 is the *table stakes*; the lifecycle work is the *signal*.
- **Deploy to a paid tier.** All three deploy targets (Hugging Face Spaces, Streamlit Community Cloud, Fly.io) have free tiers that fit the capstone. If your model needs a paid tier, your model is too big; either prune (Week 14 topic) or pick a different problem.
- **Skip the fairness audit because "my dataset has no protected attributes".** The audit applies even to weather data — you can audit by geography, by season, by sensor type. The Mitchell-template prompt is "for what populations does the model not work" and that prompt has a non-trivial answer for every model. The model card explains the answer.
- **Skip the model card because "it's just a portfolio piece".** The model card is the portfolio piece. The trained model is the artifact; the card is the explanation. Hiring managers read the card; the model is the supporting evidence.
- **Use a paid API in the inference layer.** No OpenAI, no Anthropic, no Cohere, no Gemini API calls in `predict.py`. The capstone is *your* model, hosted on *your* deploy, end-to-end. We respect the paid APIs and will use them in industry; this week is about owning the stack.
- **Roll your own monitoring stack.** SQLite plus `scipy.stats.ks_2samp` is enough for the capstone. Prometheus, Grafana, Sentry, Arize, WhyLabs, Fiddler, Truera, and the rest of the production-monitoring landscape are mentioned in Lecture 3 and revisited if you take C5's optional industry-tier follow-up (C5+).
- **Pad the model card.** The Mitchell template is one page in the original paper; the C5 template is one page on purpose. Bullet-point density, not paragraph mass.

That list is deliberate. The point of Week 12 is to *ship one thing well*, not five things half-way. SOTA is not the goal; the lifecycle is.

---

## A note on the EXPERIMENT cards

Lectures continue to use `EXPERIMENT` callouts:

> **EXPERIMENT — short title.** A one-paragraph drill you can run in a Python REPL in five to fifteen minutes. The drill makes a specific claim from the lecture concrete. Run them. Do not skip them.

The Lecture 1 experiment that loads UCI Adult Income and runs `df.isna().sum()` and `df['workclass'].value_counts()` is the experiment that makes "the literal string `'?'` is how this dataset encodes missing values" stop being a footnote. The Lecture 2 experiment that splits the test predictions by `sex` and reports false-positive-rate per slice is the experiment that makes the fairness audit a number, not a feeling. The Lecture 3 experiment that runs `gradio.Interface` locally and visits `http://localhost:7860` is the experiment that closes the gap between "I have a model" and "anyone with a browser can use my model." All three are under fifteen minutes; all three are the difference between "I read about it" and "I have done it."

---

## Up next

There is no Week 13. C5 ends here.

The optional follow-up is **C5+ · Industry Track** (a 12-week sequel that picks up the topics Week 12 deliberately deferred: prompt engineering and the modern foundation-model stack, fine-tuning LLMs, evaluation harnesses for generative models, the production monitoring stack at depth, MLflow vs Weights & Biases, on-call rotations for ML services, and cost engineering). C5+ runs once a year and is not part of the core curriculum.

Your capstone URL goes at the top of your portfolio README. The URL is the deliverable. The repository is the supporting evidence. The model card is the closer. You are now an ML engineer; congratulations.
