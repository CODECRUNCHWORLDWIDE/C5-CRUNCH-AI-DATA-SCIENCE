# Week 12 — Resources

Every resource on this page is **free** and **publicly accessible**. No paywalled books, no proprietary PDFs, no signup-required courses. If a link breaks, open an issue.

## Required reading (work it into your week)

- **Mitchell, Margaret; Wu, Simone; Zaldivar, Andrew; Barnes, Parker; Vasserman, Lucy; Hutchinson, Ben; Spitzer, Elena; Raji, Inioluwa Deborah; Gebru, Timnit — "Model Cards for Model Reporting"** (FAT* 2019; the paper that established model cards as the standard documentation artifact for an ML model):
  <https://arxiv.org/abs/1810.03993>
  Read **Sunday evening** before Monday's lecture. Eight pages. The single most important paper for the lifecycle half of ML. Section 4 is the template; section 4.1 is the section-by-section explanation. Pin the URL; reference it again on Wednesday when you draft your `MODEL_CARD.md`. The paper is short and the template is concrete; there is no excuse for not reading it.
- **Gebru, Timnit; Morgenstern, Jamie; Vecchione, Briana; Vaughan, Jennifer Wortman; Wallach, Hanna; Daumé III, Hal; Crawford, Kate — "Datasheets for Datasets"** (CACM 2021; the dataset analogue of the Mitchell paper):
  <https://arxiv.org/abs/1803.09010>
  Read **Sunday evening** before Monday's lecture. Twenty-four pages. Section 3 is the seven-section questionnaire (motivation, composition, collection process, preprocessing, uses, distribution, maintenance). The C5 `DATASET_CARD_TEMPLATE.md` is a one-page adaptation. Read once before writing your dataset card; refer back as you write.
- **Hugging Face — "Spaces overview"** (the deploy platform documentation):
  <https://huggingface.co/docs/hub/spaces-overview>
  Read **Friday morning** before Lecture 3 if you have not already. Thirty minutes. The C5 default deploy target. Pin the "Configuring Spaces" subpage and the "Spaces persistent storage" subpage. The free tier runs a 2-vCPU, 16 GB RAM CPU machine indefinitely; that fits any sklearn or PyTorch CPU-inference model under 1 GB.
- **Gradio — "Quickstart"** (the simplest "wrap a Python function in a web UI" library):
  <https://www.gradio.app/guides/quickstart>
  Read **Friday morning** before Lecture 3. Twenty minutes. The `gr.Interface(fn=predict, inputs=..., outputs=...)` line in `app.py` is the entire UI; Gradio is the C5 default for capstone deployment because the API is two function calls.
- **Streamlit — "Get started"** (the Python-only dashboard framework):
  <https://docs.streamlit.io/get-started>
  Read **Friday morning** before Lecture 3 if you plan to deploy to Streamlit Cloud. Twenty minutes. The `st.title`, `st.write`, `st.slider`, `st.button` API; the script reruns top to bottom on every interaction; the `st.session_state` dictionary for in-memory state. Slick for dashboards, less suited than Gradio for "one prediction at a time" capstones.
- **FastAPI — "First Steps"** (the production-grade Python web framework):
  <https://fastapi.tiangolo.com/tutorial/first-steps/>
  Read **Friday morning** before Lecture 3 if you plan to deploy to Fly.io. Twenty minutes. The `@app.post("/predict")` decorator wraps your `predict()` function in an HTTP endpoint with automatic Swagger documentation at `/docs`. The C5 third deploy option.
- **Fly.io — "Launch a new app"** (the platform-as-a-service with a free allowance):
  <https://fly.io/docs/launch/>
  Read **Friday morning** before Lecture 3 if you plan to deploy to Fly. Twenty minutes. `fly launch` reads your `Dockerfile`, asks four questions, and deploys. The free allowance covers a `shared-cpu-1x` machine with 256 MB RAM, which fits a sklearn model server but not a transformer.
- **DVC — "Get started"** (the data-versioning tool):
  <https://dvc.org/doc/start>
  Read **Wednesday morning** before Lecture 2. Forty minutes. `dvc init`, `dvc add data/raw.csv`, `dvc push` against a free remote. The `.dvc` file is committed to git; the bytes are not. The C5 default for dataset versioning.

## The papers (free PDFs)

- **Mitchell et al. 2019 — "Model Cards for Model Reporting"** (FAT* 2019):
  <https://arxiv.org/abs/1810.03993>
  The model-card paper. Required reading. The Mitchell template is the industry-standard documentation format.
- **Gebru et al. 2021 — "Datasheets for Datasets"** (CACM 2021; also free at <https://dl.acm.org/doi/10.1145/3458723>):
  <https://arxiv.org/abs/1803.09010>
  The dataset-card paper. Required reading.
- **Buolamwini, Joy; Gebru, Timnit — "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification"** (PMLR 2018; the paper that empirically grounded slice-based fairness audits):
  <https://proceedings.mlr.press/v81/buolamwini18a.html>
  Fifteen pages. The empirical case for fairness audits in three pages of tables: commercial face-classification systems had 0.8 percent error on light-skinned men and 34.7 percent error on dark-skinned women — a 43x gap that no aggregate accuracy number revealed. Read once before drafting your fairness audit.
- **Breck, Eric; Cai, Shanqing; Nielsen, Eric; Salib, Michael; Sculley, D. — "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"** (Google IEEE Big Data 2017; the production-readiness checklist):
  <https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>
  Eight pages. The 28-question rubric; the C5 capstone rubric is a 25-point adaptation. Read once before Friday.
- **Sculley, D.; Holt, Gary; Golovin, Daniel; Davydov, Eugene; Phillips, Todd; Ebner, Dietmar; Chaudhary, Vinay; Young, Michael; Crespo, Jean-François; Dennison, Dan — "Hidden Technical Debt in Machine Learning Systems"** (NeurIPS 2015; the canonical critique of "ML is the small fraction of the system"):
  <https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
  Nine pages. The diagram that shows ML code as a small box surrounded by twelve larger boxes (data collection, feature extraction, configuration, machine resource management, monitoring, serving infrastructure, ...) is the most-cited illustration in MLOps. Read after deploying your capstone — it will land harder.
- **Polyzotis, Neoklis; Roy, Sudip; Whang, Steven Euijong; Zinkevich, Martin — "Data Lifecycle Challenges in Production Machine Learning: A Survey"** (SIGMOD 2018; free preprint <https://research.google/pubs/data-lifecycle-challenges-in-production-machine-learning-a-survey/>):
  Eight pages. The production data-lifecycle paper from Google's perspective. Worth reading after the capstone — gives you the vocabulary for what the next ten projects will look like.
- **Pushkarna, Mahima; Zaldivar, Andrew; Kjartansson, Oddur — "Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI"** (Google FAccT 2022; an evolution of the Gebru template):
  <https://arxiv.org/abs/2204.01075>
  Sixteen pages. Google's adaptation of the Gebru datasheet, with a focus on cards that are read by non-ML stakeholders. Optional but excellent.
- **Bender, Emily M.; Friedman, Batya — "Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science"** (TACL 2018; the NLP-specific equivalent of the Gebru template):
  <https://aclanthology.org/Q18-1041/>
  Sixteen pages. If your capstone is an NLP project, read this in addition to Gebru et al. The data-statement template is more specific about language, demographic, and curatorial decisions.

## The deploy-platform official docs

### Hugging Face Spaces (the C5 default)

- **Spaces overview** (the platform):
  <https://huggingface.co/docs/hub/spaces-overview>
- **Spaces SDKs** (Gradio, Streamlit, Docker, Static):
  <https://huggingface.co/docs/hub/spaces-sdks>
- **Spaces config — `README.md` YAML frontmatter** (the metadata block that picks the SDK and the hardware):
  <https://huggingface.co/docs/hub/spaces-config-reference>
- **Spaces persistent storage** (read this if you log predictions to a file):
  <https://huggingface.co/docs/hub/spaces-storage>
- **Spaces secrets and variables** (read this if your inference layer needs an API key):
  <https://huggingface.co/docs/hub/spaces-overview#managing-secrets>
- **Spaces free-tier hardware** (current as of 2026; check the page for updates):
  <https://huggingface.co/docs/hub/spaces-gpus>

### Gradio (the C5 default UI library)

- **Gradio quickstart**:
  <https://www.gradio.app/guides/quickstart>
- **`gr.Interface` reference** (the high-level "wrap a function" API):
  <https://www.gradio.app/docs/interface>
- **`gr.Blocks` reference** (the low-level "build a custom layout" API; rarely needed for the capstone):
  <https://www.gradio.app/docs/blocks>
- **Sharing your app** (the temporary `share=True` URL and the permanent Hugging Face Spaces deploy):
  <https://www.gradio.app/guides/sharing-your-app>

### Streamlit

- **Streamlit docs root**:
  <https://docs.streamlit.io/>
- **Streamlit get started**:
  <https://docs.streamlit.io/get-started>
- **Streamlit Community Cloud** (the free hosting):
  <https://docs.streamlit.io/streamlit-community-cloud>
- **`st.cache_data` and `st.cache_resource`** (read this; the C5 starter caches the model load):
  <https://docs.streamlit.io/develop/concepts/architecture/caching>

### FastAPI + Fly.io

- **FastAPI tutorial root**:
  <https://fastapi.tiangolo.com/tutorial/>
- **FastAPI `Depends` and dependency injection** (the C5 pattern for loading the model once at startup):
  <https://fastapi.tiangolo.com/tutorial/dependencies/>
- **FastAPI Pydantic models for request validation** (the C5 pattern for typed `/predict` request bodies):
  <https://fastapi.tiangolo.com/tutorial/body/>
- **Fly.io overview**:
  <https://fly.io/docs/about/introduction/>
- **Fly.io launch a new app**:
  <https://fly.io/docs/launch/>
- **Fly.io pricing and free allowance**:
  <https://fly.io/docs/about/pricing/>
- **Fly.io machines (the underlying VM)**:
  <https://fly.io/docs/machines/>

## Versioning datasets and models

- **DVC docs root**:
  <https://dvc.org/doc>
- **DVC get started**:
  <https://dvc.org/doc/start>
- **DVC remote storage** (S3, GCS, Azure, SSH, WebDAV, and free options):
  <https://dvc.org/doc/user-guide/data-management/remote-storage>
- **DVC and CI/CD** (running training pipelines from GitHub Actions; the next step after the capstone):
  <https://dvc.org/doc/use-cases/ci-cd-for-machine-learning>
- **LakeFS docs** (the alternative "git for data"; the LakeFS Cloud free tier covers small projects):
  <https://docs.lakefs.io/>
- **LakeFS Cloud free tier pricing**:
  <https://lakefs.io/pricing/>
- **Hugging Face Hub — model repositories** (the C5 alternative for versioning trained-model artifacts):
  <https://huggingface.co/docs/hub/models>
- **Hugging Face Hub — dataset repositories** (the C5 alternative for versioning the raw dataset):
  <https://huggingface.co/docs/hub/datasets>
- **`git-lfs`** (the underlying technology under Hugging Face Hub for files larger than 10 MB):
  <https://git-lfs.com/>

## Fairness, bias, and slice-based evaluation

- **Fairlearn** (the Microsoft open-source fairness toolkit; the C5 reference):
  <https://fairlearn.org/>
- **Fairlearn API reference** (the `MetricFrame` class is what you want for the slice-based audit):
  <https://fairlearn.org/v0.10/api_reference/index.html>
- **Aequitas** (the alternative fairness toolkit; University of Chicago Data Science for Social Good):
  <http://aequitas.dssg.io/>
- **Google's "AI Explanations Whitepaper"** (the production-grade explanation framework; vendor-specific but well-written):
  <https://cloud.google.com/vertex-ai/docs/explainable-ai/overview>
- **Buolamwini and Gebru 2018 — "Gender Shades"**:
  <https://proceedings.mlr.press/v81/buolamwini18a.html>
- **Barocas, Solon; Hardt, Moritz; Narayanan, Arvind — "Fairness and Machine Learning"** (the free online textbook):
  <https://fairmlbook.org/>
  About 250 pages, fully online. The most rigorous treatment of the topic. Optional but the canonical reference.
- **AIF360** (IBM's fairness toolkit):
  <https://aif360.res.ibm.com/>
- **What-If Tool** (Google's interactive fairness-and-counterfactual visualizer):
  <https://pair-code.github.io/what-if-tool/>

## Drift detection and monitoring

- **Alibi Detect** (Seldon's open-source drift and outlier detection library; the C5 reference):
  <https://github.com/SeldonIO/alibi-detect>
- **Alibi Detect docs**:
  <https://docs.seldon.io/projects/alibi-detect/en/stable/>
- **Evidently AI** (an open-source library for data drift, model drift, and data quality reports):
  <https://github.com/evidentlyai/evidently>
- **`scipy.stats.ks_2samp`** (the two-sample Kolmogorov-Smirnov test the C5 monitor uses by default):
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html>
- **`scipy.stats.wasserstein_distance`** (the alternative drift metric; less commonly used but more interpretable for continuous features):
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>
- **WhyLabs whylogs** (a logging library that produces statistical profiles of every prediction):
  <https://github.com/whylabs/whylogs>
- **Arize Phoenix** (the open-source observability platform for ML; a heavier-weight option than the SQLite-based C5 monitor):
  <https://github.com/Arize-ai/phoenix>

## Datasets you might use for the capstone

- **Hugging Face Datasets Hub** (the modern default; thousands of curated datasets with license metadata):
  <https://huggingface.co/datasets>
- **OpenML** (tabular focus; great metadata; the right place to find a "classic" benchmark):
  <https://www.openml.org/>
- **Kaggle Datasets** (large variety, variable quality, free with a Kaggle account):
  <https://www.kaggle.com/datasets>
- **UCI Machine Learning Repository** (the classic small-tabular-dataset archive):
  <https://archive.ics.uci.edu/>
- **data.gov** (US government open data):
  <https://www.data.gov/>
- **New York City Open Data**:
  <https://opendata.cityofnewyork.us/>
- **Chicago Data Portal**:
  <https://data.cityofchicago.org/>
- **City of Austin Open Data Portal**:
  <https://data.austintexas.gov/>
- **California Open Data Portal**:
  <https://data.ca.gov/>
- **Project Gutenberg** (public-domain text; same source as Weeks 10-11):
  <https://www.gutenberg.org/>
- **Wikipedia dumps** (full-corpus text):
  <https://dumps.wikimedia.org/>
- **Common Crawl** (web-scale text; out of scope for a one-week capstone but listed for completeness):
  <https://commoncrawl.org/>
- **Google Dataset Search**:
  <https://datasetsearch.research.google.com/>
- **AwesomeData / awesome-public-datasets** (a curated GitHub list):
  <https://github.com/awesomedata/awesome-public-datasets>

## Tabular ML libraries (W4-W5 references restated)

- **scikit-learn** (the C5 default):
  <https://scikit-learn.org/stable/>
- **LightGBM** (Microsoft's gradient-boosted-tree library; the C5 default for tabular when accuracy matters):
  <https://lightgbm.readthedocs.io/>
- **XGBoost** (the original gradient-boosted-tree library):
  <https://xgboost.readthedocs.io/>
- **CatBoost** (Yandex's gradient-boosted-tree library; handles categorical features without one-hot encoding):
  <https://catboost.ai/>

## Hyperparameter search

- **`sklearn.model_selection.GridSearchCV`** (the C5 default for small search spaces):
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>
- **`sklearn.model_selection.RandomizedSearchCV`** (the default for large search spaces):
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html>
- **Optuna** (the Bayesian optimization library; out of scope for the capstone but excellent):
  <https://optuna.org/>

## Companion blog posts and tutorials (free)

- **Hugging Face — "Build your first ML demo with Gradio"** (the canonical step-by-step):
  <https://huggingface.co/blog/gradio>
- **Hugging Face — "Hosting your models on the Hub"**:
  <https://huggingface.co/docs/hub/models-uploading>
- **Karpathy, Andrej — "A Recipe for Training Neural Networks"** (the timeless 2019 essay; required reading for anyone training a model in 2026):
  <http://karpathy.github.io/2019/04/25/recipe/>
- **Google — "Rules of Machine Learning: Best Practices for ML Engineering"** (Martin Zinkevich's 43-rule guide; the single most-recommended internal-Google ML doc):
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
- **Made With ML** (a free MLOps-focused course; longer than the C5 capstone covers but excellent):
  <https://madewithml.com/>
- **Full Stack Deep Learning** (an annual online course; recordings free):
  <https://fullstackdeeplearning.com/>
- **Sebastian Raschka — "Machine Learning Q and AI"** (free online; covers many lifecycle topics):
  <https://sebastianraschka.com/books/ml-q-and-ai/>

## Reproducibility

- **`requirements.txt`** versus `pyproject.toml` versus `environment.yml`:
  <https://packaging.python.org/en/latest/discussions/install-requirements-linux/>
- **`pip-tools`** (the tool that pins transitive dependencies):
  <https://github.com/jazzband/pip-tools>
- **`pipreqs`** (the tool that scans your code and generates a minimal `requirements.txt`):
  <https://github.com/bndr/pipreqs>
- **`pip-audit`** (the tool that checks your `requirements.txt` against the Python Package Index vulnerability database):
  <https://pypi.org/project/pip-audit/>
- **Docker — "Get started"** (the C5 reference for containerized deploys):
  <https://docs.docker.com/get-started/>
- **`mlflow.set_tracking_uri`** (MLflow; an alternative to DVC for experiment tracking; free local mode):
  <https://mlflow.org/docs/latest/index.html>
- **Weights & Biases** (the SaaS experiment tracker; free tier for individuals):
  <https://wandb.ai/>

## Containers and the cloud-free-tier landscape

- **Docker — "Dockerfile reference"**:
  <https://docs.docker.com/engine/reference/builder/>
- **Docker Hub free tier** (free public image hosting):
  <https://hub.docker.com/>
- **GitHub Container Registry** (free for public images, integrated with `gh` and GitHub Actions):
  <https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry>
- **Render free tier** (an alternative to Fly.io; spins down after fifteen minutes of inactivity):
  <https://docs.render.com/free>
- **Railway free tier** (an alternative to Fly.io; usage-based after the trial credit):
  <https://docs.railway.app/reference/pricing/free-trial>
- **Cloudflare R2 free allowance** (the free S3-compatible bucket the C5 DVC remote uses):
  <https://developers.cloudflare.com/r2/pricing/>
- **Backblaze B2 free tier** (an alternative S3-compatible bucket):
  <https://www.backblaze.com/cloud-storage/pricing>
- **Oracle Always Free** (the most generous free-tier compute on the planet, if you like Oracle Cloud):
  <https://www.oracle.com/cloud/free/>

## The portfolio README

- **GitHub — "Customizing your profile README"**:
  <https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/managing-your-profile-readme>
- **A pinned-repository guide** (how to pin your capstone repo to your GitHub profile):
  <https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/pinning-items-to-your-profile>
- **Shields.io** (the badge generator for your README — build status, license, Python version):
  <https://shields.io/>

## The "what comes next" reading list

- **MLOps Community Slack** (the free community of practice; sign up at):
  <https://mlops.community/>
- **The Batch by Andrew Ng** (weekly newsletter; free):
  <https://www.deeplearning.ai/the-batch/>
- **Sebastian Raschka — Ahead of AI** (weekly newsletter; free tier):
  <https://magazine.sebastianraschka.com/>
- **The Gradient** (a long-form ML-focused publication; free):
  <https://thegradient.pub/>
- **Distill** (the now-archived but still-readable visual-explanations journal):
  <https://distill.pub/>

## Quick API reference card

```text
# Save a model.
import joblib
joblib.dump(model, "model.joblib")

# Load a model.
model = joblib.load("model.joblib")

# Save a PyTorch model.
import torch
torch.save(model.state_dict(), "model.pt")

# Load a PyTorch model.
model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# A Gradio interface.
import gradio as gr
def predict(text: str) -> str:
    return model.predict([text])[0]
demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()

# A FastAPI endpoint.
from fastapi import FastAPI
from pydantic import BaseModel
class Req(BaseModel):
    text: str
app = FastAPI()
@app.post("/predict")
def predict_endpoint(req: Req) -> dict[str, str]:
    return {"label": model.predict([req.text])[0]}

# A Streamlit dashboard.
import streamlit as st
st.title("My Capstone")
text = st.text_input("Input")
if st.button("Predict"):
    st.write(model.predict([text])[0])

# A DVC dataset add.
# $ dvc init
# $ dvc add data/raw.csv
# $ git add data/raw.csv.dvc data/.gitignore
# $ git commit -m "track raw.csv with dvc"
# $ dvc remote add -d storage s3://my-bucket/dvc-store
# $ dvc push

# A SHA256 of a file (for model card versioning).
import hashlib
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

Pin this page. You will return to it.
