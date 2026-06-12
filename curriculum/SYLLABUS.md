# C5 · Crunch AI / Data Science — Full Syllabus

**12 weeks · ~432 hours full-time · ~36 hrs/week · C1 graduate → mid-level ML / data scientist**

This is the table of contents for the C5 track. The existing `Unit-0` through `Unit-9` notebooks remain as legacy material; this syllabus describes the **full 12-week structure** the track is moving to, with the same standard week layout as every other Code Crunch course (lectures, exercises, challenges, quiz, homework, mini-project).

---

## Who this is for

- You've completed **C1 · Code Crunch Convos** (or have equivalent Python proficiency through Week 14).
- You want to be employable as a data analyst, data scientist, or junior ML engineer at the end.
- You're willing to do real math. We don't dodge it; we just don't make it the point.

If you don't yet know what a list comprehension is, finish C1 first.

---

## What you will be able to do at the end of 12 weeks

- **Read and clean** any reasonable real-world dataset in pandas without panicking — CSVs, JSON, Parquet, Excel, SQL — including all the edge cases (mixed types, missing values, encoding errors, time zones).
- **Build a complete ML pipeline** with scikit-learn from raw data to a trained model: split, vectorize, fit, evaluate, cross-validate, save.
- **Choose the right model** for a tabular problem: when linear regression suffices, when you need trees, when neural nets pay off, when classical statistics is more honest.
- **Train a small neural network** in PyTorch end-to-end: dataset, dataloader, `nn.Module`, optimizer, training loop, checkpointing, inference.
- **Explain a model's predictions** with SHAP or permutation importance, and know what those numbers do and don't mean.
- **Set up an experiment**: hold-out set, cross-validation, leakage-free preprocessing, fair baselines, statistical significance.
- **Ship a model behind an API** (FastAPI) and call it from a small frontend.
- **Read papers** at the level of mainstream ML conferences (NeurIPS, ICML) — at least the abstract + figures — without getting lost.

---

## Program at a glance

| Phase | Weeks | Outcome |
|-------|-------|---------|
| **Phase 1 — Foundations of Data** | 01 – 03 | NumPy + pandas + visualization fluency |
| **Phase 2 — Classical Machine Learning** | 04 – 06 | Regression, classification, clustering with scikit-learn |
| **Phase 3 — Deep Learning** | 07 – 09 | PyTorch from scratch; CNN, RNN/transformer basics |
| **Phase 4 — ML in the Real World** | 10 – 12 | MLOps, ethics, capstone |

---

## How the weekly load adds up

| Component | hrs/wk |
|-----------|------:|
| Lectures / readings | 6 |
| Hands-on exercises | 8 |
| Coding challenges | 4 |
| Quiz + readings | 3 |
| Homework problems | 6 |
| Mini-project | 7 |
| Self-study & review | 2 |
| **Total** | **36** |

---

## Weekly breakdown

### Phase 1 — Foundations of Data

#### Week 1 — NumPy from Scratch

The N-dimensional array. dtype matters. View vs. copy. Broadcasting until it's boring. Vectorization vs. for-loops. The first lecture where "you don't actually need pandas yet" earns its keep.

- **Mini-project:** A vectorized image filter (grayscale, blur, edge detection) on a real image — no PIL, no OpenCV. Pure NumPy.

#### Week 2 — pandas, Honestly

Series, DataFrame, MultiIndex, the `merge` family, group-by aggregation, time series. Reading and writing CSV, JSON, Parquet. Handling missing data. The `apply` foot-gun.

- **Mini-project:** Clean and explore a real, messy public dataset (we provide one — a city open-data feed). Produce a 2-page EDA report.

#### Week 3 — Visualization that Doesn't Lie

matplotlib fundamentals. When to use seaborn. When to use plotly. Color, scale, axis tricks, perceptual encoding. Reading and *making* charts that an executive can act on.

- **Mini-project:** Re-create three published charts from real news sources (e.g., the New York Times' *Upshot* or the *Financial Times*), then critique the original — what works, what's misleading.

---

### Phase 2 — Classical Machine Learning

#### Week 4 — The ML Workflow & Linear Models

Train/val/test split, the bias-variance tradeoff, the no-free-lunch theorem. Linear regression and logistic regression done properly: assumptions, residual analysis, regularization (L1, L2), interpretation.

- **Mini-project:** Predict house prices on a real dataset. Beat a strong baseline. Document your features. Justify your choice.

#### Week 5 — Trees, Forests, and Gradient Boosting

Decision trees from scratch (Gini and entropy). Random forests. Gradient boosting (XGBoost, LightGBM, scikit-learn's HistGradientBoosting). Why these dominate Kaggle.

- **Mini-project:** Same dataset as Week 4. Beat your linear-model score by ≥10% relative error. Explain why the trees won — or, surprisingly, why they didn't.

#### Week 6 — Clustering, Dimensionality Reduction, Unsupervised Methods

K-Means, hierarchical, DBSCAN, Gaussian mixtures. PCA, t-SNE, UMAP. When clustering is the answer; more often when it's not.

- **Mini-project:** Take a real-world set of customer or product records. Find meaningful clusters. Justify the number of clusters. Visualize in 2D.

---

### Phase 3 — Deep Learning

#### Week 7 — Neural Networks from Scratch

A 2-layer neural network in NumPy. Forward pass, loss, backprop by hand. Why this exercise still matters even when PyTorch does it for you.

- **Mini-project:** Implement and train a 2-layer NN on MNIST using only NumPy. Reach ≥95% test accuracy.

#### Week 8 — PyTorch and the Modern Training Loop

Tensors, autograd, `nn.Module`, `Dataset`, `DataLoader`, optimizers. The training loop you'll write 100 more times. `torch.compile` in 2026.

- **Mini-project:** Re-do Week 7 in PyTorch. Add data augmentation, better init, dropout, and a learning-rate schedule. Beat your pure-NumPy result.

#### Week 9 — CNNs, RNNs, and Just Enough Transformers

Convolutional neural networks for vision. RNNs (LSTM, GRU) briefly. Transformer architecture in plain English — what attention computes, why it matters. We don't pre-train one; we use a small pre-trained one.

- **Mini-project:** Fine-tune a small pre-trained CNN on a real image dataset (your photo collection or a public Kaggle one). Reach a solid test accuracy. Document the training run.

---

### Phase 4 — ML in the Real World

#### Week 10 — Experimentation & Statistics for ML

P-values, confidence intervals, A/B testing fundamentals. Multiple-comparison corrections. Why "deep learning will solve it" is rarely the right first answer.

- **Mini-project:** Design an end-to-end A/B test for a real product question (we provide a fake company; you write the experiment plan, the analysis script, and the report).

#### Week 11 — MLOps and Serving Models

`mlflow` for experiment tracking. Model versioning. Serving with FastAPI. Building a small data pipeline. A first taste of `dvc`.

- **Mini-project:** Take your Week 8 PyTorch model. Wrap it in a FastAPI service. Add experiment tracking. Containerize with Docker. Expose a `/predict` endpoint.

#### Week 12 — Ethics, Bias, and the Capstone

Real bias case studies (COMPAS, hiring algorithms, facial recognition). Fairness metrics. Reading audits. Then: capstone.

- **Capstone:** A full ML project end-to-end on a dataset you choose: from EDA through a deployed model behind an API, with a 5-page write-up that includes a fairness/limitations section. This is what you point employers at.

---

## Skills progression chart

```text
W1  ─ NumPy
W2  │ pandas
W3  ─ visualization
W4  ─ linear / logistic regression
W5  │ trees, forests, boosting
W6  ─ clustering, dimensionality reduction
W7  ─ NN from scratch
W8  │ PyTorch
W9  ─ CNN / RNN / transformer basics
W10 ─ experimentation & stats
W11 │ MLOps
W12 ─ ethics + CAPSTONE
```

---

## What you won't learn (but should)

- **NLP at scale, LLM training, RLHF** — these are full sub-tracks. After C5, the next step is targeted reading + the [HuggingFace course](https://huggingface.co/learn) (free).
- **Computer vision at scale** — same as above; we get the basics in Week 9.
- **Reinforcement learning** — out of scope; we point at OpenAI's [Spinning Up](https://spinningup.openai.com/) (free) for next steps.
- **Distributed training (Spark / Ray)** — see [C15 · Crunch DevOps](../C15-CRUNCH-DEVOPS/) and the stretch reading in Week 11.
- **MLOps at scale (feature stores, model registries, online serving systems)** — touched on but not deep.

For the runtime parts (PyTorch internals, performance, async data loading, custom CUDA kernels): see **[C17 · Crunch Pro Python Advanced](../C17-CRUNCH-PRO-PYTHON-ADVANCED/)**.

---

## Migration from the legacy units

The existing `Unit-0` through `Unit-9` notebooks remain available for now. As the new weekly structure rolls out, units will be replaced with full weekly modules following the standard Code Crunch layout (README, resources, lecture-notes, exercises, challenges, quiz, homework, mini-project).

- `Unit-0` → Week 1 prerequisites + setup material
- `Unit-1` → Week 1 (NumPy)
- `Unit-2` → Week 2 (pandas, data mining basics)
- `Unit-3` → Week 4 (modeling intro)
- `Unit-4` → Week 4 (regression)
- `Unit-5` → Week 6 (clustering)

Contributors welcome — see the master `CONTRIBUTING.md` for how to PR a new week.

---

## License

GPL-3.0.
