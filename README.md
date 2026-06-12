# C5 · Crunch AI / Data Science

> A free, open-source **12-week data-science + classical-ML track** for engineers who already know Python. NumPy through PyTorch, with the experimentation discipline employers actually want. C1 graduate → mid-level ML / data scientist.

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Built in the open](https://img.shields.io/badge/built-in%20the%20open-7C3AED.svg)](https://github.com/CODE-CRUNCH-CLUB)

C5 takes you from "I know Python" to "I can take a real dataset, do honest analysis, build a model, evaluate it without fooling myself, and ship it behind an API." It is not a survey course — every week ends with a deliverable that pushes to a portfolio repo.

---

## Pathway summary

- **Full-time:** 12 weeks · ~36 hrs/week · ~432 hours
- **Working-engineer pace:** 6 months · ~18 hrs/week
- **Evening pace:** 12 months · ~9 hrs/week

See [`SYLLABUS.md`](SYLLABUS.md) for the full week-by-week breakdown across four phases.

---

## What you will be able to do at the end of 12 weeks

- **Read and clean** any reasonable real-world dataset in pandas — including all the edge cases (mixed types, missing values, encoding errors, time zones).
- **Build a complete ML pipeline** with scikit-learn from raw data to a trained model: split, vectorize, fit, evaluate, cross-validate, save.
- **Choose the right model** for a tabular problem: when linear regression suffices, when you need trees, when neural nets pay off, when classical statistics is more honest.
- **Train a small neural network** in PyTorch end-to-end: dataset, dataloader, `nn.Module`, optimizer, training loop, checkpointing, inference.
- **Explain a model's predictions** with SHAP or permutation importance — and know what those numbers do and don't mean.
- **Set up an experiment**: hold-out set, cross-validation, leakage-free preprocessing, fair baselines, statistical significance.
- **Ship a model behind an API** (FastAPI) and call it from a small frontend.
- **Read papers** at the level of mainstream ML conferences without getting lost.

---

## Who this is for

- **C1 graduate** who wants the data-science specialization.
- **Self-taught engineer** comfortable with Python and itching for data work.
- **CS / stats / engineering student** preparing for an ML-adjacent first job.
- **Working engineer** transitioning into a data team.

Not for: pure beginners (do [C1](../C1-Code-Crunch-Convos/) first), researchers wanting publication-grade depth (this is engineering-grade, not academic), or people who specifically want LLMs/agents (those are in the upcoming Crunch Labs **C23 · Crunch Agents** track).

---

## Prerequisites

- **C1 Weeks 1–14** completed, or equivalent.
- Comfortable with: functions, classes, decorators, generators, list/dict comprehensions, exception handling, `pip` / `venv`.
- Some exposure to algebra and calculus — derivatives, vectors, basic matrix math. We re-teach the bits we need, but a complete blank slate makes Week 7 (NN from scratch) tough.
- A computer with ≥8 GB RAM. A GPU is nice for Week 8+ but not required — Google Colab's free tier handles every Week.

---

## What you ship

By the end of the program, your `crunch-ai-portfolio-<yourhandle>` GitHub repo contains:

1. **A vectorized image-processing notebook** (Week 1) — pure NumPy, no PIL.
2. **An EDA report** on a real public dataset (Week 2).
3. **A re-creation + critique** of three published charts (Week 3).
4. **A house-price prediction** project with both linear and tree-based models (Weeks 4–5).
5. **A clustering analysis** with justified k and 2D visualization (Week 6).
6. **A 2-layer NN trained from scratch in pure NumPy** on MNIST ≥ 95% accuracy (Week 7).
7. **A PyTorch version** of the same, with data augmentation, beating the NumPy version (Week 8).
8. **A fine-tuned pre-trained CNN** on a real image task (Week 9).
9. **An A/B test plan** for a real product question (Week 10).
10. **A FastAPI-served model** with experiment tracking and Docker (Week 11).
11. **A capstone project** end-to-end with a fairness/limitations section (Week 12).

That repo is the artifact you hand recruiters.

---

## Tools we use

| Tool | Role |
|------|------|
| **Python 3.11+** | Language |
| **NumPy** | The N-dimensional array; vectorization |
| **pandas** | DataFrames, IO, cleaning |
| **matplotlib · seaborn · plotly** | Visualization |
| **scikit-learn** | Classical ML |
| **PyTorch** | Deep learning |
| **Jupyter / VS Code notebooks** | Exploration |
| **FastAPI** | Model serving |
| **mlflow** | Experiment tracking |
| **Docker** | Packaging |
| **DuckDB / Polars** | Mentioned as faster pandas alternatives in stretch readings |

Everything is **free** and **open-source**. No paid Kaggle Pro, no proprietary AutoML SaaS, no required GPU rental — Google Colab's free tier handles the deep-learning weeks.

---

## Migration from legacy units

The existing `Unit-0` through `Unit-9` Jupyter notebooks remain available for now. Each is being rolled forward into the standardized weekly module layout used by C1, C16, C17, and C2. See the [SYLLABUS.md migration table](SYLLABUS.md#migration-from-the-legacy-units) for the mapping.

Contributors: each new week's PR should follow the [Code Crunch contribution guide](../CONTRIBUTING.md) and match the depth of [C1 Week 1](../C1-Code-Crunch-Convos/curriculum/week-01-python-foundations/).

---

## Next track after C5

- **[C17 · Crunch Pro Python Advanced](../C17-CRUNCH-PRO-PYTHON-ADVANCED/)** — for the runtime depth (async, performance, PyTorch internals).
- **[C15 · Crunch DevOps](../C15-CRUNCH-DEVOPS/)** — deploy your models in production.
- **C23 · Crunch Agents** (Tier 2 Labs) — for LLM / agentic systems beyond classical ML.

---

## License

GPL-3.0. See [LICENSE](LICENSE).

---

*C5 is part of the Code Crunch open-source curriculum.* [Master catalog ↗](../MASTER-CURRICULUM.md) · [Brand family ↗](../../assets/brand/BRAND-FAMILY.md)


---

<!-- CCWW:AUTO-INDEX:START — generated by scripts/restructure_course_repos.py; edit ABOVE this marker -->

## Course at a glance

| Section | Count |
| --- | --- |
| Curriculum entries | 13 |
| Projects | 1 |
| Past sessions | 10 |

## Curriculum

- [SYLLABUS](curriculum/SYLLABUS.md)
- [week 01 numpy from scratch](curriculum/week-01-numpy-from-scratch/README.md)
- [week 02 pandas honestly](curriculum/week-02-pandas-honestly/README.md)
- [week 03 visualization that doesnt lie](curriculum/week-03-visualization-that-doesnt-lie/README.md)
- [week 04 ml workflow and linear models](curriculum/week-04-ml-workflow-and-linear-models/README.md)
- [week 05 trees forests boosting](curriculum/week-05-trees-forests-boosting/README.md)
- [week 06 clustering and dimensionality reduction](curriculum/week-06-clustering-and-dimensionality-reduction/README.md)
- [week 07 neural networks from scratch](curriculum/week-07-neural-networks-from-scratch/README.md)
- [week 08 pytorch fundamentals](curriculum/week-08-pytorch-fundamentals/README.md)
- [week 09 cnns and transfer learning](curriculum/week-09-cnns-and-transfer-learning/README.md)
- [week 10 sequence models rnn lstm gru](curriculum/week-10-sequence-models-rnn-lstm-gru/README.md)
- [week 11 attention transformers and mini gpt](curriculum/week-11-attention-transformers-and-mini-gpt/README.md)
- [week 12 capstone end to end ml with deploy](curriculum/week-12-capstone-end-to-end-ml-with-deploy/README.md)

## In this course

- **Community** — [community/](community/)
- **Curriculum** — [curriculum/](curriculum/)
- **Projects** — [projects/](projects/)
- **Resources** — [resources/](resources/)
- **Past sessions** — [past-sessions/](past-sessions/)

<!-- CCWW:AUTO-INDEX:END -->
