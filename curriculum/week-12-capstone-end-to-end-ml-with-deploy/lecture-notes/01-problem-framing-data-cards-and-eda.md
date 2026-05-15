# Lecture 1 — Problem framing, dataset cards, and exploratory data analysis

> *Reading time: about 75 minutes. Companion experiments at the end take another 30. Required reading for this lecture: Gebru et al. 2021 ("Datasheets for Datasets", <https://arxiv.org/abs/1803.09010>), at least sections 1 and 3.*

---

## 1. The setup the previous eleven weeks did not cover

Every prior week of C5 handed you a problem and a dataset. Week 4 said "classify UCI Adult Income". Week 9 said "fine-tune a ResNet-18 on Caltech-101". Week 11 said "train a char-LM on `Pride and Prejudice`". The problem framing was pre-cooked; the dataset card was implicit; the success metric was named in the assignment. The capstone removes the scaffolding. You walk into Week 12 with no problem, no dataset, no metric, and seven days to ship a deployed model. This lecture is the first three of those days.

The decision tree that turns the blank-page state into a tractable one-week project has four nodes:

1. **What kind of problem is this?** Classification (binary, multi-class), regression, sequence-to-label, sequence-to-sequence, clustering, recommendation, time-series forecasting. The architectural family of the model and the metric family follow from the answer.
2. **What is the prediction target, in concrete terms?** "Churn" is not a target. "Will this customer cancel their subscription within the next 30 days, measured from the snapshot date" is a target. "House price" is not a target. "Median sale price in US dollars for a single-family home, log-transformed, predicted at signing-of-contract time" is a target.
3. **What is the metric, in units?** F1 at the macro-average, with a threshold chosen on the validation set, reported on the held-out test set. RMSE in dollars, on the original (non-log) scale, reported as a percentage of the median target. BLEU-4 with the SacreBLEU tokenizer, computed against a held-out 1k-sample test split.
4. **What is the baseline, as a number?** `DummyClassifier(strategy="most_frequent")` gives accuracy `p_majority`; an F1 calculation on that prediction gives the macro-F1 baseline. The candidate model has to beat that number, in expectation, on the held-out test set, by an effect size large enough to be worth deploying.

Each of those four decisions is recorded in the capstone's `PROBLEM.md` file. The grader reads `PROBLEM.md` first; if your problem framing is mush, every downstream artifact is mush.

### 1.1 An aside on stakeholder translation

In industry, a one-paragraph "we want to know who is likely to churn" arrives by Slack DM. The first task is *not* to open `pandas`; it is to translate that paragraph into the four bullet points above and reply to the stakeholder with "here is what I think you are asking; please confirm before I spend two weeks on it." The reply-and-confirm step is the most underrated workflow in industrial ML. You do not have a stakeholder this week — the C5 grader is a poor stand-in — but you should write the problem statement *as if* you did, and the rubric grades the document on the clarity it would have for a hypothetical product manager.

The template the C5 capstone uses:

```text
# Problem statement -- <your capstone name>

## What kind of problem is this?
<binary classification | multi-class classification | regression | sequence-to-label | sequence-to-sequence | recommendation | time-series forecasting | clustering>

## Prediction target
<one sentence: what we are predicting, on what scale, at what time>

## Success metric
<one sentence: which metric, on which split, computed how>

## Baseline
<one sentence: what dumb thing we have to beat, with a number>

## Acceptance threshold
<one sentence: what improvement over baseline justifies a deploy>

## Out of scope (deliberately)
<two-to-five bullets: things this capstone is not trying to do>
```

The "out of scope" section is load-bearing. Naming what you are *not* doing is the move that prevents the capstone from spiralling into three projects at once. "I am not predicting churn for B2B customers; I am not building a real-time scoring API; I am not handling streaming data" is the kind of declaration that makes the capstone tractable.

> **EXPERIMENT 1.1 — write your problem statement in twenty minutes.** Pick the dataset you are leaning toward (Adult Income, NYC 311 service requests, a Hugging Face dataset, etc.) and write the seven-section template above. Then close the file, leave it for an hour, come back, and read it as if you had not written it. The success metric, in particular, almost always reads as fuzzy on the first pass. Tighten it until it reads as a single number computable from a single function call.

---

## 2. The six problem archetypes the capstone covers

The capstone rubric scores any of six problem archetypes equally; pick the one that fits your dataset and your interest, not the one you think will "look hardest" on a portfolio. The C5 grader has read 200 capstone repos and can tell within thirty seconds whether the project chose a problem that fit its data; mismatches penalize.

### 2.1 Binary classification (the C5 default)

The target is a single 0/1 label. The metric is one of:

- **Accuracy.** Use only when classes are balanced (rare in industry).
- **F1.** Use when classes are imbalanced and false positives and false negatives are roughly equally costly.
- **Precision at fixed recall** (or recall at fixed precision). Use when the *operating point* is constrained by business — e.g., "we will only contact 1,000 customers a day, so what is the precision at the top 1,000 ranked by predicted probability".
- **ROC-AUC.** Use when you want a threshold-free comparison; a sanity check, not a deploy metric.
- **PR-AUC.** Use when classes are very imbalanced (positive rate < 5 percent); the area under the precision-recall curve dominates ROC-AUC in that regime.

Datasets that fit binary classification: UCI Adult Income (predict income > 50k; 33 percent positive class), Kaggle Titanic (predict survival; 38 percent positive), Lending Club default (predict loan default; ~20 percent positive depending on the cohort), a custom text-spam dataset, a custom medical-imaging "tumor / not tumor" dataset.

The C5 starter `mini-project/starter.py` defaults to binary classification on a small tabular dataset, because that combination is the highest-success-rate path through the capstone.

### 2.2 Multi-class classification

The target is one of `K` discrete labels with `K > 2`. The metric family is the binary one extended: per-class precision and recall, the confusion matrix, macro and micro F1, the top-1 and top-5 accuracy (if `K > 10`). Datasets: MNIST (`K = 10`; trivial in 2026 but is the canonical first multi-class), CIFAR-10 (`K = 10`; harder), Fashion-MNIST (`K = 10`; harder than MNIST, easier than CIFAR), Caltech-101 (`K = 102`; the Week 9 default), 20 Newsgroups (`K = 20`; text), AG News (`K = 4`; text).

A multi-class capstone scores well when the analysis goes beyond the headline accuracy — a per-class precision-recall breakdown that flags the two classes the model confuses most often, plus a one-paragraph hypothesis for *why*, is the kind of EDA-meets-evaluation that distinguishes a B from an A.

### 2.3 Regression

The target is a continuous real number. The metric family:

- **RMSE.** Use when large errors are disproportionately costly.
- **MAE.** Use when all errors are equally costly per unit (i.e., a 10-dollar error is exactly twice as bad as a 5-dollar error).
- **MAPE / sMAPE.** Use when the target spans orders of magnitude (e.g., house prices from 50k to 5M) and percentage errors are the natural unit.
- **R-squared.** Sanity check, not a deploy metric. Reported alongside RMSE.
- **The residual plot.** Always. A scatter plot of residuals against predicted values reveals heteroskedasticity, non-linearity, and outliers in a way no scalar metric does.

Datasets: California Housing (predict median home price; the C5 W4 example), Diamonds (Kaggle; predict diamond price from carat, cut, color, clarity), Bike Sharing (UCI; predict daily ridership). Predict-the-price problems are the C5 regression default because the metric is naturally interpretable and the residual plot reveals structure.

### 2.4 Time-series forecasting

A specialization of regression where the target is the value of a single series at a future time and the features are lags of the same series (plus exogenous variables). The metric family adds MAPE and **MASE** (the mean absolute scaled error, which compares your model to a naive last-value baseline; <https://otexts.com/fpp3/accuracy.html>).

The C5 capstone time-series default is one of: NYC daily taxi rides, Austin daily ridership, a stock price series from <https://www.alphavantage.co/> free tier (do not over-claim the result; predicting stock prices well is impossible, and the rubric grades the *methodology* — backtesting, walk-forward validation, naive baselines — not the headline number).

A time-series capstone has its own discipline. The train/test split is by time, never by random sample (a random sample leaks future information into the past). The validation is walk-forward — for a 1000-point series, train on points 1-700, validate on 701-800; retrain on 1-800, validate on 801-900; and so on. The baseline is `predict_last_value` (the naive baseline; `y_pred[t] = y_obs[t-1]`); the candidate model has to beat MASE = 1 by enough to be worth deploying.

### 2.5 Sequence-to-label NLP

Text in, single label out. Sentiment analysis (positive / negative), spam detection (spam / ham), topic classification (sports / politics / tech), language identification (eng / spa / fra / ...). The metric family is the multi-class family.

The C5 sequence-to-label default is one of:

- **Sentiment on IMDB.** 25k train, 25k test, binary. The 2011 benchmark (<https://ai.stanford.edu/~amaas/data/sentiment/>); two pretrained-DistilBERT epochs achieves ~93 percent.
- **Topic on AG News.** 120k train, 7.6k test, 4 classes.
- **Spam on the SMS Spam Collection.** 5.5k samples, binary, free at <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>.

For the capstone, the natural recipe is `LogisticRegression` on `TfidfVectorizer` features (the strong baseline; trains in seconds, deploys in megabytes) as the deploy candidate, with a fine-tuned `distilbert-base-uncased` (the stretch goal; deploys in 250 MB and needs the Hugging Face Spaces hardware to serve fast enough). The C5 rubric accepts either as the final model; the TF-IDF logistic regression is the *recommended* one because it is the model the rubric was designed around.

### 2.6 Recommendation

Given a user-item interaction history, predict the items a user will engage with next. The metric family is **Recall@K**, **Precision@K**, and **NDCG@K**, all evaluated on a held-out future-interactions slice.

Recommendation capstones are technically interesting but operationally hard for the one-week budget — the data is sparse, the baseline (popularity) is surprisingly strong, the evaluation is fiddly. The C5 capstone allows recommendation as a problem type but does not list it as the default. If you pick recommendation, the C5 dataset of choice is MovieLens-1M (<https://grouplens.org/datasets/movielens/1m/>); the baseline is "predict the user's top 10 by global popularity"; the candidate model is `implicit`'s alternating-least-squares (<https://github.com/benfred/implicit>) or a two-tower matrix-factorization model in PyTorch.

---

## 3. Picking a dataset

There are eight C5-approved free dataset sources. The capstone rubric accepts any of the eight, plus any other source that meets the licensing and reproducibility bar (a license that permits redistribution and modeling; a stable URL; a public dataset card or an equivalent that you can summarize).

### 3.1 The eight sources, ranked

1. **Hugging Face Datasets Hub** (<https://huggingface.co/datasets>). The modern default. Every dataset has a dataset card by default (Hugging Face enforces a template); every dataset has a commit hash you can pin in the capstone code; every dataset can be loaded with `load_dataset(name, revision=commit_hash)`. The first place to look. The downside is that "Hugging Face Datasets" subtly biases toward NLP and image data; for tabular data, OpenML and UCI are better-curated.
2. **OpenML** (<https://www.openml.org/>). Tabular focus. Every dataset has rich metadata, a DOI, a license tag, and a per-dataset "tasks" page that lists the canonical train/test splits and the leaderboard. The best place for a tabular capstone.
3. **UCI Machine Learning Repository** (<https://archive.ics.uci.edu/>). The classical archive. Smaller datasets, older provenance, lighter metadata than OpenML, but still the *de facto* home of "Adult Income", "Wine Quality", "Mushroom", "Iris", "Wisconsin Breast Cancer", and a hundred more.
4. **Kaggle Datasets** (<https://www.kaggle.com/datasets>). Large variety, variable quality. The Kaggle competition leaderboards are the second-best place after OpenML to learn what a strong baseline looks like for a given dataset. The downsides are that licensing is dataset-specific (read the license tab before committing), and that many "datasets" are scraped re-uploads of original data hosted elsewhere.
5. **data.gov** (<https://www.data.gov/>). US-government open data; large, well-curated, mostly tabular, mostly free of licensing issues. The right home for capstone problems with a public-policy angle.
6. **City open-data portals.** Every major US city now publishes open data: New York (<https://opendata.cityofnewyork.us/>), Chicago (<https://data.cityofchicago.org/>), Austin (<https://data.austintexas.gov/>), Los Angeles (<https://data.lacity.org/>), Seattle (<https://data.seattle.gov/>), etc. Granular, current, real-world; the right home for capstone problems with a city-scale operational angle.
7. **Project Gutenberg** (<https://www.gutenberg.org/>). Public-domain text. Used for the Week 10 and Week 11 sequence-model capstones; still relevant for a Week 12 NLP capstone.
8. **Wikipedia dumps** (<https://dumps.wikimedia.org/>). Full-corpus text. Larger than Project Gutenberg, more current, more topical. Out of scope for "load it all in memory" but reasonable if your capstone uses a subset (e.g., all Wikipedia articles in the `Category:Cities_in_Texas` tree).

### 3.2 The licensing question

Every dataset you use must permit redistribution and the kind of modeling you are doing. The acceptable licenses for the C5 capstone are:

- **CC0 / Public Domain.** Use it for anything.
- **CC BY 4.0.** Use it; cite the original.
- **CC BY-SA 4.0.** Use it; cite; redistribute under the same license. Tricky for a portfolio because it taints your downstream artifacts.
- **MIT, BSD, Apache 2.0.** Permissive software-style licenses sometimes used on datasets. Use freely.
- **Open Data Commons Attribution Licence (ODC-By).** Use it; cite.
- **A custom-license that allows redistribution and modeling.** Read it; if it allows your intended use, use it.

Avoid:

- **Non-commercial-only licenses** (CC BY-NC). The C5 portfolio is a non-commercial educational project, but you do not control where your portfolio's readers will use your work, and a non-commercial dataset taints downstream artifacts.
- **Datasets without a stated license.** "Found it on someone's GitHub" is not a license. If you cannot find a license, do not use it.
- **Scraped commercial data.** Scraping is a legal grey area; the C5 capstone does not litigate it.

The dataset card has a section for the license; fill it in. The grader checks.

> **EXPERIMENT 1.2 — pick a dataset in fifteen minutes.** Open three of the eight sources above. Browse for fifteen minutes. Pick *one* dataset that meets all three of: (a) the problem archetype you committed to in Experiment 1.1; (b) a permissive license; (c) under 1 GB on disk. Do not pick the "biggest available" or the "most prestigious". The rubric does not reward dataset size or fame; the rubric rewards lifecycle rigor on whatever dataset you pick.

---

## 4. The dataset card

Gebru et al. 2021's "Datasheets for Datasets" (<https://arxiv.org/abs/1803.09010>) is the paper that defined the modern dataset-documentation template. The paper's section 3 has the full questionnaire — about 50 questions across seven sections. The C5 dataset card at `mini-project/DATASET_CARD_TEMPLATE.md` is a one-page adaptation that keeps the seven-section structure and reduces the question count to the eight or nine that fit a one-week capstone.

The seven sections:

1. **Motivation.** Why does the dataset exist? Who created it? Who funded it?
2. **Composition.** What does each instance represent? How many instances? Any labels, and how were they assigned? Any missing values, and what they mean?
3. **Collection process.** How was the data acquired? Over what time period? With what consent process, if any?
4. **Preprocessing / cleaning / labeling.** What pre-existing cleaning or labeling has been applied? What was discarded? Is the raw data available, or only the cleaned version?
5. **Uses.** What is the dataset's intended use? Are there uses it should *not* be put to?
6. **Distribution.** How is the dataset available? What is the license? Is there a DOI?
7. **Maintenance.** Who maintains the dataset? Is it being updated? Is there a contact?

The grader reads section by section. Each section is two to four sentences in the C5 template — bullet points are acceptable. The total length target is one printed page; the Mitchell and Gebru papers are clear that the format is the discipline.

### 4.1 A worked example: the UCI Adult Income dataset card

**Motivation.** Predict whether an individual's annual income exceeds 50,000 US dollars based on demographic and employment census features. Extracted by Barry Becker from the 1994 US Census Bureau database. Donated to UCI in 1996. Originally intended as a benchmark for machine-learning algorithms in the late-1990s tabular-data research community.

**Composition.** 48,842 instances (32,561 train; 16,281 test). Each instance is one anonymized adult worker. 14 features: 6 continuous (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week), 8 categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country). One binary target: `income` is `<=50K` or `>50K`. Approximately 24 percent of instances are positive (income > 50K). The categorical columns encode missing values as the literal string `?`; 6.4 percent of `workclass` and 5.7 percent of `occupation` values are `?`.

**Collection process.** Drawn from the 1994 US Census Current Population Survey. Sample of individuals who completed the long-form census questionnaire. Reasonable to assume informed consent for census participation; no additional consent for ML benchmarking was obtained. Sample restricted to individuals aged 16+, with income > 100 USD, weeks-worked > 0, hours-per-week > 0.

**Preprocessing / cleaning / labeling.** Continuous features preserved; categorical features kept as strings. The fnlwgt column is the survey-weight; never use it as a feature in modeling. The `education` and `education-num` columns are redundant — `education-num` is the integer encoding of `education`; pick one. The 24 percent positive rate creates a class imbalance the model must handle.

**Uses.** Suitable for: benchmarking tabular classification algorithms; teaching ML workflow; demonstrating fairness audits (race and sex are protected attributes). Not suitable for: any real income-classification application — the data is 30 years old, US-specific, sampled before substantial economic and demographic changes; the dataset reflects 1994 US labor-market patterns.

**Distribution.** Available at <https://archive.ics.uci.edu/dataset/2/adult>. Released under CC BY 4.0 (the UCI default since 2023). DOI: `10.24432/C5XW20`.

**Maintenance.** Donated in 1996; static since. Maintained by UCI as part of the archival ML repository. Contact: <ml-repository@ics.uci.edu>.

Note the length: roughly half a printed page. Every claim is verifiable from the UCI page or from the original 1994 census documentation. Every statistic is checkable in fifteen seconds with `pandas`. That is the bar.

> **EXPERIMENT 1.3 — write your dataset card in thirty minutes.** Load your chosen dataset, fill the seven sections with the data in front of you, and stop at one printed page. The grader rewards completeness over depth; the section that says "this dataset has no protected attributes — see section 5" scores higher than one that omits the section entirely.

---

## 5. Exploratory data analysis as written evidence

EDA is the bit where you open the dataset and look at it. In the textbooks it is presented as a free exploration; in the capstone rubric it is *written evidence*. The grader cannot see the histograms you plotted and rejected; the grader can only see the histograms you saved with one-sentence captions in `notebooks/01-eda.md` or `notebooks/01-eda.ipynb`.

### 5.1 The eight-step EDA checklist

The C5 EDA checklist, applied to *every* capstone regardless of problem type:

1. **Shape.** `df.shape`. Save the row and column counts; check they match the dataset card.
2. **Dtypes.** `df.dtypes` and `df.info()`. Flag any column whose dtype is `object` and that you expected to be numeric — that almost always means non-numeric tokens (the literal `'?'`, the literal `'N/A'`, the literal `'unknown'`) in the column.
3. **Missingness.** `df.isna().sum()`. Plot a missingness bar chart. For each column with > 1 percent missingness, write one sentence stating *which* missingness pattern you believe applies — Missing Completely At Random (MCAR), Missing At Random (MAR), or Missing Not At Random (MNAR). The capstone's most-common modeling mistakes flow from misclassifying MNAR as MCAR.
4. **Univariate distributions.** For each numeric column, a histogram. For each categorical column, a value-counts bar. Save each as a PNG; caption each in one sentence.
5. **Target distribution.** If you have a target, plot it. For classification, the class-balance bar. For regression, the histogram and a log-scaled histogram. The class imbalance and the regression-target skew are the two diagnostics that drive metric and model choice.
6. **Bivariate relationships.** A heatmap of `df.corr()` for numeric columns; a small-multiples scatter of each predictor against the target (use `seaborn`'s `pairplot` for `<= 6` numeric features; use a hand-rolled small-multiples grid for more).
7. **Outliers.** Identify the top-5 and bottom-5 instances by each numeric feature. Look at them. Are they data-entry errors, real edge cases, or rare-but-legitimate observations? The decision drives the cleaning rule.
8. **A one-paragraph EDA summary.** "Here are the four things I learned that change my modeling plan." This paragraph is the most-read paragraph in the EDA notebook by the grader.

### 5.2 The pandas one-liners you will use most

```python
import pandas as pd

df = pd.read_csv("data/raw.csv")

# Shape and dtypes.
print(df.shape)
print(df.dtypes)
df.info()

# Missingness.
df.isna().sum().sort_values(ascending=False)

# Univariate.
df.describe(include="all")  # both numeric and categorical.
df["category_col"].value_counts(dropna=False)  # dropna=False shows NaN as its own bucket.

# Bivariate.
df.corr(numeric_only=True)
df.groupby("category_col")["target"].mean()  # target rate per category.

# Outliers.
q_low, q_high = df["numeric_col"].quantile([0.01, 0.99])
df[(df["numeric_col"] < q_low) | (df["numeric_col"] > q_high)]
```

The Week 2 lecture covered every line above. The capstone EDA is Week 2 with intent — every output goes into the notebook with a caption, every decision goes into a one-sentence note that the grader will read.

### 5.3 The three EDA findings that change the modeling plan

Across the 200 capstones the C5 grader has read, the same three EDA findings change the modeling plan for the majority of submissions:

- **Class imbalance.** If the positive class is < 20 percent, you cannot use accuracy as the metric, you cannot use the default `LogisticRegression` threshold of 0.5 without tuning, and you should consider class-weighted loss or `imbalanced-learn` resampling. The Week 4 lecture covered the trade-offs.
- **MNAR columns.** If a column has missingness > 10 percent and the missingness is not random (e.g., `salary` is missing for the lowest-earning workers because they are not asked the question), imputing the mean is wrong. Either drop the column, fit a separate model on the non-missing subset, or use an explicit "missing" indicator column.
- **A leakage column.** If a column has correlation > 0.99 with the target on the training set, it is either the target itself (you have made a copy and forgotten) or a feature that was computed *after* the target was known. Either way, the model will achieve 100 percent test accuracy and produce nothing on deploy. Find the leakage; drop the column; redo the analysis.

The capstone rubric rewards EDA notebooks that flag at least one of those three findings. If your EDA "looks fine" on every diagnostic, you have not yet looked hard enough.

> **EXPERIMENT 1.4 — run the eight-step EDA on your dataset in sixty minutes.** Open the dataset, run each of the eight steps, save the plots, and write the one-paragraph summary at the top. If the dataset is "clean" and "nothing is interesting", you have not yet found the interesting things; pick three columns and pivot the target against them and look at the histograms.

---

## 6. The train/val/test split

The capstone's *first* code-level decision after EDA is the split. The split is sacred:

- **Train.** ~70 percent. Fit on this.
- **Validation.** ~15 percent. Use for hyperparameter search and for early stopping.
- **Test.** ~15 percent. Touch *once*, at the very end, for the headline metric in the model card.

The Week 4 lecture covered the split; the capstone rubric enforces it. The single most-common cause of "my model got 95 percent on test but 20 percent on the deployed UI" is leakage between the split — usually because the split was random when it should have been time-based, or because the same patient / customer / household appears in both train and test, or because feature engineering touched the full dataset before the split.

The C5 capstone split rules:

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

# Step 1: split off the test set first. This set never gets touched again.
X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42,
)

# Step 2: split the remainder into train and validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_pool, y_pool, test_size=0.1765, stratify=y_pool, random_state=42,
)  # 0.1765 of 85% gives 15% of original.

# Step 3: drop X_pool and y_pool; you do not need them again.
del X_pool, y_pool

print(X_train.shape, X_val.shape, X_test.shape)
```

For time-series data the split rule is different: split *by time*, never by random sample. For grouped data (the same patient appears in multiple rows) the split rule is *by group*, using `sklearn.model_selection.GroupShuffleSplit`. The capstone's `PROBLEM.md` states which split rule applies and why.

### 6.1 The "do not touch test" rule, mechanically

The capstone's `train.py` should mention `X_test` and `y_test` exactly twice — once when the split happens, once when the final headline metric is computed at the very end. If `X_test` appears in any hyperparameter-search loop, any cross-validation block, any plotting helper, your submission fails the rubric's discipline check.

The C5 mechanical enforcement: use a `final_eval(model)` function at the bottom of `train.py` that takes the trained model and prints the test metric. The function is called once at the end; the grader greps for `X_test` and counts the occurrences. Three or more is a fail.

---

## 7. The baseline

The baseline is the dumbest thing that could possibly work. It is not a placeholder; it is a graded artifact. The baseline lives in `train.py` next to the candidate model and is reported in the model card alongside the candidate.

The baselines by problem archetype:

- **Binary classification:** `DummyClassifier(strategy="most_frequent")`. Predicts the majority class for every input. Reports accuracy = `p_majority`.
- **Multi-class classification:** `DummyClassifier(strategy="most_frequent")` or `DummyClassifier(strategy="stratified")` (samples from the training class distribution). The stratified version is the harder baseline.
- **Regression:** `DummyRegressor(strategy="mean")`. Predicts the training-set mean for every input. Reports RMSE equal to the training-target standard deviation.
- **Time-series forecasting:** `predict_last_value`. Predicts `y[t-1]` for `y[t]`. Reports MASE = 1 by definition.
- **Sequence-to-label NLP:** `LogisticRegression` on `TfidfVectorizer` features with `max_features=10000`. A surprisingly strong baseline; in many cases the headline test metric is within 2 points of a fine-tuned transformer.
- **Recommendation:** "predict the K most-popular items globally", regardless of user.

The candidate model must beat the baseline by an effect size that justifies deploying the more complex model. The rubric's table of thresholds:

| Improvement over baseline | Verdict |
|---:|---|
| 0 to 1 point | The candidate is not worth deploying. Use the baseline. |
| 1 to 3 points | Deploy if and only if the candidate is operationally similar (same dependencies, similar latency). |
| 3 to 10 points | Deploy the candidate. The improvement justifies the complexity. |
| 10+ points | Either deploy the candidate or check for leakage. Improvements that large usually flag a problem. |

This is not a rule; it is a guideline. The model card explains the trade-off in your problem's terms.

> **EXPERIMENT 1.5 — train the baseline first.** Before you train anything else, fit the baseline. Print the headline metric. Save it to `runs/baseline/metric.json`. You now have a number you must beat. Every candidate model is judged against that number.

---

## 8. What you have at the end of Lecture 1

By Tuesday morning you should have:

- **`PROBLEM.md`** with the seven-section problem statement. Two pages.
- **`data/DATASET_CARD.md`** with the seven-section dataset card. One page.
- **`notebooks/01-eda.ipynb`** or `notebooks/01-eda.md` with the eight-step EDA. Three to five pages.
- **`runs/baseline/metric.json`** with the baseline metric on the validation set.
- **`train.py` skeleton** with the train/val/test split and a placeholder candidate model.

That is a third of the capstone deliverable, done in the first 24 hours. The remaining four days are model selection, fairness audit, model card, packaging, deploy, monitoring, and writeup. Lecture 2 covers metrics, fairness, and model cards. Lecture 3 covers packaging and deploy.

---

## 9. Recap and next

- Translate the stakeholder request into a written problem statement before opening pandas.
- Pick one of the six problem archetypes (binary classification, multi-class, regression, time-series, sequence-to-label, recommendation) and stick with it.
- Pick one dataset from one of the eight C5-approved sources, with a permissive license.
- Write a one-page dataset card following the Gebru et al. 2021 template.
- Run the eight-step EDA and write the findings as evidence the grader can read.
- Split into train, validation, test before anything else; touch test only at the very end.
- Train the baseline first. The candidate model must beat it.

Lecture 2 covers evaluation metrics that fit the problem, slice-based fairness audits, the Mitchell et al. 2019 model card, and the versioning of datasets and models with DVC. Lecture 3 covers packaging and deploy to Hugging Face Spaces, Streamlit Community Cloud, or Fly.io. The mini-project is where it all lands.
