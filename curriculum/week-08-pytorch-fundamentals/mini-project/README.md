# Mini-Project — A FashionMNIST (or CIFAR-10) Image Classifier in PyTorch

> Build an image classifier end to end in PyTorch 2.x. Train it on FashionMNIST (the default) or CIFAR-10 (the stretch). Save the trained model as a `state_dict`. Write a separate evaluation script that reloads the checkpoint and reproduces the test accuracy. Produce a training-curve plot and a confusion-matrix heatmap. Write a 600-900 word report that an engineering manager could read in five minutes and conclude "this person can build a PyTorch model from scratch and ship a reproducible artifact."

This is the eighth artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 7 was the from-scratch NumPy MLP — the proof that you understand what the framework is doing. Week 8 is the framework version — the proof that you can ship it. Recruiters with taste read Week 8 as the answer to "okay, you understand backprop; can you actually train and ship a model?"

**Estimated time:** 8 hours, spread across Thursday-Sunday.

---

## What you will build

Two scripts plus a notebook plus a report:

1. **`train.py`** — runs the full training pipeline: loads FashionMNIST via `torchvision`, builds the model, trains for `n_epochs`, evaluates on the test set each epoch, saves checkpoints, produces the training-curve plot.
2. **`evaluate.py`** — loads the saved `state_dict`, reconstructs the model, runs one evaluation pass, prints the test accuracy and the confusion matrix. Must produce the same accuracy as the final epoch of `train.py`.
3. **`exploration.ipynb`** — a Jupyter notebook that loads the dataset, shows a 3x3 grid of sample images per class (with class names), and inspects the trained model's predictions on a handful of test examples (including a couple of misclassifications). The notebook is for the human reader; the scripts are for the machine reader.
4. **`report.md`** — a 600-900 word executive summary that a non-technical reader can finish in five minutes.

The two scripts together implement the **train-then-reload-then-evaluate** workflow that every production PyTorch project uses. If your `evaluate.py` reproduces the `train.py` test accuracy to the third decimal, the workflow is correct.

---

## The dataset

Two default options.

### Option A — FashionMNIST (the default; 28x28 grayscale; ten classes)

Available as `torchvision.datasets.FashionMNIST`. 60k training images, 10k test images. The C5 minimum is **≥85% test accuracy** with a 2-layer MLP and 10 epochs of Adam at `lr=1e-3`. Reaching 88-89% is the common range; reaching 90%+ with an MLP alone (no CNN, no augmentation) is unusual and worth documenting.

Reference: <https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html>
Paper: <https://arxiv.org/abs/1708.07747>

### Option B — CIFAR-10 (the stretch; 32x32 RGB; ten classes)

Available as `torchvision.datasets.CIFAR10`. 50k training images, 10k test images. The C5 minimum for an MLP-only baseline is **≥48% test accuracy**; with a tiny CNN (as in Challenge 1) the target is **≥65%**.

The honest C5 reading: an MLP on CIFAR-10 is *bad*, and that is the point. The reason CNNs took over vision in 2012 is exactly this gap. If you choose CIFAR-10 for the mini-project, you should also include the CNN comparison from Challenge 1 — the report's headline is the MLP-vs-CNN gap, not the MLP number alone.

Reference: <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>

### Picking between them

- **Pick FashionMNIST if** you have a deadline, this is your first PyTorch project, or you want the cleanest possible report.
- **Pick CIFAR-10 if** you want to motivate Week 9 (CNNs) in your own portfolio, you have a GPU, or you finished Challenge 1 and want to extend it.

The mini-project rubric does *not* penalize you for picking A; it *does* reward the honest CIFAR-10 report that makes the architecture-matters argument with data.

---

## Acceptance criteria

- [ ] A new directory `week-08/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `torch>=2.4,<3`, `torchvision>=0.19,<1`, `numpy>=2,<3`, `matplotlib>=3.8,<4`, `scikit-learn>=1.5,<2`, `jupyter`.
- [ ] **`python train.py`** runs end-to-end without errors, downloads the dataset on first run, trains for ≥5 epochs, prints per-epoch summaries, and produces a `model.pt` file plus a `training_curves.png` figure.
- [ ] **`python evaluate.py`** loads `model.pt`, runs one evaluation pass on the test loader, prints the test accuracy, and saves `confusion_matrix.png`. The accuracy must match the final-epoch accuracy from `train.py` exactly (to the third decimal).
- [ ] The training script uses **`torch.manual_seed(42)`** at the top; the same run twice produces the same final accuracy.
- [ ] The model is defined as a **subclass of `nn.Module`** (not just an `nn.Sequential` at module scope). The class lives in `model.py`; both `train.py` and `evaluate.py` import it from there. This is the production pattern.
- [ ] The training loop is the **bare loop from Lecture 2 Section 7**. No Lightning, no fastai, no `accelerate`.
- [ ] The model is saved as a **`state_dict`** with `torch.save(model.state_dict(), "model.pt")`, not as a pickled object. The load side uses **`torch.load("model.pt", weights_only=True)`**.
- [ ] The script detects the **best available device** (`cuda > mps > cpu`) and moves the model and every batch to it.
- [ ] **Test accuracy reaches the C5 minimum** for your chosen dataset (≥0.85 for FashionMNIST MLP; ≥0.48 for CIFAR-10 MLP; ≥0.65 for CIFAR-10 CNN).
- [ ] **`exploration.ipynb`** loads the data, shows a 3x3 sample grid with class names, and inspects 5 model predictions (correct and incorrect mix). Plots are labeled.
- [ ] **`report.md`** is 600-900 words, structured as below.
- [ ] **`README.md`** in `week-08/` explains: setup, dataset download path, file layout, how to reproduce the result.

---

## Suggested layout

```text
crunch-ai-portfolio-<yourhandle>/
├── README.md
├── requirements.txt
└── week-08/
    ├── README.md
    ├── train.py             # the training pipeline
    ├── evaluate.py          # the reload-and-evaluate script
    ├── model.py             # the nn.Module subclass (shared)
    ├── data.py              # the build_loaders function (shared)
    ├── exploration.ipynb    # the Jupyter exploration notebook
    ├── model.pt             # the trained state_dict (small; commit it)
    ├── report.md            # the 600-900 word write-up
    └── images/
        ├── training_curves.png
        └── confusion_matrix.png
```

The `model.py` / `data.py` factoring is what makes `evaluate.py` short. Without it, the two scripts would duplicate the model class and the data loader; with it, both import a single source of truth.

`starter.py` in this directory is a working skeleton that you can copy into `train.py`. Read it before you write your own.

---

## The report structure

`report.md` is the artifact a hiring manager reads. Structure it as:

### 1. The problem (one paragraph)

What you trained, on what data, against what baseline. Be specific: "I trained a 2-layer MLP with 128 hidden units (101,770 parameters) on FashionMNIST. The baseline I compared against was a logistic regression at the same task (~84% test accuracy)."

### 2. The model and the training setup (one paragraph)

The architecture, the hyperparameters, the optimizer. "The model is `Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)`, trained with `nn.CrossEntropyLoss` and `torch.optim.Adam(lr=1e-3)` for 10 epochs with batch size 128. Random seed 42 throughout."

### 3. The training curves (one paragraph plus the figure)

Show `training_curves.png`. Describe what is happening: does the loss go down monotonically? Does the test accuracy plateau? Is there a sign of overfitting (train accuracy rising while test accuracy falls)?

### 4. The confusion matrix (one paragraph plus the figure)

Show `confusion_matrix.png`. Identify the worst-performing class and the most-common confusion. For FashionMNIST, the answer is almost always "Shirt is the worst, confused with T-shirt/Top and Pullover." Include the per-class accuracy distribution (best, worst, mean).

### 5. The honest paragraph (the most important paragraph; ~100 words)

What does this model *not* do? Three honest statements:

1. **The MLP architecture is not the right prior for images.** It ignores the 2D spatial structure of the input. A CNN of similar parameter count beats it by ~4 percentage points on FashionMNIST and ~15 on CIFAR-10 (Challenge 1).
2. **No regularization, no augmentation.** Adding dropout, batch normalization, or random-flip / random-crop augmentation would likely add another 1-2 percentage points. The C5 default for Week 8 is the simplest possible model.
3. **The test set is in-distribution.** Test images come from the same source as training. The model's performance on out-of-distribution images (different camera, different background, different aspect ratio) is unknown and likely much worse.

### 6. Reproducibility (one paragraph)

State the seed, the PyTorch version, and the device. "Trained on `torch==2.4.1` with `random_state=42` on a 2023 M2 Mac Mini (`mps` device). Reproduces to the third decimal."

### 7. What you would do next (one paragraph; the "interview talk-track")

Three to five things you would try if you had another week. Examples:

- "Switch to a CNN (Week 9 preview); expect ~4pp accuracy lift."
- "Add a learning-rate scheduler — `optim.lr_scheduler.CosineAnnealingLR` is the modern default."
- "Run a small architecture search over `n_hidden ∈ [64, 128, 256, 512]` and pick by validation accuracy."
- "Add `torchmetrics.Accuracy` instead of computing it manually; better for production."

Word budget: 600-900. Keep it tight.

---

## Grading

See `rubric.md` for the full rubric. The headline:

- **A**: every acceptance criterion checked; report is structured as above; the honest paragraph names ≥2 specific limitations; `evaluate.py` reproduces `train.py` exactly.
- **B**: all criteria checked except one (typically the confusion matrix is missing class names, or the report is under 600 words).
- **C**: training works but the script is not reproducible (random seed not set, or `evaluate.py` produces a different number than `train.py`).
- **F**: training does not work; the model class is not a `nn.Module`; or the script uses Lightning or fastai.

---

## What to do when you are stuck

The C5 Week 8 escalation order:

1. **Re-read Lecture 3 Section 9** (the debugging checklist) and apply it line by line.
2. **Reread the relevant PyTorch tutorial** — the data and saving tutorials are short and they cover the most common bugs.
3. **Read your own training curves.** A NaN loss is a learning-rate issue. A flat loss is a missing `zero_grad`. An accuracy stuck at 10% is a device or transform issue.
4. **Print the shapes.** Add `print(X.shape, y.shape, logits.shape)` at the top of the inner loop. Mismatched shapes are the most common forward-pass bug.
5. **Run on CPU.** GPU errors are asynchronous and the stack traces are in C++. CPU errors are immediate and the stack traces are in Python.
6. **Ask a teammate.** The C5 Discord channel for Week 8 is the right place.

---

## Submission

When complete:

1. Push the `week-08/` directory to your portfolio repo.
2. Run `python evaluate.py` from a fresh clone to verify reproducibility.
3. Open a PR or issue tagging the C5 channel for review.
4. The reviewer will load `model.pt`, run `python evaluate.py`, and check that the printed accuracy matches your reported number.

When the reviewer signs off, you have shipped your first reproducible PyTorch model. The rest of the curriculum builds on this loop.
