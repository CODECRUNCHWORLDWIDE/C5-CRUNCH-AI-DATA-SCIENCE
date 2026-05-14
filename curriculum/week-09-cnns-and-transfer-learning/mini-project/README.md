# Mini-Project — A Transfer-Learning Image Classifier in PyTorch

> Build an image classifier on CIFAR-10 (the default) or Oxford Flowers-102 (the stretch) that reaches **90%+ test accuracy** via transfer learning from a pretrained ResNet-18. Include a hand-built `TinyVGG` from-scratch baseline so the report can quantify the transfer-learning benefit on the same dataset. Save both trained models as `state_dict`s, write a separate evaluation script for each, produce a training-curve plot and a confusion matrix, and write a 600-900 word report that an engineering manager could read in five minutes and conclude "this person can ship a vision model in 2026."

This is the ninth artifact of your `crunch-ai-portfolio-<yourhandle>` repo. Week 8 was the bare PyTorch training loop on FashionMNIST — the proof that you can write the loop. Week 9 is the proof that you can ship a vision model: you load a pretrained backbone, you swap the head, and you reach an accuracy that would have been state-of-the-art ten years ago in under twenty minutes of training. The "wow factor" of this project is exactly this: the same student who hand-rolled backprop in Week 7 now hits 90%+ on CIFAR-10 in three lines of model code.

**Estimated time:** 8 hours, spread across Thursday-Sunday.

---

## What you will build

Two trained models, two evaluation scripts, one notebook, and one report:

1. **`baseline_train.py`** — trains the `TinyVGG` from-scratch CNN from Lecture 2 Section 5 on CIFAR-10 for 15 epochs of Adam at lr=1e-3 with `RandomCrop` and `RandomHorizontalFlip` augmentation. Saves `baseline_model.pt`. Expected test accuracy: 70-73%.
2. **`transfer_train.py`** — loads `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`, replaces the head with `nn.Linear(512, 10)`, fine-tunes end-to-end with differential learning rates (backbone at `1e-4`, head at `1e-3`) for 10 epochs. Saves `transfer_model.pt`. Expected test accuracy: 90-93%.
3. **`baseline_evaluate.py`** and **`transfer_evaluate.py`** — each loads its corresponding checkpoint, runs one evaluation pass on the test loader, prints the accuracy and saves a confusion matrix figure. Each script's accuracy must match the final-epoch accuracy from its training script to the third decimal.
4. **`exploration.ipynb`** — a Jupyter notebook that loads CIFAR-10, shows a 3×3 grid of sample images per class with class names, runs both trained models on 10 test images (including a couple of misclassifications), and side-by-sides their predictions. The notebook is for the human reader; the scripts are for the machine reader.
5. **`report.md`** — a 600-900 word executive summary. The headline is the from-scratch vs. transfer-learning accuracy gap on the same compute budget. The narrative is "we measured the gap; here is why it exists; here is when each approach is appropriate."

The two scripts together (baseline + transfer) implement the **comparison protocol** that the report is built around. Same dataset, same loss function, same eight-line training loop; the only differences are the model class and the optimizer's parameter groups. The 20-percentage-point gap is the deliverable.

---

## The dataset

Two default options.

### Option A — CIFAR-10 (the default; 32×32 RGB; ten classes)

Available as `torchvision.datasets.CIFAR10`. 50k training images, 10k test images, ten classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The C5 minimum target is **≥90% test accuracy** via transfer learning. Reaching 92-93% is the common range; reaching 95%+ requires more aggressive augmentation, a learning-rate schedule, or a larger backbone (ResNet-50). Document the gap from your number to the published ResNet-18-on-CIFAR-10 state-of-the-art if you want a stretch achievement.

References:
- Dataset: <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>
- The original paper: <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>

### Option B — Oxford Flowers-102 (the stretch; 102 classes; ~8k images)

Available as `torchvision.datasets.Flowers102`. About 8k images across 102 categories of flowers; a much smaller dataset than CIFAR-10 and a harder task because the per-class images are fewer (~80 per class vs. CIFAR-10's 5000 per class). The C5 minimum target is **≥85% test accuracy** via transfer learning.

Flowers-102 is the better choice for demonstrating transfer learning's data-efficiency advantage. With only 8k training images, a from-scratch CNN reaches maybe 40-50%; the transfer-learning ResNet-18 reaches 85-90% in the same compute budget. The gap is larger than on CIFAR-10 and the story is more compelling.

References:
- Dataset: <https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html>
- The original paper: <https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/nilsback08.pdf>

### Picking between them

- **Pick CIFAR-10 if** you have a deadline, this is your first transfer-learning project, or you want a clean comparison with the Week 8 / Week 9 lecture results.
- **Pick Flowers-102 if** you finished CIFAR-10 in a previous course / hobby project, you want a more impressive headline number, or you want to make the "transfer learning eats small datasets" argument as strongly as possible.

The mini-project rubric does *not* penalize you for picking A; it *does* reward the honest Flowers-102 report that makes the data-efficiency argument with concrete numbers.

---

## Acceptance criteria

- [ ] A new directory `week-09/` exists in your `crunch-ai-portfolio-<yourhandle>` repo.
- [ ] `pip install -r requirements.txt` from a fresh clone works. The file pins `torch>=2.4,<3`, `torchvision>=0.19,<1`, `numpy>=2,<3`, `matplotlib>=3.8,<4`, `scikit-learn>=1.5,<2`, `jupyter`.
- [ ] **`python baseline_train.py`** runs end-to-end, downloads the dataset on first run, trains for ≥15 epochs, prints per-epoch summaries, and produces `baseline_model.pt` and `baseline_curves.png`.
- [ ] **`python transfer_train.py`** runs end-to-end, downloads the pretrained ResNet-18 weights on first run, trains for ≥10 epochs, prints per-epoch summaries, and produces `transfer_model.pt` and `transfer_curves.png`.
- [ ] **`python baseline_evaluate.py`** loads `baseline_model.pt`, runs one evaluation pass, prints the test accuracy, and saves `baseline_confusion.png`. The accuracy must match the final-epoch accuracy from `baseline_train.py` to the third decimal.
- [ ] **`python transfer_evaluate.py`** does the same for the transfer model and produces `transfer_confusion.png`.
- [ ] Both training scripts use **`torch.manual_seed(42)`** at the top; the same run twice produces the same final accuracy.
- [ ] The models are defined as **subclasses of `nn.Module`** (or, for the transfer model, the unmodified `torchvision.models.resnet18` with the head swap). The class definitions and the `build_*` functions live in `model.py`; both train scripts and both evaluate scripts import from there. This is the production pattern.
- [ ] The training loop is the **bare loop from Week 8 Lecture 2 Section 7**. No Lightning, no fastai, no `accelerate`.
- [ ] The baseline model reaches **≥70% test accuracy** on CIFAR-10 (or ≥40% on Flowers-102).
- [ ] The transfer-learning model reaches **≥90% test accuracy** on CIFAR-10 (or ≥85% on Flowers-102).
- [ ] `python -m py_compile *.py` succeeds for every Python file in `week-09/`.
- [ ] `report.md` is 600-900 words. No more, no less.

---

## File layout

```text
week-09/
├── requirements.txt
├── model.py              # TinyVGG class; build_pretrained_model helper
├── data.py               # build_cifar10_loaders_32 + build_cifar10_loaders_224
├── baseline_train.py
├── baseline_evaluate.py
├── transfer_train.py
├── transfer_evaluate.py
├── exploration.ipynb
├── report.md
├── baseline_model.pt          # generated
├── transfer_model.pt          # generated
├── baseline_curves.png        # generated
├── transfer_curves.png        # generated
├── baseline_confusion.png     # generated
├── transfer_confusion.png     # generated
└── README.md             # your own copy of these instructions
```

The split between `model.py` / `data.py` and the four scripts is the production pattern: model and data are libraries, the four scripts are entry points. The same model class is loaded by `*_train.py` and `*_evaluate.py`; the same data-loader function is called from both train and evaluate.

---

## The report

The report is the artifact that recruiters read. It should be readable in five minutes and answer four questions:

1. **What did you build?** One paragraph. Architecture choices, dataset, framework.
2. **What were the numbers?** A short table:

| Variant                    | Trainable params | Epochs | Final test acc | Wall time |
|----------------------------|------------------|-------:|---------------:|----------:|
| Baseline (TinyVGG)         | ~1.1M            |   15   | _e.g._ 72.5%   | _e.g._ 12 min on Colab T4 |
| Transfer-learning ResNet-18| ~11.2M           |   10   | _e.g._ 92.7%   | _e.g._ 25 min on Colab T4 |

3. **What does the gap mean?** One or two paragraphs. The pretrained ResNet-18 starts from weights that have already learned a useful visual representation on 1.28M ImageNet images; the from-scratch `TinyVGG` does not. The 20-percentage-point gap is the value of that pretraining. State this in your own words; be specific about what "useful visual representation" means (early layers learn edge detectors, deeper layers learn part detectors, etc.).
4. **When would you reach for each?** One paragraph. The honest answer: transfer learning is the right default for any vision task in 2026 with under 100k labeled images. From-scratch CNNs are a teaching exercise (or appropriate for self-supervised pretraining on a large unlabeled corpus). State this with the data from your own experiment as the evidence.

Add a fifth section ("Implementation notes") of ~150 words documenting the hyperparameters that mattered:

- The choice of differential learning rates (`backbone_lr=1e-4`, `head_lr=1e-3`) — why these values, what happens if you use the same LR everywhere.
- The 224×224 resize and ImageNet normalization — why match the pretrained model's input distribution.
- The augmentation choices (`RandomCrop`, `RandomHorizontalFlip`) — why they help and what their hyperparameters are.

End with a one-sentence "Next steps" mentioning what you would try if you had more compute (longer training, learning-rate schedule, ResNet-50 backbone, more aggressive augmentation).

---

## What "good" looks like

The C5 grading rubric (in `rubric.md`) checks:

1. **Correctness:** Both scripts run end-to-end without errors; both reach the target accuracy.
2. **Reproducibility:** `torch.manual_seed(42)` set; same run produces same number.
3. **Engineering hygiene:** Model class in `model.py`; data loaders in `data.py`; no copy-paste between scripts.
4. **Report quality:** Headline numbers in a table; the "why does transfer learning help" question answered concretely; word count respected.
5. **Comparison honesty:** Both variants trained with the same eight-line loop; the only differences are the model class and the optimizer.

The C5 reading: this is the project where a recruiter who reads your `crunch-ai-portfolio` repo concludes "this candidate has done real vision work." It is the artifact that gets you a phone screen.

---

## Hints

- **Cache the dataset before the timed runs.** The first call to `torchvision.datasets.CIFAR10(download=True)` downloads ~170 MB. Run it once in `exploration.ipynb` before timing the training scripts.
- **Cache the pretrained weights too.** First call to `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` downloads ~45 MB. Same routine.
- **Use `num_workers=2` or `num_workers=4` on the DataLoader.** Without it the data pipeline becomes the bottleneck at 224×224.
- **Pin memory only on CUDA.** `pin_memory=torch.cuda.is_available()`.
- **Use `@torch.no_grad()` on `evaluate`.** Doubles the eval-pass throughput; halves the memory.
- **Save the model with `weights_only=True`-safe `state_dict()`.** Use `torch.save(model.state_dict(), path)`, not `torch.save(model, path)`. Load with `model.load_state_dict(torch.load(path, weights_only=True))`. Same pattern as Week 8.
- **Confusion matrix plotting.** `sklearn.metrics.confusion_matrix` returns a NumPy array; pass it to `matplotlib.pyplot.imshow` with `cmap="Blues"` and annotate cells with `ax.text(j, i, str(value))` in a nested loop. Add `ax.set_xticks` and `ax.set_yticks` for the class names. ~30 lines of plotting code total.
- **The reload-and-evaluate test of correctness.** After saving, load into a fresh model instance and run evaluation; the accuracy must match. If it does not, you have a `model.train()` / `model.eval()` bug, a BatchNorm bug, or you saved the wrong dict.

## Stretch goals

These are not required for full credit; they are credit-eligible if the basic criteria are met.

- **Add a third variant: feature extraction (frozen backbone).** Train it for 5 epochs and add a third row to the report's table. The story "frozen reaches 87% in 5 epochs, fine-tune reaches 92% in 10 epochs" is more nuanced than the two-variant version.
- **Implement a learning-rate schedule.** Use `torch.optim.lr_scheduler.CosineAnnealingLR` (<https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html>) or `OneCycleLR`. The schedule should add 1-2 percentage points to the transfer-learning final accuracy.
- **Replace ResNet-18 with ResNet-50.** Same training script, different model factory. The 50-layer model reaches 95%+ on CIFAR-10 with the same recipe, at ~3x the compute cost. Worth doing once for the experience; document the additional compute in the report.
- **Repeat on Flowers-102 even if you picked CIFAR-10 originally.** The data-efficiency argument is more visceral on 8k images than on 50k.
- **Try a vision transformer.** `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)`. Same head-swap recipe (`model.heads.head = nn.Linear(...)`). Compare the transfer-learning accuracy and compute against ResNet-18. Spoiler: ViT-B/16 has more parameters and a larger compute cost; on CIFAR-10 with limited training, it does not always beat ResNet-18. Document what you see.
