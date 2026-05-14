# Week 9 Mini-Project — Grading Rubric

Total: 100 points. Pass threshold: 70.

The rubric is split across five categories that match the C5 standard from Weeks 1-8. The "self-grade before submission" recommendation: walk through every checkbox, mark it honestly, and submit only if you would be willing to defend the marks in a five-minute conversation.

---

## 1. Correctness (30 points)

| Item | Points |
|------|-------:|
| Both `baseline_train.py` and `transfer_train.py` run end-to-end without errors on a fresh clone (after `pip install -r requirements.txt`). | 5 |
| Baseline `TinyVGG` reaches **≥70% test accuracy** on CIFAR-10 (or ≥40% on Flowers-102). | 5 |
| Transfer-learning ResNet-18 reaches **≥90% test accuracy** on CIFAR-10 (or ≥85% on Flowers-102). | 10 |
| Both `*_evaluate.py` scripts reproduce the final-epoch accuracy from their `*_train.py` to the third decimal. | 5 |
| `python -m py_compile *.py` succeeds for every Python file in `week-09/`. | 5 |

If the transfer-learning model misses 90% by less than 1 percentage point, the grader will read your `report.md` for the explanation. A clearly-documented "the run reached 89.2% with this seed; see Implementation Notes for why" is acceptable; an undocumented 88% is not.

---

## 2. Engineering hygiene (20 points)

| Item | Points |
|------|-------:|
| Model class definitions live in `model.py`; both train scripts and both evaluate scripts import from there. No copy-paste of the model class. | 5 |
| Data-loader functions live in `data.py`; both train and evaluate scripts call them. | 3 |
| `torch.manual_seed(42)` set at the top of both training scripts; same run twice produces the same final accuracy (verify this yourself). | 3 |
| `state_dict` is saved with `torch.save(model.state_dict(), path)`, not `torch.save(model, path)`. Loaded with `weights_only=True`. | 3 |
| Type hints on every Python function. `model.train()` / `model.eval()` called in the right places. `with torch.no_grad():` used in `evaluate`. | 3 |
| The training loop is the bare loop from Week 8 Lecture 2 Section 7; no Lightning, fastai, or accelerate. | 3 |

---

## 3. Comparison protocol (15 points)

| Item | Points |
|------|-------:|
| Same dataset, same `random_state`, same loss function, same eight-line loop across baseline and transfer-learning runs. | 5 |
| The only differences between the two runs are the model class, the data loader (32×32 vs. 224×224), and the optimizer's parameter groups. Document the differences in the report. | 5 |
| Wall-clock time is recorded for both runs and reported in the table. | 3 |
| Final-accuracy table is rendered in `report.md` with both variants, trainable parameter counts, and epoch counts. | 2 |

---

## 4. Visualizations (15 points)

| Item | Points |
|------|-------:|
| `baseline_curves.png` shows train loss and test accuracy per epoch with axis labels and a legend. | 3 |
| `transfer_curves.png` does the same for the transfer model. | 3 |
| `baseline_confusion.png` and `transfer_confusion.png` show 10×10 confusion matrices with class names on both axes, cell-value annotations, and a colorbar. | 6 |
| `exploration.ipynb` contains the 3×3-images-per-class grid, ten test-image side-by-side predictions, and at least two misclassified examples with the predicted/true labels printed. | 3 |

---

## 5. Report (20 points)

| Item | Points |
|------|-------:|
| Word count is between **600 and 900**. (Outside the window is an automatic -3.) | 2 |
| The report opens with a one-paragraph summary that names the architecture, dataset, and headline numbers. | 3 |
| The numerical comparison table (baseline vs. transfer) appears in the first half of the report. | 3 |
| The "why is there a gap" section is at least 100 words and references concrete properties of pretraining (the backbone has learned edge / texture / part detectors on 1.28M ImageNet images). | 5 |
| The "when to reach for each" section is concrete: it states that transfer learning is the right default below 100k labeled images and explains *why* with the data from this experiment. | 3 |
| Implementation notes section names the hyperparameters that mattered (differential learning rates, 224×224 resize, ImageNet normalization, augmentation). | 2 |
| Next-steps section names at least two ideas you would try with more compute. | 2 |

---

## Auto-failures (any one of these caps the grade at 50)

- Code uses Lightning, fastai, transformers, accelerate, or any other training-loop wrapper.
- Code uses `pretrained=True` instead of the modern `weights=` API.
- The transfer-learning model uses CIFAR-10 normalization (instead of ImageNet's) on the pretrained backbone. (See Lecture 3 Section 7.)
- The training loop is not the bare Week 8 loop. (Hidden `for` over the same iterable, no zero_grad, wrong order — common bugs.)
- `report.md` exceeds 900 words or is shorter than 600.
- Emojis present in any file.

---

## Stretch goals (credit-eligible up to +10 above 100)

These do not affect the base grade but can compensate for a missed criterion. Document any stretch goal you attempted in the report's "Implementation notes" section.

| Stretch | Points |
|---------|-------:|
| Add a third "feature extraction" variant (frozen backbone) to the comparison table. Train it for 5 epochs and report the numbers. | 3 |
| Implement a learning-rate schedule (`CosineAnnealingLR` or `OneCycleLR`) on the transfer-learning run; document any accuracy delta. | 2 |
| Repeat the experiment with ResNet-50 instead of ResNet-18 and document the compute / accuracy trade-off. | 2 |
| Run on Flowers-102 in addition to CIFAR-10 and report on the data-efficiency argument with numbers from both datasets. | 3 |

---

## Common mistakes and how to avoid them

1. **The transfer model "reaches 90%" on the training set but 60% on test.** Classic data-augmentation mismatch: you have flip-and-crop on the test loader. Augmentation belongs in the train transform only.
2. **The transfer model's accuracy decreases over epochs.** Catastrophic forgetting: backbone LR too high. Drop the backbone LR by 5x and retry.
3. **The two evaluate scripts produce slightly different accuracies than the train scripts' final epoch.** You probably did not call `model.eval()` in train, so BN's running statistics were still using batch stats at the final epoch's eval pass. Always call `model.eval()` inside `evaluate`.
4. **The save/load fails with "missing keys" warnings.** You saved a `state_dict` from `model` and tried to load into a different model class. The fix: save and load the same class; if you renamed the class, regenerate the checkpoint.
5. **`weights=ResNet18_Weights.IMAGENET1K_V1` raises an AttributeError.** You are on torchvision 0.12 or older. Upgrade: `pip install --upgrade torchvision`. The `weights=` API was introduced in 0.13.

---

## Self-grade template

Save this as `self-grade.md` in your repo:

```text
Section 1 (Correctness):           __ / 30
Section 2 (Engineering hygiene):   __ / 20
Section 3 (Comparison protocol):   __ / 15
Section 4 (Visualizations):        __ / 15
Section 5 (Report):                __ / 20
Stretch goals:                     __ / 10
                                   -------
Total:                             __ / 110

Self-assessment of the strongest part of this submission:
...

Self-assessment of the weakest part:
...
```

The C5 conviction: students who fill out the self-grade honestly are the same students who write good reports. Submission without a self-grade is not strictly penalized, but the grader will assume the absence is for a reason.
