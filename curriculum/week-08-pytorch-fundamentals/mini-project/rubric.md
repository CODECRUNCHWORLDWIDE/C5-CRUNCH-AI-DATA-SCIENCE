# Week 8 Mini-Project — Grading Rubric

100 points total. The mini-project is the second-largest single grade in Week 8, after the homework set.

## Scripts and code (40 points)

| Criterion | Points | Notes |
|-----------|-------:|-------|
| `train.py` runs end to end from a fresh clone | 5 | No interactive prompts, no missing-data errors |
| `evaluate.py` reloads `model.pt` and prints test accuracy | 5 | Must reproduce `train.py`'s final accuracy to 3 decimal places |
| Model is a subclass of `nn.Module` in `model.py` | 4 | Not a top-level `nn.Sequential` in a script |
| `train.py` and `evaluate.py` both `import` from `model.py` and `data.py` | 4 | The factoring is the production pattern |
| Training loop is the bare 8-line form from Lecture 2 Section 7 | 5 | No Lightning, no fastai, no `accelerate` |
| Model is saved as `state_dict` (not pickled) | 3 | `torch.save(model.state_dict(), ...)` |
| `torch.load` uses `weights_only=True` | 2 | PyTorch 2.4+ safe loader |
| Device detection (`cuda > mps > cpu`) | 3 | The `best_device()` pattern |
| `torch.manual_seed(42)` is set | 3 | Reproducibility |
| `python -m py_compile train.py evaluate.py model.py data.py` clean | 3 | AST-level validity |
| Per-epoch print statement summarizes train_loss, test_loss, test_acc | 3 | The minimum logging |

## Correctness and accuracy (20 points)

| Criterion | Points | Notes |
|-----------|-------:|-------|
| Final test accuracy meets the C5 threshold | 10 | FashionMNIST MLP ≥ 0.85; CIFAR-10 MLP ≥ 0.48; CIFAR-10 CNN ≥ 0.65 |
| `evaluate.py` accuracy matches `train.py` final-epoch accuracy | 5 | Exact match to 3 decimals (the reproducibility headline) |
| Train and test transforms are consistent | 3 | Same Normalize on both; no augmentation on test |
| No `softmax` applied before `CrossEntropyLoss` | 2 | The standard Lecture 2 Section 6 bug |

## Plots and notebook (15 points)

| Criterion | Points | Notes |
|-----------|-------:|-------|
| `training_curves.png` shows train+test loss and test accuracy | 4 | 2-panel figure, axis labels, legend |
| `confusion_matrix.png` shows class names on both axes | 4 | Class names, not class indices |
| `confusion_matrix.png` annotates cell values | 2 | Numbers visible in each cell |
| `exploration.ipynb` shows a 3x3 sample grid with class names | 3 | The "what does the data look like" panel |
| `exploration.ipynb` inspects 5 model predictions (correct + incorrect) | 2 | At least one of each |

## Report (`report.md`) (25 points)

| Criterion | Points | Notes |
|-----------|-------:|-------|
| 600-900 words | 3 | Outside that range loses 1pt; outside 400-1200 loses all 3 |
| The seven-section structure from README.md | 5 | Problem, model+setup, training curves, confusion, honest paragraph, reproducibility, what's next |
| The honest paragraph names ≥2 specific limitations | 5 | "The MLP ignores spatial structure" + "no regularization" is the typical pair |
| References Week 7 result for context | 2 | "Last week's NumPy MLP did X; this PyTorch version reaches Y." |
| Cites PyTorch 2.x docs by URL at least once | 2 | The docs habit is the habit |
| The "what next" section names ≥3 specific actions | 4 | "Switch to CNN," "add cosine LR schedule," "search over n_hidden" |
| Honest about wall-clock and hardware | 2 | "Trained on M2 Mac mini, mps device, 5 minutes total" |
| No vague claims ("state of the art," "industry standard," "we found that") | 2 | Specifics or none |

---

## Final score band

| Total | Letter |
|------:|--------|
| 90-100 | A |
| 80-89  | B |
| 70-79  | C |
| 60-69  | D |
| < 60   | F |

---

## The fast-fail conditions (any one fails the mini-project)

These conditions skip the rubric and result in an automatic F. Avoid all of them.

1. **`evaluate.py` produces a different accuracy than `train.py`.** The whole point of the project is reproducibility. If they disagree, fix the disagreement before submitting.
2. **The model class is not a `nn.Module` subclass.** It must be one; that is the central skill of Week 8.
3. **The training loop uses Lightning, fastai, or any wrapper.** Bare loop only. The point is to write the loop.
4. **The script does not run without manual intervention.** No interactive prompts, no manual file moves, no "after you do X, then run Y." `python train.py` must work from a fresh clone.
5. **The model was not actually trained by your script.** A pre-loaded checkpoint without a training pipeline is not a mini-project.

If any of those conditions trigger, the submission is graded F regardless of other quality. The C5 expectation is that you ship a reproducible artifact; the artifact's correctness is non-negotiable.
